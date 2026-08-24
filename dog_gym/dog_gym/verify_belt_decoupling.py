#!/usr/bin/env python3
"""Interactively visualize the belt/pulley calf-decoupling fix. Opens the MuJoCo viewer and sweeps one leg's THIGH back ..."""

import argparse
import os
import sys
import time

import gymnasium as gym
import mujoco
import numpy as np

import dog_gym  # noqa: F401  (registers Dog-Stand-v0/Dog-Walk-v0)
from dog_gym.envs.dog_env import MAX_SLEW_DEG_PER_S, load_motor_joint_names

# 180deg rotation about world +X (MuJoCo quat is w,x,y,z)
UPSIDE_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--leg', choices=['a', 'b', 'c', 'd'], default='a')
    p.add_argument('--amplitude-deg', type=float, default=30.0)
    p.add_argument('--full-range', action='store_true',
                    help="sweep the thigh across its ENTIRE <joint range> (read from the "
                         "model) instead of +-amplitude-deg around home -- shows the whole "
                         "path. Starts from whichever range end is nearer the home pose. "
                         "--period-s is auto-lengthened if needed so the moving target "
                         "never outruns the effective slew-rate limit (DogEnv's own "
                         f"MAX_SLEW_DEG_PER_S={MAX_SLEW_DEG_PER_S:.0f}deg/s by default, or "
                         "--max-slew-deg-per-s if set) -- otherwise the real motion would "
                         "lag/flatten instead of tracking the sweep.")
    p.add_argument('--period-s', type=float, default=4.0)
    p.add_argument('--duration-s', type=float, default=0.0,
                    help='0 = run until the viewer window is closed')
    p.add_argument('--upside-down', action=argparse.BooleanOptionalAction, default=True,
                    help="pin the torso upside-down in mid-air, clear of the floor and not "
                         "occluded by the body/other legs, for an unobstructed view (default: on). "
                         "The torso is kinematically held in place (re-set every step) -- gravity "
                         "still applies normally to the swinging leg itself.")
    p.add_argument('--pin-height-m', type=float, default=0.6,
                    help='--upside-down only: world z height the torso is held at')
    p.add_argument('--joint-stiffness', type=float, default=None,
                    help="runtime override of the MJCF's baked-in per-leg-joint stiffness "
                         "(see DogEnv.__init__'s joint_stiffness comment) -- watch how a "
                         "passive restoring spring on the SWINGING leg's own joints changes "
                         "how visibly it fights the sweep. None (default) leaves it at the "
                         "physically-correct 0. Does not affect the belt-decoupling result "
                         "itself (that's a kinematic constraint on the calf's COMMANDED "
                         "target, computed independently of joint stiffness), only how "
                         "hard the actuator has to work to hit that target.")
    p.add_argument('--max-slew-deg-per-s', type=float, default=None,
                    help='runtime override of DogEnv\'s own max_slew_deg_per_s starting '
                         'ceiling (see DogEnv.__init__\'s max_slew_deg_per_s comment). None '
                         f'(default) leaves it at MAX_SLEW_DEG_PER_S={MAX_SLEW_DEG_PER_S:.0f}. '
                         'A tighter value visibly slows/flattens how fast the commanded '
                         'target can move each tick -- --full-range\'s own auto period-'
                         'lengthening accounts for whatever this is set to.')
    p.add_argument('--no-coupling', action='store_true',
                    help="illustrate the OPPOSITE case, for comparison: instead of holding the "
                         "calf's ABSOLUTE world orientation fixed (the real, working "
                         "belt-decoupled behavior this tool normally verifies), holds its RAW "
                         "thigh-relative hinge angle fixed instead -- exactly what a normal, "
                         "non-decoupled elbow joint would do, so the calf visibly swings along "
                         "WITH the thigh the whole time. Implemented by zeroing calf_belt_sign "
                         "for this one leg (in-memory on this process's own env instance only, "
                         "never touching the model file) so DogEnv's own ctrl_calf = "
                         "action_calf + calf_belt_sign*thigh_qpos formula reduces to plain "
                         "ctrl_calf = action_calf. Zeroing the coefficient keeps the fed value "
                         "small/bounded so it never needs action_space clipping, regardless of "
                         "how far the thigh sweeps.")
    return p.parse_args()


def _pin_torso(env, height_m):
    env.data.qpos[0:3] = [0.0, 0.0, height_m]
    env.data.qpos[3:7] = UPSIDE_DOWN_QUAT
    env.data.qvel[0:6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def main():
    args = parse_args()

    joint_names = load_motor_joint_names()
    thigh_idx = joint_names.index(f'leg_{args.leg}_thigh')
    calf_idx = joint_names.index(f'leg_{args.leg}_calf')

    env_kwargs = {}
    if args.joint_stiffness is not None:
        env_kwargs['joint_stiffness'] = args.joint_stiffness
    if args.max_slew_deg_per_s is not None:
        env_kwargs['max_slew_deg_per_s'] = args.max_slew_deg_per_s
    env = gym.make('Dog-Stand-v0', render_mode='human', **env_kwargs).unwrapped
    effective_slew_deg_per_s = (
        args.max_slew_deg_per_s if args.max_slew_deg_per_s is not None else MAX_SLEW_DEG_PER_S)
    obs, _ = env.reset()
    if args.upside_down:
        _pin_torso(env, args.pin_height_m)

    calf_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f'leg_{args.leg}_calf')

    def calf_world_rotmat():
        return env.data.xmat[calf_body_id].reshape(3, 3).copy()

    def full_rotation_angle_deg(R0, R1):
        # Angle of the full 3D rotation R1 @ R0.T via the trace formula.
        cos_theta = (np.trace(R1 @ R0.T) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    action = obs[:8].copy()  # hold every motor at its current absolute target...
    calf_hold_target = action[calf_idx]  # ...this is the one being checked
    R0 = calf_world_rotmat()

    if args.no_coupling:
        # calf_idx/calf_belt_sign are built by DogEnv iterating over its own calf_idx array in the same order
        belt_pos = list(env.calf_idx).index(calf_idx)
        # Read the calf's CURRENT raw (thigh-relative) hinge angle directly off qpos
        raw_calf_hold_target = env.data.qpos[env.motor_qpos_adr[calf_idx]]
        # Zero the compensation coefficient itself (in-memory, this process's own env instance only
        env.calf_belt_sign[belt_pos] = 0.0
        print(f"--no-coupling: illustrating the OPPOSITE of the fix -- holding leg_{args.leg}'s "
              f"calf's RAW thigh-relative hinge fixed (like a normal, non-decoupled elbow) "
              f"instead of its absolute world orientation. The calf SHOULD visibly swing "
              f"along with the thigh now.")

    period_s = args.period_s
    if args.full_range:
        jid = env.model.joint(f'leg_{args.leg}_thigh').id
        lo, hi = env.model.jnt_range[jid]  # radians
        center = (lo + hi) / 2
        amp = (hi - lo) / 2
        # Start the sweep from whichever range end sits nearer the home pose (qpos=0), so there's no big initial jump: cos starts at +1, so `center + amp*cos` starts at hi, `center - amp*cos` at lo.
        start_sign = 1.0 if abs(hi) <= abs(lo) else -1.0
        # DogEnv.step() slew-limits the target to effective_slew_deg_per_s (MAX_SLEW_DEG_PER_S by default, or --max-slew-deg-per-s if set); the sine target's peak speed is amp*2pi/T.
        slew_margin_deg_per_s = 0.9 * effective_slew_deg_per_s
        min_period = np.degrees(amp) * 2 * np.pi / slew_margin_deg_per_s
        if period_s < min_period:
            period_s = min_period
            print(f'note: --period-s raised to {period_s:.1f}s so the sweep target stays '
                  f'within the {effective_slew_deg_per_s:.0f}deg/s slew-rate limit.')

        def thigh_target(t):
            return center + start_sign * amp * np.cos(2 * np.pi * t / period_s)

        print(f"Sweeping leg_{args.leg}'s thigh across its FULL range "
              f"({np.degrees(lo):.1f} .. {np.degrees(hi):.1f}deg), starting from the "
              f"{'upper' if start_sign > 0 else 'lower'} end (nearest home), "
              f"holding leg_{args.leg}'s calf fixed at {np.degrees(calf_hold_target):.1f}deg (absolute).")
    else:
        def thigh_target(t):
            return np.radians(args.amplitude_deg) * np.sin(2 * np.pi * t / period_s)

        print(f"Sweeping leg_{args.leg}'s thigh +-{args.amplitude_deg:.0f}deg around "
              f"{np.degrees(action[thigh_idx]):.1f}deg, holding leg_{args.leg}'s calf fixed at "
              f"{np.degrees(calf_hold_target):.1f}deg (absolute).")
    if args.no_coupling:
        print('Watch the calf in the viewer -- it should visibly swing WITH the thigh now '
              '(the illustration, not the real fix).')
    else:
        print('Watch the calf in the viewer -- it should barely move while the thigh swings.')
    print('Close the viewer window to stop.' if args.duration_s <= 0
          else f'Running for {args.duration_s:.0f}s (or close the viewer window to stop early).')

    dt = env.model.opt.timestep
    t = 0.0
    start = time.time()
    last_print = 0.0
    while env.renderer is None or env.renderer.is_running():
        if args.duration_s > 0 and time.time() - start > args.duration_s:
            break
        action[thigh_idx] = thigh_target(t)
        # --no-coupling: calf_belt_sign was zeroed above, so this same fixed-target assignment now produces ctrl_calf = raw_calf_hold_target directly.
        action[calf_idx] = raw_calf_hold_target if args.no_coupling else calf_hold_target
        # terminated/truncated are ignored here on purpose
        obs, reward, terminated, truncated, _ = env.step(action)
        if args.upside_down:
            _pin_torso(env, args.pin_height_m)
        t += dt
        # Sim steps execute far faster than real time
        time.sleep(dt)

        if time.time() - start - last_print > 0.5:
            last_print = time.time() - start
            drift_deg = full_rotation_angle_deg(R0, calf_world_rotmat())
            print(f't={t:5.1f}s  thigh={np.degrees(obs[thigh_idx]):+7.2f}deg  '
                  f'calf={np.degrees(obs[calf_idx]):+7.2f}deg (target {np.degrees(calf_hold_target):+7.2f}deg)  '
                  f'calf world-orientation drift={drift_deg:5.2f}deg')

    env.close()

    # mujoco.viewer.launch_passive's pause-key_callback thread doesn't join cleanly on close()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
