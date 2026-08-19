#!/usr/bin/env python3
"""Interactively visualize the belt/pulley calf-decoupling fix (see
daniel_cl_context.md's "Real root cause found: belt/pulley calf
decoupling was never modeled in sim" section, 2026-07-25).

Opens the MuJoCo viewer and sweeps one leg's THIGH back and forth while
holding every other action -- including that leg's own CALF -- fixed at
an absolute target. On the real robot, a belt ties the calf's lower
pulley to a torso-mounted motor, so rotating only the thigh never changes
the calf's real-world orientation. If the fix is working, the calf should
visibly stay pointing the same direction the whole time the thigh swings,
instead of swinging along with it like a normal elbow.

This runs the actual DogEnv.step()/_get_obs() code path training uses
(via `gym.make`), not a standalone reimplementation, so it's testing the
real fix.

Usage:
    python3 -m dog_gym.verify_belt_decoupling [--leg a] [--amplitude-deg 30] [--period-s 4] [--duration-s 30]
    python3 -m dog_gym.verify_belt_decoupling --leg a --full-range   # sweep the thigh's whole joint range
    python3 -m dog_gym.verify_belt_decoupling --leg a --no-coupling  # illustrate the OPPOSITE
                                                                      # case for comparison --
                                                                      # calf swings WITH the
                                                                      # thigh, like a normal
                                                                      # non-decoupled elbow

Needs a display (mujoco.viewer.launch_passive) -- run this on your dev
machine, not the headless training VM.
"""
import argparse
import os
import sys
import time

import gymnasium as gym
import mujoco
import numpy as np

import dog_gym  # noqa: F401  (registers Dog-Stand-v0/Dog-Walk-v0)
from dog_gym.envs.dog_env import MAX_SLEW_DEG_PER_S, load_motor_joint_names

# 180deg rotation about world +X (MuJoCo quat is w,x,y,z) -- flips the
# torso belly-up. Which of front/back vs left/right ends up mirrored
# doesn't matter for this test, only that it's clear of the floor and
# not occluded by the torso/other legs.
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
                    help="illustrate the OPPOSITE case, for comparison (2026-08-17, user "
                         "request -- 'show what the robot would look like if it was not "
                         "coupled and the thigh was swinging'): instead of holding the calf's "
                         "ABSOLUTE world orientation fixed (the real, working belt-decoupled "
                         "behavior this tool normally verifies), holds its RAW thigh-relative "
                         "hinge angle fixed instead -- exactly what a normal, non-decoupled "
                         "elbow joint would do, so the calf visibly swings along WITH the "
                         "thigh the whole time. Implemented by zeroing calf_belt_sign for this "
                         "one leg (in-memory on this process's own env instance only, never "
                         "touching the model file) so DogEnv's own ctrl_calf = action_calf + "
                         "calf_belt_sign*thigh_qpos formula reduces to plain ctrl_calf = "
                         "action_calf -- an EARLIER version instead tried to feed action_calf a "
                         "value that CANCELS the unmodified formula every tick, which needed "
                         "action_calf to swing from +11.6 to -230.6deg across a --full-range "
                         "sweep, but the calf's action_space is bounded to [0, 206.1]deg -- "
                         "step()'s own np.clip(action, action_space.low, action_space.high), "
                         "its very first line, silently clamped that and broke the "
                         "illustration specifically during large excursions (caught via user "
                         "report: 'the calf does not stay tucked in like it's supposed to' "
                         "when combined with --full-range). Zeroing the coefficient instead "
                         "keeps the fed value small/bounded (just the calf's own starting "
                         "angle) so it never needs clipping, regardless of how far the thigh "
                         "sweeps.")
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
        # An earlier version of this script compared a single body axis
        # (xmat's z column) instead -- BLIND to rotation about that axis,
        # which for this calf is exactly the hinge axis, so it printed
        # ~0 drift even while the calf visibly rotated (the same flawed
        # metric that once hid a real sign bug -- see daniel_cl_context.md's
        # "flip the sign" section). Never compare a single axis vector.
        cos_theta = (np.trace(R1 @ R0.T) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    action = obs[:8].copy()  # hold every motor at its current absolute target...
    calf_hold_target = action[calf_idx]  # ...this is the one being checked
    R0 = calf_world_rotmat()

    if args.no_coupling:
        # calf_idx/calf_belt_sign are built by DogEnv iterating over its
        # OWN calf_idx array in the same order (see that constant's own
        # comment in dog_env.py) -- this position lookup finds where
        # THIS script's chosen leg lands in that order.
        belt_pos = list(env.calf_idx).index(calf_idx)
        # Read the calf's CURRENT raw (thigh-relative) hinge angle
        # directly off qpos -- unambiguous, doesn't depend on
        # calf_belt_sign at all (unlike calf_hold_target above, which
        # was captured through the STILL-ACTIVE belt compensation).
        raw_calf_hold_target = env.data.qpos[env.motor_qpos_adr[calf_idx]]
        # Zero the compensation coefficient itself (in-memory, this
        # process's own env instance only -- never touches the model
        # file) so DogEnv's own ctrl_calf = action_calf +
        # calf_belt_sign*thigh_qpos formula reduces to plain ctrl_calf =
        # action_calf. See this flag's own --help for why this replaced
        # an earlier, more complex "feed action_calf a cancelling value"
        # approach that broke under --full-range (action_space clipping).
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
        # Start the sweep from whichever range end sits nearer the home
        # pose (qpos=0), so there's no big initial jump: cos starts at
        # +1, so `center + amp*cos` starts at hi, `center - amp*cos` at lo.
        start_sign = 1.0 if abs(hi) <= abs(lo) else -1.0
        # DogEnv.step() slew-limits the target to effective_slew_deg_per_s
        # (MAX_SLEW_DEG_PER_S by default, or --max-slew-deg-per-s if set);
        # the sine target's peak speed is amp*2pi/T. Cap at 90% of that
        # ceiling (small margin) or the real motion lags the target and
        # never shows the true path.
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
        # --no-coupling: calf_belt_sign was zeroed above, so this same
        # simple fixed-target assignment (identical to the normal/coupled
        # branch) now produces ctrl_calf = raw_calf_hold_target directly,
        # with no cancellation math and no risk of the calf's own bounded
        # action_space clipping it during a wide --full-range sweep.
        action[calf_idx] = raw_calf_hold_target if args.no_coupling else calf_hold_target
        # terminated/truncated are ignored here on purpose -- this tool
        # manually drives the pose every step rather than running a real
        # RL episode, and (when --upside-down) the robot is permanently
        # "fallen" by _is_fallen()'s definition, so auto-resetting on that
        # would just snap every joint back to qpos=0 every single step.
        obs, reward, terminated, truncated, _ = env.step(action)
        if args.upside_down:
            _pin_torso(env, args.pin_height_m)
        t += dt
        # Sim steps execute far faster than real time -- without pacing,
        # the whole sweep blows by in a fraction of a second and there's
        # nothing to actually watch (same reasoning as train.py's --test).
        time.sleep(dt)

        if time.time() - start - last_print > 0.5:
            last_print = time.time() - start
            drift_deg = full_rotation_angle_deg(R0, calf_world_rotmat())
            print(f't={t:5.1f}s  thigh={np.degrees(obs[thigh_idx]):+7.2f}deg  '
                  f'calf={np.degrees(obs[calf_idx]):+7.2f}deg (target {np.degrees(calf_hold_target):+7.2f}deg)  '
                  f'calf world-orientation drift={drift_deg:5.2f}deg')

    env.close()

    # mujoco.viewer.launch_passive's pause-key_callback thread doesn't join
    # cleanly on close() -- same issue train.py's --test works around.
    # Force-exit rather than hang here; nothing else is pending.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
