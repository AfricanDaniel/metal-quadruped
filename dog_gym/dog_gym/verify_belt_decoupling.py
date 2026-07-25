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
from dog_gym.envs.dog_env import load_motor_joint_names

# 180deg rotation about world +X (MuJoCo quat is w,x,y,z) -- flips the
# torso belly-up. Which of front/back vs left/right ends up mirrored
# doesn't matter for this test, only that it's clear of the floor and
# not occluded by the torso/other legs.
UPSIDE_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--leg', choices=['a', 'b', 'c', 'd'], default='a')
    p.add_argument('--amplitude-deg', type=float, default=30.0)
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

    env = gym.make('Dog-Stand-v0', render_mode='human').unwrapped
    obs, _ = env.reset()
    if args.upside_down:
        _pin_torso(env, args.pin_height_m)

    calf_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f'leg_{args.leg}_calf')

    def calf_world_zaxis():
        xmat = env.data.xmat[calf_body_id].reshape(3, 3)
        return xmat[:, 2].copy()

    action = obs[:8].copy()  # hold every motor at its current absolute target...
    calf_hold_target = action[calf_idx]  # ...this is the one being checked
    z0 = calf_world_zaxis()

    print(f"Sweeping leg_{args.leg}'s thigh +-{args.amplitude_deg:.0f}deg around "
          f"{np.degrees(action[thigh_idx]):.1f}deg, holding leg_{args.leg}'s calf fixed at "
          f"{np.degrees(calf_hold_target):.1f}deg (absolute).")
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
        action[thigh_idx] = np.radians(args.amplitude_deg) * np.sin(2 * np.pi * t / args.period_s)
        action[calf_idx] = calf_hold_target
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
            drift_deg = np.degrees(np.arccos(np.clip(np.dot(z0, calf_world_zaxis()), -1.0, 1.0)))
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
