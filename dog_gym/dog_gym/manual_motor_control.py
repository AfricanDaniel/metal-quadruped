#!/usr/bin/env python3
"""Interactively drive each motor's target by hand, to double-check the
belt/pulley calf-decoupling fix yourself instead of watching the
automated sweep in verify_belt_decoupling.py (see that file's docstring
for the root cause: belt/pulley calf decoupling was never modeled in
sim, found 2026-07-25).

Deliberately NOT using MuJoCo's native Control-tab sliders: those are a
direct read/write on the raw MJCF actuator's ctrl value, which for a calf
motor is the UNCOMPENSATED thigh-relative hinge target -- dragging one
would show the OLD, broken (thigh-dragging-calf) behavior, not the fix.
Keyboard control below goes through the real DogEnv.step() (same as
verify_belt_decoupling.py and training), so calf motors keep their
ABSOLUTE-angle semantics -- same code path, not a reimplementation.

Keys:
    1-8         select motor 1-8 (see the printed motor->joint table)
    Up / Down   nudge the selected motor's target by --step-deg
    R           reset every motor's target back to its startup value
    Space       pause/resume physics

Usage:
    python3 -m dog_gym.manual_motor_control [--step-deg 2] [--orientation upside-down] [--pin-height-m 0.6]

Needs a display (mujoco.viewer.launch_passive) -- run this on your dev
machine, not the headless training VM.
"""
import argparse
import os
import sys
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import dog_gym  # noqa: F401  (registers Dog-Stand-v0/Dog-Walk-v0)
from dog_gym.envs.dog_env import load_motor_joint_names

GLFW_KEY_SPACE = 32
GLFW_KEY_UP = 265
GLFW_KEY_DOWN = 264

_COS45, _SIN45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
# Quaternions (w,x,y,z), pinning the torso in mid-air. 'upside-down' is a
# 180deg flip about world X (belly-up). 'sideways' is a 90deg roll about
# world Y -- for THIS robot specifically (not a general "any 90deg roll
# works" fact), every hip/knee joint's axis happens to already point
# along local/world X in the normal standing orientation (verified
# directly via data.xaxis at qpos=0), so this particular roll swings
# every joint axis to vertical (parallel to gravity) -- gravity torque
# about a joint is proportional to axis . (r x gravity_force), which is
# exactly zero when axis is parallel to gravity, regardless of the leg's
# configuration. That's what "legs not affected by gravity" means here:
# not zero weight, zero TORQUE about each joint's own rotation axis, so
# the position PD controller isn't fighting gravity droop at all in this
# orientation.
PIN_QUATS = {
    'upside-down': np.array([0.0, 1.0, 0.0, 0.0]),
    'sideways': np.array([_COS45, 0.0, _SIN45, 0.0]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--step-deg', type=float, default=10.0,
                    help='how far one Up/Down key press nudges the selected motor')
    p.add_argument('--orientation', choices=['upside-down', 'sideways', 'normal'], default='upside-down',
                    help="how to pin the torso in mid-air, clear of the floor and not occluded by "
                         "the body/other legs, for an unobstructed view. 'sideways' additionally "
                         "puts every leg joint's axis parallel to gravity, so the position "
                         "controller isn't fighting any gravity droop at all (see PIN_QUATS' "
                         "comment). 'normal' disables pinning entirely -- gravity/falling applies "
                         "as usual. (default: upside-down)")
    p.add_argument('--pin-height-m', type=float, default=0.6,
                    help='--orientation upside-down/sideways only: world z height the torso is held at')
    return p.parse_args()


def _pin_torso(env, quat, height_m):
    env.data.qpos[0:3] = [0.0, 0.0, height_m]
    env.data.qpos[3:7] = quat
    env.data.qvel[0:6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def main():
    args = parse_args()
    step_rad = np.radians(args.step_deg)

    joint_names = load_motor_joint_names()

    pin_quat = PIN_QUATS.get(args.orientation)  # None for 'normal' -- no pinning

    env = gym.make('Dog-Stand-v0').unwrapped  # no render_mode -- we drive our own viewer below
    obs, _ = env.reset()
    if pin_quat is not None:
        _pin_torso(env, pin_quat, args.pin_height_m)

    action = obs[:8].copy()  # absolute target per motor -- see calf_idx's comment in dog_env.py
    initial_action = action.copy()

    print('Motor -> joint:')
    for i, name in enumerate(joint_names):
        print(f'  {i + 1}: {name}')
    print(f"\nOrientation: {args.orientation}")
    print(f'Keys: 1-8 select motor, Up/Down nudge by {args.step_deg:.1f}deg, '
          'R reset all, Space pause/resume.')

    state = {'selected': 0, 'paused': False}

    def on_key(keycode):
        if keycode == GLFW_KEY_SPACE:
            state['paused'] = not state['paused']
        elif ord('1') <= keycode <= ord('8'):
            state['selected'] = keycode - ord('1')
            print(f"selected motor {state['selected'] + 1} ({joint_names[state['selected']]}), "
                  f"target={np.degrees(action[state['selected']]):+.1f}deg")
        elif keycode == GLFW_KEY_UP:
            action[state['selected']] += step_rad
            print(f"motor {state['selected'] + 1} ({joint_names[state['selected']]}) "
                  f"target -> {np.degrees(action[state['selected']]):+.1f}deg")
        elif keycode == GLFW_KEY_DOWN:
            action[state['selected']] -= step_rad
            print(f"motor {state['selected'] + 1} ({joint_names[state['selected']]}) "
                  f"target -> {np.degrees(action[state['selected']]):+.1f}deg")
        elif keycode == ord('R'):
            action[:] = initial_action
            print('reset all motor targets to startup values')

    dt = env.model.opt.timestep
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        while viewer.is_running():
            if not state['paused']:
                obs, reward, terminated, truncated, _ = env.step(action)
                if pin_quat is not None:
                    _pin_torso(env, pin_quat, args.pin_height_m)
            viewer.sync()
            time.sleep(dt)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
