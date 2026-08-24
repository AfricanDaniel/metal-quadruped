#!/usr/bin/env python3
"""Sweeps ONE motor's absolute target smoothly back and forth between two angles you choose (."""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

import dog_gym  # noqa: F401  (registers Dog-Stand-v0/Dog-Walk-v0)
from dog_gym.envs.dog_env import MAX_SLEW_DEG_PER_S, load_motor_joint_names

HERE = Path(__file__).resolve().parent

_COS45, _SIN45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
# Same pin quaternions as manual_motor_control.py
PIN_QUATS = {
    'upside-down': np.array([0.0, 1.0, 0.0, 0.0]),
    'sideways': np.array([_COS45, 0.0, _SIN45, 0.0]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--moving-motor', type=int, default=1, choices=range(1, 9), metavar='1-8',
                    help='motor that sweeps between --x-deg and --z-deg (default: 1, leg_a_thigh)')
    p.add_argument('--held-motor', type=int, default=2, choices=range(1, 9), metavar='1-8',
                    help='motor held fixed at its own startup value the whole run '
                         '(default: 2, leg_a_calf)')
    p.add_argument('--x-deg', type=float, default=-11.6,
                    help='sweep start angle, output-shaft degrees, absolute (default: -11.6, '
                         "leg_a_thigh's declared range low end)")
    p.add_argument('--z-deg', type=float, default=230.6,
                    help='sweep other end, output-shaft degrees, absolute (default: 230.6, '
                         "leg_a_thigh's declared range high end)")
    p.add_argument('--period-s', type=float, default=8.0,
                    help='seconds for one x->z->x cycle (default: 8.0; auto-lengthened if the '
                         'sweep speed would outrun the slew-rate limit -- see the note printed '
                         'at startup if that happens)')
    p.add_argument('--n-periods', type=float, default=2.0,
                    help='how many full x->z->x cycles to run (default: 2.0)')
    p.add_argument('--orientation', choices=['upside-down', 'sideways', 'normal'], default='upside-down',
                    help='how to pin the torso in mid-air (default: upside-down). "normal" disables '
                         'pinning entirely.')
    p.add_argument('--pin-height-m', type=float, default=0.6,
                    help='--orientation upside-down/sideways only: world z height the torso is held at')
    p.add_argument('--out', type=Path, default=None,
                    help='CSV output path (default: motor_sweep_<timestamp>.csv in the current directory)')
    p.add_argument('--headless', action='store_true',
                    help="don't open the MuJoCo viewer window -- just simulate and log, faster for "
                         'repeated automated runs. Default: viewer opens, same as the other manual '
                         'test tools.')
    return p.parse_args()


def _pin_torso(env, quat, height_m):
    env.data.qpos[0:3] = [0.0, 0.0, height_m]
    env.data.qpos[3:7] = quat
    env.data.qvel[0:6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def main():
    args = parse_args()
    joint_names = load_motor_joint_names()
    moving_idx = args.moving_motor - 1
    held_idx = args.held_motor - 1
    moving_joint = joint_names[moving_idx]
    held_joint = joint_names[held_idx]

    out_path = args.out or Path.cwd() / f'motor_sweep_{datetime.now():%Y%m%d_%H%M%S}.csv'

    env = gym.make('Dog-Stand-v0', render_mode=None).unwrapped
    obs, _ = env.reset()

    pin_quat = PIN_QUATS.get(args.orientation)
    if pin_quat is not None:
        _pin_torso(env, pin_quat, args.pin_height_m)

    action = obs[:8].copy()  # absolute target per motor
    held_target = action[held_idx]  # captured once, NEVER touched again

    x_rad, z_rad = np.radians(args.x_deg), np.radians(args.z_deg)

    # Auto-lengthen period so the sweep's peak commanded speed stays under MAX_SLEW_DEG_PER_S (with a 10% margin)
    amp_deg = abs(args.z_deg - args.x_deg) / 2
    min_period = amp_deg * 2 * np.pi / (0.9 * MAX_SLEW_DEG_PER_S)
    period_s = args.period_s
    if period_s < min_period:
        period_s = min_period
        print(f'note: --period-s raised to {period_s:.2f}s so the sweep target stays within '
              f'{0.9 * MAX_SLEW_DEG_PER_S:.0f}deg/s (90% of MAX_SLEW_DEG_PER_S={MAX_SLEW_DEG_PER_S:.0f}).')

    def moving_target(t):
        # Starts exactly at x_rad (t=0), reaches z_rad at t=period/2, back
        # to x_rad at t=period -- smooth, zero-velocity at both ends.
        return x_rad + (z_rad - x_rad) * (1 - np.cos(2 * np.pi * t / period_s)) / 2

    moving_act = env.model.actuator(f'motor_{args.moving_motor}')
    held_act = env.model.actuator(f'motor_{args.held_motor}')
    moving_dofadr = env.model.joint(moving_joint).dofadr[0]
    held_dofadr = env.model.joint(held_joint).dofadr[0]

    print(f'Sweeping motor {args.moving_motor} ({moving_joint}) between {args.x_deg:.1f} and '
          f'{args.z_deg:.1f}deg, motor {args.held_motor} ({held_joint}) held at '
          f'{np.degrees(held_target):.1f}deg (absolute). Orientation: {args.orientation}. '
          f'{args.n_periods:.1f} cycle(s) over {period_s:.1f}s each.')
    print(f'Logging to {out_path}')

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    dt = env.model.opt.timestep
    n_steps = int(args.n_periods * period_s / dt)
    t = 0.0
    tick = 0

    def saturated(qfrc, forcerange):
        lo, hi = forcerange
        if lo == 0.0 and hi == 0.0:  # forcelimited=False -- never saturated
            return False
        return abs(qfrc) >= 0.99 * max(abs(lo), abs(hi))

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tick', 'elapsed_s', 'role', 'motor_id', 'joint', 'target_deg', 'actual_deg',
                          'velocity_deg_s', 'torque_nm', 'saturated'])

        for i in range(n_steps):
            action[moving_idx] = moving_target(t)
            action[held_idx] = held_target
            obs, r, term, trunc, info = env.step(action)
            if pin_quat is not None:
                _pin_torso(env, pin_quat, args.pin_height_m)
            if viewer is not None:
                if not viewer.is_running():
                    print('viewer closed, stopping early.')
                    break
                viewer.sync()
                time.sleep(dt)

            moving_torque = env.data.qfrc_actuator[moving_dofadr]
            held_torque = env.data.qfrc_actuator[held_dofadr]
            writer.writerow([tick, f'{t:.3f}', 'moving', args.moving_motor, moving_joint,
                              f'{np.degrees(action[moving_idx]):.3f}', f'{np.degrees(obs[moving_idx]):.3f}',
                              f'{np.degrees(env.data.qvel[moving_dofadr]):.3f}', f'{moving_torque:.4f}',
                              int(saturated(moving_torque, moving_act.forcerange))])
            writer.writerow([tick, f'{t:.3f}', 'held', args.held_motor, held_joint,
                              f'{np.degrees(held_target):.3f}', f'{np.degrees(obs[held_idx]):.3f}',
                              f'{np.degrees(env.data.qvel[held_dofadr]):.3f}', f'{held_torque:.4f}',
                              int(saturated(held_torque, held_act.forcerange))])

            if tick % 40 == 0:
                print(f't={t:6.2f}s  motor{args.moving_motor} target={np.degrees(action[moving_idx]):7.2f}deg '
                      f'actual={np.degrees(obs[moving_idx]):7.2f}deg torque={moving_torque:6.2f}Nm  |  '
                      f'motor{args.held_motor} actual={np.degrees(obs[held_idx]):7.2f}deg '
                      f'torque={held_torque:6.2f}Nm')
            t += dt
            tick += 1

    if viewer is not None:
        viewer.close()
    env.close()
    print(f'Done. {tick} ticks logged to {out_path}')


if __name__ == '__main__':
    main()
