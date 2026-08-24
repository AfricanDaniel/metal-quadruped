#!/usr/bin/env python3
"""Behavior-cloning warm start for position-mode WALK training. Position-mode WALK has struggled to learn a gait from sc..."""


import argparse

import dog_gym  # noqa: F401  (registers Dog-v0)
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from dog_gym.envs.dog_env import WALK_ACTION_RESIDUAL_RANGE_RAD


def collect_teacher_data(teacher_path, env_id, walk_start_pose, domain_randomization,
                          rollout_steps, seed):
    """Rolls out the torque teacher and returns (obs, target_action) arrays, target_action already converted to position-mod..."""

    teacher_env = gym.make(env_id, control_mode='torque', task='walk',
                            walk_start_pose=walk_start_pose,
                            domain_randomization=domain_randomization).unwrapped
    teacher = PPO.load(teacher_path)

    obs_list = []
    action_list = []
    obs, _ = teacher_env.reset(seed=seed)
    collected = 0
    episode = 0
    while collected < rollout_steps:
        # Stochastic (not deterministic) sampling
        action, _ = teacher.predict(obs, deterministic=False)
        obs_before = obs
        obs, reward, terminated, truncated, info = teacher_env.step(action)
        # Target = the ACTUAL resulting joint angle (what a position controller would need to command to reproduce this motion), NOT the teacher's own torque action
        resulting_qpos = teacher_env.data.qpos[teacher_env.motor_qpos_adr].copy()
        target = np.clip(resulting_qpos - teacher_env._walk_default_action_rad,
                          -WALK_ACTION_RESIDUAL_RANGE_RAD, WALK_ACTION_RESIDUAL_RANGE_RAD)
        obs_list.append(obs_before)
        action_list.append(target.astype(np.float32))
        collected += 1
        if terminated or truncated:
            episode += 1
            obs, _ = teacher_env.reset(seed=seed + episode)
    teacher_env.close()
    return np.array(obs_list, dtype=np.float32), np.array(action_list, dtype=np.float32)


def pretrain(model, obs, targets, epochs, batch_size):
    """Supervised regression of model.policy's Gaussian mean action against targets."""

    device = model.device
    obs_t = torch.as_tensor(obs, device=device)
    targets_t = torch.as_tensor(targets, device=device)
    n = obs_t.shape[0]
    optimizer = model.policy.optimizer

    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_obs = obs_t[idx]
            batch_targets = targets_t[idx]

            distribution = model.policy.get_distribution(batch_obs)
            mean_actions = distribution.distribution.mean
            loss = nn.functional.mse_loss(mean_actions, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f'  epoch {epoch + 1}/{epochs}: mse={total_loss / n_batches:.5f}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--teacher-path', required=True,
                         help='SB3 checkpoint of a working torque-mode WALK policy, e.g. '
                              'models/walk_home_torque/PPO_9000000_walk_policy_torque_v8.zip')
    parser.add_argument('--output-path', required=True,
                         help='Where to save the warm-started PPO model (no .zip suffix -- SB3 '
                              'appends it). Feed this into train.py --init-from to continue with '
                              'RL fine-tuning under real reward.')
    parser.add_argument('--env-id', default='Dog-Walk-v0')
    parser.add_argument('--walk-start-pose', default='home', choices=['standing', 'home'],
                         help='Must match whatever the teacher checkpoint was actually trained '
                              'with, or the collected trajectories won\'t reflect real teacher '
                              'behavior.')
    parser.add_argument('--domain-randomization', action='store_true')
    parser.add_argument('--position-kp', type=float, default=20.0,
                         help='Student (position-mode) env gain, see train.py --position-kp. Only '
                              'affects the STUDENT env used to build the action space/network -- '
                              'has no effect on the teacher rollout (torque mode ignores it). Pass '
                              'the SAME value to train.py --init-from afterward.')
    parser.add_argument('--position-kd', type=float, default=2.0)
    parser.add_argument('--rollout-steps', type=int, default=200_000,
                         help='Total ticks of teacher rollout to collect, across as many episodes '
                              'as needed.')
    parser.add_argument('--epochs', type=int, default=30,
                         help='Supervised regression epochs over the collected dataset.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print(f'Collecting {args.rollout_steps} ticks from teacher {args.teacher_path} '
          f'(control_mode=torque, task=walk, walk_start_pose={args.walk_start_pose})...')
    obs, targets = collect_teacher_data(
        args.teacher_path, args.env_id, args.walk_start_pose, args.domain_randomization,
        args.rollout_steps, args.seed)
    print(f'Collected {obs.shape[0]} (obs, target_action) pairs.')

    print(f'Building student env (control_mode=position, task=walk, '
          f'position_kp={args.position_kp}, position_kd={args.position_kd})...')
    student_env = gym.make(args.env_id, control_mode='position', task='walk',
                            walk_start_pose=args.walk_start_pose,
                            domain_randomization=args.domain_randomization,
                            position_kp=args.position_kp, position_kd=args.position_kd)

    # Same net_arch/activation as train.py's PPO
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=nn.Tanh,
    )
    model = PPO('MlpPolicy', student_env, policy_kwargs=policy_kwargs,
                device=args.device, verbose=0)

    print(f'Pretraining (behavior cloning) for {args.epochs} epochs...')
    pretrain(model, obs, targets, args.epochs, args.batch_size)

    model.save(args.output_path)
    print(f'Saved warm-started model to {args.output_path}.zip -- continue with:\n'
          f'  python3 -m dog_gym.train --train --env-id {args.env_id} --algo PPO '
          f'--control-mode position --position-kp {args.position_kp} '
          f'--position-kd {args.position_kd} --walk-start-pose {args.walk_start_pose} '
          f'--init-from {args.output_path}.zip --fname <your_fname>')

    student_env.close()


if __name__ == '__main__':
    main()
