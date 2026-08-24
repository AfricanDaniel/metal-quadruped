#!/usr/bin/env python3
"""Behavior-cloning warm start for position-mode WALK training.

Position-mode WALK has never learned to walk from scratch, even at 27-35M
steps (DR_walk/walk_decay lines). Torque-mode
WALK, by contrast, already works (models/walk_home_torque/*_v8.zip). The
working hypothesis (see train.py's --position-kp/--position-kd) is that
free RL exploration under a stiff PD servo (kp=60) turns exploration noise
into violent, near-instantaneous corrective force -- a much harsher
reward landscape to discover a gait in than torque control's graceful,
inertia-smoothed degradation.

Rather than have position-mode discover a gait via free exploration AGAIN,
this script bootstraps a position-mode PPO policy by imitating an
already-working torque-mode policy's own resulting joint trajectories:

  1. Roll out the torque teacher for many steps, recording each tick's
     observation (control-mode-agnostic -- see DogEnv._get_obs(), which
     doesn't branch on control_mode) and the joint angle it ACTUALLY
     reached next.
  2. Convert that resulting angle into position-mode's own action
     convention (residual from _walk_default_action_rad, clipped to
     +-WALK_ACTION_RESIDUAL_RANGE_RAD) -- this is exactly the action a
     position controller would need to command to reproduce that same
     motion, assuming reasonable PD tracking.
  3. Supervised-regress a fresh PPO policy's mean action output against
     those targets (standard behavior-cloning pretraining -- this bypasses
     PPO's own rollout collection/GAE/clipping entirely; it's imitation,
     not RL).
  4. Save the result in the same format train.py's --init-from expects, so
     RL fine-tuning can continue from here under real reward feedback
     instead of from a random network.

Usage:
    python3 -m dog_gym.imitation_pretrain \\
        --teacher-path models/walk_home_torque/PPO_9000000_walk_policy_torque_v8.zip \\
        --output-path models/walk_position_imitation_init \\
        --position-kp 20 --position-kd 2 --rollout-steps 200000 --epochs 30

Then continue with real RL under the SAME position_kp/kd:
    python3 -m dog_gym.train --train --env-id Dog-Walk-v0 --algo PPO \\
        --control-mode position --position-kp 20 --position-kd 2 \\
        --walk-start-pose home --init-from models/walk_position_imitation_init.zip \\
        --fname walk_position_imitation_v1
"""

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
    """Rolls out the torque teacher and returns (obs, target_action) arrays,
    target_action already converted to position-mode's residual convention."""
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
        # Stochastic (not deterministic) sampling -- more state diversity
        # across the dataset than a single repeated deterministic
        # trajectory would give, same reasoning as any on-policy rollout
        # collection.
        action, _ = teacher.predict(obs, deterministic=False)
        obs_before = obs
        obs, reward, terminated, truncated, info = teacher_env.step(action)
        # Target = the ACTUAL resulting joint angle (what a position
        # controller would need to command to reproduce this motion), NOT
        # the teacher's own torque action -- torque and position actions
        # live in incompatible spaces (N*m vs radians, see
        # export_policy.py's --control-mode action-space-mismatch error).
        # motor_qpos_adr already returns calf entries in the SAME
        # thigh-relative convention _walk_default_action_rad uses (see
        # __init__'s calf_belt_sign-derived construction of both), so
        # subtracting the baseline directly is the correct residual, no
        # extra per-leg conversion needed.
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
    """Supervised regression of model.policy's Gaussian mean action
    against targets -- standard SB3 behavior-cloning pretrain pattern.
    Uses policy.get_distribution() (public SB3 API) rather than
    reimplementing ActorCriticPolicy.forward()'s internals, so this stays
    correct across SB3's own feature-extractor/mlp_extractor wiring
    instead of assuming a specific internal shape. Only the actor (mean
    action) is fitted -- the critic (value function) is left at its
    random init, standard for BC-then-RL-finetune pipelines, since it
    adapts quickly once real reward feedback starts under --init-from."""
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

    # Same net_arch/activation as train.py's PPO -- MUST match exactly so
    # this checkpoint's saved policy loads correctly under --init-from
    # (SB3 restores weights by architecture, not just observation/action
    # space shape).
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
