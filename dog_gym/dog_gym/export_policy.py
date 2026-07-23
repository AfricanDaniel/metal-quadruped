#!/usr/bin/env python3
"""Export a trained Stable-Baselines3 PPO policy to TorchScript.

dog_deploy runs on the Jetson and only needs `torch` at inference time --
not `stable_baselines3`/`gymnasium`/`mujoco` -- so training-side artifacts
(.zip SB3 checkpoints) get converted to a plain TorchScript module here.

Usage:
    ros2 run dog_gym export_policy models/PPO_1000000_dog_policy.zip \
        models/dog_policy.pt
"""

import argparse

import dog_gym  # noqa: F401  (registers Dog-v0, needed to load the model's env)
import gymnasium as gym
import torch
from stable_baselines3 import PPO


class DeterministicPolicy(torch.nn.Module):
    """Wraps an SB3 ActorCriticPolicy as a plain (obs) -> action module.

    Must be a real nn.Module (not e.g. a lambda closing over `policy`) so
    torch.jit.trace treats the wrapped policy's parameters as part of the
    traced graph instead of trying to inline them as constants, which
    raises "Cannot insert a Tensor that requires grad as a constant".
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy(observation, deterministic=True)[0]


def export(model_path, output_path):
    env = gym.make('Dog-v0')
    model = PPO.load(model_path, env=env)

    obs_dim = env.observation_space.shape[0]
    example_obs = torch.zeros(1, obs_dim)

    model.policy.set_training_mode(False)
    wrapped = DeterministicPolicy(model.policy)

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_obs)
    traced.save(output_path)
    print(f'Exported TorchScript policy to {output_path} '
          f'(input dim {obs_dim}, output dim {env.action_space.shape[0]})')
    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_path', help='Path to the SB3 .zip checkpoint')
    parser.add_argument('output_path', help='Where to write the TorchScript .pt file')
    args = parser.parse_args()
    export(args.model_path, args.output_path)


if __name__ == '__main__':
    main()
