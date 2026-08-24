#!/usr/bin/env python3
"""Export a trained Stable-Baselines3 PPO policy to TorchScript. dog_deploy runs on the Jetson and only needs `torch` at..."""


import argparse

import dog_gym  # noqa: F401  (registers Dog-Stand-v0/Dog-Walk-v0, needed to load the model's env)
import gymnasium as gym
import torch
from stable_baselines3 import PPO


class DeterministicPolicy(torch.nn.Module):
    """Wraps an SB3 ActorCriticPolicy as a plain (obs) -> action module. Must be a real nn.Module (not e.g."""


    def __init__(self, policy, action_low, action_high, default_action=None):
        super().__init__()
        self.policy = policy
        # SB3's own predict() (stable_baselines3/common/policies.py, BasePolicy.predict()) clips ActorCriticPolicy.forward()'s raw action
        self.register_buffer('action_low', torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.as_tensor(action_high, dtype=torch.float32))
        # WALK trains a RESIDUAL action around dog_env.py's own _walk_default_action_rad (see WALK_ACTION_RESIDUAL_RANGE_RAD), not an absolute joint target
        if default_action is not None:
            self.register_buffer('default_action', torch.as_tensor(default_action, dtype=torch.float32))
        else:
            self.default_action = None

    def forward(self, observation):
        action = self.policy(observation, deterministic=True)[0]
        action = torch.clamp(action, self.action_low, self.action_high)
        if self.default_action is not None:
            action = action + self.default_action
        return action


def export(model_path, output_path, env_id='Dog-Stand-v0', control_mode='position', model_xml_path=None):
    kwargs = dict(control_mode=control_mode)
    if model_xml_path is not None:
        kwargs['model_path'] = model_xml_path
    env = gym.make(env_id, **kwargs)
    # Force CPU regardless of what device the checkpoint was trained/saved on (e.g.
    model = PPO.load(model_path, env=env, device='cpu')

    obs_dim = env.observation_space.shape[0]
    example_obs = torch.zeros(1, obs_dim)

    model.policy.set_training_mode(False)

    default_action = None
    if getattr(env.unwrapped, 'task', None) == 'walk' and getattr(env.unwrapped, 'control_mode', None) == 'position':
        default_action = env.unwrapped._walk_default_action_rad

    wrapped = DeterministicPolicy(model.policy, env.action_space.low, env.action_space.high, default_action)

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
    parser.add_argument('--env-id', default='Dog-Stand-v0', choices=['Dog-Stand-v0', 'Dog-Walk-v0'])
    parser.add_argument('--control-mode', default='position',
                         choices=['position', 'torque', 'torque_belt'],
                         help='Must match whatever control_mode the checkpoint was actually '
                              'trained with (dog_gym.train\'s --control-mode) -- a mismatch fails '
                              'with an action-space error from SB3\'s PPO.load(), same as --test.')
    parser.add_argument('--model-xml-path', default=None,
                         help='Override which MJCF file the env loads (same role as train.py\'s '
                              '--model-path) -- needed if a shared file like dog_torque.mjcf.xml '
                              'was regenerated with different values after this checkpoint was '
                              'trained.')
    args = parser.parse_args()
    export(args.model_path, args.output_path, args.env_id, args.control_mode, args.model_xml_path)


if __name__ == '__main__':
    main()
