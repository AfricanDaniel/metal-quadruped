#!/usr/bin/env python3
"""Export a trained Stable-Baselines3 PPO policy to TorchScript.

dog_deploy runs on the Jetson and only needs `torch` at inference time --
not `stable_baselines3`/`gymnasium`/`mujoco` -- so training-side artifacts
(.zip SB3 checkpoints) get converted to a plain TorchScript module here.

Usage:
    ros2 run dog_gym export_policy models/PPO_7000000_stand_policy_v4.zip \
        models/stand_policy_v4.pt --env-id Dog-Stand-v0

    # torque-mode checkpoint (dog_deploy/policy_node.py's control_mode='torque'):
    ros2 run dog_gym export_policy models/PPO_5000000_stand_policy_torque_v8.zip \
        policies/stand_policy_torque_v8.pt --env-id Dog-Stand-v0 --control-mode torque
"""

import argparse

import dog_gym  # noqa: F401  (registers Dog-Stand-v0/Dog-Walk-v0, needed to load the model's env)
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

    def __init__(self, policy, action_low, action_high):
        super().__init__()
        self.policy = policy
        # SB3's own predict() (stable_baselines3/common/policies.py,
        # BasePolicy.predict()) clips ActorCriticPolicy.forward()'s raw
        # action -- an unbounded Gaussian mean, not something PPO's own
        # network ever squashes -- to the action space AFTER calling
        # forward(). forward() (what DeterministicPolicy.forward() below
        # calls into) never does this itself. Registered as buffers (not
        # plain Python floats) so they're saved as part of the traced
        # module's state, not baked in as untyped constants.
        self.register_buffer('action_low', torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer('action_high', torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, observation):
        action = self.policy(observation, deterministic=True)[0]
        return torch.clamp(action, self.action_low, self.action_high)


def export(model_path, output_path, env_id='Dog-Stand-v0', control_mode='position', model_xml_path=None):
    kwargs = dict(control_mode=control_mode)
    if model_xml_path is not None:
        kwargs['model_path'] = model_xml_path
    env = gym.make(env_id, **kwargs)
    # Force CPU regardless of what device the checkpoint was trained/saved
    # on (e.g. v4 was trained with device='cuda' on the VM) -- the exported
    # TorchScript module is meant to run on the Jetson's CPU anyway, and
    # torch.jit.trace below needs the model and its example input on the
    # same device, which is simplest to guarantee by just forcing CPU here.
    model = PPO.load(model_path, env=env, device='cpu')

    obs_dim = env.observation_space.shape[0]
    example_obs = torch.zeros(1, obs_dim)

    model.policy.set_training_mode(False)
    wrapped = DeterministicPolicy(model.policy, env.action_space.low, env.action_space.high)

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
    parser.add_argument('--control-mode', default='position', choices=['position', 'torque'],
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
