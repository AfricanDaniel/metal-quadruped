#!/usr/bin/env python3
"""Train or test a PPO/SAC/A2C policy on a Dog-Stand-v0/Dog-Walk-v0
environment (see dog_gym/envs/dog_env.py's module docstring for what
each task means and why they're separate).

Restructured from a teammate's full_model_script_multi_processs.py
(shane_ws/Fast-Quadruped-), but on modern mujoco + current Gymnasium
instead of mujoco_py, and without SB3.

Usage:
    python3 -m dog_gym.train --train --env-id Dog-Stand-v0 --algo PPO --fname stand_model
    python3 -m dog_gym.train --train --env-id Dog-Walk-v0 --algo PPO --fname walk_model
    python3 -m dog_gym.train --test models/PPO_1000000_stand_model --env-id Dog-Stand-v0

Fine-tuning from an existing checkpoint (e.g. a good stand policy as the
starting point for walk training -- valid because Dog-Stand-v0/
Dog-Walk-v0 share the exact same DogEnv observation/action space, only
task=stand/walk's reward+reset differ, see dog_env.py's module
docstring):
    python3 -m dog_gym.train --train --env-id Dog-Walk-v0 --algo PPO \\
        --init-from models/PPO_32000000_DR_stand_policy_v1.zip --fname DR_walk_policy_v1
--init-from loads the saved policy/value network weights (a real head
start over random init) AND optimizer state via PPO.load(), rebinding to
the new env -- --n-steps/--batch-size/--n-epochs/device still apply,
overriding whatever was saved. The timestep counter and checkpoint
filenames restart at this run's own 0 regardless of how many steps the
source checkpoint had already seen (reset_num_timesteps=True on the
first .learn() call only) -- so DR_walk_policy_v1's own PPO_1000000_...
name means "1M steps of WALK fine-tuning", not "33M cumulative".
"""

import argparse
import os
import sys
import time

import dog_gym  # noqa: F401  (registers Dog-v0)
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

ALGOS = {'PPO': PPO, 'SAC': SAC, 'A2C': A2C}


class DecayScheduleCallback(BaseCallback):
    """Linearly decays ent_coef and learning_rate from their start value
    to a floor over `decay_steps` CUMULATIVE timesteps
    (self.model.num_timesteps, correctly maintained across repeated
    .learn() calls via reset_num_timesteps=False), then holds at the
    floor.

    NOT using SB3's own built-in learning_rate schedule support (passing
    a callable at PPO construction) -- that machinery computes progress
    relative to a single .learn() call's own total_timesteps budget,
    which doesn't fit this script's indefinite `while True` training
    loop (see train()): each new .learn() call would otherwise reset or
    distort progress_remaining rather than continuing a single smooth
    decay across the whole run. This callback sidesteps that by reading
    model.num_timesteps directly (a real cumulative counter) and forcing
    the applied values every step, independent of SB3's own progress
    tracking.

    Added 2026-07-30 to investigate a real, repeated finding logged in
    daniel_cl_context.md's TODO 16: every stand-task reward variant
    tried (flat action_rate_penalty, gated -1.0, gated -0.4) looked good
    early in training and then degraded substantially with enough
    further training -- the SAME divergence pattern regardless of the
    reward shape, pointing at something more fundamental than reward
    tuning. Leading suspect: ent_coef=0.01 never decayed (constant
    exploration pressure for the entire run, no mechanism to let the
    policy settle) combined with a constant learning_rate the whole way
    through -- nothing stops continued gradient noise from drifting an
    already-good policy away from it. Both defaults (ent_coef_end ==
    ent_coef_start, lr_end == learning_rate) leave decay INERT unless
    explicitly configured differently via the CLI, so this is fully
    opt-in and doesn't change any existing run's behavior by default.

    For ent_coef: mutating self.model.ent_coef directly is safe and
    takes effect on the next PPO.train() call, which reads
    self.ent_coef fresh every time (not cached into a schedule object
    the way learning_rate is).
    For learning_rate: overrides self.model.lr_schedule with a lambda
    that always returns the current decayed value regardless of its
    progress_remaining input -- SB3's own _update_learning_rate() (called
    internally once per rollout update) applies whatever that function
    returns to the optimizer, so this correctly forces the real applied
    LR without needing to fight SB3's own progress-tracking internals."""

    def __init__(self, ent_coef_start, ent_coef_end, lr_start, lr_end, decay_steps, verbose=0):
        super().__init__(verbose)
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_steps = decay_steps

    def _on_step(self):
        progress = min(1.0, self.model.num_timesteps / self.decay_steps) if self.decay_steps > 0 else 1.0
        self.model.ent_coef = self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * progress
        current_lr = self.lr_start + (self.lr_end - self.lr_start) * progress
        self.model.lr_schedule = lambda progress_remaining, _lr=current_lr: _lr
        return True


def make_env(env_id, domain_randomization):
    return lambda: gym.make(env_id, domain_randomization=domain_randomization)


def train(env_id, algo, fname, env_type, num_envs, total_timesteps_per_iter,
          log_dir, model_dir, domain_randomization, n_steps, batch_size, n_epochs,
          learning_rate, ent_coef, ent_coef_end, learning_rate_end, decay_steps,
          init_from=None):
    print(f'Training {algo} on {env_id} ({env_type}, {num_envs} envs)')

    env_fns = [make_env(env_id, domain_randomization) for _ in range(num_envs)]
    if env_type == 'dummy':
        env = DummyVecEnv(env_fns)
    elif env_type == 'subproc':
        env = SubprocVecEnv(env_fns, start_method='spawn')
    else:
        raise ValueError(f'Unknown env_type: {env_type}')
    env = VecMonitor(env)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.Tanh,
    )

    device = 'cuda'  # VM with a real GPU -- swap for 'cpu' on a dev machine (small MLP,
                      # GPU transfer overhead isn't worth it there)

    if init_from:
        # Fine-tuning: PPO.load() restores the saved policy/value network
        # weights AND optimizer state, then rebinds to `env` (a NEW env,
        # e.g. Dog-Walk-v0 while the checkpoint was trained on
        # Dog-Stand-v0 -- valid exactly because both tasks share the same
        # DogEnv observation/action space, see this module's docstring).
        # The n_steps/batch_size/n_epochs/learning_rate/device kwargs here
        # OVERRIDE whatever was saved in the checkpoint, matching this
        # run's own CLI flags rather than silently inheriting the source
        # run's. --learning-rate matters MORE here than for a fresh run:
        # fine-tuning from an already-good policy generally wants a LOWER
        # rate than training from scratch (2026-07-28 -- a
        # penaltyFix fine-tune at the same 3e-4 default used for fresh
        # training got WORSE, not better, over 19M further steps --
        # raw-action swing at a settled stand pose went from 74.7deg
        # mean at 1M steps to 148.5deg at 20M, roughly back to the
        # original pre-fix policy's level -- large gradient steps on an
        # already-near-optimal network are more likely to destabilize
        # existing behavior than refine it smoothly; see
        # daniel_cl_context.md).
        print(f'Fine-tuning from {init_from} on {env_id}')
        if algo != 'PPO':
            raise ValueError('--init-from is only wired up for PPO so far')
        model = PPO.load(
            init_from, env=env, device=device,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            learning_rate=learning_rate, ent_coef=ent_coef, tensorboard_log=log_dir)
    elif algo == 'PPO':
        # n_steps*num_envs is the rollout buffer size collected before each
        # round of updates -- SB3 just warns (doesn't error) if batch_size
        # doesn't evenly divide it, verified directly; not re-validated
        # here, so a bad combination will silently proceed with a
        # truncated final minibatch each epoch. Watch stdout for that
        # warning after changing any of these three.
        model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,  # was hardcoded 0.01 (itself raised from 0.0 -- no exploration
                                 # pressure beyond the policy's own action noise, which let training
                                 # settle into a "sit still and level" local optimum) -- now a CLI
                                 # flag, see --ent-coef/--ent-coef-end/--decay-steps and
                                 # DecayScheduleCallback for optional decay over training.
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
        )
    elif algo in ('SAC', 'A2C'):
        model = ALGOS[algo]('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    # Constructed once, reused across every .learn() call below (not
    # per-iteration) so its decay stays continuous over the whole
    # cumulative run instead of restarting each iteration -- see
    # DecayScheduleCallback's docstring. Inert (no-op) by default: both
    # end values default to their start values in main(), so a run that
    # doesn't explicitly configure decay behaves exactly as before.
    schedule_callback = DecayScheduleCallback(
        ent_coef_start=ent_coef, ent_coef_end=ent_coef_end,
        lr_start=learning_rate, lr_end=learning_rate_end,
        decay_steps=decay_steps)

    iteration = 0
    while True:
        iteration += 1
        print(f'Starting iteration {iteration}')
        # First .learn() call after a fresh --init-from load starts this
        # run's own timestep counter (and therefore checkpoint filenames
        # below) at 0, regardless of how many steps the source checkpoint
        # had already accumulated -- see this module's docstring.
        reset_num_timesteps = bool(init_from) and iteration == 1
        model.learn(total_timesteps=total_timesteps_per_iter,
                    reset_num_timesteps=reset_num_timesteps,
                    callback=schedule_callback)
        save_path = os.path.join(
            model_dir, f'{algo}_{total_timesteps_per_iter * iteration}_{fname}')
        model.save(save_path)
        print(f'Completed iteration {iteration}, model saved to {save_path}')


def test(env_id, algo, path_to_model, episodes, domain_randomization=False, log_csv=None):
    env = gym.make(env_id, render_mode='human', domain_randomization=domain_randomization)

    if algo not in ALGOS:
        raise ValueError(f'Unknown algorithm: {algo}')
    model = ALGOS[algo].load(path_to_model, env=env)

    # Sim steps execute far faster than real time -- without pacing, a
    # short (falls-quickly) episode blows by in a fraction of a second and
    # the viewer window closes before there's anything to watch.
    dt = env.unwrapped.model.opt.timestep

    csv_file = csv_writer = None
    if log_csv:
        import csv
        csv_file = open(log_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # x_m/y_m: world-frame torso position. dog.mjcf.xml is still in
        # the CAD's own native frame (+y=front, +x=right), NOT ROS
        # REP-103 -- so y_m is forward progress and x_m is sideways
        # drift, not the other way around. See dog_env.py's
        # _compute_reward_walk comment for why this matters.
        # feet_grounded: any calf-capsule/floor contact, knee end included.
        # feet_tip/feet_non_tip: same contacts, split by whether they're
        # near the actual foot site (standing on the foot) or not
        # (standing on the knee/shin) -- see dog_env.py's
        # _foot_tip_contact_count(). leg_a/b/c/d_contact: per-leg
        # breakdown of the same thing ('air'/'tip'/'nontip') -- for
        # spotting asymmetric issues (e.g. "front legs dragging, back
        # legs fine") the aggregate counts alone can't show.
        csv_writer.writerow(['episode', 'step', 'x_m', 'y_m', 'height_m', 'upright',
                              'feet_grounded', 'feet_tip', 'feet_non_tip',
                              'leg_a_contact', 'leg_b_contact', 'leg_c_contact', 'leg_d_contact',
                              'reward', 'terminated'])

    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        step = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if csv_writer:
                u = env.unwrapped
                torso_xy = u.data.xpos[u.torso_body_id][0:2]
                num_tip, num_non_tip = u._foot_tip_contact_count()
                per_leg = u._foot_contact_state_per_leg()
                csv_writer.writerow([episode, step, torso_xy[0], torso_xy[1], u._torso_height(),
                                      u._torso_up_z(), u._num_feet_grounded(), num_tip, num_non_tip,
                                      *per_leg, reward, terminated])
            step += 1
            time.sleep(dt)
        print(f'Episode {episode}: total_reward={total_reward:.2f}')

    if csv_file:
        csv_file.close()
        print(f'Wrote per-step log to {log_csv}')

    env.close()

    # mujoco.viewer.launch_passive's pause-key_callback thread doesn't join
    # cleanly on close() -- the interpreter otherwise hangs here forever
    # instead of returning control to the terminal. Safe to force-exit:
    # this is the terminal step of the CLI's --test path, nothing else is
    # pending. os._exit() skips the normal stdout/stderr flush, so do that
    # explicitly first or piped/redirected output (e.g. `| tee log.txt`)
    # silently loses the "Episode N: total_reward=" lines above.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description='Train or test a Dog-Stand-v0/Dog-Walk-v0 policy.')
    parser.add_argument('--env-id', default='Dog-Stand-v0', choices=['Dog-Stand-v0', 'Dog-Walk-v0'])
    parser.add_argument('--algo', default='PPO', choices=list(ALGOS.keys()))
    parser.add_argument('--env-type', default='subproc', choices=['dummy', 'subproc'])
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--timesteps-per-iter', type=int, default=1_000_000)
    # Not 'logs' -- collides visually with colcon's own log/ dir at the
    # workspace root (unrelated: this is SB3's tensorboard_log output).
    parser.add_argument('--log-dir', default='dogGymTrain_logs')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--domain-randomization', action='store_true')
    parser.add_argument('--n-steps', type=int, default=2048,
                         help='PPO only: rollout length per env before each update '
                              '(buffer size = n_steps * num_envs)')
    parser.add_argument('--batch-size', type=int, default=64,
                         help='PPO only: SGD minibatch size (must evenly divide '
                              'n_steps * num_envs)')
    parser.add_argument('--n-epochs', type=int, default=10,
                         help='PPO only: number of SGD passes over the rollout buffer per update')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                         help='PPO only: Adam learning rate (or decay START value, see '
                              '--learning-rate-end). Consider a LOWER value (e.g. 1e-4 or '
                              '5e-5) when using --init-from -- fine-tuning an already-good '
                              'policy at the fresh-training default risks destabilizing '
                              'existing behavior rather than refining it (observed directly, '
                              'see train.py\'s --init-from comment)')
    parser.add_argument('--learning-rate-end', type=float, default=None,
                         help='PPO only: learning rate floor, linearly decayed toward over '
                              '--decay-steps CUMULATIVE timesteps (see DecayScheduleCallback). '
                              'Default: same as --learning-rate, i.e. no decay -- opt-in. '
                              'Added 2026-07-30 to test whether a never-decaying learning '
                              'rate/ent_coef is why every stand reward variant tried degraded '
                              'with enough further training regardless of reward shape (TODO '
                              '16 in daniel_cl_context.md).')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                         help='PPO only: entropy bonus coefficient (or decay START value, see '
                              '--ent-coef-end). 0.01 matches this project\'s established '
                              'default (raised from 0.0 -- no exploration pressure let training '
                              'settle into a "sit still and level" local optimum).')
    parser.add_argument('--ent-coef-end', type=float, default=None,
                         help='PPO only: ent_coef floor, linearly decayed toward over '
                              '--decay-steps CUMULATIVE timesteps. Default: same as '
                              '--ent-coef, i.e. no decay -- opt-in, see --learning-rate-end.')
    parser.add_argument('--decay-steps', type=int, default=20_000_000,
                         help='PPO only: cumulative timesteps over which --ent-coef/'
                              '--learning-rate linearly decay toward their -end floors (holds '
                              'at the floor after). Only matters if either -end flag is set '
                              'to something different from its start value.')
    parser.add_argument('--init-from', metavar='PATH_TO_MODEL',
                         help='--train only, PPO only: load this checkpoint\'s policy/value '
                              'weights + optimizer state as the starting point instead of '
                              'random init, then continue training on --env-id (may differ '
                              'from the checkpoint\'s original task, e.g. fine-tune a stand '
                              'policy for walk -- see this module\'s docstring). This run\'s '
                              'own timestep counter/checkpoint filenames restart at 0.')
    parser.add_argument('--fname', default='dog_policy')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', metavar='PATH_TO_MODEL')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--log-csv', metavar='PATH',
                         help='--test only: write per-step height/upright/feet-grounded/reward to this CSV')
    args = parser.parse_args()

    # Unset -end flags default to their start value -- no decay unless
    # explicitly configured differently (see --ent-coef-end/
    # --learning-rate-end's help).
    ent_coef_end = args.ent_coef_end if args.ent_coef_end is not None else args.ent_coef
    learning_rate_end = args.learning_rate_end if args.learning_rate_end is not None else args.learning_rate

    if args.train:
        train(args.env_id, args.algo, args.fname, args.env_type, args.num_envs,
              args.timesteps_per_iter, args.log_dir, args.model_dir,
              args.domain_randomization, args.n_steps, args.batch_size, args.n_epochs,
              args.learning_rate, args.ent_coef, ent_coef_end, learning_rate_end,
              args.decay_steps, args.init_from)
    elif args.test:
        test(args.env_id, args.algo, args.test, args.episodes, args.domain_randomization, args.log_csv)
    else:
        parser.error('Pass either --train or --test PATH_TO_MODEL')


if __name__ == '__main__':
    main()
