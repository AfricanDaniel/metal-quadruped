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
import numpy as np
import torch.nn as nn
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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

    def __init__(self, ent_coef_start, ent_coef_end, lr_start, lr_end, decay_steps,
                 adjust_lr=True, verbose=0):
        super().__init__(verbose)
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_steps = decay_steps
        # --lr-schedule adaptive (2026-08-06): AdaptiveKLLearningRateCallback
        # owns model.lr_schedule instead in that mode -- both callbacks
        # writing to it every step would just fight each other, last-
        # write-wins each _on_step() in an undefined order. ent_coef decay
        # is independent and still applies either way.
        self.adjust_lr = adjust_lr

    def _on_step(self):
        progress = min(1.0, self.model.num_timesteps / self.decay_steps) if self.decay_steps > 0 else 1.0
        self.model.ent_coef = self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * progress
        if self.adjust_lr:
            current_lr = self.lr_start + (self.lr_end - self.lr_start) * progress
            self.model.lr_schedule = lambda progress_remaining, _lr=current_lr: _lr
        return True


class AdaptiveKLLearningRateCallback(BaseCallback):
    """Mirrors rsl_rl's PPO "adaptive" learning-rate schedule (the same
    mechanism .../go2-sim2real-locomotion-rl/examples/locomotion/final/
    go2_train_walk.py uses -- desired_kl=0.01, learning_rate=0.001,
    schedule="adaptive"; rsl_rl is the standard PPO implementation from
    ETH Zurich's legged-robot RL work, e.g. ANYmal/Isaac Lab). After
    each rollout's train() call, reads the ACTUAL KL divergence that
    update produced (SB3 already computes and logs this as
    'train/approx_kl') and adjusts the learning rate:
      - shrinks it (/1.5) if the update moved the policy MORE than 2x
        desired_kl -- too aggressive, risks destabilizing an
        already-decent policy (this project has direct prior evidence
        of exactly that: a fixed 3e-4 --init-from fine-tune made an
        already-good stand policy WORSE over 19M further steps, see
        that flag's own comment).
      - grows it (*1.5) if the update moved the policy LESS than 0.5x
        desired_kl -- barely changing anything, e.g. stuck reinforcing
        whatever local optimum it already found (the "tumbling" walk
        pattern) with no pressure to actually escape it.
      - otherwise leaves it alone.
    Clamped to rsl_rl's own [1e-5, 1e-2] bounds.

    Uses _on_rollout_start() (fires exactly once per rollout, right
    after the PREVIOUS train() call finished and before the NEXT one
    starts) rather than _on_step() (fires once per env step -- thousands
    of times per rollout, all reading the SAME stale approx_kl from the
    last train() call, which would misapply the same adjustment that
    many times before a fresh KL reading ever exists)."""

    def __init__(self, desired_kl=0.01, initial_lr=1e-3, min_lr=1e-5, max_lr=1e-2, verbose=0):
        super().__init__(verbose)
        self.desired_kl = desired_kl
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr

    def _on_rollout_start(self):
        approx_kl = self.model.logger.name_to_value.get('train/approx_kl')
        # None on the very first rollout -- no train() call has happened
        # yet, so there's nothing to react to; keep the initial LR.
        if approx_kl is not None:
            if approx_kl > self.desired_kl * 2.0:
                self.current_lr = max(self.min_lr, self.current_lr / 1.5)
            elif approx_kl < self.desired_kl / 2.0:
                self.current_lr = min(self.max_lr, self.current_lr * 1.5)
            if self.verbose:
                print(f'AdaptiveKLLearningRateCallback: approx_kl={approx_kl:.5f} '
                      f'(target={self.desired_kl}) -> lr={self.current_lr:.2e}')
        self.model.lr_schedule = lambda progress_remaining, _lr=self.current_lr: _lr

    def _on_step(self):
        return True


class GainRangeCurriculumCallback(BaseCallback):
    """Widens the (low, high) range DogEnv.reset() samples position_kp/
    position_kd from every episode, linearly, from a gentle starting
    range to a wider (real-hardware-matching) ending range, over
    `decay_steps` cumulative timesteps -- see DogEnv.__init__'s
    position_kp_range/position_kd_range comment for the full motivation
    (domain randomization over servo stiffness for sim-to-real
    robustness, PLUS a curriculum so early, near-random rollouts only
    ever see the gentle end of the range instead of risking a violent
    crash on a stiff-gain episode before the policy has any competence).

    Pushes the updated bounds into every parallel sub-environment via
    VecEnv.env_method('set_position_gain_range', ...) -- works for both
    DummyVecEnv (same process) and SubprocVecEnv (dispatches to each
    worker). Only takes effect on each env's NEXT reset(), not the
    episode already in progress -- see set_position_gain_range()'s own
    docstring for why that's deliberate.

    Throttled to update_interval_steps (default 2000) rather than every
    single _on_step() call -- env_method() round-trips to every
    subprocess worker, and the range only needs to move smoothly over
    MILLIONS of steps, so updating every step would be pure overhead for
    no meaningful precision gain."""

    def __init__(self, kp_range_start, kp_range_end, kd_range_start, kd_range_end,
                 decay_steps, update_interval_steps=2000, verbose=0):
        super().__init__(verbose)
        self.kp_range_start = kp_range_start
        self.kp_range_end = kp_range_end
        self.kd_range_start = kd_range_start
        self.kd_range_end = kd_range_end
        self.decay_steps = decay_steps
        self.update_interval_steps = update_interval_steps
        self._last_update_step = -update_interval_steps  # forces an update on the very first call

    def _on_step(self):
        if self.num_timesteps - self._last_update_step < self.update_interval_steps:
            return True
        self._last_update_step = self.num_timesteps
        progress = min(1.0, self.num_timesteps / self.decay_steps) if self.decay_steps > 0 else 1.0
        kp_range = tuple(
            s + (e - s) * progress for s, e in zip(self.kp_range_start, self.kp_range_end))
        kd_range = tuple(
            s + (e - s) * progress for s, e in zip(self.kd_range_start, self.kd_range_end))
        self.training_env.env_method('set_position_gain_range', kp_range, kd_range)
        return True


def make_env(env_id, domain_randomization, walk_start_pose, walk_height_fraction, control_mode,
             model_path=None, position_kp=None, position_kd=None,
             position_kp_range=None, position_kd_range=None):
    kwargs = dict(domain_randomization=domain_randomization,
                  walk_start_pose=walk_start_pose,
                  walk_height_fraction=walk_height_fraction,
                  control_mode=control_mode)
    # --model-path (2026-08-04): lets --test load a checkpoint against the
    # EXACT MJCF/ctrlrange it was actually trained on, overriding
    # control_mode's own default resolution -- needed once a shared file
    # like dog_torque.mjcf.xml gets regenerated with different values
    # (e.g. --torque-limit fixed 5.0 -> 20.0) out from under an older
    # checkpoint's saved action_space, which otherwise fails to load at
    # all (SB3's check_for_correct_spaces).
    if model_path is not None:
        kwargs['model_path'] = model_path
    # --position-kp/--position-kd (2026-08-05): see DogEnv.__init__'s
    # position_kp/position_kd comment -- runtime override of the MJCF's
    # baked-in <position> actuator gains, for comparing training under
    # softer gains against the real-hardware-matching default (60/4).
    # control_mode='position' only; harmless to pass otherwise since
    # DogEnv itself gates on control_mode too.
    if position_kp is not None:
        kwargs['position_kp'] = position_kp
    if position_kd is not None:
        kwargs['position_kd'] = position_kd
    # --position-kp-range-start/--position-kp-range-end (2026-08-06): the
    # env is constructed with the STARTING range -- GainRangeCurriculumCallback
    # widens it over training via set_position_gain_range(), this is just
    # the initial value so the range is well-defined before the callback's
    # first update. See DogEnv.__init__'s position_kp_range comment.
    if position_kp_range is not None:
        kwargs['position_kp_range'] = position_kp_range
    if position_kd_range is not None:
        kwargs['position_kd_range'] = position_kd_range
    return lambda: gym.make(env_id, **kwargs)


def train(env_id, algo, fname, env_type, num_envs, total_timesteps_per_iter,
          log_dir, model_dir, domain_randomization, n_steps, batch_size, n_epochs,
          learning_rate, ent_coef, ent_coef_end, learning_rate_end, decay_steps,
          init_from=None, walk_start_pose='standing', walk_height_fraction=0.90,
          control_mode='position', model_path=None, position_kp=None, position_kd=None,
          position_kp_range_start=None, position_kp_range_end=None,
          position_kd_range_start=None, position_kd_range_end=None,
          gain_curriculum_steps=None, lr_schedule='linear', desired_kl=0.01):
    print(f'Training {algo} on {env_id} ({env_type}, {num_envs} envs, '
          f'walk_start_pose={walk_start_pose}, walk_height_fraction={walk_height_fraction}, '
          f'control_mode={control_mode}, position_kp={position_kp}, position_kd={position_kd}, '
          f'position_kp_range_start={position_kp_range_start}, position_kp_range_end={position_kp_range_end}, '
          f'position_kd_range_start={position_kd_range_start}, position_kd_range_end={position_kd_range_end})')

    env_fns = [make_env(env_id, domain_randomization, walk_start_pose, walk_height_fraction,
                         control_mode, model_path, position_kp, position_kd,
                         position_kp_range_start, position_kd_range_start)
               for _ in range(num_envs)]
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
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=nn.Tanh,
    )
    # control_mode='torque' only (2026-08-04): SB3's PPO default
    # log_std_init=0.0 means the initial per-motor exploration std is
    # 1.0, in RAW action-space units. Position mode's action range is
    # already a few radians wide, so std=1 explores a real chunk of it.
    # Torque mode's range is the actuator's full +-ctrlrange (e.g. +-20
    # N*m) -- with std=1, exploration noise barely reaches +-3, well
    # short of the ~15-20 N*m a stand policy actually needs (measured
    # directly: a stand_policy_torque_v3 checkpoint at 1M steps still
    # had std~1.08 despite the effort_penalty/action_rate_penalty scale
    # fix, and mean |action| stayed 0.1-0.6 out of +-20 -- the policy
    # was never SAMPLING large torques during rollout collection in the
    # first place, so there was no gradient signal that they help,
    # independent of whether they'd be rewarded once tried). Scaling the
    # initial std to a meaningful fraction (1/3) of the actuator's own
    # range fixes this at the source rather than waiting for ent_coef to
    # slowly grow it. Reads the env's OWN action_space (not a hardcoded
    # 20) so this stays correct if --torque-limit ever changes.
    if control_mode == 'torque':
        torque_range = float(env.action_space.high[0])
        policy_kwargs['log_std_init'] = float(np.log(torque_range / 3.0))

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
    # --lr-schedule adaptive (2026-08-06): AdaptiveKLLearningRateCallback
    # takes over model.lr_schedule instead of DecayScheduleCallback's own
    # linear decay -- adjust_lr=False keeps DecayScheduleCallback around
    # ONLY for ent_coef decay in that mode, so the two don't fight over
    # the same attribute. Default ('linear') behaves exactly as before.
    schedule_callback = DecayScheduleCallback(
        ent_coef_start=ent_coef, ent_coef_end=ent_coef_end,
        lr_start=learning_rate, lr_end=learning_rate_end,
        decay_steps=decay_steps, adjust_lr=(lr_schedule != 'adaptive'))
    callbacks = [schedule_callback]
    if lr_schedule == 'adaptive':
        callbacks.append(AdaptiveKLLearningRateCallback(
            desired_kl=desired_kl, initial_lr=learning_rate))
    # Opt-in: only constructed if all 4 range endpoints were given (see
    # main()'s validation) -- a run that doesn't pass any of the
    # --position-kp-range-* flags behaves exactly as before, no curriculum.
    if position_kp_range_start is not None:
        callbacks.append(GainRangeCurriculumCallback(
            kp_range_start=position_kp_range_start, kp_range_end=position_kp_range_end,
            kd_range_start=position_kd_range_start, kd_range_end=position_kd_range_end,
            decay_steps=gain_curriculum_steps if gain_curriculum_steps is not None else decay_steps))
    callback = CallbackList(callbacks)

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
                    callback=callback)
        save_path = os.path.join(
            model_dir, f'{algo}_{total_timesteps_per_iter * iteration}_{fname}')
        model.save(save_path)
        print(f'Completed iteration {iteration}, model saved to {save_path}')


def test(env_id, algo, path_to_model, episodes, domain_randomization=False, log_csv=None,
         walk_start_pose='standing', walk_height_fraction=0.90, control_mode='position',
         model_path=None, position_kp=None, position_kd=None):
    kwargs = dict(render_mode='human', domain_randomization=domain_randomization,
                  walk_start_pose=walk_start_pose, walk_height_fraction=walk_height_fraction,
                  control_mode=control_mode)
    if model_path is not None:
        kwargs['model_path'] = model_path
    if position_kp is not None:
        kwargs['position_kp'] = position_kp
    if position_kd is not None:
        kwargs['position_kd'] = position_kd
    env = gym.make(env_id, **kwargs)

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
    parser.add_argument('--walk-start-pose', default='standing', choices=['standing', 'home'],
                         help='Dog-Walk-v0 only: episode starting pose. \'standing\' (default) '
                              'is the original behavior -- starts already standing, see this '
                              'module\'s docstring for why (composing a stand policy + a walk '
                              'policy was the original plan). \'home\' starts from the sitting/'
                              'home pose instead (same start state STAND uses) -- one policy '
                              'has to climb to standing height AND walk forward, inspired by '
                              'friend_code\'s approach of training a single policy end-to-end '
                              'rather than assuming a separate stand policy always runs first. '
                              'No effect on Dog-Stand-v0.')
    parser.add_argument('--walk-height-fraction', type=float, default=0.90,
                         help='Dog-Walk-v0 only: target torso height during walking, as a '
                              'fraction of STAND_HEIGHT_M (0.313m). Was a hardcoded constant '
                              '(WALK_HEIGHT_FRACTION in dog_env.py), now a CLI flag. A crouched '
                              'target (< 1.0) keeps the CoM lower and legs bent -- standard '
                              'practice for quadruped locomotion RL. No effect on Dog-Stand-v0.')
    parser.add_argument('--control-mode', default='position',
                         choices=['position', 'torque', 'torque_belt'],
                         help="'position' (default, unchanged): <position> PD actuators, action "
                              '= target joint angle -- the only mode that matches real hardware '
                              "(actuator package exposes position/velocity services, no torque "
                              "command yet). 'torque' (2026-08-03, sim-only comparison run): "
                              '<motor> actuators (dog_torque.mjcf.xml), action = raw joint '
                              "torque directly -- NOT deployable to real hardware as-is. 'torque_belt' "
                              '(2026-08-05, sim-only): same as torque, but calf motors additionally '
                              'get an automatic belt-compensation servo (dog_torque_belt.mjcf.xml, '
                              '<fixed> tendons) so the policy\'s calf output represents only genuine '
                              'flexion effort, not the "carried along by the thigh" motion the real '
                              'belt cancels for free -- see daniel_cl_context.md TODO 4\'s refinement. '
                              'Same --fname/checkpoints as any other run; keep torque-mode runs under '
                              'a clearly separate --fname so they never get mixed up with a '
                              'position-mode lineage (the action spaces are incompatible, same '
                              'as WALK_ACTION_RESIDUAL_RANGE_RAD\'s old-checkpoint break).')
    parser.add_argument('--model-path', default=None,
                         help='Override which MJCF file DogEnv loads, instead of control_mode\'s '
                              'own default (dog.mjcf.xml for position, dog_torque.mjcf.xml for '
                              'torque). Mainly for --test: a checkpoint\'s saved action_space must '
                              'exactly match ctrlrange in whatever MJCF is loaded now, so if a '
                              'shared file like dog_torque.mjcf.xml gets regenerated with '
                              'different values (e.g. --torque-limit) after a checkpoint was '
                              'trained, that older checkpoint needs this pointed at a copy of the '
                              'file with its original values, or it fails to load.')
    parser.add_argument('--position-kp', type=float, default=None,
                         help='control_mode=position only (2026-08-05): runtime override of the '
                              "MJCF's baked-in <position> actuator kp (default 60, matches real "
                              'hardware). Position-mode WALK has never learned successfully from '
                              'scratch even at 27-35M steps -- the hypothesis is that kp=60 turns '
                              'exploration noise into violent, near-instantaneous snapping instead '
                              'of graceful degradation, unlike torque mode. Must be passed together '
                              'with --position-kd (both or neither) -- see DogEnv.__init__\'s '
                              'position_kp/position_kd comment. Does not touch the MJCF file itself, '
                              'so real-hardware-matching runs that omit this flag are unaffected.')
    parser.add_argument('--position-kd', type=float, default=None,
                         help='control_mode=position only: runtime override of the MJCF\'s baked-in '
                              '<position> actuator kv/damping (default 4). See --position-kp.')
    parser.add_argument('--position-kp-range-start', type=float, nargs=2, default=None,
                         metavar=('LOW', 'HIGH'),
                         help='control_mode=position only, --train only (2026-08-06): domain-'
                              'randomizes position_kp per EPISODE from this (low, high) range at '
                              'the start of training, linearly widening to --position-kp-range-end '
                              'over --gain-curriculum-steps. Mutually exclusive with --position-kp/'
                              '--position-kd (fixed). All four of --position-kp-range-start/-end and '
                              '--position-kd-range-start/-end must be given together. Motivated by '
                              'real-hardware deployment of a kp=20-trained policy showing insufficient '
                              'torque to lift the back legs at kp=20, but training AT the real kp=60/'
                              'kd=8 from scratch producing a policy that lost ~27%% of standing height '
                              'in 0.14s early in a rollout (PPO_6000000_walk_position_scratch_v8) -- '
                              'PPO\'s exploration noise is roughly fixed-size in action-space units, '
                              'but the physical force it produces scales with the gain, so a near-'
                              'random early policy is far more likely to crash at a stiff gain than a '
                              'soft one. Starting the randomization range narrow and gentle (e.g. '
                              '10 30) and widening it toward the real value (e.g. 40 80) over training '
                              'gives both robustness (randomization) and a safe on-ramp (curriculum). '
                              'Example: --position-kp-range-start 10 30 --position-kp-range-end 40 80 '
                              '--position-kd-range-start 1 3 --position-kd-range-end 4 10.')
    parser.add_argument('--position-kp-range-end', type=float, nargs=2, default=None,
                         metavar=('LOW', 'HIGH'), help='See --position-kp-range-start.')
    parser.add_argument('--position-kd-range-start', type=float, nargs=2, default=None,
                         metavar=('LOW', 'HIGH'), help='See --position-kp-range-start.')
    parser.add_argument('--position-kd-range-end', type=float, nargs=2, default=None,
                         metavar=('LOW', 'HIGH'), help='See --position-kp-range-start.')
    parser.add_argument('--gain-curriculum-steps', type=int, default=None,
                         help='Cumulative timesteps over which --position-kp-range-*/--position-kd-'
                              'range-* linearly widen from start to end, then hold at end. Defaults '
                              'to --decay-steps if not set (same "how long is the ramp" knob as '
                              'ent_coef/learning_rate decay, but independently overridable here).')
    parser.add_argument('--lr-schedule', default='linear', choices=['linear', 'adaptive'],
                         help='PPO only, --train only (2026-08-06): "linear" (default, unchanged) '
                              'uses --learning-rate/--learning-rate-end/--decay-steps\' linear decay '
                              '(DecayScheduleCallback). "adaptive" instead mirrors rsl_rl\'s PPO '
                              '"adaptive" schedule (the same mechanism go2-sim2real-locomotion-rl\'s '
                              'go2_train_walk.py uses) -- after every rollout\'s train() call, reads '
                              'the ACTUAL KL divergence that update produced and shrinks the learning '
                              'rate if it moved the policy more than 2x --desired-kl (too aggressive), '
                              'grows it if less than 0.5x (barely changing -- e.g. stuck reinforcing a '
                              'local optimum with no pressure to escape it). Starts at --learning-rate, '
                              'clamped to [1e-5, 1e-2] (rsl_rl\'s own bounds). ent_coef decay (if '
                              'configured via --ent-coef-end) still applies independently either way.')
    parser.add_argument('--desired-kl', type=float, default=0.01,
                         help='--lr-schedule adaptive only: target KL divergence per update. Default '
                              'matches rsl_rl\'s own default (and go2_train_walk.py\'s).')
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

    if (args.position_kp is None) != (args.position_kd is None):
        raise ValueError('--position-kp and --position-kd must be passed together (both or neither)')

    gain_range_flags = (args.position_kp_range_start, args.position_kp_range_end,
                         args.position_kd_range_start, args.position_kd_range_end)
    if any(f is not None for f in gain_range_flags) and not all(f is not None for f in gain_range_flags):
        raise ValueError('--position-kp-range-start/-end and --position-kd-range-start/-end must '
                          'all be given together (all four, or none)')
    if gain_range_flags[0] is not None and args.position_kp is not None:
        raise ValueError('Pass either --position-kp/--position-kd (fixed) or the --position-kp-range-*/'
                          '--position-kd-range-* flags (randomized + curriculum), not both')

    if args.train:
        train(args.env_id, args.algo, args.fname, args.env_type, args.num_envs,
              args.timesteps_per_iter, args.log_dir, args.model_dir,
              args.domain_randomization, args.n_steps, args.batch_size, args.n_epochs,
              args.learning_rate, args.ent_coef, ent_coef_end, learning_rate_end,
              args.decay_steps, args.init_from, args.walk_start_pose, args.walk_height_fraction,
              args.control_mode, args.model_path, args.position_kp, args.position_kd,
              args.position_kp_range_start, args.position_kp_range_end,
              args.position_kd_range_start, args.position_kd_range_end,
              args.gain_curriculum_steps, args.lr_schedule, args.desired_kl)
    elif args.test:
        test(args.env_id, args.algo, args.test, args.episodes, args.domain_randomization, args.log_csv,
             args.walk_start_pose, args.walk_height_fraction, args.control_mode, args.model_path,
             args.position_kp, args.position_kd)
    else:
        parser.error('Pass either --train or --test PATH_TO_MODEL')


if __name__ == '__main__':
    main()
