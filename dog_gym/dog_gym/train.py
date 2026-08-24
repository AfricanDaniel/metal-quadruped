#!/usr/bin/env python3
"""Train or test a PPO/SAC/A2C policy on a Dog-Stand-v0/Dog-Walk-v0 environment (see dog_gym/envs/dog_env.py's module do..."""


import argparse
import os
import sys
import time

import dog_gym  # noqa: F401  (registers Dog-v0)
from dog_gym.envs.dog_env import (
    MAX_SLEW_DEG_PER_S, SLEW_CURRICULUM_TARGET_DEG_PER_S, WALK_FORWARD_PROGRESS_TARGET_M_S,
)
import gymnasium as gym
import numpy as np
import torch.nn as nn
import wandb
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

ALGOS = {'PPO': PPO, 'SAC': SAC, 'A2C': A2C}


class DecayScheduleCallback(BaseCallback):
    """Linearly decays ent_coef and learning_rate from their start value to a floor over `decay_steps` CUMULATIVE timesteps ..."""


    def __init__(self, ent_coef_start, ent_coef_end, lr_start, lr_end, decay_steps,
                 adjust_lr=True, verbose=0):
        super().__init__(verbose)
        self.ent_coef_start = ent_coef_start
        self.ent_coef_end = ent_coef_end
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.decay_steps = decay_steps
        # With --lr-schedule adaptive, AdaptiveKLLearningRateCallback owns model.lr_schedule instead
        self.adjust_lr = adjust_lr

    def _on_step(self):
        progress = min(1.0, self.model.num_timesteps / self.decay_steps) if self.decay_steps > 0 else 1.0
        self.model.ent_coef = self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * progress
        if self.adjust_lr:
            current_lr = self.lr_start + (self.lr_end - self.lr_start) * progress
            self.model.lr_schedule = lambda progress_remaining, _lr=current_lr: _lr
        return True


class AdaptiveKLLearningRateCallback(BaseCallback):
    """Mirrors rsl_rl's PPO "adaptive" learning-rate schedule (the standard PPO implementation from ETH Zurich's legged-robo..."""


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
    """Widens the (low, high) range DogEnv.reset() samples position_kp/ position_kd from every episode, linearly, from a gen..."""


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


class HomeStartCurriculumCallback(BaseCallback):
    """Widens the probability DogEnv.reset() uses to start a WALK episode from 'home' (tucked) instead of 'standing', linear..."""


    def __init__(self, prob_start, prob_end, decay_steps, update_interval_steps=2000, verbose=0):
        super().__init__(verbose)
        self.prob_start = prob_start
        self.prob_end = prob_end
        self.decay_steps = decay_steps
        self.update_interval_steps = update_interval_steps
        self._last_update_step = -update_interval_steps  # forces an update on the very first call

    def _on_step(self):
        if self.num_timesteps - self._last_update_step < self.update_interval_steps:
            return True
        self._last_update_step = self.num_timesteps
        progress = min(1.0, self.num_timesteps / self.decay_steps) if self.decay_steps > 0 else 1.0
        prob = self.prob_start + (self.prob_end - self.prob_start) * progress
        self.training_env.env_method('set_home_start_prob', prob)
        return True


class SlewCurriculumCallback(BaseCallback):
    """Tightens DogEnv's per-step slew clamp (see MAX_SLEW_DEG_PER_S/ SLEW_CURRICULUM_TARGET_DEG_PER_S's own comments in dog..."""


    def __init__(self, start_step, decay_steps, start_value=None, end_value=None,
                 update_interval_steps=2000, verbose=0):
        super().__init__(verbose)
        self.start_step = start_step
        self.decay_steps = decay_steps
        self.start_value = start_value if start_value is not None else MAX_SLEW_DEG_PER_S
        self.end_value = end_value if end_value is not None else SLEW_CURRICULUM_TARGET_DEG_PER_S
        self.update_interval_steps = update_interval_steps
        self._last_update_step = -update_interval_steps  # forces an update on the very first call

    def _on_step(self):
        if self.num_timesteps - self._last_update_step < self.update_interval_steps:
            return True
        self._last_update_step = self.num_timesteps
        if self.num_timesteps <= self.start_step:
            progress = 0.0
        elif self.decay_steps > 0:
            progress = min(1.0, (self.num_timesteps - self.start_step) / self.decay_steps)
        else:
            progress = 1.0
        value = self.start_value + (self.end_value - self.start_value) * progress
        self.training_env.env_method('set_max_slew_deg_per_s', value)
        return True


class ForwardSpeedCurriculumCallback(BaseCallback):
    """Raises DogEnv's WALK forward-speed reward target (see WALK_FORWARD_ PROGRESS_TARGET_M_S/set_walk_forward_progress_tar..."""


    def __init__(self, start_step, decay_steps, start_value=None, end_value=None,
                 update_interval_steps=2000, verbose=0):
        super().__init__(verbose)
        self.start_step = start_step
        self.decay_steps = decay_steps
        self.start_value = start_value if start_value is not None else WALK_FORWARD_PROGRESS_TARGET_M_S
        self.end_value = end_value if end_value is not None else WALK_FORWARD_PROGRESS_TARGET_M_S
        self.update_interval_steps = update_interval_steps
        self._last_update_step = -update_interval_steps  # forces an update on the very first call

    def _on_step(self):
        if self.num_timesteps - self._last_update_step < self.update_interval_steps:
            return True
        self._last_update_step = self.num_timesteps
        if self.num_timesteps <= self.start_step:
            progress = 0.0
        elif self.decay_steps > 0:
            progress = min(1.0, (self.num_timesteps - self.start_step) / self.decay_steps)
        else:
            progress = 1.0
        value = self.start_value + (self.end_value - self.start_value) * progress
        self.training_env.env_method('set_walk_forward_progress_target_m_s', value)
        return True


def make_env(env_id, domain_randomization, walk_start_pose, walk_height_fraction, control_mode,
             model_path=None, position_kp=None, position_kd=None,
             position_kp_range=None, position_kd_range=None, home_start_prob_start=None,
             max_slew_deg_per_s=None, gait_style='trot', joint_stiffness=None):
    kwargs = dict(domain_randomization=domain_randomization,
                  walk_start_pose=walk_start_pose,
                  walk_height_fraction=walk_height_fraction,
                  control_mode=control_mode,
                  gait_style=gait_style)
    # --model-path: lets --test load a checkpoint against the EXACT MJCF/ ctrlrange it was actually trained on, overriding control_mode's own default resolution
    if model_path is not None:
        kwargs['model_path'] = model_path
    # --position-kp/--position-kd: see DogEnv.__init__'s position_kp/ position_kd comment
    if position_kp is not None:
        kwargs['position_kp'] = position_kp
    if position_kd is not None:
        kwargs['position_kd'] = position_kd
    # --position-kp-range-start/--position-kp-range-end: the env is constructed with the STARTING range
    if position_kp_range is not None:
        kwargs['position_kp_range'] = position_kp_range
    if position_kd_range is not None:
        kwargs['position_kd_range'] = position_kd_range
    # --home-start-prob-start: env is constructed with the STARTING probability
    if home_start_prob_start is not None:
        kwargs['home_start_prob'] = home_start_prob_start
    # --max-slew-deg-per-s: runtime override of dog_env.py's MAX_SLEW_DEG_PER_S starting ceiling
    if max_slew_deg_per_s is not None:
        kwargs['max_slew_deg_per_s'] = max_slew_deg_per_s
    # --joint-stiffness: runtime override of generate_dog_mjcf.py's baked-in per-leg-joint stiffness="0"
    if joint_stiffness is not None:
        kwargs['joint_stiffness'] = joint_stiffness
    return lambda: gym.make(env_id, **kwargs)


def train(env_id, algo, fname, env_type, num_envs, total_timesteps_per_iter,
          log_dir, model_dir, domain_randomization, n_steps, batch_size, n_epochs,
          learning_rate, ent_coef, ent_coef_end, learning_rate_end, decay_steps,
          init_from=None, walk_start_pose='standing', walk_height_fraction=0.90,
          control_mode='position', model_path=None, position_kp=None, position_kd=None,
          position_kp_range_start=None, position_kp_range_end=None,
          position_kd_range_start=None, position_kd_range_end=None,
          gain_curriculum_steps=None, lr_schedule='linear', desired_kl=0.01,
          home_start_prob_start=None, home_start_prob_end=None,
          home_start_curriculum_steps=None, slew_curriculum_start_step=None,
          slew_curriculum_decay_steps=None, max_slew_deg_per_s=None,
          forward_speed_curriculum_start_step=None, forward_speed_curriculum_decay_steps=None,
          forward_speed_curriculum_target=None, gait_style='trot', joint_stiffness=None,
          use_sde=False, sde_sample_freq=-1,
          wandb_project='dog-quadruped', wandb_entity=None):
    # W&B logging is always on.
    wandb_config = {k: v for k, v in locals().items() if k not in ('wandb_project', 'wandb_entity')}
    os.makedirs(log_dir, exist_ok=True)  # wandb.init(dir=...) needs this to already exist
    wandb.init(project=wandb_project, entity=wandb_entity, name=fname,
               config=wandb_config, sync_tensorboard=True, dir=log_dir)

    print(f'Training {algo} on {env_id} ({env_type}, {num_envs} envs, '
          f'walk_start_pose={walk_start_pose}, walk_height_fraction={walk_height_fraction}, '
          f'control_mode={control_mode}, position_kp={position_kp}, position_kd={position_kd}, '
          f'position_kp_range_start={position_kp_range_start}, position_kp_range_end={position_kp_range_end}, '
          f'position_kd_range_start={position_kd_range_start}, position_kd_range_end={position_kd_range_end}, '
          f'home_start_prob_start={home_start_prob_start}, home_start_prob_end={home_start_prob_end}, '
          f'slew_curriculum_start_step={slew_curriculum_start_step}, '
          f'slew_curriculum_decay_steps={slew_curriculum_decay_steps}, '
          f'max_slew_deg_per_s={max_slew_deg_per_s}, '
          f'forward_speed_curriculum_start_step={forward_speed_curriculum_start_step}, '
          f'forward_speed_curriculum_decay_steps={forward_speed_curriculum_decay_steps}, '
          f'forward_speed_curriculum_target={forward_speed_curriculum_target}, '
          f'gait_style={gait_style}, joint_stiffness={joint_stiffness})')

    env_fns = [make_env(env_id, domain_randomization, walk_start_pose, walk_height_fraction,
                         control_mode, model_path, position_kp, position_kd,
                         position_kp_range_start, position_kd_range_start, home_start_prob_start,
                         max_slew_deg_per_s, gait_style, joint_stiffness)
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
    # control_mode='torque' only: SB3's PPO default log_std_init=0.0 means the initial per-motor exploration std is 1.0, in RAW action-space units.
    if control_mode == 'torque':
        torque_range = float(env.action_space.high[0])
        policy_kwargs['log_std_init'] = float(np.log(torque_range / 3.0))

    device = 'cuda'  # VM with a real GPU -- swap for 'cpu' on a dev machine (small MLP,
                      # GPU transfer overhead isn't worth it there)

    if init_from:
        # Fine-tuning: PPO.load() restores the saved policy/value network weights AND optimizer state, then rebinds to `env` (a NEW env, e.g.
        print(f'Fine-tuning from {init_from} on {env_id}')
        if algo != 'PPO':
            raise ValueError('--init-from is only wired up for PPO so far')
        model = PPO.load(
            init_from, env=env, device=device,
            n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
            learning_rate=learning_rate, ent_coef=ent_coef, tensorboard_log=log_dir)
    elif algo == 'PPO':
        # n_steps*num_envs is the rollout buffer size collected before each round of updates
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
            ent_coef=ent_coef,  # entropy bonus -- see --ent-coef/--ent-coef-end/--decay-steps
                                 # and DecayScheduleCallback for optional decay over training.
            vf_coef=0.5,
            max_grad_norm=0.5,
            # gSDE replaces PPO's default per-tick independent Gaussian action noise with noise sampled once per sde_sample_freq steps and held smoothly correlated in between (State- Dependent Exploration, Raffin et al.
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
        )
    elif algo in ('SAC', 'A2C'):
        model = ALGOS[algo]('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    # Constructed once, reused across every .learn() call below (not per-iteration) so its decay stays continuous over the whole cumulative run instead of restarting each iteration
    schedule_callback = DecayScheduleCallback(
        ent_coef_start=ent_coef, ent_coef_end=ent_coef_end,
        lr_start=learning_rate, lr_end=learning_rate_end,
        decay_steps=decay_steps, adjust_lr=(lr_schedule != 'adaptive'))
    callbacks = [schedule_callback]
    if lr_schedule == 'adaptive':
        callbacks.append(AdaptiveKLLearningRateCallback(
            desired_kl=desired_kl, initial_lr=learning_rate))
    # Opt-in: only constructed if all 4 range endpoints were given (see main()'s validation)
    if position_kp_range_start is not None:
        callbacks.append(GainRangeCurriculumCallback(
            kp_range_start=position_kp_range_start, kp_range_end=position_kp_range_end,
            kd_range_start=position_kd_range_start, kd_range_end=position_kd_range_end,
            decay_steps=gain_curriculum_steps if gain_curriculum_steps is not None else decay_steps))
    # Opt-in: only constructed if walk_start_pose='random' (see main()'s validation
    if walk_start_pose == 'random':
        callbacks.append(HomeStartCurriculumCallback(
            prob_start=home_start_prob_start, prob_end=home_start_prob_end,
            decay_steps=home_start_curriculum_steps if home_start_curriculum_steps is not None
            else decay_steps))
    # Opt-in: only constructed if --slew-curriculum-start-step was given
    if slew_curriculum_start_step is not None:
        callbacks.append(SlewCurriculumCallback(
            start_step=slew_curriculum_start_step,
            decay_steps=slew_curriculum_decay_steps if slew_curriculum_decay_steps is not None
            else decay_steps))
    # Opt-in: only constructed if --forward-speed-curriculum-start-step was given
    if forward_speed_curriculum_start_step is not None:
        callbacks.append(ForwardSpeedCurriculumCallback(
            start_step=forward_speed_curriculum_start_step,
            decay_steps=forward_speed_curriculum_decay_steps if forward_speed_curriculum_decay_steps is not None
            else decay_steps,
            end_value=forward_speed_curriculum_target))
    callback = CallbackList(callbacks)

    iteration = 0
    while True:
        iteration += 1
        print(f'Starting iteration {iteration}')
        # First .learn() call after a fresh --init-from load starts this run's own timestep counter (and therefore checkpoint filenames below) at 0, regardless of how many steps the source checkpoint had already accumulated
        reset_num_timesteps = bool(init_from) and iteration == 1
        # tb_log_name=fname: without an explicit name, SB3 falls back to its own auto-incrementing {algo}_{n} folder naming under log_dir, scoped per PROCESS not per fname, so unrelated runs can collide on the same folder.
        model.learn(total_timesteps=total_timesteps_per_iter,
                    reset_num_timesteps=reset_num_timesteps,
                    callback=callback,
                    tb_log_name=fname)
        save_path = os.path.join(
            model_dir, f'{algo}_{total_timesteps_per_iter * iteration}_{fname}')
        model.save(save_path)
        print(f'Completed iteration {iteration}, model saved to {save_path}')


def test(env_id, algo, path_to_model, episodes, domain_randomization=False, log_csv=None,
         walk_start_pose='standing', walk_height_fraction=0.90, control_mode='position',
         model_path=None, position_kp=None, position_kd=None, home_start_prob=None,
         max_slew_deg_per_s=None, gait_style='trot', joint_stiffness=None):
    kwargs = dict(render_mode='human', domain_randomization=domain_randomization,
                  walk_start_pose=walk_start_pose, walk_height_fraction=walk_height_fraction,
                  control_mode=control_mode, gait_style=gait_style)
    if model_path is not None:
        kwargs['model_path'] = model_path
    if position_kp is not None:
        kwargs['position_kp'] = position_kp
    if position_kd is not None:
        kwargs['position_kd'] = position_kd
    if joint_stiffness is not None:
        kwargs['joint_stiffness'] = joint_stiffness
    # walk_start_pose='random' only
    if home_start_prob is not None:
        kwargs['home_start_prob'] = home_start_prob
    if max_slew_deg_per_s is not None:
        kwargs['max_slew_deg_per_s'] = max_slew_deg_per_s
    env = gym.make(env_id, **kwargs)

    if algo not in ALGOS:
        raise ValueError(f'Unknown algorithm: {algo}')
    model = ALGOS[algo].load(path_to_model, env=env)

    # Sim steps execute far faster than real time
    dt = env.unwrapped.model.opt.timestep

    csv_file = csv_writer = None
    if log_csv:
        import csv
        csv_file = open(log_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # x_m/y_m: world-frame torso position.
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

    # mujoco.viewer.launch_passive's pause-key_callback thread doesn't join cleanly on close()
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
    parser.add_argument('--walk-start-pose', default='standing', choices=['standing', 'home', 'random'],
                         help='Dog-Walk-v0 only: episode starting pose. \'standing\' (default) '
                              'starts already standing (for composing a separate stand policy + '
                              'a walk policy). \'home\' starts from the sitting/home pose instead '
                              '(same start state STAND uses) -- one policy has to climb to '
                              'standing height AND walk forward. \'random\' mixes both within ONE '
                              'training run -- each episode independently coin-flips \'home\' vs '
                              '\'standing\' (weighted by --home-start-prob-start/-end, see those '
                              'flags), so a single policy learns to stand up from home AND walk. '
                              'No effect on Dog-Stand-v0.')
    parser.add_argument('--walk-height-fraction', type=float, default=0.90,
                         help='Dog-Walk-v0 only: target torso height during walking, as a '
                              'fraction of STAND_HEIGHT_M (0.313m). A crouched target (< 1.0) '
                              'keeps the CoM lower and legs bent -- standard practice for '
                              'quadruped locomotion RL. No effect on Dog-Stand-v0.')
    parser.add_argument('--control-mode', default='position',
                         choices=['position', 'torque', 'torque_belt'],
                         help="'position' (default): <position> PD actuators, action = target "
                              "joint angle -- the only mode that matches real hardware (actuator "
                              "package exposes position/velocity services, no torque command "
                              "yet). 'torque' (sim-only comparison): <motor> actuators "
                              '(dog_torque.mjcf.xml), action = raw joint torque directly -- NOT '
                              "deployable to real hardware as-is. 'torque_belt' (sim-only): same "
                              'as torque, but calf motors additionally get an automatic belt-'
                              'compensation servo (dog_torque_belt.mjcf.xml, <fixed> tendons) so '
                              'the policy\'s calf output represents only genuine flexion effort, '
                              'not the "carried along by the thigh" motion the real belt cancels '
                              'for free. Keep torque-mode runs under a clearly separate --fname '
                              'so they never get mixed up with a position-mode lineage -- the '
                              'action spaces are incompatible.')
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
                         help='control_mode=position only: runtime override of the MJCF\'s '
                              "baked-in <position> actuator kp (default 60, matches real "
                              'hardware). A softer kp lets exploration noise degrade gracefully '
                              'instead of snapping violently, which may help WALK learn from '
                              'scratch. Must be passed together with --position-kd (both or '
                              'neither) -- see DogEnv.__init__\'s position_kp/position_kd comment. '
                              'Does not touch the MJCF file itself, so real-hardware-matching runs '
                              'that omit this flag are unaffected.')
    parser.add_argument('--position-kd', type=float, default=None,
                         help='control_mode=position only: runtime override of the MJCF\'s baked-in '
                              '<position> actuator kv/damping (default 4). See --position-kp.')
    parser.add_argument('--position-kp-range-start', type=float, nargs=2, default=None,
                         metavar=('LOW', 'HIGH'),
                         help='control_mode=position only, --train only: domain-randomizes '
                              'position_kp per EPISODE from this (low, high) range at the start '
                              'of training, linearly widening to --position-kp-range-end over '
                              '--gain-curriculum-steps. Mutually exclusive with --position-kp/'
                              '--position-kd (fixed). All four of --position-kp-range-start/-end '
                              'and --position-kd-range-start/-end must be given together. PPO\'s '
                              'exploration noise is roughly fixed-size in action-space units, but '
                              'the physical force it produces scales with the gain, so a near-'
                              'random early policy is far more likely to crash at a stiff gain '
                              'than a soft one -- starting the randomization range narrow and '
                              'gentle and widening it toward the real gain over training gives '
                              'both robustness (randomization) and a safe on-ramp (curriculum). '
                              'Example: --position-kp-range-start 10 30 --position-kp-range-end '
                              '40 80 --position-kd-range-start 1 3 --position-kd-range-end 4 10.')
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
    parser.add_argument('--home-start-prob-start', type=float, default=0.0,
                         help='--walk-start-pose random only, --train only (2026-08-09): starting '
                              'probability (0-1) that a WALK episode begins from \'home\' instead of '
                              '\'standing\' -- HomeStartCurriculumCallback linearly widens this to '
                              '--home-start-prob-end over --home-start-curriculum-steps. Default 0.0 '
                              '(behaves like \'standing\' until the curriculum ramps up). Also used '
                              'directly (no ramp) as --test\'s single fixed probability when testing '
                              'a \'random\'-trained checkpoint.')
    parser.add_argument('--home-start-prob-end', type=float, default=0.5,
                         help='--walk-start-pose random, --train only: ending probability the '
                              'curriculum ramps toward. Default 0.5 (even mix at full ramp) rather '
                              'than 1.0, so the policy keeps seeing \'standing\'-start episodes '
                              'throughout training too, not just early on. See --home-start-prob-start.')
    parser.add_argument('--home-start-curriculum-steps', type=int, default=None,
                         help='Cumulative timesteps over which --home-start-prob-start/-end linearly '
                              'ramp, then hold at -end. Defaults to --decay-steps if not set, same '
                              'convention as --gain-curriculum-steps.')
    parser.add_argument('--slew-curriculum-start-step', type=int, default=None,
                         help='--train only (2026-08-16): cumulative timesteps to wait before '
                              'SlewCurriculumCallback starts tightening dog_env.py\'s per-step slew '
                              'clamp down from MAX_SLEW_DEG_PER_S (1000, loose) toward '
                              'SLEW_CURRICULUM_TARGET_DEG_PER_S (250, matching dog_deploy/'
                              'policy_node.py\'s real deployment clamp). Opt-in -- omitting this flag '
                              'entirely disables the curriculum (trains at the fixed 1000 the whole '
                              'run, unchanged from before). UNLIKE --gain-curriculum-steps/--home-'
                              'start-curriculum-steps (which both start ramping from step 0), this '
                              'stays pinned at 1000 until this step is reached -- tightening before a '
                              'gait has been discovered risks reproducing the 2026-08-07 MAX_SLEW_DEG_'
                              'PER_S failure (PPO\'s early exploration noise needs the full 1000 '
                              'headroom). 2_000_000 was chosen as a confident buffer past '
                              'gait discovery.')
    parser.add_argument('--slew-curriculum-decay-steps', type=int, default=None,
                         help='Cumulative timesteps AFTER --slew-curriculum-start-step over which the '
                              'slew clamp linearly tightens from 1000 to 250, then holds at 250. '
                              'Defaults to --decay-steps if not set, same convention as --gain-'
                              'curriculum-steps. 3_000_000 was chosen to finish '
                              'the anneal around 5M total steps with the default 2M start.')
    parser.add_argument('--max-slew-deg-per-s', type=float, default=None,
                         help='--train and --test (2026-08-16, user request): runtime override of '
                              "dog_env.py's MAX_SLEW_DEG_PER_S starting ceiling (module default 1000, "
                              'chosen for PPO gait-discovery headroom -- see that constant\'s own long '
                              'comment for the full history of why it needs to start this loose). '
                              'Independent of --slew-curriculum-start-step/-decay-steps, which control '
                              'whether/how this starting value tightens over training, not what it '
                              'starts at. Default None leaves the module constant untouched. Exposed '
                              'so different starting ceilings can be tried directly (e.g. investigating '
                              'whether a tighter start changes early-training stability) without '
                              'editing dog_env.py.')
    parser.add_argument('--forward-speed-curriculum-start-step', type=int, default=None,
                         help='WALK only, --train only (2026-08-18): cumulative timesteps to wait '
                              'before ForwardSpeedCurriculumCallback starts raising dog_env.py\'s '
                              'WALK_FORWARD_PROGRESS_TARGET_M_S (0.15) up toward --forward-speed-'
                              'curriculum-target. Opt-in -- omitting this flag entirely disables the '
                              'curriculum (trains at the fixed 0.15 the whole run, unchanged from '
                              'before). Same "pinned at the start value until this step, not ramping '
                              'from step 0" shape as --slew-curriculum-start-step, for the same reason: '
                              'raising the speed target before a policy has settled at its current '
                              'fine-tune point risks collapsing the forward_progress-gated terms '
                              '(upright_reward/trot_symmetry_reward/clearance gates) it currently relies '
                              'on. In practice this is best started from an already-converged checkpoint '
                              'rather than step 0 of a fresh run.')
    parser.add_argument('--forward-speed-curriculum-decay-steps', type=int, default=None,
                         help='Cumulative timesteps AFTER --forward-speed-curriculum-start-step over '
                              'which the speed target linearly rises from 0.15 to --forward-speed-'
                              'curriculum-target, then holds there. Defaults to --decay-steps if not '
                              'set, same convention as --slew-curriculum-decay-steps.')
    parser.add_argument('--forward-speed-curriculum-target', type=float, default=None,
                         help='WALK_FORWARD_PROGRESS_TARGET_M_S value the curriculum ramps toward. '
                              'Defaults to 0.15 (the module constant, i.e. a no-op) if the curriculum '
                              'is enabled without this set. 0.25-0.3 '
                              '(roughly double the original 0.15) is a reasonable first target given a real measured '
                              'steady-state of ~0.165 m/s on PPO_10500000_init_from_hf_v17_home_v1 -- '
                              'close to something already demonstrated as reachable rather than an '
                              'untested leap.')
    parser.add_argument('--gait-style', default='trot', choices=['trot', 'bound'],
                         help='WALK only, --train and --test (2026-08-18): which cross-leg gait '
                              'symmetry term DogEnv.__init__ rewards. \'trot\' (default, unchanged '
                              'behavior) diagonal pairs (a+d, b+c) offset from each other -- '
                              'currently a no-op weight-wise (WALK_TROT_SYMMETRY_WEIGHT zeroed, see '
                              'that term\'s DROPPED comment in dog_env.py), left in place only for '
                              'reference/restorability. \'bound\' rewards front pair (a+b) synced, '
                              'back pair (c+d) synced, offset from each other -- the pattern a genuine '
                              'aerial-phase running gait needs (all 4 feet airborne together, not '
                              'just fast walking), gated in via WALK_BOUND_SYMMETRY_GATE_START/_FULL '
                              'so it doesn\'t disturb an already-converged trot-style policy. '
                              'Should be trained as a '
                              '--gait-style bound run FROM SCRATCH (no --init-from), not by '
                              'fine-tuning a trot checkpoint, since the two terms actively disagree '
                              'on what "good" looks like.')
    parser.add_argument('--joint-stiffness', type=float, default=None,
                         help='WALK/STAND, --train and --test (2026-08-18, "T_FAKE" staged-training '
                              'idea -- trot_home_v1/trot_stand_1ms_v1 both plateaued far below their '
                              'forward-speed-curriculum targets): runtime override of generate_dog_'
                              'mjcf.py\'s baked-in per-leg-joint stiffness="0" (see that script\'s '
                              '2026-08-16 comment for why 0 is the physically-correct value). A '
                              'nonzero value adds a passive restoring spring on every leg joint, '
                              'which may make a fast gait easier for PPO to discover from scratch '
                              'even though it does not match real hardware -- the idea is to train a '
                              'first-phase policy under this easier physics (plus a loosened '
                              '--max-slew-deg-per-s), then --init-from it into a second phase with '
                              'this flag OMITTED (back to the correct 0) and the correct '
                              '--max-slew-deg-per-s, so the learned gait transfers into '
                              'physically-accurate dynamics instead of being learned under them from '
                              'a random policy. Omitting this flag entirely (default) leaves the '
                              'MJCF\'s own stiffness="0" untouched, matching every existing run.')
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
    parser.add_argument('--use-sde', action='store_true',
                         help='PPO only, --train only (2026-08-17): use gSDE (generalized State-'
                              'Dependent Exploration) instead of PPO\'s default per-tick independent '
                              'Gaussian action noise. gSDE samples noise once per --sde-sample-freq '
                              'steps and holds it smoothly correlated in between, instead of '
                              'redrawing independently every tick -- directly targets a confirmed '
                              'mechanism where, under a tight slew clamp, a single tick\'s ACTUAL '
                              'physical consequence is nearly identical regardless of how aggressive '
                              'the commanded action is, so independent per-tick noise never '
                              'accidentally produces the sustained multi-tick commitment a real leg '
                              'swing needs. Default False -- old behavior (independent per-tick '
                              'noise) unchanged unless passed.')
    parser.add_argument('--sde-sample-freq', type=int, default=-1,
                         help='--use-sde only: resample gSDE noise every N steps (-1 = SB3\'s own '
                              'default, sample once per rollout collection instead of periodically '
                              'mid-rollout).')
    parser.add_argument('--wandb-project', default='dog-quadruped',
                         help='Weights & Biases project every --train run logs to (always on -- '
                              'run `wandb login` once beforehand, see train.py\'s wandb.init() call).')
    parser.add_argument('--wandb-entity', default=None,
                         help='W&B entity (team/user) to log under -- default None uses whatever '
                              '`wandb login` set as your default entity.')
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
                              'with enough further training regardless of reward shape.')
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

    # Unset -end flags default to their start value
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
              args.gain_curriculum_steps, args.lr_schedule, args.desired_kl,
              args.home_start_prob_start, args.home_start_prob_end,
              args.home_start_curriculum_steps, args.slew_curriculum_start_step,
              args.slew_curriculum_decay_steps, args.max_slew_deg_per_s,
              args.forward_speed_curriculum_start_step, args.forward_speed_curriculum_decay_steps,
              args.forward_speed_curriculum_target, args.gait_style, args.joint_stiffness,
              args.use_sde, args.sde_sample_freq,
              args.wandb_project, args.wandb_entity)
    elif args.test:
        test(args.env_id, args.algo, args.test, args.episodes, args.domain_randomization, args.log_csv,
             args.walk_start_pose, args.walk_height_fraction, args.control_mode, args.model_path,
             args.position_kp, args.position_kd, args.home_start_prob_start, args.max_slew_deg_per_s,
             args.gait_style, args.joint_stiffness)
    else:
        parser.error('Pass either --train or --test PATH_TO_MODEL')


if __name__ == '__main__':
    main()
