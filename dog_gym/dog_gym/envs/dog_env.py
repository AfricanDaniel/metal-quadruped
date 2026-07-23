"""Gymnasium environment for the 8-DOF dog, on the modern `mujoco` bindings
(not the deprecated `mujoco_py`).

Action space: 8 target joint angles (radians), one per motor_1..motor_8
position actuator defined in dog_description/mjcf/dog.mjcf.xml, in that
same order -- this is exactly the quantity actuator's SetMotorTargets
service takes (after dog_deploy applies the sign flips from
motor_mapping.yaml), so a trained policy's raw output needs no further
reshaping to run on real hardware.

Observation: deliberately proprioceptive-only, matching what's actually
available on the real robot -- no absolute torso position/orientation (no
motion-capture/localization exists), and no privileged simulator state.
It's the 8 joint angles + 8 joint velocities (explicitly gathered in
motor_1..motor_8 order via motor_mapping.yaml -- raw qpos/qvel array order
follows MJCF body-tree order, which is NOT motor order, so it can't be
sliced directly) + simulated IMU (accelerometer + gyro, matching what
dog_imu's real LSM6DSO32 driver publishes) + the previous action (lets the
policy be penalized for jerky control, which matters for the real
gearboxes). dog_deploy builds this exact vector from
actuator/read_motor_positions + dog_imu on real hardware.

Reward computation is free to use privileged simulator state (true torso
height/tilt, true forward velocity) even though the observation can't --
that's standard practice and does not create a sim-to-real gap, since
reward is only used during training, never at deployment.

Two tasks share this one env/observation/action space (`task='stand'` or
`task='walk'`, registered as separate gym ids `Dog-Stand-v0`/`Dog-Walk-v0`
in dog_gym/__init__.py so they train as distinct policies/checkpoints via
train.py's --env-id) -- only reset()'s initial pose and _compute_reward()
differ:
  - 'stand': episode starts from the sitting/home pose (qpos=0, matches
    actuator/config/preset_pose.yaml's "home" preset and dog.mjcf.xml's
    CAD-captured qpos=0 -- confirmed to agree via mjcf/save_pose.py,
    2026-07-23). Reward pushes the torso up to STAND_HEIGHT_M and keeps
    it level and symmetric, no forward-motion term.
  - 'walk': episode starts already standing (STANDING_QPOS_DEG below) and
    rewards forward velocity -- learning to stand up AND walk from a
    crouch simultaneously is a harder joint problem than composing two
    single-purpose policies, hence separate tasks per the user's request.
"""

import os
import time

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from gymnasium import spaces

DOG_DESCRIPTION_SHARE = get_package_share_directory('dog_description')
DEFAULT_MODEL_PATH = os.path.join(DOG_DESCRIPTION_SHARE, 'mjcf', 'dog.mjcf.xml')
DEFAULT_MOTOR_MAPPING_PATH = os.path.join(DOG_DESCRIPTION_SHARE, 'config', 'motor_mapping.yaml')

# Standing height: measured 2026-07-23 by setting qpos to STANDING_QPOS_DEG
# below (the real "standing" preset from actuator/config/preset_pose.yaml,
# converted through motor_mapping.yaml's sign) and reading the settled
# torso height back via mjcf/save_pose.py -- 0.277m. Replaces an earlier
# 0.340m guess (a hand-measured bracket+thigh+calf sum against the old,
# since-rebuilt CAD -- stale, no longer matches this model's link lengths).
STAND_HEIGHT_M = 0.277
STAND_HEIGHT_TOLERANCE_M = 0.02  # user: "small range allowed for error"

# Same "standing" preset, motor-id order (1..8), magnitudes taken from
# actuator/config/preset_pose.yaml's real "standing" values (sign-converted
# through motor_mapping.yaml where that convention is verified correct),
# used as the walk task's reset pose.
#
# IMPORTANT, UNRESOLVED (2026-07-23): motor_mapping.yaml documents thigh
# "positive = away from front" as the same physical meaning for all 4
# legs, and that IS verified true in the current dog.mjcf.xml (checked
# directly -- every leg's foot moves in the away-from-front direction for
# positive thigh qpos). But a hinge sweeps an arc, and a full +-180deg
# sweep found each leg's actual foot-height MINIMUM (i.e. the standing-
# extended configuration) at asymmetric signs: leg_a/leg_c (the two
# right-side legs) minimize around thigh=-75deg, leg_b/leg_d (left-side)
# around thigh=+75deg -- traced to the two sides having mirrored world-
# frame rotation axes (+X vs -X), an expected property of a mirror-
# symmetric leg mount. Converting the real "standing" preset through
# motor_mapping.yaml's documented sign gives POSITIVE angles for legs
# a/c too, which -- per this sim's own kinematics -- would fold those two
# legs up instead of standing on them. Since the real "standing" preset
# is a working pose on the actual hardware, real hardware and the current
# CAD-derived sim kinematics disagree for the two right-side legs
# specifically. Root cause not identified (CAD mounting difference for
# the right side vs. a motor_mapping.yaml sign error vs. something else)
# -- deliberately NOT changing motor_mapping.yaml (shared with dog_deploy,
# hardware-facing) on this unverified a guess. Per user decision
# (2026-07-23): use the real preset's per-motor MAGNITUDES (a
# hardware-calibrated fact) but flip the SIGN of the thigh angle for
# leg_a/leg_c specifically, matching what this sim's own verified
# kinematics require to actually stand -- i.e. trust real magnitudes,
# trust this sim's own verified sign, don't trust the cross-referenced
# motor_mapping.yaml sign for these two joints until the real
# discrepancy above is investigated on hardware/CAD.
STANDING_QPOS_DEG = np.array([-107.41, 5.91, -0.49, 107.62, -113.66, 4.68, 1.14, 107.25])

# Motor-order (0-indexed, i.e. motor_id - 1) indices of the 4 thigh and 4
# calf joints -- used by the stand task's symmetry penalty. Calf axis is
# uniform across all 4 legs (verified), so raw qpos compares directly.
# Thigh axis is NOT uniform (see the note above) -- THIGH_SYMMETRY_SIGN
# corrects for it before comparing, so the penalty rewards the real
# symmetric standing configuration instead of fighting it.
SYMMETRIC_THIGH_IDX = [0, 3, 4, 7]  # motors 1, 4, 5, 8 (leg_a, leg_b, leg_c, leg_d)
THIGH_SYMMETRY_SIGN = np.array([-1, 1, -1, 1])  # leg_a/leg_c flipped, see note above
SYMMETRIC_CALF_IDX = [1, 2, 5, 6]   # motors 2, 3, 6, 7

# Sitting/home height settles around 0.14m (measured the same way as
# STAND_HEIGHT_M, at qpos=0) -- FALL_HEIGHT_M must stay below that or the
# stand task's own starting pose would immediately count as "fallen".
FALL_HEIGHT_M = 0.10
MAX_TILT_RAD = 0.9  # ~51 degrees from vertical before an episode ends
MAX_EPISODE_STEPS = 1000
NUM_MOTORS = 8


def load_motor_joint_names(motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH):
    """Returns MJCF joint names ["leg_a_thigh", ...] indexed motor 1..8,
    resolved from motor_mapping.yaml's {leg, joint} pairs -- the same file
    dog_deploy uses, so sim and real can never disagree on motor ordering.
    """
    with open(motor_mapping_path) as f:
        mapping = yaml.safe_load(f)['motors']
    return [
        f"{mapping[motor_id]['leg']}_{mapping[motor_id]['joint']}"
        for motor_id in range(1, NUM_MOTORS + 1)
    ]


class DogEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 50}

    def __init__(self, model_path=DEFAULT_MODEL_PATH,
                 motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH,
                 render_mode=None, domain_randomization=False, task='stand'):
        super().__init__()
        if task not in ('stand', 'walk'):
            raise ValueError(f"task must be 'stand' or 'walk', got {task!r}")
        self.task = task
        self.model_path = model_path
        self.render_mode = render_mode
        self.domain_randomization = domain_randomization

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self._paused = False

        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        self.floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.default_floor_friction = self.model.geom_friction[self.floor_geom_id].copy()

        # motor_1..motor_8 -> this joint's qpos/qvel address, so the
        # observation is built in a fixed, motor-id order regardless of
        # the MJCF's body-tree traversal order.
        joint_names = load_motor_joint_names(motor_mapping_path)
        self.motor_qpos_adr = np.array(
            [self.model.joint(name).qposadr[0] for name in joint_names])
        self.motor_dof_adr = np.array(
            [self.model.joint(name).dofadr[0] for name in joint_names])

        ctrlrange = self.model.actuator_ctrlrange.copy()
        self.action_space = spaces.Box(
            low=ctrlrange[:, 0].astype(np.float32),
            high=ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32)

        self.prev_action = np.zeros(NUM_MOTORS, dtype=np.float32)

        # motor qpos (8) + motor qvel (8) + IMU sensordata + prev_action (8)
        obs_dim = NUM_MOTORS + NUM_MOTORS + self.model.nsensordata + NUM_MOTORS
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.prev_action = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._step_count = 0

        if self.task == 'walk':
            # Start already standing -- see module docstring: composing a
            # stand policy + a walk policy is the plan, not one policy
            # learning to stand up and walk forward at once.
            qpos_rad = np.radians(STANDING_QPOS_DEG)
            if self.domain_randomization:
                qpos_rad = qpos_rad + self.np_random.uniform(-0.02, 0.02, size=NUM_MOTORS)
            self.data.qpos[self.motor_qpos_adr] = qpos_rad
            # The free joint's own z (qpos[2]) still defaults to the
            # model's qpos=0 spawn height (0.287m, chosen to clear the
            # floor at the CAD-captured/sitting leg pose) -- leaving it
            # there while the legs are suddenly snapped to the standing
            # pose drives the feet deep through the floor on the very
            # first physics step (extended legs + unchanged torso height
            # = feet well below z=0), producing a violent contact impulse
            # that was observed to tip the robot over and leave it stuck
            # in a partial, off-balance crouch instead of standing
            # (verified: torso settled around 0.19m tilted ~16deg, not
            # 0.277m level, regardless of how far the commanded target
            # was pushed beyond the real standing angles -- confirms the
            # cause was the initial impact, not insufficient PD gain).
            # Starting at the real settled standing height directly
            # avoids that transient.
            self.data.qpos[2] = STAND_HEIGHT_M
        # 'stand' task starts from the model's default qpos=0 (the
        # sitting/home pose, see STAND_HEIGHT_M's comment) -- nothing to do.

        if self.domain_randomization:
            self._randomize_ground()
        else:
            self.model.geom_friction[self.floor_geom_id] = self.default_floor_friction

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _randomize_ground(self):
        friction = self.np_random.uniform(0.3, 0.8)
        self.model.geom_friction[self.floor_geom_id] = [friction, friction * 0.1, friction * 0.1]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        if self.render_mode == 'human':
            self._ensure_viewer()
            # Space bar toggles self._paused via _on_key. mj_step only
            # advances while unpaused; the viewer keeps redrawing (and
            # stays responsive to mouse/keyboard) either way.
            while self._paused and self.renderer.is_running():
                self.renderer.sync()
                time.sleep(0.02)

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_fallen()
        truncated = self._step_count >= MAX_EPISODE_STEPS

        self.prev_action = action.astype(np.float32)

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        motor_qpos = self.data.qpos[self.motor_qpos_adr]
        motor_qvel = self.data.qvel[self.motor_dof_adr]
        return np.concatenate([
            motor_qpos,
            motor_qvel,
            self.data.sensordata,
            self.prev_action,
        ]).astype(np.float32)

    def _torso_height(self):
        return self.data.xpos[self.torso_body_id][2]

    def _torso_up_z(self):
        """World-frame z-component of the torso's local up axis: 1.0 =
        perfectly upright, drops towards 0 as the robot tips over."""
        xmat = self.data.xmat[self.torso_body_id].reshape(3, 3)
        return xmat[2, 2]

    def _is_fallen(self):
        if self._torso_height() < FALL_HEIGHT_M:
            return True
        if self._torso_up_z() < np.cos(MAX_TILT_RAD):
            return True
        return False

    def _common_penalties(self, action):
        """Terms both tasks share: IMU-based stability penalties + effort/
        action-rate (hardware-friendliness) + a per-step survival bonus."""
        # sensordata layout matches the <sensor> block in dog.mjcf.xml:
        # [0:3] accelerometer, [3:6] gyro.
        linear_accel = self.data.sensordata[0:3]
        angular_vel = self.data.sensordata[3:6]
        gravity_m_s2 = 9.81
        accel_shock_penalty = -0.01 * (np.linalg.norm(linear_accel) - gravity_m_s2) ** 2
        angular_vel_penalty = -0.02 * float(np.dot(angular_vel, angular_vel))

        effort_penalty = -0.001 * float(np.dot(action, action))
        action_rate_penalty = -0.01 * float(np.sum((action - self.prev_action) ** 2))

        survival_bonus = 0.05

        return (accel_shock_penalty + angular_vel_penalty
                + effort_penalty + action_rate_penalty + survival_bonus)

    def _compute_reward(self, action):
        if self.task == 'stand':
            return self._compute_reward_stand(action)
        return self._compute_reward_walk(action)

    def _compute_reward_stand(self, action):
        height_error = abs(self._torso_height() - STAND_HEIGHT_M)
        height_reward = -height_error
        # Bonus on top of the continuous shaping term once within
        # tolerance -- gives "close enough" a distinctly better return
        # than "still closing in", per the requested error tolerance.
        height_bonus = 0.2 if height_error < STAND_HEIGHT_TOLERANCE_M else 0.0

        upright_reward = self._torso_up_z()  # 1.0 = perfectly level ("flat")

        motor_qpos = self.data.qpos[self.motor_qpos_adr]
        # THIGH_SYMMETRY_SIGN corrects for leg_a/leg_c's mirrored joint
        # axis before comparing -- see STANDING_QPOS_DEG's comment.
        thigh_spread = np.var(motor_qpos[SYMMETRIC_THIGH_IDX] * THIGH_SYMMETRY_SIGN)
        calf_spread = np.var(motor_qpos[SYMMETRIC_CALF_IDX])
        symmetry_penalty = -1.0 * (thigh_spread + calf_spread)

        # Standing still means staying in place, not just staying up --
        # qvel[0:2] is the free joint's world-frame x/y linear velocity.
        drift_penalty = -0.1 * float(np.dot(self.data.qvel[0:2], self.data.qvel[0:2]))

        return (
            3.0 * height_reward
            + height_bonus
            + 2.0 * upright_reward
            + symmetry_penalty
            + drift_penalty
            + self._common_penalties(action)
        )

    def _compute_reward_walk(self, action):
        # qvel[0] is the free joint's world-frame x-linear-velocity --
        # privileged sim state, fine to use for reward (see module docstring).
        forward_velocity_reward = self.data.qvel[0]
        upright_reward = self._torso_up_z()
        # Loose height regularizer (much lower weight than the stand
        # task's) -- just discourages crawling/belly-flopping, doesn't
        # demand the exact standing height while walking.
        height_reward = -abs(self._torso_height() - STAND_HEIGHT_M)

        return (
            2.0 * forward_velocity_reward
            + 0.5 * upright_reward
            + 0.2 * height_reward
            + self._common_penalties(action)
        )

    def _ensure_viewer(self):
        if self.renderer is None:
            import mujoco.viewer
            self.renderer = mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self._on_key)
            # Group 3 = the simple collision-primitive geoms (capsules/box)
            # kept alongside the real CAD visual meshes (group 2) for fast
            # collision -- hide them by default so they don't overlap-render
            # with the meshes.
            self.renderer.opt.geomgroup[3] = 0

    def _on_key(self, keycode):
        if keycode == 32:  # GLFW_KEY_SPACE
            self._paused = not self._paused

    def render(self):
        if self.render_mode != 'human':
            return
        self._ensure_viewer()
        self.renderer.sync()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
