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

# Standing height and STANDING_QPOS_DEG below: measured 2026-07-23 by
# directly hand-posing the robot in the interactive viewer (mjcf/
# save_pose.py) until all 4 limbs looked fully extended down and the
# robot was near its max height, then reading qpos back -- NOT derived
# from actuator/config/preset_pose.yaml's real hardware values this time
# (an earlier version of this file was, and got the calf angles wrong,
# see below). This is real, directly-observed sim ground truth, the most
# reliable source available without hardware access.
STAND_HEIGHT_M = 0.313
STAND_HEIGHT_TOLERANCE_M = 0.02  # user: "small range allowed for error"
# Sitting/home height (see FALL_HEIGHT_M's comment) -- used below as the
# "0% standing progress" reference point for gating the uprightness
# reward, so it can't be collected just by sitting still and level.
# Matches the user's own hand-posed "home" capture almost exactly
# (0.1405m, qpos~0 on every joint -- consistent with preset_pose.yaml's
# "home" preset being all-zero too).
SIT_HEIGHT_M = 0.14

# Motor-id order (1..8), hand-verified sim qpos at the standing pose
# above (NOT converted from real hardware degrees -- see the note above).
#
# IMPORTANT, UNRESOLVED (2026-07-23): this replaced an earlier version
# derived from actuator/config/preset_pose.yaml's real "standing" preset,
# converted through motor_mapping.yaml's documented sign. That version
# had the right idea for the thighs (see THIGH_SYMMETRY_SIGN -- the
# left/right mirrored-axis finding still holds and is confirmed again by
# this new data) but was flat wrong for the calves: it assumed calves
# barely move (~0-6deg, taken straight from the real preset) when hand-
# posing the sim shows they need to rotate almost as much as the thighs
# (~100-120deg) to reach a fully extended stance. The calves ALSO turned
# out to have a non-uniform sign across legs, just split front/back
# instead of left/right: leg_a/leg_b (front) go negative, leg_c/leg_d
# (back) go positive, consistent across both the semi-standing and full-
# standing hand-posed captures (not noise). See CALF_SYMMETRY_SIGN below.
# Root cause of why real-hardware-preset-converted values don't match
# this sim's own kinematics is still not identified for either joint --
# deliberately NOT touching motor_mapping.yaml (shared with dog_deploy,
# hardware-facing) on unverified sim-only findings. Check both thigh AND
# calf signs on real hardware before trusting a sim-trained policy's
# action signs at deployment time.
STANDING_QPOS_DEG = np.array([-116.970, -120.650, -105.144, 97.769, -110.876, 104.772, 98.950, 104.126])

# Motor-order (0-indexed, i.e. motor_id - 1) indices of the 4 thigh and 4
# calf joints -- used by the stand task's symmetry penalty. Neither axis
# is uniform across all 4 legs (see the note above) -- THIGH_SYMMETRY_SIGN/
# CALF_SYMMETRY_SIGN correct for it before comparing, so the penalty
# rewards the real symmetric standing configuration instead of fighting it.
SYMMETRIC_THIGH_IDX = [0, 3, 4, 7]  # motors 1, 4, 5, 8 (leg_a, leg_b, leg_c, leg_d)
THIGH_SYMMETRY_SIGN = np.array([-1, 1, -1, 1])  # left/right split, see note above
SYMMETRIC_CALF_IDX = [1, 2, 5, 6]   # motors 2, 3, 6, 7 (leg_a, leg_b, leg_c, leg_d)
CALF_SYMMETRY_SIGN = np.array([-1, -1, 1, 1])   # front/back split, see note above

# Sitting/home height settles around 0.14m (measured the same way as
# STAND_HEIGHT_M, at qpos=0) -- FALL_HEIGHT_M must stay below that or the
# stand task's own starting pose would immediately count as "fallen".
FALL_HEIGHT_M = 0.10
MAX_TILT_RAD = 0.9  # ~51 degrees from vertical before an episode ends
MAX_EPISODE_STEPS = 1000
NUM_MOTORS = 8

# Per-motor max target slew rate, applied every step() -- see the long
# comment where it's used. Matches dog_deploy/policy_node.py's real
# safety clamp (5deg per 20Hz tick = 100deg/s), not a fresh guess.
MAX_SLEW_DEG_PER_S = 100.0

# How close a ground contact must be to a leg's foot site to count as
# "standing on the foot" rather than "standing on the knee/shin" -- see
# _foot_tip_contact_count(). The calf capsule (knee->foot) is ~0.077m
# long (2x its half-length) with a 0.017m radius, so 0.03m is close to
# the tip specifically, not a point along the middle of the segment.
FOOT_CONTACT_RADIUS_M = 0.03


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

        # Each leg's collision capsule is named "<leg>_calf" (same name as
        # the calf joint, different MuJoCo namespace) and runs knee->foot
        # as ONE capsule (no separate foot geom) -- so a contact anywhere
        # along it, knee-end included, counts as "grounded" by geom alone.
        # foot_site_ids (the existing "<leg>_foot" site, already at the
        # true foot-tip position) lets contact position be compared
        # against the tip specifically, to tell "standing on the foot"
        # apart from "standing on the knee/shin" -- see
        # _foot_tip_contact_count().
        self.calf_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'{leg}_calf')
            for leg in ('leg_a', 'leg_b', 'leg_c', 'leg_d')
        ]
        self.foot_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{leg}_foot')
            for leg in ('leg_a', 'leg_b', 'leg_c', 'leg_d')
        ]

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
            # step()'s slew-rate limiter clips each new action around
            # prev_action -- leaving prev_action at its zero default here
            # while the legs are actually at the standing pose would make
            # the very first step() think the last commanded target was 0
            # and yank every leg toward 0 (rate-limited, so a slow fake
            # "collapse" rather than an instant snap, but still wrong).
            self.prev_action = qpos_rad.astype(np.float32)
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
        # Rate-limit how far any single motor's target can move per step,
        # same idea (and same underlying number) as dog_deploy/policy_node.py's
        # real-hardware safety clamp: that clamps 5deg per 20Hz control tick
        # = 100deg/s. Sim steps at 100Hz (dog.mjcf.xml's 0.01s timestep, one
        # physics step per env.step(), no frame-skip), so the equivalent
        # per-sim-step limit is 100deg/s / 100Hz = 1deg/step -- same overall
        # speed limit, just expressed at the sim's own step rate. Without
        # this, a policy is free to snap straight to any target every step
        # (visibly "stands up too fast"), which also doesn't reflect what
        # the real, rate-limited deployment path will actually allow --
        # training under the same limit avoids that train/deploy mismatch.
        max_delta_rad = np.radians(MAX_SLEW_DEG_PER_S) * self.model.opt.timestep
        action = np.clip(action, self.prev_action - max_delta_rad, self.prev_action + max_delta_rad)
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

    def _num_feet_grounded(self):
        """How many of the 4 calf capsules are in actual contact with the
        floor geom right now (0-4) -- a real physical ground-contact
        check via MuJoCo's own contact list, not a height-based guess.
        Doesn't distinguish WHERE on the capsule contact happens (knee
        end counts the same as the foot tip) -- see
        _foot_tip_contact_count() for that."""
        grounded = set()
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.floor_geom_id and c.geom2 in self.calf_geom_ids:
                grounded.add(c.geom2)
            elif c.geom2 == self.floor_geom_id and c.geom1 in self.calf_geom_ids:
                grounded.add(c.geom1)
        return len(grounded)

    def _foot_tip_contact_count(self):
        """(num_tip, num_non_tip): of the calf-floor contacts right now,
        how many are within FOOT_CONTACT_RADIUS_M of that leg's actual
        foot site (standing on the foot, as intended) vs. further away
        along the capsule (standing on the knee/shin -- e.g. a trained
        walk policy observed doing exactly this, since
        _num_feet_grounded() alone can't tell the two apart)."""
        num_tip = 0
        num_non_tip = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.floor_geom_id and c.geom2 in self.calf_geom_ids:
                calf_geom_id = c.geom2
            elif c.geom2 == self.floor_geom_id and c.geom1 in self.calf_geom_ids:
                calf_geom_id = c.geom1
            else:
                continue
            leg_idx = self.calf_geom_ids.index(calf_geom_id)
            foot_pos = self.data.site_xpos[self.foot_site_ids[leg_idx]]
            if np.linalg.norm(np.array(c.pos) - foot_pos) < FOOT_CONTACT_RADIUS_M:
                num_tip += 1
            else:
                num_non_tip += 1
        return num_tip, num_non_tip

    def _foot_placement_terms(self):
        """(tip_reward, non_tip_penalty) shared by both tasks: tip_reward
        in [0, 1] is the fraction of the 4 feet grounded at the actual
        foot tip; non_tip_penalty is <=0, scaled by how many contacts are
        happening away from the tip instead (knee/shin dragging -- e.g.
        a trained walk policy observed doing exactly this)."""
        num_tip, num_non_tip = self._foot_tip_contact_count()
        return num_tip / 4.0, -1.0 * num_non_tip

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

        # Gate uprightness by standing progress (0 at sitting height, 1 at
        # the target) instead of paying it out unconditionally -- an
        # unconditional +upright term made "sit still and level" collect
        # most of the achievable reward on its own (observed: trained
        # policy barely moved 2 of 8 motors, ep_rew_mean matched the
        # sit-still-forever estimate almost exactly), removing any real
        # pressure to actually climb to standing height. Now sitting
        # level scores near zero on this term instead of the max.
        height_progress = np.clip(
            (self._torso_height() - SIT_HEIGHT_M) / (STAND_HEIGHT_M - SIT_HEIGHT_M), 0.0, 1.0)
        upright_reward = self._torso_up_z() * height_progress  # 1.0 = perfectly level AND at height

        motor_qpos = self.data.qpos[self.motor_qpos_adr]
        # THIGH_SYMMETRY_SIGN/CALF_SYMMETRY_SIGN correct for the left/right
        # (thigh) and front/back (calf) mirrored joint axes before
        # comparing -- see STANDING_QPOS_DEG's comment.
        thigh_spread = np.var(motor_qpos[SYMMETRIC_THIGH_IDX] * THIGH_SYMMETRY_SIGN)
        calf_spread = np.var(motor_qpos[SYMMETRIC_CALF_IDX] * CALF_SYMMETRY_SIGN)
        symmetry_penalty = -1.0 * (thigh_spread + calf_spread)

        # Standing still means staying in place, not just staying up --
        # qvel[0:2] is the free joint's world-frame x/y linear velocity.
        drift_penalty = -0.1 * float(np.dot(self.data.qvel[0:2], self.data.qvel[0:2]))

        # All 4 feet should be on the ground AT THE FOOT TIP once actually
        # standing (user observed a trained policy reaching the target
        # height on only 3 legs, and separately a walk policy standing on
        # its knees instead of its feet -- _foot_placement_terms()
        # distinguishes tip contact from knee/shin contact, unlike a raw
        # calf-capsule contact check). Gated by height_progress the same
        # way upright_reward is, since feet legitimately leave the ground
        # (and can brush the knee/shin) while still climbing from the
        # sitting pose -- only penalize/reward this once close to standing.
        tip_reward, non_tip_penalty = self._foot_placement_terms()
        grounded_reward = (tip_reward + non_tip_penalty) * height_progress

        return (
            3.0 * height_reward
            + height_bonus
            + 2.0 * upright_reward
            + 1.5 * grounded_reward
            + symmetry_penalty
            + drift_penalty
            + self._common_penalties(action)
        )

    def _compute_reward_walk(self, action):
        # dog.mjcf.xml is still in the CAD's own native frame (+y=front,
        # +x=right -- see the model's own header comment), NOT yet
        # remapped to ROS REP-103 (+x=forward). qvel[1] is therefore the
        # real forward velocity here, not qvel[0] (an earlier version of
        # this line used qvel[0] under the wrong assumption that +x was
        # already forward -- confirmed as the cause of a policy that
        # leaned/drifted rightward instead of walking forward after a
        # full 22M-timestep training run, since qvel[0] literally rewards
        # rightward motion in this frame). Revisit this line together with
        # the frame remap if that ever gets applied.
        forward_velocity_reward = self.data.qvel[1]
        upright_reward = self._torso_up_z()
        # Loose height regularizer (much lower weight than the stand
        # task's) -- just discourages crawling/belly-flopping, doesn't
        # demand the exact standing height while walking.
        height_reward = -abs(self._torso_height() - STAND_HEIGHT_M)

        # Walk on the feet, not the knees/shins -- this task previously had
        # NO foot-placement term at all, which is exactly why a trained
        # policy was observed walking on its knees: nothing in the reward
        # distinguished that from walking on its feet. Not gated by height
        # progress (unlike the stand task's version) since the walk task
        # is always meant to be standing/walking, never mid-climb.
        tip_reward, non_tip_penalty = self._foot_placement_terms()

        return (
            2.0 * forward_velocity_reward
            + 0.5 * upright_reward
            + 0.2 * height_reward
            + 1.5 * tip_reward
            + non_tip_penalty
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
