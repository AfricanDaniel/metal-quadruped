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

# Walk task's target torso height, as a fraction of STAND_HEIGHT_M
# (2026-07-28, user request: "the robot stays at 75% its maximum height
# ... this way the robot is guaranteed to stay on 4 legs"). A crouched
# target -- legs more bent than full standing extension -- is standard
# practice for quadruped locomotion RL: lower CoM for stability, more
# leg travel available for the swing phase without needing near-maximal
# joint excursions. Easy to change: edit this one fraction, everything
# else derives from it. See _compute_reward_walk()'s height_reward.
WALK_HEIGHT_FRACTION = 0.75
WALK_TARGET_HEIGHT_M = WALK_HEIGHT_FRACTION * STAND_HEIGHT_M
# Sitting/home height (see FALL_HEIGHT_M's comment) -- used below as the
# "0% standing progress" reference point for gating the uprightness
# reward, so it can't be collected just by sitting still and level.
# Matches the user's own hand-posed "home" capture almost exactly
# (0.1405m, qpos~0 on every joint -- consistent with preset_pose.yaml's
# "home" preset being all-zero too).
SIT_HEIGHT_M = 0.14

# Motor-id order (1..8), hand-verified sim qpos at the standing pose
# above (NOT converted from real hardware degrees).
#
# RECAPTURED 2026-07-27 (via mjcf/save_pose.py, post-AXIS_FLIP AND
# post-calf-range-fix -- supersedes both the original 2026-07-23 capture
# and its 2026-07-26 AXIS_FLIP renegotiation, which predated the belt-
# decoupling fix's change to what a calf's action/obs means and was
# flagged stale ever since, see daniel_cl_context.md's TODO 9). Every
# value fits comfortably inside the real-hardware-measured joint ranges
# (generate_dog_mjcf.py's JOINT_RANGE_OVERRIDES_DEG) -- confirms this
# hand-posed stance is physically achievable on the real robot, not just
# a sim artifact. Torso height at capture was 0.3132m, matching
# STAND_HEIGHT_M=0.313 almost exactly -- a genuine standing pose, not a
# coincidence. User note: a second, real-hardware-referenced candidate
# pose looked "more appealing" by eye but didn't match how the CURRENT
# sim actually stands -- deliberately using this sim-native capture
# instead, since this constant's job is to seed DogEnv's own kinematics
# consistently, not to exactly reproduce an external reference.
STANDING_QPOS_DEG = np.array([107.507, 104.071, -86.789, -98.804, 98.743, 98.011, -93.049, -103.804])

# Motor-order (0-indexed, i.e. motor_id - 1) indices of the 4 thigh and 4
# calf joints -- used by the stand task's symmetry penalty. Neither axis
# is uniform across all 4 legs (see the note above) -- THIGH_SYMMETRY_SIGN/
# CALF_SYMMETRY_SIGN correct for it before comparing, so the penalty
# rewards the real symmetric standing configuration instead of fighting it.
SYMMETRIC_THIGH_IDX = [0, 3, 4, 7]  # motors 1, 4, 5, 8 (leg_a, leg_b, leg_c, leg_d)
# [1,-1,1,-1] since the 2026-07-26 AXIS_FLIP (generate_dog_mjcf.py):
# sim's raw directions now match the real robot motor-for-motor, and the
# real robot's left/right motors are mirror-mounted -- so "extend toward
# standing" is POSITIVE qpos for the right legs' thighs (a, c) and
# NEGATIVE for the left legs' (b, d). These signs normalize all four to
# positive before the variance comparison. Derived from (and verified
# against) the same direct FK sweep that determined AXIS_FLIP itself --
# if the axes ever change again, re-derive this from FK, never from a
# documented convention (see daniel_cl_context.md's sign saga).
THIGH_SYMMETRY_SIGN = np.array([1, -1, 1, -1])
SYMMETRIC_CALF_IDX = [1, 2, 5, 6]   # motors 2, 3, 6, 7 (leg_a, leg_b, leg_c, leg_d)
# [1,-1,1,-1] since the same AXIS_FLIP (was [-1,-1,1,1] under the old
# axes): "calf towards front" is now POSITIVE qpos for the right legs'
# calves (a, c) and NEGATIVE for the left legs' (b, d), matching the
# real robot's mirror-mounted motors -- left/right split now, not the
# old front/back split (motors 2 and 7 flipped; 3 and 6 didn't).
CALF_SYMMETRY_SIGN = np.array([1, -1, 1, -1])
# NOT YET AUDITED (flagged 2026-07-26, unlike THIGH_SYMMETRY_SIGN above):
# _compute_reward_stand() computes calf_spread from self.data.qpos
# directly -- the RAW, thigh-relative calf hinge value -- NOT the
# ABSOLUTE (belt-decoupled) calf angle _get_obs() computes via
# calf_belt_sign. Since a calf's raw qpos = absolute_angle -
# calf_belt_sign*thigh_qpos, two legs' raw values aren't comparable on
# their own -- they're each confounded by wherever that leg's OWN thigh
# happens to be, which can legitimately differ leg-to-leg during a
# stand-up motion. This sign was derived from (and is being applied to)
# that same raw, thigh-confounded quantity, so it may be self-consistent
# rather than actively wrong the way THIGH_SYMMETRY_SIGN was -- but it's
# unclear this raw-qpos-based symmetry measure is even the right thing to
# reward post-belt-decoupling-fix, vs. computing it on the ABSOLUTE calf
# angle instead (matching what the observation/action actually use).
# Deliberately not changed without further verification -- see
# daniel_cl_context.md.

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

# Real-hardware-realistic SENSOR noise (motor position/velocity + IMU),
# opt-in via domain_randomization -- see _get_obs()'s comment for why
# this was added (2026-07-27) and PLACEHOLDER STATUS (typical magnetic-
# encoder + LSM6DSO32 noise scales, not a formal characterization of
# this specific hardware -- refine using real dog_deploy log_csv data if
# higher fidelity is wanted).
MOTOR_POS_NOISE_STD_RAD = np.radians(0.1)     # ~0.1deg encoder noise
MOTOR_VEL_NOISE_STD_RAD_S = np.radians(0.5)   # velocity noisier than position (often derived by differentiating)
ACCEL_NOISE_STD_M_S2 = 0.05
GYRO_NOISE_STD_RAD_S = 0.01

# How close a ground contact must be to a leg's foot site to count as
# "standing on the foot" rather than "standing on the knee/shin" -- see
# _foot_tip_contact_count(). The calf capsule (knee->foot) is ~0.077m
# long (2x its half-length) with a 0.017m radius, so 0.03m is close to
# the tip specifically, not a point along the middle of the segment.
FOOT_CONTACT_RADIUS_M = 0.03

# Target foot height (world z, meters) during swing phase for
# _foot_clearance_reward() -- borrowed from the standard "foot clearance"
# reward pattern in quadruped locomotion RL. Placeholder, not derived from
# a real measured gait -- a few cm is enough to clear the ground without
# demanding an unnaturally high step.
FOOT_CLEARANCE_TARGET_M = 0.03


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

        # Belt/pulley coupling compensation (real robot confirmed
        # 2026-07-25): each leg's calf motor drives its lower pulley
        # through a timing belt to an upper pulley mounted on the
        # (non-rotating-with-the-thigh) torso/shoulder; the lower pulley
        # itself free-spins on a bearing relative to the thigh. Net
        # effect: rotating ONLY the thigh does not change the calf's real
        # orientation at all -- the belt cancels the "carried along"
        # rotation a plain serial hinge would otherwise apply. Only the
        # calf motor's own rotation changes the calf's real (torso-
        # relative, i.e. ABSOLUTE) angle. dog.mjcf.xml's calf joint is a
        # plain hinge relative to the thigh body (a normal serial elbow)
        # -- the opposite behavior. Rather than rebuild the belt/pulley
        # physically (two more bodies + a tendon per leg, real pulley
        # radii not measured), this compensates at the control/
        # observation layer: action[calf]/obs[calf] represent the same
        # ABSOLUTE angle the real motor's own encoder reports (matching
        # actuator/dog_deploy, which read/command that motor directly and
        # need no change at all -- the real hardware already behaves this
        # way natively). Only the raw ctrl sent to the MJCF's
        # thigh-relative position actuator, and the raw qpos/qvel read
        # back from it, need converting -- see step()/_get_obs(). qpos
        # itself (physical DOF, reward/contact/geometry code) is
        # completely unaffected, still the same MuJoCo tree-composed
        # hinge angle it always was.
        #
        # Sign, corrected 2026-07-25, and NOT uniform across legs: ctrl_calf
        # = action_calf + calf_belt_sign*qpos_thigh. An initial
        # from-first-principles derivation assumed a single global minus
        # sign; user caught by eye (dog_gym.manual_motor_control) that the
        # calf was visibly rotating the SAME way as the thigh instead of
        # counter-rotating. Verified with a proper metric this time (the
        # calf body's FULL 3D orientation change via the rotation-matrix
        # angle -- comparing only one axis vector, as an earlier check did,
        # silently missed rotation about that same axis, which is exactly
        # what was happening): the minus sign leaves ~48deg of real drift
        # at thigh=20deg on leg_a specifically (thigh and calf rotations
        # ADDING instead of cancelling), while plus cuts it to ~2deg
        # (matching ordinary PD droop under load). But testing all 4 legs
        # the same way showed leg_a/leg_d need PLUS while leg_b/leg_c need
        # MINUS -- flipping to a single global plus (the first fix
        # attempted) would have fixed leg_a/leg_d while breaking leg_b/
        # leg_c, which were already correct.
        #
        # calf_belt_sign is therefore computed per leg, not hardcoded,
        # from the actual loaded geometry -- robust to the CAD
        # regeneration this is about to go through (cylindrical-mate fix
        # rolling out to all 4 legs) rather than a guessed per-leg-letter
        # table that would silently go stale on the next re-export. The
        # sign only depends on whether the knee axis, expressed in the
        # thigh's own frame (rotate by the calf body's quat, which is
        # relative to its thigh parent), points the same way as the hip
        # axis (same direction -> rotations in thigh-frame ADD for a
        # fixed calf qpos, so cancelling needs MINUS) or the opposite way
        # (needs PLUS). Confirmed this closed form reproduces the
        # empirically-measured per-leg signs above exactly (see
        # daniel_cl_context.md's "flip the sign" section for the
        # verification script).
        self.calf_idx = np.array(
            [i for i, n in enumerate(joint_names) if n.endswith('_calf')])
        self.calf_thigh_idx = np.array([
            joint_names.index(joint_names[i][:-len('_calf')] + '_thigh')
            for i in self.calf_idx
        ])
        self.calf_thigh_qpos_adr = self.motor_qpos_adr[self.calf_thigh_idx]
        self.calf_thigh_dof_adr = self.motor_dof_adr[self.calf_thigh_idx]

        calf_belt_sign = []
        for i in self.calf_idx:
            calf_joint_id = self.model.joint(joint_names[i]).id
            hip_joint_id = self.model.joint(joint_names[joint_names.index(
                joint_names[i][:-len('_calf')] + '_thigh')]).id
            calf_body_id = self.model.jnt_bodyid[calf_joint_id]
            knee_axis_in_thigh_frame = np.zeros(3)
            mujoco.mju_rotVecQuat(
                knee_axis_in_thigh_frame, self.model.jnt_axis[calf_joint_id],
                self.model.body_quat[calf_body_id])
            same_direction = np.dot(knee_axis_in_thigh_frame, self.model.jnt_axis[hip_joint_id]) > 0
            calf_belt_sign.append(-1.0 if same_direction else 1.0)
        self.calf_belt_sign = np.array(calf_belt_sign)

        # NOTE: for calf motors, this is the raw MJCF actuator's
        # thigh-relative ctrlrange, used as-is for the ABSOLUTE action
        # space too (see calf_idx's comment above) -- an approximation,
        # since the true reachable absolute range shifts with the
        # paired thigh's current angle. Not a safety gap: the joint's own
        # <joint range>/limited="true"> in dog.mjcf.xml is the real hard
        # clamp MuJoCo enforces regardless of what ctrl requests, this
        # box is only PPO's action-distribution bound.
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
            # prev_action lives in the policy's ABSOLUTE action space (see
            # calf_idx's comment in __init__), so the calf entries of this
            # raw (thigh-relative) qpos snapshot need the same
            # qpos[calf]-qpos[paired thigh] conversion _get_obs() uses.
            prev_action = qpos_rad.copy()
            prev_action[self.calf_idx] -= self.calf_belt_sign * qpos_rad[self.calf_thigh_idx]
            self.prev_action = prev_action.astype(np.float32)
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

        # Belt/pulley compensation (see calf_idx's comment in __init__,
        # including the sign note): action[calf] is an ABSOLUTE angle;
        # dog.mjcf.xml's calf actuator servos a thigh-RELATIVE hinge, so
        # ADD the thigh's current angle before writing ctrl. Thigh entries
        # (and any non-calf motor) pass through unchanged.
        ctrl = action.copy()
        ctrl[self.calf_idx] = action[self.calf_idx] + self.calf_belt_sign * self.data.qpos[self.calf_thigh_qpos_adr]
        self.data.ctrl[:] = ctrl

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
        # Belt/pulley compensation (see calf_idx's comment in __init__):
        # convert the calf hinge's raw, thigh-relative qpos/qvel into the
        # ABSOLUTE angle/rate the real motor's own encoder would report,
        # so the policy's observation matches actuator/dog_deploy's
        # read_motor_positions on real hardware with no further change
        # needed there.
        motor_qpos = self.data.qpos[self.motor_qpos_adr].copy()
        motor_qvel = self.data.qvel[self.motor_dof_adr].copy()
        motor_qpos[self.calf_idx] -= self.calf_belt_sign * self.data.qpos[self.calf_thigh_qpos_adr]
        motor_qvel[self.calf_idx] -= self.calf_belt_sign * self.data.qvel[self.calf_thigh_dof_adr]
        sensordata = self.data.sensordata

        # SENSOR NOISE (2026-07-27, opt-in via domain_randomization):
        # real-hardware deployment of a "converged"-looking stand policy
        # showed persistent chatter at a settled stand pose, traced
        # directly to the raw policy action itself being unstable at
        # steady state (see action_rate_penalty's comment in
        # _common_penalties()) -- sim's observation had ALWAYS been
        # perfectly noise-free, unlike anything the real robot's
        # encoders/IMU actually report, so a policy never has any reason
        # to learn robustness to small measurement noise on its own.
        # Added here so training can optionally include that noise (see
        # MOTOR_POS_NOISE_STD_RAD's comment for magnitudes/placeholder
        # status). NOT applied to prev_action -- that's the policy's own
        # last commanded value, not a sensor reading, and stays exact on
        # both sim and real (dog_deploy/policy_node.py's prev_action is
        # likewise the exact last clamped command, never a measurement).
        if self.domain_randomization:
            motor_qpos = motor_qpos + self.np_random.normal(
                0.0, MOTOR_POS_NOISE_STD_RAD, size=NUM_MOTORS)
            motor_qvel = motor_qvel + self.np_random.normal(
                0.0, MOTOR_VEL_NOISE_STD_RAD_S, size=NUM_MOTORS)
            sensordata = sensordata.copy()
            sensordata[0:3] += self.np_random.normal(0.0, ACCEL_NOISE_STD_M_S2, size=3)
            sensordata[3:6] += self.np_random.normal(0.0, GYRO_NOISE_STD_RAD_S, size=3)

        return np.concatenate([
            motor_qpos,
            motor_qvel,
            sensordata,
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

    def _foot_contact_per_leg(self):
        """[bool x4] (leg_a, leg_b, leg_c, leg_d order, matching
        calf_geom_ids/foot_site_ids): True if that leg has ANY floor
        contact right now (tip or knee/shin), False if fully airborne
        (mid-swing). Used by _foot_clearance_reward() to know which legs
        should be judged on swing height vs. which are legitimately
        planted."""
        contacted = [False] * len(self.calf_geom_ids)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == self.floor_geom_id and c.geom2 in self.calf_geom_ids:
                contacted[self.calf_geom_ids.index(c.geom2)] = True
            elif c.geom2 == self.floor_geom_id and c.geom1 in self.calf_geom_ids:
                contacted[self.calf_geom_ids.index(c.geom1)] = True
        return contacted

    def _foot_clearance_reward(self):
        """Rewards a SWINGING (non-contacting) foot for actually being
        elevated, not dragged along the ground -- adapted from the
        standard feet_air_time/foot_clearance pattern used in quadruped
        locomotion RL (e.g. legged_gym/ANYmal-style reward suites).

        Why this is needed on top of _foot_placement_terms(): that reward
        only judges WHERE contact happens when it happens (tip vs.
        knee/shin) -- it says nothing about whether the leg ever lifts at
        all. A policy that drags the true foot tip along the ground
        continuously, never swinging, would score perfectly on tip-
        contact while still never producing a real walking gait -- which
        can look a lot like "walking on its knees" even with tip contact
        technically satisfied. This term specifically rewards legs that
        ARE mid-swing for being off the ground, closing that gap.

        Returns a value in [0, 1]: 1.0 means every currently-swinging leg
        is at or above FOOT_CLEARANCE_TARGET_M; legs currently in contact
        don't count either way (this isn't a "lift all feet" reward, only
        a "when swinging, actually swing" one)."""
        contacted = self._foot_contact_per_leg()
        swinging = [not c for c in contacted]
        if not any(swinging):
            return 1.0  # no leg is swinging right now (e.g. standing still) -- nothing to penalize
        total = 0.0
        for i, site_id in enumerate(self.foot_site_ids):
            if swinging[i]:
                foot_z = self.data.site_xpos[site_id][2]
                total += min(max(foot_z, 0.0) / FOOT_CLEARANCE_TARGET_M, 1.0)
        return total / sum(swinging)

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

    def _foot_contact_state_per_leg(self):
        """['air'|'tip'|'nontip'] x4, in (leg_a, leg_b, leg_c, leg_d)
        order -- per-leg breakdown of _foot_tip_contact_count(), for
        diagnosing asymmetric issues (e.g. "front legs dragging, back
        legs fine") that the aggregate counts alone can't show. If a leg
        somehow registers more than one contact this step, the LAST one
        found wins -- rare (a capsule vs. a plane is usually one contact
        point) and only matters for this diagnostic, not for reward."""
        state = ['air'] * len(self.calf_geom_ids)
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
            is_tip = np.linalg.norm(np.array(c.pos) - foot_pos) < FOOT_CONTACT_RADIUS_M
            state[leg_idx] = 'tip' if is_tip else 'nontip'
        return state

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
        # Raised 10x (-0.01 -> -0.1) 2026-07-27: real-hardware deployment
        # of a "converged"-looking stand policy (calfFix_stand_policy_v5,
        # 3-6M timesteps) showed persistent tick-to-tick chatter even at
        # a fully settled stand pose -- traced to the RAW POLICY ACTION
        # itself (not firmware, not real sensor noise, not the deploy
        # clamp -- confirmed directly in clean sim with zero real-
        # hardware noise involved): consecutive deterministic actions at
        # a genuinely settled state swung by 10-30deg step-to-step.
        # DogEnv.step()'s own slew clamp (MAX_SLEW_DEG_PER_S) rate-limits
        # the PHYSICAL consequence during training, same as
        # dog_deploy/policy_node.py's clamp does at deploy time -- but
        # neither clamp can fix an underlying signal that keeps
        # reversing direction; they only rate-limit how fast it chases
        # itself back and forth forever. At the OLD weight (-0.01), a
        # representative multi-motor jitter contributed roughly -0.01 to
        # total reward per step -- negligible next to height_reward
        # (weight 3.0) or upright_reward (weight 2.0), which routinely
        # swing by whole units, or even the flat +0.05 survival_bonus
        # alone. Checked checkpoints 1M/3M/4M/5M/6M of that run: no
        # real downward trend in swing magnitude across 3-6M, i.e. more
        # timesteps under the old weight wasn't fixing it on its own.
        # New weight makes a typical jittering motor cost a real,
        # competitive fraction of per-step reward instead of background
        # noise. Still a placeholder value, not derived from any formal
        # tuning -- re-adjust based on the next retrain's actual smoothness
        # (same iterative-tuning pattern as position_kp/position_kd on
        # the actuator side, see daniel_cl_context.md).
        action_rate_penalty = -0.1 * float(np.sum((action - self.prev_action) ** 2))

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
        # Targets WALK_TARGET_HEIGHT_M (WALK_HEIGHT_FRACTION * full
        # standing height, 2026-07-28 -- see that constant's comment),
        # not the stand task's full STAND_HEIGHT_M: a crouched walking
        # height keeps the CoM lower and legs bent, discouraging both
        # crawling/belly-flopping AND an overextended, easy-to-topple
        # near-max-height stance. Weight raised 0.2 -> 0.6 alongside the
        # target change (still well below forward_velocity_reward's 2.0,
        # so it shapes rather than dominates, but a "loose regularizer"
        # weight wasn't enough to reliably hold this if the goal is a
        # real height-based stability guarantee -- reassess based on
        # the next walk training run's actual height-tracking behavior).
        height_reward = -abs(self._torso_height() - WALK_TARGET_HEIGHT_M)

        # Walk on the feet, not the knees/shins -- this task previously had
        # NO foot-placement term at all, which is exactly why a trained
        # policy was observed walking on its knees: nothing in the reward
        # distinguished that from walking on its feet. Not gated by height
        # progress (unlike the stand task's version) since the walk task
        # is always meant to be standing/walking, never mid-climb.
        tip_reward, non_tip_penalty = self._foot_placement_terms()

        # This alone wasn't enough -- a later, much-longer-trained walk
        # policy (43M timesteps) converged back to knee-walking despite
        # the above, because tip_reward only judges WHERE contact happens
        # when it happens, not whether a leg ever lifts at all. A policy
        # that drags the true foot tip along the ground continuously,
        # never swinging, satisfies tip_reward perfectly while still never
        # producing a real gait. _foot_clearance_reward() closes that gap
        # by rewarding swinging (non-contacting) legs for actually being
        # elevated -- adapted from the standard feet_air_time/
        # foot_clearance pattern in quadruped locomotion RL (a labmate's
        # Go2/Genesis implementation, friend_code/go2_env_walk.py, uses
        # the same idea).
        foot_clearance_reward = self._foot_clearance_reward()

        return (
            2.0 * forward_velocity_reward
            + 0.5 * upright_reward
            + 0.6 * height_reward
            + 1.5 * tip_reward
            + non_tip_penalty
            + 0.75 * foot_clearance_reward
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
