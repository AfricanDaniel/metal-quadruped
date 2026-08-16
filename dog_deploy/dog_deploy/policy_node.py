#!/usr/bin/env python3
"""Sim-to-real bridge: runs a TorchScript policy against the real robot.

Builds the same observation dog_gym's DogEnv trains on (8 motor qpos + 8
motor qvel + IMU accel/gyro + previous action, all in motor_1..motor_8
order -- see dog_gym/envs/dog_env.py) from actuator's read_motor_positions
service and dog_imu's IMU topic, runs the policy, and sends the result to
actuator's set_motor_targets service (control_mode='position', the
default) or its set_motor_torque service (control_mode='torque' -- see
this docstring's "control_mode" section below). motor_mapping.yaml
(dog_description) supplies the per-motor sign flip between the sim's
joint-angle convention and the real motor's command direction -- see that
file's docstring/README for exactly what `sign` means.

Safety: `max_delta_deg_per_step` clamps how far any single motor's target
may move in one control tick, and `dry_run_hold_pose` bypasses the policy
entirely (each tick just re-commands each motor's current position) so the
whole read -> observe -> command loop can be exercised safely before ever
loading a trained policy.

Home offset: `actuator`'s `read_motor_positions` returns each motor's RAW
ABSOLUTE angle (confirmed directly against basic_control.cpp -- it never
subtracts `set_home`'s stored reference, only `go_to_pose` does that), an
arbitrary per-motor nonzero value at the physical tucked/home stance. But
`DogEnv`'s 'stand' task always resets to EXACTLY qpos=0 at that same
physical stance in sim. Feeding the policy raw absolute degrees directly
(as an earlier version of this file did) silently offsets every real
observation from what the policy was actually trained on. Fixed by
capturing a home reference once at startup (`home_position_deg` param, or
auto-captured from the current reading if not provided -- see __init__)
and subtracting it before building the observation / adding it back when
converting an action to a real target -- see _on_positions_read().

Sliding calf range: each leg's calf motor already reports its own real
ABSOLUTE (torso-relative) angle directly from its encoder -- the belt
does that decoupling in hardware natively, no software compensation
needed on this side (see dog_gym/envs/dog_env.py's belt-decoupling
comments for the sim-side half of this story). But the calf's real
physical limit -- the calf link hitting the thigh link -- is fixed in a
RELATIVE coordinate (calf_absolute + thigh_absolute, both home-relative)
and therefore SLIDES in absolute terms as the thigh moves (see
daniel_cl_context.md's TODO 13 for the full measurement). MuJoCo enforces
this automatically via dog.mjcf.xml's <joint range> in sim; the real
robot has no such clamp, so _on_positions_read() enforces it explicitly
using the live thigh reading -- see CALF_RANGE_DEG below.

IMU frame: subscribes to dog_imu's CALIBRATED `imu/data` (forward/left/up,
ROS REP-103 -- see dog_imu/calibrate_imu_node.py), not the raw
`imu/data_raw`. But the sim the policy was trained in (dog_description/
mjcf/dog.mjcf.xml) is still in the CAD's own native frame (right=x,
front=y, up=z -- the ROS remap is a separate, still-outstanding TODO on
that file) -- so _ros_to_cad() below applies one more fixed, already-known
rotation (not something recalibration ever needs to redo) to match what
the policy actually saw during training. Remove this conversion (and just
use imu/data directly) if dog.mjcf.xml ever gets the full ROS remap and
policies get retrained against it.

control_mode (2026-08-04): 'position' (default, unchanged) sends
actuator's set_motor_targets service an absolute output-shaft ANGLE per
motor, tracked by the firmware's own position-mode PD -- this is the
only path that existed before. 'torque' sends actuator's NEW
set_motor_torque service a raw output-shaft TORQUE (N*m) per motor
instead, for a policy trained with dog_gym's control_mode='torque'
(dog_torque.mjcf.xml's <motor> actuators, no PD tracking at all). The
observation build (motor qpos/qvel + IMU + prev_action) is IDENTICAL
either way -- only how `action_rad` gets turned into a real command
differs, see _on_positions_read()'s branch. Torque mode has real,
different safety characteristics from position mode: there is no
firmware-side <joint range>-equivalent hard stop the way MuJoCo enforces
in sim, and a stale/un-refreshed torque command is actively dangerous
(unlike a stale position target, which is at least a bounded, PD-held
pose) -- see max_torque_nm/max_delta_torque_nm_per_step below and
actuator's own torque_timeout_s watchdog in basic_control.cpp.
"""

import csv
import os
import time

import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Imu

from actuator.srv import ReadMotorPositions, SetMotorTargets, SetMotorTorque
from dog_deploy.home_correction import apply_back_leg_correction

NUM_MOTORS = 8
DEFAULT_MOTOR_MAPPING_PATH = os.path.join(
    get_package_share_directory('dog_description'), 'config', 'motor_mapping.yaml')

# 3-tick observation-history stacking (2026-08-07) -- MUST match
# dog_gym/envs/dog_env.py's single_obs_dim/obs_history_len exactly (30 =
# 8 qpos + 8 qvel + 6 IMU + 8 prev_action, 3 ticks stacked = 90 total).
# Policies exported after dog_gym's history-stacking feature was added
# expect this 90-dim input; feeding the old single-tick 30-dim vector
# fails at the network's first layer (RuntimeError: mat1 and mat2 shapes
# cannot be multiplied, (1x30 and 90x512)) -- confirmed directly against
# a real PPO_9000000_walk_position_obshistory_v14 deployment attempt.
SINGLE_OBS_DIM = NUM_MOTORS + NUM_MOTORS + 6 + NUM_MOTORS
OBS_HISTORY_LEN = 3

DEG_TO_RAD = 0.017453292519943295
RAD_TO_DEG = 57.29577951308232

# Real-hardware-measured calf RAW/RELATIVE range, home-relative degrees
# -- MUST match generate_dog_mjcf.py's JOINT_RANGE_OVERRIDES_DEG's calf
# entries exactly (that file is the single source of truth; keep these
# in sync if the range is ever re-measured). See this module's
# docstring's "Sliding calf range" section and daniel_cl_context.md's
# TODO 13 for the full derivation (one stop is home itself, by design,
# for every leg -- home-side has no margin; the other side has the
# usual 5% pulled in from the measured stop).
CALF_RANGE_DEG = {
    'leg_a_calf': (0, 206.1),
    'leg_b_calf': (-213.3, 0),
    'leg_c_calf': (0, 215.8),
    'leg_d_calf': (-213.7, 0),
}

# calf_belt_sign in dog_env.py's terms -- came out uniformly +1 for all
# 4 legs after AXIS_FLIP (verified directly, see daniel_cl_context.md's
# "AXIS_FLIP" section), i.e. raw_hinge = calf_absolute + thigh_absolute
# for every leg, no per-leg sign needed. Re-verify (recompute per leg
# from the model's own joint axes, matching dog_env.py's __init__) if
# the CAD or AXIS_FLIP ever changes.
CALF_BELT_SIGN = 1.0


def ros_to_cad(x, y, z):
    """(forward, left, up) [ROS REP-103, what imu/data publishes] ->
    (right, front, up) [CAD-native, what dog.mjcf.xml/the policy actually
    uses] -- see this module's docstring. Derived directly from the known
    CAD<->ROS relationship (x_ros=y_cad, y_ros=-x_cad, z_ros=z_cad),
    algebraically inverted -- not something a recalibration changes."""
    return -y, x, z


def load_motor_signs(motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH):
    """Returns [sign_motor_1, ..., sign_motor_8] from motor_mapping.yaml."""
    with open(motor_mapping_path) as f:
        mapping = yaml.safe_load(f)['motors']
    return [float(mapping[motor_id]['sign']) for motor_id in range(1, NUM_MOTORS + 1)]


def load_motor_joint_names(motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH):
    """Returns ["leg_a_thigh", ...] motor 1..8, for log_csv readability only
    -- duplicated in miniature from dog_gym/envs/dog_env.py's own
    load_motor_joint_names() rather than imported, since dog_deploy is
    meant to run on the Jetson without gymnasium/mujoco/dog_gym
    installed (see this module's docstring)."""
    with open(motor_mapping_path) as f:
        mapping = yaml.safe_load(f)['motors']
    return [
        f"{mapping[motor_id]['leg']}_{mapping[motor_id]['joint']}"
        for motor_id in range(1, NUM_MOTORS + 1)
    ]


def find_calf_thigh_pairs(motor_joint_names):
    """Returns {calf_motor_index: thigh_motor_index} for all 4 legs
    (0-indexed, i.e. motor_id - 1), derived generically from
    motor_joint_names -- same pairing dog_gym/envs/dog_env.py's
    calf_idx/calf_thigh_idx computes, duplicated here since dog_deploy
    doesn't import dog_gym (see this module's docstring)."""
    pairs = {}
    for i, name in enumerate(motor_joint_names):
        if name.endswith('_calf'):
            thigh_name = name[:-len('_calf')] + '_thigh'
            pairs[i] = motor_joint_names.index(thigh_name)
    return pairs


class PolicyNode(Node):

    def __init__(self):
        super().__init__('policy_node')

        self.declare_parameter('policy_path', '')
        # 50.0 (raised from 20.0, 2026-08-16, chatbot.md "can the Jetson
        # actually sustain your recommended 50Hz control_rate_hz?" thread):
        # gated on empirically proving the Jetson could sustain it first --
        # real hardware measured at 61.4Hz sustained control_rate_hz=100.0
        # request AFTER fixing the executor-contention bug that had been
        # capping actual throughput far below any requested rate. Also
        # directly narrows the sim-to-real slew gap on the deployment side:
        # effective real ceiling is control_rate_hz * max_delta_deg_per_step,
        # was 20*5=100 deg/s, now 50*5=250 deg/s -- much closer to the real
        # measured hardware slew ceiling (~300-340 deg/s, see the empirical
        # slew characterization test) than the old 100 deg/s was.
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('max_delta_deg_per_step', 5.0)
        self.declare_parameter('imu_timeout_sec', 0.5)
        self.declare_parameter('dry_run_hold_pose', True)
        # 'position' (default, unchanged) or 'torque' -- see this module's
        # docstring's "control_mode" section. MUST match whatever
        # control_mode the loaded policy_path was actually trained with
        # (dog_gym.train's --control-mode) -- nothing here can detect a
        # mismatch the way SB3's check_for_correct_spaces does at training
        # time, since the exported TorchScript module has no action-space
        # metadata attached.
        self.declare_parameter('control_mode', 'position')
        # TORQUE MODE ONLY, both below. Client-side clamps -- DELIBERATELY
        # redundant with actuator's own max_torque_nm/torque_timeout_s
        # server-side safety net (basic_control.cpp) -- neither should be
        # the only thing standing between a bad policy output and full
        # motor torque. max_delta_torque_nm_per_step is NOT something
        # dog_gym's sim training itself enforces (control_mode='torque'
        # has no slew clamp in DogEnv.step()) -- it's an extra
        # conservative safety net specific to this being the first-ever
        # real torque deployment, not a sim-fidelity requirement.
        #
        # max_torque_nm: PER-MOTOR array, 8 values, motor 1..8 order (was
        # a single shared double -- see actuator's matching
        # declare_parameter comment in basic_control.cpp for the full
        # 2026-08-04 finding: at a uniform 1.0 N*m, the 4 thigh motors
        # were pinned at the ceiling 99.7% of a real run -- genuinely
        # underpowered -- while the 4 calf motors swung 150-200+ degrees
        # at that SAME limit -- a slipping foot with nothing checking the
        # leg's motion. One shared number can't serve both. Kept in sync
        # with actuator's own max_torque_nm default -- a client
        # requesting more than the server allows just gets silently
        # clamped lower there, which would be a confusing mismatch if the
        # two drifted apart.
        self.declare_parameter('max_torque_nm', [1.0] * NUM_MOTORS)
        self.declare_parameter('max_delta_torque_nm_per_step', 2.0)
        # Overridable so a corrected/test copy of motor_mapping.yaml can be
        # used for a specific run (e.g. while a known sign issue in the
        # shared file is still being investigated) without touching the
        # shared file itself, which dog_gym and everything else depends on.
        self.declare_parameter('motor_mapping_path', str(DEFAULT_MOTOR_MAPPING_PATH))
        # '' (default) disables logging. When set, writes one CSV row per
        # motor per control tick -- real position/velocity/torque actually
        # read (real_torque_nm added 2026-08-05 -- the motor's own
        # MEASURED torque, from actuator's read_motor_positions, present
        # in every control_mode, not just torque; distinct from
        # command_torque_nm, which is what control_mode='torque' actually
        # COMMANDED), the sim-convention qpos that became part of the
        # observation, the policy's raw (pre-clamp) action, the clamped
        # action, and the real degrees actually sent -- so a real-hardware
        # run's exact per-motor behavior can be inspected after the fact
        # instead of only watched live. See _open_log()/_log_row().
        self.declare_parameter('log_csv', '')
        # 8 floats, or empty (default) to auto-capture from the current
        # reading at startup instead -- see __init__'s home-capture block
        # below and this module's docstring's "Home offset" section.
        # Provide explicitly to reuse an exact previously-known-good home
        # (e.g. matching a specific /set_home call's response) without
        # needing the robot physically re-posed at that exact stance.
        self.declare_parameter('home_position_deg', [])
        # '' (default) disables this -- unchanged behavior. When set (and
        # home_position_deg above is NOT also given -- that still takes
        # priority), loads home_position_deg from this YAML cache file
        # instead of auto-capturing from whatever position the robot is
        # CURRENTLY in. See dog_deploy/set_home_and_cache.py -- written by
        # that node's own /set_home call, at the moment the robot is
        # PROVABLY tucked (not whenever this policy_node happens to
        # start). Added 2026-08-10 (chatbot.md "running STAND then
        # WALK-from-standing back to back") after auto-capture was found
        # to silently corrupt a SECOND policy_node run's home reference if
        # it starts while the robot is already standing (from a prior
        # policy) rather than tucked -- every subsequent observation then
        # reads as "near home" even though the robot is actually fully
        # extended, producing wrong, potentially dangerous actions. Use
        # this for any workflow that runs more than one policy_node in
        # sequence against the same physical home reference.
        self.declare_parameter('home_position_deg_cache_path', '')
        # Windup guard for the prev_action-anchored slew limiter (see
        # _on_positions_read): max degrees the commanded target may LEAD
        # the measured position. Large enough to never bind during
        # normal tracking (ordinary lag is under max_delta_deg_per_step
        # + a few degrees of PD sag), small enough that a jammed motor
        # can't wind up a big error and violently catch up on release.
        self.declare_parameter('max_target_lead_deg', 10.0)
        # 0.0 (default) disables freezing entirely -- current clamp/slew
        # behavior unchanged. When > 0, once this many seconds of control
        # ticks have elapsed, the target is snapshotted and held FIXED
        # forever after -- the policy keeps running every tick (for
        # logging/visibility), but its output is ignored for target-
        # setting purposes. Firmware position-mode PD stays fully active
        # on the frozen target (this does NOT cut torque/go passive) --
        # it just stops re-commanding a new target every tick, which is
        # what was causing the small continuous oscillation: real-hardware
        # data (2026-07-29, PPO_5000000_DR_stand_policy_v3_noNoise.csv)
        # showed the policy's raw action never actually settles quiet on
        # its own even once the robot is visibly done standing (still
        # 8-41deg mean tick-to-tick swings late in the run, saturating
        # max_delta_deg_per_step on most ticks) -- so triggering on "the
        # policy's own output went quiet" wouldn't reliably fire. A fixed
        # time, tuned to when you visually observe standing is complete,
        # is simpler and actually works. See _on_positions_read().
        self.declare_parameter('freeze_after_sec', 0.0)
        # '' (default) disables this entirely -- unchanged behavior, a
        # single fixed home_position_deg for the whole run, same as
        # before this parameter existed. When set to a second
        # ~/.dog_home_cache.yaml-shaped file, the EFFECTIVE home
        # reference used every tick starts at home_position_deg (above)
        # and linearly ramps over home_switch_ramp_sec, starting
        # home_switch_after_sec into the run, to this second cache's
        # values -- then stays there. Added 2026-08-15 (daniel_cl_
        # context.md, back-leg home-offset recalibration): the recalibrated
        # cache gets the back legs' STEADY-STATE walking posture much
        # closer to sim, but starting a run already anchored to it makes
        # the robot tip backward during the climb-to-standing phase --
        # climbing was fine under the ORIGINAL reference. This lets a
        # deploy start on the original (safe climb) reference and hand off
        # to the recalibrated one once standing is complete, instead of
        # forcing one fixed reference for the whole run. The ramp (not an
        # instant swap) exists so the commanded target doesn't jump by the
        # full home delta in one control tick -- see _on_positions_read()'s
        # blend computation.
        self.declare_parameter('home_switch_cache_path', '')
        # Simpler alternative to home_switch_cache_path above, for the
        # specific back-leg (motor 5/8) correction (see home_correction.py):
        # 0.0 (default) disables this. When nonzero, the switch target is
        # computed FRESH from whatever home_position_deg was just loaded
        # (regular, above) + fraction * the known correction, instead of
        # requiring a second pre-written cache file -- so trying a
        # different fraction is just a different number on this same
        # deploy, not a re-calibration. If BOTH this and
        # home_switch_cache_path are given, home_switch_cache_path wins
        # (more specific/explicit).
        self.declare_parameter('home_switch_back_leg_fraction', 0.0)
        self.declare_parameter('home_switch_after_sec', 3.0)
        self.declare_parameter('home_switch_ramp_sec', 1.5)

        self.control_mode = self.get_parameter('control_mode').value
        if self.control_mode not in ('position', 'torque'):
            raise ValueError(f"control_mode must be 'position' or 'torque', got {self.control_mode!r}")
        self.control_rate_hz = self.get_parameter('control_rate_hz').value
        self.max_delta_rad = (
            self.get_parameter('max_delta_deg_per_step').value * DEG_TO_RAD)
        self.max_target_lead_rad = (
            self.get_parameter('max_target_lead_deg').value * DEG_TO_RAD)
        self.imu_timeout_sec = self.get_parameter('imu_timeout_sec').value
        self.dry_run_hold_pose = self.get_parameter('dry_run_hold_pose').value
        self.freeze_after_sec = self.get_parameter('freeze_after_sec').value
        self._control_tick_count = 0
        self._frozen = False
        self._frozen_action_rad = None

        motor_mapping_path = self.get_parameter('motor_mapping_path').value
        self.get_logger().info(f'Loading motor signs from {motor_mapping_path}')
        self.motor_sign = load_motor_signs(motor_mapping_path)
        self.motor_joint_names = load_motor_joint_names(motor_mapping_path)
        self.calf_thigh_pairs = find_calf_thigh_pairs(self.motor_joint_names)
        self.calf_range_rad = {
            calf_i: tuple(v * DEG_TO_RAD for v in CALF_RANGE_DEG[self.motor_joint_names[calf_i]])
            for calf_i in self.calf_thigh_pairs
        }

        self.tick = 0
        self.csv_file = None
        self.csv_writer = None
        log_csv_path = self.get_parameter('log_csv').value
        if log_csv_path:
            self._open_log(log_csv_path)

        self.policy = None
        if not self.dry_run_hold_pose:
            policy_path = self.get_parameter('policy_path').value
            if not policy_path:
                raise ValueError(
                    'policy_path parameter is required when dry_run_hold_pose is false')
            self.policy = torch.jit.load(policy_path)
            self.policy.eval()
            self.get_logger().info(f'Loaded policy from {policy_path}')
        else:
            self.get_logger().warning(
                'dry_run_hold_pose=true: NOT running a policy. Every control '
                'tick just re-commands each motor to hold its current '
                'position. Set dry_run_hold_pose:=false and provide '
                'policy_path once this dry run looks safe.')

        self.prev_action = None  # seeded from the first measured pose, see _on_positions_read
        self._obs_history = None  # seeded on the first control tick, see _on_positions_read
        self.latest_imu = None
        self.busy = False
        self._pending_timer = None  # see _schedule_next_tick()

        self.imu_sub = self.create_subscription(Imu, 'imu/data', self._on_imu, 10)
        self.read_client = self.create_client(ReadMotorPositions, 'read_motor_positions')
        # Both clients are always created (harmless), but only the one
        # actually needed for control_mode is waited on below -- no reason
        # to block startup on a service this run will never call.
        self.set_targets_client = self.create_client(SetMotorTargets, 'set_motor_targets')
        self.set_torque_client = self.create_client(SetMotorTorque, 'set_motor_torque')
        command_client, command_service_name = (
            (self.set_torque_client, 'set_motor_torque') if self.control_mode == 'torque'
            else (self.set_targets_client, 'set_motor_targets'))

        for client, name in (
            (self.read_client, 'read_motor_positions'),
            (command_client, command_service_name),
        ):
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(f'Waiting for {name} service (is actuator running?)...')

        home_position_deg = list(self.get_parameter('home_position_deg').value)
        home_cache_path = self.get_parameter('home_position_deg_cache_path').value
        if home_position_deg:
            if len(home_position_deg) != NUM_MOTORS:
                raise ValueError(
                    f'home_position_deg must have exactly {NUM_MOTORS} values, '
                    f'got {len(home_position_deg)}')
            self.home_position_deg = home_position_deg
            self.get_logger().info(f'Using provided home_position_deg: {self.home_position_deg}')
        elif home_cache_path:
            # See home_position_deg_cache_path's declare_parameter comment
            # -- written by set_home_and_cache.py, at the moment the robot
            # was PROVABLY tucked (when /set_home was called), not
            # whatever position it's in right now.
            cache_path = os.path.expanduser(home_cache_path)
            self.get_logger().info(f'Loading home_position_deg from cache: {cache_path}')
            with open(cache_path) as f:
                cached = yaml.safe_load(f)['home_position_deg']
            self.home_position_deg = [cached[i] for i in range(1, NUM_MOTORS + 1)]
            self.get_logger().info(f'Loaded home_position_deg from cache: {self.home_position_deg}')
        else:
            self.get_logger().info(
                'No home_position_deg or home_position_deg_cache_path provided -- reading '
                'current motor positions as home. The robot MUST already be physically posed '
                'at the tucked/home stance now.')
            request = ReadMotorPositions.Request()
            request.motor_id = list(range(1, NUM_MOTORS + 1))
            future = self.read_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response is None:
                raise RuntimeError('read_motor_positions failed while capturing home -- aborting startup')
            self.home_position_deg = list(response.position_deg)
            self.get_logger().info(f'Captured home_position_deg: {self.home_position_deg}')

        # See home_switch_cache_path's declare_parameter comment above.
        # None (default, '' param) = feature off, home_position_deg is
        # used as a fixed constant for the whole run exactly like before
        # this existed -- _on_positions_read()'s blend is a no-op in that
        # case (blend stays 0.0 forever).
        self.home_position_deg_after_switch = None
        home_switch_cache_path = self.get_parameter('home_switch_cache_path').value
        home_switch_back_leg_fraction = self.get_parameter('home_switch_back_leg_fraction').value
        if home_switch_cache_path:
            switch_cache_path = os.path.expanduser(home_switch_cache_path)
            self.get_logger().info(
                f'Loading post-switch home_position_deg from cache: {switch_cache_path}')
            with open(switch_cache_path) as f:
                switch_cached = yaml.safe_load(f)['home_position_deg']
            self.home_position_deg_after_switch = [switch_cached[i] for i in range(1, NUM_MOTORS + 1)]
            self.get_logger().info(
                f'Loaded post-switch home_position_deg: {self.home_position_deg_after_switch} -- '
                f'will start ramping {self.get_parameter("home_switch_after_sec").value}s into the run, '
                f'over {self.get_parameter("home_switch_ramp_sec").value}s.')
        elif home_switch_back_leg_fraction != 0.0:
            # See home_switch_back_leg_fraction's declare_parameter
            # comment -- computed fresh from home_position_deg (whatever
            # 'regular' reference was just loaded/captured above), not
            # read from a second file.
            self.home_position_deg_after_switch = apply_back_leg_correction(
                self.home_position_deg, home_switch_back_leg_fraction)
            self.get_logger().info(
                f'home_switch_back_leg_fraction={home_switch_back_leg_fraction} -- post-switch '
                f'home_position_deg computed as {self.home_position_deg_after_switch} -- '
                f'will start ramping {self.get_parameter("home_switch_after_sec").value}s into the run, '
                f'over {self.get_parameter("home_switch_ramp_sec").value}s.')
        self.home_switch_after_sec = self.get_parameter('home_switch_after_sec').value
        self.home_switch_ramp_sec = self.get_parameter('home_switch_ramp_sec').value
        self._home_switch_started_logged = False
        self._home_switch_complete_logged = False

        # CHAINED, not a fixed-rate create_timer (2026-08-16, see
        # _schedule_next_tick()'s own docstring for the full mechanism/
        # measurement this replaces) -- kicks off the first cycle
        # directly; every cycle after that reschedules itself from
        # _on_command_sent() once it actually finishes.
        self._schedule_next_tick(0.0)
        if self.control_mode == 'torque':
            self.get_logger().warning(
                f'policy_node ready: control_mode=torque, control_rate_hz={self.control_rate_hz}, '
                f'max_torque_nm={self.get_parameter("max_torque_nm").value}, '
                f'max_delta_torque_nm_per_step={self.get_parameter("max_delta_torque_nm_per_step").value}, '
                f'dry_run_hold_pose={self.dry_run_hold_pose}. TORQUE MODE: no firmware position '
                'clamp protects this the way position mode has -- start with dry_run_hold_pose=true '
                'and a low max_torque_nm, confirm expected behavior before raising either.')
        else:
            self.get_logger().info(
                f'policy_node ready: control_mode=position, control_rate_hz={self.control_rate_hz}, '
                f'max_delta_deg_per_step={self.get_parameter("max_delta_deg_per_step").value}, '
                f'dry_run_hold_pose={self.dry_run_hold_pose}')

    def _open_log(self, path):
        self.csv_file = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Last column's meaning depends on control_mode: 'target_deg' (real
        # absolute output-shaft angle sent to set_motor_targets) for
        # position, 'command_torque_nm' (real output-shaft torque sent to
        # set_motor_torque) for torque -- raw_action/clamped_action are
        # always in the policy's own action-space units (radians for
        # position, N*m for torque).
        last_col = 'command_torque_nm' if self.control_mode == 'torque' else 'target_deg'
        self.csv_writer.writerow([
            'tick', 'motor_id', 'joint', 'sign',
            'real_position_deg', 'real_velocity_deg_s', 'real_torque_nm',
            'sim_qpos_rad', 'raw_action', 'clamped_action', last_col, 'frozen',
        ])
        self.get_logger().info(f'Logging per-motor control data to {path}')

    def _log_row(self, response, motor_qpos_rad, raw_action, clamped_action, command_value):
        for i in range(NUM_MOTORS):
            self.csv_writer.writerow([
                self.tick, i + 1, self.motor_joint_names[i], self.motor_sign[i],
                response.position_deg[i], response.velocity_deg_s[i], response.torque_nm[i],
                motor_qpos_rad[i], raw_action[i], clamped_action[i], command_value[i],
                self._frozen,
            ])
        self.csv_file.flush()  # real-hardware runs are often killed abruptly (Ctrl-C) -- don't lose rows
        self.tick += 1

    def _close_log(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

    def _on_imu(self, msg):
        self.latest_imu = msg

    def _schedule_next_tick(self, delay_s):
        """Runs _control_step() again after delay_s seconds -- immediately
        (delay_s<=0) if the previous cycle already used up its whole
        period or more, otherwise via a self-canceling one-shot timer for
        the remainder. Replaces the old fixed-rate `create_timer(1.0 /
        control_rate_hz, self._control_step)` (2026-08-16, chatbot.md
        "timer overrun and chaining fix", multi-agent review w/
        Antigravity): measured directly that a plain fixed-rate timer +
        busy-flag-skip silently DOUBLES effective latency whenever a
        cycle's real work (read+inference+write) exceeds one period --
        the timer's next firing lands while still busy, gets dropped as a
        no-op, and the tick doesn't actually resume until the NEXT firing
        after that (measured: ~55-65ms of real work but ~103ms actual
        spacing at a 50ms/20Hz period, real hardware). Chaining from the
        previous cycle's own completion instead means a slow cycle is
        immediately followed by the next one (no double-period stall),
        while a fast cycle still waits out the rest of its period
        (Antigravity's explicit requirement: sim trains against a fixed
        dt, so running unboundedly fast on 'good' ticks would itself
        become a new sim-to-real timing mismatch) -- control_rate_hz is a
        rate CAP now, not a guaranteed rate."""
        if delay_s <= 0.0:
            self._control_step()
            return
        self._pending_timer = self.create_timer(delay_s, self._on_pending_timer_fire)

    def _on_pending_timer_fire(self):
        self._pending_timer.cancel()
        self.destroy_timer(self._pending_timer)
        self._pending_timer = None
        self._control_step()

    def _control_step(self):
        if self.busy:
            return  # defensive only -- chaining means this shouldn't ever actually happen

        if self.latest_imu is None:
            self.get_logger().warning('No IMU data received yet, skipping control step',
                                       throttle_duration_sec=2.0)
            self._schedule_next_tick(1.0 / self.control_rate_hz)
            return

        imu_age_sec = (
            self.get_clock().now() - rclpy.time.Time.from_msg(self.latest_imu.header.stamp)
        ).nanoseconds / 1e9
        if imu_age_sec > self.imu_timeout_sec:
            self.get_logger().warning(
                f'IMU data is {imu_age_sec:.2f}s old (> {self.imu_timeout_sec}s timeout), '
                'skipping control step', throttle_duration_sec=2.0)
            self._schedule_next_tick(1.0 / self.control_rate_hz)
            return

        self.busy = True
        self._tick_start_time = time.monotonic()
        request = ReadMotorPositions.Request()
        request.motor_id = list(range(1, NUM_MOTORS + 1))
        future = self.read_client.call_async(request)
        future.add_done_callback(self._on_positions_read)

    def _on_positions_read(self, future):
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'read_motor_positions call failed: {e}')
            self.busy = False
            self._schedule_next_tick(1.0 / self.control_rate_hz)
            return

        # Always incremented once per control tick (moved out of the
        # freeze_after_sec-only block below so home_switch_cache_path's
        # blend, which needs elapsed time regardless of whether freezing
        # is even enabled, can use the same counter) -- freeze_after_sec's
        # own check further down just READS this, doesn't increment it.
        self._control_tick_count += 1

        # See home_switch_cache_path's declare_parameter comment: blend
        # from home_position_deg toward home_position_deg_after_switch,
        # starting home_switch_after_sec into the run, over
        # home_switch_ramp_sec. Stays at effective_home ==
        # home_position_deg (blend=0.0) for the entire run if the feature
        # is off (home_position_deg_after_switch is None).
        effective_home_position_deg = self.home_position_deg
        if self.home_position_deg_after_switch is not None:
            elapsed_sec = self._control_tick_count / self.control_rate_hz
            blend = 0.0
            if elapsed_sec > self.home_switch_after_sec:
                if self.home_switch_ramp_sec > 0.0:
                    blend = min(1.0, (elapsed_sec - self.home_switch_after_sec) / self.home_switch_ramp_sec)
                else:
                    blend = 1.0
            if blend > 0.0 and not self._home_switch_started_logged:
                self._home_switch_started_logged = True
                self.get_logger().info(f'home_switch_after_sec={self.home_switch_after_sec}s reached -- '
                                        'starting home reference ramp.')
            if blend >= 1.0 and not self._home_switch_complete_logged:
                self._home_switch_complete_logged = True
                self.get_logger().info('Home reference ramp complete -- now fully on '
                                        f'{self.home_position_deg_after_switch}.')
            effective_home_position_deg = [
                (1.0 - blend) * self.home_position_deg[i] + blend * self.home_position_deg_after_switch[i]
                for i in range(NUM_MOTORS)
            ]

        # Home-relative, matching sim's own qpos=0-at-tucked-home reference
        # -- see this module's docstring's "Home offset" section.
        motor_qpos_rad = [
            self.motor_sign[i] * (response.position_deg[i] - effective_home_position_deg[i]) * DEG_TO_RAD
            for i in range(NUM_MOTORS)
        ]
        motor_qvel_rad_s = [
            self.motor_sign[i] * response.velocity_deg_s[i] * DEG_TO_RAD
            for i in range(NUM_MOTORS)
        ]

        # First control cycle: seed the slew-limiter anchor (and the
        # observation's prev_action slot). 'position': the actual measured
        # pose, matching DogEnv.reset()'s "prev_action = current pose"
        # semantics for control_mode='position'. 'torque': zero -- no
        # torque has been commanded yet, matching DogEnv.reset()'s
        # control_mode='torque' branch (prev_action stays at its zero
        # default regardless of the legs' actual starting pose; seeding it
        # from motor_qpos_rad here would wrongly treat a position-radians
        # reading as if it were an N*m torque).
        if self.prev_action is None:
            self.prev_action = [0.0] * NUM_MOTORS if self.control_mode == 'torque' else list(motor_qpos_rad)

        imu = self.latest_imu
        accel_cad = ros_to_cad(imu.linear_acceleration.x, imu.linear_acceleration.y,
                                imu.linear_acceleration.z)
        gyro_cad = ros_to_cad(imu.angular_velocity.x, imu.angular_velocity.y,
                               imu.angular_velocity.z)
        obs = (
            motor_qpos_rad
            + motor_qvel_rad_s
            + list(accel_cad)
            + list(gyro_cad)
            + self.prev_action
        )

        # 3-tick observation-history stacking -- see SINGLE_OBS_DIM's
        # comment. First control tick: no history yet, fill all
        # OBS_HISTORY_LEN slots with this SAME single-tick obs, matching
        # DogEnv.reset()'s identical behavior. Every tick after: drop the
        # oldest 30-dim block, append this tick's obs at the end -- same
        # shift-and-append DogEnv.step() does, most-recent-last.
        if self._obs_history is None:
            self._obs_history = list(obs) * OBS_HISTORY_LEN
        else:
            self._obs_history = self._obs_history[SINGLE_OBS_DIM:] + list(obs)
        stacked_obs = self._obs_history

        if self.dry_run_hold_pose:
            # 'position': hold the current pose (matches this parameter's
            # name literally). 'torque': there's no meaningful "current
            # torque" to hold -- motor_qpos_rad is a POSITION in radians,
            # reusing it as a torque value here would silently send
            # nonsense-scaled torque commands. Zero torque is the correct
            # dry-run no-op for torque mode (passive, doesn't move
            # anything on its own).
            action_rad = [0.0] * NUM_MOTORS if self.control_mode == 'torque' else list(motor_qpos_rad)
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor([stacked_obs], dtype=torch.float32)
                action_rad = self.policy(obs_tensor)[0].tolist()

        if self.control_mode == 'torque':
            self._send_torque_command(response, motor_qpos_rad, action_rad)
            return

        # Safety clamp, anchored to the PREVIOUS COMMANDED TARGET (not
        # the measured position) -- exactly like DogEnv.step()'s slew
        # limiter. An earlier version anchored to the measured position;
        # with stiff firmware gains the motor overshoots each staircase
        # step within one tick, and a measurement-anchored clamp feeds
        # that overshoot straight back into the next target, flipping it
        # to the other side every tick -- a measurement-coupled limit
        # cycle, observed as severe stand-up chatter (velocity direction
        # flipping on ~half of all ticks in the v5 run log, see
        # daniel_cl_context.md 2026-07-27). Anchoring to prev_action
        # gives the firmware PD a clean, monotone ramp regardless of how
        # the motor rings around it.
        clamped_action_rad = [
            self.prev_action[i] + max(
                -self.max_delta_rad,
                min(self.max_delta_rad, action_rad[i] - self.prev_action[i]))
            for i in range(NUM_MOTORS)
        ]
        # Windup guard: never let the commanded target lead the measured
        # position by more than max_target_lead_rad. Without this, a
        # jammed/blocked motor no longer stops the reference (that was
        # the one virtue of measurement-anchoring) -- prev_action would
        # keep marching away from the stuck motor and the eventual
        # catch-up would be violent. With it, the ramp stalls near a
        # jammed motor and resumes when it frees, catch-up bounded.
        clamped_action_rad = [
            max(motor_qpos_rad[i] - self.max_target_lead_rad,
                min(motor_qpos_rad[i] + self.max_target_lead_rad, clamped_action_rad[i]))
            for i in range(NUM_MOTORS)
        ]

        # Sliding calf-range clamp -- see this module's docstring's
        # "Sliding calf range" section and CALF_RANGE_DEG's comment.
        # Converts the (already slew+windup clamped) absolute calf
        # target into the equivalent raw/relative coordinate using the
        # LIVE current thigh reading, clips to the measured physical
        # band, converts back. Applied last so it reflects the true
        # final constraint on what gets sent.
        for calf_i, thigh_i in self.calf_thigh_pairs.items():
            lo, hi = self.calf_range_rad[calf_i]
            raw_equivalent = clamped_action_rad[calf_i] + CALF_BELT_SIGN * motor_qpos_rad[thigh_i]
            raw_clamped = max(lo, min(hi, raw_equivalent))
            clamped_action_rad[calf_i] = raw_clamped - CALF_BELT_SIGN * motor_qpos_rad[thigh_i]

        # Freeze (see freeze_after_sec's declare_parameter comment):
        # snapshot the target ONCE, this many ticks in, then hold it fixed
        # forever -- the policy above still runs every tick (kept for
        # logging/visibility) but its output stops being used once frozen.
        if self.freeze_after_sec > 0.0:
            if not self._frozen and (
                    self._control_tick_count / self.control_rate_hz >= self.freeze_after_sec):
                self._frozen = True
                self._frozen_action_rad = list(clamped_action_rad)
                self.get_logger().info(
                    f'freeze_after_sec={self.freeze_after_sec}s reached -- '
                    'freezing motor targets at the current pose.')
            if self._frozen:
                clamped_action_rad = list(self._frozen_action_rad)

        self.prev_action = clamped_action_rad

        # Inverse of the home-relative conversion above: real ABSOLUTE
        # degrees (same frame set_motor_targets/read_motor_positions use)
        # = home + sign*sim_value (sign is +-1, so sign*sign=1 undoes the
        # sign applied when motor_qpos_rad was built).
        target_deg = [
            effective_home_position_deg[i] + self.motor_sign[i] * clamped_action_rad[i] * RAD_TO_DEG
            for i in range(NUM_MOTORS)
        ]

        if self.csv_writer is not None:
            self._log_row(response, motor_qpos_rad, action_rad, clamped_action_rad, target_deg)

        set_request = SetMotorTargets.Request()
        set_request.motor_id = list(range(1, NUM_MOTORS + 1))
        set_request.position_deg = target_deg
        future = self.set_targets_client.call_async(set_request)
        future.add_done_callback(self._on_command_sent)

    def _send_torque_command(self, response, motor_qpos_rad, action_rad):
        """control_mode='torque' counterpart to the position-mode tail of
        _on_positions_read() above -- action_rad is already the policy's
        raw output (or the dry-run zero-torque no-op), in N*m, sim-frame
        sign convention. Separate method (not inlined into
        _on_positions_read) since torque needs a genuinely different
        sequence of clamps -- no slew-anchored-to-target/windup-guard
        pair (those are position-specific concepts), no home-offset
        degree conversion (torque isn't a frame-relative angle), but a
        NEW calf mechanical end-stop protection position mode doesn't
        need (see below)."""
        # Magnitude clamp -- defense in depth, mirroring actuator's own
        # PER-MOTOR max_torque_nm server-side clamp (basic_control.cpp).
        # See max_torque_nm's declare_parameter comment.
        max_torque_nm = self.get_parameter('max_torque_nm').value
        if len(max_torque_nm) != NUM_MOTORS:
            raise ValueError(
                f'max_torque_nm must have exactly {NUM_MOTORS} values (motor 1..8 order), '
                f'got {len(max_torque_nm)}')
        magnitude_clamped = [
            max(-max_torque_nm[i], min(max_torque_nm[i], action_rad[i]))
            for i in range(NUM_MOTORS)
        ]
        # Rate clamp, anchored to the previous COMMANDED torque (same
        # "anchor to commanded, not measured" reasoning as position
        # mode's slew clamp above) -- an EXTRA safety net beyond what sim
        # training itself enforced, see max_delta_torque_nm_per_step's
        # declare_parameter comment.
        max_delta = self.get_parameter('max_delta_torque_nm_per_step').value
        clamped_action = [
            self.prev_action[i] + max(
                -max_delta, min(max_delta, magnitude_clamped[i] - self.prev_action[i]))
            for i in range(NUM_MOTORS)
        ]

        # Calf mechanical end-stop protection (see this module's
        # docstring's "Sliding calf range" section) -- position mode
        # clamps the TARGET ANGLE to stay inside CALF_RANGE_DEG; torque
        # has no target angle to clamp the same way, so instead this
        # zeros a calf motor's torque outright if its CURRENT measured
        # position is already outside the known mechanical range, rather
        # than letting it keep grinding a real physical stop under real
        # force. Same raw/relative conversion as the position-mode
        # clamp (calf_absolute + thigh_absolute).
        for calf_i, thigh_i in self.calf_thigh_pairs.items():
            lo, hi = self.calf_range_rad[calf_i]
            raw_equivalent = motor_qpos_rad[calf_i] + CALF_BELT_SIGN * motor_qpos_rad[thigh_i]
            if not (lo <= raw_equivalent <= hi):
                clamped_action[calf_i] = 0.0

        # Freeze (torque mode): going PASSIVE (zero torque) once
        # triggered, NOT holding the last nonzero torque forever --
        # unlike position mode, where the firmware PD genuinely holds a
        # frozen TARGET pose, holding a frozen nonzero TORQUE forever
        # would just keep applying constant, unopposed force (a runaway
        # risk, not a stable hold).
        if self.freeze_after_sec > 0.0:
            if not self._frozen and (
                    self._control_tick_count / self.control_rate_hz >= self.freeze_after_sec):
                self._frozen = True
                self.get_logger().info(
                    f'freeze_after_sec={self.freeze_after_sec}s reached -- '
                    'freezing (torque mode: going passive, zero torque).')
            if self._frozen:
                clamped_action = [0.0] * NUM_MOTORS

        self.prev_action = clamped_action

        # Sign flip only -- no home-offset/degree conversion needed
        # (torque isn't a frame-relative angle the way position is),
        # same sign convention as the qpos/qvel observation build.
        torque_nm = [self.motor_sign[i] * clamped_action[i] for i in range(NUM_MOTORS)]

        if self.csv_writer is not None:
            self._log_row(response, motor_qpos_rad, action_rad, clamped_action, torque_nm)

        set_request = SetMotorTorque.Request()
        set_request.motor_id = list(range(1, NUM_MOTORS + 1))
        set_request.torque_nm = torque_nm
        future = self.set_torque_client.call_async(set_request)
        future.add_done_callback(self._on_command_sent)

    def _on_command_sent(self, future):
        try:
            future.result()
        except Exception as e:
            service = 'set_motor_torque' if self.control_mode == 'torque' else 'set_motor_targets'
            self.get_logger().error(f'{service} call failed: {e}')
        now = time.monotonic()
        self.busy = False

        # See _schedule_next_tick()'s own docstring for the full
        # mechanism -- chains the next cycle from here instead of relying
        # on a fixed-rate timer, immediately if this cycle already used a
        # whole period or more, otherwise waiting out the remainder.
        elapsed = now - self._tick_start_time
        self._schedule_next_tick((1.0 / self.control_rate_hz) - elapsed)


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close_log()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
