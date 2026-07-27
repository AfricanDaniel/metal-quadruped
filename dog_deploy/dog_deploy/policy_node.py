#!/usr/bin/env python3
"""Sim-to-real bridge: runs a TorchScript policy against the real robot.

Builds the same observation dog_gym's DogEnv trains on (8 motor qpos + 8
motor qvel + IMU accel/gyro + previous action, all in motor_1..motor_8
order -- see dog_gym/envs/dog_env.py) from actuator's read_motor_positions
service and dog_imu's IMU topic, runs the policy, and sends the result to
actuator's set_motor_targets service. motor_mapping.yaml (dog_description)
supplies the per-motor sign flip between the sim's joint-angle convention
and the real motor's command direction -- see that file's docstring/README
for exactly what `sign` means.

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
"""

import csv
import os

import rclpy
import torch
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Imu

from actuator.srv import ReadMotorPositions, SetMotorTargets

NUM_MOTORS = 8
DEFAULT_MOTOR_MAPPING_PATH = os.path.join(
    get_package_share_directory('dog_description'), 'config', 'motor_mapping.yaml')

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
        self.declare_parameter('control_rate_hz', 20.0)
        self.declare_parameter('max_delta_deg_per_step', 5.0)
        self.declare_parameter('imu_timeout_sec', 0.5)
        self.declare_parameter('dry_run_hold_pose', True)
        # Overridable so a corrected/test copy of motor_mapping.yaml can be
        # used for a specific run (e.g. while a known sign issue in the
        # shared file is still being investigated) without touching the
        # shared file itself, which dog_gym and everything else depends on.
        self.declare_parameter('motor_mapping_path', str(DEFAULT_MOTOR_MAPPING_PATH))
        # '' (default) disables logging. When set, writes one CSV row per
        # motor per control tick -- real position/velocity actually read,
        # the sim-convention qpos that became part of the observation, the
        # policy's raw (pre-clamp) action, the clamped action, and the
        # real degrees actually sent -- so a real-hardware run's exact
        # per-motor behavior can be inspected after the fact instead of
        # only watched live. See _open_log()/_log_row().
        self.declare_parameter('log_csv', '')
        # 8 floats, or empty (default) to auto-capture from the current
        # reading at startup instead -- see __init__'s home-capture block
        # below and this module's docstring's "Home offset" section.
        # Provide explicitly to reuse an exact previously-known-good home
        # (e.g. matching a specific /set_home call's response) without
        # needing the robot physically re-posed at that exact stance.
        self.declare_parameter('home_position_deg', [])
        # Windup guard for the prev_action-anchored slew limiter (see
        # _on_positions_read): max degrees the commanded target may LEAD
        # the measured position. Large enough to never bind during
        # normal tracking (ordinary lag is under max_delta_deg_per_step
        # + a few degrees of PD sag), small enough that a jammed motor
        # can't wind up a big error and violently catch up on release.
        self.declare_parameter('max_target_lead_deg', 10.0)

        self.control_rate_hz = self.get_parameter('control_rate_hz').value
        self.max_delta_rad = (
            self.get_parameter('max_delta_deg_per_step').value * DEG_TO_RAD)
        self.max_target_lead_rad = (
            self.get_parameter('max_target_lead_deg').value * DEG_TO_RAD)
        self.imu_timeout_sec = self.get_parameter('imu_timeout_sec').value
        self.dry_run_hold_pose = self.get_parameter('dry_run_hold_pose').value

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
        self.latest_imu = None
        self.busy = False

        self.imu_sub = self.create_subscription(Imu, 'imu/data', self._on_imu, 10)
        self.read_client = self.create_client(ReadMotorPositions, 'read_motor_positions')
        self.set_targets_client = self.create_client(SetMotorTargets, 'set_motor_targets')

        for client, name in (
            (self.read_client, 'read_motor_positions'),
            (self.set_targets_client, 'set_motor_targets'),
        ):
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(f'Waiting for {name} service (is actuator running?)...')

        home_position_deg = list(self.get_parameter('home_position_deg').value)
        if home_position_deg:
            if len(home_position_deg) != NUM_MOTORS:
                raise ValueError(
                    f'home_position_deg must have exactly {NUM_MOTORS} values, '
                    f'got {len(home_position_deg)}')
            self.home_position_deg = home_position_deg
            self.get_logger().info(f'Using provided home_position_deg: {self.home_position_deg}')
        else:
            self.get_logger().info(
                'No home_position_deg provided -- reading current motor positions as home. '
                'The robot MUST already be physically posed at the tucked/home stance now.')
            request = ReadMotorPositions.Request()
            request.motor_id = list(range(1, NUM_MOTORS + 1))
            future = self.read_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response is None:
                raise RuntimeError('read_motor_positions failed while capturing home -- aborting startup')
            self.home_position_deg = list(response.position_deg)
            self.get_logger().info(f'Captured home_position_deg: {self.home_position_deg}')

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._control_step)
        self.get_logger().info(
            f'policy_node ready: control_rate_hz={self.control_rate_hz}, '
            f'max_delta_deg_per_step={self.get_parameter("max_delta_deg_per_step").value}, '
            f'dry_run_hold_pose={self.dry_run_hold_pose}')

    def _open_log(self, path):
        self.csv_file = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'tick', 'motor_id', 'joint', 'sign',
            'real_position_deg', 'real_velocity_deg_s',
            'sim_qpos_rad', 'raw_action_rad', 'clamped_action_rad', 'target_deg',
        ])
        self.get_logger().info(f'Logging per-motor control data to {path}')

    def _log_row(self, response, motor_qpos_rad, raw_action_rad, clamped_action_rad, target_deg):
        for i in range(NUM_MOTORS):
            self.csv_writer.writerow([
                self.tick, i + 1, self.motor_joint_names[i], self.motor_sign[i],
                response.position_deg[i], response.velocity_deg_s[i],
                motor_qpos_rad[i], raw_action_rad[i], clamped_action_rad[i], target_deg[i],
            ])
        self.csv_file.flush()  # real-hardware runs are often killed abruptly (Ctrl-C) -- don't lose rows
        self.tick += 1

    def _close_log(self):
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

    def _on_imu(self, msg):
        self.latest_imu = msg

    def _control_step(self):
        if self.busy:
            return  # previous cycle's service calls haven't completed yet

        if self.latest_imu is None:
            self.get_logger().warning('No IMU data received yet, skipping control step',
                                       throttle_duration_sec=2.0)
            return

        imu_age_sec = (
            self.get_clock().now() - rclpy.time.Time.from_msg(self.latest_imu.header.stamp)
        ).nanoseconds / 1e9
        if imu_age_sec > self.imu_timeout_sec:
            self.get_logger().warning(
                f'IMU data is {imu_age_sec:.2f}s old (> {self.imu_timeout_sec}s timeout), '
                'skipping control step', throttle_duration_sec=2.0)
            return

        self.busy = True
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
            return

        # Home-relative, matching sim's own qpos=0-at-tucked-home reference
        # -- see this module's docstring's "Home offset" section.
        motor_qpos_rad = [
            self.motor_sign[i] * (response.position_deg[i] - self.home_position_deg[i]) * DEG_TO_RAD
            for i in range(NUM_MOTORS)
        ]
        motor_qvel_rad_s = [
            self.motor_sign[i] * response.velocity_deg_s[i] * DEG_TO_RAD
            for i in range(NUM_MOTORS)
        ]

        # First control cycle: seed the slew-limiter anchor (and the
        # observation's prev_action slot) with the actual measured pose,
        # matching DogEnv.reset()'s "prev_action = current pose" semantics.
        if self.prev_action is None:
            self.prev_action = list(motor_qpos_rad)

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

        if self.dry_run_hold_pose:
            action_rad = list(motor_qpos_rad)
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor([obs], dtype=torch.float32)
                action_rad = self.policy(obs_tensor)[0].tolist()

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

        self.prev_action = clamped_action_rad

        # Inverse of the home-relative conversion above: real ABSOLUTE
        # degrees (same frame set_motor_targets/read_motor_positions use)
        # = home + sign*sim_value (sign is +-1, so sign*sign=1 undoes the
        # sign applied when motor_qpos_rad was built).
        target_deg = [
            self.home_position_deg[i] + self.motor_sign[i] * clamped_action_rad[i] * RAD_TO_DEG
            for i in range(NUM_MOTORS)
        ]

        if self.csv_writer is not None:
            self._log_row(response, motor_qpos_rad, action_rad, clamped_action_rad, target_deg)

        set_request = SetMotorTargets.Request()
        set_request.motor_id = list(range(1, NUM_MOTORS + 1))
        set_request.position_deg = target_deg
        future = self.set_targets_client.call_async(set_request)
        future.add_done_callback(self._on_targets_set)

    def _on_targets_set(self, future):
        try:
            future.result()
        except Exception as e:
            self.get_logger().error(f'set_motor_targets call failed: {e}')
        self.busy = False


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
