#!/usr/bin/env python3
"""Sim-to-real bridge: runs a TorchScript policy against the real robot. Builds the same observation dog_gym's DogEnv tr..."""


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
from dog_deploy.home_correction import apply_back_leg_correction, apply_front_leg_correction

NUM_MOTORS = 8
DEFAULT_MOTOR_MAPPING_PATH = os.path.join(
    get_package_share_directory('dog_description'), 'config', 'motor_mapping.yaml')

# 3-tick observation-history stacking
SINGLE_OBS_DIM = NUM_MOTORS + NUM_MOTORS + 6 + NUM_MOTORS
OBS_HISTORY_LEN = 3

DEG_TO_RAD = 0.017453292519943295
RAD_TO_DEG = 57.29577951308232

# Real-hardware-measured calf RAW/RELATIVE range, home-relative degrees
CALF_RANGE_DEG = {
    'leg_a_calf': (0, 206.1),
    'leg_b_calf': (-213.3, 0),
    'leg_c_calf': (0, 215.8),
    'leg_d_calf': (-213.7, 0),
}

# calf_belt_sign in dog_env.py's terms
CALF_BELT_SIGN = 1.0


def ros_to_cad(x, y, z):
    """(forward, left, up) [ROS REP-103, what imu/data publishes] -> (right, front, up) [CAD-native, what dog.mjcf.xml/the p..."""

    return -y, x, z


def load_motor_signs(motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH):
    """Returns [sign_motor_1, ..., sign_motor_8] from motor_mapping.yaml."""
    with open(motor_mapping_path) as f:
        mapping = yaml.safe_load(f)['motors']
    return [float(mapping[motor_id]['sign']) for motor_id in range(1, NUM_MOTORS + 1)]


def load_motor_joint_names(motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH):
    """Returns ["leg_a_thigh", ...] motor 1..8, for log_csv readability only"""

    with open(motor_mapping_path) as f:
        mapping = yaml.safe_load(f)['motors']
    return [
        f"{mapping[motor_id]['leg']}_{mapping[motor_id]['joint']}"
        for motor_id in range(1, NUM_MOTORS + 1)
    ]


def find_calf_thigh_pairs(motor_joint_names):
    """Returns {calf_motor_index: thigh_motor_index} for all 4 legs (0-indexed, i.e."""

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
        # Rate CAP, not a guaranteed rate (see _schedule_next_tick()).
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('max_delta_deg_per_step', 5.0)
        self.declare_parameter('imu_timeout_sec', 0.5)
        self.declare_parameter('dry_run_hold_pose', True)
        # 'position' (default) or 'torque'
        self.declare_parameter('control_mode', 'position')
        # TORQUE MODE ONLY, both below.
        self.declare_parameter('max_torque_nm', [1.0] * NUM_MOTORS)
        self.declare_parameter('max_delta_torque_nm_per_step', 2.0)
        # Overridable so a corrected/test copy of motor_mapping.yaml can be used for a specific run (e.g.
        self.declare_parameter('motor_mapping_path', str(DEFAULT_MOTOR_MAPPING_PATH))
        # '' (default) disables logging.
        self.declare_parameter('log_csv', '')
        # 8 floats, or empty (default) to auto-capture from the current reading at startup instead
        self.declare_parameter('home_position_deg', [])
        # '' (default) disables this.
        self.declare_parameter('home_position_deg_cache_path', '')
        # Windup guard for the prev_action-anchored slew limiter (see _on_positions_read): max degrees the commanded target may LEAD the measured position.
        self.declare_parameter('max_target_lead_deg', 10.0)
        # 0.0 (default) disables freezing.
        self.declare_parameter('freeze_after_sec', 0.0)
        # '' (default) disables this
        self.declare_parameter('home_switch_cache_path', '')
        # Simpler alternative to home_switch_cache_path above, for the specific back-leg (motor 5/8) correction (see home_correction.py): 0.0 (default) disables this.
        self.declare_parameter('home_switch_back_leg_fraction', 0.0)
        # Same mechanism as home_switch_back_leg_fraction above, for the front legs (motors 1/4, leg_a/leg_b thigh
        self.declare_parameter('home_switch_front_leg_fraction', 0.0)
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
        # Both clients are always created (harmless), but only the one actually needed for control_mode is waited on below
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
        self.home_position_deg_after_switch = None
        home_switch_cache_path = self.get_parameter('home_switch_cache_path').value
        home_switch_back_leg_fraction = self.get_parameter('home_switch_back_leg_fraction').value
        home_switch_front_leg_fraction = self.get_parameter('home_switch_front_leg_fraction').value
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
        elif home_switch_back_leg_fraction != 0.0 or home_switch_front_leg_fraction != 0.0:
            # See home_switch_back_leg_fraction/home_switch_front_leg_ fraction's own declare_parameter comments
            self.home_position_deg_after_switch = self.home_position_deg
            if home_switch_back_leg_fraction != 0.0:
                self.home_position_deg_after_switch = apply_back_leg_correction(
                    self.home_position_deg_after_switch, home_switch_back_leg_fraction)
            if home_switch_front_leg_fraction != 0.0:
                self.home_position_deg_after_switch = apply_front_leg_correction(
                    self.home_position_deg_after_switch, home_switch_front_leg_fraction)
            self.get_logger().info(
                f'home_switch_back_leg_fraction={home_switch_back_leg_fraction}, '
                f'home_switch_front_leg_fraction={home_switch_front_leg_fraction} -- post-switch '
                f'home_position_deg computed as {self.home_position_deg_after_switch} -- '
                f'will start ramping {self.get_parameter("home_switch_after_sec").value}s into the run, '
                f'over {self.get_parameter("home_switch_ramp_sec").value}s.')
        self.home_switch_after_sec = self.get_parameter('home_switch_after_sec').value
        self.home_switch_ramp_sec = self.get_parameter('home_switch_ramp_sec').value
        self._home_switch_started_logged = False
        self._home_switch_complete_logged = False

        # CHAINED, not a fixed-rate create_timer 's own docstring for the full mechanism/ measurement this replaces)
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
        # Last column's meaning depends on control_mode: 'target_deg' (real absolute output-shaft angle sent to set_motor_targets) for position, 'command_torque_nm' (real output-shaft torque sent to set_motor_torque) for torque
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
        """Runs _control_step() again after delay_s seconds."""

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

        # Always incremented once per control tick (moved out of the freeze_after_sec-only block below so home_switch_cache_path's blend, which needs elapsed time regardless of whether freezing is even enabled, can use the same counter)
        self._control_tick_count += 1

        # See home_switch_cache_path's declare_parameter comment: blend from home_position_deg toward home_position_deg_after_switch, starting home_switch_after_sec into the run, over home_switch_ramp_sec.
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

        # First control cycle: seed the slew-limiter anchor (and the observation's prev_action slot).
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

        # 3-tick observation-history stacking
        if self._obs_history is None:
            self._obs_history = list(obs) * OBS_HISTORY_LEN
        else:
            self._obs_history = self._obs_history[SINGLE_OBS_DIM:] + list(obs)
        stacked_obs = self._obs_history

        if self.dry_run_hold_pose:
            # 'position': hold the current pose (matches this parameter's name literally).
            action_rad = [0.0] * NUM_MOTORS if self.control_mode == 'torque' else list(motor_qpos_rad)
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor([stacked_obs], dtype=torch.float32)
                action_rad = self.policy(obs_tensor)[0].tolist()

        if self.control_mode == 'torque':
            self._send_torque_command(response, motor_qpos_rad, action_rad)
            return

        # Safety clamp, anchored to the PREVIOUS COMMANDED TARGET (not the measured position)
        clamped_action_rad = [
            self.prev_action[i] + max(
                -self.max_delta_rad,
                min(self.max_delta_rad, action_rad[i] - self.prev_action[i]))
            for i in range(NUM_MOTORS)
        ]
        # Windup guard: never let the commanded target lead the measured position by more than max_target_lead_rad.
        clamped_action_rad = [
            max(motor_qpos_rad[i] - self.max_target_lead_rad,
                min(motor_qpos_rad[i] + self.max_target_lead_rad, clamped_action_rad[i]))
            for i in range(NUM_MOTORS)
        ]

        # Sliding calf-range clamp
        for calf_i, thigh_i in self.calf_thigh_pairs.items():
            lo, hi = self.calf_range_rad[calf_i]
            raw_equivalent = clamped_action_rad[calf_i] + CALF_BELT_SIGN * motor_qpos_rad[thigh_i]
            raw_clamped = max(lo, min(hi, raw_equivalent))
            clamped_action_rad[calf_i] = raw_clamped - CALF_BELT_SIGN * motor_qpos_rad[thigh_i]

        # Freeze (see freeze_after_sec's declare_parameter comment): snapshot the target ONCE, this many ticks in, then hold it fixed forever
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

        # Inverse of the home-relative conversion above: real ABSOLUTE degrees (same frame set_motor_targets/read_motor_positions use) = home + sign*sim_value (sign is +-1, so sign*sign=1 undoes the sign applied when motor_qpos_rad was built).
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
        """control_mode='torque' counterpart to the position-mode tail of _on_positions_read() above."""

        # Magnitude clamp
        max_torque_nm = self.get_parameter('max_torque_nm').value
        if len(max_torque_nm) != NUM_MOTORS:
            raise ValueError(
                f'max_torque_nm must have exactly {NUM_MOTORS} values (motor 1..8 order), '
                f'got {len(max_torque_nm)}')
        magnitude_clamped = [
            max(-max_torque_nm[i], min(max_torque_nm[i], action_rad[i]))
            for i in range(NUM_MOTORS)
        ]
        # Rate clamp, anchored to the previous COMMANDED torque (same "anchor to commanded, not measured" reasoning as position mode's slew clamp above)
        max_delta = self.get_parameter('max_delta_torque_nm_per_step').value
        clamped_action = [
            self.prev_action[i] + max(
                -max_delta, min(max_delta, magnitude_clamped[i] - self.prev_action[i]))
            for i in range(NUM_MOTORS)
        ]

        # Calf mechanical end-stop protection (see this module's docstring's "Sliding calf range" section)
        for calf_i, thigh_i in self.calf_thigh_pairs.items():
            lo, hi = self.calf_range_rad[calf_i]
            raw_equivalent = motor_qpos_rad[calf_i] + CALF_BELT_SIGN * motor_qpos_rad[thigh_i]
            if not (lo <= raw_equivalent <= hi):
                clamped_action[calf_i] = 0.0

        # Freeze (torque mode): going PASSIVE (zero torque) once triggered, NOT holding the last nonzero torque forever
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

        # Sign flip only
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

        # See _schedule_next_tick()'s own docstring for the full mechanism
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
