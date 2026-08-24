#!/usr/bin/env python3
"""Unloaded-observation calibration check (2026-08-12, "front
knees tucked in / walking on shins" -- real-hardware deficit isolated to
motors 1-4's front-leg thigh amplitude, next step is checking whether
that's an observation-build/calibration bug vs. a genuine physical
loading difference sim never modeled).

READ-ONLY: this node calls ONLY `read_motor_positions` and subscribes to
`imu/data` -- it never calls `set_motor_targets`/`set_motor_torque` and
holds no command client at all, so it is structurally incapable of
moving the robot, unlike policy_node.py's `dry_run_hold_pose`. Suspend
the robot off the ground (or otherwise support its own weight) yourself,
then optionally pose it by hand at a known reference -- this node just
watches and logs what the software THINKS that pose is.

Builds the exact same `motor_qpos_rad` (home-relative, sign-corrected)
policy_node.py computes for the policy's own observation -- same
load_motor_signs()/home-offset logic, imported directly from
policy_node so the two can never silently drift apart. Prints a live
per-motor table each tick: real absolute degrees, the computed
home-relative "observation" value, and (if --expected-pose is given) how
far that is from what a correctly-calibrated reading SHOULD show at a
known reference pose -- 'home' (all motors should read ~0) or 'standing'
(should match dog_env.py's STANDING_QPOS_DEG, duplicated below rather
than imported since dog_deploy is meant to run without dog_gym
installed -- see policy_node.py's own module docstring for why. Compared
in the ABSOLUTE/belt-decoupled frame, matching obs_deg -- see
_standing_absolute_deg()).

Usage (home reference must match whatever the real deployed policy used
-- pass the SAME home_position_deg/home_position_deg_cache_path you'd
give policy_node, not a fresh auto-capture, or the comparison is
meaningless):
    # 1) Physically pose the robot (suspended, unloaded) at its tucked
    #    home stance, then:
    ros2 run dog_deploy log_static_observation --ros-args \\
        -p home_position_deg_cache_path:=~/.dog_home_cache.yaml \\
        -p expected_pose:=home

    # 2) Physically pose the robot at roughly the standing stance, then:
    ros2 run dog_deploy log_static_observation --ros-args \\
        -p home_position_deg_cache_path:=~/.dog_home_cache.yaml \\
        -p expected_pose:=standing

What to look for: motors 1-4 (leg_a_thigh, leg_a_calf, leg_b_calf,
leg_b_thigh) showing a large, persistent diff_deg while motors 5-8 read
correctly at the SAME physical pose is direct evidence of a front-leg-
specific calibration bug (motor_mapping.yaml sign, or a bad home
capture) independent of any physics/training question -- since nothing
is loaded and nothing is moving, there's no gravity/friction/dynamics
left to blame.
"""
import os

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Imu

from actuator.srv import ReadMotorPositions
from dog_deploy.policy_node import (
    CALF_BELT_SIGN, NUM_MOTORS, find_calf_thigh_pairs, load_motor_joint_names,
    load_motor_signs, ros_to_cad,
)

# Motor 1..8 order -- duplicated from dog_gym/envs/dog_env.py's own
# STANDING_QPOS_DEG (see that constant's comment there for its
# derivation/history). Keep in sync if it's ever re-tuned.
#
# THIS IS THE RAW, THIGH-COUPLED (MuJoCo-native) FRAME -- confirmed by
# dog_env.py's own conversion right below where it's defined:
# `walk_default_rad[calf_idx] -= calf_belt_sign * qpos[thigh_idx]` builds
# the ABSOLUTE/belt-decoupled equivalent FROM this constant, meaning the
# constant itself is NOT already absolute. Bug found 2026-08-12 (user
# caught it -- "front knees tucked in / walking on shins"): an
# earlier version of this file compared this raw array directly against
# `obs_deg` (which IS absolute/decoupled, matching real hardware's own
# motor_qpos_deg convention) -- an apples-to-oranges comparison that
# produced a misleading ~90deg "diff" on every calf motor. See
# _standing_absolute_deg() below for the fix -- same raw-to-absolute
# transform dog_env.py itself applies, not duplicated ad hoc.
STANDING_QPOS_DEG_RAW = np.array(
    [107.507, 104.071, -86.789, -98.804, 98.743, 98.011, -93.049, -103.804])


def _standing_absolute_deg(motor_joint_names):
    """STANDING_QPOS_DEG_RAW converted to the ABSOLUTE, belt-decoupled
    frame -- directly comparable to obs_deg/motor_qpos_deg, matching
    dog_env.py's own `walk_default_rad` conversion exactly (calf_idx -=
    calf_belt_sign * qpos[thigh_idx]; thighs are unaffected, they're not
    belt-coupled)."""
    absolute_deg = STANDING_QPOS_DEG_RAW.copy()
    for calf_i, thigh_i in find_calf_thigh_pairs(motor_joint_names).items():
        absolute_deg[calf_i] -= CALF_BELT_SIGN * STANDING_QPOS_DEG_RAW[thigh_i]
    return absolute_deg


class LogStaticObservation(Node):

    def __init__(self):
        super().__init__('log_static_observation')
        self.declare_parameter('motor_mapping_path', '')
        self.declare_parameter('log_csv', '')
        self.declare_parameter('log_rate_hz', 5.0)
        # 'none' (default): just log raw readings, no comparison column.
        # 'home': every motor's motor_qpos_deg should read ~0 (that's the
        # definition of home). 'standing': should match STANDING_QPOS_DEG
        # above. Set to whatever pose you've actually physically posed
        # the (unloaded/suspended) robot at.
        self.declare_parameter('expected_pose', 'none')
        # Same two options, same semantics, as policy_node.py's own
        # params -- see that file's matching declare_parameter comments.
        # Passing neither auto-captures from the CURRENT reading, which
        # is almost never what you want here: you want to compare against
        # the SAME home a real deployed policy used, not a fresh capture.
        self.declare_parameter('home_position_deg', [])
        self.declare_parameter('home_position_deg_cache_path', '')

        motor_mapping_path = self.get_parameter('motor_mapping_path').value
        if not motor_mapping_path:
            from dog_deploy.policy_node import DEFAULT_MOTOR_MAPPING_PATH
            motor_mapping_path = str(DEFAULT_MOTOR_MAPPING_PATH)
        self.get_logger().info(f'Loading motor signs from {motor_mapping_path}')
        self.motor_sign = load_motor_signs(motor_mapping_path)
        self.motor_joint_names = load_motor_joint_names(motor_mapping_path)

        expected_pose = self.get_parameter('expected_pose').value
        if expected_pose not in ('none', 'home', 'standing'):
            raise ValueError(
                f"expected_pose must be 'none', 'home', or 'standing', got {expected_pose!r}")
        self.expected_deg = (
            None if expected_pose == 'none'
            else np.zeros(NUM_MOTORS) if expected_pose == 'home'
            else _standing_absolute_deg(self.motor_joint_names))

        self.tick = 0
        self.csv_file = None
        self.csv_writer = None
        log_csv_path = self.get_parameter('log_csv').value
        if log_csv_path:
            self._open_log(log_csv_path)

        self.latest_imu = None
        self.busy = False
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self._on_imu, 10)

        # READ-ONLY: no set_motor_targets/set_motor_torque client exists
        # anywhere in this node -- see this module's docstring.
        self.read_client = self.create_client(ReadMotorPositions, 'read_motor_positions')
        while not self.read_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warning('Waiting for read_motor_positions service (is actuator running?)...')

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
            cache_path = os.path.expanduser(home_cache_path)
            self.get_logger().info(f'Loading home_position_deg from cache: {cache_path}')
            with open(cache_path) as f:
                cached = yaml.safe_load(f)['home_position_deg']
            self.home_position_deg = [cached[i] for i in range(1, NUM_MOTORS + 1)]
            self.get_logger().info(f'Loaded home_position_deg from cache: {self.home_position_deg}')
        else:
            self.get_logger().warning(
                'No home_position_deg or home_position_deg_cache_path given -- auto-capturing '
                'from the CURRENT reading. This is almost never what you want for a calibration '
                'check: pass the SAME home reference a real deployed policy used, or this '
                'comparison is meaningless (everything will trivially read ~0 relative to itself).')
            request = ReadMotorPositions.Request()
            request.motor_id = list(range(1, NUM_MOTORS + 1))
            future = self.read_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response is None:
                raise RuntimeError('read_motor_positions failed while capturing home -- aborting')
            self.home_position_deg = list(response.position_deg)
            self.get_logger().info(f'Auto-captured home_position_deg: {self.home_position_deg}')

        log_rate_hz = self.get_parameter('log_rate_hz').value
        self.timer = self.create_timer(1.0 / log_rate_hz, self._tick)
        self.get_logger().info(
            f'log_static_observation ready: log_rate_hz={log_rate_hz}, expected_pose={expected_pose}. '
            'READ-ONLY -- no motor commands will ever be sent by this node.')

    def _open_log(self, path):
        import csv
        self.csv_file = open(path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'tick', 'motor_id', 'joint', 'sign',
            'real_position_deg', 'motor_qpos_deg', 'expected_deg', 'diff_deg',
        ])
        self.get_logger().info(f'Logging to {path}')

    def _on_imu(self, msg):
        self.latest_imu = msg

    def _tick(self):
        if self.busy:
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

        # Same home-relative, sign-corrected convention as policy_node.py's
        # own observation build -- see that module's docstring's "Home
        # offset" section.
        motor_qpos_deg = [
            self.motor_sign[i] * (response.position_deg[i] - self.home_position_deg[i])
            for i in range(NUM_MOTORS)
        ]

        print(f'--- tick {self.tick} ---')
        header = f'{"motor":>5} {"joint":>14} {"real_deg":>10} {"obs_deg":>10}'
        if self.expected_deg is not None:
            header += f' {"expected":>10} {"diff":>10}'
        print(header)
        for i in range(NUM_MOTORS):
            line = (f'{i + 1:>5} {self.motor_joint_names[i]:>14} '
                    f'{response.position_deg[i]:>10.2f} {motor_qpos_deg[i]:>10.2f}')
            if self.expected_deg is not None:
                diff = motor_qpos_deg[i] - self.expected_deg[i]
                flag = '  <-- LARGE DIFF' if abs(diff) > 15.0 else ''
                line += f' {self.expected_deg[i]:>10.2f} {diff:>10.2f}{flag}'
            print(line)

        if self.latest_imu is not None:
            imu = self.latest_imu
            accel_cad = ros_to_cad(imu.linear_acceleration.x, imu.linear_acceleration.y,
                                    imu.linear_acceleration.z)
            gyro_cad = ros_to_cad(imu.angular_velocity.x, imu.angular_velocity.y,
                                   imu.angular_velocity.z)
            print(f'  IMU (CAD frame) accel={tuple(round(a, 3) for a in accel_cad)} '
                  f'gyro={tuple(round(g, 3) for g in gyro_cad)}')
        else:
            print('  IMU: no data received yet')

        if self.csv_writer is not None:
            for i in range(NUM_MOTORS):
                expected = self.expected_deg[i] if self.expected_deg is not None else ''
                diff = (motor_qpos_deg[i] - self.expected_deg[i]) if self.expected_deg is not None else ''
                self.csv_writer.writerow([
                    self.tick, i + 1, self.motor_joint_names[i], self.motor_sign[i],
                    response.position_deg[i], motor_qpos_deg[i], expected, diff,
                ])
            self.csv_file.flush()

        self.tick += 1
        self.busy = False

    def destroy_node(self):
        if self.csv_file is not None:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LogStaticObservation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
