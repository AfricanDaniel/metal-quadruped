#!/usr/bin/env python3
"""Real-hardware isolation test: distinguishes a real leg_d_thigh hardware/
CoM issue from a WALK-policy-specific learned asymmetry (see chatbot.md,
"does leg_d_thigh actually have a hardware fault?" -- STAND deployment
showed no issue, WALK deployments showed leg_d_thigh as a dominant,
one-directional outlier both times).

No RL policy involved at all -- this only uses actuator's existing
go_to_pose/read_motor_positions services plus dog_imu, to isolate the
STATIC/DYNAMIC-loading variable directly:

  1. go_to_pose('standing'), settle.
  2. For each leg in turn (front-right, front-left, back-right,
     back-left): log a baseline window, go_to_pose('standing_lift_<leg>')
     -- a preset_pose.yaml entry identical to 'standing' except that
     leg's thigh target -- hold and log, go_to_pose('standing') to lower
     it back, log a settled window.
  3. Every phase logs all 8 motors (position/velocity/torque) + IMU
     (accel/gyro) continuously, tagged with which phase/leg is active.

USES go_to_pose EXCLUSIVELY, not adjust_motor_position (2026-08-09):
an earlier version used adjust_motor_position for the per-leg lift --
tested for real, and it never actually moved the commanded motor at all
(position_deg stayed completely flat, std=0.00, across every phase of a
full run, including its own "lifted" phase). Root cause not isolated.
go_to_pose is proven working (used successfully for the STAND
deployment), so switched to that entirely instead of debugging
adjust_motor_position further -- see preset_pose.yaml's
standing_lift_a/b/c/d entries.

Read the result looking specifically at leg_d_thigh's (motor 8) torque
during EACH OTHER leg's lift phase:
  - If it spikes toward the -5N*m range specifically when leg_a or leg_c
    (the RIGHT-side legs) lift, that's evidence for a real CoM offset/
    weight-distribution issue -- independent of any policy.
  - If it stays unremarkable through all 4 phases, the WALK-specific
    loading is more likely a policy-learned asymmetry, not hardware.

SAFETY: preset_pose.yaml's standing_lift_a/b/c/d entries are PLACEHOLDER
+-20deg shifts, not verified-safe values for this specific robot's real
range of motion. Before running the full sequence, verify manually --
call `ros2 service call /go_to_pose actuator/srv/GoToPose
"{pose_name: standing_lift_a}"` for just ONE of them first, watch it,
confirm it actually lifts that foot clear of the ground rather than
driving it further down or toward a joint limit -- thigh rotation
direction for "lift" depends on this robot's specific standing geometry,
not just the sign convention in motor_mapping.yaml. Ramp SPEED is
governed by actuator's own `pose_speed_deg_s` parameter -- confirm it's
set conservatively (`ros2 param get /actuator pose_speed_deg_s`) before
running.

Usage:
    ros2 run dog_deploy leg_isolation_test --ros-args -p log_csv:=/path/to/output.csv
"""
import csv
import os
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Imu

from actuator.srv import GoToPose, ReadMotorPositions

NUM_MOTORS = 8
DEFAULT_MOTOR_MAPPING_PATH = os.path.join(
    get_package_share_directory('dog_description'), 'config', 'motor_mapping.yaml')

# motor_id (for logging labels only, the pose name is what actually
# drives the move) -> (preset_pose.yaml lift pose name, leg label).
# PLACEHOLDER lift amounts (+-20deg) live in preset_pose.yaml itself --
# see that file's standing_lift_a/b/c/d comments and this module's
# docstring for why they're unverified.
LEG_LIFT_POSES = {
    1: ('standing_lift_a', 'leg_a_front_right'),
    4: ('standing_lift_b', 'leg_b_front_left'),
    5: ('standing_lift_c', 'leg_c_back_right'),
    8: ('standing_lift_d', 'leg_d_back_left'),
}

BASELINE_S = 2.0
HOLD_S = 3.0
SETTLE_S = 2.0
LOG_HZ = 20.0


def load_motor_joint_names(motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH):
    with open(motor_mapping_path) as f:
        mapping = yaml.safe_load(f)['motors']
    return [f"{mapping[i]['leg']}_{mapping[i]['joint']}" for i in range(1, NUM_MOTORS + 1)]


class LegIsolationTest(Node):

    def __init__(self):
        super().__init__('leg_isolation_test')
        self.declare_parameter('log_csv', 'leg_isolation_test.csv')
        self.motor_joint_names = load_motor_joint_names()

        self.latest_imu = None
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self._on_imu, 10)

        self.read_client = self.create_client(ReadMotorPositions, 'read_motor_positions')
        self.pose_client = self.create_client(GoToPose, 'go_to_pose')
        for client, name in ((self.read_client, 'read_motor_positions'),
                              (self.pose_client, 'go_to_pose')):
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(f'Waiting for {name} service...')

        log_path = self.get_parameter('log_csv').value
        self.csv_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'elapsed_s', 'phase', 'motor_id', 'joint',
            'position_deg', 'velocity_deg_s', 'torque_nm',
            'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
        ])
        self.get_logger().info(f'Logging to {log_path}')
        self.start_time = time.time()

    def _on_imu(self, msg):
        self.latest_imu = msg

    def _log_once(self, phase):
        request = ReadMotorPositions.Request()
        request.motor_id = list(range(1, NUM_MOTORS + 1))
        future = self.read_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            self.get_logger().error('read_motor_positions failed during logging -- skipping row')
            return

        imu = self.latest_imu
        accel = (imu.linear_acceleration.x, imu.linear_acceleration.y,
                  imu.linear_acceleration.z) if imu is not None else (float('nan'),) * 3
        gyro = (imu.angular_velocity.x, imu.angular_velocity.y,
                 imu.angular_velocity.z) if imu is not None else (float('nan'),) * 3

        elapsed = time.time() - self.start_time
        for i in range(NUM_MOTORS):
            self.csv_writer.writerow([
                f'{elapsed:.3f}', phase, i + 1, self.motor_joint_names[i],
                response.position_deg[i], response.velocity_deg_s[i], response.torque_nm[i],
                *accel, *gyro,
            ])
        self.csv_file.flush()

    def _log_for(self, phase, duration_s):
        self.get_logger().info(f'Logging phase "{phase}" for {duration_s}s')
        end = time.time() + duration_s
        period = 1.0 / LOG_HZ
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.0)  # pump IMU callback
            self._log_once(phase)
            time.sleep(period)

    def _go_to_pose(self, pose_name):
        self.get_logger().info(f'go_to_pose({pose_name})')
        request = GoToPose.Request()
        request.pose_name = pose_name
        request.speed_deg_s = 0.0  # 0 = actuator's own default, see pose_speed_deg_s
        future = self.pose_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            msg = response.message if response is not None else 'no response'
            raise RuntimeError(f'go_to_pose({pose_name}) failed: {msg}')

    def run(self):
        self.get_logger().info(
            'Starting leg isolation test -- robot MUST already be homed '
            '(set_home called while physically tucked) before this runs, '
            'same precondition go_to_pose always has.')
        self._go_to_pose('standing')
        self._log_for('standing_settle', BASELINE_S)

        for motor_id in (1, 4, 5, 8):
            lift_pose, leg_name = LEG_LIFT_POSES[motor_id]

            self._log_for(f'baseline_before_{leg_name}', BASELINE_S)

            self._go_to_pose(lift_pose)
            self._log_for(f'lifted_{leg_name}', HOLD_S)

            self._go_to_pose('standing')
            self._log_for(f'settled_after_{leg_name}', SETTLE_S)

        self.get_logger().info('Test sequence complete -- returning to standing.')
        self._go_to_pose('standing')
        self._log_for('final_settle', BASELINE_S)
        self.csv_file.close()
        self.get_logger().info('Done. Log closed.')


def main(args=None):
    rclpy.init(args=args)
    node = LegIsolationTest()
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warning('Interrupted -- log file left as-is up to this point.')
    except Exception as e:
        node.get_logger().error(f'Test aborted: {e}')
    finally:
        if not node.csv_file.closed:
            node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
