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

        self.control_rate_hz = self.get_parameter('control_rate_hz').value
        self.max_delta_rad = (
            self.get_parameter('max_delta_deg_per_step').value * DEG_TO_RAD)
        self.imu_timeout_sec = self.get_parameter('imu_timeout_sec').value
        self.dry_run_hold_pose = self.get_parameter('dry_run_hold_pose').value

        motor_mapping_path = self.get_parameter('motor_mapping_path').value
        self.get_logger().info(f'Loading motor signs from {motor_mapping_path}')
        self.motor_sign = load_motor_signs(motor_mapping_path)

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

        self.prev_action = [0.0] * NUM_MOTORS
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

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._control_step)
        self.get_logger().info(
            f'policy_node ready: control_rate_hz={self.control_rate_hz}, '
            f'max_delta_deg_per_step={self.get_parameter("max_delta_deg_per_step").value}, '
            f'dry_run_hold_pose={self.dry_run_hold_pose}')

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

        motor_qpos_rad = [
            self.motor_sign[i] * response.position_deg[i] * DEG_TO_RAD
            for i in range(NUM_MOTORS)
        ]
        motor_qvel_rad_s = [
            self.motor_sign[i] * response.velocity_deg_s[i] * DEG_TO_RAD
            for i in range(NUM_MOTORS)
        ]

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

        # Safety clamp: never move any motor more than max_delta_rad this tick.
        clamped_action_rad = [
            motor_qpos_rad[i] + max(
                -self.max_delta_rad,
                min(self.max_delta_rad, action_rad[i] - motor_qpos_rad[i]))
            for i in range(NUM_MOTORS)
        ]
        self.prev_action = clamped_action_rad

        target_deg = [
            self.motor_sign[i] * clamped_action_rad[i] * RAD_TO_DEG
            for i in range(NUM_MOTORS)
        ]

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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
