#!/usr/bin/env python3
"""ROS 2 driver node for the LSM6DSO32 IMU on the Jetson's I2C bus.

Promoted from the actuator/src/imu_reader.py prototype: same register-level
driver logic, but publishes sensor_msgs/Imu on a topic (at a configurable
rate) instead of printing to the console.

Publishes two topics:
  - imu/data_raw: the sensor's own raw X/Y/Z axes, always. Whatever
    "forward"/"left"/"up" mean here depends entirely on how the physical
    board happens to be mounted -- not meaningful on its own.
  - imu/data: the SAME readings rotated into a forward/left/up (ROS
    REP-103) body frame, via config/imu_calibration.yaml -- only
    published once that file has been produced by
    `ros2 run dog_imu calibrate_imu` (see that node's docstring for why
    this can't just be hardcoded: it depends on physical mounting, which
    changes if the IMU ever gets unplugged/remounted).
"""

from pathlib import Path

import rclpy
import smbus
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from sensor_msgs.msg import Imu

# LSM6DSO32 register map (see README.md for the full note on why these
# specific values were chosen).
WHO_AM_I_REG = 0x0F
CTRL1_XL_REG = 0x10  # Accelerometer: ODR + full-scale
CTRL2_G_REG = 0x11   # Gyroscope: ODR + full-scale
OUTX_L_G = 0x22      # Gyro X low byte; gyro Y/Z and accel X/Y/Z follow at +2 each
OUTX_L_A = 0x28      # Accel X low byte

# Sensitivity for the CTRL1_XL/CTRL2_G settings written below.
GYRO_SENSITIVITY_DPS_PER_LSB = 0.0175    # +-500 dps range
ACCEL_SENSITIVITY_G_PER_LSB = 0.000122   # +-4g range
GRAVITY_M_S2 = 9.81
DEG_TO_RAD = 0.017453292519943295


class ImuNode(Node):

    def __init__(self):
        super().__init__('imu_node')

        self.declare_parameter('bus_id', 7)
        self.declare_parameter('i2c_addr', 0x6A)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('calibration_path', str(
            Path(get_package_share_directory('dog_imu')) / 'config' / 'imu_calibration.yaml'))

        bus_id = self.get_parameter('bus_id').value
        self.addr = self.get_parameter('i2c_addr').value
        self.frame_id = self.get_parameter('frame_id').value
        publish_rate_hz = self.get_parameter('publish_rate_hz').value

        self.calibration = self._load_calibration(self.get_parameter('calibration_path').value)

        self.bus = smbus.SMBus(bus_id)

        who = self.bus.read_byte_data(self.addr, WHO_AM_I_REG)
        self.get_logger().info(f'LSM6DSO32 WHO_AM_I = 0x{who:02X}')

        # 104 Hz, +-4g
        self.bus.write_byte_data(self.addr, CTRL1_XL_REG, 0x58)
        # 104 Hz, 500 dps
        self.bus.write_byte_data(self.addr, CTRL2_G_REG, 0x54)

        self.publisher_ = self.create_publisher(Imu, 'imu/data_raw', 10)
        self.calibrated_publisher_ = (
            self.create_publisher(Imu, 'imu/data', 10) if self.calibration else None)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self.publish_imu)

    def _load_calibration(self, path):
        """Returns the parsed axes dict, or None if no calibration has
        been run yet -- imu/data (calibrated) is only published when this
        is present. See calibrate_imu_node.py."""
        path = Path(path)
        if not path.exists():
            self.get_logger().warn(
                f'No calibration file at {path} -- only imu/data_raw (raw sensor axes) will '
                "be published. Run 'ros2 run dog_imu calibrate_imu' to calibrate.")
            return None
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data.get('calibrated'):
            self.get_logger().warn(
                f'{path} exists but is not yet calibrated -- only imu/data_raw will be '
                "published. Run 'ros2 run dog_imu calibrate_imu' to calibrate.")
            return None
        self.get_logger().info(f'Loaded IMU calibration from {path} -- publishing imu/data too.')
        return data['axes']

    def _apply_calibration(self, raw_xyz):
        """(x, y, z) raw -> (forward, left, up), per self.calibration."""
        axes = self.calibration
        return (
            axes['forward']['sign'] * raw_xyz[axes['forward']['source']],
            axes['left']['sign'] * raw_xyz[axes['left']['source']],
            axes['up']['sign'] * raw_xyz[axes['up']['source']],
        )

    def read_word(self, reg):
        low = self.bus.read_byte_data(self.addr, reg)
        high = self.bus.read_byte_data(self.addr, reg + 1)
        value = (high << 8) | low
        if value > 32767:
            value -= 65536
        return value

    def publish_imu(self):
        gx_raw = self.read_word(OUTX_L_G)
        gy_raw = self.read_word(OUTX_L_G + 2)
        gz_raw = self.read_word(OUTX_L_G + 4)

        ax_raw = self.read_word(OUTX_L_A)
        ay_raw = self.read_word(OUTX_L_A + 2)
        az_raw = self.read_word(OUTX_L_A + 4)

        gyro_xyz = (gx_raw * GYRO_SENSITIVITY_DPS_PER_LSB * DEG_TO_RAD,
                    gy_raw * GYRO_SENSITIVITY_DPS_PER_LSB * DEG_TO_RAD,
                    gz_raw * GYRO_SENSITIVITY_DPS_PER_LSB * DEG_TO_RAD)
        accel_xyz = (ax_raw * ACCEL_SENSITIVITY_G_PER_LSB * GRAVITY_M_S2,
                     ay_raw * ACCEL_SENSITIVITY_G_PER_LSB * GRAVITY_M_S2,
                     az_raw * ACCEL_SENSITIVITY_G_PER_LSB * GRAVITY_M_S2)

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # No orientation estimate available from this sensor (no onboard
        # fusion) -- identity quaternion + covariance[0] = -1 tells
        # consumers to ignore the orientation field, per the sensor_msgs/Imu
        # convention.
        msg.orientation.w = 1.0
        msg.orientation_covariance[0] = -1.0

        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro_xyz
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel_xyz

        self.publisher_.publish(msg)

        if self.calibrated_publisher_ is not None:
            cal_msg = Imu()
            cal_msg.header = msg.header
            cal_msg.orientation = msg.orientation
            cal_msg.orientation_covariance = msg.orientation_covariance
            (cal_msg.angular_velocity.x, cal_msg.angular_velocity.y,
             cal_msg.angular_velocity.z) = self._apply_calibration(gyro_xyz)
            (cal_msg.linear_acceleration.x, cal_msg.linear_acceleration.y,
             cal_msg.linear_acceleration.z) = self._apply_calibration(accel_xyz)
            self.calibrated_publisher_.publish(cal_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
