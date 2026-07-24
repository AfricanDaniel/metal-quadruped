#!/usr/bin/env python3
"""Guided IMU mounting-orientation calibration.

Why this exists: the LSM6DSO32's raw X/Y/Z axes are whatever direction
the physical board happens to be glued/screwed down in -- there's no way
to know which raw axis is "forward", "left", or "up" on the robot without
actually watching how the readings respond to a real, physical tilt. This
node walks through that physical test, records the raw /imu/data_raw
readings to a CSV, and turns them into a saved calibration (config/
imu_calibration.yaml) that imu_node.py then applies automatically to
publish a second, already-oriented topic (/imu/data) -- forward/left/up,
matching ROS REP-103 -- regardless of how the board is actually mounted.

Re-run this any time the IMU gets unplugged, remounted, or otherwise
might have moved -- it's a ~30 second guided procedure, not a one-time
setup step.

How the calibration is derived (see the long comment in _compute_calibration
for the physical reasoning): for each of the 4 tilts, the raw axis with
the largest deviation from the level baseline is identified as the
pitch-sensitive (forward) or roll-sensitive (left) axis; the remaining
axis is "up". Sign is chosen so that tilting the nose down reads negative
on the calibrated forward axis and tilting left reads negative on the
calibrated left axis (the standard accelerometer convention -- e.g. a
forward-pointing axis reads negative once the nose dips toward the
ground, exactly mirroring how the up axis reads positive at rest and
would read negative if the sensor were held upside down). Verified
against a real captured session before writing this: the nose-down/
nose-up pair excited one raw axis strongly while the other stayed near
baseline, the left/right pair excited a different raw axis, and the gyro
rotation-rate signs at the start of each tilt were self-consistent with
this same reasoning applied to angular velocity.

Usage:
    ros2 run dog_imu calibrate_imu
    ros2 run dog_imu calibrate_imu --ros-args -p hold_duration_s:=5.0

Output:
    - A timestamped CSV of every raw sample collected, under
      dog_imu/data/imu_calibration/ (or --ros-args -p csv_dir:=PATH).
    - config/imu_calibration.yaml (or --ros-args -p calibration_path:=PATH),
      loaded by imu_node.py on its next start.

To make a calibration persist across future `colcon build`s (a plain,
non-symlink install only copies config/imu_calibration.yaml from the
*source* tree at build time, so a calibration written to the installed
copy would otherwise get overwritten by the checked-in placeholder on the
next rebuild): copy the generated file over
src/dog_imu/config/imu_calibration.yaml once you're happy with it.
"""
import csv
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Imu

PHASES = ['baseline', 'nose_down', 'nose_up', 'tilt_left', 'tilt_right']
PROMPTS = {
    'baseline': 'Set the robot down LEVEL and STILL.',
    'nose_down': "Tilt the robot's NOSE (front) DOWN and hold steady.",
    'nose_up': "Tilt the robot's NOSE (front) UP and hold steady.",
    'tilt_left': "Tilt the robot so its LEFT side goes DOWN and hold steady.",
    'tilt_right': "Tilt the robot so its RIGHT side goes DOWN and hold steady.",
}
AXIS_NAMES = ['x', 'y', 'z']


class CalibrateImuNode(Node):

    def __init__(self):
        super().__init__('calibrate_imu_node')

        self.declare_parameter('settle_duration_s', 3.0)
        self.declare_parameter('hold_duration_s', 4.0)
        self.declare_parameter('csv_dir', str(
            Path(get_package_share_directory('dog_imu')) / 'data' / 'imu_calibration'))
        self.declare_parameter('calibration_path', str(
            Path(get_package_share_directory('dog_imu')) / 'config' / 'imu_calibration.yaml'))

        self.settle_s = self.get_parameter('settle_duration_s').value
        self.hold_s = self.get_parameter('hold_duration_s').value
        self.csv_dir = Path(self.get_parameter('csv_dir').value)
        self.calibration_path = Path(self.get_parameter('calibration_path').value)

        self.latest_msg = None
        self.create_subscription(Imu, 'imu/data_raw', self._on_imu, 10)

    def _on_imu(self, msg):
        self.latest_msg = msg

    def _wait_for_data(self, timeout_s=10.0):
        deadline = time.time() + timeout_s
        while self.latest_msg is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.latest_msg is None:
            raise RuntimeError(
                "No messages on imu/data_raw -- is 'ros2 run dog_imu imu_node' running?")

    def _record_phase(self, phase):
        samples = []
        end_time = time.time() + self.hold_s
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.latest_msg is not None:
                m = self.latest_msg
                samples.append({
                    'phase': phase,
                    't': time.time(),
                    'ax': m.linear_acceleration.x, 'ay': m.linear_acceleration.y,
                    'az': m.linear_acceleration.z,
                    'gx': m.angular_velocity.x, 'gy': m.angular_velocity.y,
                    'gz': m.angular_velocity.z,
                })
                self.latest_msg = None  # don't record the same message twice
        return samples

    def run(self):
        self._wait_for_data()

        all_samples = []
        for phase in PHASES:
            print(f'\n=== {phase.upper()} ===')
            print(PROMPTS[phase])
            for remaining in range(int(self.settle_s), 0, -1):
                print(f'  get in position... {remaining}', end='\r')
                time.sleep(1.0)
            print(f'  RECORDING for {self.hold_s:.0f}s -- hold still...     ')
            phase_samples = self._record_phase(phase)
            print(f'  got {len(phase_samples)} samples.')
            all_samples.extend(phase_samples)

        self.csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.csv_dir / f'imu_calibration_{int(time.time())}.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['phase', 't', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
            writer.writeheader()
            writer.writerows(all_samples)
        print(f'\nWrote raw samples to {csv_path}')

        calibration = self._compute_calibration(all_samples)

        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_path, 'w') as f:
            yaml.safe_dump(calibration, f, default_flow_style=False, sort_keys=False)
        print(f'Wrote calibration to {self.calibration_path}\n')
        self._print_summary(calibration)

    def _phase_mean_accel(self, samples, phase):
        pts = np.array([[s['ax'], s['ay'], s['az']] for s in samples if s['phase'] == phase])
        if len(pts) == 0:
            raise RuntimeError(f"No samples recorded for phase '{phase}'")
        return pts.mean(axis=0)

    def _compute_calibration(self, samples):
        """Derives (source_axis, sign) for calibrated forward/left/up from
        the recorded tilts. See this module's docstring for the physical
        reasoning -- short version: whichever raw axis moves most during
        nose-down/nose-up is "forward" (pitch-sensitive), whichever moves
        most during tilt-left/tilt-right is "left" (roll-sensitive), and
        the sign on each is chosen so tilting that direction reads
        negative -- the same convention that makes "up" read +9.8 at rest
        and negative if the sensor were upside down."""
        baseline = self._phase_mean_accel(samples, 'baseline')
        dev = {p: self._phase_mean_accel(samples, p) - baseline
               for p in ('nose_down', 'nose_up', 'tilt_left', 'tilt_right')}

        pitch_score = np.abs(dev['nose_down']) + np.abs(dev['nose_up'])
        forward_axis = int(np.argmax(pitch_score))

        roll_score = np.abs(dev['tilt_left']) + np.abs(dev['tilt_right'])
        roll_score[forward_axis] = -1  # exclude the axis already claimed by pitch
        left_axis = int(np.argmax(roll_score))

        if forward_axis == left_axis:
            raise RuntimeError(
                'Could not tell the pitch and roll axes apart -- the nose-down/up and '
                'left/right tilts excited the same raw axis most strongly. Redo the '
                'calibration with clearer, more distinct tilts.')
        up_axis = ({0, 1, 2} - {forward_axis, left_axis}).pop()

        forward_sign = 1 if dev['nose_down'][forward_axis] < 0 else -1
        left_sign = 1 if dev['tilt_left'][left_axis] < 0 else -1
        up_sign = 1 if baseline[up_axis] > 0 else -1

        return {
            'calibrated': True,
            'axes': {
                'forward': {'source': forward_axis, 'sign': forward_sign},
                'left': {'source': left_axis, 'sign': left_sign},
                'up': {'source': up_axis, 'sign': up_sign},
            },
        }

    def _print_summary(self, calibration):
        axes = calibration['axes']
        print('Derived calibration:')
        for name in ('forward', 'left', 'up'):
            a = axes[name]
            sign_str = '+' if a['sign'] > 0 else '-'
            print(f"  calibrated {name:8s} = {sign_str}raw_{AXIS_NAMES[a['source']]}")
        print('\nSanity check this makes sense (e.g. "up" should be the axis that was ~9.8 at '
              'rest, not one of the two that swung during a tilt). If anything looks wrong, '
              'redo the calibration with more deliberate tilts.\n'
              'imu_node.py will pick this up on its next start and begin publishing the '
              'calibrated /imu/data topic alongside the existing raw /imu/data_raw.')


def main(args=None):
    rclpy.init(args=args)
    node = CalibrateImuNode()
    try:
        node.run()
    except (RuntimeError, KeyboardInterrupt) as e:
        if isinstance(e, RuntimeError):
            print(f'\nCalibration failed: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
