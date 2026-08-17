#!/usr/bin/env python3
"""Real-hardware counterpart to dog_gym's motor_sweep_log.py (sim): holds
ONE motor fixed at its current position while smoothly sweeping a SECOND
motor back and forth between two angles you choose, logging real
position/velocity/torque telemetry for BOTH motors to a CSV every tick --
so the result can be read/plotted afterward, not just watched live.

Uses actuator's own set_motor_targets/read_motor_positions services --
the same real motor-control path everything else in this project uses,
not a separate mechanism. set_motor_targets jumps STRAIGHT to whatever
position_deg it's given, with NO ramp of its own (unlike go_to_pose) --
so unlike leg_isolation_test.py (which uses go_to_pose's own ramped
speed), the safety here comes entirely from how small/slow THIS script's
own per-tick steps are. See --max-step-deg-per-tick/--rate-hz.

x_deg/z_deg and the held motor's target are in the SAME raw output-shaft-
degree frame read_motor_positions/set_motor_targets already report/expect
-- NOT a set_home-relative offset (that's go_to_pose's own convention,
see its own .srv comment). Run with the robot already positioned near
x_deg if possible; if not, the very first move (current position -> x_deg)
is ramped at the SAME --max-step-deg-per-tick safety clamp as every later
step of the sweep, never an instant jump.

SAFETY, read before running:
  - Robot must already be safely mounted (hung/on a stand), motors free
    to move without hitting anything or bearing weight.
  - Confirm the printed current position/torque of BOTH motors looks
    sane before pressing Enter to actually start moving.
  - --max-step-deg-per-tick * --rate-hz is the real, hard speed ceiling
    this script will ever command, independent of how far apart x_deg/
    z_deg are or how short --period-s is -- if the sweep's own smooth-
    target math would want to move faster than that in a single tick,
    the ACTUAL commanded step is clamped down to this instead (never
    sped back up to "catch up" later). Defaults are deliberately gentle
    (20deg/s) since this is a newer, less-exercised tool than go_to_pose.

Usage:
    ros2 run dog_deploy motor_sweep_test --ros-args \\
        -p moving_motor:=1 -p held_motor:=2 -p x_deg:=48.0 -p z_deg:=260.0

    # Faster sweep, once the gentle defaults are confirmed safe:
    ros2 run dog_deploy motor_sweep_test --ros-args \\
        -p moving_motor:=1 -p held_motor:=2 -p x_deg:=48.0 -p z_deg:=260.0 \\
        -p max_step_deg_per_tick:=2.0 -p rate_hz:=30.0
"""
import csv
import math
import os
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node

from actuator.srv import ReadMotorPositions, SetMotorTargets

NUM_MOTORS = 8
DEFAULT_MOTOR_MAPPING_PATH = os.path.join(
    get_package_share_directory('dog_description'), 'config', 'motor_mapping.yaml')


def load_motor_joint_names(motor_mapping_path=DEFAULT_MOTOR_MAPPING_PATH):
    with open(motor_mapping_path) as f:
        mapping = yaml.safe_load(f)['motors']
    return [f"{mapping[i]['leg']}_{mapping[i]['joint']}" for i in range(1, NUM_MOTORS + 1)]


class MotorSweepTest(Node):

    def __init__(self):
        super().__init__('motor_sweep_test')
        self.declare_parameter('moving_motor', 1)
        self.declare_parameter('held_motor', 2)
        self.declare_parameter('x_deg', 0.0)
        self.declare_parameter('z_deg', 90.0)
        self.declare_parameter('period_s', 8.0)
        self.declare_parameter('n_periods', 2.0)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('max_step_deg_per_tick', 1.0)
        self.declare_parameter('log_csv', 'motor_sweep_test.csv')

        self.moving_motor = self.get_parameter('moving_motor').value
        self.held_motor = self.get_parameter('held_motor').value
        self.x_deg = self.get_parameter('x_deg').value
        self.z_deg = self.get_parameter('z_deg').value
        self.period_s = self.get_parameter('period_s').value
        self.n_periods = self.get_parameter('n_periods').value
        self.rate_hz = self.get_parameter('rate_hz').value
        self.max_step_deg_per_tick = self.get_parameter('max_step_deg_per_tick').value

        self.motor_joint_names = load_motor_joint_names()

        self.read_client = self.create_client(ReadMotorPositions, 'read_motor_positions')
        self.set_targets_client = self.create_client(SetMotorTargets, 'set_motor_targets')
        for client, name in ((self.read_client, 'read_motor_positions'),
                              (self.set_targets_client, 'set_motor_targets')):
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(f'Waiting for {name} service (is actuator running?)...')

        log_path = self.get_parameter('log_csv').value
        self.csv_file = open(log_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['tick', 'elapsed_s', 'role', 'motor_id', 'joint',
                                   'target_deg', 'actual_deg', 'velocity_deg_s', 'torque_nm'])
        self.get_logger().info(f'Logging to {log_path}')

    def _read_positions(self, motor_ids):
        request = ReadMotorPositions.Request()
        request.motor_id = list(motor_ids)
        future = self.read_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            raise RuntimeError('read_motor_positions failed')
        return {mid: (pos, vel, torque) for mid, pos, vel, torque in
                zip(response.motor_id, response.position_deg, response.velocity_deg_s, response.torque_nm)}

    def _send_targets(self, targets_by_motor_id):
        request = SetMotorTargets.Request()
        request.motor_id = list(targets_by_motor_id.keys())
        request.position_deg = list(targets_by_motor_id.values())
        future = self.set_targets_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError('set_motor_targets failed')

    def _log_tick(self, tick, elapsed_s, moving_target, held_target, readings):
        for role, motor_id, target in ((('moving', self.moving_motor, moving_target),
                                         ('held', self.held_motor, held_target))):
            pos, vel, torque = readings[motor_id]
            self.csv_writer.writerow([tick, f'{elapsed_s:.3f}', role, motor_id,
                                       self.motor_joint_names[motor_id - 1],
                                       f'{target:.3f}', f'{pos:.3f}', f'{vel:.3f}', f'{torque:.4f}'])
        self.csv_file.flush()

    def run(self):
        readings = self._read_positions([self.moving_motor, self.held_motor])
        moving_current = readings[self.moving_motor][0]
        held_target = readings[self.held_motor][0]  # captured once, NEVER changed again

        self.get_logger().info(
            f'motor {self.moving_motor} ({self.motor_joint_names[self.moving_motor - 1]}) currently at '
            f'{moving_current:.2f}deg, torque={readings[self.moving_motor][2]:.2f}Nm')
        self.get_logger().info(
            f'motor {self.held_motor} ({self.motor_joint_names[self.held_motor - 1]}) currently at '
            f'{held_target:.2f}deg (this is what it will be held at), '
            f'torque={readings[self.held_motor][2]:.2f}Nm')

        # Auto-lengthen period_s so the sine sweep's own PEAK commanded
        # speed (amplitude * 2pi / period) never exceeds the hard safety
        # ceiling (max_step_deg_per_tick * rate_hz) -- without this, the
        # per-tick clamp still keeps every single step safe, but the
        # ACTUAL motor permanently lags behind a target that keeps
        # outrunning it, and the sweep never actually reaches either end
        # (exactly what happened: 212deg over 8s demands an 83deg/s peak
        # against only a 20deg/s ceiling). Same fix dog_gym's
        # verify_belt_decoupling.py/motor_sweep_log.py already apply for
        # the identical reason -- see their own comments.
        ceiling_deg_s = self.max_step_deg_per_tick * self.rate_hz
        amp_deg = abs(self.z_deg - self.x_deg) / 2
        min_period = amp_deg * 2 * math.pi / (0.9 * ceiling_deg_s)
        if self.period_s < min_period:
            self.get_logger().warning(
                f'--period-s {self.period_s:.1f}s would need a peak speed of '
                f'{amp_deg * 2 * math.pi / self.period_s:.1f}deg/s, faster than the '
                f'{ceiling_deg_s:.1f}deg/s safety ceiling -- the real motor would just lag behind '
                f'and never reach either end (this is what happened last run). Raising --period-s to '
                f'{min_period:.1f}s so the commanded sweep stays within what the clamp can actually '
                f'track. Use a higher --max-step-deg-per-tick/--rate-hz instead if you want it faster.')
            self.period_s = min_period

        self.get_logger().info(
            f'Plan: ramp motor {self.moving_motor} from {moving_current:.2f}deg to {self.x_deg:.2f}deg, '
            f'then sweep it between {self.x_deg:.2f} and {self.z_deg:.2f}deg for {self.n_periods:.1f} '
            f'cycle(s) over {self.period_s:.1f}s each, at {self.rate_hz:.0f}Hz, max '
            f'{self.max_step_deg_per_tick:.2f}deg/tick ({ceiling_deg_s:.1f}deg/s hard ceiling).')
        input('Confirm the robot is safely mounted and clear, then press Enter to start '
              '(Ctrl-C to abort)...')

        dt = 1.0 / self.rate_hz
        tick = 0
        start_time = time.time()

        def step_toward(current, target):
            delta = target - current
            step = max(-self.max_step_deg_per_tick, min(self.max_step_deg_per_tick, delta))
            return current + step

        # Phase 1: ramp smoothly from wherever the motor currently is to x_deg
        # -- never an instant jump, same per-tick clamp as the sweep itself.
        moving_cmd = moving_current
        while abs(moving_cmd - self.x_deg) > 1e-6:
            moving_cmd = step_toward(moving_cmd, self.x_deg)
            self._send_targets({self.moving_motor: moving_cmd, self.held_motor: held_target})
            readings = self._read_positions([self.moving_motor, self.held_motor])
            self._log_tick(tick, time.time() - start_time, moving_cmd, held_target, readings)
            tick += 1
            time.sleep(dt)
        self.get_logger().info(f'Reached start position ({self.x_deg:.2f}deg). Beginning sweep.')

        # Phase 2: smooth x -> z -> x sine sweep, n_periods cycles, same
        # per-tick clamp applied to whatever the smooth target math wants.
        n_steps = int(self.n_periods * self.period_s / dt)
        t = 0.0
        for i in range(n_steps):
            smooth_target = (self.x_deg + (self.z_deg - self.x_deg)
                              * (1 - math.cos(2 * math.pi * t / self.period_s)) / 2)
            moving_cmd = step_toward(moving_cmd, smooth_target)
            self._send_targets({self.moving_motor: moving_cmd, self.held_motor: held_target})
            readings = self._read_positions([self.moving_motor, self.held_motor])
            self._log_tick(tick, time.time() - start_time, moving_cmd, held_target, readings)

            if tick % 20 == 0:
                actual, vel, torque = readings[self.moving_motor]
                held_actual, held_vel, held_torque = readings[self.held_motor]
                self.get_logger().info(
                    f't={t:6.2f}s  motor{self.moving_motor} target={moving_cmd:7.2f}deg '
                    f'actual={actual:7.2f}deg torque={torque:6.2f}Nm  |  '
                    f'motor{self.held_motor} actual={held_actual:7.2f}deg torque={held_torque:6.2f}Nm')
            t += dt
            tick += 1
            time.sleep(dt)

        self.csv_file.close()
        self.get_logger().info(f'Done. {tick} ticks logged.')


def main(args=None):
    rclpy.init(args=args)
    node = MotorSweepTest()
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
