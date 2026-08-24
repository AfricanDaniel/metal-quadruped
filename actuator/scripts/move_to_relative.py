#!/usr/bin/env python3
"""Move one motor to an exact target position specified RELATIVE TO HOME, regardless of where it currently sits. Why thi..."""

import argparse
import time

import rclpy
from rclpy.node import Node

from actuator.srv import AdjustMotorPosition, ReadMotorPositions

DEFAULT_POSE_SPEED_DEG_S = 30.0  # matches actuator's pose_speed_deg_s default


class MoveToRelative(Node):

    def __init__(self):
        super().__init__('move_to_relative')
        self.read_client = self.create_client(ReadMotorPositions, 'read_motor_positions')
        self.adjust_client = self.create_client(AdjustMotorPosition, 'adjust_motor_position')
        for client, name in ((self.read_client, 'read_motor_positions'),
                              (self.adjust_client, 'adjust_motor_position')):
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().warning(f'Waiting for {name} service (is actuator running?)...')

    def read_current(self, motor_id):
        req = ReadMotorPositions.Request()
        req.motor_id = [motor_id]
        future = self.read_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().position_deg[0]

    def adjust(self, motor_id, delta_deg):
        req = AdjustMotorPosition.Request()
        req.motor_id = motor_id
        req.degrees = delta_deg
        future = self.adjust_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--motor', type=int, required=True, help='motor_id (1-8)')
    parser.add_argument('--home-deg', type=float, required=True,
                         help="this motor's home_deg from the most recent set_home response")
    parser.add_argument('--target-deg', type=float, required=True,
                         help='desired position, in degrees RELATIVE TO HOME')
    args = parser.parse_args()

    rclpy.init()
    node = MoveToRelative()

    current = node.read_current(args.motor)
    target_absolute = args.home_deg + args.target_deg
    delta = target_absolute - current

    print(f'Motor {args.motor}: current={current:.2f} deg absolute, '
          f'target=home+{args.target_deg:.2f}={target_absolute:.2f} deg absolute '
          f'-> sending delta={delta:.2f}')

    result = node.adjust(args.motor, delta)
    print(f'adjust_motor_position: success={result.success}, '
          f'resulting_position_deg={result.resulting_position_deg:.2f} '
          f'(this is the accepted TARGET, not a confirmation it has arrived yet -- '
          f'the move is ramped, not instant)')

    # The move ramps at DEFAULT_POSE_SPEED_DEG_S deg/s in a background timer, not synchronously within the service call above
    ramp_s = abs(delta) / DEFAULT_POSE_SPEED_DEG_S + 0.3
    print(f'Waiting {ramp_s:.2f}s for the ramp to finish...')
    time.sleep(ramp_s)

    final = node.read_current(args.motor)
    print(f'Confirmed read-back: {final:.2f} deg absolute '
          f'(= home+{final - args.home_deg:.2f})')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
