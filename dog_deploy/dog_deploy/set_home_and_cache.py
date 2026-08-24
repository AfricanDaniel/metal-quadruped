#!/usr/bin/env python3
"""Replaces the raw `ros2 service call /set_home actuator/srv/SetHome "{}"` call with a single command that ALSO caches ..."""

import os

import rclpy
import yaml
from rclpy.node import Node

from actuator.srv import SetHome
from dog_deploy.home_correction import apply_back_leg_correction

NUM_MOTORS = 8
DEFAULT_CACHE_PATH = os.path.expanduser('~/.dog_home_cache.yaml')
# 'regular' IS DEFAULT_CACHE_PATH above
DEFAULT_EDITED_CACHE_PATH = os.path.expanduser('~/.dog_home_cache_edited.yaml')
# fraction=1.0 reproduces the original correction exactly -- see
# home_correction.py's docstring.
DEFAULT_EDITED_FRACTION = 1.0


class SetHomeAndCache(Node):

    def __init__(self):
        super().__init__('set_home_and_cache')
        self.declare_parameter('cache_path', DEFAULT_CACHE_PATH)
        self.cache_path = os.path.expanduser(self.get_parameter('cache_path').value)
        self.declare_parameter('edited_cache_path', DEFAULT_EDITED_CACHE_PATH)
        self.edited_cache_path = os.path.expanduser(self.get_parameter('edited_cache_path').value)
        self.declare_parameter('edited_fraction', DEFAULT_EDITED_FRACTION)
        self.edited_fraction = self.get_parameter('edited_fraction').value

        self.set_home_client = self.create_client(SetHome, 'set_home')
        while not self.set_home_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warning('Waiting for set_home service (is actuator running?)...')

    def run(self):
        self.get_logger().info(
            'Calling /set_home -- robot MUST already be physically tucked at the '
            'home stance now.')
        future = self.set_home_client.call_async(SetHome.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            raise RuntimeError('set_home service call failed -- aborting, cache NOT written')
        if not response.success:
            raise RuntimeError('set_home reported success=false -- aborting, cache NOT written')

        # response.motor_id/home_deg are parallel arrays, not guaranteed to already be in motor_id-ascending order (SetHome.srv doesn't document an order)
        home_by_motor_id = dict(zip(response.motor_id, response.home_deg))
        if sorted(home_by_motor_id.keys()) != list(range(1, NUM_MOTORS + 1)):
            raise RuntimeError(
                f'set_home response motor_id set {sorted(home_by_motor_id.keys())} does not '
                f'match expected 1..{NUM_MOTORS} -- aborting, cache NOT written')
        home_position_deg = {i: float(home_by_motor_id[i]) for i in range(1, NUM_MOTORS + 1)}

        with open(self.cache_path, 'w') as f:
            yaml.safe_dump({'home_position_deg': home_position_deg}, f, default_flow_style=False)
        self.get_logger().info(f'set_home succeeded, cached (regular) to {self.cache_path}: {home_position_deg}')

        # See DEFAULT_EDITED_CACHE_PATH's comment above -- a reference
        # snapshot only, deploy time doesn't read this file.
        ordered = [home_position_deg[i] for i in range(1, NUM_MOTORS + 1)]
        edited_ordered = apply_back_leg_correction(ordered, self.edited_fraction)
        edited_home_position_deg = {i: edited_ordered[i - 1] for i in range(1, NUM_MOTORS + 1)}
        with open(self.edited_cache_path, 'w') as f:
            yaml.safe_dump({'home_position_deg': edited_home_position_deg}, f, default_flow_style=False)
        self.get_logger().info(
            f'Also cached (edited, fraction={self.edited_fraction}) to '
            f'{self.edited_cache_path}: {edited_home_position_deg}')


def main(args=None):
    rclpy.init(args=args)
    node = SetHomeAndCache()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
