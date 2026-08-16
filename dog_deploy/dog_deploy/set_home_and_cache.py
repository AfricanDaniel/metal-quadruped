#!/usr/bin/env python3
"""Replaces the raw `ros2 service call /set_home actuator/srv/SetHome "{}"`
call with a single command that ALSO caches the resulting home reference to
a local file -- so `policy_node.py` can load it explicitly (via
`home_position_deg_cache_path`) instead of auto-capturing its own
`home_position_deg` from whatever position the robot happens to be in when
it starts.

Why this exists (2026-08-10, chatbot.md "running STAND then WALK-from-
standing back to back"): `policy_node.py`'s own auto-capture reads whatever
the CURRENT motor positions are at THAT node's startup -- correct only if
the robot happens to be physically tucked at that exact moment. Running two
policy_node processes in sequence (e.g. a STAND policy, then a separate
WALK-from-standing policy once the robot is already up) breaks this: the
SECOND node's auto-capture would treat the STANDING pose as home, silently
corrupting every subsequent observation/action for that policy. The fix is
to capture home ONCE, at the moment it's actually true (right when /set_home
is called, since that's the definition of "the robot is tucked right now"),
and have every later policy_node run load that SAME cached reference
explicitly instead of re-capturing.

This node calls actuator's own `/set_home` service (SetHome.srv) and writes
its response's home_deg (already exactly what real hardware just captured
as home, in motor_id order -- no separate read_motor_positions call needed,
avoiding any risk of the two calls disagreeing) to a small YAML cache file.

Usage (replaces the raw service call in your workflow):
    ros2 run dog_deploy set_home_and_cache
    ros2 run dog_deploy set_home_and_cache --ros-args -p cache_path:=/some/other/path.yaml

Then point policy_node at the cache instead of letting it auto-capture:
    ros2 run dog_deploy policy_node --ros-args ... -p home_position_deg_cache_path:=~/.dog_home_cache.yaml
"""
import os

import rclpy
import yaml
from rclpy.node import Node

from actuator.srv import SetHome
from dog_deploy.home_correction import apply_back_leg_correction

NUM_MOTORS = 8
DEFAULT_CACHE_PATH = os.path.expanduser('~/.dog_home_cache.yaml')
# 'regular' IS DEFAULT_CACHE_PATH above -- unchanged path/meaning, so
# anything already pointing at ~/.dog_home_cache.yaml keeps working
# exactly as before. 'edited' is a NEW, second, always-written snapshot
# (see home_correction.py) -- a reference/inspection copy only; deploy
# time (policy_node.py's home_switch_back_leg_fraction) recomputes the
# same correction fresh from whatever 'regular' just captured, it does
# NOT read this file, so a stale edited snapshot here can't cause a
# deploy to silently use an out-of-date correction.
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

        # response.motor_id/home_deg are parallel arrays, not guaranteed to
        # already be in motor_id-ascending order (SetHome.srv doesn't
        # document an order) -- sort explicitly so the cache is always
        # written in the canonical motor_id 1..8 order policy_node.py
        # expects, regardless of what order actuator happens to respond in.
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
