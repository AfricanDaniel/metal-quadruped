"""Starts the two nodes policy_node depends on: actuator (motor services)
and dog_imu (imu/data_raw + imu/data). Doesn't start policy_node itself --
that one's still run manually (`ros2 run dog_deploy policy_node --ros-args
...`), since its parameters (dry_run_hold_pose, policy_path,
max_delta_deg_per_step) get changed often between test runs, unlike these
two which just need to be up and running underneath it.

position_kp/position_kd/pose_speed_deg_s launch arguments (2026-08-17,
user request -- dashboard "set live + persist across restarts" for
actuator tuning): optional overrides for the SAME-named parameters
basic_control.cpp already declares (see actuator/README.md's parameter
table). Default '' means "don't pass this key at all" -- the node then
falls back to its own declare_parameter default exactly as before this
change, rather than this launch file duplicating/hardcoding a second
copy of that default that could silently drift out of sync with the
real one over time (the README already flags position_kp/kd as "under
active hand-tuning... check the source, this table has gone stale
before" -- a second copy here would be exactly that risk).

Usage:
    ros2 launch dog_deploy hardware_bringup.launch.py
    ros2 launch dog_deploy hardware_bringup.launch.py position_kp:=45.0 position_kd:=5.0 pose_speed_deg_s:=40.0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_ACTUATOR_PARAM_NAMES = ('position_kp', 'position_kd', 'pose_speed_deg_s')


def _launch_setup(context, *args, **kwargs):
    actuator_params = {}
    for name in _ACTUATOR_PARAM_NAMES:
        value = LaunchConfiguration(name).perform(context)
        if value != '':
            actuator_params[name] = float(value)

    actuator_node_kwargs = dict(
        package='actuator',
        executable='basic_control',
        name='actuator',
        output='screen',
    )
    if actuator_params:
        actuator_node_kwargs['parameters'] = [actuator_params]

    return [
        Node(**actuator_node_kwargs),
        Node(
            package='dog_imu',
            executable='imu_node',
            name='imu_node',
            output='screen',
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('position_kp', default_value=''),
        DeclareLaunchArgument('position_kd', default_value=''),
        DeclareLaunchArgument('pose_speed_deg_s', default_value=''),
        OpaqueFunction(function=_launch_setup),
    ])
