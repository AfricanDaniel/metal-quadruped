"""Starts the two nodes policy_node depends on: actuator (motor services) and dog_imu (imu/data_raw + imu/data)."""

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
