"""Starts the two nodes policy_node depends on: actuator (motor services)
and dog_imu (imu/data_raw + imu/data). Doesn't start policy_node itself --
that one's still run manually (`ros2 run dog_deploy policy_node --ros-args
...`), since its parameters (dry_run_hold_pose, policy_path,
max_delta_deg_per_step) get changed often between test runs, unlike these
two which just need to be up and running underneath it.

Usage:
    ros2 launch dog_deploy hardware_bringup.launch.py
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='actuator',
            executable='basic_control',
            name='actuator',
            output='screen',
        ),
        Node(
            package='dog_imu',
            executable='imu_node',
            name='imu_node',
            output='screen',
        ),
    ])
