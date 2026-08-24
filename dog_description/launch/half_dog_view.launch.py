"""Visualize the CAD-placed front-right leg with two controllable joints. The Onshape URDF export preserves the CAD geom..."""


from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def make_robot_description(belt_multiplier: str) -> str:
    share = Path(get_package_share_directory('dog_description'))
    urdf_path = share / 'onshape_folders' / 'urdf_half_dog_1' / 'half_dog' / 'urdf' / 'half_dog.urdf'
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    # Make installed package resources resolvable by RViz.
    old_prefix = 'package://half_dog/meshes/'
    new_prefix = 'package://dog_description/onshape_folders/urdf_half_dog_1/half_dog/meshes/'
    for mesh in robot.findall('.//mesh'):
        filename = mesh.get('filename', '')
        mesh.set('filename', filename.replace(old_prefix, new_prefix))

    # A captive spherical rubber foot is sufficient for visualization.  The
    # export expands its ball mate into a broken three-joint self-parent chain.
    for joint in list(robot.findall('joint')):
        if joint.get('name', '').startswith('ball_2__1_'):
            robot.remove(joint)
    for link in list(robot.findall('link')):
        if link.get('name', '') in {'ball_2__1_', 'ball_2__1__1'}:
            robot.remove(link)
    foot_joint = ET.SubElement(robot, 'joint', name='foot_ball_fixed', type='fixed')
    ET.SubElement(foot_joint, 'origin', xyz='0.00784525 0.00335 -0.028333', rpy='-2.60714 0.31393 -0.927346')
    ET.SubElement(foot_joint, 'parent', link='feet_connector')
    ET.SubElement(foot_joint, 'child', link='ball_for_feet')

    for joint in robot.findall('joint'):
        name = joint.get('name')
        if name == 'revolute_1__1_':
            joint.set('name', 'thigh_joint')
        elif name == 'revolute_2__1_':
            joint.set('name', 'calf_motor_joint')
        elif name == 'revolute_1':
            joint.set('name', 'calf_knee_joint')
            ET.SubElement(joint, 'mimic', joint='calf_motor_joint', multiplier=belt_multiplier, offset='0')
        elif name == 'slider_1__1_':
            # The lower pulley and lower shaft are press-fit together.
            joint.set('name', 'lower_pulley_to_shaft_fixed')
            joint.set('type', 'fixed')
            for child in list(joint):
                if child.tag in {'axis', 'limit'}:
                    joint.remove(child)

    return ET.tostring(robot, encoding='unicode')


def launch_nodes(context):
    description = make_robot_description(LaunchConfiguration('belt_multiplier').perform(context))
    share = Path(get_package_share_directory('dog_description'))
    rviz_config = share / 'rviz' / 'half_dog.rviz'
    return [
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            parameters=[{'robot_description': description}], output='screen'),
        Node(
            package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
            parameters=[{'robot_description': description}], output='screen'),
        Node(
            package='rviz2', executable='rviz2', arguments=['-d', str(rviz_config)], output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'belt_multiplier', default_value='1.0',
            description='Lower-pulley angle per upper-pulley angle. Change sign for belt direction.'),
        OpaqueFunction(function=launch_nodes),
    ])
