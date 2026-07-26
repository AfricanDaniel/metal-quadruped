"""Visualize the full 4-leg dog in RViz with slider-controlled joints.

Loads onshape_folders/urdf_dog/full_dog/urdf/full_dog.urdf (the same URDF
generate_dog_mjcf.py builds dog.mjcf.xml from -- see that script's module
docstring) and, at launch time only (the exported files on disk are never
touched):
  - rewrites its mesh package:// paths so RViz can resolve them from the
    installed dog_description share directory (the raw export's paths
    assume a standalone "full_dog" package that doesn't exist here).
  - renames the 8 continuous hip/knee joints from their raw Onshape names
    to leg_<a-d>_<thigh,calf> (matching dog_description/config/
    motor_mapping.yaml / dog.mjcf.xml's own joint names), purely so the
    joint_state_publisher_gui sliders are readable -- no structural
    change, same pattern as half_dog_view.launch.py's renames.

This URDF has no prismatic/loop-closure joints to fix up (unlike
half_dog_view.launch.py's single-leg export) -- it's the ORIGINAL 4-leg
export, from before the (not-yet-rolled-out, see daniel_cl_context.md's
"Cylindrical mate fix" section) cylindrical-mate CAD fix. Every leg joint
here is a plain `continuous` hinge.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import OpaqueFunction
from launch_ros.actions import Node

# Raw Onshape joint name -> friendly name, matching motor_mapping.yaml's
# leg_<letter>_<thigh,calf> convention. Derived from each joint's own
# name prefix (frl=front_right_leg=leg_a, fll=front_left_leg=leg_b,
# brl=back_right_leg=leg_c, bll=back_left_leg=leg_d -- confirmed against
# motor_mapping.yaml's leg_a=front_right/leg_b=front_left/
# leg_c=back_right/leg_d=back_left) and joint kind (*_motor_N_and_thigh =
# hip/thigh joint, *_thigh_and_lower_pulley = knee/calf joint).
JOINT_RENAME = {
    '_frl__revolute_motor_1_and_thigh': 'leg_a_thigh',
    '_frl__revolute_thigh_and_lower_pulley': 'leg_a_calf',
    '_fll__revolute_motor_4_and_thigh': 'leg_b_thigh',
    '_fll__revolute_thigh_and_lower_pulley': 'leg_b_calf',
    '_brl__revolute_motor_5_and_thigh': 'leg_c_thigh',
    '_brl__revolute_thigh_and_lower_pulley': 'leg_c_calf',
    '_bll__revolute_motor_8_and_thigh': 'leg_d_thigh',
    '_bll__revolute_thigh_and_lower_pulley': 'leg_d_calf',
}


def make_robot_description() -> str:
    share = Path(get_package_share_directory('dog_description'))
    urdf_path = share / 'onshape_folders' / 'urdf_dog' / 'full_dog' / 'urdf' / 'full_dog.urdf'
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    old_prefix = 'package://full_dog/meshes/'
    new_prefix = 'package://dog_description/onshape_folders/urdf_dog/full_dog/meshes/'
    for mesh in robot.findall('.//mesh'):
        filename = mesh.get('filename', '')
        mesh.set('filename', filename.replace(old_prefix, new_prefix))

    for joint in robot.findall('joint'):
        new_name = JOINT_RENAME.get(joint.get('name'))
        if new_name is not None:
            joint.set('name', new_name)

    return ET.tostring(robot, encoding='unicode')


def launch_nodes(context):
    description = make_robot_description()
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
        OpaqueFunction(function=launch_nodes),
    ])
