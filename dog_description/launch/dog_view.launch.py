"""Visualize the full 4-leg dog in RViz with slider-controlled joints.

Loads onshape_folders/urdf_dog/full_dog/urdf/full_dog.urdf (the same URDF
generate_dog_mjcf.py builds dog.mjcf.xml from -- see that script's module
docstring) and, at launch time only (the exported files on disk are never
touched):
  - rewrites its mesh package:// paths so RViz can resolve them from the
    installed dog_description share directory (the raw export's paths
    assume a standalone "full_dog" package that doesn't exist here).
  - renames the 8 hip/knee joints from their raw Onshape names to
    leg_<a-d>_<thigh,calf> (matching dog_description/config/
    motor_mapping.yaml / dog.mjcf.xml's own joint names), so the
    joint_state_publisher_gui sliders are readable.
  - applies the SAME axis flips and joint ranges dog.mjcf.xml gets from
    generate_dog_mjcf.py, read live out of that script's own AXIS_FLIP /
    JOINT_RANGE_OVERRIDES_DEG definitions (parsed via ast, not imported
    -- the generator's module-level imports need mujoco/trimesh, which
    the ROS python running this launch file doesn't have). The raw URDF
    still carries the pre-AXIS_FLIP axes and unlimited `continuous`
    joints, so without this the sliders would spin 6 of 8 joints
    OPPOSITE to real life and run far past the real mechanical limits.
    Each leg joint becomes `revolute` with the sim's exact degree range
    converted to radians -- slider directions and endpoints then match
    both dog.mjcf.xml and the real robot 1:1.

This URDF has no prismatic/loop-closure joints to fix up (unlike
half_dog_view.launch.py's single-leg export) -- it's the ORIGINAL 4-leg
export, from before the (not-rolled-out, see daniel_cl_context.md's
"Cylindrical mate fix" section) cylindrical-mate CAD fix.
"""

import ast
import math
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


def load_generator_constants(share: Path):
    """Reads AXIS_FLIP and JOINT_RANGE_OVERRIDES_DEG out of the installed
    generate_dog_mjcf.py via ast (single source of truth -- importing the
    module would drag in mujoco/numpy/trimesh, unavailable here)."""
    src = (share / 'mjcf' / 'generate_dog_mjcf.py').read_text()
    wanted = {'AXIS_FLIP': None, 'JOINT_RANGE_OVERRIDES_DEG': None}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    wanted[target.id] = ast.literal_eval(node.value)
    missing = [k for k, v in wanted.items() if v is None]
    if missing:
        raise RuntimeError(f'could not parse {missing} from generate_dog_mjcf.py -- '
                           'was the dict/set renamed or made non-literal?')
    return wanted['AXIS_FLIP'], wanted['JOINT_RANGE_OVERRIDES_DEG']


def make_robot_description() -> str:
    share = Path(get_package_share_directory('dog_description'))
    urdf_path = share / 'onshape_folders' / 'urdf_dog' / 'full_dog' / 'urdf' / 'full_dog.urdf'
    tree = ET.parse(urdf_path)
    robot = tree.getroot()

    axis_flip, ranges_deg = load_generator_constants(share)

    old_prefix = 'package://full_dog/meshes/'
    new_prefix = 'package://dog_description/onshape_folders/urdf_dog/full_dog/meshes/'
    for mesh in robot.findall('.//mesh'):
        filename = mesh.get('filename', '')
        mesh.set('filename', filename.replace(old_prefix, new_prefix))

    for joint in robot.findall('joint'):
        new_name = JOINT_RENAME.get(joint.get('name'))
        if new_name is None:
            continue
        joint.set('name', new_name)

        if new_name in axis_flip:
            axis = joint.find('axis')
            axis.set('xyz', ' '.join(str(-float(v)) for v in axis.get('xyz').split()))

        lo_deg, hi_deg = ranges_deg[new_name]
        joint.set('type', 'revolute')
        limit = joint.find('limit')
        if limit is None:
            limit = ET.SubElement(joint, 'limit')
        limit.set('lower', f'{math.radians(lo_deg):.6f}')
        limit.set('upper', f'{math.radians(hi_deg):.6f}')
        # Nominal GO-M8010-6 figures -- URDF requires these attributes on
        # a revolute joint; nothing in this visualization consumes them.
        limit.set('effort', '23.7')
        limit.set('velocity', '30.0')

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
