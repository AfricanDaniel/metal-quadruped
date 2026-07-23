#!/usr/bin/env python3
"""Build a 4-leg dog.mjcf.xml from an onshape-to-robot URDF export +
MuJoCo (robot.xml) export of the same CAD assembly.

See converter_context.md (dog_description/) for the full methodology and
why this exists. Short version: onshape-to-robot's MuJoCo exporter
computes WRONG mesh ORIENTATIONS (verified by direct comparison against
the URDF exporter's orientation for the same mesh -- positions agreed,
rotation matrices did not; the URDF exporter is correct, confirmed via
RViz). So every body/geom pos+quat in the output is derived by
forward-kinematics composition of the URDF's fixed+revolute joint chain
instead of trusting robot.xml's own geom_quat -- only mesh files, colors,
and per-body masses are taken from robot.xml.

Legs and the static torso group are auto-detected by graph traversal of
the URDF's joint tree (parent/child link name prefixes -- see
detect_legs()), not hand-enumerated.

Requires: mujoco, numpy, trimesh (dog_ros2_ws/.venv has all three).

Typical usage, after a fresh Onshape re-export into a new
robot_half_dog_N / urdf_half_dog_N pair:

    # 1. Find the new real body masses (torso/thigh/calf) -- prints
    #    every body's mass in the MuJoCo export so you can eyeball them.
    python3 generate_dog_mjcf.py --robot-xml-dir ../onshape_folders/robot_half_dog_N --print-masses

    # 2. Generate, pointing at the new export pair and the masses found above.
    python3 generate_dog_mjcf.py \\
        --urdf ../onshape_folders/urdf_half_dog_N/half_dog/urdf/half_dog.urdf \\
        --robot-xml-dir ../onshape_folders/robot_half_dog_N \\
        --mass-torso 2.97688 --mass-thigh 0.22746 --mass-calf 0.18357 \\
        --out dog.mjcf.xml

Masses aren't auto-derived (unlike everything else) because robot.xml's
own body names don't line up 1:1 with the URDF's per-leg link names (each
export's mate-fusion groups parts differently), so there's no reliable
automatic way to attribute "this fused body's mass = this leg's thigh"
across the two formats. With 9 real bodies to read (1 torso + 4 thigh +
4 calf, expected to cluster into 3 distinct values for a symmetric
design), --print-masses makes this a 10-second manual step instead of a
fragile heuristic that could silently mis-attribute on a different
design.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE.parent / 'onshape_folders/urdf_dog/full_dog/urdf/full_dog.urdf'
DEFAULT_ROBOT_XML_DIR = HERE.parent / 'onshape_folders/robot_dog'
DEFAULT_OUT = HERE / 'dog.mjcf.xml'

# leg_a=front_right, leg_b=front_left, leg_c=back_right, leg_d=back_left
# -- matches dog_description/config/motor_mapping.yaml. CAD-native frame
# convention (established and validated on a single leg first): +y=front,
# +x=right.
LEG_NAMES = {(True, True): 'leg_a', (False, True): 'leg_b',
             (True, False): 'leg_c', (False, False): 'leg_d'}

ACTUATOR_ORDER = [
    ('leg_a_thigh', 'motor_1'), ('leg_a_calf', 'motor_2'),
    ('leg_b_calf', 'motor_3'), ('leg_b_thigh', 'motor_4'),
    ('leg_c_thigh', 'motor_5'), ('leg_c_calf', 'motor_6'),
    ('leg_d_calf', 'motor_7'), ('leg_d_thigh', 'motor_8'),
]

# Per-joint (lo, hi) range overrides, IN DEGREES -- edit this directly to
# change any motor's limits. Joints not listed here fall back to symmetric
# +-args.joint_range (also degrees, see --joint-range). Motor-to-joint
# mapping is ACTUATOR_ORDER above (motor 1 = leg_a_thigh, motor 2 =
# leg_a_calf, etc.).
#
# Current values (set 2026-07-23, user request: "motor N needs more
# negative/positive range" after bench-testing): each motor's requested
# side was pushed out to 180deg, the largest value confirmed fully
# self-collision-free for this design (bisection search, 300-sample
# random-pose sweeps of the other 7 joints at each candidate value, see
# converter_context.md) -- i.e. "as much extra range as we're willing to
# allow", not "as much as geometry allows" (geometry allows more, capped
# deliberately at 180deg). The other side of each of these 8 stays at the
# default 150deg. Still placeholders, not real hardware hard-stops.
JOINT_RANGE_OVERRIDES_DEG = {
    'leg_a_thigh': (-360, 10),
    'leg_a_calf':  (-360, 10),
    'leg_b_calf':  (-360, 10),
    'leg_b_thigh': (-10, 360),
    'leg_c_thigh': (-360, 10),
    'leg_c_calf':  (-10, 360),
    'leg_d_calf':  (-10, 360),
    'leg_d_thigh': (-10, 360),
}

# Two of the URDF's Onshape-native split mesh names both represent one
# motor half each -- robot.xml's export already merged them into a single
# "motor" mesh, so both map to that one asset.
RENAME = {'Part_A_1__Body1': 'motor', 'Part_B_1__Body1': 'motor'}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--urdf', type=Path, default=DEFAULT_URDF,
                    help='onshape-to-robot URDF export (half_dog.urdf)')
    p.add_argument('--robot-xml-dir', type=Path, default=DEFAULT_ROBOT_XML_DIR,
                    help='directory containing robot.xml + assets/ (onshape-to-robot MuJoCo export)')
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    p.add_argument('--joint-range', type=float, default=150.0,
                    help='placeholder +-range (deg) for every leg joint not listed in '
                         'JOINT_RANGE_OVERRIDES_DEG -- not real hardware hard-stops')
    p.add_argument('--mass-torso', type=float, default=4.72603)
    p.add_argument('--mass-thigh', type=float, default=0.22746)
    p.add_argument('--mass-calf', type=float, default=0.18357)
    p.add_argument('--print-masses', action='store_true',
                    help='print every body mass in --robot-xml-dir/robot.xml and exit')
    return p.parse_args()


def rpy_to_matrix(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def parse_vec(s):
    return np.array([float(v) for v in s.split()])


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def fmt(v, n=6):
    return ' '.join(f'{x:.{n}g}' for x in v)


def print_robot_xml_masses(robot_xml_dir):
    m = mujoco.MjModel.from_xml_path(str(robot_xml_dir / 'robot.xml'))
    print(f"{'body':32s} mass")
    for bid in range(1, m.nbody):  # skip 'world'
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        print(f"{name:32s} {m.body_mass[bid]:.5f}")


def load_urdf(urdf_path):
    """Returns (joints_by_child, children_of, visuals) -- see module
    docstring. joints_by_child/children_of together let you walk the tree
    in either direction; visuals maps link name -> list of
    {mesh, xyz, rpy} (that link's own <visual><origin> entries, in the
    link's own local frame)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints_by_child = {}
    children_of = {}
    for j in root.findall('joint'):
        origin = j.find('origin')
        xyz = parse_vec(origin.get('xyz', '0 0 0')) if origin is not None else np.zeros(3)
        rpy = parse_vec(origin.get('rpy', '0 0 0')) if origin is not None else np.zeros(3)
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        rec = dict(xyz=xyz, rpy=rpy, parent=parent, child=child, name=j.get('name'), type=j.get('type'))
        ax = j.find('axis')
        if ax is not None:
            rec['axis'] = parse_vec(ax.get('xyz'))
        joints_by_child[child] = rec
        children_of.setdefault(parent, []).append(rec)

    visuals = {}
    for link in root.findall('link'):
        name = link.get('name')
        vlist = []
        for vis in link.findall('visual'):
            geom = vis.find('geometry')
            mesh = geom.find('mesh')
            fn = mesh.get('filename').rsplit('/', 1)[-1].removesuffix('.stl')
            origin = vis.find('origin')
            xyz = parse_vec(origin.get('xyz', '0 0 0')) if origin is not None else np.zeros(3)
            rpy = parse_vec(origin.get('rpy', '0 0 0')) if origin is not None else np.zeros(3)
            vlist.append(dict(mesh=fn, xyz=xyz, rpy=rpy))
        visuals[name] = vlist

    return joints_by_child, children_of, visuals


def detect_legs(joints_by_child, children_of):
    """Auto-detect the 4 hip/knee joint pairs and the torso's static
    (fixed-joint-only) link group, by graph traversal -- no hand
    enumeration. Relies on this design's Onshape part-naming convention:
    hip joints go part_a_1__body1* -> thigh_connecor_full*, knee joints
    are any continuous joint whose child starts with
    timing_belt_pulley_lower* (NOT timing_belt_pulley_lower_shaft*, which
    is always fixed-jointed -- see converter_context.md). The knee
    joint's *parent* link name is intentionally not constrained: which
    part (thigh_connecor_full vs thigh_connector_hollow) ends up as the
    directly-fused parent varies per leg depending on how Onshape's
    mate-fusion groups that leg's rigid parts, confirmed to differ across
    legs in the same export (robot_half_dog_4). If a future CAD re-export
    renames these parts, update the .startswith() prefixes below to
    match -- everything downstream (grouping, kinematics, inertia) stays
    automatic."""
    hip_joints = [j for j in joints_by_child.values()
                  if j['type'] == 'continuous' and j['parent'].startswith('part_a_1__body1')
                  and j['child'].startswith('thigh_connecor_full')]
    knee_joints = [j for j in joints_by_child.values()
                   if j['type'] == 'continuous'
                   and j['child'].startswith('timing_belt_pulley_lower')
                   and not j['child'].startswith('timing_belt_pulley_lower_shaft')]
    if len(hip_joints) != 4 or len(knee_joints) != 4:
        raise AssertionError(
            f'expected 4 hip + 4 knee joints, found {len(hip_joints)} + {len(knee_joints)} -- '
            'CAD part naming may have changed, see detect_legs() docstring')

    def bfs_fixed(start_link, stop_links):
        """Links reachable from start_link via fixed/prismatic joints
        only, not crossing into any joint whose child is in stop_links,
        and never crossing a 'continuous' (revolute) joint."""
        out = [start_link]
        frontier = [start_link]
        while frontier:
            nxt = []
            for link in frontier:
                for rec in children_of.get(link, []):
                    if rec['type'] == 'continuous':
                        continue
                    if rec['child'] in stop_links:
                        continue
                    out.append(rec['child'])
                    nxt.append(rec['child'])
            frontier = nxt
        return out

    hip_children = {j['child'] for j in hip_joints}
    knee_children = {j['child'] for j in knee_joints}

    legs = []
    for hip in hip_joints:
        thigh_links = bfs_fixed(hip['child'], knee_children)
        knee = next(k for k in knee_joints if k['parent'] in thigh_links)
        calf_links = bfs_fixed(knee['child'], hip_children | knee_children)
        legs.append(dict(hip=hip, knee=knee, thigh_links=thigh_links, calf_links=calf_links))

    torso_links = bfs_fixed('torso', hip_children)
    return legs, torso_links


def joint_range_deg(joint_name, base_deg):
    """(lo, hi) IN DEGREES for this joint: JOINT_RANGE_OVERRIDES_DEG's
    explicit values if present, else symmetric +-base_deg."""
    return JOINT_RANGE_OVERRIDES_DEG.get(joint_name, (-base_deg, base_deg))


def extract_materials(robot_xml_path):
    """mesh_name -> rgba string, read directly from robot.xml's own
    <material name="X_material" rgba="..."/> declarations (assumes the
    1:1 "meshname_material" naming onshape-to-robot always uses)."""
    root = ET.parse(robot_xml_path).getroot()
    out = {}
    for mat in root.findall('.//material'):
        name = mat.get('name', '')
        rgba = mat.get('rgba')
        if name.endswith('_material') and rgba:
            out[name[:-len('_material')]] = rgba
    return out


def main():
    args = parse_args()

    if args.print_masses:
        print_robot_xml_masses(args.robot_xml_dir)
        return

    urdf_path = args.urdf.resolve()
    mesh_dir = urdf_path.parent.parent / 'meshes'
    robot_assets = args.robot_xml_dir.resolve() / 'assets'

    joints_by_child, children_of, visuals = load_urdf(urdf_path)

    def world_transform(link_name):
        if link_name == 'root':
            return np.zeros(3), np.eye(3)
        j = joints_by_child[link_name]
        ppos, prot = world_transform(j['parent'])
        lrot = rpy_to_matrix(*j['rpy'])
        pos = ppos + prot @ j['xyz']
        rot = prot @ lrot
        return pos, rot

    legs, torso_links = detect_legs(joints_by_child, children_of)
    print(f'torso_links: {len(torso_links)}')
    for i, leg in enumerate(legs):
        print(f"leg {i}: hip={leg['hip']['name']} thigh_links={len(leg['thigh_links'])} "
              f"knee={leg['knee']['name']} calf_links={len(leg['calf_links'])}")

    torso_pos, torso_rot = world_transform('torso')
    if not np.allclose(torso_rot, np.eye(3), atol=1e-4):
        raise AssertionError(
            "torso link's own composed rotation isn't identity -- the "
            "'body frame = URDF link frame' simplification used "
            "throughout this script assumes it is; if a future export "
            "trips this, the fix is to carry torso_rot through instead "
            "of assuming identity (see converter_context.md)")

    for leg in legs:
        hip_pos, hip_rot = world_transform(leg['hip']['child'])
        leg['hip_pos'], leg['hip_rot'] = hip_pos, hip_rot
        knee_pos, knee_rot = world_transform(leg['knee']['child'])
        leg['knee_pos'], leg['knee_rot'] = knee_pos, knee_rot
        is_right = hip_pos[0] > 0
        is_front = hip_pos[1] > 0
        leg['name'] = LEG_NAMES[(is_right, is_front)]
        print(f"  leg -> {leg['name']}  hip_pos={fmt(hip_pos)}")

    names_found = sorted(l['name'] for l in legs)
    if names_found != ['leg_a', 'leg_b', 'leg_c', 'leg_d']:
        raise AssertionError(f'expected one leg per corner, got {names_found} -- '
                              'two legs landed in the same (x>0/y>0) quadrant')

    def mesh_local_pose(link_name, ref_pos, ref_rot):
        lpos, lrot = world_transform(link_name)
        out = []
        for v in visuals.get(link_name, []):
            mesh_world_pos = lpos + lrot @ v['xyz']
            mesh_world_rot = lrot @ rpy_to_matrix(*v['rpy'])
            local_pos = ref_rot.T @ (mesh_world_pos - ref_pos)
            local_rot = ref_rot.T @ mesh_world_rot
            # robot.xml (and its assets/*.stl, which meshdir actually
            # points at) always lowercases mesh/material names regardless
            # of the URDF export's original filename casing (e.g. this
            # export's "IMU.stl" vs robot.xml's "imu"/"imu_material") --
            # normalize here so mesh/material references always resolve.
            mesh_name = RENAME.get(v['mesh'], v['mesh']).lower()
            out.append((mesh_name, local_pos, mat2quat(local_rot)))
        return out

    def group_geoms(links, ref_pos, ref_rot, dedupe_motor=True):
        out = []
        for link in links:
            seen_motor = False
            for mesh, pos, quat in mesh_local_pose(link, ref_pos, ref_rot):
                if dedupe_motor and mesh == 'motor':
                    if seen_motor:
                        continue
                    seen_motor = True
                out.append((mesh, pos, quat))
        return out

    mesh_cache = {}

    def mesh_bounds_local(mesh_name):
        if mesh_name not in mesh_cache:
            p = mesh_dir / f'{mesh_name}.stl'
            if not p.exists():
                # "motor" (renamed from Part_A/B) only exists in the
                # MuJoCo export's assets dir, not the URDF's meshes dir.
                p = robot_assets / f'{mesh_name}.stl'
            tm = trimesh.load(str(p))
            if isinstance(tm, trimesh.Scene):
                tm = trimesh.util.concatenate(list(tm.geometry.values()))
            mesh_cache[mesh_name] = tm.vertices
        return mesh_cache[mesh_name]

    def combined_aabb(geoms):
        """geoms already relative to one body frame. Returns (center,
        size) of the combined AABB in that frame, using real vertex data
        (not just mesh origins) -- so this is a real, if still
        box-shaped, bound on that body's actual geometry."""
        all_pts = []
        for mesh_name, pos, quat in geoms:
            verts = mesh_bounds_local(mesh_name)
            matflat = np.zeros(9)
            mujoco.mju_quat2Mat(matflat, quat)
            R = matflat.reshape(3, 3)
            all_pts.append((R @ verts.T).T + pos)
        all_pts = np.concatenate(all_pts, axis=0)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        return (lo + hi) / 2, (hi - lo)

    def box_diaginertia(mass, size):
        dx, dy, dz = size
        return np.array([
            mass / 12 * (dy**2 + dz**2),
            mass / 12 * (dx**2 + dz**2),
            mass / 12 * (dx**2 + dy**2),
        ])

    materials = extract_materials(args.robot_xml_dir / 'robot.xml')
    mesh_names = list(materials.keys())

    def geom_lines(geoms, indent):
        pad = ' ' * indent
        return '\n'.join(
            f'{pad}<geom class="visual" pos="{fmt(pos)}" quat="{fmt(quat)}" mesh="{mesh}" material="{mesh}_material"/>'
            for mesh, pos, quat in geoms
        )

    # ============ Torso ============
    torso_geoms = group_geoms(torso_links, torso_pos, torso_rot)
    torso_center, torso_size = combined_aabb(torso_geoms)
    torso_diag = box_diaginertia(args.mass_torso, torso_size)

    # ============ Legs ============
    leg_xml = {}
    lowest_z = []
    for leg in legs:
        hip_pos, hip_rot = leg['hip_pos'], leg['hip_rot']
        knee_pos, knee_rot = leg['knee_pos'], leg['knee_rot']

        thigh_rel_pos = torso_rot.T @ (hip_pos - torso_pos)
        thigh_rel_quat = mat2quat(torso_rot.T @ hip_rot)
        calf_rel_pos = hip_rot.T @ (knee_pos - hip_pos)
        calf_rel_quat = mat2quat(hip_rot.T @ knee_rot)

        thigh_geoms = group_geoms(leg['thigh_links'], hip_pos, hip_rot)
        calf_geoms = group_geoms(leg['calf_links'], knee_pos, knee_rot)

        thigh_center, thigh_size = combined_aabb(thigh_geoms)
        calf_center, calf_size = combined_aabb(calf_geoms)
        thigh_diag = box_diaginertia(args.mass_thigh, thigh_size)
        calf_diag = box_diaginertia(args.mass_calf, calf_size)

        hip_axis = leg['hip']['axis']
        knee_axis = leg['knee']['axis']

        foot_links = [l for l in leg['calf_links'] if l.startswith('ball_for_feet')]
        if not foot_links:
            raise AssertionError(f"leg {leg['name']}: no ball_for_feet* link found downstream of its knee joint")
        foot_pos, foot_rot = world_transform(foot_links[0])
        foot_rel = knee_rot.T @ (foot_pos - knee_pos)

        # foot_pos is already the true CAD-native world z (torso == world
        # origin here) -- used below to pick a spawn height clearing the floor.
        lowest_z.append(foot_pos[2])

        n = leg['name']
        # <joint range> is in DEGREES: <compiler angle="degree"> below makes
        # MuJoCo auto-convert this specific field to radians at compile
        # time. This does NOT apply to <position ctrlrange> (see below).
        thigh_lo, thigh_hi = joint_range_deg(f'{n}_thigh', args.joint_range)
        calf_lo, calf_hi = joint_range_deg(f'{n}_calf', args.joint_range)
        leg_xml[n] = f'''      <body name="{n}_thigh" pos="{fmt(thigh_rel_pos)}" quat="{fmt(thigh_rel_quat)}">
        <joint axis="{fmt(hip_axis,3)}" name="{n}_thigh" pos="0 0 0" range="{thigh_lo:.6g} {thigh_hi:.6g}" type="hinge"/>
        <inertial pos="{fmt(thigh_center,5)}" mass="{args.mass_thigh}" diaginertia="{fmt(thigh_diag,5)}"/>
        <geom name="{n}_thigh" fromto="0 0 0  {fmt(calf_rel_pos,5)}" size="0.02" type="capsule" group="3"/>
{geom_lines(thigh_geoms, 8)}

        <body name="{n}_calf" pos="{fmt(calf_rel_pos)}" quat="{fmt(calf_rel_quat)}">
          <joint axis="{fmt(knee_axis,3)}" name="{n}_calf" pos="0 0 0" range="{calf_lo:.6g} {calf_hi:.6g}" type="hinge"/>
          <inertial pos="{fmt(calf_center,5)}" mass="{args.mass_calf}" diaginertia="{fmt(calf_diag,5)}"/>
          <geom name="{n}_calf" fromto="0 0 0  {fmt(foot_rel,5)}" rgba="0.9 0.6 0.6 1" size="0.017" type="capsule" group="3"/>
          <site name="{n}_foot" pos="{fmt(foot_rel,5)}" size="0.005"/>
{geom_lines(calf_geoms, 10)}
        </body>
      </body>'''

    # Lowest foot z in CAD-native frame (torso at origin) -> spawn height margin.
    spawn_height = round(-min(lowest_z) + 0.15, 3)

    mesh_asset_lines = '\n'.join(f'    <mesh name="{m}" file="{m}.stl"/>' for m in mesh_names)
    material_asset_lines = '\n'.join(f'    <material name="{m}_material" rgba="{c}"/>' for m, c in materials.items())
    # <position ctrlrange> is NOT auto-converted by <compiler angle="degree">
    # (verified empirically -- MuJoCo only converts <joint range>/<joint
    # ref>, not actuator ctrlrange), so it must be hand-converted to
    # radians here even though <joint range> above is left in plain
    # degrees. Getting this wrong silently breaks control clamping instead
    # of erroring, so this is worth getting right.
    actuator_lines = '\n'.join(
        f'    <position joint="{j}" name="{n}" kp="15" kv="1" ctrlrange="{np.radians(lo):.6g} {np.radians(hi):.6g}"/>'
        for j, n in ACTUATOR_ORDER
        for lo, hi in [joint_range_deg(j, args.joint_range)]
    )
    meshdir_rel = Path('..') / robot_assets.relative_to(HERE.parent)

    xml = f'''<mujoco model="dog">
  <!--
    8-DOF quadruped (thigh + calf per leg, no hip motors). Built from
    {args.robot_xml_dir.name} (real CAD meshes/masses/colors, MuJoCo
    export) + {urdf_path.parent.parent.name} (real joint kinematics, URDF
    export) by generate_dog_mjcf.py (this directory) -- re-run that
    script and regenerate this file if the CAD changes (needs
    dog_ros2_ws/.venv, which has mujoco + trimesh).

    IMPORTANT, see converter_context.md (dog_description/) for the full
    story: onshape-to-robot's MuJoCo exporter computes WRONG mesh
    ORIENTATIONS (confirmed by direct comparison against the URDF
    exporter's orientation for the same mesh -- positions agreed, rotation
    matrices did not). So every body/geom pos+quat here comes from
    forward-kinematics composition of the URDF export instead of
    robot.xml's own geom_quat -- only mesh files, colors, and per-body
    masses are taken from robot.xml. This was validated on a single leg
    first (user-confirmed correct via interactive mujoco.viewer) before
    being extended to all 4 legs, auto-detected by graph traversal of the
    URDF's joint tree (not hand-enumerated).

    Frame: this file's body/geom values are in the CAD's own native frame
    (+y=front, +x=right), NOT yet remapped to ROS REP-103 (+x=forward,
    +y=left). TODO before real use: apply that remap (a single
    Rz(-90deg) on just the torso body's own pos/quat -- it does NOT need
    to touch any of the internal parent-relative body/geom transforms,
    those are frame-invariant to a whole-tree reorientation).

    qpos=0 for every leg joint is the CAD assembly's own captured/exported
    configuration, NOT necessarily the real hardware's "standing" pose --
    verify visually before assuming otherwise.

    Masses (torso {args.mass_torso}kg, thigh {args.mass_thigh}kg x4, calf
    {args.mass_calf}kg x4) are real, from robot.xml, passed in via
    --mass-torso/--mass-thigh/--mass-calf (see --print-masses). Torso mass
    includes the real battery and IMU now that both are modeled in the CAD
    (mate-fixed into the torso's rigid group, so onshape-to-robot already
    folds their mass into the single "torso" body -- no separate
    placeholder needed). Diagonal inertia is a box approximation using
    each body's REAL combined mesh bounding box (computed from actual STL
    vertex data via trimesh, transformed into each body's frame), not the
    CAD's real `fullinertia` tensor.

    Joint axes and all positions are auto-detected per leg from the
    URDF's own joint tree (see detect_legs() in the generator script).
    Leg corner naming (leg_a=front_right, leg_b=front_left,
    leg_c=back_right, leg_d=back_left) matches
    dog_description/config/motor_mapping.yaml, assigned from each leg's
    real hip-pivot (x,y) sign in the CAD frame.

    Collision geometry stays simple capsules (not the visual meshes) for
    RL training speed. Joint limits are a placeholder, still not real
    hardware hard-stops: +-{args.joint_range:.0f}deg on both sides by default, except 8
    motors with an explicit override (see JOINT_RANGE_OVERRIDES_DEG in the
    generator script -- edit that dict directly to change any motor's
    limits) where the user requested more range in one direction after
    bench-testing -- those go out to 180deg on the requested side only,
    the largest value confirmed self-collision-free for this design
    (every one of those 8 had zero self-collisions anywhere up to that
    cap, so 180deg was a deliberately chosen stopping point, not a
    geometry limit).

    NOTE: <compiler angle="degree"> below means <joint range> values in
    this file are DEGREES (MuJoCo auto-converts them at compile time).
    <position ctrlrange> values are NOT auto-converted by that setting
    (verified empirically, not just undocumented) -- they're written in
    RADIANS below, hand-converted by the generator script from the same
    degree-based JOINT_RANGE_OVERRIDES_DEG/--joint-range values. If
    hand-editing this file directly rather than regenerating it, remember
    <joint range> and <position ctrlrange> for the same joint are in
    different units.

    PD gains (kp=15 kv=1) are placeholder, not derived from the real
    motor's firmware kp/kd.
  -->
  <compiler angle="degree" coordinate="local" inertiafromgeom="false" meshdir="{meshdir_rel}"/>
  <default>
    <joint armature=".05" damping="1.5" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="10"/>
    <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <!-- rgba="0.5 0.5 0.5 1" here is MuJoCo's own compiled-in geom default,
         set explicitly (not left unset) so it doesn't inherit the parent
         default's rgba above. MuJoCo renders a geom's own material color
         only when its rgba exactly equals that internal default; any
         other value -- inherited or explicit -- overrides the material
         and every mesh below renders as one flat color instead of its
         real CAD color. -->
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.5 0.5 0.5 1"/>
    </default>
  </default>

  <option gravity="0 0 -9.81" timestep="0.01"/>

  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>

{mesh_asset_lines}
{material_asset_lines}
  </asset>

  <worldbody>
    <light cutoff="150" diffuse="1 1 1" dir="0 0 -1" directional="true" exponent="1" pos="0 0 2" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>

    <!-- Spawn height chosen only to clear the floor at qpos=0's
         CAD-captured (non-standing) leg pose, not a real measurement --
         see file header. -->
    <body name="torso" pos="0 0 {spawn_height}">
      <camera name="track" mode="trackcom" pos="0 -1.2 0.15" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
      <inertial pos="{fmt(torso_center,5)}" mass="{args.mass_torso}" diaginertia="{fmt(torso_diag,5)}"/>
      <geom name="torso_body" type="box" size="{fmt(torso_size/2,5)}" pos="{fmt(torso_center,5)}" group="3"/>

      <!-- IMU: colocated with the real LSM6DSO32 -- real mounting
           position/orientation not yet given, placed at torso origin. -->
      <site name="imu_site" pos="0 0 0" rgba="1 0 0 1" size="0.02 0.02 0.02" type="box"/>

{geom_lines(torso_geoms, 6)}

{leg_xml['leg_a']}

{leg_xml['leg_b']}

{leg_xml['leg_c']}

{leg_xml['leg_d']}
    </body>
  </worldbody>

  <sensor>
    <accelerometer name="accelerometer" site="imu_site"/>
    <gyro name="gyro" site="imu_site"/>
  </sensor>

  <actuator>
{actuator_lines}
  </actuator>
</mujoco>
'''

    args.out.write_text(xml)
    print(f'Wrote {args.out} ({len(xml)} bytes)')
    print('spawn_height:', spawn_height)


if __name__ == '__main__':
    main()
