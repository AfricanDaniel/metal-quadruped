# CAD -> dog.mjcf.xml converter: context for other tools/LLMs

This explains how `dog_description/mjcf/dog.mjcf.xml` is generated from
an Onshape CAD export, why it's built the way it is, and how to redo it
after a future CAD change. Written to be handed to a different LLM or
person with no other context on this project.

## The inputs: two exports of the same CAD assembly

[`onshape-to-robot`](https://github.com/Rhoban/onshape-to-robot) exports
one Onshape assembly into two different formats, both checked into
`dog_description/onshape_folders/`:

- **`robot_half_dog_N/`** -- MuJoCo format: `robot.xml` + `scene.xml` +
  `assets/*.stl`. `config.json` has `"output_format": "mujoco"`.
- **`urdf_half_dog_N/<assembly_name>/`** -- URDF format:
  `urdf/<assembly_name>.urdf` + `meshes/*.stl` (or `.gltf` in older
  exports -- MuJoCo 3.1.6 can't load `.gltf` directly, only `.stl`). The
  inner folder/file basename tracks whatever the Onshape assembly is
  named at export time (`half_dog` for `_1`-`_3`, `full_dog` for `_4`) --
  always pass the actual path via `--urdf`, don't assume the basename.

`N` increments each time the CAD assembly is re-exported (`_1`, `_2`,
`_3`, `_4`, ...) -- except the naming convention isn't fully stable:
later re-exports dropped the numeric suffix (`robot_full_dog` /
`urdf_full_dog`, then `robot_dog` / `urdf_dog`). Don't assume a folder
name pattern; check `onshape_folders/` directly for the newest pair. The
**current** pair, used by the generator script's defaults, is
`robot_dog` / `urdf_dog` (all 4 legs, same design as `robot_full_dog`
plus a real battery + real IMU added to the CAD -- see
`claude_context.md` for what changed). Earlier pairs (`_1`, `_2`) were
single-leg validation exports; `_3` was the first all-4-legs export;
`_4` and `robot_full_dog` were later re-exports of the same design;
all kept for history.

## The core problem: the MuJoCo exporter's mesh orientations are wrong

This is the whole reason the converter script exists instead of just
using `robot.xml` directly.

**Finding**: `onshape-to-robot`'s MuJoCo exporter computes a **wrong
mesh orientation** for at least some bodies. Directly compared, for the
same mesh (the torso plate), the two exports' own reported transforms:

- `robot.xml`'s geom position and the URDF's forward-kinematics-composed
  position agreed to within ~4mm.
- Their **rotation matrices were completely different** -- not
  numerically close, an actually different rotation.

The URDF export's orientation is the correct one (independently
confirmed: loading the URDF into RViz via `robot_state_publisher` shows
the robot in the correct real-world orientation/pose). So:

- **Trust `robot.xml` for**: mesh files (`assets/*.stl`), per-body
  masses, and material colors. These are unaffected by the orientation
  bug -- geometry/mass computation doesn't depend on the same placement
  transform that's buggy.
- **Do NOT trust `robot.xml` for**: any geom or body `pos`/`quat`. Its
  joints are also useless for a different reason -- see next section.
- **Trust the URDF for**: every position and orientation, and joint
  axes. Derive everything geometric by forward-kinematics composition of
  the URDF's `<joint><origin xyz=".." rpy=".."/></joint>` chain, not by
  reading `robot.xml`'s `geom_quat`/`geom_pos` arrays.

This was found by building a single-leg test file first
(`robot_half_dog_2/leg_test.mjcf.xml`), getting it visually confirmed
wrong twice by direct human inspection in the interactive MuJoCo viewer
(`python3 -m mujoco.viewer --mjcf=...`) despite passing my own offscreen
`mujoco.Renderer` comparisons -- the offscreen checks were worthless
because they compared the rebuild against `robot.xml`'s own (equally
buggy) raw appearance, not an independent ground truth. Only after
switching to URDF-derived transforms and re-checking interactively did
it come out correct. **Lesson: an automated render check is only
meaningful against an independent ground truth. If you rebuild from
source A and then verify by rendering-and-comparing against source A's
own output, you will not catch a bug that's already in source A.**

## The second problem: `robot.xml`'s own joints are unusable

Separately from the orientation bug: unless every Onshape mate involved
is a "real" revolute/mate type the exporter recognizes, `onshape-to-robot`'s
MuJoCo export drops joint information entirely and emits disconnected
`<freejoint>` bodies (`njnt` = number of floating bodies, no
`<equality>`/real hinges). This project's mates hit that case. The URDF
export, by contrast, DOES correctly recover the real revolute mates (as
`<joint type="continuous">`) even when the MuJoCo export fails to. So
joint axes and pivot locations also have to come from the URDF, not
`robot.xml`.

The URDF also has some of its own noise to clean up (not silently
wrong, just needs handling):
- A slider/prismatic joint (`slider_1__1_` in this design) that's really
  a press-fit, fixed connection, not a real telescoping joint (unbounded
  `+-10000` limit is the tell). Not load-bearing for the converter script
  (it's inside a fixed/rigid sub-chain that gets grouped as one rigid
  MJCF body regardless of whether that link's own joint is literally
  `fixed` or `prismatic`-but-treated-as-fixed), but worth knowing about
  if you're reading the raw URDF by hand.
- Two Onshape-native split mesh files per motor (`Part_A_1__Body1.stl`,
  `Part_B_1__Body1.stl`, sharing one local origin) that `robot.xml`'s
  export already merged into one `motor.stl` mesh -- the converter
  script maps both split names to `robot.xml`'s single merged mesh and
  emits it once (see `RENAME` in the generator script).
- The URDF export preserves whatever casing the Onshape part had
  (`IMU.stl`, added in the `robot_dog`/`urdf_dog` export), while
  `robot.xml` always lowercases mesh/material names (`imu`,
  `imu_material`). Since the final MJCF's `meshdir` points at
  `robot.xml`'s own `assets/` (not the URDF's `meshes/`), every mesh
  name the converter emits must match `robot.xml`'s lowercase spelling
  or the mesh/material references in the output file won't resolve.
  Fixed by lowercasing every URDF-derived mesh name (after the `RENAME`
  step) in `mesh_local_pose()`.

## The third problem: MuJoCo's own default-inheritance color trap

Not a CAD-export bug -- a MuJoCo semantics gotcha the generator script
itself fell into, worth flagging since it's non-obvious and easy to
reintroduce. MuJoCo's rule for `<geom rgba="...">` vs. an assigned
`material`: **the geom's own material color is only used if that geom's
*compiled* `rgba` exactly equals MuJoCo's internal default, `0.5 0.5 0.5
1`.** Any other value -- however it got there, explicit or inherited
through `<default>` classes -- overrides the material and the geom
renders as flat `rgba`, not the material's color.

The script emits every visual geom under `<default class="visual">`,
nested inside the file's top-level `<default>` which sets `rgba="0.8 0.6
.4 1"` for collision geoms. Two versions of this script both got the
color wrong the same way:
1. First version set `rgba="1 1 1 1"` on the visual class explicitly --
   differs from `0.5 0.5 0.5 1`, so every mesh rendered plain white,
   masking every material color (even correct ones already in
   `robot.xml`).
2. Removing that line entirely didn't fix it either: MJCF `<default>`
   classes inherit unset attributes from their parent, so the visual
   class geoms inherited the *parent* default's `rgba="0.8 0.6 .4 1"` --
   same bug, different flat color.

The actual fix: give `<default class="visual">` its own explicit
`rgba="0.5 0.5 0.5 1"` -- MuJoCo's real internal default, spelled out on
purpose so it can't inherit anything else -- which breaks the
inheritance chain and lets each geom's own `material` govern its color.
Verified by checking `mjv_updateScene`'s resolved `rgba` per geom (not
just the compiled model's `geom_rgba`, which doesn't tell you what
actually renders) matches each material's declared color.

## The conversion method

Implemented in `mjcf/generate_dog_mjcf.py`. In order:

1. **Parse the URDF** into two lookups: `joints_by_child` (child link
   name -> `{xyz, rpy, parent, type, axis?}`, i.e. that joint's fixed
   transform in its parent's frame) and `visuals` (link name -> list of
   that link's own `<visual><origin xyz=".." rpy=".."/><mesh
   filename=".."/></visual>` entries, i.e. mesh placement within its own
   link's frame).

2. **Forward-kinematics composition** (`world_transform(link)`):
   recursively compose from the URDF's `root` link down to any target
   link, using standard rigid transform composition (`pos = parent_pos +
   parent_rot @ local_xyz`, `rot = parent_rot @ Rz(yaw)@Ry(pitch)@Rx(roll)`
   for each joint's own `rpy`). This is evaluated at joint value 0 for
   every joint (i.e. it computes the CAD assembly's own
   as-exported/at-rest configuration) -- revolute and prismatic joints
   contribute nothing beyond their own fixed `origin`, since a joint
   variable of 0 means "no additional rotation/translation past the
   origin."

3. **Auto-detect the 4 legs and the static torso group** by graph
   traversal, not hand enumeration (72 links / 71 joints in the 4-leg
   URDF -- too many and too oddly-named, e.g. `revolute_1__1___1___1_`,
   to reliably do by hand):
   - Hip joints: any `type="continuous"` joint whose parent link name
     starts with `part_a_1__body1` and child starts with
     `thigh_connecor_full`.
   - Knee joints: any `type="continuous"` joint whose child link name
     starts with `timing_belt_pulley_lower` but NOT
     `timing_belt_pulley_lower_shaft` (that longer name is a different
     part, always joined by a `fixed` joint, so the `continuous` filter
     alone would already exclude it -- excluded explicitly anyway for
     clarity). The knee joint's *parent* link is deliberately NOT
     constrained by name: confirmed in the `_4` export that which part
     (`thigh_connecor_full` vs `thigh_connector_hollow`) ends up as the
     directly-fused parent varies leg-to-leg, depending on how Onshape's
     mate-fusion happened to group that leg's rigid parts on that
     export. Constraining on parent name (as an earlier version of the
     script did) silently under-counts knee joints on any export where
     this varies -- if it happens again, the fix is the same: drop the
     parent-name filter, keep only the child-name + `continuous` filter.
   - For each hip joint, BFS forward through **fixed joints only**
     (never crossing a `continuous` joint) from the hip's child link,
     stopping at any knee joint's child -- this walks exactly that leg's
     thigh-side static sub-assembly (spacers, connectors) and finds
     which knee joint is downstream of it.
   - Same BFS from each knee's child link (through fixed *and*
     prismatic joints, since the "slider" is really fixed) gives that
     leg's calf-side sub-assembly down to the foot ball.
   - Everything reachable from the URDF's `torso` link via fixed joints,
     without ever crossing into any hip joint's child, is the static
     torso group (spine + all 8 motors/casings + 4 calf-drive upper
     pulleys -- all rigidly fused in this design, per an earlier CAD-mate
     fix; see "belt mechanism" note below).
   - This traversal is why the converter is robust to *pose* changes
     (leg repositioned in Onshape) but depends on Onshape's *part naming
     staying consistent* -- if a future re-export renames these parts,
     update the four name prefixes in `detect_legs()`.

4. **Classify each leg's corner** from its hip pivot's `(x, y)` sign in
   the CAD's own native frame (`+x = right, +y = front`, established and
   validated against the single leg the user identified by hand as
   "front right"). Maps to `leg_a`/`leg_b`/`leg_c`/`leg_d` matching
   `dog_description/config/motor_mapping.yaml`
   (`leg_a`=front_right, `leg_b`=front_left, `leg_c`=back_right,
   `leg_d`=back_left).

5. **Build each MJCF body's frame = the corresponding URDF link's own
   composed-from-root frame.** Concretely: the `leg_a_thigh` MJCF body's
   `pos`/`quat` (relative to its MJCF parent, the torso body) is set so
   that body's frame exactly equals `world_transform('thigh_connecor_full_N')`
   (the hip pivot's own real frame), and similarly `leg_a_calf`'s frame =
   `world_transform('timing_belt_pulley_lower_N')` (the knee pivot's
   real frame) relative to the thigh body. Every mesh geom inside a body
   is then just that mesh's own `<visual><origin>` value, composed
   through its link's world transform and re-expressed relative to the
   containing body's frame (`local = body_rot^T @ (mesh_world -
   body_world)`). Since all of this is *relative* geometry (differences
   between sibling-composed frames), a global re-orientation of the
   whole tree cancels out everywhere except at the single top-level body
   -- see the "ROS frame remap" note below.

6. **Joint axis** = the URDF's own `<joint><axis xyz=".."/>` value
   directly, no sign-flipping or reinterpretation -- it's already
   expressed in the child link's own local frame, which is now that
   MJCF body's frame by construction (step 5).

7. **Masses**: real per-body values read from `robot.xml` (NOT
   auto-attributed across formats -- see "why masses need a manual
   step" below). Diagonal inertia is a **box approximation using each
   body's real combined mesh bounding box** -- not a guessed box size,
   the actual STL vertex data (loaded via `trimesh`) for every mesh
   assigned to that body, transformed by that mesh's already-computed
   local pos/quat, unioned into one AABB. Still an approximation (a box,
   not the CAD's true `fullinertia` tensor), but a real one.

8. **Collision geometry** stays simple capsules (hip-pivot-to-knee-pivot,
   knee-pivot-to-foot), not the visual meshes -- for RL training speed.
   The capsule `fromto` vectors are the real measured hip->knee and
   knee->foot vectors (not assumed axis-aligned).

9. **Emit the MJCF text directly** (an f-string template), not XML-tree
   manipulation -- with ~80 geoms across 4 legs, hand-transcribing
   printed fragments (an earlier, smaller-scale approach used for the
   single-leg file) becomes error-prone; direct generation removes that
   step entirely.

10. **Joint ranges are symmetric +-`--joint-range` degrees by default**,
    except explicit per-joint overrides in `JOINT_RANGE_OVERRIDES_DEG`
    (joint name -> `(lo_deg, hi_deg)`, directly editable -- delete a
    joint's entry to fall back to the symmetric default). Used when a
    specific motor needs asymmetric range (e.g. bench-testing showed one
    direction needed more room). Don't just pick a number for that:
    bisection-search that joint's requested extreme (holding the other 7
    joints at random samples within their own range each step) up to a
    hardware-plausible cap (180deg used so far), checking `ncon==0`, so
    the value is "confirmed self-collision-free up to this cap" rather
    than a guess. Every joint checked so far cleared all the way to
    180deg (geometry was never the binding constraint, the cap was); if a
    future joint's search finds the cap is NOT collision-free, back off
    to whatever bisection converges to instead.

    **Units: the whole file is authored in degrees**
    (`<compiler angle="degree">`), but with one sharp edge worth knowing
    before hand-editing the raw XML: MuJoCo's `angle="degree"` setting
    auto-converts `<joint range>` (and `<joint ref>`) but does **NOT**
    auto-convert `<position ctrlrange>` -- verified empirically (built a
    minimal test file, compared `MjModel.jnt_range` vs
    `MjModel.actuator_ctrlrange` after loading, the former came back
    correctly converted to radians, the latter came back as the raw
    typed-in number, unconverted). So the generator writes `<joint
    range="lo hi">` in plain degrees but writes the corresponding
    `<position ctrlrange="lo hi">` in hand-converted radians
    (`np.radians(...)`) -- same joint, same underlying limit, two
    different units in the two XML attributes. Getting this backwards
    (e.g. writing ctrlrange in raw degree numbers, expecting MuJoCo to
    convert it) doesn't error, it silently produces a control range that
    doesn't match the joint's real range.

## Two things NOT yet handled by the converter (flagged in the output file)

1. **ROS REP-103 frame remap.** The output is in the CAD's own native
   frame (`+y=front, +x=right`), not ROS convention (`+x=forward,
   +y=left, +z=up`). Because step 5's transforms are all relative
   (parent-to-child), this remap -- if/when needed -- is a **single**
   `Rz(-90deg)` rotation applied only to the outermost torso body's own
   `pos`/`quat` in the final MJCF. It does not need to touch anything
   else in the tree. Not yet applied in the current output.

2. **Standing-pose joint zero.** `qpos=0` for every leg joint is
   whatever configuration the CAD assembly happened to be captured in at
   export time (visibly bent/folded, not "legs straight down"). An
   earlier, superseded hand-built version of this file used a convention
   where `qpos=0` == the real robot's standing pose; the converter-built
   file does not have that property. Reconciling this needs either a
   real standing-pose calibration measurement or a geometric
   straight-down-leg IK solve per leg, then baking the result in as a
   `<key>` default pose or a joint-zero shift.

## Why masses need one manual step, not full automation

Everything above (geometry, orientation, joint axes, leg/torso grouping,
inertia) is fully automatic from the URDF alone. Masses are the one
exception: they only exist in `robot.xml`, and `robot.xml`'s own body
names don't line up 1:1 with the URDF's per-leg link names (each
export's own mate-fusion groups CAD parts differently -- e.g. one leg's
thigh body ends up named after a `thigh_connecor_full` part, another
leg's ends up named after a `spacer_between_thigh_connector` part,
purely due to which part happened to become that fused rigid group's
"root" during export). There's no reliable automatic way to attribute
"this specific fused body's mass belongs to leg X's thigh" across the
two formats without either assuming symmetry (fragile on an asymmetric
design) or a fragile mesh-matching heuristic.

Instead: `generate_dog_mjcf.py --print-masses` loads `robot.xml` and
prints every real body's mass. For this design (1 static torso + 4
thighs + 4 calves, symmetric), that's 9 numbers that visibly cluster
into 3 distinct values -- a 10-second manual read, passed back in via
`--mass-torso`/`--mass-thigh`/`--mass-calf`. Safer than a heuristic that
could silently mis-attribute mass on a future, less-symmetric design.

The battery and IMU (added to the CAD in the `robot_dog`/`urdf_dog`
export) don't need their own `--mass-*` flag: both are mate-fixed
directly to the torso in the CAD, so `onshape-to-robot` already folds
their mass into the single "torso" body's own reported mass -- torso
mass jumped from 2.977kg (`robot_full_dog`, no battery/IMU) to 4.726kg
(`robot_dog`, both added), read straight off `--print-masses`, no
separate placeholder or manual battery-mass estimate needed anymore.
Their real mesh position/color come along for free the same way every
other torso-group part does.

## Re-running after a new CAD export

```
# 1. Export the Onshape assembly twice (onshape-to-robot, once per format)
#    into new onshape_folders/robot_half_dog_N and urdf_half_dog_N dirs.

# 2. Find the new masses:
python3 generate_dog_mjcf.py --robot-xml-dir ../onshape_folders/robot_half_dog_N --print-masses

# 3. Generate:
python3 generate_dog_mjcf.py \
    --urdf ../onshape_folders/urdf_half_dog_N/half_dog/urdf/half_dog.urdf \
    --robot-xml-dir ../onshape_folders/robot_half_dog_N \
    --mass-torso <T> --mass-thigh <H> --mass-calf <C> \
    --out dog.mjcf.xml
```

Needs `dog_ros2_ws/.venv` (has `mujoco`, `numpy`, `trimesh`) -- that's
also the venv version-matched to the `python3 -m mujoco.viewer` CLI used
to verify the result interactively.

## Verification checklist (what was actually done, do this again after any change)

1. **Load check**: `mujoco.MjModel.from_xml_path(...)` succeeds, `nu`,
   `nbody`, `nmesh` are the expected counts.
2. **Self-collision sweep**: hundreds of random joint-angle samples
   within the joint range, `mujoco.mj_forward` + check `d.ncon == 0` for
   each. Catches gross interpenetration bugs cheaply.
3. **Offscreen renders** from a few angles (top/front/side/iso), with
   collision-only geoms (`group=3`) hidden via `MjvOption.geomgroup`, to
   visually sanity-check the assembly looks like a plausible robot (not
   scrambled/flipped/floating parts) before asking for human review.
4. **Interactive human confirmation is mandatory, not optional** --
   `python3 -m mujoco.viewer --mjcf=dog.mjcf.xml`. Step 3's offscreen
   renders already passed once on a build that was actually wrong (see
   the orientation-bug section above) -- they're a cheap first filter,
   not a substitute for a human actually looking at the interactive
   viewer.
