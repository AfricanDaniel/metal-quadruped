# dog_description

Data-only package: the MuJoCo model of the 8-DOF dog (thigh + calf per leg,
no hip motors) and the canonical motor-to-joint mapping other packages read.
No code here, just files installed to `share/dog_description/`.

```
dog_description/
├── CMakeLists.txt
├── package.xml
├── mjcf/
│   ├── dog.mjcf.xml          # MuJoCo model, loaded by dog_gym -- GENERATED, don't hand-edit
│   ├── generate_dog_mjcf.py  # (re)generates dog.mjcf.xml from onshape_folders/ -- rerun
│   │                          # and rebuild if the CAD or JOINT_RANGE_OVERRIDES_DEG changes
│   └── save_pose.py          # interactively pose the model in the viewer, save qpos to a .txt
├── onshape_folders/
│   ├── robot_dog/            # onshape-to-robot MuJoCo export: real meshes/masses/colors
│   │                          # (its own joint kinematics are unusable -- every joint comes
│   │                          #  back as a `freejoint`, onshape-to-robot free-floats
│   │                          #  disconnected subassemblies it can't resolve into hinges)
│   └── urdf_dog/              # onshape-to-robot URDF export of the SAME assembly: real joint
│                               # kinematics (axes/positions), used for everything robot_dog
│                               # itself can't provide -- see generate_dog_mjcf.py's docstring
│                               # for the full two-export methodology and why it's needed
├── launch/
│   └── dog_view.launch.py    # RViz + joint_state_publisher_gui slider view of the 4-leg URDF
└── config/
    └── motor_mapping.yaml    # motor_id -> {leg, joint, sign}, loaded by dog_gym + dog_deploy
```

See `converter_context.md` (this directory) for the full onshape-to-robot
export methodology and its history, and `daniel_cl_context.md` (workspace
root, gitignored personal working notes) for the day-by-day debugging
log this package's current state is the result of.

## Where this came from

Kinematic structure (joint axes/positions) is auto-detected by graph
traversal of `onshape_folders/urdf_dog`'s joint tree (`detect_legs()` in
`generate_dog_mjcf.py`) — not hand-enumerated, and not derived from any
other reference project. Visual meshes, colors, and real per-body masses
come from `onshape_folders/robot_dog`'s MuJoCo export (its own joint
kinematics are unusable, see above, but its mesh/mass/material data is
real and correct). `generate_dog_mjcf.py` combines the two into
`dog.mjcf.xml`'s final joint tree via forward-kinematics composition —
see that script's own docstring for the full transform.

It's a belt-driven-calf design: both a leg's thigh motor AND its calf
motor mount on the torso/shoulder, not on the swinging thigh link itself
— the calf motor drives its joint through a timing belt + pulley
routed through the thigh, not a direct-drive gearbox on the calf. This
has a real kinematic consequence `dog.mjcf.xml` itself doesn't capture
(the raw MJCF calf hinge is a plain thigh-relative joint, since the
belt/pulley/tendon was never physically modeled) — see `dog_gym/README.md`'s
"Belt/pulley calf decoupling" section for how `DogEnv` compensates for
this in software instead.

Collision physics deliberately still uses simple capsule/box primitives
(not the detailed meshes) for training speed — mesh-mesh collision across
~100 geoms would be considerably slower for RL. Visual geoms are MuJoCo
group 2, collision primitives are group 3; `DogEnv`'s viewer hides group 3
by default so they don't overlap-render.

## Geometry — real measurements, with some approximations still flagged

- Diagonal inertia is a box approximation using each body's real combined
  mesh bounding box (from actual STL vertex data), not the CAD's real
  `fullinertia` tensor.
- Battery and IMU are real CAD parts now (real mass, real mounting
  position/orientation) — both are mate-fixed into the torso's rigid
  group in the CAD, so their mass and geometry fold into the torso body
  automatically via `onshape-to-robot`/`generate_dog_mjcf.py`, no
  placeholder needed.
- Thigh/calf link cross-section (the *collision* capsule's radius —
  visual mesh shape is real) is still a placeholder guess.
- **Joint limits: real for the 4 thighs, still placeholder for the 4
  calves.** Thigh ranges in `generate_dog_mjcf.py`'s
  `JOINT_RANGE_OVERRIDES_DEG` are real hardware mechanical hard-stops
  (bench-measured, sign-corrected, 5% margin) — see that dict's own
  comment for the full derivation and its history, including a real,
  multi-round sign/topology mixup for leg_a/leg_c specifically that took
  a direct forward-kinematics check (sweep qpos, read the resulting foot
  height) to settle definitively (2026-07-26) — earlier fixes based on
  comparing against other sim-only references got this wrong more than
  once. Calf ranges are deliberately a wide `+-360deg` placeholder —
  since the belt-decoupling compensation (see `dog_gym/README.md`), a
  calf's raw `<joint range>` is just headroom for that compensation
  math, not a real limit; the real absolute calf angle limit isn't
  enforced anywhere yet (a deferred TODO, see `daniel_cl_context.md`).

Treat sim results accordingly — geometry, masses, and thigh limits are
now real, but a trained policy still won't sim-to-real transfer
perfectly until the remaining approximations (capsule radius, real calf
limit enforcement) are replaced with real numbers.

## Motor mapping (`config/motor_mapping.yaml`)

Single source of truth for which motor drives which joint, and in which
direction, shared by `dog_gym` (sim) and `dog_deploy` (real hardware) so
the two can never drift out of sync. The convention:

- **Since 2026-07-26, `dog.mjcf.xml`'s raw joint directions match the
  real robot motor-for-motor, and every `sign` is `+1` by
  construction.** The generator (`generate_dog_mjcf.py`'s `AXIS_FLIP`)
  deliberately negates the URDF's auto-detected axis for 6 of the 8
  joints — exactly the ones whose raw sim direction disagreed with the
  real motor's, determined by measuring both sides independently (real:
  the user's bench table of which way each motor moves its link; sim: a
  direct forward-kinematics sweep reading `d(foot_y)/d(theta)`). So
  `python3 -m mujoco.viewer --mjcf=dog.mjcf.xml` now shows real-life
  directions directly.
- Directions are still mirrored left-vs-right — matching the real
  robot's mirror-mounted motors, in sim and reality alike: thigh "away
  from front" is *increasing* theta for the right legs (motors 1, 5)
  but *decreasing* for the left legs (motors 4, 8); calf "towards
  front" is *increasing* for the right legs (motors 2, 6) but
  *decreasing* for the left (motors 3, 7). Don't assume uniformity
  across legs — that assumption is what caused this project's long sign
  saga (see `daniel_cl_context.md`).
- If a future CAD re-export or motor re-mount changes any direction:
  re-measure BOTH sides per motor (real: isolated single-motor delta
  with no policy running; sim: the FK `d(foot_y)/d(theta)` sweep)
  before touching `sign` or `AXIS_FLIP` — never infer from a documented
  convention or from comparing two sim-only references.
- **Calf motors' `sign` is about the real motor's own ABSOLUTE (torso-
  relative) angle**, not a thigh-relative one — see `dog_gym/README.md`'s
  "Belt/pulley calf decoupling" section for why a calf's real and sim
  angle conventions both need to be absolute, not thigh-relative.

Physical corners are confirmed (see `legs:` in `config/motor_mapping.yaml`):

| Internal name | Physical corner | Motors |
|---|---|---|
| `leg_a` | front right | 1, 2 |
| `leg_b` | front left  | 3, 4 |
| `leg_c` | back right  | 5, 6 |
| `leg_d` | back left   | 7, 8 |

## Why `<position>` actuators, not `<motor>`

The real hardware (`actuator/basic_control.cpp`) only exposes position-mode
control (firmware-side kp/kd) via its services — never raw torque. So
`dog.mjcf.xml` uses MuJoCo `<position>` (PD, target-angle) actuators
instead of `<motor>` (torque) actuators, making the RL action space
directly "target angle per motor" — the same quantity the real
`SetMotorTargets` service (see `actuator/README.md`) takes, with no
unit/mode translation needed in `dog_deploy` beyond the `sign` flip above.

The `kp`/`kv` gains on those actuators (`--kp`/`--kv` in
`generate_dog_mjcf.py`, currently 60/4, raised 2026-07-26 from an
original 15/1 after that was found to droop ~40% short of a commanded
target for a fully-extended, gravity-loaded leg) are placeholder sim-side
PD gains, not derived from the real motor's `position_kp`/`position_kd`
(those are rotor-side and GO-M8010-6-firmware-specific) — tune them
independently in sim.

## Usage

```bash
colcon build --packages-select dog_description
```

Other packages locate installed files via
`ament_index_python.packages.get_package_share_directory('dog_description')`.

Sanity-check the model parses and reports 8 actuators:

```bash
python3 -c "
import mujoco
m = mujoco.MjModel.from_xml_path('install/dog_description/share/dog_description/mjcf/dog.mjcf.xml')
print('actuators:', m.nu, 'qpos size:', m.nq)
"
```
