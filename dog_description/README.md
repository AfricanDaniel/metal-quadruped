# dog_description

Data-only package: the MuJoCo model of the 8-DOF dog (thigh + calf per leg,
no hip motors) and the canonical motor-to-joint mapping other packages read.
No code here, just files installed to `share/dog_description/`.

```
dog_description/
├── CMakeLists.txt
├── package.xml
├── mjcf/
│   ├── dog.mjcf.xml          # MuJoCo model, loaded by dog_gym
│   └── robot_onshape/        # real Onshape CAD export -- source of the
│       ├── robot.xml         # visual meshes + real masses in dog.mjcf.xml.
│       ├── assets/*.stl      # NOT itself usable as a sim model (see below)
│       ├── generate_dog_mjcf.py  # regenerates dog.mjcf.xml's mesh/mass
│       │                          # sections from robot.xml -- rerun and
│       │                          # copy its output in if the CAD changes
│       └── view_labeled.py   # standalone viewer for inspecting robot.xml,
│                               # with body/geom labels and part isolation
└── config/
    └── motor_mapping.yaml    # motor_id -> {leg, joint, sign}, loaded by dog_gym + dog_deploy
```

## Where this came from

Kinematic structure (joint locations/axes, the standing-pose-as-qpos-zero
convention, joint ranges) started from a teammate's (Shane's) reference
project at `shane_ws/Fast-Quadruped-` — the closest existing topology
match (thigh + calf per leg, no hip) — then was rebuilt from real
measurements of this robot (see `claude_context.md` at the workspace root
for the full derivation: thigh/calf/bracket lengths, leg spacing, and how
the standing pose — thigh and calf both pointing straight down — became
qpos=0, so the feet land exactly on the floor at reset).

Visual appearance and masses come from a real Onshape CAD export
(`mjcf/robot_onshape/`). **That raw export is not directly usable as a sim
model** — the Onshape assembly has no mates set up as proper joints yet
(`onshape-to-robot` found 0 degrees of freedom: one fixed body + 22
disconnected free-floating parts, no actuators). `generate_dog_mjcf.py`
extracts the real meshes/masses from it and repositions them into
`dog.mjcf.xml`'s existing jointed hierarchy — see that script's docstring
for the full transform (axis remap between Onshape's frame and this
file's, per-body mass grouping, etc.) and `claude_context.md`'s "Onshape
CAD export" section for how each CAD part's real-world role was identified
(it's a belt-driven-calf design — both the thigh and calf-drive motors
mount on the torso, not the swinging links, which is why the torso is
~9kg and the legs are only a few hundred grams each).

Collision physics deliberately still uses simple capsule/box primitives
(not the detailed meshes) for training speed — mesh-mesh collision across
~100 geoms would be considerably slower for RL. Visual geoms are MuJoCo
group 2, collision primitives are group 3; `DogEnv`'s viewer hides group 3
by default so they don't overlap-render.

## Geometry — real measurements, with some approximations still flagged

**Search for `APPROXIMATION`/`PLACEHOLDER` comments in `dog.mjcf.xml`**
for what's still not a direct measurement:
- Diagonal inertia is a box approximation using each body's real combined
  mesh bounding box (from actual STL vertex data), not the CAD's real
  `fullinertia` tensor.
- Battery and IMU are real CAD parts now (real mass, real mounting
  position/orientation) — both are mate-fixed into the torso's rigid
  group in the CAD, so their mass and geometry fold into the torso body
  automatically via `onshape-to-robot`/`generate_dog_mjcf.py`, no
  placeholder needed.
- Thigh/calf link cross-section (the *collision* capsule's radius —
  visual mesh shape is real) and joint hard-stop limits are still
  placeholder guesses (no real data yet).

Treat sim results accordingly — geometry, masses, and the standing pose
are now real, but a trained policy still won't sim-to-real transfer
perfectly until the remaining approximations are replaced with real
numbers.

## Motor mapping (`config/motor_mapping.yaml`)

Single source of truth for which motor drives which joint, and in which
direction, shared by `dog_gym` (sim) and `dog_deploy` (real hardware) so
the two can never drift out of sync. The convention:

- In `dog.mjcf.xml`, every **thigh** joint's positive direction means "away
  from the front", and every **calf** joint's positive direction means
  "towards the front" — the same physical meaning for all four legs.
- Each motor's `sign` in the YAML says whether the real motor's "increase
  commanded degrees" direction agrees with that sim convention (`1`) or is
  inverted (`-1`). This was determined by bench-testing each motor by hand;
  see `claude_context.md` at the workspace root for the raw notes it was
  transcribed from.

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

The `kp`/`kv` gains on those actuators are placeholder sim-side PD gains,
not derived from the real motor's `position_kp`/`position_kd` (those are
rotor-side and GO-M8010-6-firmware-specific) — tune them independently in
sim.

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
