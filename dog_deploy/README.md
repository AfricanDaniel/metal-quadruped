# dog_deploy

The sim-to-real bridge: runs a `dog_gym`-trained, TorchScript-exported
policy against the real robot. Runs on the Jetson, needs only `torch` at
inference time (not `stable_baselines3`/`gymnasium`/`mujoco`).

```
dog_deploy/
├── package.xml
├── setup.py / setup.cfg
├── resource/dog_deploy
└── dog_deploy/
    └── policy_node.py
```

## Node: `policy_node`

Every control tick:

1. Calls `actuator`'s `read_motor_positions` for motors 1–8 (position +
   velocity).
2. Combines that with the latest `dog_imu` reading to build **the exact
   same observation vector** `dog_gym`'s `DogEnv` trains on — 8 motor
   qpos + 8 motor qvel + IMU accel/gyro + previous action, in `motor_1`..
   `motor_8` order (see `dog_gym/envs/dog_env.py`'s docstring) — applying
   the per-motor `sign` from `dog_description/config/motor_mapping.yaml`
   to translate between the real motor's degrees and the sim's radians
   convention.
3. Runs the policy (or, in `dry_run_hold_pose` mode, skips the policy
   entirely and just re-commands each motor to hold its current reading).
4. Clamps each motor's requested move to `max_delta_deg_per_step`.
5. Sends the result to `actuator`'s `set_motor_targets` service.

All service calls are async (`call_async` + done-callbacks) so a slow
response never blocks the executor or re-enters a service call before the
previous one finished (`self.busy` guards that).

### Parameters

| Name                      | Type   | Default | Description |
|----------------------------|--------|---------|--------------|
| `policy_path`              | string | `''`    | Path to a TorchScript `.pt` file (from `dog_gym/export_policy.py`). Required unless `dry_run_hold_pose` is true. |
| `control_rate_hz`          | double | `20.0`  | Policy inference / command rate. |
| `max_delta_deg_per_step`   | double | `5.0`   | Safety clamp: max per-motor movement per control tick. |
| `imu_timeout_sec`          | double | `0.5`   | Skip a control step if the latest IMU reading is older than this. |
| `dry_run_hold_pose`        | bool   | `true`  | **Default is safe-by-default.** When true, ignores `policy_path` and just holds current position every tick — exercises the full read/observe/command loop without any policy risk. |

## Running with `torch` in a venv

If `torch` is installed in a venv on the Jetson (rather than system-wide),
use `python3 -m dog_deploy.policy_node`, **not** `ros2 run dog_deploy
policy_node`, once that venv is active — `ros2 run` executes the installed
script by its baked-in shebang (fixed to whatever Python built it, which is
system Python regardless of an active venv, since `colcon`/`ros2`
themselves always run under system Python), so it won't see the venv's
`torch` even after rebuilding. `python3 -m` uses your shell's active
`python3` instead. `--ros-args -p key:=value` works identically either way.
`actuator`/`dog_imu` don't have this problem (no extra pip deps), so
`ros2 run` is fine for those.

## Before running against real hardware

**Always dry-run first.** With the defaults (`dry_run_hold_pose:=true`),
this node reads the real motors, builds an observation, and re-commands
each motor to stay exactly where it already is — confirms the whole
pipeline (IMU, `actuator` services, timing) works safely before a trained
policy ever touches the hardware:

```bash
ros2 run actuator basic_control &
ros2 run dog_imu imu_node &
python3 -m dog_deploy.policy_node
```

Only once that looks right, run with an actual policy:

```bash
python3 -m dog_deploy.policy_node --ros-args \
  -p dry_run_hold_pose:=false \
  -p policy_path:=/path/to/dog_policy.pt \
  -p max_delta_deg_per_step:=2.0
```

Start `max_delta_deg_per_step` small (a couple of degrees) for the first
real run with any new/undertrained policy and increase it once you trust
the policy's behavior.

## Open calibration TODOs (do not skip)

- **Homing alignment.** `actuator`'s `position_deg` is relative to
  whatever `set_home` last captured (see `actuator/README.md`'s Homing
  section) — it is **not** aligned to `dog.mjcf.xml`'s zero-joint-angle
  pose by construction. For the sign-flip math above to be meaningful,
  physically pose the robot in the same neutral stance the MJCF's default
  pose represents, then call `set_home`, **before** starting
  `policy_node`. Until this is verified on hardware, treat any policy
  behavior as offset by an unknown amount per joint.
- **IMU mounting orientation.** `policy_node` assumes the real IMU's axes
  (as mounted on the Jetson) line up with `dog.mjcf.xml`'s `imu_site`
  axes. If the physical mount is rotated relative to the robot's
  forward/up axes, the accelerometer/gyro readings need a fixed rotation
  applied before use — not yet implemented here.
