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
| `max_delta_deg_per_step`   | double | `5.0`   | Safety clamp: max per-motor target movement per control tick. Since 2026-07-27 this slews the target relative to the **previous commanded target** (matching sim's slew limiter), not the measured position — measurement-anchoring fed motor overshoot back into the reference and caused severe stand-up chatter (see daniel_cl_context.md). Note 5°@20Hz = 100°/s = exactly sim's training slew rate. |
| `max_target_lead_deg`      | double | `10.0`  | Windup guard for the prev-target-anchored clamp above: max degrees the commanded target may lead the measured position. Keeps a jammed motor from winding up a large error and violently catching up on release. |
| `imu_timeout_sec`          | double | `0.5`   | Skip a control step if the latest IMU reading is older than this. |
| `dry_run_hold_pose`        | bool   | `true`  | **Default is safe-by-default.** When true, ignores `policy_path` and just holds current position every tick — exercises the full read/observe/command loop without any policy risk. |
| `motor_mapping_path`       | string | `dog_description/config/motor_mapping.yaml` | Override to point at a test/corrected copy of the mapping (e.g. while a sign issue is under investigation) without touching the shared canonical file. |
| `home_position_deg`        | double[8] | `[]` | Home reference used to make the observation match sim's qpos=0-at-home convention (see "Homing/observation offset" below). Empty (default) auto-captures from the current reading at startup — **robot must already be physically posed at the tucked/home stance** when `policy_node` starts. Provide explicitly to reuse a known-good home without re-posing the robot. |
| `log_csv`                  | string | `''`    | When set, writes one CSV row per motor per control tick (real position/velocity, the sim-convention qpos built into the observation, the policy's raw pre-clamp action, the clamped action, the real degrees actually sent, and whether that tick was frozen) to this path — for inspecting a real run after the fact. Flushed every tick so a Ctrl-C won't lose data. |
| `freeze_after_sec`         | double | `0.0`   | 2026-07-29. `0.0` disables (default, unchanged behavior). When `> 0`, once this many seconds of control ticks have elapsed, the target is snapshotted and held **fixed forever after** — the policy keeps running every tick (still logged) but its output stops being used. Firmware position-mode PD stays fully active on the frozen target (does NOT go passive/cut torque). Added because real-hardware data showed the policy's raw action never actually settles quiet on its own even once standing is visibly complete (still large tick-to-tick swings late in a run, saturating `max_delta_deg_per_step` on most ticks) — a "wait until the policy goes quiet" trigger wouldn't reliably fire, so this is a fixed time instead, tuned by watching when the robot visibly finishes standing. |

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

**Sliding calf range — DONE (2026-07-27).** Each calf motor already
reports its real ABSOLUTE angle directly (the belt does the decoupling
in hardware, no compensation needed here) — but the calf's real physical
limit (calf link hitting the thigh link) is fixed in a RELATIVE
coordinate and slides in absolute terms as the thigh moves. `policy_node`
now enforces this explicitly, using the live thigh reading, in
`_on_positions_read()` — see `CALF_RANGE_DEG`'s comment (values MUST
match `generate_dog_mjcf.py`'s `JOINT_RANGE_OVERRIDES_DEG` calf entries)
and `daniel_cl_context.md`'s TODO 13 for the full measurement. Verified
against a synthetic test matching a case sim itself hit: an out-of-range
request gets clamped identically to how MuJoCo's own `<joint range>`
clamps it in sim.

**Homing/observation offset — DONE (2026-07-26).** `policy_node` used to
build `motor_qpos_rad` directly from `read_motor_positions`'s raw
`position_deg` (`sign * position_deg * DEG_TO_RAD`). Checked directly
against `actuator/src/basic_control.cpp`: `read_motor_positions` returns
the motor's **raw absolute** angle (computed straight from
`motor.data.q`) and never subtracts `home_deg_` — only `go_to_pose` uses
that. `DogEnv.reset()`'s 'stand' task always starts from exactly
`qpos=0` in sim; on real hardware the raw absolute reading at that same
physical tucked pose is some arbitrary nonzero value per motor (e.g.
~47deg for motor 1, ~51deg for motor 5, confirmed via
`stand_policy_v1_fixed.csv`'s tick-0 `sim_qpos_rad` column). Every real
run before this fix fed the policy an observation offset from what it
was trained on, for every motor, on every tick — not just at reset.
Fixed: `policy_node` now captures a home reference once at startup (the
new `home_position_deg` param, or auto-captured from the current reading
if that param is left empty — **the robot must be physically posed at
the tucked/home stance at that moment**) and subtracts it before
building the observation / adds it back when converting an action to a
real target. Verified algebraically: at home, the built observation is
now exactly `0.0` for every motor, matching sim's own reset state; the
whole read→observe→command round-trip is confirmed to reconstruct the
exact original real position in dry-run mode, for either sign.

**IMU mounting orientation — DONE (2026-07-25).** The real IMU is
mounted in the same physical location on the torso as modeled in the
CAD, and axis calibration was carried out directly on the real robot
(tilt-test procedure, see `dog_imu/calibrate_imu_node.py` and
`dog_imu/config/imu_calibration.yaml`). Result: identity mapping — no
axis flips or rotation needed between the real sensor and
`dog.mjcf.xml`'s `imu_site` convention. `policy_node` subscribes to the
calibrated `imu/data` topic (not raw `imu/data_raw`), so this is already
applied at runtime, not an outstanding gap.

## `export_policy.py`'s clipping bug (fixed 2026-07-26)

Every `.pt` file exported before 2026-07-26 is unsafe to run — the
exported `DeterministicPolicy.forward()` called
`ActorCriticPolicy.forward()` directly, which returns SB3's **raw,
unclipped** Gaussian-mean action. `BasePolicy.predict()` (what
`dog_gym.train`'s `--test` and every other correctness check in this
project actually calls) applies an additional
`np.clip(actions, action_space.low, action_space.high)` afterward that
`forward()` never does. Real-hardware testing found the old exported
`.pt` outputting raw actions in the thousands of degrees (confirmed:
`policy_run.csv`, pre-fix) — real behavior was then dominated entirely by
`max_delta_deg_per_step`'s safety clamp rather than the policy's actual
(clipped) intention. `export_policy.py` now applies the same clip inside
the traced module; verified the fixed export's output matches
`model.predict(obs, deterministic=True)` exactly, motor-by-motor, at the
home state. **Re-export and redeploy any `.pt` file that predates this
fix** — its outputs were never representative of the trained policy.
