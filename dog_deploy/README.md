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
   entirely and just re-commands each motor to hold its current reading —
   in `control_mode='torque'`, "hold" means zero torque, the passive
   no-op, not a position to hold).
4. `control_mode='position'` (default): clamps each motor's requested
   move to `max_delta_deg_per_step`, then sends the result to `actuator`'s
   `set_motor_targets` service. `control_mode='torque'`: clamps to
   `max_torque_nm`/`max_delta_torque_nm_per_step` instead, then sends to
   `actuator`'s `set_motor_torque` service — see
   [Torque-mode deployment](#torque-mode-deployment-2026-08-04) below,
   this is NOT interchangeable with position mode and has different
   safety characteristics.

All service calls are async (`call_async` + done-callbacks) so a slow
response never blocks the executor or re-enters a service call before the
previous one finished (`self.busy` guards that).

### Parameters

| Name                      | Type   | Default | Description |
|----------------------------|--------|---------|--------------|
| `policy_path`              | string | `''`    | Path to a TorchScript `.pt` file (from `dog_gym/export_policy.py`). Required unless `dry_run_hold_pose` is true. |
| `control_rate_hz`          | double | `20.0`  | Policy inference / command rate. |
| `control_mode`              | string | `'position'` | `'position'` (default, unchanged) sends `set_motor_targets`; `'torque'` sends the new `set_motor_torque` instead — see [Torque-mode deployment](#torque-mode-deployment-2026-08-04). **Must match whatever `control_mode` the loaded `policy_path` was actually trained with** (`dog_gym.train`'s `--control-mode`) — nothing here can detect a mismatch (the exported `.pt` has no action-space metadata), it'll just silently send nonsense-scaled commands. |
| `max_delta_deg_per_step`   | double | `5.0`   | `control_mode='position'` only. Safety clamp: max per-motor target movement per control tick. Since 2026-07-27 this slews the target relative to the **previous commanded target** (matching sim's slew limiter), not the measured position — measurement-anchoring fed motor overshoot back into the reference and caused severe stand-up chatter (see daniel_cl_context.md). Note 5°@20Hz = 100°/s = exactly sim's training slew rate. |
| `max_torque_nm`            | double[8] | `[1.0]*8` | `control_mode='torque'` only. **Per-motor** (motor 1..8 order) client-side torque magnitude clamp — deliberately redundant with `actuator`'s own per-motor server-side `max_torque_nm` (neither should be the only thing standing between a bad policy output and full motor torque). Was a single shared double — changed 2026-08-04 after real data (uniform `1.0`) showed the 4 thigh motors pinned at the ceiling 99.7% of a run (genuinely underpowered) while the 4 calf motors swung 150-200+ degrees at that SAME limit (a slipping foot). `dog_gym` sim training itself used `±20 N·m`. Example: `-p max_torque_nm:="[2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0]"` (thighs 2.0, calves 1.0). |
| `max_delta_torque_nm_per_step` | double | `2.0` | `control_mode='torque'` only. Per-tick torque rate clamp, anchored to the previous commanded torque (same reasoning as `max_delta_deg_per_step`). NOT something `dog_gym`'s sim training itself enforces (`control_mode='torque'` has no slew clamp in `DogEnv.step()`) — an extra conservative safety net specific to real hardware, not a sim-fidelity requirement. |
| `max_target_lead_deg`      | double | `10.0`  | Windup guard for the prev-target-anchored clamp above: max degrees the commanded target may lead the measured position. Keeps a jammed motor from winding up a large error and violently catching up on release. |
| `imu_timeout_sec`          | double | `0.5`   | Skip a control step if the latest IMU reading is older than this. |
| `dry_run_hold_pose`        | bool   | `true`  | **Default is safe-by-default.** When true, ignores `policy_path` and just holds current position every tick — exercises the full read/observe/command loop without any policy risk. |
| `motor_mapping_path`       | string | `dog_description/config/motor_mapping.yaml` | Override to point at a test/corrected copy of the mapping (e.g. while a sign issue is under investigation) without touching the shared canonical file. |
| `home_position_deg`        | double[8] | `[]` | Home reference used to make the observation match sim's qpos=0-at-home convention (see "Homing/observation offset" below). Empty (default) auto-captures from the current reading at startup — **robot must already be physically posed at the tucked/home stance** when `policy_node` starts. Provide explicitly to reuse a known-good home without re-posing the robot. |
| `log_csv`                  | string | `''`    | When set, writes one CSV row per motor per control tick (real position/velocity, the sim-convention qpos built into the observation, the policy's raw pre-clamp action, the clamped action, the real command actually sent -- `target_deg` for position mode or `command_torque_nm` for torque mode -- and whether that tick was frozen) to this path — for inspecting a real run after the fact. Flushed every tick so a Ctrl-C won't lose data. |
| `freeze_after_sec`         | double | `0.0`   | 2026-07-29. `0.0` disables (default, unchanged behavior). When `> 0`, once this many seconds of control ticks have elapsed, freezing kicks in — the policy keeps running every tick (still logged) but its output stops being used. **`control_mode='position'`**: the target is snapshotted and held fixed forever after; firmware PD stays fully active on it (does NOT go passive/cut torque). **`control_mode='torque'`**: freezing means going PASSIVE (zero torque) instead — holding a frozen NONZERO torque forever would just keep applying constant, unopposed force, not a stable hold the way a frozen position target is. Added because real-hardware data showed the policy's raw action never actually settles quiet on its own even once standing is visibly complete (still large tick-to-tick swings late in a run, saturating `max_delta_deg_per_step` on most ticks) — a "wait until the policy goes quiet" trigger wouldn't reliably fire, so this is a fixed time instead, tuned by watching when the robot visibly finishes standing. |

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

## Torque-mode deployment (2026-08-04)

`control_mode='torque'` is genuinely different from everything above, not
just an alternate parameter set. Every other command path in this node
(`set_motor_targets`, `adjust_motor_position`, `go_to_pose`) drives a
firmware-side PD loop that **holds a bounded target** — even a bad or
stale command just means the motor sits at (or ramps toward) some
angle. `set_motor_torque` has no PD tracking at all: the number you send
**is** the force applied to the joint, with no firmware-side
`<joint range>`-equivalent hard stop protecting it the way MuJoCo enforces
one in sim. Treat a torque-mode policy with more caution than a
position-mode one, especially the first time:

1. **Dry-run first, same as always** — `dry_run_hold_pose:=true` with
   `control_mode:=torque` sends zero torque every tick (the correct
   torque-mode no-op — there's no "current torque" to hold the way
   position mode holds a current pose). Confirms the read → observe →
   command loop and the `set_motor_torque` service call path work before
   any real force is ever applied.
2. **Start with `max_torque_nm` low, and per-motor** — the default
   (`[1.0]*8` client-side and server-side in `actuator`) is well below
   what `dog_gym` sim training actually used (`±20 N·m`) on purpose. It's
   a per-motor array (motor 1..8 order), not one shared number — real
   data has already shown the 4 thighs need MORE than the 4 calves can
   safely take (thighs pinned at a uniform `1.0` 99.7% of a run,
   genuinely underpowered; calves swung 150-200+ degrees at that same
   limit, a slipping foot). Confirm the robot behaves as expected (legs
   move in the right direction, velocities in the logged CSV stay
   reasonable, nothing grinds against a mechanical stop) before raising
   any entry — both parameters are live (`ros2 param set`), no restart
   needed, so you can raise (or instantly lower) mid-session.
3. **Physically spot the robot** — for at least the first several runs at
   any new `max_torque_nm` level, have a hand ready to catch/support it.
   Unlike position mode's bounded target, a torque policy that's still
   converging can produce genuinely unexpected motion.
4. **Watch for the watchdog firing** — `actuator` logs a warning
   (`torque_timeout_s`) if a motor's torque command goes stale and gets
   force-zeroed; if you see this during normal operation (not a
   deliberate Ctrl-C), something upstream (network, `policy_node` itself)
   is failing to keep up with `control_rate_hz`.

```bash
ros2 run actuator basic_control &
ros2 run dog_imu imu_node &
python3 -m dog_deploy.policy_node --ros-args \
  -p control_mode:=torque \
  -p dry_run_hold_pose:=true
```

Only once that looks right:

```bash
python3 -m dog_deploy.policy_node --ros-args \
  -p control_mode:=torque \
  -p dry_run_hold_pose:=false \
  -p policy_path:=/path/to/torque_policy.pt \
  -p max_torque_nm:="[2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0]" \
  -p max_delta_torque_nm_per_step:=2.0
```

(`max_torque_nm` above: thighs — motors 1,4,5,8 — at `2.0`, calves — motors
2,3,6,7 — at `1.0`, matching `motor_mapping.yaml`'s order. `actuator`'s own
`max_torque_nm` must be raised to match, or its lower per-motor default
will silently clamp these requests down further.)

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
