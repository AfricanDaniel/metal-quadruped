# actuator

ROS 2 package for driving 8x Unitree GO-M8010-6 motors (one quadruped dog)
over a shared serial bus using the Unitree Actuator SDK.

```
actuator/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── preset_pose.yaml   # named poses (motor_id -> offset from home), read by go_to_pose
├── data/                  # CSV logs written at runtime (see Data logging)
│   ├── position/
│   └── velocity/
├── srv/                   # service definitions
└── src/
    ├── basic_control.cpp  # ROS 2 node — services for all 8 motors
    └── motor_cli.cpp      # plain C++ CLI — no ROS 2, tests one motor
```

Two ways to test a motor, built from this one package:

- **`motor_cli`** — a small, dependency-free (no `rclcpp`) command-line tool.
  Good for bench-testing a single motor by ID before it's wired into the
  full system, or on a machine without ROS 2 sourced.
  ```bash
  ros2 run actuator motor_cli <motor_id> [port]
  # or, without ROS 2 involved at all:
  ./install/actuator/lib/actuator/motor_cli <motor_id> [port]
  ```
- **`basic_control`** — the ROS 2 node, below. Does nothing on startup except
  open the serial port — motors are only commanded once you call a service
  naming a `motor_id`.

## Node: `basic_control`

On startup the node:
1. Opens a serial connection on the configured `port`.
2. Starts a 100 Hz control-loop timer, but with no motors registered yet it's
   a no-op — nothing is sent over the bus.
3. Advertises four services: `set_motor_velocity`, `adjust_motor_position`,
   `read_motor_positions`, and `go_to_pose`.

The first time a service call names a given `motor_id`, that motor is
registered (its `MotorCmd`/`MotorData` state is created) and the control loop
starts sending it commands every cycle. Motors you never call a service for
are never touched.

- `set_motor_velocity` puts a motor into continuous velocity control at the
  requested rad/s, using the `kd_gain` parameter.
- `adjust_motor_position` switches a motor into position control: the first
  call latches the motor's current position, and each call after that adds
  the requested degrees (positive or negative) to a running target, using the
  `position_kp`/`position_kd` parameters. A motor stays in position mode until
  `set_motor_velocity` is called for it again.
- `read_motor_positions` returns the current angle (degrees, output-shaft) of
  motors 1 through 8 in one call, without commanding any motion.
- `set_home` captures the current angle of motors 1–8 as this session's
  home/reference position — see [Homing](#homing--reference-position) below,
  this must be called (with the robot physically posed correctly) before
  `go_to_pose` will do anything.
- `go_to_pose` looks up a named pose in `config/preset_pose.yaml` — each pose
  is a set of **offsets from home** — and drives every motor listed in that
  pose to `home + offset`, from whatever pose the dog currently happens to
  be in.

Per-cycle state (position/velocity/torque/temp/error) is logged at `DEBUG`
level so it doesn't spam the terminal — pass `--ros-args --log-level debug`
to see it.

On shutdown (Ctrl+C), every motor that was ever registered gets zeroed
(kp/kd/tau/velocity = 0) before the node exits, so nothing keeps spinning.

### Units and the gearbox

The GO-M8010-6's SDK commands/reads position and velocity on the **rotor**
side of the gearbox, not the output shaft. This node converts to/from
output-shaft units (the `degrees`/`velocity` you pass in services, and the
`position`/`velocity` logged to CSV) using `queryGearRatio(GO_M8010_6)`
(≈6.33):

- `degrees` (position mode) and `velocity` (rad/s, velocity mode) are
  multiplied by the gear ratio before being sent as `cmd.q`/`cmd.dq`.
- `position_kp`/`position_kd` are divided by `gear_ratio²` before being sent
  as `cmd.kp`/`cmd.kd` — this matches the vendor SDK's position-control
  example and is required for position mode to be stable (skipping it causes
  the controller to massively overreact to position error).
- `kd_gain` (velocity mode) is used **as-is**, matching the vendor SDK's own
  GO_M8010_6 velocity example — velocity mode does not need the `r²`
  conversion the way position mode does.
- Logged/reported position and velocity are divided back down to output-shaft
  units, so a `resulting_position_deg` of `30` really means the output shaft
  moved 30°, not the rotor.

### Homing / reference position

The GO-M8010-6 has no absolute position memory across power cycles — `q` is
a multi-turn count accumulated in firmware since power-on, not a
battery-backed absolute encoder. After a power cycle, "0" (or any other raw
reading) does **not** correspond to the same physical joint angle as before
— especially once you account for the ~6.33:1 gearbox, where the rotor only
needs to complete one full revolution (≈57° of output-shaft travel) to lose
track of which "wrap" it's in.

The fix is to re-establish a known reference by hand every session instead
of trusting raw absolute readings:

1. Power on, then **physically pose the robot** in a repeatable reference
   stance (ideally against something you can register against consistently
   — a hard stop, jig, fixture — free-hand positioning will only be as
   repeatable as your eye).
2. Call `set_home` — do this *before* any other motor command in the
   session, while the motors are still torque-free (a motor is only
   commanded once some service targets it, so a freshly-started node hasn't
   applied any torque yet). This captures each motor's current angle as
   `home_deg_[motor_id]` for this session only — it is **not** persisted to
   disk or across restarts.
3. From then on, `go_to_pose` resolves each pose as `home + offset` (the
   offsets living in `config/preset_pose.yaml`), so poses are correct
   regardless of what the raw encoder reading happened to reset to this
   power cycle.

If you call `go_to_pose` before `set_home` has captured a motor that pose
needs, the service fails safely (`success: false`) without moving anything,
rather than guessing.

Re-homing mid-session (without power-cycling) works the same way, but only
if the motors aren't currently under active position/velocity control —
otherwise they'll resist you trying to pose them by hand.

### Data logging

Every control cycle a motor is active, its state is appended as a CSV row
(`timestamp,motor_id,position,velocity,torque`) to a file under this
package's `data/` directory (`actuator/data`), regardless of where
`ros2 run` is invoked from — the path is baked in at build time:

- While in velocity mode (`set_motor_velocity`): `actuator/data/velocity/motor_<id>_<timestamp>.csv`
- While in position mode (`adjust_motor_position`): `actuator/data/position/motor_<id>_<timestamp>.csv`

A new timestamped file is created each time a motor (re)enters a mode — e.g.
switching a motor from velocity to position and back opens a fresh file for
each session. Directories are created automatically if they don't exist.

> Note: `[WARNING] SerialPort::recv...`/`motor id=X does not reply` lines come
> from the vendored Unitree SDK (a prebuilt `.so`), not from this node's ROS
> logging, so `--log-level` doesn't affect them. They mean the motor isn't
> answering on the bus — check power, port, baud rate, and `motor_id` if you
> see them.

### Parameters

| Name               | Type   | Default        | Description                                              |
|--------------------|--------|----------------|------------------------------------------------------------|
| `port`             | string | `/dev/ttyUSB0` | Serial device shared by all motors on the bus              |
| `kd_gain`          | double | `0.05`         | Velocity tracking gain, applied in velocity mode            |
| `position_kp`      | double | `2.0`          | Position gain, applied in position mode                     |
| `position_kd`      | double | `0.2`          | Damping gain, applied in position mode                      |
| `pose_speed_deg_s` | double | `30.0`         | Default ramp speed (deg/s) for position-mode moves — see [Ramped moves](#ramped-position-moves) |

`ros2 param set /motor_test pose_speed_deg_s 10.0` takes effect on the very
next move — it's read fresh each time, not cached at startup.

### Service: `set_motor_velocity` (`actuator/srv/SetMotorVelocity`)

```
int32 motor_id
float32 velocity
---
bool success
```

Registers `motor_id` if it isn't already active and commands it to spin at
`velocity` rad/s. Calling it again on a motor that's in position mode
switches it back to velocity control.

### Service: `adjust_motor_position` (`actuator/srv/AdjustMotorPosition`)

```
int32 motor_id
float32 degrees
---
bool success
float32 resulting_position_deg
```

- Registers `motor_id` if it isn't already active.
- `degrees` is a **relative** offset: positive increases the target position,
  negative decreases it.
- `resulting_position_deg` is the new cumulative target position after
  applying the offset.

### Service: `read_motor_positions` (`actuator/srv/ReadMotorPositions`)

```
int32[] motor_id
---
int32[] motor_id
float32[] position_deg
float32[] velocity_deg_s
```

An empty request `motor_id` (or omitting it) reads all of motors 1–8;
otherwise only the listed IDs are read (see `{motor_id: [1]}` example
below). Response `motor_id[i]`/`position_deg[i]`/`velocity_deg_s[i]` are
parallel arrays, in the order requested (or 1–8 order if the request was
empty). `velocity_deg_s` is `dog_deploy`'s reason for existing — it's what
lets a real-hardware observation match the joint-velocity half of
`dog_gym`'s training observation. Motors already active
(velocity or position mode) report their latest reading from the 100 Hz
control loop; a motor seen for the first time gets registered with a
zero-effort probe read (kp/kd/tau/velocity all 0, so it can't move) purely to
fetch its real position — after that it stays registered and gets polled
every cycle like any other active motor, even though nothing is commanding
it to move.

The `ros2 service call` response itself is a flat array (not easy to read at
a glance), so the node also logs a human-readable, one-motor-per-line summary
in the terminal running `basic_control` every time this service is called:

```
[INFO] [motor_test]: Motor positions:
[INFO] [motor_test]:   Motor 1: 42.25 deg
[INFO] [motor_test]:   Motor 2: 52.11 deg
[INFO] [motor_test]:   Motor 3: 22.20 deg
[INFO] [motor_test]:   Motor 4: 48.50 deg
[INFO] [motor_test]:   Motor 5: 48.65 deg
[INFO] [motor_test]:   Motor 6: 16.61 deg
[INFO] [motor_test]:   Motor 7: 22.89 deg
[INFO] [motor_test]:   Motor 8: 54.81 deg
```

### Ramped position moves

`adjust_motor_position` and `go_to_pose` don't step straight to the target —
`control_loop` linearly ramps the commanded position from wherever the motor
currently is to the target over `distance / speed` seconds, where `speed` is
`pose_speed_deg_s` by default (`go_to_pose` can override it per-call). Once
the ramp finishes, the motor holds at the target as normal. Calling either
service again mid-ramp retargets smoothly from the motor's current position,
it doesn't jump back to the old target first.

### Service: `go_to_pose` (`actuator/srv/GoToPose`)

```
string pose_name
float32 speed_deg_s
---
bool success
string message
```

- `speed_deg_s` is optional — `0` (or omitting it) uses the node's
  `pose_speed_deg_s` parameter; a positive value overrides it for this call
  only, per motor (all motors in the pose move at the same deg/s, so ones
  with farther to travel take longer to arrive).
- Looks up `pose_name` under `poses:` in `config/preset_pose.yaml`, e.g.:
  ```yaml
  poses:
    standing:
      1: 107.41
      2: 5.91
      3: 0.49
      4: -107.62
      5: 113.66
      6: 4.68
      7: -1.14
      8: -107.25
  ```
  Each key under a pose is a `motor_id`, each value is its target angle in
  degrees **relative to that motor's home position** (see
  [Homing](#homing--reference-position)) — not an absolute reading.
- Requires `set_home` to have been called this session for every motor the
  pose touches; otherwise fails safely (`success: false`, `message` explains
  which motor is missing a home) without moving anything.
- The file is **re-read on every call** — add or edit poses in the YAML and
  call the service again, no rebuild or node restart needed.
- Only the motors listed in the pose are commanded; motors omitted from a
  pose are left alone (useful if you want a pose that only touches, say, the
  front legs).
- Also fails safely if the pose name isn't found or the YAML can't be
  parsed.

### Service: `set_home` (`actuator/srv/SetHome`)

```
---
bool success
int32[] motor_id
float32[] home_deg
```

Captures the current angle of motors 1–8 as this session's home/reference
position (used by `go_to_pose` to resolve pose offsets — see
[Homing](#homing--reference-position)). Call this only after physically
posing the robot in its reference stance. Not persisted — a node restart
clears it and `set_home` must be called again.

### Service: `set_motor_targets` (`actuator/srv/SetMotorTargets`)

```
int32[] motor_id
float32[] position_deg
---
bool success
```

Batch, absolute, **immediate** position control: sets each listed motor's
target to `position_deg` (output-shaft degrees) with no ramp — unlike
`adjust_motor_position` (relative, ramped) and `go_to_pose` (ramped,
YAML-driven), this jumps straight to the target on the very next control
cycle. Intended for a caller that already commands a full trajectory at a
fixed rate itself (e.g. an RL policy in `dog_deploy`), where ramping inside
this node would just fight the caller's own timing.

`motor_id` and `position_deg` must be the same length (parallel arrays,
like `read_motor_positions`'s response) or the call fails
(`success: false`) without moving anything. Registers any motor_id not
already active, same as the other services.

```bash
ros2 service call /set_motor_targets actuator/srv/SetMotorTargets \
  "{motor_id: [1, 2], position_deg: [10.0, -5.0]}"
```

## Building

```bash
colcon build --packages-select actuator
source install/setup.bash
```

## Usage

Start the node — it stays idle until a service is called:

```bash
ros2 run actuator basic_control
# [INFO] Actuator node ready on port /dev/ttyUSB0. No motors active — call
#        set_motor_velocity or adjust_motor_position to start one.
```

Run against a different serial port:

```bash
ros2 run actuator basic_control --ros-args -p port:=/dev/ttyUSB1
```

### Spin motor 5 at 1 rad/s

```bash
`ros2 service call /set_motor_velocity actuator/srv/SetMotorVelocity \
  "{motor_id: 5, velocity: 1.0}"`
```

Stop it by commanding zero velocity:

```bash
ros2 service call /set_motor_velocity actuator/srv/SetMotorVelocity \
  "{motor_id: 5, velocity: 0.0}"
```

### Nudge motor 5 by degrees

Move motor 5 forward 30 degrees (registers it in position mode if it wasn't
already active):

```bash
```ros2 service call /adjust_motor_position actuator/srv/AdjustMotorPosition \
  "{motor_id: 5, degrees: 30.0}"```
```

Move it back 10 degrees (net position is now +20 degrees from where it
started):

```bash
ros2 service call /adjust_motor_position actuator/srv/AdjustMotorPosition \
  "{motor_id: 5, degrees: -10.0}"
```

### Read motor angles 1-8

```bash
ros2 service call /read_motor_positions actuator/srv/ReadMotorPositions "{}"
```

### Go to the standing pose

Physically pose the robot in the home stance first, then capture it:

```bash
ros2 service call /set_home actuator/srv/SetHome "{}"
```

Then, at the default speed (`pose_speed_deg_s`, 30 deg/s unless changed):

```bash
ros2 service call /go_to_pose actuator/srv/GoToPose "{pose_name: standing}"
```

Slower, just for this call (10 deg/s):

```bash
ros2 service call /go_to_pose actuator/srv/GoToPose \
  "{pose_name: standing, speed_deg_s: 10.0}"
```

Return to the home/reference stance from wherever the dog currently is —
`home` is just the all-zero-offset pose (offset `0` from home is home), so
it works out of the box once `set_home` has been called:

```bash
ros2 service call /go_to_pose actuator/srv/GoToPose "{pose_name: home}"
```

Or change the default for every future move (this node instance) instead of
passing `speed_deg_s` every time:

```bash
ros2 param set /motor_test pose_speed_deg_s 10.0
```

### Multiple motors on the same bus

Since `motor_id` is per-call, one node instance can drive several motors on
the same serial bus independently:

```bash
ros2 service call /set_motor_velocity actuator/srv/SetMotorVelocity \
  "{motor_id: 3, velocity: 0.5}"
ros2 service call /adjust_motor_position actuator/srv/AdjustMotorPosition \
  "{motor_id: 7, degrees: 45.0}"
```
