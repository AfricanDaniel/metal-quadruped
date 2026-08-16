# dashboard

A local web control panel for the day-to-day workflow this project
settled into: find a checkpoint, look at it in MuJoCo, export it, and/or
deploy it to the Jetson -- without hand-typing `python3 -m dog_gym.train
--test ...` / `python3 -m dog_gym.export_policy ...` / `ssh jetson ...` /
`ros2 service call ...` every time.

Three tabs: **Local** (browse `models/`, launch the MuJoCo viewer, export
a policy), **Jetson** (build, browse deployed `.pt` policies, deploy/stop,
basic motor control), **Sheep** (browse the training machine's `models/`,
download checkpoints, or view-training them directly).

It's a thin UI over the exact same CLIs/services already used manually
throughout this project -- it does not reimplement training, export, or
rollout logic.

## Running it

Either of these works identically:

```bash
python3 -m dashboard                  # dev-mode, matches this project's
                                       # own python3 -m dog_gym.train /
                                       # python3 -m dog_deploy.policy_node
                                       # convention
ros2 run dashboard dashboard          # after colcon build + source install/setup.bash
```

Opens a browser tab to `http://127.0.0.1:5055/` (override with the
`DASHBOARD_PORT` env var).

## Layout

```
dashboard/
├── app.py            # Flask routes -- thin, wires everything else together
├── config.py          # paths + remote host constants (jetson/sheep)
├── local_fs.py         # models/ browsing + the PPO_<steps>_<fname> parsing
│                        # rules, reused by remote_fs.py for both remote hosts
├── remote_fs.py         # same browsing, over SSH (sheep's models/, Jetson's
│                        # policies_{position,torque}/{stand,walk}/policy/)
├── ssh.py             # subprocess wrappers: connect test, run, run_background
│                        # (captures a remote PID via `echo $!` so Stop can
│                        # `kill` the real remote process), scp_download, tail_log
├── procs.py           # in-memory registry: connection flags + the Jetson
│                        # hardware-bringup/deploy/build PIDs -- status is
│                        # always re-checked live (ssh.is_running), never
│                        # trusted as stale local state
├── policy_actions.py   # dog_gym.train --test / export_policy.py subprocess
│                        # construction, incl. the control-mode/env-id ->
│                        # policies_{position,torque}/{stand,walk}/policy/
│                        # path mapping
├── ros_actions.py      # the actuator ros2 service calls used by Jetson's
│                        # "basics" section
└── templates/, static/
```

## Known caveats

- **`reset_motors` is a proposed mechanism, not a battle-tested one.**
  Nothing like it existed before this dashboard. It kills and restarts
  just the `actuator` (`basic_control`) process (leaving `dog_imu`
  running) on the theory that an unresponsive motor is almost always
  that node's own RS485/serial connection wedged, not the motor itself.
  Validate this against real hardware before relying on it.
- Every Jetson action other than plain browsing sends a real command to
  the physical robot (motor moves, policy deployment, hardware bringup).
  There are no extra confirmation dialogs beyond the button itself --
  intentional, matching the explicit button-by-button design this was
  built from, but worth knowing before clicking around.
- `dry_run_hold_pose` defaults to **unchecked** in the deploy form (the
  dashboard's own default), which is the opposite of `policy_node`'s own
  parameter default (`true`, safe-by-default). Deliberate per the
  original request -- double check this is what you want before hitting
  Deploy.
- Jetson-specific flows (build streaming, deploy start/stop, basics
  actions) are implemented and template/route-verified, but not yet
  exercised against live hardware in this session (the Jetson was
  unreachable throughout development/testing) -- the connect-failure
  path itself IS verified (confirmed against the actual unreachable
  Jetson). Test the rest for real before trusting it unattended.
