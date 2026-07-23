# metal-quadruped

An 8-DOF quadruped (thigh + calf per leg, no hip motors) driven by 8x
Unitree GO-M8010-6 motors, controlled from a Jetson.

**Goal:** train an RL policy (MuJoCo + Gymnasium + PPO) that makes the
robot walk and run, deploy it on the Jetson, and iterate between
simulation and real hardware until it performs well in both.

## Packages

| Package | Purpose |
|---|---|
| [`actuator`](actuator/README.md) | ROS 2 node driving the 8 real motors over RS485 (services: velocity/position control, homing, batch position targets). |
| [`dog_imu`](dog_imu/README.md) | Driver node for the LSM6DSO32 IMU on the Jetson; publishes `sensor_msgs/Imu`. |
| [`dog_description`](dog_description/README.md) | The MuJoCo model (placeholder geometry — see its README) and the canonical motor-to-joint/sign mapping shared by sim and real code. |
| [`dog_gym`](dog_gym/README.md) | Gymnasium sim environment + PPO training/export pipeline (dev machine only — heavy deps). |
| [`dog_deploy`](dog_deploy/README.md) | Sim-to-real bridge: runs a trained, exported policy against the real robot on the Jetson. |

## Data flow

```
                    ┌─────────────┐
                    │  dog_imu    │  sensor_msgs/Imu
                    │ (real IMU)  │───────────┐
                    └─────────────┘           │
                                               ▼
┌──────────────┐  read_motor_positions  ┌─────────────┐  set_motor_targets  ┌──────────────┐
│  actuator    │◄───────────────────────│ dog_deploy  │────────────────────►│  actuator    │
│ (real motors)│    (position+velocity) │(policy_node)│   (8 target angles) │ (real motors)│
└──────────────┘                        └─────────────┘                    └──────────────┘
                                               ▲
                                     policy trained in
                                               │
                                        ┌─────────────┐
                                        │  dog_gym     │  same observation/action
                                        │ (DogEnv, sim)│  shape as the real loop
                                        └─────────────┘
                                               ▲
                                        ┌─────────────┐
                                        │dog_description│  MJCF model +
                                        │ (shared)      │  motor_mapping.yaml
                                        └─────────────┘
```

`dog_description/config/motor_mapping.yaml` is the single source of truth
for motor-id-to-joint ordering and sign, loaded by both `dog_gym` and
`dog_deploy` — see that package's README for exactly what it encodes.

## Building

```bash
cd dog_ros2_ws
colcon build
source install/setup.bash
```

`dog_gym` needs extra, heavy, non-rosdep dependencies (`mujoco`, `torch`,
`stable_baselines3`) — see `dog_gym/README.md` for a `--system-site-packages`
venv setup that doesn't fight ROS's own Python packages. `dog_deploy` needs
`torch` on the Jetson at inference time only.

## Where this came from

The RL/sim pieces (`dog_description`, `dog_gym`) started from a teammate's
reference project (`shane_ws/Fast-Quadruped-`) — an 8-DOF, thigh+calf,
no-hip MuJoCo cheetah model and a PPO/Stable-Baselines3 training pipeline.
They were rewritten on the modern `mujoco` Python bindings + current
Gymnasium API rather than ported as-is (the reference project used the
deprecated `mujoco_py`). See `dog_gym/README.md` and
`dog_description/README.md` for what changed and why.

## Status / open items

See `daniel_cl_context.md` (gitignored, not shared/committed — personal
working notes) for a running log of design decisions, what's built, and
what's still open (placeholder robot geometry, unconfirmed leg-corner
naming, IMU mounting-orientation calibration, no trained policy yet).
