# dog_gym

MuJoCo/Gymnasium simulation environment and RL training pipeline for the
dog. Runs on a dev machine (GPU recommended for training) — the heavy deps
(`mujoco`, `torch`, `stable_baselines3`) are not meant to run on the Jetson;
see `dog_deploy` for what actually ships to the robot.

Rewritten on the modern `mujoco` Python bindings + current Gymnasium API,
**not** a port of the `mujoco_py`-based training code it's derived from
(`shane_ws/Fast-Quadruped-`) — `mujoco_py` is unmaintained and painful to
install (see that repo's `additional_steps.md` for the GLEW/patchelf/numpy
pinning it required). This package has none of that.

```
dog_gym/
├── package.xml
├── setup.py / setup.cfg
├── requirements.txt
├── resource/dog_gym
└── dog_gym/
    ├── __init__.py       # registers "Dog-Stand-v0" and "Dog-Walk-v0"
    ├── envs/
    │   └── dog_env.py    # DogEnv: the actual sim environment, both tasks
    ├── train.py           # PPO/SAC/A2C training + testing CLI
    └── export_policy.py   # SB3 checkpoint -> TorchScript, for dog_deploy
```

## Setup

```bash
cd dog_ros2_ws
colcon build --packages-select dog_description dog_gym
source install/setup.bash
# --system-site-packages: DogEnv imports ament_index_python (from the ROS
# install) to locate dog_description's share directory, so the venv still
# needs to see it alongside the heavier pip-only deps below.
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r src/dog_gym/requirements.txt
```

**Use `python3 -m dog_gym.<module>`, not `ros2 run dog_gym <module>`, once the
venv is active.** `ros2 run` executes the *installed script file*, whose
shebang line is fixed to whatever Python built it — and `colcon`/`ros2`
themselves always run under system Python (their own shebang), regardless
of whether a venv is active in your shell. So the installed script's
shebang always points at system Python, which doesn't have `mujoco`/
`torch`/`stable_baselines3` — you'd get `ModuleNotFoundError` even with the
venv active and even after rebuilding. `python3 -m dog_gym.train` instead
uses *your shell's* `python3` (the venv's, once activated), which is what
you actually want. Both commands below assume `source install/setup.bash`
**and** `source .venv/bin/activate` are done first, in that order.

## `DogEnv` (`dog_gym/envs/dog_env.py`)

- Loads `dog_description`'s `mjcf/dog.mjcf.xml` via
  `ament_index_python` — **remember the geometry in that model is
  placeholder**, see `dog_description/README.md`.
- **Action**: 8 target joint angles (radians), in `motor_1..motor_8` order
  — directly usable by `dog_deploy` after the `motor_mapping.yaml` sign
  flip, no unit conversion needed (see "Why `<position>` actuators" in
  `dog_description/README.md`).
- **Observation**: deliberately proprioceptive-only, matching what's
  actually available on the real robot — no absolute torso position or
  orientation (there's no motion-capture/localization system). 8 motor
  joint angles + 8 joint velocities (gathered explicitly in `motor_1`..
  `motor_8` order via `motor_mapping.yaml` — raw MuJoCo `qpos`/`qvel` array
  order follows body-tree order, not motor order, so it can't be sliced
  directly) + simulated IMU (`accelerometer`/`gyro` sensordata, matching
  what `dog_imu`'s real driver publishes) + the previous action (so a
  policy can be penalized for jerky control, which matters for the real
  gearboxes). `dog_deploy` builds this exact vector from
  `actuator/read_motor_positions` + `dog_imu` on real hardware. Reward
  computation is still free to use privileged sim-only state (true torso
  height/tilt/velocity) since reward never runs at deployment time.
- **Two tasks, one env class** (`task='stand'`/`'walk'`, registered as
  `Dog-Stand-v0`/`Dog-Walk-v0` in `__init__.py`): stand-up-in-place and
  walk-forward are trained as separate policies rather than one policy
  learning both at once. Both share the observation/action space and
  physics-stepping code; only `reset()`'s initial pose and
  `_compute_reward()` differ per task. See `dog_env.py`'s module
  docstring for the reasoning.
  - `Dog-Stand-v0`: starts from the sitting/home pose (qpos=0). Reward:
    height error to `STAND_HEIGHT_M` (+ a bonus once within
    `STAND_HEIGHT_TOLERANCE_M`) + uprightness + a symmetry penalty across
    matching thigh/calf joints + a stay-in-place penalty. No
    forward-velocity term.
  - `Dog-Walk-v0`: starts already standing (`STANDING_QPOS_DEG`) and
    rewards forward velocity, with height/upright kept only as loose
    regularizers.
  - Both also carry an IMU-based stability penalty (shock/angular-velocity
    spikes) + a small effort/action-rate penalty (discourages jerky,
    hardware-unfriendly control) + a per-step survival bonus. Weights in
    `_compute_reward_stand`/`_compute_reward_walk` are a starting point,
    not tuned.
  - **Belt/pulley calf decoupling (2026-07-25).** Each leg's calf motor
    drives its lower pulley through a timing belt to a torso-mounted
    upper pulley; the lower pulley free-spins on a bearing relative to
    the thigh. Real effect: rotating ONLY the thigh never changes the
    calf's real-world orientation — the belt cancels the "carried along"
    rotation a plain hinge would otherwise apply. `dog.mjcf.xml`'s calf
    joint is a plain thigh-relative hinge (the belt/pulley/tendon itself
    was never physically modeled), so `DogEnv` compensates in software:
    `action`/`obs` for a calf motor represent the real motor's ABSOLUTE
    (torso-relative) angle, matching what `actuator`/`dog_deploy` already
    read/command natively — only the raw MJCF `ctrl`/`qpos` for a calf
    get converted to/from that absolute value (`step()`/`_get_obs()`, via
    `calf_belt_sign` — a per-leg sign, NOT uniform across all 4, derived
    from each leg's own joint-axis geometry, see `dog_env.py`'s
    `__init__`). **Any checkpoint trained before this fix landed is
    stale** — its calf semantics don't match the current env.
  - **Sim directions match the real robot motor-for-motor since
    2026-07-26** (`AXIS_FLIP` in `generate_dog_mjcf.py`, the final
    chapter of this project's long sign saga — see that dict's comment
    and daniel_cl_context.md for the full history of wrong turns that
    preceded it). The generator deliberately negates the URDF's
    auto-detected joint axis for 6 of the 8 joints so that every
    joint's raw sim qpos direction equals the real motor's direction;
    consequently `motor_mapping.yaml`'s sign is `+1` for all 8 motors,
    and the raw `mujoco.viewer` shows real-life directions directly.
    Directions are still mirrored left-vs-right (real motors are
    mirror-mounted; sim now mirrors that faithfully): "extend toward
    standing" is POSITIVE thigh qpos for the right legs (a, c),
    NEGATIVE for the left (b, d) — which is exactly what
    `THIGH_SYMMETRY_SIGN`/`CALF_SYMMETRY_SIGN` (`[1,-1,1,-1]`) correct
    for in the stand task's symmetry reward. Any sign/direction
    question should be settled by direct forward-kinematics (sweep the
    joint, read the foot's world position via `mj_forward`), never by
    trusting a documented convention — that's the method that ended
    the saga.
- **Termination**: torso falls below `FALL_HEIGHT_M` or tips past
  `MAX_TILT_RAD`. **Truncation**: `MAX_EPISODE_STEPS`.
- `domain_randomization=True` randomizes ground friction on every reset
  (ported from the reference repo's domain-randomization script) and, as
  of 2026-07-27, also injects gaussian noise into every observation
  (motor qpos/qvel + IMU accel/gyro, NOT `prev_action`) — see
  `_get_obs()`'s comment and `MOTOR_POS_NOISE_STD_RAD` et al. in
  `dog_env.py` for magnitudes (placeholder sensor-noise scales, refine
  using real `dog_deploy` log_csv data if wanted). Added after real
  deployment showed a policy that looked converged in sim still
  chattering on real hardware — sim's observation had always been
  perfectly clean, so a policy never had reason to learn robustness to
  the small measurement noise real encoders/IMUs actually produce.
  Opt-in, off by default.

## Training

```bash
python3 -m dog_gym.train --train --env-id Dog-Stand-v0 --algo PPO --env-type subproc \
  --num-envs 8 --fname stand_policy

python3 -m dog_gym.train --train --env-id Dog-Walk-v0 --algo PPO --env-type subproc \
  --num-envs 8 --fname walk_policy
```

`--env-id` defaults to `Dog-Stand-v0`.

Saves checkpoints to `models/` and TensorBoard logs to `dogGymTrain_logs/`
(both relative to the current directory — override with `--model-dir`/
`--log-dir`; not named `logs/` — that collides visually with colcon's own
`log/` at the workspace root, an unrelated directory). Runs indefinitely,
saving a new checkpoint every `--timesteps-per-iter` (default 1,000,000)
— stop with Ctrl+C once you're happy with a checkpoint.

Watch training progress:

```bash
tensorboard --logdir dogGymTrain_logs/
```

Fine-tune walk from a good stand checkpoint instead of starting from
random init (2026-07-28) — valid because `Dog-Stand-v0`/`Dog-Walk-v0`
share the exact same observation/action space, only `task`'s reward and
reset differ:

```bash
python3 -m dog_gym.train --train --env-id Dog-Walk-v0 --algo PPO --env-type subproc \
    --num-envs 32 --fname DR_walk_policy_v1 --domain-randomization \
    --init-from models/PPO_32000000_DR_stand_policy_v1.zip
```

`--init-from` loads the checkpoint's policy/value network weights AND
optimizer state via `PPO.load()`, then rebinds to the new env — verified
directly (a loaded model's action on a given observation matches the
original checkpoint's exactly, confirming genuine weight transfer, not
silent reinitialization). `--n-steps`/`--batch-size`/`--n-epochs`/
`--learning-rate`/device still apply and override whatever the checkpoint
saved. **Use a lower `--learning-rate` than the `3e-4` fresh-training
default when fine-tuning** — observed directly (2026-07-28): a
`penaltyFix` stand fine-tune at the default rate got WORSE, not better,
over 19M further steps (raw-action swing at a settled pose went from
74.7deg mean at 1M steps to 148.5deg at 20M, back near the original
pre-fix policy's level) — large gradient steps on an already-near-optimal
network are more likely to destabilize existing behavior than refine it;
try `1e-4` or `5e-5` instead. This run's own timestep counter and
checkpoint filenames restart at 0 regardless of the source checkpoint's
step count, so `DR_walk_policy_v1`'s own
`PPO_1000000_...` means "1M steps of walk fine-tuning," not "33M
cumulative."

Visualize a trained checkpoint (paced to real time, so it's actually
watchable — a falls-quickly policy otherwise finishes an episode in a
fraction of a second):

```bash
python3 -m dog_gym.train --test models/PPO_1000000_stand_policy.zip --env-id Dog-Stand-v0 --episodes 5
```

Without `--domain-randomization`, every episode is bit-for-bit identical
(deterministic policy + deterministic `reset()`, no initial-state noise) —
add the flag for varied episodes:

```bash
python3 -m dog_gym.train --test models/PPO_1000000_stand_policy.zip --env-id Dog-Stand-v0 --episodes 5 --domain-randomization
```

**Spacebar pauses/resumes** in the viewer window (needs the window
focused). Mouse still orbits/pans/zooms as usual either way.

**No pretrained model ships in this package.** The reference repo's
`.PPO` file was trained on a different action/observation space
(`mujoco_py`, torque actuators, no IMU-in-observation) and is not
compatible with `DogEnv`.

## Exporting for deployment

```bash
python3 -m dog_gym.export_policy models/PPO_1000000_dog_policy.zip models/dog_policy.pt --env-id Dog-Stand-v0
```

Produces a TorchScript module `dog_deploy` can load with just `torch` — no
`stable_baselines3`/`gymnasium`/`mujoco` needed on the Jetson.

**Any `.pt` file exported before 2026-07-26 is unsafe — re-export it.**
The traced module used to call `ActorCriticPolicy.forward()` directly,
which returns SB3's raw, UNCLIPPED action (PPO's action head has no
inherent bound). `model.predict()` — what every correctness check in
this project actually uses — applies an extra `np.clip()` to the action
space afterward that `forward()` never did. Real-hardware testing found
the old export outputting raw actions in the thousands of degrees, with
real behavior then dominated entirely by `dog_deploy`'s safety clamp
rather than the policy's real (clipped) intention. Fixed: the traced
module now clips to the action space itself, verified to match
`model.predict(obs, deterministic=True)` exactly.

## Smoke-testing without training

Before spending compute on a real training run, confirm the environment
itself is sound:

```python
import dog_gym
import gymnasium as gym

env = gym.make('Dog-Stand-v0')  # or 'Dog-Walk-v0'
obs, _ = env.reset()
for _ in range(500):
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
print("ok")
```
