# Position-control RL policy never learns to reciprocate (swing legs back and forth) — torque control does

## Summary

Training a quadruped locomotion policy in MuJoCo + Gymnasium + Stable-Baselines3 PPO, with two
available action spaces for the same robot/task: **position control** (action = target joint
angle, tracked by a `<position>` PD actuator) and **torque control** (action = raw joint torque,
applied via a `<motor>` actuator). Both use the same reward function, same network architecture,
same PPO hyperparameters.

- **Torque control** learns a genuine walking gait: legs swing forward, then reverse and swing
  back, in a repeating cycle, producing real forward locomotion.
- **Position control** never does this. Every training run (multiple random seeds, multiple
  reward-shaping iterations, an imitation-learning warm-start from the working torque policy —
  see below) converges to the same failure mode: one or more leg joints monotonically ramp their
  target angle toward the edge of the allowed action range and just **hold there**, never
  reversing. The robot's forward motion in this state comes from the torso slowly tipping/leaning
  forward over the extended leg — not from a walking gait — and it eventually falls or is
  terminated.

We want to know: **is this a known failure mode in position/PD-servo-controlled legged RL, and
are there established techniques for avoiding it** — beyond what's listed under "what we've tried"
below?

## Environment details relevant to the question

- MuJoCo physics, 100 Hz control (`0.01s` timestep, no frame-skip).
- **Position mode**: action is a *residual* target angle (±75° from a fixed reference pose),
  converted to an absolute target, then rate-limited by a slew clamp (max ~1°/tick, matching a
  real-hardware safety limit of 100°/s) before being written to a `<position>` actuator with
  `kp`/`kd` gains. `kp`/`kd` were deliberately softened for training (from a real-hardware-matching
  `kp=60` down to `kp=20`) after finding that `kp=60` made PPO's exploration noise translate into
  violent, destabilizing snapping — training never progressed past falling immediately at the
  stiffer gain.
- **Torque mode**: action is raw torque (±20 N·m), applied directly, plus a small fixed
  velocity-damping term (`-kd_gain * qvel`) matching a real breakaway-protection feature on the
  actual hardware.
- Reward: `forward_velocity_reward` (dominant term, weight 5.0) + height/upright terms + several
  gait-quality terms (trot-symmetry, foot-clearance, foot-slip penalty, etc.) + effort/action-rate
  penalties.
- The **calf joint** (only) has a "belt/pulley compensation" step: the real robot's calf motor is
  coupled to the thigh via a mechanical belt drive, so the raw MuJoCo joint angle (thigh-relative)
  is converted to an absolute (world-relative) angle before being used as the action/observation
  for that joint, via `absolute_calf = raw_calf_joint + sign * thigh_joint`. In torque mode there's
  an additional variant of this (`torque_belt`) that implements the coupling as a real MuJoCo
  `<fixed>` tendon instead of a Python-side arithmetic conversion. **The thigh joints have no belt
  compensation applied to them at all, in any mode.**

## A hypothesis we considered and ruled out: is this caused by the belt/pulley mechanism?

Our first guess was that the belt-compensation logic (see above) — which isn't represented as a
real kinematic constraint in the MJCF/URDF for plain position mode, only as a runtime angle
conversion — might be responsible for the non-reversing behavior, e.g. by distorting the calf's
effective action space or introducing some asymmetry that discourages reversal.

**This does not hold up.** The non-reversing, monotonic-drift-to-the-limit behavior is observed on
the **thigh** joints, not just the calf — and thighs never touch any belt-compensation code path,
in any control mode. We isolated this directly: commanding a single thigh's action to its maximum
value and holding it constant (nothing else) reproduces the exact same "smooth monotonic ramp to
the limit, then holds" behavior under position control, with no calf or belt logic involved at
all. So whatever is causing this, it is not specific to the belt/pulley mechanism or its
(currently informal, non-URDF) representation.

## What we've tried (results, not just ideas)

1. **Reward shaping** to penalize the specific exploit pattern (a leg holding one contact phase —
   swinging or planted — for an implausibly long time relative to a real gait cycle; an unbounded
   "reward more angular velocity" term that a runaway one-directional rotation could exploit
   better than genuine bounded reciprocation; zero penalty for knee/shin ground contact).
   **Result**: confirmed via direct measurement that these fixes correctly identify and suppress
   the exploit (reward for the pattern dropped from ~1.4–1.6/tick to ~0.6/tick; the specific
   "reward the pattern as a perfect gait" loophole is closed). **The underlying leg behavior did
   not change** — it's now correctly scored as bad, but the policy hasn't found anything better.

2. **Early episode termination** for the same failure signature (a leg stuck in one phase too
   long, sustained torso pitch divergence, no net forward progress after a grace period), meant to
   stop wasting rollout time on doomed episodes. **Result**: had to be tuned carefully — an initial
   aggressive threshold caused every episode to terminate at the exact same early tick regardless
   of what the policy did, which (we believe) prevented random exploration from ever running long
   enough to attempt a leg-lift at all. Loosened significantly; confirmed episodes now get enough
   runway, but the underlying pattern still emerges within that runway.

3. **Softening `kp`/`kd`** (see above) — necessary just to get *any* forward exploration instead of
   immediate falling, but is also suspected to be part of why reversal never gets discovered (see
   root-cause discussion below).

4. **Checked whether the position action-space's rate limiter (slew clamp) is what's making
   monotonic drift "safe."** Directly tested: removed the slew clamp entirely (effectively
   infinite allowed rate). **Result: the robot still never falls or destabilizes** from a
   sustained, non-reversing action. This ruled out the slew clamp as the reason position control
   lacks torque control's natural "bad exploration gets physically punished" property — the
   real reason is structural (see below), not the rate limiter.

5. **Imitation-learning warm start**: rolled out the working torque policy, recorded its actual
   resulting joint-angle trajectories, and supervised-pretrained a position-mode policy to imitate
   those trajectories (behavior cloning) before any RL fine-tuning. **Result**: at 1M steps of
   subsequent RL fine-tuning, the policy showed genuine reciprocating leg swinging for the first
   time (confirmed by directly measuring legs spending most of the episode in ground contact with
   real swing-phase motion, not one-directional). **By 4M steps of further fine-tuning under the
   same setup, it had regressed back to the same monotonic-drift pattern**, without any change to
   the environment/reward in between — purely from continued PPO optimization.

## Best current understanding of the root cause

We believe this is a **local-optimum / credit-assignment problem specific to how PD-servo-mediated
(position) control interacts with PPO's reward-maximization**, not a bug in any single reward term
or a modeling omission:

- In **position control**, a policy that outputs a constant, saturated action (target = the edge
  of the allowed range) produces smooth, safe, *reliably survivable* motion — verified directly,
  it never causes a fall, with or without the rate limiter. Because the torso genuinely does move
  forward as it slowly leans/tips over the extended leg, this strategy also **reliably earns real
  reward** every episode. Once PPO discovers this, its gradient has every reason to reinforce it
  further — it's a stable, safe, positive-return attractor.

- In **torque control**, the equivalent strategy (constant maximal torque, never released) is
  **not safe** — verified directly: applying it to a single joint causes the robot to fall within
  about 0.3 seconds. There is no safe, reliably-rewarded monotonic-torque attractor for PPO to
  find and lock onto. Every rollout that tries sustained one-directional torque gets a strongly
  negative outcome, which forces exploration toward modulated (push, then release) torque profiles
  from very early in training — not because torque exploration is inherently smarter, but because
  nothing else survives.

- Given that, a single exploratory reversal of one joint's action, at a random moment,
  uncoordinated with the other legs' timing, tends to disrupt the *already-reliable* position-mode
  monotonic strategy without reproducing anything better (a real gait needs multiple joints
  reversing in mutual, properly-timed coordination — a much harder thing to stumble into via
  independent per-step exploration noise than "keep doing the one thing that already works").
  So there's a strong asymmetry: torque mode is structurally *pushed off* the bad attractor by
  physics; position mode has no equivalent push and gets stuck reinforcing it.

## What we're considering next (not yet implemented / not yet conclusive)

A `kp`/`kd` **curriculum** — start training at the softened gain (needed for exploration to
survive at all), then gradually stiffen toward the real-hardware-matching gain later in training,
once basic competence exists — specifically to remove the "safety" of the monotonic-drift strategy
partway through training, without reintroducing the original stiff-gain problem (immediate,
un-survivable snapping) from the very start. Not yet implemented or tested. Open question: is
there a principled/established way to pace or gate this (time-based vs. performance-gated) in the
legged-RL literature, and is a `kp` curriculum even the right lever, versus some other way to
introduce the "bad exploration should be unsafe" property into position control?

## Question for outside input

Is this a documented failure mode for PD/position-controlled legged locomotion RL (as opposed to
torque control)? Are there known techniques — reward design, curriculum, exploration strategy,
action-space reparameterization — for specifically preventing convergence to a
"monotonic-drift-to-limit" pseudo-gait, beyond gain curricula and imitation warm-starts (both
of which we've tried, and both of which have shown promise but not fully solved it)?
