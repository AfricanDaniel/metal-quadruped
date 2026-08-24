"""Shared back-leg home-reference correction, used by both
set_home_and_cache.py (to write a reference 'edited' snapshot alongside
the regular capture) and policy_node.py (to compute the mid-walk switch
target from whatever 'regular' reference was just loaded, at deploy
time -- see home_switch_back_leg_fraction's declare_parameter comment).

Origin (2026-08-15, hf_v17_fixed: real-hardware back thighs sit ~35-40deg
past where swing_amplitude_penalty holds them in sim): real-hardware back
thighs (leg_c/motor 5, leg_d/motor 8) sit
much further from home than sim ever produces for the same trained
policy -- a genuine sim-to-real gap, not a reward-shaping problem.
Manually re-posing the back legs further forward at set_home time and
re-measuring confirmed the correction is close to linear/proportional
over the range tested; MOTOR_5_CORRECTION_DEG/MOTOR_8_CORRECTION_DEG
below are the exact deltas of the first software-only step taken on top
of that physical recalibration (fraction=1.0 reproduces that step
exactly: regular was motor5=-8.671062469482422, motor8=57.61500549316406;
edited was motor5=-27.0, motor8=72.0)."""

MOTOR_5_CORRECTION_DEG = -27.0 - (-8.671062469482422)   # -18.328937530517578
MOTOR_8_CORRECTION_DEG = 72.0 - 57.61500549316406        # 14.38499450683594

BACK_LEG_HOME_CORRECTION_DEG = {5: MOTOR_5_CORRECTION_DEG, 8: MOTOR_8_CORRECTION_DEG}

# Front legs (2026-08-18, user report -- "front legs still too close to
# the face when the policy is deployed, need to be a little further
# away, the exact opposite issue as the back legs"): motors 1/4 (leg_a/
# leg_b thigh, see dog_description/config/motor_mapping.yaml) need an
# analogous correction, same empirical procedure as the back legs
# (same "hf_v17_fixed" origin noted above): read
# motor 1/4 at the regular home pose, physically re-pose to the correct-
# looking position, read again, delta = correction. MEASURED 2026-08-18
# on real hardware: regular was motor1=48.16, motor4=45.70; re-posed was
# motor1=72.94, motor4=24.62. Directional sanity check against motor_
# mapping.yaml's own convention ("away from front" is INCREASING theta
# for right-side thighs (motor 1) but DECREASING for left-side (motor
# 4)): motor 1 increased and motor 4 decreased -- both correctly mean
# "moved away from the face" for their respective sides, consistent with
# the reported problem. User flagged some possible tilt error while
# physically placing the robot for this measurement -- same "first
# empirical pass, not a formal sweep" caveat the back-leg constants
# above carry; consider a lower fraction (e.g. 0.5) for the first real
# test given that caveat, same as MOTOR_5/8_CORRECTION_DEG's own
# fraction=1.0-is-not-mandatory design.
MOTOR_1_CORRECTION_DEG = 72.94 - 48.16   # 24.780000000000058
MOTOR_4_CORRECTION_DEG = 24.62 - 45.70   # -21.079999999999984

FRONT_LEG_HOME_CORRECTION_DEG = {1: MOTOR_1_CORRECTION_DEG, 4: MOTOR_4_CORRECTION_DEG}


def apply_home_correction(home_position_deg, corrections, fraction):
    """Generic version of apply_back_leg_correction/apply_front_leg_
    correction below -- home_position_deg: list of 8 floats, motor 1..8
    order (same convention as policy_node.py/set_home_and_cache.py
    everywhere else). corrections: {motor_id (1-indexed): correction_deg}.
    Returns a NEW list -- each motor in `corrections` shifted by
    fraction * its own correction constant, every other motor unchanged.
    fraction=1.0 reproduces the measured correction exactly; fraction=0.0
    returns home_position_deg unchanged (as a new list); intermediate/
    out-of-[0,1] fractions are allowed (caller's choice), since the back-
    leg correction was found to behave close to linearly over the tested
    range. fraction=0.0 is ALWAYS a safe no-op (returns an unmodified
    copy immediately) regardless of whether any correction in
    `corrections` is still None/unmeasured -- only a nonzero fraction
    requires every motor it touches to already have a real measured
    value, so an unmeasured correction (e.g. FRONT_LEG_HOME_CORRECTION_
    DEG right now) can sit in this file inertly without breaking the
    default-disabled (fraction=0.0) case."""
    corrected = list(home_position_deg)
    if fraction == 0.0:
        return corrected
    for motor_id, delta in corrections.items():
        if delta is None:
            raise ValueError(
                f'motor {motor_id} has no measured correction yet (None) -- '
                'cannot apply a nonzero fraction until it is measured and filled in.')
        corrected[motor_id - 1] += fraction * delta
    return corrected


def apply_back_leg_correction(home_position_deg, fraction):
    return apply_home_correction(home_position_deg, BACK_LEG_HOME_CORRECTION_DEG, fraction)


def apply_front_leg_correction(home_position_deg, fraction):
    return apply_home_correction(home_position_deg, FRONT_LEG_HOME_CORRECTION_DEG, fraction)
