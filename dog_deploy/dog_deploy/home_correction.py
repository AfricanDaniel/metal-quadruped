"""Shared back-leg home-reference correction, used by both
set_home_and_cache.py (to write a reference 'edited' snapshot alongside
the regular capture) and policy_node.py (to compute the mid-walk switch
target from whatever 'regular' reference was just loaded, at deploy
time -- see home_switch_back_leg_fraction's declare_parameter comment).

Origin (daniel_cl_context.md, 2026-08-15, "hf_v17_fixed: real-hardware
back thighs sit ~35-40deg past where swing_amplitude_penalty holds them
in sim"): real-hardware back thighs (leg_c/motor 5, leg_d/motor 8) sit
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


def apply_back_leg_correction(home_position_deg, fraction):
    """home_position_deg: list of 8 floats, motor 1..8 order (same
    convention as policy_node.py/set_home_and_cache.py everywhere else).
    Returns a NEW list -- motors 5 and 8 (indices 4 and 7) shifted by
    fraction * their own correction constant above, every other motor
    unchanged. fraction=1.0 reproduces the original software correction
    exactly; fraction=0.0 returns home_position_deg unchanged (as new
    list); intermediate/out-of-[0,1] fractions are allowed (caller's
    choice), since the correction was found to behave close to linearly
    over the tested range."""
    corrected = list(home_position_deg)
    corrected[4] += fraction * MOTOR_5_CORRECTION_DEG   # motor 5 (leg_c_thigh)
    corrected[7] += fraction * MOTOR_8_CORRECTION_DEG   # motor 8 (leg_d_thigh)
    return corrected
