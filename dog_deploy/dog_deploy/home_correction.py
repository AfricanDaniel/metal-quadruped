"""Shared back-leg home-reference correction, used by both set_home_and_cache.py (to write a reference 'edited' snapshot..."""


MOTOR_5_CORRECTION_DEG = -27.0 - (-8.671062469482422)   # -18.328937530517578
MOTOR_8_CORRECTION_DEG = 72.0 - 57.61500549316406        # 14.38499450683594

BACK_LEG_HOME_CORRECTION_DEG = {5: MOTOR_5_CORRECTION_DEG, 8: MOTOR_8_CORRECTION_DEG}

# Front legs : motors 1/4 (leg_a/ leg_b thigh, see dog_description/config/motor_mapping.yaml) need an analogous correction, same empirical procedure as the back legs (same "hf_v17_fixed" origin noted above): read motor 1/4 at the regular home pose, physically re-pose to the correct- looking position, read again, delta = correction.
MOTOR_1_CORRECTION_DEG = 72.94 - 48.16   # 24.780000000000058
MOTOR_4_CORRECTION_DEG = 24.62 - 45.70   # -21.079999999999984

FRONT_LEG_HOME_CORRECTION_DEG = {1: MOTOR_1_CORRECTION_DEG, 4: MOTOR_4_CORRECTION_DEG}


def apply_home_correction(home_position_deg, corrections, fraction):
    """Generic version of apply_back_leg_correction/apply_front_leg_ correction below."""

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
