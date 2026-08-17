# Belt/pulley decoupling mechanism -- real-hardware calibration data

Raw `read_motor_positions` snapshots collected while manually swinging each
leg's THIGH by hand (robot upside-down, per this session's earlier
`motor_sweep_test`/`verify_belt_decoupling.py` investigation) and reading
back both motors -- the CALF was deliberately driven to (or near) its own
real mechanical min/max limits via the belt/pulley coupling as the thigh
moved, not independently commanded.

**This is a living document** -- more calibration sessions will be
appended below as separate dated sections, not overwritten. Values here
are RAW motor-degree readings (the exact same convention
`read_motor_positions`/`set_motor_targets` use, i.e. NOT relative to
`set_home`'s own reference) -- see `generate_dog_mjcf.py`'s
`JOINT_RANGE_OVERRIDES_DEG` for where bench-measured hard-stops like this
eventually get consumed once fully calibrated (that dict is the sim-side
consumer; this file is the raw-data side, same relationship
`JOINT_RANGE_OVERRIDES_DEG`'s own comment describes for `daniel_cl_context.md`,
just scoped specifically to the belt-decoupling recalibration effort
instead of the general log).

Motor -> joint mapping (`dog_description/config/motor_mapping.yaml`):
`1`=leg_a_thigh, `2`=leg_a_calf, `3`=leg_b_calf, `4`=leg_b_thigh,
`5`=leg_c_thigh, `6`=leg_c_calf, `7`=leg_d_calf, `8`=leg_d_thigh.

## 2026-08-16 session

### Home reference at time of this data
(`ros2 run dog_deploy set_home_and_cache`, robot physically tucked)

| motor | joint | home_deg (regular) |
|---|---|---|
| 1 | leg_a_thigh | 47.2205 |
| 2 | leg_a_calf | 47.6023 |
| 3 | leg_b_calf | 20.7786 |
| 4 | leg_b_thigh | 46.0143 |
| 5 | leg_c_thigh | 4.9118 |
| 6 | leg_c_calf | 16.3546 |
| 7 | leg_d_calf | 101.8260 |
| 8 | leg_d_thigh | 45.3165 |

(Also cached to the "edited"/back-leg-corrected snapshot at fraction=1.0,
per `home_correction.py` -- motor 5 -> -13.4172deg, motor 8 -> 59.7015deg.
Unrelated to the calf-range calibration below, noted here only because it
was produced by the same `set_home_and_cache` call.)

### leg_a (motor 1 = thigh, motor 2 = calf)

| # | thigh_deg | calf_deg | thigh_vel_deg_s | calf_vel_deg_s | thigh_torque_nm | calf_torque_nm |
|---|---|---|---|---|---|---|
| 1 | 52.34 | 268.27 | 1.11 | -3.78 | 0.000 | 0.000 |
| 2 | 40.47 | 56.96 | 0.89 | -2.22 | 0.000 | 0.000 |
| 3 | 145.86 | 173.99 | -4.89 | 0.67 | 0.000 | 0.000 |
| 4 | 152.63 | -56.68 | 0.89 | 0.00 | 0.000 | 0.000 |
| 5 | 245.78 | 76.84 | 0.00 | 0.22 | 0.000 | -0.025 |
| 6 | 234.18 | -133.95 | 2.00 | -0.22 | 0.000 | 0.025 |

**Observed range this session**: thigh 40.47 to 245.78deg, **calf -133.95 to 268.27deg**.

### leg_b (motor 3 = calf, motor 4 = thigh)

| # | calf_deg | thigh_deg | calf_vel_deg_s | thigh_vel_deg_s | calf_torque_nm | thigh_torque_nm |
|---|---|---|---|---|---|---|
| 1 | -194.25 | 35.42 | -4.89 | 0.22 | 0.000 | 0.000 |
| 2 | 30.22 | 36.85 | 3.11 | 1.33 | 0.000 | 0.000 |
| 3 | -99.34 | -58.94 | 1.11 | 0.89 | 0.000 | 0.025 |
| 4 | 126.03 | -58.28 | -4.00 | -1.56 | 0.000 | 0.000 |
| 5 | -8.13 | -149.44 | 0.89 | -0.89 | 0.000 | -0.025 |
| 6 | 211.50 | -146.75 | -2.22 | -2.22 | 0.000 | 0.000 |

**Observed range this session**: **calf -194.25 to 211.50deg**, thigh -149.44 to 36.85deg.

### leg_c (motor 5 = thigh, motor 6 = calf)

| # | thigh_deg | calf_deg | thigh_vel_deg_s | calf_vel_deg_s | thigh_torque_nm | calf_torque_nm |
|---|---|---|---|---|---|---|
| 1 | 24.44 | 221.73 | 2.22 | -1.78 | 0.000 | 0.025 |
| 2 | 22.20 | -1.43 | 1.11 | -3.78 | -0.025 | 0.000 |
| 3 | 108.45 | 139.56 | 0.44 | 3.33 | 0.000 | -0.025 |
| 4 | 107.07 | -84.44 | -3.11 | -2.22 | 0.025 | 0.000 |
| 5 | 204.97 | 43.07 | 0.00 | 0.89 | 0.000 | 0.000 |
| 6 | 207.63 | -183.52 | -2.67 | -1.33 | 0.000 | 0.000 |

**Observed range this session**: thigh 22.20 to 207.63deg, **calf -183.52 to 221.73deg**.

### leg_d (motor 7 = calf, motor 8 = thigh)

| # | calf_deg | thigh_deg | calf_vel_deg_s | thigh_vel_deg_s | calf_torque_nm | thigh_torque_nm |
|---|---|---|---|---|---|---|
| 1 | -106.14 | 28.08 | 0.00 | -6.22 | 0.000 | 0.000 |
| 2 | 121.92 | 25.64 | 0.00 | -2.67 | 0.025 | 0.000 |
| 3 | -13.85 | -63.89 | -2.00 | 1.11 | 0.025 | 0.000 |
| 4 | 205.89 | -55.88 | -2.22 | -2.22 | 0.000 | 0.000 |
| 5 | 136.17 | -196.17 | -4.89 | -3.33 | 0.000 | 0.000 |
| 6 | 325.03 | -178.14 | -4.22 | -2.89 | 0.000 | 0.000 |

**Observed range this session**: calf -106.14 to 325.03deg, thigh -196.17 to 28.08deg.

### Summary -- calf raw-degree extremes observed this session

| leg | calf motor | min_deg | max_deg | span_deg |
|---|---|---|---|---|
| leg_a | 2 | -133.95 | 268.27 | 402.22 |
| leg_b | 3 | -194.25 | 211.50 | 405.75 |
| leg_c | 6 | -183.52 | 221.73 | 405.25 |
| leg_d | 7 | -106.14 | 325.03 | 431.17 |

**Torque stayed near zero throughout** (max magnitude observed: 0.025 N·m,
essentially the measurement floor) -- consistent with this being purely
hand-guided, unloaded motion, not a motor-driven or force-limited test.

**Not yet concluded**: whether these observed extremes represent the
calf's TRUE mechanical hard-stops (i.e. the thigh was moved far enough in
both directions to actually bottom the calf out) or just how far this
particular session's manual sweep happened to reach -- more calibration
sessions planned to confirm/refine these numbers before they get fed into
`generate_dog_mjcf.py`'s `JOINT_RANGE_OVERRIDES_DEG` or anywhere else
calibrated values are consumed.
