# dog_imu

ROS 2 driver node for the LSM6DSO32 6-axis IMU, mounted on the Jetson and
read over I2C. Publishes `sensor_msgs/msg/Imu`.

Promoted from the `actuator/src/imu_reader.py` prototype (same
register-level driver logic, WHO_AM_I check included), but wired up as a
real publisher instead of printing to the console — this is what
`dog_deploy` subscribes to for the IMU half of a policy's observation, and
what `dog_gym`'s simulated `accelerometer`/`gyro` sensors are meant to
match the shape of.

```
dog_imu/
├── package.xml
├── setup.py / setup.cfg
├── resource/dog_imu
└── dog_imu/
    └── imu_node.py
```

## Node: `imu_node`

On startup: opens the I2C bus, reads `WHO_AM_I` (logged for a sanity
check), configures the accelerometer (104 Hz, ±4g) and gyroscope (104 Hz,
500 dps), then publishes `sensor_msgs/Imu` on `imu/data_raw` on a timer.

- `angular_velocity` — rad/s (converted from the sensor's deg/s).
- `linear_acceleration` — m/s².
- `orientation` — left as identity with `orientation_covariance[0] = -1`,
  the `sensor_msgs/Imu` convention for "no orientation estimate available"
  (this sensor has no onboard sensor fusion — it's raw accel + gyro only).

### Parameters

| Name               | Type   | Default    | Description                                    |
|---------------------|--------|------------|--------------------------------------------------|
| `bus_id`           | int    | `7`        | I2C bus number (Jetson Orin Nano: bus 7)          |
| `i2c_addr`         | int    | `0x6A`     | LSM6DSO32 I2C address                             |
| `frame_id`         | string | `imu_link` | Frame ID stamped on published messages            |
| `publish_rate_hz`  | double | `100.0`    | Publish rate (independent of the sensor's own ODR)|

## Usage

```bash
colcon build --packages-select dog_imu
source install/setup.bash
ros2 run dog_imu imu_node
```

```bash
ros2 topic echo /imu/data_raw
```

Requires hardware (the LSM6DSO32 over I2C) — there's no simulated fallback
here, so this can only be run/verified on the Jetson, not the dev machine.

## Register map reference

Carried over from the original prototype:

- `0x0F` `WHO_AM_I` — read-only ID check.
- `0x10` `CTRL1_XL` — accelerometer ODR + full-scale. `0x58` = 104 Hz, ±4g.
- `0x11` `CTRL2_G` — gyroscope ODR + full-scale. `0x54` = 104 Hz, 500 dps.
- `0x22`-`0x27` — gyro X/Y/Z, 16-bit signed, low byte first.
- `0x28`-`0x2D` — accel X/Y/Z, 16-bit signed, low byte first.

Sensitivity for the settings above: gyro `0.0175 dps/LSB`, accel
`0.000122 g/LSB`.
