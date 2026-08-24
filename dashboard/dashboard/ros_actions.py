"""Thin wrappers over the exact `ros2 service call ...` commands used manually against the Jetson throughout this project"""

import subprocess

from dashboard import ssh
from dashboard.config import JETSON_HOST, JETSON_WS_ROOT, SOURCE_PREFIX

_PREFIX = f'source /opt/ros/*/setup.bash && source {JETSON_WS_ROOT}/install/setup.bash && '


def _call(cmd, timeout_s=15):
    return ssh.run(JETSON_HOST, _PREFIX + cmd, timeout_s=timeout_s)


def _local_call(cmd, timeout_s=15):
    """Local equivalent of _call()."""

    try:
        proc = subprocess.run(
            ['bash', '-c', SOURCE_PREFIX + cmd],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return ssh.SshResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired:
        return ssh.SshResult(False, '', f'Timed out after {timeout_s}s.', None)
    except OSError as e:
        return ssh.SshResult(False, '', str(e), None)


def local_read_motor_positions(motor_ids=None):
    """Local (not Jetson) equivalent of read_motor_positions()."""

    if motor_ids:
        ids = ', '.join(str(i) for i in motor_ids)
        req = f'{{motor_id: [{ids}]}}'
    else:
        req = '{}'
    return _local_call(f'ros2 service call /read_motor_positions '
                        f'actuator/srv/ReadMotorPositions "{req}"')


def local_adjust_motor_position(motor_id, degrees):
    """Local (not Jetson) equivalent of adjust_motor_position()."""
    return _local_call(f'ros2 service call /adjust_motor_position '
                        f'actuator/srv/AdjustMotorPosition '
                        f'"{{motor_id: {int(motor_id)}, degrees: {float(degrees)}}}"')


def set_home():
    """Uses set_home_and_cache (dog_deploy), not the raw /set_home service call directly."""

    return _call('ros2 run dog_deploy set_home_and_cache')


def go_to_pose(pose_name, speed_deg_s=None, log_csv=False, log_csv_path=None):
    """speed_deg_s (2026-08-17, user request."""

    req = f'pose_name: {pose_name}'
    if speed_deg_s:
        req += f', speed_deg_s: {speed_deg_s}'
    if log_csv:
        req += ', log_csv: true'
        if log_csv_path:
            req += f', log_csv_path: "{log_csv_path}"'
    return _call(f'ros2 service call /go_to_pose actuator/srv/GoToPose "{{{req}}}"')


def set_actuator_param(name, value):
    """Live `ros2 param set /actuator <name> <value>` (2026-08-17, user request."""

    return _call(f'ros2 param set /actuator {name} {value}')


def read_motor_positions(motor_ids=None):
    """motor_ids: list[int] or None (None/empty = all motors, per ReadMotorPositions.srv's own convention)."""

    if motor_ids:
        ids = ', '.join(str(i) for i in motor_ids)
        req = f'{{motor_id: [{ids}]}}'
    else:
        req = '{}'
    return _call(f'ros2 service call /read_motor_positions '
                 f'actuator/srv/ReadMotorPositions "{req}"')


def adjust_motor_position(motor_id, degrees):
    return _call(f'ros2 service call /adjust_motor_position '
                 f'actuator/srv/AdjustMotorPosition '
                 f'"{{motor_id: {int(motor_id)}, degrees: {float(degrees)}}}"')


def disable_motors(motor_ids=range(1, 9)):
    """Zeros torque (kp=0, dq=0, tau=0) on the given motors via the ALREADY-EXPOSED set_motor_torque service, so they're fre..."""

    ids = ', '.join(str(i) for i in motor_ids)
    torques = ', '.join('0.0' for _ in motor_ids)
    return _call(f'ros2 service call /set_motor_torque '
                 f'actuator/srv/SetMotorTorque "{{motor_id: [{ids}], torque_nm: [{torques}]}}"')


def reset_motors():
    """Best-effort fix for a motor stuck executing an old command."""

    kill_result = _call('pkill -x basic_control 2>/dev/null; sleep 1; echo done',
                         timeout_s=15)
    log = f'{JETSON_WS_ROOT}/.dashboard_logs/reset_motors_basic_control.log'
    restart_cmd = (f'mkdir -p {JETSON_WS_ROOT}/.dashboard_logs && '
                    f'source /opt/ros/*/setup.bash && '
                    f'source {JETSON_WS_ROOT}/install/setup.bash && '
                    'ros2 run actuator basic_control')
    pid = ssh.run_background(JETSON_HOST, restart_cmd, log)
    return kill_result.ok and pid is not None, pid
