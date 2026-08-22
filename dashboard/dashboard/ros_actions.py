"""Thin wrappers over the exact `ros2 service call ...` commands used
manually against the Jetson throughout this project -- each one sources
ROS + the workspace install first (a fresh `ssh host "cmd"` is a new
shell every time, nothing persists across calls) then runs the service
call, over SSH.

local_read_motor_positions()/local_adjust_motor_position() (2026-08-18,
user request -- Motors section for the Local Tools tab, talking to
local_actuator_actions.py's own locally-run basic_control instead of the
Jetson's) are the same two service calls run as a LOCAL subprocess
instead, for bench-debugging motors plugged straight into this laptop.
"""
import subprocess

from dashboard import ssh
from dashboard.config import JETSON_HOST, JETSON_WS_ROOT, SOURCE_PREFIX

_PREFIX = f'source /opt/ros/*/setup.bash && source {JETSON_WS_ROOT}/install/setup.bash && '


def _call(cmd, timeout_s=15):
    return ssh.run(JETSON_HOST, _PREFIX + cmd, timeout_s=timeout_s)


def _local_call(cmd, timeout_s=15):
    """Local equivalent of _call() -- same SshResult-shaped return (.ok/
    .stdout/.stderr) so app.py's route handlers can treat local and
    Jetson results identically."""
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
    """Local (not Jetson) equivalent of read_motor_positions() -- see
    that function's own docstring for the request format."""
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
    """Uses set_home_and_cache (dog_deploy), not the raw /set_home
    service call directly -- it calls the same service but ALSO writes
    the resulting home reference to ~/.dog_home_cache.yaml, so a later
    policy_node run can load it explicitly via
    home_position_deg_cache_path instead of auto-capturing its own
    (wrong, if the robot isn't tucked at that exact moment) home. Exits
    on its own once done (see dog_deploy/set_home_and_cache.py), so this
    can stay a plain synchronous call like the others here."""
    return _call('ros2 run dog_deploy set_home_and_cache')


def go_to_pose(pose_name, speed_deg_s=None, log_csv=False, log_csv_path=None):
    """speed_deg_s (2026-08-17, user request -- dashboard "Go to pose"
    section): GoToPose.srv already has a per-call speed_deg_s field
    (0/omitted = actuator's own pose_speed_deg_s param default, a
    positive value overrides it for just this call) -- previously never
    threaded through from the dashboard at all. None/0 omits the field
    from the request entirely, matching the service's own "0 = default"
    convention exactly rather than sending an explicit 0.0.

    log_csv/log_csv_path (2026-08-20, user request -- "is there a way
    for me to collect data from the basics tab... when i run go to pose
    home or standing"): GoToPose.srv already supports per-call CSV
    logging (position/velocity/torque/temp/error per motor for the
    move's duration) server-side in basic_control.cpp -- this was just
    never threaded through from the dashboard. log_csv_path left
    None/blank auto-generates a timestamped path under the actuator
    package's own data/csv_logs/ dir on whichever host runs this
    (Jetson for the jetson_ caller)."""
    req = f'pose_name: {pose_name}'
    if speed_deg_s:
        req += f', speed_deg_s: {speed_deg_s}'
    if log_csv:
        req += ', log_csv: true'
        if log_csv_path:
            req += f', log_csv_path: "{log_csv_path}"'
    return _call(f'ros2 service call /go_to_pose actuator/srv/GoToPose "{{{req}}}"')


def set_actuator_param(name, value):
    """Live `ros2 param set /actuator <name> <value>` (2026-08-17, user
    request -- "expose position_kp/position_kd/pose_speed_deg_s... test
    with different values without going to the code"). These are real,
    already-declared parameters on basic_control.cpp (see actuator/
    README.md's own documented `ros2 param set` pattern) -- this takes
    effect on the actuator node immediately, no restart/rebuild needed.
    Only affects the CURRENTLY RUNNING node; see procs.start_hardware_
    bringup()'s own comment for how a value also survives a restart."""
    return _call(f'ros2 param set /actuator {name} {value}')


def read_motor_positions(motor_ids=None):
    """motor_ids: list[int] or None (None/empty = all motors, per
    ReadMotorPositions.srv's own convention)."""
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
    """Zeros torque (kp=0, dq=0, tau=0) on the given motors via the
    ALREADY-EXPOSED set_motor_torque service, so they're free to move by
    hand -- default is 1..8, every motor. kd stays at basic_control's
    own torque_kd_gain default (0.2 N*m/(rad/s) output-shaft, light
    velocity damping) rather than the fully-zero kd the C++
    destructor's own cleanup uses -- gentle resistance to fast motion,
    not a hard lock, and correct because set_motor_torque's own handler
    (basic_control.cpp) always sets kd this way, not something this
    call can override.

    Called from procs.stop_hardware_bringup() BEFORE any kill signal is
    sent, deliberately NOT relying on basic_control's own destructor-
    based zeroing to fire on shutdown. Confirmed directly, the hard way:
    a real stop's own hardware_bringup log had zero "Motor N disabled
    and zeroed" lines (the destructor's own log message) anywhere after
    the last command sent -- some combination of ssh.kill's SIGKILL
    escalation and/or ros2 launch's own child-shutdown timeout can beat
    the destructor's multi-motor serial round-trips to the punch, and
    SIGKILL can't be caught at all, so the C++ cleanup path can't be
    trusted to run before the process is actually gone. This call is
    synchronous and made while the node is still definitely alive."""
    ids = ', '.join(str(i) for i in motor_ids)
    torques = ', '.join('0.0' for _ in motor_ids)
    return _call(f'ros2 service call /set_motor_torque '
                 f'actuator/srv/SetMotorTorque "{{motor_id: [{ids}], torque_nm: [{torques}]}}"')


def reset_motors():
    """Best-effort fix for a motor stuck executing an old command --
    proposed design, not an existing/battle-tested mechanism (see the
    plan's own note on this). Reasoning: a genuinely unresponsive motor
    almost always means the `actuator` node's own RS485/serial
    connection is wedged, not the motor itself -- so this kills just the
    `basic_control` process (leaving `dog_imu` running, unlike killing
    the whole hardware_bringup launch would) and starts a fresh one,
    which re-opens the serial connection from scratch. Runs standalone
    (`ros2 run`, not re-joining the original hardware_bringup launch
    tree) so it doesn't depend on tracking/restarting that launch's
    whole process group.

    Uses `pkill -x basic_control` (exact process NAME), not `-f
    "basic_control"` (full command line). Confirmed directly, the hard
    way, that `-f` is genuinely dangerous here: the remote shell
    invoked to RUN this pkill command has "basic_control" sitting right
    there in its own command line (as pkill's own argument), and pkill
    -f matches against that too -- so it kills its own parent shell
    mid-command, cutting off the SSH session before it can return
    anything. That is exactly what surfaced as "Failed to reset motors
    -- check the Jetson directly" with nothing actually wrong on the
    Jetson. `-x` matches only the exact process name (confirmed:
    basic_control's own `comm` is 'basic_control'; the wrapping `ros2
    run actuator basic_control` process's `comm` is just 'ros2', so -x
    doesn't touch it directly -- but confirmed separately that this
    wrapper exits on its own once its child dies, so nothing is left
    orphaned)."""
    kill_result = _call('pkill -x basic_control 2>/dev/null; sleep 1; echo done',
                         timeout_s=15)
    log = f'{JETSON_WS_ROOT}/.dashboard_logs/reset_motors_basic_control.log'
    restart_cmd = (f'mkdir -p {JETSON_WS_ROOT}/.dashboard_logs && '
                    f'source /opt/ros/*/setup.bash && '
                    f'source {JETSON_WS_ROOT}/install/setup.bash && '
                    'ros2 run actuator basic_control')
    pid = ssh.run_background(JETSON_HOST, restart_cmd, log)
    return kill_result.ok and pid is not None, pid
