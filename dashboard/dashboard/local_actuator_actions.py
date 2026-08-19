"""Runs actuator's basic_control node directly on THIS machine (not the
Jetson) -- for bench-debugging motors plugged straight into this laptop
(2026-08-18, user request: "Motors section on Local Tools ... it runs
the actuator node so i can debug motors when they are plugged in to my
laptop"). Mirrors training_actions.py's own "PID + log marker file,
status always re-verified live" pattern for LOCAL long-running processes
-- procs.py's in-memory-only state (fine for Jetson/Sheep, since those
processes are independently alive from the dashboard's own process
either way) isn't enough for a local child: a dashboard restart would
forget its PID (see training_actions.py's own module docstring for the
full reasoning already established for local training runs).

Deliberately just basic_control, not the full hardware_bringup.launch.py
actuator+dog_imu pair the Jetson uses -- IMU isn't relevant for bench
motor debugging and may not even be plugged in on a laptop.
"""
import json
import os
import signal
import subprocess
import time

from dashboard.config import SOURCE_PREFIX, WS_ROOT

LOCAL_ACTUATOR_LOG_DIR = os.path.join(WS_ROOT, '.dashboard_logs')
_MARKER_PATH = os.path.join(LOCAL_ACTUATOR_LOG_DIR, 'local_actuator.run.json')
_LOG_PATH = os.path.join(LOCAL_ACTUATOR_LOG_DIR, 'local_actuator.stdout.log')


def _pid_alive(pid):
    """Same zombie-reaping caveat as training_actions.py's own
    _pid_alive -- see that function's docstring for the full reasoning
    (a Popen'd child this process never wait()s on becomes a zombie that
    still answers kill(pid, 0) as 'exists' until reaped)."""
    if not pid:
        return False
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, TypeError):
        return False


def _read_marker():
    if not os.path.exists(_MARKER_PATH):
        return None
    try:
        with open(_MARKER_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def local_actuator_status():
    """{'running': bool, 'log_tail': str} -- ALWAYS re-verified live via
    the marker's own PID, never trusted from memory alone (matches every
    other status check in this dashboard)."""
    marker = _read_marker()
    if marker is None or not _pid_alive(marker.get('pid')):
        if marker is not None:
            try:
                os.remove(_MARKER_PATH)
            except OSError:
                pass
        return {'running': False, 'log_tail': ''}
    log_tail = ''
    if os.path.exists(_LOG_PATH):
        with open(_LOG_PATH, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8000))
            log_tail = f.read().decode(errors='replace')
    return {'running': True, 'log_tail': log_tail}


def start_local_actuator():
    """(ok, message). Refuses to launch a second instance on top of one
    already running, checked live (see local_actuator_status)."""
    if local_actuator_status()['running']:
        return True, 'basic_control is already running locally -- not starting a second one.'
    os.makedirs(LOCAL_ACTUATOR_LOG_DIR, exist_ok=True)
    cmd = SOURCE_PREFIX + 'ros2 run actuator basic_control'
    try:
        with open(_LOG_PATH, 'ab') as logf:
            proc = subprocess.Popen(
                ['bash', '-c', cmd], cwd=WS_ROOT,
                stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,  # detach -- survives the dashboard itself restarting
            )
    except OSError as e:
        return False, str(e)
    with open(_MARKER_PATH, 'w') as f:
        json.dump({'pid': proc.pid, 'started_at': time.time()}, f)
    return True, f'Started basic_control locally (pid {proc.pid}).'


def stop_local_actuator():
    """Best-effort TERM then KILL, matching training_actions.
    stop_training()'s own pattern. Doesn't zero motor torque first
    (unlike Jetson's stop_hardware_bringup, via ros_actions.
    disable_motors()) -- there's no local equivalent yet, and a bench
    setup being actively debugged (usually a single motor on a stand,
    not a loaded leg) is lower-stakes than the full assembled robot."""
    marker = _read_marker()
    if marker is None:
        return True
    pid = marker.get('pid')
    if pid and _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        time.sleep(1.0)
        if _pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            time.sleep(0.5)
    ok = not _pid_alive(pid) if pid else True
    if ok:
        try:
            os.remove(_MARKER_PATH)
        except OSError:
            pass
    return ok
