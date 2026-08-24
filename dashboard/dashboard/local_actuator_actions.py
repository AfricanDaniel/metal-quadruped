"""Runs actuator's basic_control node directly on THIS machine (not the Jetson)."""

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
    """Same zombie-reaping caveat as training_actions.py's own _pid_alive"""

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
    """{'running': bool, 'log_tail': str}."""

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
    """(ok, message)."""

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
    """Best-effort TERM then KILL, matching training_actions. stop_training()'s own pattern."""

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
