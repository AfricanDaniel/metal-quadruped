"""colcon build (+ source install/setup.bash) for Local and Sheep, from the Trainings sub-tab."""

import json
import os
import subprocess
import threading
import time

from dashboard import ssh
from dashboard.config import INSTALL_SETUP_BASH, ROS_SETUP_BASH, SHEEP_HOST, SHEEP_WS_ROOT, WS_ROOT

# --- local ---------------------------------------------------------------

_LOCAL_LOG_PATH = os.path.join(WS_ROOT, '.dashboard_local_build.log')
_LOCAL_MARKER_PATH = os.path.join(WS_ROOT, '.dashboard_local_build.run.json')


def _pid_alive(pid):
    """See training_actions._pid_alive for why this reaps before checking: an un-wait()'d Popen child becomes a zombie on ex..."""

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
    except (OSError, ProcessLookupError):
        return False


def _tail_local_log(max_bytes=20000):
    if not os.path.exists(_LOCAL_LOG_PATH):
        return ''
    with open(_LOCAL_LOG_PATH, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode(errors='replace')


def start_local_build():
    if local_build_status()['running']:
        return False, 'A local build is already running.'
    cmd = f'source {ROS_SETUP_BASH} 2>/dev/null; cd {WS_ROOT} && colcon build 2>&1; source {INSTALL_SETUP_BASH} 2>/dev/null'
    try:
        with open(_LOCAL_LOG_PATH, 'wb') as logf:
            proc = subprocess.Popen(
                ['bash', '-c', cmd], cwd=WS_ROOT,
                stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    except OSError as e:
        return False, str(e)
    with open(_LOCAL_MARKER_PATH, 'w') as f:
        json.dump({'pid': proc.pid, 'started_at': time.time()}, f)
    return True, f'Build started (pid {proc.pid}).'


def local_build_status():
    """{'running': bool, 'log_tail': str}."""

    running = False
    if os.path.exists(_LOCAL_MARKER_PATH):
        try:
            with open(_LOCAL_MARKER_PATH) as f:
                marker = json.load(f)
            running = _pid_alive(marker.get('pid'))
        except (OSError, ValueError):
            running = False
        if not running:
            try:
                os.remove(_LOCAL_MARKER_PATH)
            except OSError:
                pass
    return {'running': running, 'log_tail': _tail_local_log()}


# --- sheep -----------------------------------------------------------------

_sheep_lock = threading.Lock()
_sheep_state = {'pid': None, 'log': None}
_SHEEP_BUILD_LOG_DIR = f'{SHEEP_WS_ROOT}/.dashboard_logs'


def start_sheep_build():
    with _sheep_lock:
        pid = _sheep_state['pid']
    if pid is not None and ssh.is_running(SHEEP_HOST, pid):
        return False, 'A build is already running on sheep.'
    ssh.run(SHEEP_HOST, f'mkdir -p {_SHEEP_BUILD_LOG_DIR}', timeout_s=10)
    log = f'{_SHEEP_BUILD_LOG_DIR}/build_{int(time.time())}.log'
    cmd = f'cd {SHEEP_WS_ROOT} && source /opt/ros/*/setup.bash && colcon build 2>&1; source install/setup.bash'
    new_pid = ssh.run_background(SHEEP_HOST, cmd, log)
    if new_pid is None:
        return False, 'Failed to start build on sheep.'
    with _sheep_lock:
        _sheep_state['pid'] = new_pid
        _sheep_state['log'] = log
    return True, f'Build started on sheep (pid {new_pid}).'


def sheep_build_status():
    with _sheep_lock:
        pid, log = _sheep_state['pid'], _sheep_state['log']
    if pid is None:
        return {'running': False, 'log_tail': ''}
    running = ssh.is_running(SHEEP_HOST, pid)
    if not running:
        with _sheep_lock:
            _sheep_state['pid'] = None
    return {'running': running, 'log_tail': ssh.tail_log(SHEEP_HOST, log) if log else ''}
