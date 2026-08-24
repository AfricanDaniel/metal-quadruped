"""In-memory process/connection registry for the Jetson tab (and the two connection flags)."""

import shlex
import threading
import time

from dashboard import ros_actions, ssh
from dashboard.config import JETSON_HOST, JETSON_WS_ROOT

_lock = threading.Lock()

REMOTE_LOG_DIR = f'{JETSON_WS_ROOT}/.dashboard_logs'

_state = {
    'jetson_connected': False,
    'sheep_connected': False,
    'hardware_bringup': {'pid': None, 'log': None},
    'deploy': {'pid': None, 'log': None, 'policy': None},
    'last_deploy_form': {},
    # Local Tools sub-tab: {tool_key: {field: value}}
    'last_tools_forms': {},
    'build': {'pid': None, 'log': None},
    # Last GET page visited within each tab, so the tab bar itself can jump back to wherever you left off (e.g.
    'last_path': {
        'local': '/local', 'jetson': '/jetson', 'sheep': '/sheep', 'jetson_policies': '/jetson/home',
        # Local/Sheep sub-tab-specific tracking, same purpose as
        # jetson_policies above.
        'local_models': '/local', 'local_trainings': '/local/trainings',
        'local_policies': '/local/policies', 'local_tools': '/local/tools',
        'sheep_models': '/sheep/home', 'sheep_trainings': '/sheep/trainings',
    },
}


def set_last_path(tab, path):
    with _lock:
        _state['last_path'][tab] = path


def get_last_path(tab):
    with _lock:
        return _state['last_path'][tab]


def set_connected(which, value):
    with _lock:
        _state[f'{which}_connected'] = value


def is_connected(which):
    with _lock:
        return _state[f'{which}_connected']


def _remote_log_path(name):
    return f'{REMOTE_LOG_DIR}/{name}_{int(time.time())}.log'


def _ensure_log_dir():
    ssh.run(JETSON_HOST, f'mkdir -p {REMOTE_LOG_DIR}', timeout_s=10)


def _find_running_pid(pattern):
    """Ground-truth check via pgrep -f <pattern> on the Jetson itself."""

    pids = _find_all_running_pids(pattern)
    return pids[0] if pids else None


def _find_all_running_pids(pattern):
    """Same live pgrep -f check as _find_running_pid, but returns every matching pid instead of just the first."""

    result = ssh.run(JETSON_HOST, f'pgrep -f {shlex.quote(pattern)}', timeout_s=10)
    if not result.ok:
        return []
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


# --- hardware bringup ----------------------------------------------------

# hardware_bringup.launch.py is a ROS launch file that starts actuator's basic_control AND dog_imu's imu_node as separate child processes that do NOT share the launch process's process group (unlike ordinary multiprocessing children, e.g.
_HW_BRINGUP_PATTERN = 'hardware_bringup.launch.py'
_HW_BRINGUP_PATTERNS = [_HW_BRINGUP_PATTERN, 'basic_control', 'lib/dog_imu/imu_node']
_NOT_FROM_DASHBOARD_LOG = '(already running, but not started from this dashboard session -- no log available here)'


def _find_hw_bringup_pid():
    """First matching pid across ALL hardware-bringup-related patterns."""

    for pattern in _HW_BRINGUP_PATTERNS:
        pid = _find_running_pid(pattern)
        if pid is not None:
            return pid
    return None


def start_hardware_bringup():
    """(ok, message)."""

    existing_pid = _find_hw_bringup_pid()
    if existing_pid is not None:
        with _lock:
            _state['hardware_bringup'] = {'pid': existing_pid, 'log': None}
        # ok=True: the end state the button promises (bringup running) is already true
        return True, 'Hardware bringup is already running (found an existing process) -- not starting a second one.'
    _ensure_log_dir()
    log = _remote_log_path('hardware_bringup')
    actuator_params = get_last_tool_form('actuator_params')
    launch_args = ' '.join(
        f'{name}:={value}' for name in ('position_kp', 'position_kd', 'pose_speed_deg_s')
        if (value := actuator_params.get(name, '')))
    cmd = (f'cd {JETSON_WS_ROOT} && source /opt/ros/*/setup.bash && '
           'source install/setup.bash && '
           f'ros2 launch dog_deploy hardware_bringup.launch.py {launch_args}'.rstrip())
    pid = ssh.run_background(JETSON_HOST, cmd, log)
    with _lock:
        _state['hardware_bringup'] = {'pid': pid, 'log': log}
    ok = pid is not None
    return ok, ('Hardware bringup started.' if ok else 'Failed to start hardware bringup.')


def stop_hardware_bringup():
    """Kills the launch process AND every node process it started, each found and killed independently by name."""

    ros_actions.disable_motors()
    all_ok = True
    for pattern in _HW_BRINGUP_PATTERNS:
        for pid in _find_all_running_pids(pattern):
            if not ssh.kill(JETSON_HOST, pid):
                all_ok = False
    with _lock:
        _state['hardware_bringup'] = {'pid': None, 'log': None}
    return all_ok


def hardware_bringup_status():
    """{'running': bool, 'log_tail': str}."""

    live_pid = _find_hw_bringup_pid()
    with _lock:
        tracked_pid, tracked_log = _state['hardware_bringup']['pid'], _state['hardware_bringup']['log']
    if live_pid is None:
        if tracked_pid is not None:
            with _lock:
                _state['hardware_bringup'] = {'pid': None, 'log': None}
        return {'running': False, 'log_tail': ''}
    if live_pid != tracked_pid:
        with _lock:
            _state['hardware_bringup'] = {'pid': live_pid, 'log': None}
        return {'running': True, 'log_tail': _NOT_FROM_DASHBOARD_LOG}
    return {'running': True, 'log_tail': ssh.tail_log(JETSON_HOST, tracked_log) if tracked_log else ''}


# --- deploy ----------------------------------------------------------------

_POLICY_NODE_PATTERN = 'dog_deploy.policy_node'


def start_deploy(policy_pt_path, ros_args):
    """(ok, message)."""

    existing_pid = _find_running_pid(_POLICY_NODE_PATTERN)
    if existing_pid is not None:
        with _lock:
            existing_policy = _state['deploy'].get('policy')
            _state['deploy'] = {'pid': existing_pid, 'log': None, 'policy': existing_policy}
        return False, 'A policy is already deployed (found an existing policy_node) -- stop it first.'
    _ensure_log_dir()
    log = _remote_log_path('deploy')
    args_str = ' '.join(ros_args)
    cmd = (f'cd {JETSON_WS_ROOT} && source /opt/ros/*/setup.bash && '
           'source install/setup.bash && '
           f'python3 -m dog_deploy.policy_node --ros-args {args_str}')
    pid = ssh.run_background(JETSON_HOST, cmd, log)
    with _lock:
        _state['deploy'] = {'pid': pid, 'log': log, 'policy': policy_pt_path}
    ok = pid is not None
    return ok, ('Deployment started.' if ok else 'Failed to start deployment.')


def stop_deploy():
    with _lock:
        pid = _state['deploy']['pid']
    if pid is None:
        pid = _find_running_pid(_POLICY_NODE_PATTERN)
        if pid is None:
            return True
    ok = ssh.kill(JETSON_HOST, pid)
    if ok:
        with _lock:
            _state['deploy'] = {'pid': None, 'log': None, 'policy': None}
    return ok


def set_last_deploy_form(form):
    """Remembers the raw submitted Deploy form (checkboxes/text/number fields) so jetson_detail() can pre-fill the same valu..."""

    with _lock:
        _state['last_deploy_form'] = dict(form)


def get_last_deploy_form():
    with _lock:
        return dict(_state['last_deploy_form'])


def set_last_tool_form(tool_key, form):
    """Same 'remember the raw submitted form' purpose as set_last_deploy_form(), just keyed per-tool."""

    with _lock:
        _state['last_tools_forms'][tool_key] = dict(form)


def get_last_tool_form(tool_key):
    with _lock:
        return dict(_state['last_tools_forms'].get(tool_key, {}))


def deploy_status():
    """{'running': bool, 'policy': str|None, 'log_tail': str}."""

    live_pid = _find_running_pid(_POLICY_NODE_PATTERN)
    with _lock:
        d = dict(_state['deploy'])
    if live_pid is None:
        if d['pid'] is not None:
            with _lock:
                _state['deploy'] = {'pid': None, 'log': None, 'policy': None}
        return {'running': False, 'policy': None, 'log_tail': ''}
    if live_pid != d['pid']:
        with _lock:
            _state['deploy'] = {'pid': live_pid, 'log': None, 'policy': d.get('policy')}
        return {'running': True, 'policy': d.get('policy'), 'log_tail': _NOT_FROM_DASHBOARD_LOG}
    return {'running': True, 'policy': d['policy'],
            'log_tail': ssh.tail_log(JETSON_HOST, d['log']) if d['log'] else ''}


# --- colcon build ------------------------------------------------------

def start_build():
    _ensure_log_dir()
    log = _remote_log_path('build')
    cmd = f'cd {JETSON_WS_ROOT} && source /opt/ros/*/setup.bash && colcon build 2>&1; source install/setup.bash'
    pid = ssh.run_background(JETSON_HOST, cmd, log)
    with _lock:
        _state['build'] = {'pid': pid, 'log': log}
    return pid is not None


def build_status():
    with _lock:
        pid, log = _state['build']['pid'], _state['build']['log']
    if pid is None:
        return {'running': False, 'log_tail': ''}
    running = ssh.is_running(JETSON_HOST, pid)
    if not running:
        with _lock:
            _state['build'] = {'pid': None, 'log': None}
    return {'running': running, 'log_tail': ssh.tail_log(JETSON_HOST, log) if log else ''}
