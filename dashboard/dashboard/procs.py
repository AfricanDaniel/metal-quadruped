"""In-memory process/connection registry for the Jetson tab (and the
two connection flags). Single-user local tool -- plain Python state
behind a lock, no database.

Design choice: rather than locally buffering "live" output from a
background thread, every long-running remote action (hardware bringup,
deploy, colcon build) is started via ssh.run_background() with its
output redirected to a REMOTE log file, and this module only remembers
that process's remote PID + log path. "What has it printed so far" is
answered fresh on every request via ssh.tail_log().

"Is it still running" for hardware bringup and deploy specifically is
NOT just a check of the remembered PID -- it's a live `pgrep -f
<pattern>` against the Jetson itself (see _find_running_pid), so it
correctly detects one already running from BEFORE a dashboard restart
(this module's own state is gone, in-memory only) or one started
directly in a terminal outside the dashboard entirely (this module
never had a PID for it in the first place). An earlier version of this
module only trusted its own remembered PID, which meant
start_hardware_bringup()/start_deploy() could -- and did -- launch a
SECOND actuator+dog_imu pair or a second policy_node on top of one
already running, with both fighting to command the same motors at
once. build_status() doesn't need this same treatment: a stray
untracked `colcon build` isn't a safety hazard the way two things
touching the real robot simultaneously is, so it's left as a simple
remembered-PID check.
"""
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
    # Local Tools sub-tab (2026-08-16): {tool_key: {field: value}} -- one
    # entry PER TOOL, not a single shared dict like last_deploy_form,
    # since that page has 5 independent forms all posting to different
    # routes. A single shared dict would use dict-replace semantics (see
    # set_last_deploy_form's own docstring) -- submitting any ONE tool's
    # form would wipe every OTHER tool's remembered values, since each
    # POST's request.form only ever contains that one form's own fields.
    'last_tools_forms': {},
    'build': {'pid': None, 'log': None},
    # Last GET page visited within each tab, so the tab bar itself can
    # jump back to wherever you left off (e.g. a specific fname's
    # checkpoint list) instead of always landing on that tab's root --
    # updated from app.py's before_request hook, on GET requests only.
    'last_path': {'local': '/local', 'jetson': '/jetson', 'sheep': '/sheep', 'jetson_policies': '/jetson/home'},
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
    """Ground-truth check via pgrep -f <pattern> on the Jetson itself --
    NOT just this process's own memory of a PID it launched. Without
    this, a dashboard restart (in-memory state is gone) or a process
    started directly in a terminal (this dashboard never knew its PID
    in the first place) would both be silently invisible to
    hardware_bringup_status()/deploy_status(), which would then happily
    let start_hardware_bringup()/start_deploy() launch a SECOND
    actuator+dog_imu pair or a SECOND policy_node on top of one already
    running -- two processes both trying to own the same serial port /
    command the same motors at once, which is exactly what was reported
    as the robot 'acting weird'. Confirmed this gap directly: neither
    function previously did any live check before launching.

    Deliberately no `| head -1` here: piping means the remote shell
    invoking pgrep can't exec-replace itself (it has to stick around to
    run the pipeline), so THAT shell process's own command line --
    which literally contains the search pattern, since it's the `bash
    -c "pgrep -f 'pattern' | head -1"` sshd used to run this --
    becomes a match for its own search. Confirmed this exact false
    positive directly: querying with the pipe reported hardware_bringup
    as running when nothing was actually running on the Jetson at all.
    Dropping the pipe and taking the first line in Python avoids it."""
    pids = _find_all_running_pids(pattern)
    return pids[0] if pids else None


def _find_all_running_pids(pattern):
    """Same live pgrep -f check as _find_running_pid, but returns every
    matching pid instead of just the first. See _find_running_pid's
    docstring for why the pipe-free form matters."""
    result = ssh.run(JETSON_HOST, f'pgrep -f {shlex.quote(pattern)}', timeout_s=10)
    if not result.ok:
        return []
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


# --- hardware bringup ----------------------------------------------------

# hardware_bringup.launch.py is a ROS launch file that starts actuator's
# basic_control AND dog_imu's imu_node as separate child processes.
# CONFIRMED THE HARD WAY: those children do NOT share the launch
# process's process group (unlike ordinary multiprocessing children,
# e.g. the sheep-training SubprocVecEnv workers -- see ssh.kill), so
# killing just the launch PID (even via ssh.kill's process-group kill)
# does not reach them. A real incident: stopping hardware bringup from
# the dashboard reported success while basic_control was still alive
# and the robot kept slowly moving on its own, because the orphaned
# actuator node still held its connection to the motors. Every function
# below checks/kills ALL THREE patterns -- the launch process itself
# plus both node executables by name -- not just the launch process,
# specifically so this can't happen again.
_HW_BRINGUP_PATTERN = 'hardware_bringup.launch.py'
_HW_BRINGUP_PATTERNS = [_HW_BRINGUP_PATTERN, 'basic_control', 'lib/dog_imu/imu_node']
_NOT_FROM_DASHBOARD_LOG = '(already running, but not started from this dashboard session -- no log available here)'


def _find_hw_bringup_pid():
    """First matching pid across ALL hardware-bringup-related patterns
    -- the launch supervisor OR either of its node children -- so an
    orphaned basic_control/imu_node with no launch parent still counts
    as 'running' here."""
    for pattern in _HW_BRINGUP_PATTERNS:
        pid = _find_running_pid(pattern)
        if pid is not None:
            return pid
    return None


def start_hardware_bringup():
    """(ok, message). Refuses to launch a second hardware_bringup on
    top of one already running -- checked live (see _find_hw_bringup_pid),
    so this catches one left over from before a dashboard restart, one
    started directly in a terminal, or an orphaned node with no launch
    parent left, not just one this exact process remembers launching.

    Passes the last-set actuator params (position_kp/position_kd/
    pose_speed_deg_s, see set_last_tool_form('actuator_params', ...) in
    app.py's live-set route) as launch arguments to hardware_bringup.
    launch.py -- 2026-08-17, user request ("persist across restarts"):
    without this, a value set live via `ros2 param set` only lasts until
    the actuator node is next restarted, silently reverting to whatever
    default is hardcoded in basic_control.cpp. Any param never explicitly
    set (empty string, get_last_tool_form's own default) is simply
    omitted from the launch command, so hardware_bringup launches exactly
    as before this feature existed until the user actually sets something."""
    existing_pid = _find_hw_bringup_pid()
    if existing_pid is not None:
        with _lock:
            _state['hardware_bringup'] = {'pid': existing_pid, 'log': None}
        # ok=True: the end state the button promises (bringup running) is
        # already true -- there's no ambiguity like deploy's "which
        # policy" question, so this isn't an error, just a no-op.
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
    """Kills the launch process AND every node process it started, each
    found and killed independently by name -- not by following any
    parent/process-group relationship from a single tracked PID, which
    is exactly what let basic_control survive a 'successful' stop
    before (see the module-level comment above _HW_BRINGUP_PATTERNS).

    Zeros motor torque FIRST, before any kill signal, via
    ros_actions.disable_motors() -- direct user request ("motors are
    locked in position... can they be disabled so I can manually move
    them"), and NOT left to basic_control's own destructor-based
    zeroing on shutdown: confirmed directly that the destructor's
    cleanup doesn't reliably fire before the process is gone (see
    disable_motors()'s own docstring). Best-effort -- if basic_control
    is already dead this call just fails harmlessly, and the kill loop
    below still runs regardless."""
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
    """{'running': bool, 'log_tail': str} -- ALWAYS re-verified live via
    pgrep, every call, not just trusted from memory. See
    _find_hw_bringup_pid for why this checks every node pattern, not
    just the launch process."""
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
    """(ok, message). ros_args: list of already-formatted
    '-p key:=value' strings. Refuses to launch a second policy_node on
    top of one already running -- see _find_running_pid's docstring;
    two policy_node processes both issuing motor commands at once is a
    real safety hazard, not just a UI inconsistency."""
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
    """Remembers the raw submitted Deploy form (checkboxes/text/number
    fields) so jetson_detail() can pre-fill the SAME values next render,
    instead of the form silently resetting to its hardcoded defaults
    after every redirect -- same 'stop losing what I just picked'
    complaint as the earlier log_csv_enabled-only fix, generalized to
    every field on the form. dict(form) turns the submitted
    ImmutableMultiDict into a plain dict; checkboxes are naturally
    key-present (checked) or key-absent (unchecked) in a form submit, so
    a plain .get() on the stored dict reproduces that directly."""
    with _lock:
        _state['last_deploy_form'] = dict(form)


def get_last_deploy_form():
    with _lock:
        return dict(_state['last_deploy_form'])


def set_last_tool_form(tool_key, form):
    """Same 'remember the raw submitted form' purpose as
    set_last_deploy_form(), just keyed per-tool -- see last_tools_forms'
    own comment in _state for why a single shared dict doesn't work here."""
    with _lock:
        _state['last_tools_forms'][tool_key] = dict(form)


def get_last_tool_form(tool_key):
    with _lock:
        return dict(_state['last_tools_forms'].get(tool_key, {}))


def deploy_status():
    """{'running': bool, 'policy': str|None, 'log_tail': str} -- ALWAYS
    re-verified live via pgrep, every call, not just trusted from
    memory. See _find_running_pid for why that distinction matters
    here."""
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
