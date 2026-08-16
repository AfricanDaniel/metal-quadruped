"""Launches dog_gym.train --train as a detached local subprocess, and
tracks it well enough to survive a dashboard restart -- mirrors ssh.py/
procs.py's "PID + log file, status always re-verified live, never
trusted from stale memory" philosophy, just for a LOCAL process instead
of a remote one (Jetson/sheep's long-running processes are already
independently alive from the dashboard's own process since they're
remote; a local subprocess needs its own persistence story since the
dashboard process itself might restart).

Marker file convention: one <fname>.run.json per running (or just-
finished-and-not-yet-pruned) local training, sitting flat in LOG_DIR
(NOT inside a <fname>_N/ TensorBoard run folder -- those are SB3's own,
this dashboard doesn't own their contents). Contains enough to report
status and to `kill` it later: {pid, fname, cmd, started_at}.
"""
import json
import os
import signal
import subprocess
import time

from dashboard import remote_fs, ssh
from dashboard.config import (
    LOG_DIR, SHEEP_HOST, SHEEP_VENV_PYTHON, SHEEP_WS_ROOT, SOURCE_PREFIX, VENV_PYTHON, WS_ROOT,
)

# --- command construction -----------------------------------------------

# (form field name, CLI flag, python type) for the "advanced" fields --
# each gated by a same-named '<field>_enabled' checkbox in the form.
# Simple, uniform enough (checkbox present -> pass --flag <value>) that a
# table beats writing out 10 near-identical if-blocks by hand.
_ADVANCED_FIELDS = [
    ('n_steps', '--n-steps', int),
    ('batch_size', '--batch-size', int),
    ('n_epochs', '--n-epochs', int),
    ('learning_rate', '--learning-rate', float),
    ('learning_rate_end', '--learning-rate-end', float),
    ('ent_coef', '--ent-coef', float),
    ('ent_coef_end', '--ent-coef-end', float),
    ('decay_steps', '--decay-steps', int),
    ('timesteps_per_iter', '--timesteps-per-iter', int),
    ('init_from', '--init-from', str),
    ('env_type', '--env-type', str),
    # Slew-rate curriculum (2026-08-16, dog_gym/dog_gym/train.py's
    # SlewCurriculumCallback -- see MAX_SLEW_DEG_PER_S/
    # SLEW_CURRICULUM_TARGET_DEG_PER_S in dog_env.py): opt-in, same as
    # every other advanced field here -- omitting start-step leaves
    # train.py's own default (no curriculum, fixed at 1000 the whole run).
    ('slew_curriculum_start_step', '--slew-curriculum-start-step', int),
    ('slew_curriculum_decay_steps', '--slew-curriculum-decay-steps', int),
]


def build_train_args(form):
    """form: a Flask request.form-like mapping (the launch-training POST
    body). Returns a list of CLI arg strings for `dog_gym.train --train`,
    matching train.py's actual argparse surface -- see the plan's
    "Ground truth" section for the full flag list this was built from."""
    env_id = form.get('env_id', 'Dog-Walk-v0')
    is_walk = env_id == 'Dog-Walk-v0'

    args = [
        '--train',
        '--env-id', env_id,
        '--algo', form.get('algo', 'PPO'),
        '--control-mode', form.get('control_mode', 'position'),
        '--fname', form['fname'],
        '--model-dir', f"models/{form['model_dir']}",
        '--num-envs', str(int(form.get('num_envs') or 32)),
    ]

    if form.get('domain_randomization'):
        args.append('--domain-randomization')

    if is_walk:
        args += ['--walk-start-pose', form.get('start_pose', 'home')]
        if form.get('walk_height_fraction_enabled'):
            args += ['--walk-height-fraction', form.get('walk_height_fraction', '0.9')]

    # Gain mode: mutually exclusive at the train.py argparse level, so
    # only ONE of these branches ever contributes anything.
    gain_mode = form.get('gain_mode', 'range')
    if gain_mode == 'range':
        pairs = [
            ('--position-kp-range-start', 'kp_start_low', 'kp_start_high'),
            ('--position-kp-range-end', 'kp_end_low', 'kp_end_high'),
            ('--position-kd-range-start', 'kd_start_low', 'kd_start_high'),
            ('--position-kd-range-end', 'kd_end_low', 'kd_end_high'),
        ]
        if all(form.get(lo) and form.get(hi) for _, lo, hi in pairs):
            for flag, lo, hi in pairs:
                args += [flag, form[lo], form[hi]]
    elif gain_mode == 'fixed':
        if form.get('position_kp') and form.get('position_kd'):
            args += ['--position-kp', form['position_kp'], '--position-kd', form['position_kd']]

    if form.get('lr_schedule_enabled'):
        schedule = form.get('lr_schedule', 'linear')
        args += ['--lr-schedule', schedule]
        if schedule == 'adaptive' and form.get('desired_kl'):
            args += ['--desired-kl', form['desired_kl']]

    for field, flag, _type in _ADVANCED_FIELDS:
        if form.get(f'{field}_enabled') and form.get(field):
            args += [flag, form[field]]

    return args


# --- launch / track / stop -----------------------------------------------

def _marker_path(fname):
    return os.path.join(LOG_DIR, f'{fname}.run.json')


def _stdout_log_path(fname):
    return os.path.join(LOG_DIR, f'{fname}.stdout.log')


def launch_local_training(form):
    """Returns (ok, message). Spawns detached, writes the PID marker."""
    fname = form.get('fname', '').strip()
    if not fname:
        return False, 'fname is required.'
    if is_training_running(fname):
        return False, f'"{fname}" is already running.'

    os.makedirs(LOG_DIR, exist_ok=True)
    train_args = build_train_args(form)
    cmd_parts = [VENV_PYTHON, '-m', 'dog_gym.train'] + train_args
    cmd = SOURCE_PREFIX + ' '.join(cmd_parts)

    log_path = _stdout_log_path(fname)
    try:
        with open(log_path, 'ab') as logf:
            proc = subprocess.Popen(
                ['bash', '-c', cmd], cwd=WS_ROOT,
                stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,  # detach -- survives the dashboard itself restarting
            )
    except OSError as e:
        return False, str(e)

    marker = {'pid': proc.pid, 'fname': fname, 'cmd': cmd, 'started_at': time.time()}
    with open(_marker_path(fname), 'w') as f:
        json.dump(marker, f)
    return True, f'Launched "{fname}" (pid {proc.pid}).'


def _pid_alive(pid):
    """os.kill(pid, 0) alone isn't enough: launch_local_training() never
    calls wait()/poll() on its Popen (deliberately -- it needs to
    outlive this request and possibly the whole dashboard process), so
    when the child exits it becomes a zombie, and a zombie still answers
    kill(pid, 0) as "exists" until reaped. In a one-shot script the
    zombie gets reaped for free when that script's process exits, which
    is why this looked fine under ad-hoc testing -- but the real
    dashboard is one long-lived process, so without an explicit reap a
    finished/killed training would report as "running" forever. Reap
    first (only succeeds if pid is actually our own child, e.g. within
    the same dashboard process lifetime -- ChildProcessError otherwise,
    which is fine, kill(pid, 0) still correctly answers that case)."""
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


def list_running_local_trainings():
    """[{fname, pid, started_at, cmd}] -- scans LOG_DIR for *.run.json
    markers, verifies each PID is genuinely still alive (never trusts the
    marker's mere existence), and PRUNES (deletes) markers for processes
    that are no longer running -- so a training that finished or crashed
    naturally drops off this list on the next page load, no manual
    cleanup needed."""
    running = []
    if not os.path.isdir(LOG_DIR):
        return running
    for name in os.listdir(LOG_DIR):
        if not name.endswith('.run.json'):
            continue
        path = os.path.join(LOG_DIR, name)
        try:
            with open(path) as f:
                marker = json.load(f)
        except (OSError, ValueError):
            continue
        if _pid_alive(marker.get('pid')):
            running.append(marker)
        else:
            try:
                os.remove(path)
            except OSError:
                pass
    running.sort(key=lambda m: m.get('started_at', 0), reverse=True)
    return running


def is_training_running(fname):
    path = _marker_path(fname)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            marker = json.load(f)
    except (OSError, ValueError):
        return False
    if _pid_alive(marker.get('pid')):
        return True
    try:
        os.remove(path)
    except OSError:
        pass
    return False


def stop_training(fname):
    """Best-effort TERM then KILL, mirroring ssh.kill()'s own pattern.
    Returns True once the process is confirmed gone (or was already
    gone)."""
    path = _marker_path(fname)
    if not os.path.exists(path):
        return True
    try:
        with open(path) as f:
            marker = json.load(f)
    except (OSError, ValueError):
        os.remove(path)
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
            os.remove(path)
        except OSError:
            pass
    return ok


def tail_stdout_log(fname, max_bytes=20000):
    path = _stdout_log_path(fname)
    if not os.path.exists(path):
        return ''
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - max_bytes))
        return f.read().decode(errors='replace')


# --- sheep (remote) training launch/stop ---------------------------------
# No local PID-persistence story needed here (unlike the local functions
# above) -- a remote training is already independently alive from the
# dashboard's own process, so remote_fs.list_remote_running_trainings()
# (a live pgrep, not a stored marker) is the sole source of truth. Reuses
# build_train_args() -- the argument LIST is host-independent, only how
# it gets launched differs.

_SHEEP_PREFIX = (f'cd {SHEEP_WS_ROOT} && source /opt/ros/*/setup.bash && '
                 f'source install/setup.bash && ')
_SHEEP_LOG_DIR = f'{SHEEP_WS_ROOT}/.dashboard_logs'


def launch_remote_training(form):
    fname = form.get('fname', '').strip()
    if not fname:
        return False, 'fname is required.'
    if any(t['fname'] == fname for t in remote_fs.list_remote_running_trainings(SHEEP_HOST)):
        return False, f'"{fname}" is already running on sheep.'

    ssh.run(SHEEP_HOST, f'mkdir -p {_SHEEP_LOG_DIR}', timeout_s=10)
    train_args = build_train_args(form)
    cmd_parts = [SHEEP_VENV_PYTHON, '-m', 'dog_gym.train'] + train_args
    cmd = _SHEEP_PREFIX + ' '.join(cmd_parts)
    log_path = f'{_SHEEP_LOG_DIR}/{fname}.stdout.log'

    pid = ssh.run_background(SHEEP_HOST, cmd, log_path)
    if pid is None:
        return False, 'Failed to launch on sheep.'
    return True, f'Launched "{fname}" on sheep (pid {pid}).'


def stop_remote_training(fname):
    trainings = remote_fs.list_remote_running_trainings(SHEEP_HOST)
    match = next((t for t in trainings if t['fname'] == fname), None)
    if match is None:
        return True  # already not running
    return ssh.kill(SHEEP_HOST, match['pid'])


def tail_remote_stdout_log(fname, max_bytes=20000):
    return ssh.tail_log(SHEEP_HOST, f'{_SHEEP_LOG_DIR}/{fname}.stdout.log', max_bytes=max_bytes)
