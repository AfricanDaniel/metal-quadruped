"""Launches dog_gym.train."""

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

# (form field name, CLI flag, python type) for the "advanced" fields
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
    # Slew-rate curriculum (train.py's SlewCurriculumCallback
    ('slew_curriculum_start_step', '--slew-curriculum-start-step', int),
    ('slew_curriculum_decay_steps', '--slew-curriculum-decay-steps', int),
    # Starting slew ceiling itself, independent of the two curriculum fields above (those control whether/how it tightens, not what it starts at).
    ('max_slew_deg_per_s', '--max-slew-deg-per-s', float),
    # Forward-speed curriculum (train.py's ForwardSpeedCurriculumCallback
    ('forward_speed_curriculum_start_step', '--forward-speed-curriculum-start-step', int),
    ('forward_speed_curriculum_decay_steps', '--forward-speed-curriculum-decay-steps', int),
    ('forward_speed_curriculum_target', '--forward-speed-curriculum-target', float),
    # --use-sde only (gSDE exploration
    ('sde_sample_freq', '--sde-sample-freq', int),
    # --joint-stiffness: opt-in override of generate_dog_mjcf.py's baked-in per-leg-joint stiffness="0" (see train.py's own --joint-stiffness help).
    ('joint_stiffness', '--joint-stiffness', float),
]


def build_train_args(form):
    """form: a Flask request.form-like mapping (the launch-training POST body)."""

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

    # gSDE: plain boolean, same pattern as domain_randomization above.
    if form.get('use_sde'):
        args.append('--use-sde')

    if is_walk:
        args += ['--walk-start-pose', form.get('start_pose', 'home')]
        if form.get('walk_height_fraction_enabled'):
            args += ['--walk-height-fraction', form.get('walk_height_fraction', '0.9')]
        # gait_style: front-and-center like start_pose above, not buried
        # in _ADVANCED_FIELDS, for easier discoverability.
        args += ['--gait-style', form.get('gait_style', 'trot')]

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
    """Returns (ok, message)."""
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
    """os.kill(pid, 0) alone isn't enough: launch_local_training() never calls wait()/poll() on its Popen (deliberately."""

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
    """[{fname, pid, started_at, cmd}]."""

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
    """Best-effort TERM then KILL, mirroring ssh.kill()'s own pattern. Returns True once the process is confirmed gone (or w..."""

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


# --- sheep (remote) training launch/stop --------------------------------- No local PID-persistence story needed here (unlike the local functions above)

_SHEEP_PREFIX = (f'cd {SHEEP_WS_ROOT} && source /opt/ros/*/setup.bash && '
                 f'source install/setup.bash && ')
_SHEEP_LOG_DIR = f'{SHEEP_WS_ROOT}/.dashboard_logs'


def _cuda_device_prefix(form):
    """'' or 'CUDA_VISIBLE_DEVICES=<n> '."""

    device = form.get('cuda_device', '').strip()
    return f'CUDA_VISIBLE_DEVICES={device} ' if device != '' else ''


def launch_remote_training(form):
    fname = form.get('fname', '').strip()
    if not fname:
        return False, 'fname is required.'
    if any(t['fname'] == fname for t in remote_fs.list_remote_running_trainings(SHEEP_HOST)):
        return False, f'"{fname}" is already running on sheep.'

    ssh.run(SHEEP_HOST, f'mkdir -p {_SHEEP_LOG_DIR}', timeout_s=10)
    train_args = build_train_args(form)
    cmd_parts = [SHEEP_VENV_PYTHON, '-m', 'dog_gym.train'] + train_args
    cmd = _SHEEP_PREFIX + _cuda_device_prefix(form) + ' '.join(cmd_parts)
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
