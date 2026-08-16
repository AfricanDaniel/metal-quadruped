"""Wraps dog_gym.train's --test entry point (MuJoCo viewer) and
dog_gym.export_policy -- both run as subprocesses of the exact same
CLIs used manually throughout this whole project, not reimplemented.
"""
import os
import subprocess

from dashboard.config import (
    POLICIES_POSITION_DIR, POLICIES_TORQUE_DIR, SOURCE_PREFIX, VENV_PYTHON, WS_ROOT,
)

CONTROL_MODES = ['position', 'torque', 'torque_belt']
START_POSES = ['home', 'standing']


def _policies_dir(control_mode):
    return POLICIES_POSITION_DIR if control_mode == 'position' else POLICIES_TORQUE_DIR


def _task_subfolder(env_id):
    return 'walk' if env_id == 'Dog-Walk-v0' else 'stand'


def export_target_group(env_id, control_mode):
    """(policies_dir, task) as the plain strings the Local Policies
    routes key on ('policies_position'/'policies_torque', 'stand'/
    'walk') -- same mapping export_target_path() uses internally, just
    exposed so app.py can build a url_for('local_policy_detail', ...)
    link to wherever a given export actually landed."""
    policies_dir = 'policies_position' if control_mode == 'position' else 'policies_torque'
    return policies_dir, _task_subfolder(env_id)


def export_target_path(basename, env_id, control_mode):
    """<repo>/src/policies_{position,torque}/{stand,walk}/policy/<basename>.pt
    -- matches this project's own established layout exactly (see the
    plan's 'Ground truth' section)."""
    return os.path.join(
        _policies_dir(control_mode), _task_subfolder(env_id), 'policy', basename + '.pt')


def launch_view_training(zip_path, env_id, episodes, start_pose=None, control_mode=None):
    """Spawns `python3 -m dog_gym.train --test ...` DETACHED (the
    dashboard's own request returns immediately; the MuJoCo viewer opens
    in its own window and keeps running independently of the dashboard
    process). Returns True if the subprocess was launched (not whether
    the rollout itself succeeds -- there's no synchronous way to know
    that for a GUI window)."""
    cmd_parts = [
        VENV_PYTHON, '-m', 'dog_gym.train',
        '--test', zip_path,
        '--env-id', env_id,
        '--episodes', str(episodes),
    ]
    if env_id == 'Dog-Walk-v0':
        if start_pose:
            cmd_parts += ['--walk-start-pose', start_pose]
        if control_mode:
            cmd_parts += ['--control-mode', control_mode]
    cmd = SOURCE_PREFIX + ' '.join(cmd_parts)
    try:
        subprocess.Popen(
            ['bash', '-c', cmd],
            cwd=WS_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach fully from the dashboard process
        )
        return True, cmd
    except OSError as e:
        return False, str(e)


def run_export_policy(zip_path, env_id, control_mode, basename):
    """Blocking (export is fast) -- runs dog_gym.export_policy and
    returns (ok, message, target_path)."""
    target_path = export_target_path(basename, env_id, control_mode)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    cmd_parts = [
        VENV_PYTHON, '-m', 'dog_gym.export_policy',
        zip_path, target_path,
        '--env-id', env_id,
        '--control-mode', control_mode,
    ]
    cmd = SOURCE_PREFIX + ' '.join(cmd_parts)
    try:
        proc = subprocess.run(
            ['bash', '-c', cmd], cwd=WS_ROOT,
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return False, 'export_policy.py timed out after 120s', target_path
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-5:]
        return False, '\n'.join(tail) or f'exited with code {proc.returncode}', target_path
    return True, f'Exported to {target_path}', target_path
