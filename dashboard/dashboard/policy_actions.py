"""Wraps dog_gym.train's"""

import os
import subprocess

from dashboard.config import (
    POLICIES_POSITION_DIR, POLICIES_TORQUE_DIR, SOURCE_PREFIX, VENV_PYTHON, WS_ROOT,
)

CONTROL_MODES = ['position', 'torque', 'torque_belt']
START_POSES = ['home', 'standing']

# View Training's own default when the max_slew_deg_per_s box is left blank.
VIEW_DEFAULT_MAX_SLEW_DEG_PER_S = 250


def _policies_dir(control_mode):
    return POLICIES_POSITION_DIR if control_mode == 'position' else POLICIES_TORQUE_DIR


# WALK lineage folders -- mirrors the 6 models/ folder names (see
# local_fs.LOCAL_POLICY_GROUPS).
_WALK_LINEAGE_FOLDERS = {
    'walk_home', 'walk_standing', 'trot_home', 'trot_stand', 'running_home', 'running_stand',
}


def _task_subfolder(env_id, source_folder=None):
    """'stand' for STAND-task checkpoints (no gait-lineage concept there). For WALK, routes by the checkpoint's own models/ ..."""

    if env_id != 'Dog-Walk-v0':
        return 'stand'
    if source_folder in _WALK_LINEAGE_FOLDERS:
        return source_folder
    return 'walk_home'


def export_target_group(env_id, control_mode, source_folder=None):
    """(policies_dir, task) as the plain strings the Local Policies routes key on ('policies_position'/'policies_torque', on..."""

    policies_dir = 'policies_position' if control_mode == 'position' else 'policies_torque'
    return policies_dir, _task_subfolder(env_id, source_folder)


def export_target_path(basename, env_id, control_mode, source_folder=None):
    """<repo>/src/policies_{position,torque}/<task>/policy/<basename>.pt"""

    return os.path.join(
        _policies_dir(control_mode), _task_subfolder(env_id, source_folder), 'policy', basename + '.pt')


def launch_view_training(zip_path, env_id, episodes, start_pose=None, control_mode=None,
                          max_slew_deg_per_s=None, joint_stiffness=None):
    """Spawns `python3 -m dog_gym.train."""

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
        if max_slew_deg_per_s:
            cmd_parts += ['--max-slew-deg-per-s', str(max_slew_deg_per_s)]
        # joint_stiffness: like max_slew_deg_per_s above, this is a runtime env-construction kwarg, not something saved inside the .zip checkpoint.
        if joint_stiffness:
            cmd_parts += ['--joint-stiffness', str(joint_stiffness)]
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


def run_export_policy(zip_path, env_id, control_mode, basename, source_folder=None):
    """Blocking (export is fast)."""

    target_path = export_target_path(basename, env_id, control_mode, source_folder)
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
