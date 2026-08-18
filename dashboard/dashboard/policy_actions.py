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

# View Training's own default when the max_slew_deg_per_s box is left
# blank (2026-08-18, user request) -- duplicates dog_gym/envs/dog_env.py's
# SLEW_CURRICULUM_TARGET_DEG_PER_S (the real-deployment-matching target
# most checkpoints are actually trained/converged toward), NOT dog_env.py's
# own bare-CLI-omission default (MAX_SLEW_DEG_PER_S=1000, the loose
# training-exploration ceiling -- viewing a checkpoint under that makes it
# look far more violent than it really is, see launch_view_training()'s
# own docstring). Keep this in sync if SLEW_CURRICULUM_TARGET_DEG_PER_S
# ever changes (same "keep duplicated constants in sync" pattern already
# used for dog_deploy/home_correction.py's CALF_RANGE_DEG).
VIEW_DEFAULT_MAX_SLEW_DEG_PER_S = 250


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


def launch_view_training(zip_path, env_id, episodes, start_pose=None, control_mode=None,
                          max_slew_deg_per_s=None):
    """Spawns `python3 -m dog_gym.train --test ...` DETACHED (the
    dashboard's own request returns immediately; the MuJoCo viewer opens
    in its own window and keeps running independently of the dashboard
    process). Returns True if the subprocess was launched (not whether
    the rollout itself succeeds -- there's no synchronous way to know
    that for a GUI window).

    max_slew_deg_per_s (2026-08-17, user request): without this, --test
    never passes --max-slew-deg-per-s, so dog_env.py's own loose module
    default (MAX_SLEW_DEG_PER_S=1000) is always used at view time --
    regardless of what slew clamp the checkpoint was actually TRAINED
    under at that step if it came from a run using --slew-curriculum-
    start-step/-decay-steps (which tightens the clamp over training, e.g.
    down to 250 by a few million steps in). Viewing under the wrong
    (looser) clamp than a checkpoint actually trained with makes it look
    more violent/unstable than it really is -- confirmed directly
    (chatbot.md-adjacent investigation, 2026-08-17): re-testing
    PPO_5500000_roll_gate_standing_v1 at its OWN real trained clamp
    (250) instead of the loose test-default (1000) dropped its peak
    front-leg swing from 7.08 to 3.41deg/tick and let it survive the
    full test window instead of falling early. This lets the caller
    override it per-viewing so a checkpoint can be watched under its own
    realistic clamp instead of always the loose default."""
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
