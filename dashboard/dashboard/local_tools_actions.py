"""Wraps four local dev/diagnostic tools this project's own workflow already runs by hand from a terminal."""

import os
import subprocess

from dashboard.config import SOURCE_PREFIX, SRC_ROOT, VENV_PYTHON, WS_ROOT

# save_pose.py lives under src/, not the workspace root
SAVE_POSE_SCRIPT = os.path.join(SRC_ROOT, 'dog_description', 'mjcf', 'save_pose.py')
MJCF_DIR = os.path.join(SRC_ROOT, 'dog_description', 'mjcf')
# Matches generate_dog_mjcf.py's actual output files (dog_description/
# README.md's own file listing) -- not hand-guessed.
MJCF_CHOICES = [
    'dog.mjcf.xml',
    'dog_torque.mjcf.xml',
    'dog_torque_belt.mjcf.xml',
    'dog_legacy_6kg.mjcf.xml',
    'dog_torque_v1_legacy_5nm.xml',
]


def _launch(cmd_parts):
    """Fire-and-forget detached local subprocess."""

    cmd = SOURCE_PREFIX + ' '.join(cmd_parts)
    try:
        subprocess.Popen(
            ['bash', '-c', cmd],
            cwd=WS_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True, cmd
    except OSError as e:
        return False, str(e)


def launch_save_pose(form):
    """save_pose.py."""

    cmd_parts = [VENV_PYTHON, SAVE_POSE_SCRIPT]
    if form.get('mjcf_enabled') and form.get('mjcf'):
        cmd_parts += ['--mjcf', os.path.join(MJCF_DIR, form['mjcf'])]
    if form.get('out_enabled') and form.get('out'):
        cmd_parts += ['--out', form['out']]
    return _launch(cmd_parts)


def launch_manual_motor_control(form):
    cmd_parts = [VENV_PYTHON, '-m', 'dog_gym.manual_motor_control']
    if form.get('step_deg_enabled') and form.get('step_deg'):
        cmd_parts += ['--step-deg', form['step_deg']]
    if form.get('orientation_enabled') and form.get('orientation'):
        cmd_parts += ['--orientation', form['orientation']]
    if form.get('mmc_pin_height_m_enabled') and form.get('mmc_pin_height_m'):
        cmd_parts += ['--pin-height-m', form['mmc_pin_height_m']]
    return _launch(cmd_parts)


def launch_verify_belt_decoupling(form):
    """."""

    cmd_parts = [VENV_PYTHON, '-m', 'dog_gym.verify_belt_decoupling']
    if form.get('leg_enabled') and form.get('leg'):
        cmd_parts += ['--leg', form['leg']]
    if form.get('amplitude_deg_enabled') and form.get('amplitude_deg'):
        cmd_parts += ['--amplitude-deg', form['amplitude_deg']]
    if form.get('full_range'):
        cmd_parts += ['--full-range']
    if form.get('no_coupling'):
        cmd_parts += ['--no-coupling']
    if form.get('period_s_enabled') and form.get('period_s'):
        cmd_parts += ['--period-s', form['period_s']]
    if form.get('joint_stiffness_enabled') and form.get('joint_stiffness'):
        cmd_parts += ['--joint-stiffness', form['joint_stiffness']]
    if form.get('vbd_max_slew_deg_per_s_enabled') and form.get('vbd_max_slew_deg_per_s'):
        cmd_parts += ['--max-slew-deg-per-s', form['vbd_max_slew_deg_per_s']]
    if form.get('duration_s_enabled') and form.get('duration_s'):
        cmd_parts += ['--duration-s', form['duration_s']]
    cmd_parts += ['--upside-down' if form.get('upside_down') else '--no-upside-down']
    if form.get('vbd_pin_height_m_enabled') and form.get('vbd_pin_height_m'):
        cmd_parts += ['--pin-height-m', form['vbd_pin_height_m']]
    return _launch(cmd_parts)


def launch_dog_view():
    """No configurable launch arguments."""

    return _launch(['ros2', 'launch', 'dog_description', 'dog_view.launch.py'])
