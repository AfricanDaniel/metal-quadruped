"""Wraps four local dev/diagnostic tools this project's own workflow
already runs by hand from a terminal -- save_pose.py,
dog_gym.manual_motor_control, dog_gym.verify_belt_decoupling, and the
dog_view.launch.py RViz launch file (2026-08-16, user request: "add
another section on the local tab ... some of them are almost the same
just with different settings"). half_dog_view.launch.py was dropped
(2026-08-16) -- it depends on onshape_folders/urdf_half_dog_1/half_dog/
urdf/half_dog.urdf, which doesn't exist in this checkout.

Each is spawned as a DETACHED subprocess -- same fire-and-forget pattern
as policy_actions.launch_view_training: these all open their own GUI
window (MuJoCo viewer or RViz) and keep running independently of the
dashboard process once launched, so there's nothing to track/stop here.

Every optional field follows this project's established "checkbox gates
inclusion" convention (see training_actions.py's _ADVANCED_FIELDS): if
the checkbox wasn't checked, the flag is omitted entirely and the
underlying script's own argparse default applies -- the dashboard never
bakes in a second, possibly-drifting copy of that default.
"""
import os
import subprocess

from dashboard.config import SOURCE_PREFIX, SRC_ROOT, VENV_PYTHON, WS_ROOT

# save_pose.py lives under src/, not the workspace root -- SRC_ROOT
# (WS_ROOT/src), NOT WS_ROOT itself, matching how config.py's own
# POLICIES_POSITION_DIR/POLICIES_TORQUE_DIR are built.
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
    """Fire-and-forget detached local subprocess -- mirrors
    policy_actions.launch_view_training's exact mechanism (SOURCE_PREFIX,
    bash -c, cwd=WS_ROOT, fully detached via start_new_session)."""
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
    """save_pose.py --mjcf/--out both default to paths relative to the
    SCRIPT's own directory (Path(__file__).resolve().parent), not cwd --
    safe to launch from anywhere. --mjcf here is a dropdown of the known
    generated MJCF variants (MJCF_CHOICES) rather than a free-text path,
    since picking the wrong one silently is easy to do by hand and hard
    to notice from the viewer alone."""
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
    """--upside-down is argparse.BooleanOptionalAction (default True),
    not a plain store_true -- unlike every other checkbox on this page,
    leaving it checked/unchecked must EXPLICITLY pass --upside-down or
    --no-upside-down (there's no bare 'omit the flag' state that still
    lets the checkbox reflect what's actually going to run)."""
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
    """No configurable launch arguments -- DeclareLaunchArgument-free,
    confirmed directly against dog_view.launch.py."""
    return _launch(['ros2', 'launch', 'dog_description', 'dog_view.launch.py'])
