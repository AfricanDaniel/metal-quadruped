"""Paths and remote-host constants for the dashboard.

WS_ROOT is found by SEARCHING upward from this file's own location, not
by a fixed number of '..' -- a fixed depth silently breaks depending on
HOW this package is being run: from source
(src/dashboard/dashboard/config.py, 3 levels up to ws root) vs. colcon's
own installed copy (install/dashboard/lib/python3.12/site-packages/
dashboard/config.py, 5 levels up) land at completely different depths
for the exact same workspace. Confirmed this the hard way: the fixed-
depth version pointed MODELS_DIR at .../install/dashboard/lib/models
when run via the installed `ros2 run dashboard dashboard` entry point,
while working fine under plain `python3 -m dashboard` from source.
"""
import os
import tempfile


def _find_ws_root(start):
    """Walks upward from `start` for the first ancestor directory that
    has BOTH a `src` and a `models` subdirectory -- this workspace's own
    unique signature, regardless of how deep `start` itself is nested
    under either `src/` (dev mode) or `install/` (colcon-installed)."""
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, 'src')) and os.path.isdir(os.path.join(d, 'models')):
            return d
        parent = os.path.dirname(d)
        if parent == d:  # reached filesystem root without finding it
            return os.path.abspath(start)  # fall back to something, rather than crash
        d = parent


WS_ROOT = os.environ.get('DOG_ROS2_WS_ROOT') or _find_ws_root(os.path.dirname(__file__))
SRC_ROOT = os.path.join(WS_ROOT, 'src')

MODELS_DIR = os.path.join(WS_ROOT, 'models')
POLICIES_POSITION_DIR = os.path.join(SRC_ROOT, 'policies_position')
POLICIES_TORQUE_DIR = os.path.join(SRC_ROOT, 'policies_torque')

# dog_gym/train.py's own --log-dir default is the RELATIVE path
# 'dogGymTrain_logs', resolved against whatever cwd it's launched from --
# every existing invocation (manual, and this dashboard's own subprocess
# launches) uses cwd=WS_ROOT, so this is that same relative path resolved
# the same way, not a separate guess. Confirmed directly: the real,
# already-populated log dir from this project's own training history
# lives at exactly WS_ROOT/dogGymTrain_logs.
LOG_DIR = os.path.join(WS_ROOT, 'dogGymTrain_logs')

# Local mirror for graphs.py's sheep_read_scalars() -- TensorBoard event
# files are small (scalar summaries only, unlike model checkpoints), so
# downloading them to view graphs is cheap. Kept SEPARATE from LOG_DIR
# (not merged into the same folder) so there's never any ambiguity about
# which run folders are this machine's own local training vs. a cached
# copy of sheep's.
SHEEP_LOG_CACHE_DIR = os.path.join(WS_ROOT, 'dogGymTrain_logs_sheep_cache')

# The venv this whole project's own training/testing/export commands have
# been run from all session (confirmed working: mujoco/gymnasium/
# stable_baselines3/torch all live here, not system python3).
VENV_PYTHON = os.path.join(WS_ROOT, '.venv', 'bin', 'python3')

ROS_SETUP_BASH = '/opt/ros/kilted/setup.bash'
INSTALL_SETUP_BASH = os.path.join(WS_ROOT, 'install', 'setup.bash')

# Sourced before every local subprocess this dashboard spawns (view
# training / export policy) -- matches how every manual invocation this
# whole project has used was itself run.
SOURCE_PREFIX = f'source {ROS_SETUP_BASH} 2>/dev/null; source {INSTALL_SETUP_BASH} 2>/dev/null; '

# --- Remote hosts -----------------------------------------------------
# Confirmed reachable this session via `ssh -o BatchMode=yes ...`.
# `jetson` is a plain ~/.bashrc alias (ssh msr@100.80.152.22), not an
# ssh-config Host -- so the dashboard connects to the same address
# directly rather than depending on that alias being sourced into
# whatever shell spawns this dashboard.
JETSON_HOST = 'msr@100.80.152.22'
JETSON_WS_ROOT = '~/dog_ros2_ws'

# `sheep` IS a real ssh-config Host (~/.ssh/config), so this can be used
# as-is as the ssh target.
SHEEP_HOST = 'sheep'
SHEEP_WS_ROOT = '~/final_project/dog_ros2_ws'
# Same relative-path convention as local LOG_DIR, just under sheep's own
# workspace root instead -- sheep runs this exact same train.py.
SHEEP_LOG_DIR = f'{SHEEP_WS_ROOT}/dogGymTrain_logs'
# Confirmed directly (not assumed): sheep's system python3 does NOT have
# mujoco/stable_baselines3 -- same .venv convention as local, just under
# sheep's own workspace root. A plain 'python3' in a remote training
# launch command would silently fail with ModuleNotFoundError.
SHEEP_VENV_PYTHON = f'{SHEEP_WS_ROOT}/.venv/bin/python3'

SSH_CONNECT_TIMEOUT_S = 5

# Connection multiplexing: without this, every single ssh/scp call in
# this module (there can be a dozen+ in a row for a group checkpoint
# download) pays its own full handshake, and if the remote host is under
# heavy load (e.g. sheep mid-training, 3 PPO runs pegging the CPU) some
# of those handshakes miss the 5s ConnectTimeout and the call just fails
# -- confirmed this is a real, not hypothetical, failure mode (a group
# download that silently landed as "1/10 checkpoints"). With
# ControlMaster, only the FIRST call to a given host pays the handshake;
# everything else for the next 60s of inactivity reuses that same
# connection, which is both faster and far less likely to intermittently
# fail under load.
SSH_CONTROL_PATH = os.path.join(tempfile.gettempdir(), 'dashboard-ssh-%r@%h:%p')
SSH_OPTS = [
    '-o', 'BatchMode=yes', '-o', f'ConnectTimeout={SSH_CONNECT_TIMEOUT_S}',
    '-o', 'ControlMaster=auto', '-o', 'ControlPersist=60s',
    '-o', f'ControlPath={SSH_CONTROL_PATH}',
]
