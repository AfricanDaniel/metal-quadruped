"""Paths and remote-host constants for the dashboard. WS_ROOT is found by SEARCHING upward from this file's own location..."""

import os
import tempfile


def _find_ws_root(start):
    """Walks upward from `start` for the first ancestor directory that has BOTH a `src` and a `models` subdirectory."""

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

# dog_gym/train.py's own --log-dir default is the relative path 'dogGymTrain_logs', resolved against cwd=WS_ROOT (how every invocation, manual and this dashboard's, launches it).
LOG_DIR = os.path.join(WS_ROOT, 'dogGymTrain_logs')

# Local mirror for graphs.py's sheep_read_scalars()
SHEEP_LOG_CACHE_DIR = os.path.join(WS_ROOT, 'dogGymTrain_logs_sheep_cache')

# The venv this project's training/testing/export commands run from
# (has mujoco/gymnasium/stable_baselines3/torch; system python3 doesn't).
VENV_PYTHON = os.path.join(WS_ROOT, '.venv', 'bin', 'python3')

ROS_SETUP_BASH = '/opt/ros/kilted/setup.bash'
INSTALL_SETUP_BASH = os.path.join(WS_ROOT, 'install', 'setup.bash')

# Sourced before every local subprocess this dashboard spawns (view
# training / export policy), matching how these are run manually.
SOURCE_PREFIX = f'source {ROS_SETUP_BASH} 2>/dev/null; source {INSTALL_SETUP_BASH} 2>/dev/null; '

# --- Remote hosts ----------------------------------------------------- `jetson` is a plain ~/.bashrc alias, not an ssh-config Host, so the dashboard connects to the same address directly rather than depending on that alias being sourced into whatever shell spawns it.
JETSON_HOST = 'msr@100.80.152.22'
JETSON_WS_ROOT = '~/dog_ros2_ws'

# `sheep` IS a real ssh-config Host (~/.ssh/config), usable as-is.
SHEEP_HOST = 'sheep'
SHEEP_WS_ROOT = '~/final_project/dog_ros2_ws'
# Same relative-path convention as local LOG_DIR, under sheep's own
# workspace root (sheep runs this same train.py).
SHEEP_LOG_DIR = f'{SHEEP_WS_ROOT}/dogGymTrain_logs'
# sheep's system python3 lacks mujoco/stable_baselines3, so use its .venv the same way local does, or a plain 'python3' launch fails with ModuleNotFoundError.
SHEEP_VENV_PYTHON = f'{SHEEP_WS_ROOT}/.venv/bin/python3'

# Tailscale (100.80.152.22) can take a few seconds to re-establish a path to a peer it hasn't talked to recently, so this gives the first connection attempt generous headroom rather than failing fast.
SSH_CONNECT_TIMEOUT_S = 15

# Connection multiplexing: without this, every ssh/scp call in this module pays its own handshake, and under load some can miss ConnectTimeout and fail outright.
SSH_CONTROL_PATH = os.path.join(tempfile.gettempdir(), 'dashboard-ssh-%r@%h:%p')
SSH_OPTS = [
    '-o', 'BatchMode=yes', '-o', f'ConnectTimeout={SSH_CONNECT_TIMEOUT_S}',
    '-o', 'ControlMaster=auto', '-o', 'ControlPersist=60s',
    '-o', f'ControlPath={SSH_CONTROL_PATH}',
]
