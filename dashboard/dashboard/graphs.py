"""Reads PPO/SB3 TensorBoard scalar logs directly via tensorboard's own EventAccumulator instead of spawning a real `ten..."""

import glob
import os

from dashboard import ssh
from dashboard.config import LOG_DIR, SHEEP_LOG_CACHE_DIR, SHEEP_LOG_DIR

# rollout/ep_rew_mean first if present
PRIMARY_TAG = 'rollout/ep_rew_mean'


class TensorboardUnavailable(Exception):
    """Raised instead of letting ImportError propagate."""



def _event_accumulator_cls():
    # Imported lazily, not at module load time: this dashboard should run under either system python3 (has flask only) or the project's .venv (has flask AND tensorboard).
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        return EventAccumulator
    except ImportError:
        raise TensorboardUnavailable(
            "tensorboard isn't installed in this Python environment -- "
            'run the dashboard via .venv/bin/python3 -m dashboard instead '
            '(it has tensorboard), or pip install tensorboard here.')


def _read_scalars_from_dir(log_dir, fname):
    """{tag: [(step, value), ...]} merged across every log_dir/<fname>_* run folder, sorted by step."""

    EventAccumulator = _event_accumulator_cls()
    run_dirs = sorted(glob.glob(os.path.join(log_dir, f'{fname}_*')))
    merged = {}
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        try:
            ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get('scalars', []):
            points = [(s.step, s.value) for s in ea.Scalars(tag)]
            merged.setdefault(tag, []).extend(points)
    for tag in merged:
        merged[tag].sort(key=lambda p: p[0])
    return merged


def read_scalars(fname):
    """Local training's own scalars."""
    return _read_scalars_from_dir(LOG_DIR, fname)


def ordered_tags(scalars):
    """Tag names with PRIMARY_TAG first (if present), rest alphabetical."""

    tags = sorted(scalars.keys())
    if PRIMARY_TAG in tags:
        tags.remove(PRIMARY_TAG)
        tags.insert(0, PRIMARY_TAG)
    return tags


def sheep_sync_and_read_scalars(fname):
    """Downloads sheep's log_dir/<fname>_* run folders into SHEEP_LOG_CACHE_DIR (small files, cheap."""

    os.makedirs(SHEEP_LOG_CACHE_DIR, exist_ok=True)
    result = ssh.run(
        'sheep', f'ls -d {SHEEP_LOG_DIR}/{fname}_* 2>/dev/null', timeout_s=15)
    remote_dirs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for remote_dir in remote_dirs:
        ssh.scp_download('sheep', remote_dir, SHEEP_LOG_CACHE_DIR, recursive=True)
    return _read_scalars_from_dir(SHEEP_LOG_CACHE_DIR, fname)
