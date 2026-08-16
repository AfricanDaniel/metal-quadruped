"""Reads PPO/SB3 TensorBoard scalar logs directly (via tensorboard's own
EventAccumulator, already installed -- no new dependency) instead of
spawning a real `tensorboard` server -- see the plan's "Graphing
approach" note for why. Relies on train.py's tb_log_name=fname fix
(dog_gym/train.py) to reliably find "this fname's own run folders" via a
glob -- pre-existing logs from before that fix used SB3's own generic
auto-incrementing {algo}_{n} naming and can't be reliably attributed to
one fname, so they simply won't show anything here (not a bug, a real
limit of what's recoverable from the old naming scheme).
"""
import glob
import os

from dashboard import ssh
from dashboard.config import LOG_DIR, SHEEP_LOG_CACHE_DIR, SHEEP_LOG_DIR

# rollout/ep_rew_mean first if present -- the "reward" graph the user
# actually asked for; everything else SB3 happens to log comes along for
# free from the same parse, shown after it.
PRIMARY_TAG = 'rollout/ep_rew_mean'


class TensorboardUnavailable(Exception):
    """Raised instead of letting ImportError propagate -- see the lazy
    import below."""


def _event_accumulator_cls():
    # Imported LAZILY, not at module load time: this dashboard is meant
    # to run fine under EITHER system python3 (has flask, confirmed
    # working) OR the project's .venv (has flask AND tensorboard) --
    # confirmed directly that system python3 does NOT have tensorboard.
    # An eager top-level import would crash the WHOLE app at startup
    # under system python3, breaking graph-less pages too, just because
    # one optional feature needs an extra package. Deferring it here
    # means only an actual attempt to read graphs needs tensorboard,
    # with a clear error instead of the app failing to start at all.
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        return EventAccumulator
    except ImportError:
        raise TensorboardUnavailable(
            "tensorboard isn't installed in this Python environment -- "
            'run the dashboard via .venv/bin/python3 -m dashboard instead '
            '(it has tensorboard), or pip install tensorboard here.')


def _read_scalars_from_dir(log_dir, fname):
    """{tag: [(step, value), ...]} merged across every log_dir/<fname>_*
    run folder, sorted by step. Empty dict (not an error) if none exist
    yet or the folders can't be parsed -- callers show 'no graphs yet'
    rather than crashing, e.g. for a training that only just started and
    hasn't flushed its first event file yet. Raises TensorboardUnavailable
    (only once real work is needed, not at import time) if this Python
    environment doesn't have tensorboard."""
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
    """Tag names with PRIMARY_TAG first (if present), rest alphabetical
    -- so the actual reward curve is always the first graph on the page,
    not wherever it happened to fall alphabetically."""
    tags = sorted(scalars.keys())
    if PRIMARY_TAG in tags:
        tags.remove(PRIMARY_TAG)
        tags.insert(0, PRIMARY_TAG)
    return tags


def sheep_sync_and_read_scalars(fname):
    """Downloads sheep's log_dir/<fname>_* run folders into
    SHEEP_LOG_CACHE_DIR (small files, cheap -- re-synced every time this
    is called, not cached-forever, since a still-running remote training
    keeps appending to them), then reads scalars from that local cache
    the same way as a local run."""
    os.makedirs(SHEEP_LOG_CACHE_DIR, exist_ok=True)
    result = ssh.run(
        'sheep', f'ls -d {SHEEP_LOG_DIR}/{fname}_* 2>/dev/null', timeout_s=15)
    remote_dirs = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for remote_dir in remote_dirs:
        ssh.scp_download('sheep', remote_dir, SHEEP_LOG_CACHE_DIR, recursive=True)
    return _read_scalars_from_dir(SHEEP_LOG_CACHE_DIR, fname)
