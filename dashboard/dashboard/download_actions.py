"""Background download jobs for Sheep's group-download button.

A group download can be dozens of checkpoints, each its own scp call --
running that synchronously in the request thread would either block the
browser for a long time or (as happened in practice: a group landing as
"1/10 checkpoints" with no visibility into why) look like a silent
partial failure. Runs in a background thread instead, tracked by
(folder, fname) so the checkpoints page can poll and show a real
progress bar -- same 'shared state + poll' shape as Jetson's build/
deploy (procs.py) and local trainings (training_actions.py), just an
in-process thread instead of a subprocess/remote PID, since these are
lightweight scp calls that don't need to survive a dashboard restart.
"""
import threading

from dashboard import local_fs, remote_fs, ssh

_lock = threading.Lock()
_jobs = {}  # (folder, fname) -> dict


def _download_one_with_retry(host, folder, basename, attempts=2):
    """One retry on failure -- covers the transient case (a single SSH
    handshake/scp hiccup under load) without masking a real, persistent
    problem (e.g. the file genuinely not existing remotely)."""
    remote_path = remote_fs.remote_checkpoint_zip_path(folder, basename)
    local_path = local_fs.checkpoint_zip_path(folder, basename)
    for _ in range(attempts):
        result = ssh.scp_download(host, remote_path, local_path)
        if result.ok:
            return True
    return False


def start_group_download(host, folder, fname):
    key = (folder, fname)
    with _lock:
        job = _jobs.get(key)
        if job and not job['finished']:
            return False, f'A download for "{fname}" is already in progress.'

    checkpoints = remote_fs.list_remote_checkpoints(host, folder, fname)
    if not checkpoints:
        return False, f'No checkpoints found for "{fname}".'

    with _lock:
        _jobs[key] = {
            'total': len(checkpoints), 'done': 0, 'ok': 0,
            'current': None, 'finished': False,
        }

    def _run():
        for c in checkpoints:
            with _lock:
                _jobs[key]['current'] = c['basename']
            ok = _download_one_with_retry(host, folder, c['basename'])
            with _lock:
                _jobs[key]['done'] += 1
                _jobs[key]['ok'] += 1 if ok else 0
        with _lock:
            _jobs[key]['current'] = None
            _jobs[key]['finished'] = True

    threading.Thread(target=_run, daemon=True).start()
    return True, f'Downloading {len(checkpoints)} checkpoint(s) for "{fname}"...'


def get_group_download_status(folder, fname):
    """A finished job is removed the FIRST time it's read as finished --
    without this, the job dict sits in _jobs forever, so every later
    page load (or a reload right after this call) keeps reading the
    exact same 'finished: true' state and the checkpoints.html poller
    re-triggers its 'Go to download?' confirm every single time,
    including right after the user dismisses it (the dismiss path
    itself reloads the page, which re-polls, which sees the same
    still-finished job again) -- confirmed this is exactly what
    produced the reported non-stop popup loop."""
    key = (folder, fname)
    with _lock:
        job = _jobs.get(key)
        if job is None:
            return {'total': 0, 'done': 0, 'ok': 0, 'current': None, 'finished': True}
        result = dict(job)
        if job['finished']:
            del _jobs[key]
        return result
