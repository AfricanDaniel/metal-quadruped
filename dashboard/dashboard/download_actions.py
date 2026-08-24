"""Background download jobs for Sheep's group-download button. A group download can be dozens of checkpoints, each its o..."""

import threading

from dashboard import local_fs, remote_fs, ssh

_lock = threading.Lock()
_jobs = {}  # (folder, fname) -> dict


def _download_one_with_retry(host, folder, basename, attempts=2):
    """One retry on failure."""

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
    """A finished job is removed the FIRST time it's read as finished."""

    key = (folder, fname)
    with _lock:
        job = _jobs.get(key)
        if job is None:
            return {'total': 0, 'done': 0, 'ok': 0, 'current': None, 'finished': True}
        result = dict(job)
        if job['finished']:
            del _jobs[key]
        return result
