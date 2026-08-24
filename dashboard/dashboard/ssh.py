"""Thin subprocess wrappers around the system `ssh`/`scp`."""

import shlex
import subprocess
import time

from dashboard.config import SSH_OPTS


def _quote_remote_path(path):
    """shlex.quote() wraps the WHOLE path in single quotes, which is correct for shell-safety but silently breaks tilde expa..."""

    if path.startswith('~/'):
        return '$HOME/' + shlex.quote(path[2:])
    if path == '~':
        return '$HOME'
    return shlex.quote(path)


def _decode_partial(x):
    """subprocess.TimeoutExpired's own .stdout/.stderr are str-or-None when the process is killed before producing any outpu..."""

    if isinstance(x, bytes):
        return x.decode(errors='replace')
    return x or ''


class SshResult:
    def __init__(self, ok, stdout, stderr, returncode):
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_connection(host, timeout_s=18):
    """True if `ssh host "echo ok"` succeeds within timeout_s. 18s (2026-08-20, raised alongside SSH_CONNECT_TIMEOUT_S"""

    try:
        proc = subprocess.run(
            ['ssh'] + SSH_OPTS + [host, 'echo ok'],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return proc.returncode == 0 and proc.stdout.strip() == 'ok'
    except (subprocess.TimeoutExpired, OSError):
        return False


def run(host, remote_cmd, timeout_s=60, _retry=True):
    """Runs remote_cmd (a single shell string, already fully composed by the caller) over SSH and waits for it to finish."""

    try:
        proc = subprocess.run(
            ['ssh'] + SSH_OPTS + [host, remote_cmd],
            capture_output=True, text=True, timeout=timeout_s,
        )
        result = SshResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired as e:
        return SshResult(False, _decode_partial(e.stdout), _decode_partial(e.stderr) + '\n[dashboard] timed out', -1)
    except OSError as e:
        return SshResult(False, '', f'[dashboard] {e}', -1)
    if _retry and result.returncode == 255 and not result.stdout and not result.stderr:
        return run(host, remote_cmd, timeout_s=timeout_s, _retry=False)
    return result


def run_background(host, remote_cmd, log_path):
    """Starts remote_cmd on host as a detached background process ( survives the SSH session ending) with stdout+stderr redi..."""

    inner = remote_cmd.replace("'", "'\\''")  # escape for the outer bash -c '...'
    wrapped = f"nohup bash -c '{inner}' > {_quote_remote_path(log_path)} 2>&1 < /dev/null & echo $!"
    result = run(host, wrapped, timeout_s=15)
    if not result.ok:
        return None
    try:
        return int(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def is_running(host, pid):
    """True if a process with this PID still exists on host."""
    result = run(host, f'kill -0 {pid} 2>/dev/null && echo alive || echo dead', timeout_s=10)
    return result.ok and 'alive' in result.stdout


def kill(host, pid, wait_s=8.0, poll_interval_s=0.5):
    """Best-effort: TERM first, and if it's still around after a grace period, KILL."""

    pgid_result = run(host, f'ps -o pgid= -p {pid} 2>/dev/null', timeout_s=10)
    pgid = pgid_result.stdout.strip()
    group_target = f'-{pgid}' if pgid.isdigit() else None

    def _wait_dead(deadline_s):
        elapsed = 0.0
        while elapsed < deadline_s:
            if not is_running(host, pid):
                return True
            time.sleep(poll_interval_s)
            elapsed += poll_interval_s
        return not is_running(host, pid)

    def _signal(sig_name):
        """sig_name must always be an explicit signal (never a bare default)."""

        run(host, f'kill -{sig_name} {pid} 2>/dev/null', timeout_s=10)
        if group_target:
            run(host, f'kill -{sig_name} {group_target} 2>/dev/null', timeout_s=10)

    _signal('TERM')
    if _wait_dead(wait_s):
        return True
    _signal('KILL')
    return _wait_dead(wait_s)


def tail_log(host, log_path, max_bytes=20000):
    """Last max_bytes of a remote log file, for polling-based 'live' output display."""

    result = run(host, f'tail -c {max_bytes} {_quote_remote_path(log_path)} 2>/dev/null', timeout_s=10)
    return result.stdout if result.ok else ''


def remove_remote_file(host, path):
    """rm -f a single remote file (never a directory."""

    result = run(host, f'rm -f {_quote_remote_path(path)}', timeout_s=10)
    return result.ok


def scp_download(host, remote_path, local_path, recursive=False):
    """scp a file or directory from host to the local machine, creating local parent directories as needed."""

    import os
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    cmd = ['scp'] + SSH_OPTS
    if recursive:
        cmd.append('-r')
    cmd += [f'{host}:{remote_path}', local_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return SshResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired as e:
        return SshResult(False, _decode_partial(e.stdout), _decode_partial(e.stderr) + '\n[dashboard] timed out', -1)


def scp_upload(host, local_path, remote_path):
    """scp a local file TO host."""

    remote_dir = remote_path.rsplit('/', 1)[0] if '/' in remote_path else '.'
    run(host, f'mkdir -p {_quote_remote_path(remote_dir)}', timeout_s=10)
    cmd = ['scp'] + SSH_OPTS + [local_path, f'{host}:{remote_path}']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return SshResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired as e:
        return SshResult(False, _decode_partial(e.stdout), _decode_partial(e.stderr) + '\n[dashboard] timed out', -1)
