"""Thin subprocess wrappers around the system `ssh`/`scp` -- matches how
every SSH interaction earlier in this project's own debugging sessions
was actually done (`ssh -o BatchMode=yes -o ConnectTimeout=... host
"cmd"`), rather than adding a new `paramiko` dependency that isn't
actually installed in this venv (only its type stubs are)."""
import shlex
import subprocess
import time

from dashboard.config import SSH_OPTS


def _quote_remote_path(path):
    """shlex.quote() wraps the WHOLE path in single quotes, which is
    correct for shell-safety but silently breaks tilde expansion for any
    path starting with '~/' -- inside single quotes bash treats '~' as a
    literal character, not $HOME, so e.g. `> '~/foo/bar.log'` fails with
    "No such file or directory" (it looks for a literal subdirectory
    named '~'). Confirmed this directly: a background job redirecting to
    a single-quoted '~/...' path never starts at all (the redirect setup
    fails before the command executes), not just "logs to the wrong
    place". Every remote log path this module handles is built from
    JETSON_WS_ROOT/SHEEP_WS_ROOT, which are '~/...', so this isn't a
    hypothetical edge case. Fix: expand the leading '~/' to the shell
    variable $HOME OUTSIDE the quotes (so it still expands), and quote
    only the remainder."""
    if path.startswith('~/'):
        return '$HOME/' + shlex.quote(path[2:])
    if path == '~':
        return '$HOME'
    return shlex.quote(path)


def _decode_partial(x):
    """subprocess.TimeoutExpired's own .stdout/.stderr are str-or-None
    when the process is killed before producing any output, but come
    back as raw BYTES (even though text=True was passed to
    subprocess.run) whenever some output was already captured before
    the timeout hit -- confirmed directly: a command that echoes
    something and then hangs raises TimeoutExpired with .stdout as
    `bytes`, not `str`. Every caller here immediately does `e.stderr +
    '...'`, which crashes with "can't concat str to bytes" the moment a
    timed-out remote command had already printed anything (e.g. a ROS
    node logging 'waiting for service...' while stuck) -- confirmed this
    exact crash for real via /jetson/basics's set_home button."""
    if isinstance(x, bytes):
        return x.decode(errors='replace')
    return x or ''


class SshResult:
    def __init__(self, ok, stdout, stderr, returncode):
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_connection(host, timeout_s=8):
    """True if `ssh host "echo ok"` succeeds within timeout_s."""
    try:
        proc = subprocess.run(
            ['ssh'] + SSH_OPTS + [host, 'echo ok'],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return proc.returncode == 0 and proc.stdout.strip() == 'ok'
    except (subprocess.TimeoutExpired, OSError):
        return False


def run(host, remote_cmd, timeout_s=60, _retry=True):
    """Runs remote_cmd (a single shell string, already fully composed by
    the caller) over SSH and waits for it to finish. Returns SshResult.

    Retries ONCE on a specific failure signature: returncode 255 with
    BOTH stdout and stderr empty. Confirmed this is a real, reproducible
    failure mode of ssh's own connection multiplexing (ControlMaster,
    see config.py's SSH_OPTS) under concurrent load -- Jetson pages
    poll /jetson/status every 1.5s for as long as they're open, all
    reusing the SAME multiplexed connection; watched directly via `ssh
    -v` that a manual command's session was killed mid-command
    (`exit-signal`) while two of those poll-driven sessions were active
    on the same connection at that exact moment. This is exactly what
    surfaced as "Failed to reset motors -- check the Jetson directly"
    with no actual problem on the Jetson at all. Retrying is safe here
    specifically because empty stdout AND stderr rules out both a real
    remote-command failure (which prints something) and a genuinely
    unreachable host (ssh itself prints a message like "Connection
    timed out" to stderr) -- this signature is specific to the
    session having been cut out from under it, not any of that."""
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
    """Starts remote_cmd on host as a detached background process (
    survives the SSH session ending) with stdout+stderr redirected to
    log_path on the REMOTE machine, and returns that process's remote
    PID (int) -- or None if this couldn't be determined.

    Approach: `nohup bash -c '<cmd>' > log 2>&1 & echo $!` inside one SSH
    call. The outer `ssh host "..."` command returns almost immediately
    (all it does remotely is background the real work and echo a PID),
    while the actual process keeps running on the remote after this SSH
    session closes. The returned PID is what a later `run(host,
    f"kill {pid}")` targets to stop it -- killing the LOCAL ssh client
    process does not reliably kill whatever it started on the far end,
    which is why this captures the real remote PID instead."""
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
    """Best-effort: TERM first, and if it's still around after a grace
    period, KILL. Returns True if the process is confirmed gone.

    Two real problems this addresses, both confirmed directly against
    sheep's actual running trainings (--env-type subproc, the default,
    spawns dozens of SubprocVecEnv worker processes per training):

    1. Killing only `pid` (the main process) leaves every worker
       orphaned -- they eventually notice their parent is gone and exit
       on their own, but confirmed this can take up to ~a minute under
       load, not "shortly after". Also kill the process GROUP (`-pgid`,
       captured up front while `pid` is still alive to query) so every
       worker gets SIGKILL directly instead of waiting for a broken pipe
       to notice. Safe to target the whole group here specifically
       because every training this dashboard launches (local or remote)
       runs as its own isolated nohup'd background job/session, not a
       job sharing a process group with anything else the user is
       doing -- confirmed directly (ps -o pgid,sid showed the group's
       PGID/SID both equal to the job's own leader PID, not the login
       shell's).
    2. A single fixed ~1s grace period was still too short: with several
       such trainings running at once (confirmed: sheep with 3
       simultaneous 32-worker trainings, ~96 processes contending for
       the CPU) even SIGKILL can take several real seconds to finish
       tearing everything down under that much scheduler contention -- a
       "Failed to stop" training that a ~2s total grace period reported
       as still alive had, in fact, fully exited. Polling repeatedly
       instead of a single fixed-delay check returns as soon as it's
       actually dead (fast in the common, lightly-loaded case) without
       giving up too early under heavy load."""
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
        """sig_name must always be an explicit signal (never a bare
        default) -- `kill -3940671` (no signal flag before a negative
        pgid) is misparsed as "send signal number 3940671", not "send
        SIGTERM to process group 3940671", and just fails silently
        (confirmed directly: this exact form returned nonzero while
        `kill -TERM -3940671` and `kill -9 -3940671` both worked)."""
        run(host, f'kill -{sig_name} {pid} 2>/dev/null', timeout_s=10)
        if group_target:
            run(host, f'kill -{sig_name} {group_target} 2>/dev/null', timeout_s=10)

    _signal('TERM')
    if _wait_dead(wait_s):
        return True
    _signal('KILL')
    return _wait_dead(wait_s)


def tail_log(host, log_path, max_bytes=20000):
    """Last max_bytes of a remote log file, for polling-based 'live'
    output display -- empty string if the file doesn't exist yet."""
    result = run(host, f'tail -c {max_bytes} {_quote_remote_path(log_path)} 2>/dev/null', timeout_s=10)
    return result.stdout if result.ok else ''


def remove_remote_file(host, path):
    """rm -f a single remote file (never a directory -- no -r). True if
    the SSH call itself succeeded; rm -f doesn't error on a path that's
    already gone, so this is True even for a delete-of-something-
    already-deleted, same as os.remove-wrapped-in-try/except would be
    for the local equivalent."""
    result = run(host, f'rm -f {_quote_remote_path(path)}', timeout_s=10)
    return result.ok


def scp_download(host, remote_path, local_path, recursive=False):
    """scp a file or directory from host to the local machine, creating
    local parent directories as needed."""
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
    """scp a local file TO host -- mirror of scp_download, just in the
    other direction. mkdir -p's the remote parent dir first since scp
    itself won't create it."""
    remote_dir = remote_path.rsplit('/', 1)[0] if '/' in remote_path else '.'
    run(host, f'mkdir -p {_quote_remote_path(remote_dir)}', timeout_s=10)
    cmd = ['scp'] + SSH_OPTS + [local_path, f'{host}:{remote_path}']
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return SshResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode)
    except subprocess.TimeoutExpired as e:
        return SshResult(False, _decode_partial(e.stdout), _decode_partial(e.stderr) + '\n[dashboard] timed out', -1)
