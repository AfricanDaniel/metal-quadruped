"""Remote-filesystem equivalents of local_fs.py's browsing, over SSH --
reuses local_fs's group_fnames()/group_checkpoints() (same ordering
rules, same regex-driven parsing) against `find ... -printf` output
instead of os.scandir. Two structures are covered:

- Sheep's models/<folder>/PPO_<steps>_<fname>.zip -- identical layout to
  local models/, just remote (list_remote_model_folders/_fnames/
  _checkpoints).
- Jetson's src/policies_{position,torque}/{stand,walk}/policy/
  PPO_<steps>_<fname>.pt -- a fixed 4-group layout (list_jetson_groups/
  _fnames/_checkpoints), plus the matching csv/ sibling folder
  (list_jetson_csvs).
"""
import os
import shlex

from dashboard import local_fs, ssh
from dashboard.config import JETSON_WS_ROOT, SHEEP_WS_ROOT

# (policies_dir, task) -> the fixed policy groups Jetson browsing shows
# as its top-level "folders". policies_dir is 'policies_position' or
# 'policies_torque' (control-mode split); task is 'stand' or one of the 6
# WALK lineage names -- exact mirror of local_fs.LOCAL_POLICY_GROUPS, see
# that constant's own comment for the full 2026-08-19 reorganization
# story (both local and Jetson's on-disk layout were migrated together,
# confirmed identical .pt counts per folder on both machines).
JETSON_POLICY_GROUPS = [
    ('policies_position', 'stand'),
    ('policies_position', 'walk_home'),
    ('policies_position', 'walk_standing'),
    ('policies_position', 'trot_home'),
    ('policies_position', 'trot_stand'),
    ('policies_position', 'running_home'),
    ('policies_position', 'running_stand'),
    ('policies_torque', 'stand'),
    ('policies_torque', 'walk'),
]


def _remote_file_entries(host, remote_dir, name_glob):
    """[(name, mtime)] for files directly inside remote_dir (maxdepth 1)
    matching name_glob. Empty list (not an error) if the dir doesn't
    exist or the command fails -- callers treat "nothing here yet" and
    "couldn't reach it" the same way at the browsing level.

    remote_dir is deliberately NOT shell-quoted -- these are always
    dashboard-constructed paths starting with '~' (never raw user
    input), and shlex.quote() would wrap it in single quotes, which
    stops the remote shell from expanding '~' at all (quoted inside
    single quotes, tilde expansion never happens) -- silently made
    every remote_dir "not found" until caught here."""
    cmd = (f'find {remote_dir} -maxdepth 1 -name '
           f'{shlex.quote(name_glob)} -printf "%T@ %f\\n" 2>/dev/null')
    result = ssh.run(host, cmd, timeout_s=20)
    entries = []
    if not result.ok:
        return entries
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        mtime_str, _, name = line.partition(' ')
        try:
            entries.append((name, float(mtime_str)))
        except ValueError:
            continue
    return entries


# --- Sheep: models/<folder>/*.zip --------------------------------------

def list_remote_model_folders(host, ws_root=SHEEP_WS_ROOT):
    """[{name, mtime}], same shape/ordering as local_fs.list_model_folders."""
    cmd = (f'find {ws_root}/models -mindepth 2 -maxdepth 2 -name "*.zip" '
           '-printf "%T@ %P\\n" 2>/dev/null')
    result = ssh.run(host, cmd, timeout_s=20)
    latest = {}
    if result.ok:
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            mtime_str, _, rel = line.partition(' ')
            folder = rel.split('/', 1)[0]
            try:
                t = float(mtime_str)
            except ValueError:
                continue
            latest[folder] = max(latest.get(folder, 0.0), t)
    folders = [{'name': name, 'mtime': mtime} for name, mtime in latest.items()]
    folders.sort(key=lambda d: d['mtime'], reverse=True)
    return folders


def list_remote_fnames(host, folder, ws_root=SHEEP_WS_ROOT):
    entries = _remote_file_entries(host, f'{ws_root}/models/{folder}', '*.zip')
    return local_fs.group_fnames(entries)


def list_remote_checkpoints(host, folder, fname, ws_root=SHEEP_WS_ROOT):
    entries = _remote_file_entries(host, f'{ws_root}/models/{folder}', '*.zip')
    return local_fs.group_checkpoints(entries, fname)


def remote_checkpoint_zip_path(folder, basename, ws_root=SHEEP_WS_ROOT):
    return f'{ws_root}/models/{folder}/{basename}.zip'


def list_remote_fnames_with_local(host, folder, ws_root=SHEEP_WS_ROOT):
    """Same shape as list_remote_fnames, plus 'local_count' -- how many
    of that fname's checkpoints already exist under the local models/
    folder -- so the Sheep Models sub-tab can flag which fnames still
    need downloading. Computed from the SAME single remote fetch already
    used for grouping, not one extra round-trip per fname (a folder can
    have 30-40+ fnames -- N+1 remote calls just to render the page would
    make it visibly slow)."""
    entries = _remote_file_entries(host, f'{ws_root}/models/{folder}', '*.zip')
    groups = local_fs.group_fnames(entries)
    for g in groups:
        checkpoints = local_fs.group_checkpoints(entries, g['fname'])
        g['local_count'] = sum(
            1 for c in checkpoints
            if os.path.exists(local_fs.checkpoint_zip_path(folder, c['basename']))
        )
    return groups


def count_remote_checkpoints_for_fname(host, fname, ws_root=SHEEP_WS_ROOT):
    """Remote equivalent of local_fs.count_checkpoints_for_fname -- one
    SSH round-trip per remote model folder, cheap after the first thanks
    to ssh.py's connection multiplexing."""
    total = 0
    for entry in list_remote_model_folders(host, ws_root):
        for g in list_remote_fnames(host, entry['name'], ws_root):
            if g['fname'] == fname:
                total += g['count']
    return total


def find_remote_folder_for_fname(host, fname, ws_root=SHEEP_WS_ROOT):
    """Remote equivalent of local_fs.find_folder_for_fname -- one SSH
    round-trip per remote model folder (Sheep currently has under 10),
    but ssh.py's connection multiplexing makes repeat calls to the same
    host cheap after the first."""
    for entry in list_remote_model_folders(host, ws_root):
        if any(g['fname'] == fname for g in list_remote_fnames(host, entry['name'], ws_root)):
            return entry['name']
    return None


def list_remote_checkpoints_with_local(host, folder, fname, ws_root=SHEEP_WS_ROOT):
    """Same shape as list_remote_checkpoints, plus a 'local' bool per
    checkpoint -- whether that exact .zip already exists locally."""
    entries = _remote_file_entries(host, f'{ws_root}/models/{folder}', '*.zip')
    checkpoints = local_fs.group_checkpoints(entries, fname)
    for c in checkpoints:
        c['local'] = os.path.exists(local_fs.checkpoint_zip_path(folder, c['basename']))
    return checkpoints


# --- Jetson: src/policies_{position,torque}/{stand,walk}/policy/*.pt ---

def _group_dir(group, ws_root=JETSON_WS_ROOT):
    policies_dir, task = group
    return f'{ws_root}/src/{policies_dir}/{task}/policy'


def _group_csv_dir(group, ws_root=JETSON_WS_ROOT):
    policies_dir, task = group
    return f'{ws_root}/src/{policies_dir}/{task}/csv'


def list_jetson_groups(host, ws_root=JETSON_WS_ROOT):
    """[{name, mtime}] for the 4 fixed policy groups, 'name' formatted
    as '<policies_dir>/<task>' e.g. 'policies_position/walk' -- sorted
    by latest .pt mtime inside each, descending, same convention as
    every other folder-listing page in this dashboard."""
    groups = []
    for g in JETSON_POLICY_GROUPS:
        entries = _remote_file_entries(host, _group_dir(g, ws_root), '*.pt')
        latest = max((m for _, m in entries), default=0.0)
        groups.append({'name': f'{g[0]}/{g[1]}', 'mtime': latest})
    groups.sort(key=lambda d: d['mtime'], reverse=True)
    return groups


def _parse_group_name(group_name):
    policies_dir, task = group_name.split('/', 1)
    return policies_dir, task


def list_jetson_fnames(host, group_name, ws_root=JETSON_WS_ROOT):
    g = _parse_group_name(group_name)
    entries = _remote_file_entries(host, _group_dir(g, ws_root), '*.pt')
    return local_fs.group_fnames(entries, pattern=local_fs.POLICY_PT_RE)


def list_jetson_checkpoints(host, group_name, fname, ws_root=JETSON_WS_ROOT):
    g = _parse_group_name(group_name)
    entries = _remote_file_entries(host, _group_dir(g, ws_root), '*.pt')
    return local_fs.group_checkpoints(entries, fname, pattern=local_fs.POLICY_PT_RE)


def list_jetson_csvs(host, group_name, basename, ws_root=JETSON_WS_ROOT):
    """[{name, mtime}] of CSVs in the sibling csv/ folder whose name
    starts with this policy's basename (e.g. basename
    'PPO_33000000_walk_position_obshistory_v31' matches
    '..._v31_1.csv', '..._v31_2.csv', ...), newest first.

    Exact basename prefix -- see local_fs.list_policy_csvs's own comment
    for why this deliberately does NOT strip/ignore a trailing _vN (tried
    that 2026-08-19, reverted same day per direct user correction: each
    version number is its own distinct identity, not interchangeable)."""
    g = _parse_group_name(group_name)
    entries = _remote_file_entries(host, _group_csv_dir(g, ws_root), f'{basename}*.csv')
    entries.sort(key=lambda e: e[1], reverse=True)
    return [{'name': name, 'mtime': mtime} for name, mtime in entries]


def jetson_policy_pt_path(group_name, basename, ws_root=JETSON_WS_ROOT):
    g = _parse_group_name(group_name)
    return f'{_group_dir(g, ws_root)}/{basename}.pt'


def jetson_csv_path(group_name, csv_name, ws_root=JETSON_WS_ROOT):
    g = _parse_group_name(group_name)
    return f'{_group_csv_dir(g, ws_root)}/{csv_name}'


def basenames_on_jetson(host, group_name, ws_root=JETSON_WS_ROOT):
    """Basenames (no extension) of .pt files currently on the jetson for
    this group -- used by the Local Policies tab to flag which local
    policies still need uploading. Best-effort: if the jetson is
    unreachable this comes back empty, same as a genuinely-empty folder
    -- matching how every other remote-presence check in this dashboard
    already treats 'unreachable' and 'not there' the same way (fails
    toward 'needs action', never silently hides a real gap)."""
    g = _parse_group_name(group_name)
    entries = _remote_file_entries(host, _group_dir(g, ws_root), '*.pt')
    return {os.path.splitext(name)[0] for name, _ in entries}


# --- sheep: currently-running training processes -----------------------

import re as _re

_FNAME_ARG_RE = _re.compile(r'--fname\s+(\S+)')


def list_remote_running_trainings(host):
    """[{pid, fname, cmd}] -- unlike Jetson's/local training's PID-marker-
    file tracking, a remote training doesn't need one: it's already
    independently alive from the dashboard's own process (surviving a
    dashboard restart is automatic), so this just greps the remote host
    directly for any currently-running `dog_gym.train --train` process
    and parses --fname out of its own command line. No state to persist,
    always fresh."""
    result = ssh.run(host, "pgrep -af 'dog_gym.train --train'", timeout_s=15)
    trainings = []
    if not result.ok:
        return trainings
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, _, cmdline = line.partition(' ')
        m = _FNAME_ARG_RE.search(cmdline)
        if not pid_str.isdigit() or not m:
            continue
        trainings.append({'pid': int(pid_str), 'fname': m.group(1), 'cmd': cmdline})
    return trainings


# --- sheep: GPU status (2026-08-16, user request -- "one gpu might have
# less people using it, can you check if this is true") -----------------

def list_sheep_gpu_status(host):
    """[{index, util_percent, mem_used_mb, mem_total_mb}, ...] -- live
    `nvidia-smi` read, one row per physical GPU, so the Trainings launch
    form can show current load next to the CUDA_VISIBLE_DEVICES picker
    (see training_actions.py's CUDA_VISIBLE_DEVICES comment for why GPU
    selection is an env var prefix, not a train.py --flag). Always a
    fresh live query, never cached -- utilization changes constantly as
    other users' jobs come and go, a stale reading would defeat the whole
    point of picking the less-loaded GPU. Empty list (not an error) if
    nvidia-smi isn't reachable -- callers show "unknown" rather than fail
    the whole Trainings page over it."""
    cmd = 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits'
    result = ssh.run(host, cmd, timeout_s=15)
    gpus = []
    if not result.ok:
        return gpus
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) != 4:
            continue
        index, util, mem_used, mem_total = parts
        try:
            gpus.append({
                'index': int(index),
                'util_percent': int(util),
                'mem_used_mb': int(mem_used),
                'mem_total_mb': int(mem_total),
            })
        except ValueError:
            continue
    gpus.sort(key=lambda g: g['index'])
    return gpus
