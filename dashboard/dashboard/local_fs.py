"""Local filesystem browsing of models/<folder>/PPO_<steps>_<fname>.zip. The grouping/sorting helpers (group_fnames, gro..."""

import os
import re

from dashboard.config import MODELS_DIR, POLICIES_POSITION_DIR, POLICIES_TORQUE_DIR

# Extension is deliberately part of the pattern, not stripped/assumed
CHECKPOINT_RE = re.compile(r'^PPO_(\d+)_(.+)\.zip$')
POLICY_PT_RE = re.compile(r'^PPO_(\d+)_(.+)\.pt$')


def group_fnames(file_entries, pattern=CHECKPOINT_RE):
    """file_entries: iterable of (name, mtime) for files directly inside one model folder."""

    groups = {}
    for name, mtime in file_entries:
        m = pattern.match(name)
        if not m:
            continue
        fname = m.group(2)
        g = groups.setdefault(fname, {'fname': fname, 'mtime': 0.0, 'count': 0})
        g['mtime'] = max(g['mtime'], mtime)
        g['count'] += 1
    result = list(groups.values())
    result.sort(key=lambda d: d['mtime'], reverse=True)
    return result


def group_checkpoints(file_entries, fname, pattern=CHECKPOINT_RE):
    """file_entries: same shape as group_fnames()."""

    checkpoints = []
    for name, mtime in file_entries:
        m = pattern.match(name)
        if not m or m.group(2) != fname:
            continue
        checkpoints.append({
            'steps': int(m.group(1)),
            'basename': os.path.splitext(name)[0],
            'mtime': mtime,
        })
    checkpoints.sort(key=lambda d: d['steps'], reverse=True)
    return checkpoints


def resolve_env_id(folder):
    """'walk...'/'trot...'/'running...' -> Dog-Walk-v0, else Dog-Stand-v0. Covers every actual models/ folder name except th..."""

    return 'Dog-Walk-v0' if folder.startswith(('walk', 'trot', 'running')) else 'Dog-Stand-v0'


def list_model_folders():
    """[{name, mtime}] under MODELS_DIR, sorted by the most-recently- modified .zip file inside each folder (not the folder'..."""

    folders = []
    if not os.path.isdir(MODELS_DIR):
        return folders
    for entry in os.scandir(MODELS_DIR):
        if not entry.is_dir():
            continue
        latest = 0.0
        with os.scandir(entry.path) as it:
            for f in it:
                if f.is_file() and f.name.endswith('.zip'):
                    latest = max(latest, f.stat().st_mtime)
        folders.append({'name': entry.name, 'mtime': latest})
    folders.sort(key=lambda d: d['mtime'], reverse=True)
    return folders


def _folder_file_entries(folder):
    folder_path = os.path.join(MODELS_DIR, folder)
    if not os.path.isdir(folder_path):
        return []
    with os.scandir(folder_path) as it:
        return [(f.name, f.stat().st_mtime) for f in it if f.is_file()]


def list_fnames(folder):
    return group_fnames(_folder_file_entries(folder))


def list_checkpoints(folder, fname):
    return group_checkpoints(_folder_file_entries(folder), fname)


def checkpoint_zip_path(folder, basename):
    return os.path.join(MODELS_DIR, folder, basename + '.zip')


def delete_local_file(path):
    """Removes a single local file."""

    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


def count_checkpoints_for_fname(fname):
    """Total checkpoints saved so far for this fname, across all model folders."""

    total = 0
    for entry in list_model_folders():
        for g in list_fnames(entry['name']):
            if g['fname'] == fname:
                total += g['count']
    return total


# --- Local: src/policies_{position,torque}/<task>/{policy,csv}/
LOCAL_POLICY_GROUPS = [
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


def _policies_base_dir(policies_dir):
    return POLICIES_POSITION_DIR if policies_dir == 'policies_position' else POLICIES_TORQUE_DIR


def _policy_dir(group):
    policies_dir, task = group
    return os.path.join(_policies_base_dir(policies_dir), task, 'policy')


def _policy_csv_dir(group):
    policies_dir, task = group
    return os.path.join(_policies_base_dir(policies_dir), task, 'csv')


def _dir_file_entries(path, suffix=None):
    if not os.path.isdir(path):
        return []
    with os.scandir(path) as it:
        return [(f.name, f.stat().st_mtime) for f in it if f.is_file() and (suffix is None or f.name.endswith(suffix))]


def _parse_policy_group_name(group_name):
    policies_dir, task = group_name.split('/', 1)
    return policies_dir, task


def list_policy_groups():
    """[{name, mtime}] for the 4 fixed local policy groups, same shape/ naming convention as remote_fs.list_jetson_groups."""

    groups = []
    for g in LOCAL_POLICY_GROUPS:
        entries = _dir_file_entries(_policy_dir(g), '.pt')
        latest = max((m for _, m in entries), default=0.0)
        groups.append({'name': f'{g[0]}/{g[1]}', 'mtime': latest})
    groups.sort(key=lambda d: d['mtime'], reverse=True)
    return groups


def list_policy_fnames(group_name):
    g = _parse_policy_group_name(group_name)
    entries = _dir_file_entries(_policy_dir(g), '.pt')
    return group_fnames(entries, pattern=POLICY_PT_RE)


def list_policy_checkpoints(group_name, fname):
    g = _parse_policy_group_name(group_name)
    entries = _dir_file_entries(_policy_dir(g), '.pt')
    return group_checkpoints(entries, fname, pattern=POLICY_PT_RE)


def policy_pt_path(group_name, basename):
    g = _parse_policy_group_name(group_name)
    return os.path.join(_policy_dir(g), basename + '.pt')


def list_policy_csvs(group_name, basename):
    """[{name, mtime}] of CSVs in the sibling csv/ folder whose name starts with this policy's basename."""

    g = _parse_policy_group_name(group_name)
    entries = _dir_file_entries(_policy_csv_dir(g))
    matches = [{'name': name, 'mtime': mtime} for name, mtime in entries if name.startswith(basename)]
    matches.sort(key=lambda d: d['mtime'], reverse=True)
    return matches


def policy_csv_path(group_name, csv_name):
    g = _parse_policy_group_name(group_name)
    return os.path.join(_policy_csv_dir(g), csv_name)


def find_folder_for_fname(fname):
    """Which models/ folder a given fname's checkpoints live under."""

    for entry in list_model_folders():
        if any(g['fname'] == fname for g in list_fnames(entry['name'])):
            return entry['name']
    return None
