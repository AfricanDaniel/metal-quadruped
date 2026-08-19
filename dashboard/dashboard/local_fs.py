"""Local filesystem browsing of models/<folder>/PPO_<steps>_<fname>.zip.

The grouping/sorting helpers (group_fnames, group_checkpoints) are pure
functions over plain (name, mtime) pairs specifically so remote_fs.py can
reuse the exact same logic against `ssh ... find` output instead of
os.scandir -- one place defines "how a fname group / checkpoint list is
built and ordered", local and remote just supply different raw listings.
"""
import os
import re

from dashboard.config import MODELS_DIR, POLICIES_POSITION_DIR, POLICIES_TORQUE_DIR

# Extension is deliberately part of the pattern, not stripped/assumed --
# local models/ browsing matches .zip (SB3 checkpoints), Jetson policy
# browsing (remote_fs.py) matches .pt (exported TorchScript) using the
# SAME group_fnames/group_checkpoints functions with a different pattern.
CHECKPOINT_RE = re.compile(r'^PPO_(\d+)_(.+)\.zip$')
POLICY_PT_RE = re.compile(r'^PPO_(\d+)_(.+)\.pt$')


def group_fnames(file_entries, pattern=CHECKPOINT_RE):
    """file_entries: iterable of (name, mtime) for files directly inside
    one model folder. Returns [{fname, mtime, count}], sorted by each
    group's max checkpoint mtime, descending."""
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
    """file_entries: same shape as group_fnames(). Returns
    [{steps, basename, mtime}] for checkpoints matching this ONE fname,
    sorted by steps descending (matches the user's own PPO_9M > PPO_8M
    > ... example ordering). basename is the filename with its extension
    stripped, whatever that extension is (os.path.splitext, not a
    hardcoded '.zip')."""
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
    """'walk...'/'trot...'/'running...' -> Dog-Walk-v0, else Dog-Stand-v0.
    Covers every actual models/ folder name except the one genuinely
    ambiguous case, 'stand_walk' -- callers show this as an editable
    dropdown defaulted to this guess, never silently trust it for that
    one case.

    'trot' ADDED 2026-08-18 (user report: "view training button, mujoco
    no longer opens, the max slew box is also gone" on models/trot_home/
    trot_stand -- both symptoms trace to this function returning Dog-
    Stand-v0 for those folders, since they don't start with 'walk': the
    start_pose/max_slew boxes are gated on env_id=='Dog-Walk-v0' in
    checkpoints.html (explaining the missing box), and launch_view_
    training() was then invoked with the WRONG env-id for a WALK-trained
    checkpoint, mismatching its observation/action space and failing
    before MuJoCo ever opened (explaining the viewer not launching).
    trot_home/trot_stand are WALK-task checkpoints (see chatbot.md "user
    wants to push toward faster walking") under a new naming scheme this
    prefix check hadn't been updated for.

    'running' ADDED same day (models/running_home, models/running_stand --
    the bound-gait/genuine-running lineage, same WALK task, DogEnv's
    gait_style='bound' mode) -- pre-empting the identical bug for this
    new naming scheme rather than waiting for the same report again."""
    return 'Dog-Walk-v0' if folder.startswith(('walk', 'trot', 'running')) else 'Dog-Stand-v0'


def list_model_folders():
    """[{name, mtime}] under MODELS_DIR, sorted by the most-recently-
    modified .zip file inside each folder (not the folder's own mtime,
    which doesn't reliably update on file changes on every filesystem),
    descending."""
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
    """Removes a single local file -- shared by every local delete
    route (model checkpoints, policy .pt files, policy CSVs), all of
    which already have a path-builder function to hand this the exact
    path. True if the file is confirmed gone (including "was already
    gone"), matching remote_file removal's same not-an-error-if-
    missing semantics (ssh.remove_remote_file's rm -f)."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


def count_checkpoints_for_fname(fname):
    """Total checkpoints saved so far for this fname, across all model
    folders -- used by the Ongoing trainings list, which only knows a
    training's fname (not which folder it landed in), to show a
    checkpoint count without the user having to click in."""
    total = 0
    for entry in list_model_folders():
        for g in list_fnames(entry['name']):
            if g['fname'] == fname:
                total += g['count']
    return total


# --- Local: src/policies_{position,torque}/<task>/{policy,csv}/ --
# Same layout as Jetson's own policies_position/policies_torque (see
# remote_fs.py's JETSON_POLICY_GROUPS) -- these are exported here via
# "Export Policy" on the Models tab, then browsed/uploaded from here.
#
# WALK split into 6 lineage folders (2026-08-19, user request -- "sync/
# fix the jetson so it has the correct policy folders, just as we have on
# local [walk, run, trot] [home/stand]"): mirrors models/'s own folder
# names exactly (walk_home, walk_standing, trot_home, trot_stand,
# running_home, running_stand), so a checkpoint's model-dir tells you
# directly which policies_position/<lineage> it exports into -- see
# policy_actions.py's export_target_group()/_task_subfolder() for how a
# checkpoint's source folder maps to one of these. Previously a single
# flat 'walk' bucket mixed every lineage together, indistinguishable
# except by filename. Migrated the 16 existing .pt files (+ matching
# csv/) into their correct lineage folder by cross-referencing each
# embedded fname against the actual models/ folder it was found in (not
# guessed) -- ambiguous/pre-split legacy files (can't be traced to a
# specific models/ folder) fall back to walk_home, per direct user
# approval. STAND and both policies_torque groups are unaffected --
# STAND has no gait-lineage concept, and policies_torque/walk has never
# had any exports (confirmed empty on both local and Jetson).
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
    """[{name, mtime}] for the 4 fixed local policy groups, same shape/
    naming convention as remote_fs.list_jetson_groups."""
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
    """[{name, mtime}] of CSVs in the sibling csv/ folder whose name
    starts with this policy's basename -- same matching convention as
    remote_fs.list_jetson_csvs, newest first.

    Exact basename prefix, deliberately NOT stripping a trailing _vN
    (tried that 2026-08-19, reverted same day per direct user correction:
    "we should not ignore v number" -- v1/v2/v3 are each their OWN
    distinct identity, every one with its own family of further-suffixed
    files, e.g. v1_1/v1_2/v1_data/v1_tscs all belong to v1 specifically,
    v2_1/v2_2/etc. to v2 -- stripping _vN merged separate versions'
    CSVs together instead of keeping them apart.)"""
    g = _parse_policy_group_name(group_name)
    entries = _dir_file_entries(_policy_csv_dir(g))
    matches = [{'name': name, 'mtime': mtime} for name, mtime in entries if name.startswith(basename)]
    matches.sort(key=lambda d: d['mtime'], reverse=True)
    return matches


def policy_csv_path(group_name, csv_name):
    g = _parse_policy_group_name(group_name)
    return os.path.join(_policy_csv_dir(g), csv_name)


def find_folder_for_fname(fname):
    """Which models/ folder a given fname's checkpoints live under --
    scans every folder (there's only a handful) instead of requiring the
    caller to already know it. Needed to link FROM a training's graphs
    page (which only ever knows the fname -- e.g. reached from the
    Ongoing trainings list) TO its checkpoints page (which needs a
    folder too)."""
    for entry in list_model_folders():
        if any(g['fname'] == fname for g in list_fnames(entry['name'])):
            return entry['name']
    return None
