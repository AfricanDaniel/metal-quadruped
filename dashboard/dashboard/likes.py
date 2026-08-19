"""Persists "liked" models (2026-08-19, user request -- a heart-shaped
like button on each checkpoint, plus a Liked section listing them all for
easy MuJoCo viewing later) across dashboard restarts -- unlike procs.py's
_state (in-memory only, fine for ephemeral "last form values" but not for
something the user explicitly wants to keep long-term), this is written
to a small JSON file on disk.

A "like" is keyed by (host, folder, fname, basename) -- host is 'local'
or 'sheep', matching this dashboard's own two checkpoint-hosting tabs;
the same basename can exist in both places (a Sheep checkpoint that's
since been downloaded locally too), so host is part of the identity, not
just a display label.
"""
import json
import os
import threading
import time

from dashboard.config import WS_ROOT

_LIKES_PATH = os.path.join(WS_ROOT, '.dashboard_likes.json')
_lock = threading.Lock()


def _key(host, folder, fname, basename):
    return f'{host}:{folder}:{fname}:{basename}'


def _load():
    if not os.path.exists(_LIKES_PATH):
        return {}
    try:
        with open(_LIKES_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(data):
    # Atomic write (tmp + os.replace) -- avoids a half-written file if the
    # dashboard is killed mid-save, same reasoning as every other on-disk
    # marker in this app (training_actions.py's own run.json markers).
    tmp = _LIKES_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _LIKES_PATH)


def is_liked(host, folder, fname, basename):
    with _lock:
        return _key(host, folder, fname, basename) in _load()


def toggle_like(host, folder, fname, basename):
    """Returns the NEW liked state (bool)."""
    with _lock:
        data = _load()
        key = _key(host, folder, fname, basename)
        if key in data:
            del data[key]
            _save(data)
            return False
        data[key] = {
            'host': host, 'folder': folder, 'fname': fname, 'basename': basename,
            'liked_at': time.time(),
        }
        _save(data)
        return True


def list_likes():
    """[{host, folder, fname, basename, liked_at}], most recently liked
    first."""
    with _lock:
        data = _load()
    items = list(data.values())
    items.sort(key=lambda d: d.get('liked_at', 0), reverse=True)
    return items
