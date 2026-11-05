"""Harness build version.

Bump this on **every** change to the harness (see AGENTS.md):

- patch: bug fixes, UI/copy tweaks, no behaviour contract change
- minor: new tools, new API endpoints, new persisted fields (backwards compatible)
- major: breaking change to config, state schema, or the board model

The version is stamped into `state.json` meta at startup, so a dashboard can
show which build is actually running. That matters because the stack runs from
a bind-mounted volume: a container that was not rebuilt keeps serving an old
build against new data, and without a visible version that is invisible.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

__version__ = "1.9.1"


def git_sha() -> str | None:
    """Short git SHA when running from a checkout; None in a built image."""
    if os.environ.get("HARNESS_GIT_SHA"):
        return os.environ["HARNESS_GIT_SHA"][:12]
    root = Path(__file__).resolve().parent.parent
    if not (root / ".git").is_dir():
        return None
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True)
    except Exception:  # noqa: BLE001 - version metadata must never break startup
        return None
    return out.stdout.strip() or None


def version_string() -> str:
    """`1.2.0` in a container, `1.2.0+abc1234` from a git checkout."""
    sha = git_sha()
    return f"{__version__}+{sha}" if sha else __version__
