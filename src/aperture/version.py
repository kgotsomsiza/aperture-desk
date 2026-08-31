"""Which build of the desk is running.

The competition measures one account over four sessions while the code is still
being worked on, so "which system produced this P&L" is a real question and the
honest answer should not require anyone to reconstruct it from timestamps.

The version changes only when the process restarts -- a container cannot swap
its own code mid-run. So a single event recorded at startup partitions the whole
audit trail: every decision after it, until the next startup event, was made by
that build. That is cheaper and clearer than stamping every line.

Resolution order, most to least trustworthy:

1. ``APERTURE_VERSION``, baked into the image at build time. This is the real
   answer in production, where there is no git repository inside the container.
2. ``git rev-parse``, for a developer running from a checkout.
3. ``"unknown"``. Never raises and never blocks a cycle -- provenance is
   evidence about trading, not a precondition for it.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache

UNKNOWN = "unknown"


@lru_cache(maxsize=1)
def desk_version() -> str:
    """A short, stable identifier for the running build."""
    stamped = (os.environ.get("APERTURE_VERSION") or "").strip()
    if stamped and stamped != "$GIT_SHA":
        return stamped[:40]

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )
    except Exception:  # noqa: BLE001 - no git, no repo, no shell: all fine
        return UNKNOWN
    if out.returncode != 0:
        return UNKNOWN
    return (out.stdout.strip() or UNKNOWN)[:40]


def is_known() -> bool:
    return desk_version() != UNKNOWN
