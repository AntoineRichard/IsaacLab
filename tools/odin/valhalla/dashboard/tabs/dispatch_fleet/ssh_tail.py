# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the last N lines of a bundle's ssh-tail.log.

Truncates at SSH_TAIL_MAX_BYTES (64 KB) — never reads more into memory.
Failed jobs' logs are typically a few KB so the threshold rarely fires; it
exists to bound memory in pathological cases.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["SSH_TAIL_DEFAULT_LINES", "SSH_TAIL_MAX_BYTES", "load_ssh_tail"]


SSH_TAIL_DEFAULT_LINES = 50
SSH_TAIL_MAX_BYTES = 64 * 1024
_TRUNCATION_MARKER = "… (truncated to last 64 KB) …"


def load_ssh_tail(
    runs_root: Path,
    dispatch_id: str,
    run_id: str,
    lines: int = SSH_TAIL_DEFAULT_LINES,
) -> list[str]:
    """Return the last ``lines`` lines of the bundle's ssh-tail.log.

    Returns an empty list if the file is missing, unreadable, or the read
    raises any ``OSError`` (e.g., PermissionError). When the file exceeds
    ``SSH_TAIL_MAX_BYTES``, only the last 64 KB are read; the first
    (potentially partial) line is dropped and a truncation marker is
    prepended to the returned list.
    """
    path = Path(runs_root) / dispatch_id / run_id / "logs" / "ssh-tail.log"
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
        if size <= SSH_TAIL_MAX_BYTES:
            with open(path) as fh:
                all_lines = fh.read().splitlines()
            return all_lines[-lines:]
        # Large file: seek to the last SSH_TAIL_MAX_BYTES.
        with open(path, "rb") as fh:
            fh.seek(-SSH_TAIL_MAX_BYTES, os.SEEK_END)
            tail_bytes = fh.read()
        text = tail_bytes.decode("utf-8", errors="ignore")
        # Drop the (probably partial) first line.
        all_lines = text.split("\n")[1:]
        # Drop a trailing empty entry if present (file ends with \n).
        if all_lines and all_lines[-1] == "":
            all_lines = all_lines[:-1]
        if lines <= 1:
            return [_TRUNCATION_MARKER]
        return [_TRUNCATION_MARKER, *all_lines[-(lines - 1) :]]
    except OSError:
        return []
