# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the last N lines of a bundle's failure log.

Tries a prioritized list of candidate log files inside ``<bundle>/logs/``
and returns the tail of the first one that exists with non-empty content.
The legacy ``ssh-tail.log`` was generated only by the legacy-PTY dispatch
path; the current detached path produces ``training.stderr.log`` (where
the actual training Python traceback lives) plus a couple of wrapper logs.

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

# Priority order: legacy PTY-mode log first (still primary when present),
# then the detached-mode logs ranked by where Python tracebacks actually
# end up. ``training.stderr.log`` is by far the most useful for diagnosing
# a hugin_crash: the benchmark script's stderr (including any uncaught
# exception traceback) is captured there.
_CANDIDATE_LOG_NAMES = (
    "ssh-tail.log",
    "training.stderr.log",
    "hugin-stderr.log",
    "startup.stderr.log",
)


def load_ssh_tail(
    runs_root: Path,
    dispatch_id: str,
    run_id: str,
    lines: int = SSH_TAIL_DEFAULT_LINES,
) -> list[str]:
    """Return the last ``lines`` lines of the bundle's failure log.

    Tries :data:`_CANDIDATE_LOG_NAMES` in order and returns the tail of the
    first existing file with non-empty content. Returns an empty list if
    none exist or all are empty / unreadable. When the chosen file exceeds
    ``SSH_TAIL_MAX_BYTES``, only the last 64 KB are read; the first
    (potentially partial) line is dropped and a truncation marker is
    prepended to the returned list.
    """
    bundle_logs = Path(runs_root) / dispatch_id / run_id / "logs"
    for name in _CANDIDATE_LOG_NAMES:
        path = bundle_logs / name
        if not path.exists():
            continue
        try:
            size = path.stat().st_size
            if size == 0:
                continue
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
            continue
    return []
