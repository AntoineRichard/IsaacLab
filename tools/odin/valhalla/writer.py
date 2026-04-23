# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic writer for ``<dispatch_dir>/aggregate.json``."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

__all__ = ["write_aggregate"]

_FILENAME = "aggregate.json"


def write_aggregate(
    dispatch_dir: Path,
    aggregate: dict,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically write ``<dispatch_dir>/aggregate.json`` and return its path.

    Writes to a sibling temporary file then ``os.replace``\\ s over the
    final path, so a concurrent reader never observes a truncated file.

    Args:
        dispatch_dir: Target ``odin_runs/<dispatch_id>/`` directory.
        aggregate: Aggregate dict to serialize (matches T4.1 schema v1.0).
        overwrite: When ``False``, raises :class:`FileExistsError` if
            ``aggregate.json`` already exists. Default ``True``.

    Returns:
        Path to the written file.

    Raises:
        FileExistsError: When ``overwrite=False`` and the target exists.
    """
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    final_path = dispatch_dir / _FILENAME
    if not overwrite and final_path.exists():
        raise FileExistsError(f"{final_path} already exists (pass overwrite=True to replace)")

    fd, tmp_path_str = tempfile.mkstemp(prefix=".aggregate_", suffix=".json.tmp", dir=str(dispatch_dir))
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(aggregate, fh, indent=2, sort_keys=False)
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return final_path
