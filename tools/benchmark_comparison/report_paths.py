# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict path validation for simulator-free benchmark reporting."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from .models import RunSet

_COMPONENT_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]*")


def validate_artifact_path(value: str, run_set: RunSet, artifact_root: Path) -> tuple[str, ...]:
    """Validate and return a canonical run-set-relative artifact path.

    Args:
        value: POSIX artifact path relative to ``artifact_root``.
        run_set: Run set that must own the artifact.
        artifact_root: Root containing both benchmark run sets.

    Returns:
        The validated POSIX path components.

    Raises:
        ValueError: If the path is ambiguous, noncanonical, or escapes its run set.
    """
    if not isinstance(value, str) or not value or "\\" in value or "%" in value:
        raise ValueError(f"artifact path is not canonical POSIX text: {value!r}")
    path = PurePosixPath(value)
    parts = path.parts
    if (
        path.is_absolute()
        or path.as_posix() != value
        or len(parts) not in {2, 3}
        or any(part in {"", ".", ".."} or _COMPONENT_PATTERN.fullmatch(part) is None for part in parts)
    ):
        raise ValueError(f"artifact path is not canonical POSIX text: {value!r}")
    if parts[0] != run_set.value:
        raise ValueError(f"artifact path does not belong to run set {run_set.value}: {value!r}")

    resolved_root = artifact_root.resolve()
    selected_root = (resolved_root / run_set.value).resolve(strict=False)
    if not _is_within(selected_root, resolved_root):
        raise ValueError(f"artifact path run-set root escapes artifact root: {value!r}")
    target = (resolved_root / Path(*parts)).resolve(strict=False)
    if not _is_within(target, selected_root):
        raise ValueError(f"artifact path escapes selected run set: {value!r}")
    return parts


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is ``root`` or one of its descendants."""
    return path == root or root in path.parents
