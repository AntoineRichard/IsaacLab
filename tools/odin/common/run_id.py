# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run ID format for Odin bundles.

Format: ``<framework>_<backend>_<task>_<date>_seed<seed>``
Example: ``rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42``
"""

from __future__ import annotations

from datetime import datetime, timezone

_FRAMEWORKS_PATH = {"rsl_rl": "rsl-rl", "skrl": "skrl"}
_FRAMEWORKS_JSON = {"rsl-rl": "rsl_rl", "skrl": "skrl"}
_BACKENDS = {"physx", "newton"}


def compute_run_id(
    framework: str,
    backend: str,
    task: str,
    seed: int,
    now: datetime | None = None,
) -> str:
    """Compute the canonical run_id for a new Odin run.

    Args:
        framework: Learning framework, e.g. ``"rsl_rl"`` or ``"skrl"``.
        backend: Physics backend, ``"physx"`` or ``"newton"``.
        task: Gym task ID, e.g. ``"Isaac-Ant-Direct-v0"``.
        seed: Integer seed.
        now: UTC datetime for the run-start timestamp. Defaults to
            :func:`datetime.now` with ``timezone.utc``.

    Returns:
        The run_id string.

    Raises:
        ValueError: if ``framework`` or ``backend`` is not recognised.
    """
    if framework not in _FRAMEWORKS_PATH:
        raise ValueError(f"Unknown framework {framework!r}; expected one of {sorted(_FRAMEWORKS_PATH)}")
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; expected 'physx' or 'newton'")
    if now is None:
        now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    fw_path = _FRAMEWORKS_PATH[framework]
    return f"{fw_path}_{backend}_{task}_{stamp}_seed{seed}"


def parse_run_id(run_id: str) -> dict:
    """Parse a run_id back into its components.

    Returns:
        Dict with keys ``framework`` (JSON form), ``backend``, ``task``,
        ``date``, ``seed``.

    Raises:
        ValueError: if the run_id is malformed.
    """
    parts = run_id.split("_")
    if len(parts) < 5:
        raise ValueError(f"Malformed run_id {run_id!r}: expected at least 5 '_'-separated parts")
    fw_path = parts[0]
    backend = parts[1]
    seed_token = parts[-1]
    date = parts[-2]
    task = "_".join(parts[2:-2])
    if fw_path not in _FRAMEWORKS_JSON:
        raise ValueError(f"Unknown framework token {fw_path!r}")
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend token {backend!r}")
    if not seed_token.startswith("seed"):
        raise ValueError(f"Expected seed token 'seed<int>', got {seed_token!r}")
    try:
        seed = int(seed_token[4:])
    except ValueError as e:
        raise ValueError(f"Invalid seed integer in {seed_token!r}") from e
    return {
        "framework": _FRAMEWORKS_JSON[fw_path],
        "backend": backend,
        "task": task,
        "date": date,
        "seed": seed,
    }
