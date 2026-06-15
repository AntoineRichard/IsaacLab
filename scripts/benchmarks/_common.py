# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for the unified benchmark entry scripts."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType


def get_backend_type(cli_backend: str) -> str:
    """Map CLI backend names to canonical backend type strings.

    Args:
        cli_backend: The backend name from CLI arguments (legacy long-form or
            short canonical form).

    Returns:
        Canonical backend type string; defaults to ``"omniperf"`` for unknown values.
    """
    mapping = {
        "OmniPerfKPIFile": "omniperf",
        "JSONFileMetrics": "json",
        "OsmoKPIFile": "osmo",
        "LocalLogMetrics": "json",
        "omniperf": "omniperf",
        "json": "json",
        "osmo": "osmo",
        "summary": "summary",
        "schema": "schema",
    }
    return mapping.get(cli_backend, "omniperf")


def get_backend_types(cli_backend: str) -> list[str]:
    """Split a comma-separated ``--benchmark_backend`` value into canonical backend types.

    Each token is normalized with :func:`get_backend_type` (so legacy long-form aliases and
    unknown-token fallback to ``"omniperf"`` still apply). Order is preserved and duplicates
    are removed. An empty input yields ``["omniperf"]``.

    Args:
        cli_backend: Raw ``--benchmark_backend`` value, e.g. ``"schema"`` or ``"schema,omniperf"``.

    Returns:
        Ordered, de-duplicated list of canonical backend type strings.
    """
    out: list[str] = []
    for tok in cli_backend.split(","):
        tok = tok.strip()
        if not tok:
            continue
        canon = get_backend_type(tok)
        if canon not in out:
            out.append(canon)
    return out or ["omniperf"]


def preset_tokens(folded: list[str]) -> list[str]:
    """Extract active preset tokens from a folded Hydra argument list.

    Searches *folded* for a ``presets=<value>`` argument and returns its
    comma-split tokens.  Returns an empty list when no ``presets=`` argument
    is present or its value is empty.

    Args:
        folded: Folded Hydra argument list (output of
            :func:`~isaaclab_tasks.utils.fold_preset_tokens`).

    Returns:
        List of active preset token strings.
    """
    for arg in folded:
        if arg.startswith("presets="):
            value = arg.split("=", 1)[1]
            return value.split(",") if value else []
    return []


def import_module_from_path(module_name: str, module_path) -> ModuleType:
    """Import a module from an explicit file path without relying on package resolution.

    Loads the module by absolute path via ``importlib.util``, avoiding the need for
    ``scripts/reinforcement_learning`` to be an importable package (it has no
    ``__init__.py``).  The loaded module is cached in ``sys.modules`` under
    *module_name* so repeated calls are free.

    Args:
        module_name: Unique module name to register in ``sys.modules``.
        module_path: Path to the Python file to import (``str`` or
            :class:`~pathlib.Path`).

    Returns:
        The imported module.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
