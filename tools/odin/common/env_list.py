# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""YAML load/merge/write + discovery helpers for Odin env-list YAMLs.

This module is the single entry point for reading, merging, and writing
the three T2.1 YAML artifacts
(``physx_envs.yaml``, ``newton_envs.yaml``, ``newton_gap_candidates.yaml``).
It also provides small pure-Python helpers used by the enumeration scripts:
:func:`derive_group`, :func:`suggest_framework`, and
:func:`load_shipped_training_defaults`.
"""

from __future__ import annotations

__all__ = ["derive_group", "suggest_framework"]


_ISAACLAB_TASKS_PREFIX = "isaaclab_tasks."


def derive_group(entry_point: str) -> str:
    """Derive a directory-style group key from a gym ``entry_point`` string.

    The ``entry_point`` is of the form ``"package.module.path:ClassName"``.
    For env registrations under ``isaaclab_tasks.direct.*`` we return
    ``"direct/<first_subpackage>"``. For ``isaaclab_tasks.manager_based.*``
    we return ``"manager_based/<family>/<subfamily>"`` — i.e. up to two
    subpackages beyond ``manager_based``. Some tasks register deeper paths
    (e.g. ``manager_based.locomotion.velocity.config.anymal_c``); the
    two-subpart cap keeps the group usefully coarse.

    Args:
        entry_point: The gym ``entry_point`` string.

    Returns:
        Group key (``"direct/ant"``, ``"manager_based/locomotion/velocity"``,
        etc.) or ``"unknown"`` when the string is empty, missing a ``:``, or
        doesn't start with ``isaaclab_tasks.``.
    """
    if not entry_point or ":" not in entry_point:
        return "unknown"
    module_path = entry_point.split(":", 1)[0]
    if not module_path.startswith(_ISAACLAB_TASKS_PREFIX):
        return "unknown"
    remainder = module_path[len(_ISAACLAB_TASKS_PREFIX) :]
    parts = remainder.split(".")
    if not parts:
        return "unknown"
    if parts[0] == "direct" and len(parts) >= 2:
        return f"direct/{parts[1]}"
    if parts[0] == "manager_based":
        # Take up to two subpackages beyond "manager_based" (three total).
        subparts = parts[1:3]
        if not subparts:
            return "unknown"
        return "manager_based/" + "/".join(subparts)
    # Fallback: first two components joined.
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def suggest_framework(has_rsl_rl: bool, has_skrl: bool) -> str | None:
    """Suggest the default learning framework for a task.

    Preference: rsl_rl whenever registered; else skrl; else ``None``. A
    ``None`` return signals the caller to force ``keep: false`` on the
    YAML row with a diagnostic ``notes`` message — the dispatcher has
    nothing to run against a frameworkless task.

    Args:
        has_rsl_rl: Whether the task registers ``rsl_rl_cfg_entry_point``.
        has_skrl: Whether the task registers ``skrl_cfg_entry_point``.

    Returns:
        ``"rsl_rl"``, ``"skrl"``, or ``None``.
    """
    if has_rsl_rl:
        return "rsl_rl"
    if has_skrl:
        return "skrl"
    return None
