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

import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "derive_group",
    "suggest_framework",
    "EnvEntry",
    "EnvList",
    "SCHEMA_VERSION",
    "load_env_list",
    "write_env_list",
    "merge",
    "extract_training_defaults_from_cfgs",
    "load_shipped_training_defaults",
    "build_entry_from_task_spec",
]


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


# -----------------------------------------------------------------------------
# Dataclasses and YAML IO
# -----------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"

# Per-row field order in written YAML. Keeps diffs stable across re-runs.
_ENTRY_FIELD_ORDER = [
    "task_id",
    "entry_point",
    "env_cfg_entry_point",
    "group",
    "has_rsl_rl",
    "has_skrl",
    "framework",
    "num_envs",
    "max_iterations",
    "keep",
    "status",
    "suspected_gap",
    "notes",
]


@dataclass
class EnvEntry:
    """One row in a T2.1 env-list YAML.

    Fields mirror the spec's schema v1.0. ``suspected_gap`` is only used
    in ``newton_gap_candidates.yaml`` and is ``None`` elsewhere; the field
    is still written as a YAML key for schema uniformity.
    """

    task_id: str
    entry_point: str
    env_cfg_entry_point: str | None
    group: str
    has_rsl_rl: bool
    has_skrl: bool
    framework: str | None
    num_envs: int | None
    max_iterations: int | None
    keep: bool
    status: str = "current"  # "current" | "new" | "stale"
    suspected_gap: str | None = None
    notes: str = ""


@dataclass
class EnvList:
    """In-memory representation of a T2.1 env-list YAML."""

    groups: dict[str, list[EnvEntry]] = field(default_factory=dict)


def _entry_to_dict(e: EnvEntry) -> dict[str, Any]:
    """Ordered dict for YAML dump, respecting ``_ENTRY_FIELD_ORDER``."""
    d = asdict(e)
    return {k: d[k] for k in _ENTRY_FIELD_ORDER}


def _entry_from_dict(d: dict[str, Any]) -> EnvEntry:
    """Build an :class:`EnvEntry` tolerantly from a loaded YAML row.

    Missing optional fields get the dataclass defaults; unknown fields are
    ignored with a warning (printed to stderr). A missing ``task_id`` is
    fatal — every row must be identifiable for merge keying.
    """
    if "task_id" not in d:
        raise ValueError(f"Row missing required 'task_id' field: {d!r}")
    known = {f: d.get(f) for f in _ENTRY_FIELD_ORDER if f in d}
    # Required fields must be present; default to "" / None on truly minimal rows.
    known.setdefault("entry_point", "")
    known.setdefault("env_cfg_entry_point", None)
    known.setdefault("group", "unknown")
    known.setdefault("has_rsl_rl", False)
    known.setdefault("has_skrl", False)
    known.setdefault("framework", None)
    known.setdefault("num_envs", None)
    known.setdefault("max_iterations", None)
    known.setdefault("keep", True)
    known.setdefault("status", "current")
    known.setdefault("notes", "")
    known.setdefault("suspected_gap", None)
    unknown = set(d) - set(_ENTRY_FIELD_ORDER)
    if unknown:
        print(
            f"WARNING env_list: ignoring unknown fields on {d.get('task_id', '?')}: {sorted(unknown)}",
            file=sys.stderr,
        )
    return EnvEntry(**known)


def load_env_list(path: Path) -> EnvList:
    """Load an env-list YAML from disk.

    Returns an empty :class:`EnvList` when the file does not exist (first
    run of an enumeration script). Raises :class:`ValueError` if the file
    exists but declares an unsupported ``schema_version``.

    Args:
        path: Path to the YAML file.

    Returns:
        Populated or empty :class:`EnvList`.
    """
    if not path.exists():
        return EnvList()
    with path.open("r") as fh:
        payload = yaml.safe_load(fh)
    if payload is None:
        return EnvList()
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level YAML mapping in {path}, got {type(payload).__name__}")
    got_version = str(payload.get("schema_version", ""))
    if got_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {got_version!r} in {path} (expected {SCHEMA_VERSION!r})")
    groups_raw = payload.get("groups") or {}
    env_list = EnvList()
    for group, rows in groups_raw.items():
        env_list.groups[group] = [_entry_from_dict(r) for r in (rows or [])]
    return env_list


def write_env_list(path: Path, env_list: EnvList, *, generator: str) -> None:
    """Write an env-list YAML to disk with stable ordering.

    Groups are written alphabetically; within each group, entries are sorted
    by ``task_id``. The schema envelope (``schema_version``, ``generated_at``,
    ``generator``) is always present.

    Args:
        path: Destination path. Parent directory created if missing.
        env_list: The list to write.
        generator: Script identity string (e.g. ``"enumerate_physx_envs.py"``).
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": generator,
        "groups": {},
    }
    for group in sorted(env_list.groups):
        rows = sorted(env_list.groups[group], key=lambda e: e.task_id)
        payload["groups"][group] = [_entry_to_dict(e) for e in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(
            payload,
            fh,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )


def merge(existing: EnvList, discovered: list[EnvEntry]) -> EnvList:
    """Merge discovered entries against an existing list, preserving user edits.

    Semantics (spec §Architecture):

    - ``task_id`` in both → preserve user fields
      (``keep``, ``framework``, ``num_envs``, ``max_iterations``, ``notes``,
      ``suspected_gap``); refresh derived fields
      (``entry_point``, ``env_cfg_entry_point``, ``has_rsl_rl``, ``has_skrl``,
      ``group``); set ``status = "current"``.
    - ``task_id`` in existing only → mark ``status = "stale"``; keep the row.
      Never delete automatically; the user removes stale rows consciously.
    - ``task_id`` in discovered only → insert with ``status = "new"``.

    Merging is keyed on ``task_id`` alone. If a task's ``group`` has changed
    upstream, the row migrates to the new group carrying the user's fields.

    Args:
        existing: Previously-written env list (from :func:`load_env_list`).
        discovered: Rows from the current enumeration pass.

    Returns:
        A new :class:`EnvList` combining both, never mutating inputs.
    """
    # Flatten existing into a dict keyed on task_id.
    existing_by_id: dict[str, EnvEntry] = {}
    for rows in existing.groups.values():
        for row in rows:
            existing_by_id[row.task_id] = row

    # Reject duplicate task_ids in discovered — ambiguous contract and a
    # programmer error (gym.registry cannot produce duplicates by design).
    seen_discovered: set[str] = set()
    for new in discovered:
        if new.task_id in seen_discovered:
            raise ValueError(f"Duplicate task_id {new.task_id!r} in discovered list")
        seen_discovered.add(new.task_id)

    merged = EnvList()
    discovered_ids: set[str] = set()

    # First pass: existing + discovered intersection, plus new-only.
    for new in discovered:
        discovered_ids.add(new.task_id)
        old = existing_by_id.get(new.task_id)
        if old is None:
            merged_entry = EnvEntry(
                task_id=new.task_id,
                entry_point=new.entry_point,
                env_cfg_entry_point=new.env_cfg_entry_point,
                group=new.group,
                has_rsl_rl=new.has_rsl_rl,
                has_skrl=new.has_skrl,
                framework=new.framework,
                num_envs=new.num_envs,
                max_iterations=new.max_iterations,
                keep=new.keep,
                status="new",
                notes=new.notes,
                suspected_gap=new.suspected_gap,
            )
        else:
            merged_entry = EnvEntry(
                task_id=new.task_id,
                # Derived / refreshed from current registry:
                entry_point=new.entry_point,
                env_cfg_entry_point=new.env_cfg_entry_point,
                group=new.group,
                has_rsl_rl=new.has_rsl_rl,
                has_skrl=new.has_skrl,
                # Preserved from user edits:
                framework=old.framework,
                num_envs=old.num_envs,
                max_iterations=old.max_iterations,
                keep=old.keep,
                notes=old.notes,
                suspected_gap=old.suspected_gap,
                status="current",
            )
        merged.groups.setdefault(merged_entry.group, []).append(merged_entry)

    # Second pass: existing-only → stale. Preserve all fields; flag status.
    for task_id, old in existing_by_id.items():
        if task_id in discovered_ids:
            continue
        stale = EnvEntry(**{**asdict(old), "status": "stale"})
        merged.groups.setdefault(stale.group, []).append(stale)

    return merged


# -----------------------------------------------------------------------------
# Training-defaults loader
# -----------------------------------------------------------------------------


def extract_training_defaults_from_cfgs(env_cfg: Any, agent_cfg: Any, framework: str) -> tuple[int | None, int | None]:
    """Pull ``(num_envs, max_iterations)`` from already-loaded cfg objects.

    Args:
        env_cfg: Env-side cfg (attribute-navigable, has a ``scene.num_envs``).
        agent_cfg: Framework-specific learning cfg. For ``rsl_rl`` it is a
            ``@configclass`` instance with ``.max_iterations``; for ``skrl``
            it is a plain ``dict`` loaded from YAML.
        framework: ``"rsl_rl"`` or ``"skrl"``.

    Returns:
        ``(num_envs, max_iterations)`` where either can be ``None`` if the
        field is absent.

    Raises:
        ValueError: If ``framework`` is not ``"rsl_rl"`` or ``"skrl"``.
    """
    if framework not in ("rsl_rl", "skrl"):
        raise ValueError(f"Unknown framework {framework!r}; expected 'rsl_rl' or 'skrl'")

    # num_envs: same place for both frameworks — env_cfg.scene.num_envs.
    scene = getattr(env_cfg, "scene", None)
    num_envs = getattr(scene, "num_envs", None) if scene is not None else None

    # max_iterations: framework-specific.
    if framework == "rsl_rl":
        max_iterations = getattr(agent_cfg, "max_iterations", None)
    else:  # skrl
        # SKRL cfgs come from YAML as dicts; trainer.timesteps is the
        # closest analog. NB: semantics differ from RSL-RL's max_iterations;
        # T1's benchmark_skrl.py handles the distinction at run time.
        if isinstance(agent_cfg, dict):
            trainer = agent_cfg.get("trainer") or {}
            max_iterations = trainer.get("timesteps") if isinstance(trainer, dict) else None
        else:
            # Some SKRL cfgs are dataclasses; fall back to attribute access.
            trainer = getattr(agent_cfg, "trainer", None)
            max_iterations = getattr(trainer, "timesteps", None) if trainer is not None else None

    return num_envs, max_iterations


def load_shipped_training_defaults(task_id: str, framework: str) -> tuple[int | None, int | None]:
    """Load the shipped ``(num_envs, max_iterations)`` for a task.

    Launches no Isaac Sim — the caller must have done so (``gym.registry``
    must be populated).

    Args:
        task_id: The gym task id (e.g. ``"Isaac-Ant-Direct-v0"``).
        framework: ``"rsl_rl"`` or ``"skrl"``.

    Returns:
        ``(num_envs, max_iterations)``; either may be ``None`` on partial
        failure (logged to stderr).
    """
    # Deferred import — isaaclab_tasks must be importable, which requires the
    # app to be up. Caller's responsibility.
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    entry_point_key = f"{framework}_cfg_entry_point"

    try:
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
    except Exception as exc:  # noqa: BLE001 — we want any failure isolated
        print(
            f"WARNING env_list: could not load env cfg for {task_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        env_cfg = None

    try:
        agent_cfg = load_cfg_from_registry(task_id, entry_point_key)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING env_list: could not load {framework} cfg for {task_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        agent_cfg = None

    if env_cfg is None or agent_cfg is None:
        return None, None

    return extract_training_defaults_from_cfgs(env_cfg, agent_cfg, framework)


# -----------------------------------------------------------------------------
# Per-task-spec row construction (used by enumerate_physx_envs.py)
# -----------------------------------------------------------------------------


def build_entry_from_task_spec(
    task_spec: Any,
    *,
    defaults_loader=load_shipped_training_defaults,
) -> EnvEntry:
    """Construct an :class:`EnvEntry` from a gym ``EnvSpec``-like object.

    Args:
        task_spec: An object with ``id``, ``entry_point``, and ``kwargs``
            attributes (gymnasium's ``EnvSpec`` satisfies this).
        defaults_loader: Callable taking ``(task_id, framework)`` and returning
            ``(num_envs, max_iterations)``. Defaults to the real
            :func:`load_shipped_training_defaults`; tests pass a stub.

    Returns:
        A freshly-built :class:`EnvEntry` with ``status="current"``.
    """
    kwargs = task_spec.kwargs or {}
    has_rsl_rl = "rsl_rl_cfg_entry_point" in kwargs
    has_skrl = "skrl_cfg_entry_point" in kwargs
    framework = suggest_framework(has_rsl_rl, has_skrl)
    group = derive_group(task_spec.entry_point or "")

    num_envs: int | None = None
    max_iterations: int | None = None
    notes = ""

    if framework is None:
        notes = "No rsl_rl or skrl entry point registered."
    else:
        try:
            num_envs, max_iterations = defaults_loader(task_spec.id, framework)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING env_list: defaults_loader raised for {task_spec.id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            num_envs = None
            max_iterations = None
        if num_envs is None or max_iterations is None:
            notes = (
                "Could not resolve training defaults from shipped framework cfg; "
                "review the cfg and fill num_envs / max_iterations manually."
            )

    keep = framework is not None and num_envs is not None and max_iterations is not None

    return EnvEntry(
        task_id=task_spec.id,
        entry_point=task_spec.entry_point or "",
        env_cfg_entry_point=kwargs.get("env_cfg_entry_point"),
        group=group,
        has_rsl_rl=has_rsl_rl,
        has_skrl=has_skrl,
        framework=framework,
        num_envs=num_envs,
        max_iterations=max_iterations,
        keep=keep,
        status="current",
        notes=notes,
        suspected_gap=None,
    )
