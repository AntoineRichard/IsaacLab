# Odin T2.1 — Environment Lists & Newton Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three committed deliverables — `tools/odin/config/physx_envs.yaml`, `tools/odin/config/newton_envs.yaml`, `docs/odin/newton_api_gaps.md` — plus the two enumeration scripts that generate them, so T3's dispatcher has a curated list of runs to queue.

**Architecture:** A small IsaacLab upstream change (promote `_has_physics_preset` to a public helper), two new Odin helpers (`tools/odin/common/env_list.py` + `tools/odin/common/presets.py`), two enumeration scripts (`tools/odin/scripts/enumerate_{physx,newton}_envs.py`) that produce YAML manifests with merge-preserving re-run semantics, followed by a human curation + categorization + prose-authoring handoff.

**Tech Stack:** Python 3.10+, PyYAML, pytest, Omniverse Kit (headless AppLauncher), `isaaclab_tasks` registry, `configclass` dataclasses.

**Spec:** `docs/superpowers/specs/2026-04-22-odin-t2-1-env-lists-design.md`.

**Branch:** `antoiner/feat/odin` (local commits only; do not push).

**Commit convention:** Imperative mood, ~50-char subject, body explains *why*, no AI co-authorship lines (per `AGENTS.md`).

---

## Task 1: Promote `has_physics_preset` to `isaaclab_tasks.utils.presets`

**Goal:** Expose the test-only helper `_has_physics_preset` as a public API so Odin (and anyone else) can use it without importing from `test/`.

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/utils/presets.py`
- Create: `source/isaaclab_tasks/test/test_presets.py`
- Modify: `source/isaaclab_tasks/test/env_test_utils.py` (turn `_has_physics_preset` into a thin alias)
- Modify: `source/isaaclab_tasks/docs/CHANGELOG.rst`
- Modify: `source/isaaclab_tasks/config/extension.toml`

- [ ] **Step 1: Write the failing test for the public helper**

Create `source/isaaclab_tasks/test/test_presets.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`isaaclab_tasks.utils.presets.has_physics_preset`."""

from dataclasses import dataclass, field

import pytest

from isaaclab_tasks.utils.presets import has_physics_preset


# --- Synthetic fixtures -----------------------------------------------------
# We build minimal classes that imitate the shape `has_physics_preset` walks:
# - a "PhysicsCfg" object carrying named preset attributes,
# - a "SimCfg" wrapping a physics object,
# - an "EnvCfg" wrapping a SimCfg,
# - a "PresetCfg" wrapper (has both __dataclass_fields__ and a `default` attr
#   and NO `class_type` on its type — the exact gate the helper checks).


@dataclass
class _PhysicsWithNewton:
    newton: object = None


@dataclass
class _PhysicsWithOther:
    mjwarp: object = None


@dataclass
class _Sim:
    physics: object = None


@dataclass
class _EnvCfg:
    sim: _Sim = field(default_factory=_Sim)


@dataclass
class _PresetWrapper:
    """Imitates a top-level PresetCfg wrapper: has `default` attr, is a
    dataclass, but its type has no `class_type` attribute."""

    default: _EnvCfg = field(default_factory=_EnvCfg)


def test_has_physics_preset_newton_present():
    cfg = _EnvCfg(sim=_Sim(physics=_PhysicsWithNewton()))
    assert has_physics_preset(cfg, "newton") is True


def test_has_physics_preset_newton_absent():
    cfg = _EnvCfg(sim=_Sim(physics=_PhysicsWithOther()))
    assert has_physics_preset(cfg, "newton") is False


def test_has_physics_preset_no_physics():
    cfg = _EnvCfg(sim=_Sim(physics=None))
    assert has_physics_preset(cfg, "newton") is False


def test_has_physics_preset_dict_short_circuits():
    # Dicts are not unwrapped — caller must pass a raw config object.
    assert has_physics_preset({"sim": {"physics": {"newton": {}}}}, "newton") is False


def test_has_physics_preset_unwraps_top_level_preset():
    inner = _EnvCfg(sim=_Sim(physics=_PhysicsWithNewton()))
    wrapper = _PresetWrapper(default=inner)
    assert has_physics_preset(wrapper, "newton") is True


def test_has_physics_preset_other_preset_name():
    cfg = _EnvCfg(sim=_Sim(physics=_PhysicsWithOther()))
    assert has_physics_preset(cfg, "mjwarp") is True
    assert has_physics_preset(cfg, "newton") is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_presets.py -v
```

Expected: `ModuleNotFoundError: No module named 'isaaclab_tasks.utils.presets'` (or equivalent import failure on the first collection attempt).

- [ ] **Step 3: Create the public helper**

Create `source/isaaclab_tasks/isaaclab_tasks/utils/presets.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for inspecting named physics presets on raw env configs."""

from __future__ import annotations

__all__ = ["has_physics_preset"]


def has_physics_preset(raw_cfg, preset_name: str) -> bool:
    """Check if a raw (unresolved) env config has a named physics preset.

    Must be called with the result of
    :func:`~isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry`, not
    :func:`~isaaclab_tasks.utils.parse_cfg.parse_env_cfg`, because the latter
    resolves all ``PresetCfg`` wrappers to their default before returning.

    Args:
        raw_cfg: Raw env config from :func:`load_cfg_from_registry`.
        preset_name: Name of the preset to check for (e.g. ``"newton"``).

    Returns:
        ``True`` if ``raw_cfg.sim.physics`` is a ``PresetCfg`` wrapper that
        defines a field named ``preset_name``, ``False`` otherwise.
    """
    if isinstance(raw_cfg, dict):
        return False
    env_cfg = raw_cfg
    # If the top-level cfg is itself a PresetCfg wrapper, unwrap to its
    # default. A PresetCfg wrapper is a dataclass that has a ``default``
    # attribute and whose type does NOT declare ``class_type`` (which is
    # how an env-config dataclass is distinguished from a preset wrapper).
    if (
        hasattr(env_cfg, "__dataclass_fields__")
        and hasattr(env_cfg, "default")
        and not hasattr(type(env_cfg), "class_type")
    ):
        env_cfg = env_cfg.default
    physics = getattr(getattr(env_cfg, "sim", None), "physics", None)
    return physics is not None and hasattr(physics, preset_name)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_presets.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Update `env_test_utils.py` to delegate**

Modify `source/isaaclab_tasks/test/env_test_utils.py`. Replace the existing `_has_physics_preset` body with a thin alias (preserve the docstring and signature so nothing downstream breaks). Current body lives around lines 58–84.

Replace:

```python
def _has_physics_preset(raw_cfg, preset_name: str) -> bool:
    """Check if a raw (unresolved) env config has a named physics preset.
    ...<existing body>...
    """
    if isinstance(raw_cfg, dict):
        return False
    # <existing logic>
```

with:

```python
def _has_physics_preset(raw_cfg, preset_name: str) -> bool:
    """Thin alias for :func:`isaaclab_tasks.utils.presets.has_physics_preset`.

    Kept for backward compatibility with existing tests; new callers should
    import the public helper directly.
    """
    from isaaclab_tasks.utils.presets import has_physics_preset

    return has_physics_preset(raw_cfg, preset_name)
```

- [ ] **Step 6: Re-run existing tests that use `_has_physics_preset`**

```bash
./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_preset_kit_decision.py -v
```

Expected: all existing tests still pass (the alias is behaviour-equivalent).

- [ ] **Step 7: Update CHANGELOG and extension version**

Modify `source/isaaclab_tasks/docs/CHANGELOG.rst`. Insert the new version entry at the top of the `Changelog` section (above `1.5.23`):

```rst
1.5.24 (2026-04-22)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :func:`~isaaclab_tasks.utils.presets.has_physics_preset` as a public
  helper for checking whether a raw env config declares a named physics preset
  (e.g. ``"newton"``). Promoted from the test-only
  ``env_test_utils._has_physics_preset``; the latter is now a thin alias.

```

Modify `source/isaaclab_tasks/config/extension.toml`. Change the version:

```toml
version = "1.5.24"
```

- [ ] **Step 8: Run pre-commit hooks**

```bash
./isaaclab.sh -f
```

Expected: clean (or re-run once after any auto-formatting). If anything was reformatted, re-stage the affected files before committing.

- [ ] **Step 9: Commit**

```bash
git add \
  source/isaaclab_tasks/isaaclab_tasks/utils/presets.py \
  source/isaaclab_tasks/test/test_presets.py \
  source/isaaclab_tasks/test/env_test_utils.py \
  source/isaaclab_tasks/docs/CHANGELOG.rst \
  source/isaaclab_tasks/config/extension.toml

git commit -m "Promote has_physics_preset to isaaclab_tasks.utils.presets

The existing _has_physics_preset test helper is useful for any code that
needs to ask 'does this env declare physics backend X?' — expose it as a
supported public API. Keep the test-utils private name as a thin alias so
existing callers continue to work unchanged."
```

---

## Task 2: Add `derive_group` helper to `tools/odin/common/env_list.py`

**Goal:** Deterministic mapping from a gym `entry_point` string (e.g. `isaaclab_tasks.direct.ant:AntEnv`) to a two-or-three-component group path (`direct/ant`) used to partition the YAML.

**Files:**
- Create: `tools/odin/common/env_list.py`
- Create: `tools/odin/tests/test_group_derivation.py`

- [ ] **Step 1: Write the failing test**

Create `tools/odin/tests/test_group_derivation.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.common.env_list.derive_group`."""

import pytest

from tools.odin.common.env_list import derive_group


@pytest.mark.parametrize(
    "entry_point, expected_group",
    [
        ("isaaclab_tasks.direct.ant:AntEnv", "direct/ant"),
        ("isaaclab_tasks.direct.anymal_c.flat_env:FlatEnv", "direct/anymal_c"),
        (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c:EnvCfg",
            "manager_based/locomotion/velocity",
        ),
        (
            "isaaclab_tasks.manager_based.manipulation.lift:Env",
            "manager_based/manipulation/lift",
        ),
        ("isaaclab_tasks.direct.factory.factory_env:FactoryEnv", "direct/factory"),
    ],
)
def test_derive_group_known_shapes(entry_point, expected_group):
    assert derive_group(entry_point) == expected_group


def test_derive_group_missing_colon_returns_unknown():
    # Malformed: no class reference; can't derive meaningfully.
    assert derive_group("isaaclab_tasks.direct.ant") == "unknown"


def test_derive_group_not_isaaclab_tasks_returns_unknown():
    # Third-party task registration: don't attempt derivation.
    assert derive_group("some_other_pkg.envs.my_env:MyEnv") == "unknown"


def test_derive_group_empty_string():
    assert derive_group("") == "unknown"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_group_derivation.py -v --confcutdir=tools/odin
```

Expected: `ModuleNotFoundError` on the `derive_group` import.

- [ ] **Step 3: Create the initial `env_list.py` with `derive_group`**

Create `tools/odin/common/env_list.py`:

```python
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

__all__ = ["derive_group"]


_ISAACLAB_TASKS_PREFIX = "isaaclab_tasks."


def derive_group(entry_point: str) -> str:
    """Derive a directory-style group key from a gym ``entry_point`` string.

    The ``entry_point`` is of the form ``"package.module.path:ClassName"``.
    For env registrations under ``isaaclab_tasks.direct.*`` we return
    ``"direct/<first_subpackage>"``. For ``isaaclab_tasks.manager_based.*``
    we return the first three components
    (``"manager_based/<family>/<subfamily>"``); some tasks register deeper
    paths (e.g. ``manager_based.locomotion.velocity.config.anymal_c``) —
    depth cap of three keeps the group usefully coarse.

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
    remainder = module_path[len(_ISAACLAB_TASKS_PREFIX):]
    parts = remainder.split(".")
    if not parts:
        return "unknown"
    if parts[0] == "direct" and len(parts) >= 2:
        return f"direct/{parts[1]}"
    if parts[0] == "manager_based":
        # Take up to two subpackages beyond "manager_based".
        subparts = parts[1:4]
        if not subparts:
            return "unknown"
        return "manager_based/" + "/".join(subparts)
    # Fallback: first two components joined.
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_group_derivation.py -v --confcutdir=tools/odin
```

Expected: all tests pass. One of the parameterized cases (`manager_based.locomotion.velocity.config.anymal_c`) will produce `manager_based/locomotion/velocity/config` under a naïve slice — confirm the depth-cap-of-three returns `manager_based/locomotion/velocity` as asserted. If the test fails on that case, the fix is in the `derive_group` slicing logic, not the test.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_group_derivation.py
git commit -m "Add derive_group helper for Odin env-list partitioning

Maps gym entry_point strings to directory-style groups (direct/<task> or
manager_based/<family>/<subfamily>). Used by the T2.1 enumeration scripts
to lay out physx_envs.yaml and newton_envs.yaml under a groups: mapping."
```

---

## Task 3: Add `suggest_framework` helper

**Goal:** Decide which framework (`rsl_rl` / `skrl` / `None`) to default to based on which entry points the task registers.

**Files:**
- Modify: `tools/odin/common/env_list.py` (append)
- Create: `tools/odin/tests/test_framework_suggestion.py`

- [ ] **Step 1: Write the failing test**

Create `tools/odin/tests/test_framework_suggestion.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.common.env_list.suggest_framework`."""

import pytest

from tools.odin.common.env_list import suggest_framework


@pytest.mark.parametrize(
    "has_rsl_rl, has_skrl, expected",
    [
        (True,  True,  "rsl_rl"),   # both → prefer rsl_rl
        (True,  False, "rsl_rl"),
        (False, True,  "skrl"),
        (False, False, None),        # neither → caller should set keep=False
    ],
)
def test_suggest_framework_decision_table(has_rsl_rl, has_skrl, expected):
    assert suggest_framework(has_rsl_rl, has_skrl) == expected
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_framework_suggestion.py -v --confcutdir=tools/odin
```

Expected: `ImportError: cannot import name 'suggest_framework'`.

- [ ] **Step 3: Add `suggest_framework` to `env_list.py`**

Append to `tools/odin/common/env_list.py`:

```python


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
```

Also add `"suggest_framework"` to the `__all__` list at the top of the file:

```python
__all__ = ["derive_group", "suggest_framework"]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_framework_suggestion.py -v --confcutdir=tools/odin
```

Expected: 4 tests pass.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_framework_suggestion.py
git commit -m "Add suggest_framework helper for Odin env-list defaults

Prefers rsl_rl when both frameworks are registered, falls back to skrl,
returns None when neither is available (forces keep:false downstream)."
```

---

## Task 4: Add `EnvEntry`/`EnvList` dataclasses + YAML round-trip

**Goal:** Define the in-memory representation and the `load_env_list` / `write_env_list` functions that round-trip the YAML schema declared in the spec.

**Files:**
- Modify: `tools/odin/common/env_list.py` (append)
- Create: `tools/odin/tests/test_env_list.py`

- [ ] **Step 1: Write the failing round-trip test**

Create `tools/odin/tests/test_env_list.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Odin env-list YAML IO + merge semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.common.env_list import (
    EnvEntry,
    EnvList,
    load_env_list,
    write_env_list,
)


def _make_entry(task_id: str, group: str = "direct/ant", **overrides) -> EnvEntry:
    defaults = dict(
        task_id=task_id,
        entry_point="isaaclab_tasks.direct.ant:AntEnv",
        env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
        group=group,
        has_rsl_rl=True,
        has_skrl=True,
        framework="rsl_rl",
        num_envs=4096,
        max_iterations=300,
        keep=True,
        status="current",
        notes="",
        suspected_gap=None,
    )
    defaults.update(overrides)
    return EnvEntry(**defaults)


def test_roundtrip_empty_file_returns_empty_envlist(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    loaded = load_env_list(missing)
    assert isinstance(loaded, EnvList)
    assert loaded.groups == {}


def test_roundtrip_single_entry(tmp_path: Path):
    original = EnvList()
    original.groups["direct/ant"] = [_make_entry("Isaac-Ant-Direct-v0")]
    out = tmp_path / "env_list.yaml"
    write_env_list(out, original, generator="test")

    reloaded = load_env_list(out)
    assert list(reloaded.groups.keys()) == ["direct/ant"]
    assert len(reloaded.groups["direct/ant"]) == 1
    assert reloaded.groups["direct/ant"][0].task_id == "Isaac-Ant-Direct-v0"
    assert reloaded.groups["direct/ant"][0].framework == "rsl_rl"


def test_roundtrip_preserves_suspected_gap(tmp_path: Path):
    original = EnvList()
    original.groups["manager_based/locomotion"] = [
        _make_entry(
            "Isaac-Velocity-Rough-Anymal-C-v0",
            group="manager_based/locomotion",
            suspected_gap="sdf_collision",
            notes="Rough terrain uses SDF colliders on heightfield.",
        )
    ]
    out = tmp_path / "gaps.yaml"
    write_env_list(out, original, generator="test")

    reloaded = load_env_list(out)
    entry = reloaded.groups["manager_based/locomotion"][0]
    assert entry.suspected_gap == "sdf_collision"
    assert entry.notes.startswith("Rough terrain")


def test_write_sorts_groups_alphabetically_and_entries_by_task_id(tmp_path: Path):
    original = EnvList()
    original.groups["direct/humanoid"] = [_make_entry("Isaac-Humanoid-Direct-v0")]
    original.groups["direct/ant"] = [
        _make_entry("Isaac-Ant-v0"),
        _make_entry("Isaac-Ant-Direct-v0"),
    ]
    out = tmp_path / "sorted.yaml"
    write_env_list(out, original, generator="test")

    # Confirm key order by reading raw text — YAML preserves dump order.
    text = out.read_text()
    first_group = text.index("direct/ant")
    second_group = text.index("direct/humanoid")
    assert first_group < second_group

    # And within a group, task IDs sort alphabetically.
    ant_v0 = text.index("Isaac-Ant-Direct-v0")
    ant_v2 = text.index("Isaac-Ant-v0")
    assert ant_v0 < ant_v2


def test_schema_version_written_and_read(tmp_path: Path):
    original = EnvList()
    original.groups["direct/ant"] = [_make_entry("Isaac-Ant-Direct-v0")]
    out = tmp_path / "sv.yaml"
    write_env_list(out, original, generator="test")

    text = out.read_text()
    assert 'schema_version: "1.0"' in text or "schema_version: '1.0'" in text or \
           "schema_version: 1.0" in text


def test_load_rejects_unknown_schema_version(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: \"99.0\"\n"
        "generated_at: \"2026-04-22T00:00:00Z\"\n"
        "generator: \"test\"\n"
        "groups: {}\n"
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_env_list(bad)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: `ImportError` on `EnvEntry` / `EnvList` / `load_env_list` / `write_env_list`.

- [ ] **Step 3: Add the dataclasses + IO functions**

Append to `tools/odin/common/env_list.py`:

```python


# -----------------------------------------------------------------------------
# Dataclasses and YAML IO
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


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
    status: str = "current"       # "current" | "new" | "stale"
    notes: str = ""
    suspected_gap: str | None = None


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
    ignored with a warning (printed to stderr).
    """
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
        import sys
        print(
            f"WARNING env_list: ignoring unknown fields on {d.get('task_id', '?')}: "
            f"{sorted(unknown)}",
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
        payload = yaml.safe_load(fh) or {}
    got_version = str(payload.get("schema_version", ""))
    if got_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {got_version!r} in {path} "
            f"(expected {SCHEMA_VERSION!r})"
        )
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
```

Also update the module-level `__all__` at the top of the file:

```python
__all__ = [
    "derive_group",
    "suggest_framework",
    "EnvEntry",
    "EnvList",
    "SCHEMA_VERSION",
    "load_env_list",
    "write_env_list",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: 6 tests pass.

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "Add EnvEntry/EnvList dataclasses and YAML round-trip

Defines schema v1.0 envelope (schema_version, generated_at, generator,
groups) and the in-memory EnvEntry dataclass with all spec fields.
load_env_list returns an empty list for missing files (first-run case)
and raises on unknown schema versions. write_env_list emits groups
alphabetically and entries sorted by task_id for diff-stable output."
```

---

## Task 5: Add `merge` semantics for re-runs

**Goal:** Preserve user edits (`keep`, `framework`, `num_envs`, `max_iterations`, `notes`, `suspected_gap`) when re-running the enumeration scripts; mark removed rows `stale` and new rows `new`.

**Files:**
- Modify: `tools/odin/common/env_list.py` (append)
- Modify: `tools/odin/tests/test_env_list.py` (append merge tests)

- [ ] **Step 1: Append the failing merge tests**

Append to `tools/odin/tests/test_env_list.py`:

```python


# -----------------------------------------------------------------------------
# Merge semantics
# -----------------------------------------------------------------------------


from tools.odin.common.env_list import merge


def _existing_list_with(task_id: str, **overrides) -> EnvList:
    el = EnvList()
    entry = _make_entry(task_id, **overrides)
    el.groups.setdefault(entry.group, []).append(entry)
    return el


def test_merge_preserves_user_keep_false():
    existing = _existing_list_with("Isaac-Ant-Direct-v0", keep=False, notes="too slow")
    discovered = [_make_entry("Isaac-Ant-Direct-v0")]  # script default keep=True
    merged = merge(existing, discovered)

    entry = merged.groups["direct/ant"][0]
    assert entry.keep is False
    assert entry.notes == "too slow"
    assert entry.status == "current"


def test_merge_preserves_user_framework_override():
    existing = _existing_list_with("Isaac-Ant-Direct-v0", framework="skrl")
    discovered = [_make_entry("Isaac-Ant-Direct-v0", framework="rsl_rl")]
    merged = merge(existing, discovered)

    assert merged.groups["direct/ant"][0].framework == "skrl"


def test_merge_preserves_user_training_knobs():
    existing = _existing_list_with(
        "Isaac-Ant-Direct-v0", num_envs=2048, max_iterations=500
    )
    discovered = [_make_entry("Isaac-Ant-Direct-v0", num_envs=4096, max_iterations=300)]
    merged = merge(existing, discovered)

    entry = merged.groups["direct/ant"][0]
    assert entry.num_envs == 2048
    assert entry.max_iterations == 500


def test_merge_refreshes_derived_fields():
    # has_rsl_rl/has_skrl/entry_point reflect the registry now, not the past.
    existing = _existing_list_with(
        "Isaac-Ant-Direct-v0", has_rsl_rl=False, has_skrl=False, entry_point="stale:X"
    )
    discovered = [_make_entry("Isaac-Ant-Direct-v0", has_rsl_rl=True, has_skrl=True)]
    merged = merge(existing, discovered)

    entry = merged.groups["direct/ant"][0]
    assert entry.has_rsl_rl is True
    assert entry.has_skrl is True
    assert entry.entry_point == "isaaclab_tasks.direct.ant:AntEnv"


def test_merge_marks_vanished_rows_stale_and_does_not_delete():
    existing = _existing_list_with("Isaac-Ant-Direct-v0", keep=True)
    discovered: list[EnvEntry] = []  # registry removed the task
    merged = merge(existing, discovered)

    entry = merged.groups["direct/ant"][0]
    assert entry.status == "stale"
    # Row is still present — user removes it consciously.


def test_merge_marks_new_rows_new():
    existing = EnvList()
    discovered = [_make_entry("Isaac-Humanoid-Direct-v0", group="direct/humanoid")]
    merged = merge(existing, discovered)

    entry = merged.groups["direct/humanoid"][0]
    assert entry.status == "new"
    assert entry.keep is True


def test_merge_handles_task_moving_between_groups():
    # Rare but possible: upstream re-filed a task under a different dir.
    existing = _existing_list_with("Isaac-Ant-Direct-v0", group="direct/ant", keep=False)
    discovered = [_make_entry("Isaac-Ant-Direct-v0", group="direct/ant_v2", keep=True)]
    merged = merge(existing, discovered)

    # User's keep=False travels with the task despite the group change.
    assert "direct/ant" not in merged.groups or not merged.groups["direct/ant"]
    entry = merged.groups["direct/ant_v2"][0]
    assert entry.keep is False
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: `ImportError: cannot import name 'merge'`.

- [ ] **Step 3: Implement `merge` in `env_list.py`**

Append to `tools/odin/common/env_list.py`:

```python


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
```

Add `"merge"` to `__all__`:

```python
__all__ = [
    "derive_group",
    "suggest_framework",
    "EnvEntry",
    "EnvList",
    "SCHEMA_VERSION",
    "load_env_list",
    "write_env_list",
    "merge",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: 13 tests pass (6 IO + 7 merge).

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "Add merge() for env-list re-run semantics

Preserves user edits (keep, framework, num_envs, max_iterations, notes,
suspected_gap) on task_id match. Refreshes derived fields (entry_point,
has_rsl_rl, has_skrl, group) from the current registry. Marks
disappeared rows status:stale (kept, not deleted) and new rows
status:new so users can see what changed between runs."
```

---

## Task 6: Add `load_shipped_training_defaults`

**Goal:** Pull `(num_envs, max_iterations)` from the shipped framework cfg for a given task. RSL-RL uses `agent_cfg.max_iterations`; SKRL uses `agent_cfg["trainer"]["timesteps"]` as the normalized equivalent. `num_envs` comes from the env cfg's `scene.num_envs` in both cases.

Because this function loads real cfg modules, it's hard to unit-test directly — we split into a pure-extraction helper (`extract_training_defaults_from_cfgs`) that takes already-loaded cfg objects and test it against synthetic classes / dicts. The IO wrapper that calls `load_cfg_from_registry` is exercised only by the integration test in Task 10.

**Files:**
- Modify: `tools/odin/common/env_list.py` (append)
- Modify: `tools/odin/tests/test_env_list.py` (append extraction tests)

- [ ] **Step 1: Append failing extraction tests**

Append to `tools/odin/tests/test_env_list.py`:

```python


# -----------------------------------------------------------------------------
# Training defaults extraction
# -----------------------------------------------------------------------------


from tools.odin.common.env_list import extract_training_defaults_from_cfgs


class _SceneCfgRsl:
    num_envs = 4096


class _EnvCfgRsl:
    scene = _SceneCfgRsl()


class _RslAgentCfg:
    max_iterations = 1000


class _SceneCfgSkrl:
    num_envs = 2048


class _EnvCfgSkrl:
    scene = _SceneCfgSkrl()


# SKRL agent cfg is a dict (loaded from YAML) in practice.
_SKRL_AGENT_CFG = {"trainer": {"timesteps": 8000}}


def test_extract_training_defaults_rsl_rl():
    n, m = extract_training_defaults_from_cfgs(_EnvCfgRsl(), _RslAgentCfg(), "rsl_rl")
    assert n == 4096
    assert m == 1000


def test_extract_training_defaults_skrl():
    n, m = extract_training_defaults_from_cfgs(_EnvCfgSkrl(), _SKRL_AGENT_CFG, "skrl")
    assert n == 2048
    assert m == 8000


def test_extract_training_defaults_missing_max_iterations():
    class _BareRslAgentCfg:  # no max_iterations
        pass
    n, m = extract_training_defaults_from_cfgs(_EnvCfgRsl(), _BareRslAgentCfg(), "rsl_rl")
    assert n == 4096
    assert m is None


def test_extract_training_defaults_missing_scene():
    class _BareEnvCfg:
        pass
    n, m = extract_training_defaults_from_cfgs(_BareEnvCfg(), _RslAgentCfg(), "rsl_rl")
    assert n is None
    assert m == 1000


def test_extract_training_defaults_skrl_missing_trainer():
    n, m = extract_training_defaults_from_cfgs(_EnvCfgSkrl(), {}, "skrl")
    assert n == 2048
    assert m is None


def test_extract_training_defaults_unknown_framework():
    with pytest.raises(ValueError, match="framework"):
        extract_training_defaults_from_cfgs(_EnvCfgRsl(), _RslAgentCfg(), "bogus")
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: `ImportError` on `extract_training_defaults_from_cfgs`.

- [ ] **Step 3: Implement extraction + IO wrapper**

Append to `tools/odin/common/env_list.py`:

```python


# -----------------------------------------------------------------------------
# Training-defaults loader
# -----------------------------------------------------------------------------


def extract_training_defaults_from_cfgs(
    env_cfg: Any, agent_cfg: Any, framework: str
) -> tuple[int | None, int | None]:
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


def load_shipped_training_defaults(
    task_id: str, framework: str
) -> tuple[int | None, int | None]:
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
        import sys
        print(
            f"WARNING env_list: could not load env cfg for {task_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        env_cfg = None

    try:
        agent_cfg = load_cfg_from_registry(task_id, entry_point_key)
    except Exception as exc:  # noqa: BLE001
        import sys
        print(
            f"WARNING env_list: could not load {framework} cfg for {task_id}: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        agent_cfg = None

    if env_cfg is None or agent_cfg is None:
        return None, None

    return extract_training_defaults_from_cfgs(env_cfg, agent_cfg, framework)
```

Add both names to `__all__`:

```python
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
]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: 19 tests pass (13 existing + 6 new).

- [ ] **Step 5: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/env_list.py tools/odin/tests/test_env_list.py
git commit -m "Add training-defaults extraction for RSL-RL and SKRL

extract_training_defaults_from_cfgs reads num_envs from env_cfg.scene and
max_iterations from agent_cfg.max_iterations (RSL-RL) or
agent_cfg['trainer']['timesteps'] (SKRL, dict loaded from YAML).
load_shipped_training_defaults wraps it with load_cfg_from_registry for
use from the enumeration scripts; per-task failures degrade to (None, None)
with a stderr warning so one bad task does not abort the scan."
```

---

## Task 7: Add `tools/odin/common/presets.py` wrapper

**Goal:** A one-line re-export so Odin scripts don't reach into `isaaclab_tasks` directly; makes the dependency explicit and swappable if Odin later ships to its own repo.

**Files:**
- Create: `tools/odin/common/presets.py`

- [ ] **Step 1: Create the wrapper module**

Create `tools/odin/common/presets.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin-side re-export of the upstream :mod:`isaaclab_tasks.utils.presets`.

Kept as a separate module so that when Odin graduates to its own repo the
upstream dependency surface is visible at a glance: any replacement backend
just needs to provide a ``has_physics_preset`` with the same signature.
"""

from isaaclab_tasks.utils.presets import has_physics_preset

__all__ = ["has_physics_preset"]
```

- [ ] **Step 2: Verify the import works**

```bash
./isaaclab.sh -p -c "from tools.odin.common.presets import has_physics_preset; print(has_physics_preset)"
```

Expected: prints `<function has_physics_preset at 0x...>`. No errors.

- [ ] **Step 3: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/common/presets.py
git commit -m "Add Odin presets re-export wrapper

Surfaces the upstream dependency on isaaclab_tasks.utils.presets as a
single-line import in tools/odin/common/, so when Odin graduates the
external-facing interface is obvious."
```

---

## Task 8: Implement `enumerate_physx_envs.py`

**Goal:** Walk `gym.registry`, build one `EnvEntry` per `Isaac*` task, merge against any existing YAML, and write `tools/odin/config/physx_envs.yaml`.

The script is thin glue over `env_list.py`. We extract the per-task logic into a pure function `build_entry_from_task_spec` so it's unit-testable without Isaac Sim.

**Files:**
- Create: `tools/odin/scripts/__init__.py` (empty, makes the dir a package)
- Create: `tools/odin/scripts/enumerate_physx_envs.py`
- Modify: `tools/odin/common/env_list.py` (append `build_entry_from_task_spec`)
- Modify: `tools/odin/tests/test_env_list.py` (append tests for it)

- [ ] **Step 1: Append failing tests for `build_entry_from_task_spec`**

Append to `tools/odin/tests/test_env_list.py`:

```python


# -----------------------------------------------------------------------------
# build_entry_from_task_spec (called from enumerate_physx_envs.py)
# -----------------------------------------------------------------------------


from tools.odin.common.env_list import build_entry_from_task_spec


class _FakeTaskSpec:
    """Imitates a gymnasium EnvSpec enough for build_entry_from_task_spec."""

    def __init__(self, task_id, entry_point, kwargs):
        self.id = task_id
        self.entry_point = entry_point
        self.kwargs = kwargs


def _noop_defaults_loader(task_id, framework):
    return 4096, 300


def test_build_entry_rsl_rl_preferred_when_both_registered():
    spec = _FakeTaskSpec(
        task_id="Isaac-Ant-Direct-v0",
        entry_point="isaaclab_tasks.direct.ant:AntEnv",
        kwargs={
            "env_cfg_entry_point": "isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            "rsl_rl_cfg_entry_point": "x:Y",
            "skrl_cfg_entry_point": "x:Y",
        },
    )
    e = build_entry_from_task_spec(spec, defaults_loader=_noop_defaults_loader)
    assert e.task_id == "Isaac-Ant-Direct-v0"
    assert e.group == "direct/ant"
    assert e.has_rsl_rl is True
    assert e.has_skrl is True
    assert e.framework == "rsl_rl"
    assert e.num_envs == 4096
    assert e.max_iterations == 300
    assert e.keep is True
    assert e.notes == ""


def test_build_entry_skrl_only():
    spec = _FakeTaskSpec(
        task_id="Isaac-Cartpole-RGB-Camera-v0",
        entry_point="isaaclab_tasks.direct.cartpole:CartpoleRGBEnv",
        kwargs={
            "env_cfg_entry_point": "isaaclab_tasks.direct.cartpole.cfg:Cfg",
            "skrl_cfg_entry_point": "x:Y",
        },
    )
    e = build_entry_from_task_spec(spec, defaults_loader=_noop_defaults_loader)
    assert e.framework == "skrl"
    assert e.keep is True


def test_build_entry_no_framework_forces_keep_false():
    spec = _FakeTaskSpec(
        task_id="Isaac-Manual-v0",
        entry_point="isaaclab_tasks.direct.manual:Env",
        kwargs={"env_cfg_entry_point": "x:Y"},
    )
    e = build_entry_from_task_spec(spec, defaults_loader=_noop_defaults_loader)
    assert e.framework is None
    assert e.keep is False
    assert "No rsl_rl or skrl" in e.notes


def test_build_entry_defaults_loader_failure_forces_keep_false():
    spec = _FakeTaskSpec(
        task_id="Isaac-Broken-v0",
        entry_point="isaaclab_tasks.direct.broken:Env",
        kwargs={
            "env_cfg_entry_point": "x:Y",
            "rsl_rl_cfg_entry_point": "x:Y",
        },
    )

    def failing_loader(task_id, framework):
        return None, None

    e = build_entry_from_task_spec(spec, defaults_loader=failing_loader)
    assert e.framework == "rsl_rl"
    assert e.num_envs is None
    assert e.max_iterations is None
    assert e.keep is False
    assert "training defaults" in e.notes.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: `ImportError: cannot import name 'build_entry_from_task_spec'`.

- [ ] **Step 3: Implement `build_entry_from_task_spec` in `env_list.py`**

Append to `tools/odin/common/env_list.py`:

```python


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
            import sys
            print(
                f"WARNING env_list: defaults_loader raised for {task_spec.id}: "
                f"{type(exc).__name__}: {exc}",
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
```

Add to `__all__`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: 23 tests pass.

- [ ] **Step 5: Create the script**

Create `tools/odin/scripts/__init__.py` (empty file).

Create `tools/odin/scripts/enumerate_physx_envs.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Enumerate IsaacLab PhysX-capable training environments into a YAML manifest.

Writes ``tools/odin/config/physx_envs.yaml`` (by default) with one row per
registered ``Isaac*`` gym task, grouped by directory-derived type. Preserves
user edits (``keep``, ``framework``, ``num_envs``, ``max_iterations``,
``notes``) on re-run via :func:`tools.odin.common.env_list.merge`.

Usage (from the repo root; PYTHONPATH=. is required so ``tools.odin.*`` is
importable):

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py \\
        [--output-path PATH] [--dry-run] [--regenerate [--force]]
"""

from __future__ import annotations

"""Launch Isaac Sim simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

_DEFAULT_OUTPUT = Path("tools/odin/config/physx_envs.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate IsaacLab PhysX envs into a YAML manifest.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output YAML path (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary, write nothing.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Discard existing YAML and start fresh. Destructive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the --regenerate confirmation prompt.",
    )
    return parser.parse_args()


args_cli = _parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import sys

import gymnasium as gym

import isaaclab_tasks  # noqa: F401  (populates the gym registry)

from tools.odin.common.env_list import (
    EnvList,
    build_entry_from_task_spec,
    load_env_list,
    merge,
    write_env_list,
)


def _confirm_regenerate() -> bool:
    if args_cli.force:
        return True
    print(
        f"--regenerate will overwrite {args_cli.output_path}, losing any "
        f"manual edits. Continue? [y/N]",
        end=" ",
        flush=True,
    )
    response = sys.stdin.readline().strip().lower()
    return response == "y"


def main() -> int:
    output_path: Path = args_cli.output_path

    if args_cli.regenerate:
        if not _confirm_regenerate():
            print("Aborted.")
            return 1
        existing = EnvList()
    else:
        existing = load_env_list(output_path)

    discovered: list = []
    errors = 0
    for task_spec in gym.registry.values():
        if "Isaac" not in task_spec.id:
            continue
        try:
            discovered.append(build_entry_from_task_spec(task_spec))
        except Exception as exc:  # noqa: BLE001 — isolate per-task failure
            errors += 1
            print(
                f"WARNING enum: {task_spec.id}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    merged = merge(existing, discovered)

    # Summary: count per status bucket.
    totals = {"current": 0, "new": 0, "stale": 0}
    frameworkless = 0
    for rows in merged.groups.values():
        for e in rows:
            totals[e.status] = totals.get(e.status, 0) + 1
            if e.framework is None:
                frameworkless += 1

    print(
        f"physx envs: {sum(totals.values())} total "
        f"({totals.get('new', 0)} new, {totals.get('stale', 0)} stale, "
        f"{totals.get('current', 0)} current), "
        f"{frameworkless} frameworkless, {errors} enumeration errors."
    )

    if args_cli.dry_run:
        print("--dry-run: not writing.")
        return 0

    write_env_list(output_path, merged, generator="enumerate_physx_envs.py")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
```

- [ ] **Step 6: Sanity-check script import (without running Isaac Sim)**

```bash
./isaaclab.sh -p -c "import ast; ast.parse(open('tools/odin/scripts/enumerate_physx_envs.py').read()); print('OK')"
```

Expected: `OK`. (A real execution is exercised by Task 10's integration test; this just verifies the file parses.)

- [ ] **Step 7: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add \
  tools/odin/scripts/__init__.py \
  tools/odin/scripts/enumerate_physx_envs.py \
  tools/odin/common/env_list.py \
  tools/odin/tests/test_env_list.py
git commit -m "Add enumerate_physx_envs.py for T2.1 env-list generation

Walks gym.registry, builds one row per Isaac* task via
build_entry_from_task_spec, merges against any existing
tools/odin/config/physx_envs.yaml preserving user edits, writes the
result. --dry-run, --regenerate, --force flags control destructive paths."
```

---

## Task 9: Implement `enumerate_newton_envs.py`

**Goal:** Read the user-filtered `physx_envs.yaml`, check each `keep:true` row for a `newton` preset, write `newton_envs.yaml` (supported) and `newton_gap_candidates.yaml` (unsupported, `suspected_gap: "tbd"`).

We split the per-row classification into a pure function `classify_for_newton` that takes a pre-loaded raw cfg and returns `("supported" | "gap")`, so we can unit-test without Isaac Sim.

**Files:**
- Create: `tools/odin/scripts/enumerate_newton_envs.py`
- Modify: `tools/odin/common/env_list.py` (append `classify_for_newton`)
- Modify: `tools/odin/tests/test_env_list.py` (append classify tests)

- [ ] **Step 1: Append failing tests for `classify_for_newton`**

Append to `tools/odin/tests/test_env_list.py`:

```python


# -----------------------------------------------------------------------------
# classify_for_newton (used by enumerate_newton_envs.py)
# -----------------------------------------------------------------------------


from tools.odin.common.env_list import classify_for_newton


class _FakeNewtonPhysicsCfg:
    newton = object()


class _FakePhysxOnlyPhysicsCfg:
    mjwarp = object()


class _FakeSimCfg:
    def __init__(self, physics):
        self.physics = physics


class _FakeRawCfg:
    def __init__(self, physics):
        self.sim = _FakeSimCfg(physics)


def test_classify_supported_when_newton_preset_present():
    cfg = _FakeRawCfg(_FakeNewtonPhysicsCfg())
    assert classify_for_newton(cfg) == "supported"


def test_classify_gap_when_no_newton_preset():
    cfg = _FakeRawCfg(_FakePhysxOnlyPhysicsCfg())
    assert classify_for_newton(cfg) == "gap"


def test_classify_gap_when_no_physics_at_all():
    cfg = _FakeRawCfg(None)
    assert classify_for_newton(cfg) == "gap"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: `ImportError: cannot import name 'classify_for_newton'`.

- [ ] **Step 3: Implement `classify_for_newton`**

Append to `tools/odin/common/env_list.py`:

```python


# -----------------------------------------------------------------------------
# classify_for_newton (used by enumerate_newton_envs.py)
# -----------------------------------------------------------------------------


def classify_for_newton(raw_cfg: Any) -> str:
    """Decide whether a raw env cfg is ``"supported"`` or ``"gap"`` on Newton.

    A cfg is ``"supported"`` iff it exposes a ``newton`` physics preset
    (see :func:`isaaclab_tasks.utils.presets.has_physics_preset`).

    Args:
        raw_cfg: Raw env cfg from ``load_cfg_from_registry``.

    Returns:
        ``"supported"`` or ``"gap"``.
    """
    # Deferred import so this module remains import-light.
    from tools.odin.common.presets import has_physics_preset

    return "supported" if has_physics_preset(raw_cfg, "newton") else "gap"
```

Add to `__all__`:

```python
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
    "classify_for_newton",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_env_list.py -v --confcutdir=tools/odin
```

Expected: 26 tests pass.

- [ ] **Step 5: Create the script**

Create `tools/odin/scripts/enumerate_newton_envs.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive Newton env list and gap-candidates from a filtered PhysX list.

Reads ``tools/odin/config/physx_envs.yaml`` (filtered by the user), visits
every ``keep: true`` row, and partitions them by Newton preset presence:

- Rows whose raw env cfg has a ``newton`` preset → ``newton_envs.yaml``.
- Rows without a ``newton`` preset → ``newton_gap_candidates.yaml`` with
  ``suspected_gap: "tbd"`` for the user to categorize.

Both outputs merge with existing files so prior categorization survives
re-runs.

Usage (from the repo root; PYTHONPATH=. is required so ``tools.odin.*`` is
importable):

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py \\
        [--physx-input PATH] [--newton-output PATH] [--gap-output PATH] \\
        [--dry-run] [--regenerate [--force]]
"""

from __future__ import annotations

"""Launch Isaac Sim simulator first."""

import argparse
import copy
from pathlib import Path

from isaaclab.app import AppLauncher

_DEFAULT_PHYSX_INPUT = Path("tools/odin/config/physx_envs.yaml")
_DEFAULT_NEWTON_OUTPUT = Path("tools/odin/config/newton_envs.yaml")
_DEFAULT_GAP_OUTPUT = Path("tools/odin/config/newton_gap_candidates.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate Newton-supported IsaacLab envs and gap candidates.",
    )
    parser.add_argument("--physx-input", type=Path, default=_DEFAULT_PHYSX_INPUT)
    parser.add_argument("--newton-output", type=Path, default=_DEFAULT_NEWTON_OUTPUT)
    parser.add_argument("--gap-output", type=Path, default=_DEFAULT_GAP_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


args_cli = _parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import sys

import isaaclab_tasks  # noqa: F401  (populates the gym registry)
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from tools.odin.common.env_list import (
    EnvList,
    classify_for_newton,
    load_env_list,
    merge,
    write_env_list,
)


def _confirm_regenerate() -> bool:
    if args_cli.force:
        return True
    print(
        f"--regenerate will overwrite {args_cli.newton_output} and "
        f"{args_cli.gap_output}, losing any manual edits. Continue? [y/N]",
        end=" ",
        flush=True,
    )
    return sys.stdin.readline().strip().lower() == "y"


def main() -> int:
    physx_path: Path = args_cli.physx_input
    physx = load_env_list(physx_path)
    if not physx.groups:
        print(
            f"No PhysX env list at {physx_path}. Run "
            f"tools/odin/scripts/enumerate_physx_envs.py first.",
            file=sys.stderr,
        )
        return 1

    kept = [
        e
        for rows in physx.groups.values()
        for e in rows
        if e.keep and e.status != "stale"
    ]
    print(f"PhysX input: {sum(len(v) for v in physx.groups.values())} total, {len(kept)} kept.")

    if args_cli.regenerate:
        if not _confirm_regenerate():
            print("Aborted.")
            return 1
        existing_newton = EnvList()
        existing_gaps = EnvList()
    else:
        existing_newton = load_env_list(args_cli.newton_output)
        existing_gaps = load_env_list(args_cli.gap_output)

    newton_discovered: list = []
    gap_discovered: list = []
    errors = 0
    for e in kept:
        try:
            raw_cfg = load_cfg_from_registry(e.task_id, "env_cfg_entry_point")
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(
                f"WARNING enum: {e.task_id}: cfg load failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue

        verdict = classify_for_newton(raw_cfg)
        if verdict == "supported":
            newton_discovered.append(copy.deepcopy(e))
        else:
            gap_entry = copy.deepcopy(e)
            gap_entry.suspected_gap = "tbd"
            gap_discovered.append(gap_entry)

    newton_merged = merge(existing_newton, newton_discovered)
    gaps_merged = merge(existing_gaps, gap_discovered)

    print(
        f"newton envs:   {sum(len(v) for v in newton_merged.groups.values())} "
        f"({len(newton_discovered)} from this run)"
    )
    print(
        f"gap candidates:{sum(len(v) for v in gaps_merged.groups.values())} "
        f"({len(gap_discovered)} from this run)"
    )
    print(f"load errors:   {errors}")

    if args_cli.dry_run:
        print("--dry-run: not writing.")
        return 0

    write_env_list(args_cli.newton_output, newton_merged,
                   generator="enumerate_newton_envs.py")
    write_env_list(args_cli.gap_output, gaps_merged,
                   generator="enumerate_newton_envs.py")
    print(f"Wrote {args_cli.newton_output}")
    print(f"Wrote {args_cli.gap_output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        simulation_app.close()
```

- [ ] **Step 6: Sanity-check script import**

```bash
./isaaclab.sh -p -c "import ast; ast.parse(open('tools/odin/scripts/enumerate_newton_envs.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 7: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add \
  tools/odin/scripts/enumerate_newton_envs.py \
  tools/odin/common/env_list.py \
  tools/odin/tests/test_env_list.py
git commit -m "Add enumerate_newton_envs.py for Newton + gap-candidate lists

Reads physx_envs.yaml, visits each keep:true row, calls
load_cfg_from_registry + has_physics_preset to split supported-on-Newton
from gap candidates. Writes newton_envs.yaml (supported) and
newton_gap_candidates.yaml (unsupported, suspected_gap: tbd) with merge
semantics that preserve the user's later gap categorization across re-runs."
```

---

## Task 10: Integration smoke test

**Goal:** One end-to-end test that launches Isaac Sim, runs both scripts against the live registry, and asserts the three YAMLs are schema-valid and internally consistent.

**Files:**
- Create: `tools/odin/tests/test_enumerate_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tools/odin/tests/test_enumerate_integration.py`:

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slow-marked integration test: enumerate scripts end-to-end.

Runs both ``enumerate_physx_envs.py`` and ``enumerate_newton_envs.py``
against the live IsaacLab registry, writing into a tmpdir. Asserts the
three YAMLs parse and are internally consistent. Does NOT assert row
content — the registry changes over time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.odin.common.env_list import load_env_list


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_SH = REPO_ROOT / "isaaclab.sh"


@pytest.mark.slow
def test_enumerate_pipeline_end_to_end(tmp_path: Path):
    physx_out = tmp_path / "physx_envs.yaml"
    newton_out = tmp_path / "newton_envs.yaml"
    gap_out = tmp_path / "newton_gap_candidates.yaml"

    # Inherit env plus PYTHONPATH=. so `tools.odin.*` imports resolve.
    import os
    child_env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}

    # --- Step 1: enumerate PhysX envs ---
    result = subprocess.run(
        [
            str(ISAACLAB_SH),
            "-p",
            "tools/odin/scripts/enumerate_physx_envs.py",
            "--output-path",
            str(physx_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=child_env,
    )
    assert result.returncode == 0, (
        f"enumerate_physx_envs.py failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert physx_out.exists(), "PhysX output YAML was not created"

    physx = load_env_list(physx_out)
    assert physx.groups, "PhysX env list is empty — registry unpopulated?"

    # --- Step 2: enumerate Newton envs ---
    result = subprocess.run(
        [
            str(ISAACLAB_SH),
            "-p",
            "tools/odin/scripts/enumerate_newton_envs.py",
            "--physx-input",
            str(physx_out),
            "--newton-output",
            str(newton_out),
            "--gap-output",
            str(gap_out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env=child_env,
    )
    assert result.returncode == 0, (
        f"enumerate_newton_envs.py failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert newton_out.exists()
    assert gap_out.exists()

    newton = load_env_list(newton_out)
    gaps = load_env_list(gap_out)

    # --- Consistency: every Newton/gap row was kept in PhysX ---
    physx_by_id = {
        e.task_id: e for rows in physx.groups.values() for e in rows
    }
    for rows in newton.groups.values():
        for e in rows:
            assert e.task_id in physx_by_id, f"Newton row {e.task_id} not in PhysX YAML"
            assert physx_by_id[e.task_id].keep, (
                f"Newton row {e.task_id} was kept in physx but keep=False there"
            )

    # --- Consistency: no row appears in both Newton and gap lists ---
    newton_ids = {e.task_id for rows in newton.groups.values() for e in rows}
    gap_ids = {e.task_id for rows in gaps.groups.values() for e in rows}
    assert not (newton_ids & gap_ids), (
        f"Rows appear in both newton_envs and gap_candidates: "
        f"{sorted(newton_ids & gap_ids)}"
    )

    # --- Every gap row has suspected_gap set (tbd is fine for fresh runs) ---
    for rows in gaps.groups.values():
        for e in rows:
            assert e.suspected_gap is not None, (
                f"Gap row {e.task_id} missing suspected_gap"
            )
```

- [ ] **Step 2: Verify the test is collected but marked slow**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_enumerate_integration.py --collect-only -v --confcutdir=tools/odin
```

Expected: the test is listed; mark `slow` is visible.

- [ ] **Step 3: Run the slow test manually (sequential — no parallel GPU)**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_enumerate_integration.py -v --confcutdir=tools/odin
```

Expected: 1 test passes. Runtime: ~1–3 minutes for the two subprocess Isaac Sim launches.

If the test fails because of an issue in one of the scripts, fix the script (not the test) and re-run.

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/tests/test_enumerate_integration.py
git commit -m "Add end-to-end integration test for Odin T2.1 enumeration

Runs enumerate_physx_envs.py then enumerate_newton_envs.py against the
live registry, writing into a tmpdir. Asserts that every Newton row was
keep:true in PhysX, that no row appears in both Newton and gap lists,
and that every gap row has a suspected_gap field. Marked @pytest.mark.slow
so the default test suite stays fast."
```

---

## Task 11: Update `tools/odin/README.md`

**Goal:** Document the two enumeration commands and the filter protocol so future users (and subagents) know how to run T2.1's pipeline.

**Files:**
- Modify: `tools/odin/README.md`

- [ ] **Step 1: Append the new section**

Open `tools/odin/README.md`. After the `Running tests` section, append:

```markdown

## Enumerating environments (T2.1)

T2.1 produces three committed artifacts that feed T3's dispatcher:

- `tools/odin/config/physx_envs.yaml` — curated PhysX run list.
- `tools/odin/config/newton_envs.yaml` — curated Newton run list (derived
  from the PhysX kept set that also has a `newton` preset).
- `docs/odin/newton_api_gaps.md` — narrative on what Newton is missing to
  unlock the remaining PhysX-kept tasks, plus a per-env appendix.

### Generate / refresh the PhysX list

Run from the repo root. `PYTHONPATH=.` makes `tools.odin.*` importable.

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py
# Options:
#   --output-path PATH   (default: tools/odin/config/physx_envs.yaml)
#   --dry-run            (print summary, write nothing)
#   --regenerate --force (discard existing YAML)
```

The script walks `gym.registry` for every `Isaac*` task, populates
`framework` / `num_envs` / `max_iterations` from the shipped framework
config, and writes the YAML grouped by directory-derived type. On re-run
it preserves your manual edits (`keep`, `framework`, `notes`, etc.) — rows
that vanish from the registry are kept with `status: stale`; new rows are
`status: new`.

### Curate the PhysX list

Edit `tools/odin/config/physx_envs.yaml` directly. Flip `keep: false` on
rows you don't want T3 to dispatch; adjust `framework` where the auto-pick
is wrong (e.g. force `skrl` on a vision task); tune `num_envs` /
`max_iterations` if the shipped defaults are wildly off for benchmarking.

### Generate the Newton list + gap candidates

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py
```

Reads your filtered `physx_envs.yaml`, writes:

- `tools/odin/config/newton_envs.yaml` for tasks whose env cfg declares a
  `newton` physics preset.
- `tools/odin/config/newton_gap_candidates.yaml` for the rest, each row
  carrying `suspected_gap: tbd`.

### Categorize the gap candidates and write the gap doc

1. Edit `tools/odin/config/newton_gap_candidates.yaml`: replace each
   `suspected_gap: tbd` with one of
   `sdf_collision | tendons | rough_terrain | manipulation_coverage | deformable | other`.
   Use `notes:` to add context where it helps the gap narrative.
2. Author `docs/odin/newton_api_gaps.md` with per-gap body sections
   (what's missing, count of affected envs, unlock value) followed by a
   per-env appendix table.
```

- [ ] **Step 2: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add tools/odin/README.md
git commit -m "Document T2.1 enumeration pipeline in tools/odin/README

Covers enumerate_physx_envs.py and enumerate_newton_envs.py invocation,
the manual filter / categorization steps between them, and the final
gap-doc authoring handoff."
```

---

## Task 12 (MANUAL — HUMAN): Run PhysX enumeration and curate

> **Agent workers stop here.** The remaining tasks require a human to read
> task IDs and decide which to keep. An agent worker should signal "ready
> for human curation" and hand off.

**Files:**
- Create: `tools/odin/config/physx_envs.yaml` (generated + curated)

- [ ] **Step 1 (human): Run the script**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py
```

Expected: writes `tools/odin/config/physx_envs.yaml`. Console summary lists total / new / stale counts.

- [ ] **Step 2 (human): Review and curate**

Open `tools/odin/config/physx_envs.yaml`. For each row:

- Decide whether to keep it. Set `keep: false` on rows you don't want
  dispatched (duplicates, play variants, obviously uninteresting tasks,
  tasks flagged as broken).
- If `framework: null` and `notes:` cites "No rsl_rl or skrl entry
  point": leave `keep: false`; these are tasks we cannot benchmark.
- If `num_envs: null` or `max_iterations: null`: either fix the shipped
  cfg (upstream concern, out of T2.1 scope) or fill the values manually
  with a `notes:` explaining the source, or set `keep: false`.
- For vision tasks that auto-picked `rsl_rl` but you want `skrl`: flip
  `framework: skrl`.

- [ ] **Step 3 (human): Commit the curated YAML**

```bash
./isaaclab.sh -f
git add tools/odin/config/physx_envs.yaml
git commit -m "Curate Odin T2.1 PhysX env list

Initial human-curated PhysX run list for T3 dispatch. Keep-flags reflect
the first pass of what we want benchmarked; num_envs / max_iterations
are the shipped per-task defaults (calibration from dry-run reward
plateaus to follow in a later iteration)."
```

---

## Task 13 (MANUAL — HUMAN): Run Newton enumeration, curate, and categorize gaps

**Files:**
- Create: `tools/odin/config/newton_envs.yaml` (generated + curated)
- Create: `tools/odin/config/newton_gap_candidates.yaml` (generated + categorized)

- [ ] **Step 1 (human): Run the script**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py
```

Expected: writes `newton_envs.yaml` and `newton_gap_candidates.yaml`. Summary reports kept counts and load errors.

- [ ] **Step 2 (human): Curate `newton_envs.yaml`**

Review every row. Flip `keep: false` on Newton envs you don't want to
dispatch (too slow on Newton, known-flaky, redundant with PhysX's pick).
Add `notes:` when the reason isn't obvious from the task name.

- [ ] **Step 3 (human): Categorize `newton_gap_candidates.yaml`**

For each row with `suspected_gap: tbd`, replace with the correct category
from the vocabulary:

- `sdf_collision` — uses SDF colliders (rough terrain, nut-and-bolt, …)
- `tendons` — tendon actuation
- `rough_terrain` — heightfield / procedural terrain (not SDF-based)
- `manipulation_coverage` — manipulation task, Newton surface untested
- `deformable` — deformable / softbody simulation
- `other` — none of the above; `notes:` required

If a row legitimately doesn't fit, leave `other` and document why in
`notes:`. Never leave `tbd` — the gap-doc render step rejects it.

To figure out which category a task falls into, open its env cfg (path
visible in `entry_point`) and look for: SDF-related mesh loading, tendon
config blocks, terrain kind, presence of cameras / manipulation scenes.

- [ ] **Step 4 (human): Commit both YAMLs**

```bash
./isaaclab.sh -f
git add \
  tools/odin/config/newton_envs.yaml \
  tools/odin/config/newton_gap_candidates.yaml
git commit -m "Curate Odin T2.1 Newton env list and categorize gap candidates

Human pass over newton_envs.yaml (keep-flags for Newton run list) and
newton_gap_candidates.yaml (suspected_gap replaced with the vocabulary
category for each PhysX-kept task that Newton cannot currently run).
Feeds the gap-doc narrative in docs/odin/newton_api_gaps.md."
```

---

## Task 14 (MANUAL — HUMAN): Author `docs/odin/newton_api_gaps.md`

**Files:**
- Create: `docs/odin/newton_api_gaps.md`

- [ ] **Step 1 (human): Write the document**

Create `docs/odin/newton_api_gaps.md` following the template in the spec
(§YAML schema and gap doc structure). Structure:

```markdown
# Newton API gaps blocking Odin environments

**Input:** `tools/odin/config/newton_gap_candidates.yaml` at <this commit>
**Scope:** gaps blocking PhysX-kept envs from running on Newton
           (`physx_envs.yaml` ∩ ¬ `newton_envs.yaml`).

## 1. <First gap category>

Envs blocked: N. Unlock value: <high | medium | low>.

<Narrative: what the missing API is, what "support" would look like,
rough effort estimate, any upstream Newton issue or discussion link.>

## 2. <Next gap category>

...

## N. Other / TBD

<Any `suspected_gap: other` rows inline with their notes.>

---

## Appendix: per-env table

| Task | Group | Gap | Notes |
|------|-------|-----|-------|
| Isaac-Velocity-Rough-Anymal-C-v0 | manager_based/locomotion/velocity | sdf_collision | Rough terrain uses SDF heightfield |
| ... | ... | ... | ... |
```

The appendix table is deterministic from `newton_gap_candidates.yaml`.
Order by group, then task_id. You can render it with a one-off Python
snippet (note `PYTHONPATH=.` so `tools.odin` resolves):

```bash
PYTHONPATH=. ./isaaclab.sh -p -c "
from tools.odin.common.env_list import load_env_list
g = load_env_list('tools/odin/config/newton_gap_candidates.yaml')
print('| Task | Group | Gap | Notes |')
print('|------|-------|-----|-------|')
for group in sorted(g.groups):
    for e in sorted(g.groups[group], key=lambda x: x.task_id):
        print(f'| {e.task_id} | {e.group} | {e.suspected_gap} | {e.notes} |')
"
```

- [ ] **Step 2 (human): Commit the gap doc**

```bash
./isaaclab.sh -f
git add docs/odin/newton_api_gaps.md
git commit -m "Author Odin Newton API gap document

Per-gap body describing each missing API with blocked-env count and
unlock value, plus a per-env appendix table derived mechanically from
newton_gap_candidates.yaml. Feeds Newton-team triage and provides the
reference for deciding whether to close a gap or skip the affected
envs."
```

---

## Task 15: Update `docs/odin/architecture.md`

**Goal:** Mark T2.1 complete in the living architecture doc, link the spec, and add a change-log entry per the doc's self-imposed rule ("commit this update in the same commit as the underlying change").

We do this as its own final commit *after* the manual deliverables because the task-map row can only be flipped to ✅ once the deliverables exist.

**Files:**
- Modify: `docs/odin/architecture.md`

- [ ] **Step 1: Update the task map (§6)**

Open `docs/odin/architecture.md`. Find the T2.1 row in the task map table (currently status `⚪`). Change it to:

```markdown
| T2.1 | Environment lists + Newton gap doc | `docs/superpowers/specs/2026-04-22-odin-t2-1-env-lists-design.md` | ✅ |
```

- [ ] **Step 2: Update the document's "Last updated" line**

Replace the existing line near the top of the file with:

```markdown
**Last updated:** 2026-04-22 (end of T2.1)
```

- [ ] **Step 3: Add a change-log entry (§9)**

Append a new row to the change log table:

```markdown
| 2026-04-22 | T2.1 delivered: physx_envs.yaml + newton_envs.yaml + newton_api_gaps.md committed. Added upstream public helper `isaaclab_tasks.utils.presets.has_physics_preset` (promoted from the test-only `_has_physics_preset`) and the Odin-side `tools/odin/common/env_list.py` + `enumerate_{physx,newton}_envs.py` scripts. Gap vocabulary: sdf_collision / tendons / rough_terrain / manipulation_coverage / deformable / other. | Odin T2.1 |
```

- [ ] **Step 4: Pre-commit and commit**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark Odin T2.1 complete in architecture reference

Flip T2.1 status to done in the task map; link the spec; add a
change-log entry noting the three committed deliverables, the upstream
has_physics_preset promotion, and the gap-categorization vocabulary."
```

---

## Self-review notes (for the implementer)

Before calling T2.1 done, verify:

1. **All tests pass.**
   - `./isaaclab.sh -p -m pytest source/isaaclab_tasks/test/test_presets.py -v` — 6 passing.
   - `./isaaclab.sh -p -m pytest tools/odin/tests/ -m 'not slow' -v --confcutdir=tools/odin` — all unit tests passing.
   - `./isaaclab.sh -p -m pytest tools/odin/tests/test_enumerate_integration.py -v --confcutdir=tools/odin` — slow integration passing.

2. **Pre-commit clean.** `./isaaclab.sh -f` clean on `HEAD`.

3. **Deliverables exist.** All four artifacts in git:
   - `tools/odin/config/physx_envs.yaml`
   - `tools/odin/config/newton_envs.yaml`
   - `tools/odin/config/newton_gap_candidates.yaml`
   - `docs/odin/newton_api_gaps.md`

4. **Cross-consistency:** every row in `newton_envs.yaml` exists in `physx_envs.yaml` with `keep: true`; no row appears in both `newton_envs.yaml` and `newton_gap_candidates.yaml`; no row in `newton_gap_candidates.yaml` has `suspected_gap: tbd`.

5. **Changelog + extension version bumped** for `isaaclab_tasks` (1.5.23 → 1.5.24).

6. **Architecture doc** reflects T2.1 ✅ with the spec link and change-log entry.

7. **No `odin_runs/` bundles referenced.** T2.1 does not consume T1's dry-run bundles — those are known-broken per project memory and will be regenerated after T1 bug fixes.
