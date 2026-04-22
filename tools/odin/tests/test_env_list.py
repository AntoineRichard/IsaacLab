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
    assert 'schema_version: "1.0"' in text or "schema_version: '1.0'" in text or "schema_version: 1.0" in text


def test_load_rejects_unknown_schema_version(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text('schema_version: "99.0"\ngenerated_at: "2026-04-22T00:00:00Z"\ngenerator: "test"\ngroups: {}\n')
    with pytest.raises(ValueError, match="schema_version"):
        load_env_list(bad)


def test_load_rejects_non_dict_yaml(tmp_path: Path):
    # Top-level YAML that parses to something other than a mapping (a list,
    # here) must raise a clear error, not an AttributeError from payload.get.
    bad = tmp_path / "list.yaml"
    bad.write_text("- foo\n- bar\n")
    with pytest.raises(ValueError, match="mapping"):
        load_env_list(bad)


def test_load_rejects_row_missing_task_id(tmp_path: Path):
    # A row without task_id can't be merge-keyed; fail loudly instead of
    # silently constructing an EnvEntry with task_id=None.
    bad = tmp_path / "no_id.yaml"
    bad.write_text(
        'schema_version: "1.0"\n'
        'generated_at: "2026-04-22T00:00:00Z"\n'
        'generator: "test"\n'
        "groups:\n"
        "  direct/ant:\n"
        "    - entry_point: isaaclab_tasks.direct.ant:AntEnv\n"
        "      framework: rsl_rl\n"
    )
    with pytest.raises(ValueError, match="task_id"):
        load_env_list(bad)


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
    existing = _existing_list_with("Isaac-Ant-Direct-v0", num_envs=2048, max_iterations=500)
    discovered = [_make_entry("Isaac-Ant-Direct-v0", num_envs=4096, max_iterations=300)]
    merged = merge(existing, discovered)

    entry = merged.groups["direct/ant"][0]
    assert entry.num_envs == 2048
    assert entry.max_iterations == 500


def test_merge_refreshes_derived_fields():
    # has_rsl_rl/has_skrl/entry_point reflect the registry now, not the past.
    existing = _existing_list_with("Isaac-Ant-Direct-v0", has_rsl_rl=False, has_skrl=False, entry_point="stale:X")
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
    assert "direct/ant" not in merged.groups
    entry = merged.groups["direct/ant_v2"][0]
    assert entry.keep is False


def test_merge_rejects_duplicate_task_ids_in_discovered():
    existing = EnvList()
    discovered = [
        _make_entry("Isaac-Ant-Direct-v0"),
        _make_entry("Isaac-Ant-Direct-v0", num_envs=2048),
    ]
    with pytest.raises(ValueError, match="Duplicate task_id"):
        merge(existing, discovered)


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
