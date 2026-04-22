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
        "schema_version: \"1.0\"\n"
        "generated_at: \"2026-04-22T00:00:00Z\"\n"
        "generator: \"test\"\n"
        "groups:\n"
        "  direct/ant:\n"
        "    - entry_point: isaaclab_tasks.direct.ant:AntEnv\n"
        "      framework: rsl_rl\n"
    )
    with pytest.raises(ValueError, match="task_id"):
        load_env_list(bad)
