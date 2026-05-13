# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curated-YAML env loader: ``timeout_class`` field handling (spec §4.1, §5.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.bifrost.cli import _build_rows
from tools.odin.bifrost.config import BifrostConfigError, load_bifrost_config

_CFG_WITH_CLASSES = """\
osmo_profile: prod
pool: rtx-pro-6000-eval
priority: NORMAL
image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
defaults:
  resources:
    cpu: 16
    gpu: 1
    memory: 64Gi
    storage: 64Gi
    platform: rtx-pro-6000
  exec_timeout: 14400
  queue_timeout: 7200
retry: {reschedule_codes: "3001-3006", restart_codes: ""}
bundle_dataset_prefix: odin
code_delivery: {mode: files_upload, source_root: tools/odin}
timeout_classes:
  short: "30m"
  medium: "2h"
  long: "8h"
default_timeout_class: medium
chunk_size: 25
"""


def _envs_yaml(timeout_class_line: str) -> str:
    return (
        "groups:\n"
        "  direct/ant:\n"
        "  - task_id: Isaac-Ant-Direct-v0\n"
        "    framework: rsl_rl\n"
        "    num_envs: 4096\n"
        "    max_iterations: 500\n"
        "    keep: true\n" + timeout_class_line
    )


def _write_cfg(tmp_path: Path, body: str = _CFG_WITH_CLASSES) -> Path:
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(body)
    return p


def _write_envs(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "physx.yaml"
    p.write_text(body)
    return p


def test_timeout_class_from_env_yaml(tmp_path: Path):
    """An env with ``timeout_class: short`` produces a row with that class."""
    cfg = load_bifrost_config(_write_cfg(tmp_path))
    envs = _write_envs(tmp_path, _envs_yaml("    timeout_class: short\n"))
    rows = _build_rows(
        physx_yaml=envs,
        newton_yaml=None,
        seeds=[42],
        include_glob=None,
        dispatch_id="20260513-000000",
        cfg=cfg,
    )
    assert len(rows) == 1
    assert rows[0].timeout_class == "short"


def test_timeout_class_falls_back_to_default(tmp_path: Path):
    """An env that omits ``timeout_class`` falls back to the config's default."""
    cfg = load_bifrost_config(_write_cfg(tmp_path))
    envs = _write_envs(tmp_path, _envs_yaml(""))
    rows = _build_rows(
        physx_yaml=envs,
        newton_yaml=None,
        seeds=[42],
        include_glob=None,
        dispatch_id="20260513-000000",
        cfg=cfg,
    )
    assert len(rows) == 1
    assert rows[0].timeout_class == "medium"


def test_unknown_timeout_class_raises(tmp_path: Path):
    """A ``timeout_class`` not listed in ``cfg.timeout_classes`` is a config error.

    Catching the typo at planner time avoids submitting an OSMO workflow
    whose ``exec_timeout`` resolves to ``None``.
    """
    cfg = load_bifrost_config(_write_cfg(tmp_path))
    envs = _write_envs(tmp_path, _envs_yaml("    timeout_class: bananas\n"))
    with pytest.raises(BifrostConfigError, match="bananas"):
        _build_rows(
            physx_yaml=envs,
            newton_yaml=None,
            seeds=[42],
            include_glob=None,
            dispatch_id="20260513-000000",
            cfg=cfg,
        )


def test_default_unknown_when_classes_set_raises(tmp_path: Path):
    """``default_timeout_class`` itself must be a known class when classes are set."""
    bad_cfg = _CFG_WITH_CLASSES.replace("default_timeout_class: medium", "default_timeout_class: oops")
    cfg = load_bifrost_config(_write_cfg(tmp_path, bad_cfg))
    envs = _write_envs(tmp_path, _envs_yaml(""))
    with pytest.raises(BifrostConfigError, match="oops"):
        _build_rows(
            physx_yaml=envs,
            newton_yaml=None,
            seeds=[42],
            include_glob=None,
            dispatch_id="20260513-000000",
            cfg=cfg,
        )
