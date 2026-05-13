# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config-loader coverage for the per-task timeout refactor.

The ``timeout_classes`` table and ``default_timeout_class`` field are
gone — per-task timeouts now come from ``job_budgets.yaml`` via
:mod:`tools.odin.asgard.budgets`. ``chunk_size`` stays. Old-style
configs that still carry the removed fields emit deprecation warnings
but continue to parse.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tools.odin.bifrost.config import BifrostConfigError, load_bifrost_config

_BASE_YAML = """\
osmo_profile: prod
pool: rtx-pro-6000-eval
priority: NORMAL
image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
  pull_credential: ngc-readonly
defaults:
  resources:
    cpu: 16
    gpu: 1
    memory: 64Gi
    storage: 64Gi
    platform: rtx-pro-6000
  exec_timeout: 14400
  queue_timeout: 7200
retry:
  reschedule_codes: "3001-3006"
  restart_codes: ""
bundle_dataset_prefix: odin
code_delivery:
  mode: files_upload
  source_root: tools/odin
"""

_CHUNK_SIZE_BLOCK = "chunk_size: 25\n"

_LEGACY_CLASSES_BLOCK = """\
timeout_classes:
  short: "30m"
  medium: "2h"
default_timeout_class: medium
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(body)
    return p


def test_chunk_size_loads(tmp_path: Path):
    """``chunk_size`` parses into an int."""
    cfg = load_bifrost_config(_write(tmp_path, _BASE_YAML + _CHUNK_SIZE_BLOCK))
    assert cfg.chunk_size == 25


def test_chunk_size_defaults_when_missing(tmp_path: Path):
    """Configs without ``chunk_size`` get the documented default 25."""
    cfg = load_bifrost_config(_write(tmp_path, _BASE_YAML))
    assert cfg.chunk_size == 25


def test_chunk_size_override(tmp_path: Path):
    """Operator-supplied ``chunk_size`` overrides the default."""
    cfg = load_bifrost_config(_write(tmp_path, _BASE_YAML + "chunk_size: 10\n"))
    assert cfg.chunk_size == 10


def test_chunk_size_invalid_raises(tmp_path: Path):
    """Negative / zero / non-int ``chunk_size`` is a config error."""
    with pytest.raises(BifrostConfigError, match="chunk_size"):
        load_bifrost_config(_write(tmp_path, _BASE_YAML + "chunk_size: 0\n"))


def test_legacy_timeout_classes_field_warns(tmp_path: Path, caplog):
    """``timeout_classes:`` is no longer consulted; surface a warning."""
    with caplog.at_level(logging.WARNING, logger="tools.odin.bifrost.config"):
        load_bifrost_config(_write(tmp_path, _BASE_YAML + _LEGACY_CLASSES_BLOCK + _CHUNK_SIZE_BLOCK))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("timeout_classes" in m and "no longer used" in m for m in warnings), warnings


def test_legacy_default_timeout_class_field_warns(tmp_path: Path, caplog):
    """``default_timeout_class:`` alone (without classes) also warns."""
    body = _BASE_YAML + "default_timeout_class: medium\n" + _CHUNK_SIZE_BLOCK
    with caplog.at_level(logging.WARNING, logger="tools.odin.bifrost.config"):
        load_bifrost_config(_write(tmp_path, body))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("default_timeout_class" in m and "no longer used" in m for m in warnings), warnings


def test_legacy_exec_timeout_warns(tmp_path: Path, caplog):
    """``defaults.exec_timeout`` is also no longer used (budgets file owns it)."""
    with caplog.at_level(logging.WARNING, logger="tools.odin.bifrost.config"):
        load_bifrost_config(_write(tmp_path, _BASE_YAML + _CHUNK_SIZE_BLOCK))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("defaults.exec_timeout" in m and "no longer used" in m for m in warnings), warnings


def test_config_has_no_timeout_classes_attribute(tmp_path: Path):
    """``BifrostConfig`` no longer carries the class table."""
    cfg = load_bifrost_config(_write(tmp_path, _BASE_YAML + _CHUNK_SIZE_BLOCK))
    assert not hasattr(cfg, "timeout_classes")
    assert not hasattr(cfg, "default_timeout_class")
