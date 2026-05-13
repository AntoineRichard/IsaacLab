# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config loader coverage for the new ``timeout_classes`` fields (spec §4.2)."""

from __future__ import annotations

import logging
from pathlib import Path

from tools.odin.bifrost.config import load_bifrost_config

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

_TIMEOUT_CLASSES_BLOCK = """\
timeout_classes:
  short: "30m"
  medium: "2h"
  long: "8h"
  very_long: "24h"
default_timeout_class: medium
chunk_size: 25
"""


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(body)
    return p


def test_timeout_classes_loaded_into_dict(tmp_path: Path):
    """``timeout_classes`` parses into ``dict[str, str]`` keyed by class name."""
    cfg = load_bifrost_config(_write(tmp_path, _BASE_YAML + _TIMEOUT_CLASSES_BLOCK))
    assert cfg.timeout_classes == {
        "short": "30m",
        "medium": "2h",
        "long": "8h",
        "very_long": "24h",
    }
    assert cfg.default_timeout_class == "medium"
    assert cfg.chunk_size == 25


def test_timeout_classes_missing_defaults_to_empty(tmp_path: Path):
    """Old configs without ``timeout_classes`` still load.

    They get an empty dict + the documented default class + chunk size, so
    callers can detect the legacy mode by ``not cfg.timeout_classes``.
    """
    cfg = load_bifrost_config(_write(tmp_path, _BASE_YAML))
    assert cfg.timeout_classes == {}
    assert cfg.default_timeout_class == "medium"
    assert cfg.chunk_size == 25


def test_chunk_size_override(tmp_path: Path):
    """Operator-supplied ``chunk_size`` overrides the default 25."""
    body = _BASE_YAML + _TIMEOUT_CLASSES_BLOCK.replace("chunk_size: 25", "chunk_size: 10")
    cfg = load_bifrost_config(_write(tmp_path, body))
    assert cfg.chunk_size == 10


def test_default_timeout_class_override(tmp_path: Path):
    body = _BASE_YAML + _TIMEOUT_CLASSES_BLOCK.replace("default_timeout_class: medium", "default_timeout_class: short")
    cfg = load_bifrost_config(_write(tmp_path, body))
    assert cfg.default_timeout_class == "short"


def test_deprecation_warning_when_legacy_exec_timeout_with_classes(tmp_path: Path, caplog):
    """Both legacy ``defaults.exec_timeout`` and new ``timeout_classes`` set → warn.

    The legacy field is still parsed (so old configs don't crash), but it
    will be ignored once the planner switches to per-chunk timeouts.
    """
    with caplog.at_level(logging.WARNING, logger="tools.odin.bifrost.config"):
        load_bifrost_config(_write(tmp_path, _BASE_YAML + _TIMEOUT_CLASSES_BLOCK))
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("defaults.exec_timeout" in m and "timeout_classes" in m for m in warnings), warnings
