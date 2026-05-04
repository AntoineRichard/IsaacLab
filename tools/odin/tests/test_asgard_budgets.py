# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`tools.odin.asgard.budgets`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.budgets import (
    Budgets,
    load_budgets,
    parse_gpu_class,
)


def _write(p: Path, body: str) -> Path:
    p.write_text(body)
    return p


# --- parse_gpu_class --------------------------------------------------------


def test_parse_gpu_class_blackwell():
    """Match the exact line shape ``nvidia-smi -L`` emits on Blackwell."""
    out = "GPU 0: NVIDIA RTX PRO 5000 Blackwell (UUID: GPU-9686ec94-27e9-cdcf-f1c0-05cf8bc623a3)"
    assert parse_gpu_class(out) == "blackwell-pro-5000"


def test_parse_gpu_class_l40():
    out = "GPU 0: NVIDIA L40 (UUID: GPU-...)"
    assert parse_gpu_class(out) == "l40"


def test_parse_gpu_class_l40_aarch64():
    """ARM L40 hosts emit a slightly different model string; same class."""
    out = "GPU 0: NVIDIA L40 SXM (UUID: GPU-...)"
    assert parse_gpu_class(out) == "l40-sxm"


def test_parse_gpu_class_a100():
    out = "GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-...)"
    assert parse_gpu_class(out) == "a100-sxm4-80gb"


def test_parse_gpu_class_empty_returns_none():
    assert parse_gpu_class("") is None


def test_parse_gpu_class_no_gpu_returns_none():
    assert parse_gpu_class("Failed to initialize NVML: Unknown Error") is None


def test_parse_gpu_class_takes_first_when_multi_gpu():
    """Multi-GPU hosts get budget-classed by the first device. We don't run
    multi-GPU per worker today, but parse defensively."""
    out = (
        "GPU 0: NVIDIA L40 (UUID: GPU-aaa)\n"
        "GPU 1: NVIDIA L40 (UUID: GPU-bbb)\n"
    )
    assert parse_gpu_class(out) == "l40"


# --- load_budgets -----------------------------------------------------------


def test_load_budgets_minimal(tmp_path):
    p = _write(
        tmp_path / "b.yaml",
        """
defaults:
  rsl_rl: 3600
  skrl: 7200
""",
    )
    b = load_budgets(p)
    assert isinstance(b, Budgets)
    assert b.defaults == {"rsl_rl": 3600, "skrl": 7200}
    assert b.budgets == {}
    # When no GPU multipliers are declared, everything passes through at 1.0.
    assert b.gpu_multipliers == {"default": 1.0}


def test_load_budgets_full(tmp_path):
    p = _write(
        tmp_path / "b.yaml",
        """
defaults:
  rsl_rl: 3600
  skrl: 7200
budgets:
  Isaac-Cartpole-Direct-v0:
    rsl_rl: 600
  Isaac-Deploy-Reach-Rizon4s-v0:
    rsl_rl: 36000
gpu_multipliers:
  default: 1.5
  blackwell-pro-5000: 1.0
  l40: 1.4
""",
    )
    b = load_budgets(p)
    assert b.budgets["Isaac-Cartpole-Direct-v0"]["rsl_rl"] == 600
    assert b.gpu_multipliers["l40"] == 1.4


def test_load_budgets_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_budgets(tmp_path / "missing.yaml")


# --- Budgets.lookup ---------------------------------------------------------


def test_lookup_per_task_value(tmp_path):
    """An entry under ``budgets[task][fw]`` overrides the framework default."""
    b = load_budgets(
        _write(
            tmp_path / "b.yaml",
            """
defaults: {rsl_rl: 3600, skrl: 7200}
budgets:
  Isaac-Cartpole-Direct-v0:
    rsl_rl: 600
""",
        )
    )
    assert b.lookup("Isaac-Cartpole-Direct-v0", "rsl_rl", "blackwell-pro-5000") == 600


def test_lookup_falls_back_to_framework_default(tmp_path):
    b = load_budgets(
        _write(
            tmp_path / "b.yaml",
            """
defaults: {rsl_rl: 3600, skrl: 7200}
budgets: {}
""",
        )
    )
    assert b.lookup("Isaac-Unknown-Task-v0", "rsl_rl", "blackwell-pro-5000") == 3600
    assert b.lookup("Isaac-Unknown-Task-v0", "skrl", "blackwell-pro-5000") == 7200


def test_lookup_applies_gpu_multiplier(tmp_path):
    """L40 is ~40% slower than Blackwell — bump the budget proportionally so a
    job that completes in 1h on Blackwell gets 1h24 on L40."""
    b = load_budgets(
        _write(
            tmp_path / "b.yaml",
            """
defaults: {rsl_rl: 3600, skrl: 7200}
gpu_multipliers:
  default: 1.5
  blackwell-pro-5000: 1.0
  l40: 1.4
""",
        )
    )
    assert b.lookup("X", "rsl_rl", "blackwell-pro-5000") == 3600
    assert b.lookup("X", "rsl_rl", "l40") == int(3600 * 1.4)


def test_lookup_unknown_gpu_class_uses_default_multiplier(tmp_path):
    """When we don't have a multiplier for a GPU class, fall back to the
    ``default`` entry so a fleet introduces a new GPU without crashing."""
    b = load_budgets(
        _write(
            tmp_path / "b.yaml",
            """
defaults: {rsl_rl: 3600, skrl: 7200}
gpu_multipliers:
  default: 2.0
  blackwell-pro-5000: 1.0
""",
        )
    )
    assert b.lookup("X", "rsl_rl", "h100-sxm") == int(3600 * 2.0)


def test_lookup_no_gpu_class_uses_default_multiplier(tmp_path):
    """``gpu_class=None`` (preflight couldn't parse it) routes through the
    default multiplier — avoids a job with no timeout if detection fails."""
    b = load_budgets(
        _write(
            tmp_path / "b.yaml",
            """
defaults: {rsl_rl: 3600, skrl: 7200}
gpu_multipliers:
  default: 1.5
""",
        )
    )
    assert b.lookup("X", "rsl_rl", None) == int(3600 * 1.5)


def test_lookup_returns_int(tmp_path):
    """Multiplier math can produce floats; the worker's ``timeout_s`` field
    expects int — ensure we always return an int."""
    b = load_budgets(
        _write(
            tmp_path / "b.yaml",
            """
defaults: {rsl_rl: 3600}
gpu_multipliers:
  default: 1.0
  l40: 1.4
""",
        )
    )
    out = b.lookup("X", "rsl_rl", "l40")
    assert isinstance(out, int)


def test_lookup_unknown_framework_falls_back_to_global_default(tmp_path):
    """A framework not listed in ``defaults`` still gets a budget — return
    a generous fallback (12h) rather than 0 so a misconfigured runner can't
    accidentally kill jobs the moment they start."""
    b = load_budgets(
        _write(
            tmp_path / "b.yaml",
            """
defaults: {rsl_rl: 3600}
""",
        )
    )
    assert b.lookup("X", "rl_games", "blackwell-pro-5000") == 12 * 3600
