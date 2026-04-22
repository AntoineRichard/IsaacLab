# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Odin run_id format."""

from datetime import datetime, timezone

import pytest

from tools.odin.common.run_id import compute_run_id, parse_run_id


def test_compute_run_id_basic():
    now = datetime(2026, 4, 22, 13, 15, 0, tzinfo=timezone.utc)
    rid = compute_run_id("rsl_rl", "physx", "Isaac-Ant-Direct-v0", 42, now=now)
    assert rid == "rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-131500_seed42"


def test_compute_run_id_rejects_unknown_framework():
    with pytest.raises(ValueError):
        compute_run_id("torch_rl", "physx", "Isaac-Ant-Direct-v0", 0)


def test_compute_run_id_rejects_unknown_backend():
    with pytest.raises(ValueError):
        compute_run_id("skrl", "mujoco", "Isaac-Ant-Direct-v0", 0)


def test_round_trip_ant():
    now = datetime(2026, 4, 22, 13, 15, 0, tzinfo=timezone.utc)
    rid = compute_run_id("skrl", "newton", "Isaac-Ant-Direct-v0", 7, now=now)
    parts = parse_run_id(rid)
    assert parts["framework"] == "skrl"
    assert parts["backend"] == "newton"
    assert parts["task"] == "Isaac-Ant-Direct-v0"
    assert parts["seed"] == 7


def test_parse_run_id_rejects_malformed():
    with pytest.raises(ValueError):
        parse_run_id("not_a_run_id")
