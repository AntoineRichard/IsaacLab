# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DataLayer's cancel-queue wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.valhalla.dashboard.data import DataLayer


def test_request_cancel_inserts_into_db(tmp_path: Path):
    layer = DataLayer(tmp_path)

    layer.request_cancel("20260504-100000", "run-a", kind="kill")

    assert layer.read_cancel_queue("20260504-100000") == {"run-a": "kill"}


def test_read_cancel_queue_empty_for_unknown_dispatch(tmp_path: Path):
    layer = DataLayer(tmp_path)

    assert layer.read_cancel_queue("nope") == {}


def test_request_cancel_rejects_invalid_kind(tmp_path: Path):
    layer = DataLayer(tmp_path)

    with pytest.raises(ValueError):
        layer.request_cancel("20260504-100000", "run-a", kind="terminate")


def test_request_cancel_replaces_kind(tmp_path: Path):
    layer = DataLayer(tmp_path)

    layer.request_cancel("20260504-100000", "run-a", kind="skip")
    layer.request_cancel("20260504-100000", "run-a", kind="kill")

    assert layer.read_cancel_queue("20260504-100000") == {"run-a": "kill"}
