# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from tools.odin.bifrost.poller import (
    TERMINAL_OSMO_STATES,
    classify_terminal_state,
    is_terminal,
)


@pytest.mark.parametrize(
    "osmo_state, expected_kind",
    [
        ("FAILED", "hugin_crash"),
        ("FAILED_EXEC_TIMEOUT", "timeout"),
        ("FAILED_BACKEND_ERROR", "infrastructure"),
        ("FAILED_PREEMPTED", "infrastructure"),
        ("FAILED_EVICTED", "infrastructure"),
        ("FAILED_IMAGE_PULL", "infrastructure"),
        ("FAILED_START_ERROR", "infrastructure"),
        ("FAILED_START_TIMEOUT", "infrastructure"),
        ("FAILED_QUEUE_TIMEOUT", "infrastructure"),
        ("FAILED_SERVER_ERROR", "infrastructure"),
        ("FAILED_CANCELED", "infrastructure"),
    ],
)
def test_failure_kind_for_each_failed_state(osmo_state: str, expected_kind: str):
    assert classify_terminal_state(osmo_state) == expected_kind


def test_completed_returns_none():
    """COMPLETED isn't a failure kind — caller decides hugin_malformed_bundle separately."""
    assert classify_terminal_state("COMPLETED") is None


def test_classify_unknown_state_defaults_to_infrastructure():
    assert classify_terminal_state("FAILED_NOVEL_THING") == "infrastructure"


def test_is_terminal_recognizes_completed_and_failed_family():
    assert is_terminal("COMPLETED")
    assert is_terminal("FAILED")
    assert is_terminal("FAILED_BACKEND_ERROR")


def test_is_terminal_excludes_in_flight_states():
    for s in ("PENDING", "WAITING", "PROCESSING", "SCHEDULING", "INITIALIZING", "RUNNING", "RESCHEDULED"):
        assert not is_terminal(s), s


def test_terminal_set_complete():
    """Sanity-check the terminal set contains all FAILED_* and COMPLETED."""
    expected_subset = {
        "COMPLETED",
        "FAILED",
        "FAILED_EXEC_TIMEOUT",
        "FAILED_BACKEND_ERROR",
        "FAILED_PREEMPTED",
        "FAILED_EVICTED",
        "FAILED_IMAGE_PULL",
        "FAILED_START_ERROR",
        "FAILED_START_TIMEOUT",
        "FAILED_QUEUE_TIMEOUT",
        "FAILED_SERVER_ERROR",
        "FAILED_CANCELED",
    }
    assert expected_subset <= TERMINAL_OSMO_STATES
