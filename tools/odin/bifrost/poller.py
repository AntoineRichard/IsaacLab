# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Poll an OSMO workflow and classify per-task terminal states.

Single source of truth for the OSMO-state → Odin-failure-kind mapping
lives in :data:`OSMO_STATE_TO_FAILURE_KIND` (spec §7).
"""

from __future__ import annotations

__all__ = [
    "OSMO_STATE_TO_FAILURE_KIND",
    "TERMINAL_OSMO_STATES",
    "classify_terminal_state",
    "is_terminal",
]


# Maps OSMO terminal task states to Odin failure kinds. Per spec §7.
# COMPLETED is intentionally absent: callers decide
# hugin_malformed_bundle vs success after manifest validation.
OSMO_STATE_TO_FAILURE_KIND: dict[str, str] = {
    "FAILED": "hugin_crash",
    "FAILED_EXEC_TIMEOUT": "timeout",
    "FAILED_BACKEND_ERROR": "infrastructure",
    "FAILED_PREEMPTED": "infrastructure",
    "FAILED_EVICTED": "infrastructure",
    "FAILED_IMAGE_PULL": "infrastructure",
    "FAILED_START_ERROR": "infrastructure",
    "FAILED_START_TIMEOUT": "infrastructure",
    "FAILED_QUEUE_TIMEOUT": "infrastructure",
    "FAILED_SERVER_ERROR": "infrastructure",
    "FAILED_CANCELED": "infrastructure",
}

TERMINAL_OSMO_STATES: frozenset[str] = frozenset({"COMPLETED", *OSMO_STATE_TO_FAILURE_KIND.keys()})


def is_terminal(osmo_state: str) -> bool:
    """Return True iff ``osmo_state`` is one of the known terminal task states."""
    return osmo_state in TERMINAL_OSMO_STATES


def classify_terminal_state(osmo_state: str) -> str | None:
    """Return the Odin failure kind for an OSMO terminal state, or ``None`` for COMPLETED.

    Unknown ``FAILED_*`` states default to ``"infrastructure"`` to keep behavior
    safe under OSMO version drift.

    Args:
        osmo_state: OSMO task state string.

    Returns:
        Odin failure kind (``hugin_crash`` | ``timeout`` | ``infrastructure``)
        for failed states, ``None`` for ``COMPLETED``, ``None`` for non-terminal
        states.
    """
    if osmo_state == "COMPLETED":
        return None
    if osmo_state in OSMO_STATE_TO_FAILURE_KIND:
        return OSMO_STATE_TO_FAILURE_KIND[osmo_state]
    if osmo_state.startswith("FAILED"):
        return "infrastructure"
    return None
