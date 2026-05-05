# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the JobEntry allowed-transition graph (transition_to tests added in Task 2)."""

from __future__ import annotations

from tools.odin.asgard.jobs import JobEntry


def _job(status: str = "pending", **overrides) -> JobEntry:
    """Build a minimal JobEntry for transition tests. All required fields populated with stubs."""
    defaults = dict(
        run_id="test-run",
        task_id="Isaac-Test-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=1,
        max_iterations=1,
        seed=42,
        bundle_dir_name="test-run",
        status=status,
    )
    defaults.update(overrides)
    return JobEntry(**defaults)


def test_allowed_transitions_graph_has_eight_legal_edges():
    """The graph encodes seven edges from spec §4.1 plus one back-compat 'assigned'→'pending' edge — eight total."""
    expected = {
        ("assigned", "pending"),
        ("pending", "running"),
        ("pending", "failed"),
        ("running", "completed"),
        ("running", "failed"),
        ("running", "pending"),
        ("failed", "pending"),
        ("completed", "pending"),
    }
    actual = {(src, dst) for src, dsts in JobEntry._ALLOWED_TRANSITIONS.items() for dst in dsts}
    assert actual == expected
