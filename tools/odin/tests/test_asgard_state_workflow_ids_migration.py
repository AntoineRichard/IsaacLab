# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DispatchState ``osmo_workflow_ids`` field + legacy migration (spec §4.4)."""

from __future__ import annotations

import json
from pathlib import Path

from tools.odin.asgard.jobs import JobEntry
from tools.odin.asgard.state import (
    SCHEMA_VERSION,
    DispatchState,
    FleetSnapshot,
    read_dispatch_state,
    write_dispatch_state,
)


def _job(run_id: str, status: str = "pending") -> JobEntry:
    return JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name=run_id,
        status=status,
    )


def _empty_state(*, dispatcher: str = "osmo") -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260513-100000",
        started_at="2026-05-13T10:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="",
        fleet=[FleetSnapshot(host="h1", status="idle")],
        jobs=[_job("r-1")],
        dispatcher=dispatcher,
    )


def test_new_dispatch_defaults_to_empty_workflow_ids_list(tmp_path: Path):
    """A freshly built ``DispatchState`` has ``osmo_workflow_ids == []``."""
    state = _empty_state()
    assert state.osmo_workflow_ids == []


def test_writing_and_reading_populated_workflow_ids(tmp_path: Path):
    """List of workflow ids survives a write/read round-trip."""
    state = _empty_state()
    state.osmo_workflow_ids = ["wf-a", "wf-b", "wf-c"]
    write_dispatch_state(tmp_path, state)
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded is not None
    assert reloaded.osmo_workflow_ids == ["wf-a", "wf-b", "wf-c"]


def test_legacy_dispatch_json_with_single_workflow_id_migrates_to_list(tmp_path: Path):
    """Old dispatch.json with ``osmo_workflow_id`` only loads as a 1-element list.

    Pre-timeout-bucket dispatches wrote a single ``osmo_workflow_id``
    string. Reading them with the new schema must populate
    ``osmo_workflow_ids = [<that id>]`` so the poller's
    multi-workflow walk picks them up.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dispatch_id": "20260413-100000",
        "started_at": "2026-04-13T10:00:00Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "",
        "fleet": [],
        "jobs": [
            {
                "run_id": "r-1",
                "task_id": "Isaac-Ant-Direct-v0",
                "framework": "rsl_rl",
                "backend": "physx",
                "num_envs": 4096,
                "max_iterations": 300,
                "seed": 42,
                "bundle_dir_name": "r-1",
                "status": "pending",
                "assigned_to": None,
                "attempts": 0,
                "started_at": None,
                "ended_at": None,
                "running_substate": None,
                "preferred_not": [],
                "per_job_timeout_s": None,
                "osmo_task_name": None,
                "last_heartbeat_at": None,
                "failure": None,
            }
        ],
        "skipped": [],
        "quarantined_hosts": [],
        "dispatcher": "osmo",
        "osmo_workflow_id": "wf-legacy",
        "parent_dispatch_id": None,
        # Note: no osmo_workflow_ids key.
    }
    (tmp_path / "dispatch.json").write_text(json.dumps(payload))

    reloaded = read_dispatch_state(tmp_path)
    assert reloaded is not None
    assert reloaded.osmo_workflow_ids == ["wf-legacy"]
    # Single-field accessor still works for old-reader compat.
    assert reloaded.osmo_workflow_id == "wf-legacy"


def test_both_fields_on_disk_list_wins(tmp_path: Path):
    """When both fields are written, the list field is authoritative.

    We write both for old-reader compatibility, so a mid-migration reader
    sees the right state.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dispatch_id": "20260513-100000",
        "started_at": "2026-05-13T10:00:00Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "",
        "fleet": [],
        "jobs": [],
        "skipped": [],
        "quarantined_hosts": [],
        "dispatcher": "osmo",
        # Stale single-id; list is the canonical source post-migration.
        "osmo_workflow_id": "wf-stale",
        "osmo_workflow_ids": ["wf-a", "wf-b"],
        "parent_dispatch_id": None,
    }
    (tmp_path / "dispatch.json").write_text(json.dumps(payload))
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded is not None
    assert reloaded.osmo_workflow_ids == ["wf-a", "wf-b"]


def test_write_emits_both_fields_for_compat(tmp_path: Path):
    """Writers populate both the list and the legacy single-id key.

    The single-id key is the FIRST entry in the list (or ``None`` if
    empty). Old readers — pre-this-migration code or external tooling
    — that only look at ``osmo_workflow_id`` get a sensible default.
    """
    state = _empty_state()
    state.osmo_workflow_ids = ["wf-a", "wf-b"]
    write_dispatch_state(tmp_path, state)
    raw = json.loads((tmp_path / "dispatch.json").read_text())
    assert raw["osmo_workflow_ids"] == ["wf-a", "wf-b"]
    assert raw["osmo_workflow_id"] == "wf-a"


def test_write_empty_list_serializes_legacy_field_as_none(tmp_path: Path):
    state = _empty_state()
    state.osmo_workflow_ids = []
    write_dispatch_state(tmp_path, state)
    raw = json.loads((tmp_path / "dispatch.json").read_text())
    assert raw["osmo_workflow_ids"] == []
    assert raw["osmo_workflow_id"] is None


def test_asgard_only_dispatch_unchanged(tmp_path: Path):
    """A dispatcher==asgard state with no OSMO workflow id stays empty.

    Asgard-only dispatches never set ``osmo_workflow_id``; the migration
    must not synthesize one for them.
    """
    state = _empty_state(dispatcher="asgard")
    write_dispatch_state(tmp_path, state)
    reloaded = read_dispatch_state(tmp_path)
    assert reloaded is not None
    assert reloaded.osmo_workflow_ids == []
    assert reloaded.osmo_workflow_id is None
