# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``fleet.json`` read/write round-trip + atomic-write semantics."""

from __future__ import annotations

from tools.odin.asgard.heimdall import (
    FLEET_JSON_SCHEMA_VERSION,
    HostHealth,
    read_fleet_json,
    write_fleet_json,
)


def test_fleet_json_round_trip(tmp_path):
    hosts = {
        "host-a": HostHealth(
            name="host-a",
            healthy=True,
            last_probe_at="2026-05-08T14:32:18Z",
            consecutive_failures=0,
            failure_reason=None,
            recovery_attempts=0,
            recovery_history=[],
            quarantined=False,
        ),
        "host-b": HostHealth(
            name="host-b",
            healthy=False,
            last_probe_at="2026-05-08T14:32:18Z",
            consecutive_failures=2,
            failure_reason="ssh_timeout",
            recovery_attempts=1,
            recovery_history=["2026-05-08T14:30:00Z"],
            quarantined=True,
        ),
    }
    recent_events = [
        {
            "ts": "2026-05-08T14:31:00Z",
            "kind": "host_flipped",
            "host": "host-b",
            "reason": "ssh_timeout",
        },
    ]

    write_fleet_json(
        tmp_path,
        generated_at="2026-05-08T14:32:18Z",
        hosts=hosts,
        recent_events=recent_events,
    )

    payload = read_fleet_json(tmp_path)
    assert payload is not None
    assert payload["schema_version"] == FLEET_JSON_SCHEMA_VERSION
    assert payload["generated_at"] == "2026-05-08T14:32:18Z"
    assert payload["hosts"]["host-a"]["healthy"] is True
    assert payload["hosts"]["host-b"]["quarantined"] is True
    assert payload["hosts"]["host-b"]["recovery_history"] == ["2026-05-08T14:30:00Z"]
    assert payload["recent_events"] == recent_events


def test_fleet_json_missing_returns_none(tmp_path):
    assert read_fleet_json(tmp_path) is None


def test_fleet_json_atomic_write_no_partial(tmp_path):
    """A failed write must not leave the .tmp behind, nor truncate fleet.json."""
    write_fleet_json(
        tmp_path,
        generated_at="2026-05-08T14:32:18Z",
        hosts={},
        recent_events=[],
    )
    initial = (tmp_path / "fleet.json").read_text()

    class NotSerializable:
        pass

    # Pass an object that asdict() will reject as not-a-dataclass.
    try:
        write_fleet_json(
            tmp_path,
            generated_at="2026-05-08T14:33:18Z",
            hosts={"host-a": NotSerializable()},  # type: ignore[dict-item]
            recent_events=[],
        )
    except (TypeError, AttributeError):
        pass

    assert (tmp_path / "fleet.json").read_text() == initial
    leftovers = list(tmp_path.glob(".fleet_*.json.tmp"))
    assert leftovers == []
