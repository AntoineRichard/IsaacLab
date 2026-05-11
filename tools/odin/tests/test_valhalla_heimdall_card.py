# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer tests for the Heimdall dashboard card."""

from __future__ import annotations

import json

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.heimdall_card import (
    render_empty_state,
    render_heimdall_card,
)


def _payload(hosts: dict, recent_events: list) -> dict:
    return {
        "schema_version": "1.0",
        "generated_at": "2026-05-08T14:32:18Z",
        "hosts": hosts,
        "recent_events": recent_events,
    }


def test_render_empty_state_returns_placeholder():
    div = render_empty_state()
    text = json.dumps(div, default=str)
    assert "Heimdall not active" in text or "no fleet.json" in text


def test_render_card_shows_last_lookup_timestamp():
    payload = _payload(
        hosts={
            "a": {
                "name": "a",
                "healthy": True,
                "last_probe_at": "2026-05-08T14:32:18Z",
                "consecutive_failures": 0,
                "failure_reason": None,
                "recovery_attempts": 0,
                "recovery_history": [],
                "quarantined": False,
            }
        },
        recent_events=[],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:05Z")
    text = json.dumps(div, default=str)
    assert "2026-05-08T14:32:18Z" in text
    assert "47s ago" in text


def test_render_card_marks_unhealthy_with_reason():
    payload = _payload(
        hosts={
            "b": {
                "name": "b",
                "healthy": False,
                "last_probe_at": "2026-05-08T14:32:18Z",
                "consecutive_failures": 2,
                "failure_reason": "ssh_timeout",
                "recovery_attempts": 0,
                "recovery_history": [],
                "quarantined": False,
            }
        },
        recent_events=[],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:00Z")
    text = json.dumps(div, default=str)
    assert "b" in text
    assert "ssh_timeout" in text


def test_render_card_marks_quarantined():
    payload = _payload(
        hosts={
            "c": {
                "name": "c",
                "healthy": False,
                "last_probe_at": "2026-05-08T14:32:18Z",
                "consecutive_failures": 5,
                "failure_reason": "ssh_timeout",
                "recovery_attempts": 1,
                "recovery_history": ["2026-05-08T14:30:00Z"],
                "quarantined": True,
            }
        },
        recent_events=[],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:00Z")
    text = json.dumps(div, default=str)
    assert "quarantined" in text.lower() or "⛔" in text


def test_render_card_shows_recent_events():
    payload = _payload(
        hosts={},
        recent_events=[
            {"ts": "2026-05-08T14:31:00Z", "kind": "host_flipped", "host": "b", "reason": "ssh_timeout"},
            {"ts": "2026-05-08T14:31:05Z", "kind": "host_quarantined", "host": "b", "reason": "recovery_failed"},
        ],
    )
    div = render_heimdall_card(payload, now_iso="2026-05-08T14:33:00Z")
    text = json.dumps(div, default=str)
    assert "host_flipped" in text
    assert "host_quarantined" in text


def test_render_card_handles_none_payload():
    div = render_heimdall_card(None, now_iso="2026-05-08T14:33:00Z")
    text = json.dumps(div, default=str)
    assert "Heimdall not active" in text or "no fleet.json" in text
