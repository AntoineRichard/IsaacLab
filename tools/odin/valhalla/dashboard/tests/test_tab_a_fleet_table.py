# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Tab A fleet table."""

from __future__ import annotations

from tools.odin.valhalla.dashboard.data import HardwareInfo
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.fleet_table import render_fleet_table


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            if child is None or isinstance(child, str):
                continue
            yield from _walk(child)
    elif not isinstance(children, str):
        yield from _walk(children)


def _text_blob(component) -> str:
    parts: list[str] = []
    for c in _walk(component):
        ch = getattr(c, "children", None)
        if isinstance(ch, str):
            parts.append(ch)
    return " ".join(parts)


def _has_class(component, target_class: str) -> bool:
    for c in _walk(component):
        cls = getattr(c, "className", "") or ""
        if target_class in cls.split():
            return True
    return False


def _payload(fleet, *, jobs=None):
    return {
        "schema_version": "1.3",
        "dispatch_id": "d",
        "fleet": fleet,
        "jobs": jobs or [],
    }


def _hw_payload(hosts):
    return {
        "schema_version": "1.0",
        "dispatch_id": "d",
        "fingerprint": "gpu:NVIDIA-L40",
        "hosts": hosts,
    }


def _hw_block(*, hostname="h", gpu="NVIDIA L40", mem=44.32, cpu="Xeon", count=16, ram=62.79):
    return {
        "hostname": hostname,
        "gpu_devices": [{"name": gpu, "mem_gb": mem, "compute_cap": "8.9"}],
        "cpu_name": cpu,
        "cpu_count": count,
        "ram_gb": ram,
        "sourced_from": "r1",
    }


def test_fleet_renders_one_row_per_host():
    fleet = [
        {"host": "v1", "status": "idle", "current_run_id": None, "last_error": None},
        {"host": "v2", "status": "busy", "current_run_id": "r1", "last_error": None},
    ]
    component = render_fleet_table(_payload(fleet), None, lambda host: None)
    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    # 1 header row + 2 data rows
    assert len(rows) == 3


def test_fleet_status_pill_idle_busy_down():
    for status, expected_class in [
        ("idle", "tab-a-fleet-status-idle"),
        ("busy", "tab-a-fleet-status-busy"),
        ("down", "tab-a-fleet-status-down"),
    ]:
        fleet = [{"host": "v1", "status": status, "current_run_id": None, "last_error": None}]
        component = render_fleet_table(_payload(fleet), None, lambda h: None)
        assert _has_class(component, expected_class), f"missing class for status={status!r}"


def test_fleet_current_run_link_for_busy_host():
    fleet = [{"host": "v1", "status": "busy", "current_run_id": "rsl-rl_physx_X_seed42", "last_error": None}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    anchors = [c for c in _walk(component) if type(c).__name__ == "A"]
    assert len(anchors) == 1
    assert "seed42" in (getattr(anchors[0], "children", "") or "")


def test_fleet_current_run_dash_when_idle():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    blob = _text_blob(component)
    # Has the em-dash placeholder somewhere on the row.
    assert "—" in blob


def test_fleet_hardware_from_hardware_json():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    hw = _hw_payload({"v1": _hw_block(hostname="Odin-Runner-5")})
    component = render_fleet_table(_payload(fleet), hw, lambda h: None)
    blob = _text_blob(component)
    assert "Odin-Runner-5" in blob
    assert "NVIDIA L40" in blob
    assert "44.32" in blob or "44.3" in blob


def test_fleet_hardware_falls_back_to_lookup():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]

    def fallback(host):
        if host == "v1":
            return HardwareInfo(
                hostname="Odin-Fallback",
                gpu_devices=[{"name": "NVIDIA L40", "mem_gb": 44.32, "compute_cap": "8.9"}],
                cpu_name="Xeon",
                cpu_count=16,
                ram_gb=62.79,
                sourced_from="prev/r1",
            )
        return None

    component = render_fleet_table(_payload(fleet), None, fallback)
    blob = _text_blob(component)
    assert "Odin-Fallback" in blob


def test_fleet_hardware_dash_when_unknown():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    blob = _text_blob(component)
    # Several em-dashes; at minimum the GPU/CPU/RAM cells are dashes.
    assert blob.count("—") >= 3


def test_fleet_last_event_recovered_pill():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": "gpu_lost: recovered"}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    assert _has_class(component, "tab-a-event-recovered")


def test_fleet_last_event_recovery_failed_pill():
    fleet = [
        {
            "host": "v1",
            "status": "down",
            "current_run_id": None,
            "last_error": "gpu_lost: recovery_failed (docker_restart_failed: daemon down)",
        }
    ]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    assert _has_class(component, "tab-a-event-recovery-failed")


def test_fleet_last_event_dash_when_no_error():
    fleet = [{"host": "v1", "status": "idle", "current_run_id": None, "last_error": None}]
    component = render_fleet_table(_payload(fleet), None, lambda h: None)
    rows = [c for c in _walk(component) if type(c).__name__ == "Tr"]
    # The last cell of the data row should be an em-dash.
    last_data_row = rows[-1]
    last_cell = list(_walk(last_data_row))[-1]
    last_cell_text = getattr(last_cell, "children", None)
    assert last_cell_text == "—"


def test_fleet_fallback_lookup_called_at_most_once_per_host():
    fleet = [
        {"host": "v1", "status": "idle", "current_run_id": None, "last_error": None},
        {"host": "v2", "status": "idle", "current_run_id": None, "last_error": None},
    ]
    call_log: list[str] = []

    def fallback(host):
        call_log.append(host)
        return

    render_fleet_table(_payload(fleet), None, fallback)
    assert call_log == ["v1", "v2"]
