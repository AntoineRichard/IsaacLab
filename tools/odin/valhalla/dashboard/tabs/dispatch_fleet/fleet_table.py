# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Tab A fleet table — one row per host, 8 inline columns."""

from __future__ import annotations

from collections.abc import Callable

from dash import html

from tools.odin.valhalla.dashboard.data import HardwareInfo

__all__ = ["render_fleet_table"]


_STATUS_CLASS = {
    "idle": "tab-a-fleet-status-idle",
    "busy": "tab-a-fleet-status-busy",
    "down": "tab-a-fleet-status-down",
}


def render_fleet_table(
    dispatch_payload: dict,
    hardware_payload: dict | None,
    fallback_lookup: Callable[[str], HardwareInfo | None],
) -> html.Div:
    """Build the fleet table.

    Args:
        dispatch_payload: Parsed dispatch.json.
        hardware_payload: Parsed hardware.json (or None for pre-feature dispatches).
        fallback_lookup: ``DataLayer.lookup_hardware`` — called at most once per host.

    Returns:
        A Div containing a table with one header row and one data row per host.
    """
    fleet = dispatch_payload.get("fleet", []) or []
    hw_hosts = (hardware_payload or {}).get("hosts", {}) or {}
    fallback_cache: dict[str, HardwareInfo | None] = {}

    header = html.Tr(
        children=[
            html.Th("Host"),
            html.Th("Hostname"),
            html.Th("Status"),
            html.Th("Current run"),
            html.Th("GPU"),
            html.Th("CPU"),
            html.Th("RAM"),
            html.Th("Last event"),
        ]
    )

    rows: list = []
    for host_entry in fleet:
        host = str(host_entry.get("host", ""))
        status = str(host_entry.get("status", "unknown"))
        current_run_id = host_entry.get("current_run_id")
        last_error = host_entry.get("last_error")

        hw = _resolve_hardware(host, hw_hosts, fallback_lookup, fallback_cache)
        hostname_cell = hw["hostname"] if hw else "—"
        gpu_cell = _gpu_cell(hw)
        cpu_cell = _cpu_cell(hw)
        ram_cell = _ram_cell(hw)

        rows.append(
            html.Tr(
                children=[
                    html.Td(host, className="tab-a-fleet-host"),
                    html.Td(hostname_cell),
                    html.Td(_status_pill(status)),
                    html.Td(_current_run_cell(current_run_id)),
                    html.Td(gpu_cell),
                    html.Td(cpu_cell),
                    html.Td(ram_cell),
                    html.Td(_last_event_cell(last_error)),
                ]
            )
        )

    return html.Div(
        id="tab-a-fleet-table-content",
        children=[
            html.Table(
                children=[
                    html.Thead(children=[header]),
                    html.Tbody(children=rows),
                ],
                className="tab-a-fleet-table",
            )
        ],
    )


def _resolve_hardware(
    host: str,
    hw_hosts: dict,
    fallback_lookup: Callable[[str], HardwareInfo | None],
    cache: dict[str, HardwareInfo | None],
) -> dict | None:
    """Return the hardware block for ``host`` as a plain dict, or None."""
    direct = hw_hosts.get(host)
    if direct:
        return direct
    if host not in cache:
        cache[host] = fallback_lookup(host)
    info = cache[host]
    if info is None:
        return None
    return {
        "hostname": info.hostname,
        "gpu_devices": info.gpu_devices,
        "cpu_name": info.cpu_name,
        "cpu_count": info.cpu_count,
        "ram_gb": info.ram_gb,
    }


def _status_pill(status: str) -> html.Span:
    cls = _STATUS_CLASS.get(status, "tab-a-fleet-status-unknown")
    label = status.capitalize()
    return html.Span(label, className=f"tab-a-pill {cls}")


def _current_run_cell(current_run_id):
    if not current_run_id:
        return "—"
    short_text = f"…{current_run_id[-30:]}" if len(current_run_id) > 30 else current_run_id
    return html.A(short_text, href="#", title=current_run_id)


def _gpu_cell(hw: dict | None):
    if not hw or not hw.get("gpu_devices"):
        return "—"
    g = hw["gpu_devices"][0]
    return f"{g.get('name', '?')} · {g.get('mem_gb', 0):.2f} GB"


def _cpu_cell(hw: dict | None):
    if not hw:
        return "—"
    return f"{hw.get('cpu_name', '?')} ×{hw.get('cpu_count', 0)}"


def _ram_cell(hw: dict | None):
    if not hw:
        return "—"
    return f"{hw.get('ram_gb', 0):.2f} GB"


def _last_event_cell(last_error):
    if not last_error:
        return "—"
    if last_error == "gpu_lost: recovered":
        return html.Span("gpu_lost: recovered", className="tab-a-pill tab-a-event-recovered")
    if last_error.startswith("gpu_lost: recovery_failed"):
        return html.Span(
            "gpu_lost: recovery_failed",
            title=last_error,
            className="tab-a-pill tab-a-event-recovery-failed",
        )
    return last_error
