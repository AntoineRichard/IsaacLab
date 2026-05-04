# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wire Tab A's dcc.Interval and pattern-matching callbacks."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import dash
from dash import ALL, Input, Output, State

from tools.odin.valhalla.dashboard.data import DataLayer
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.fleet_table import render_fleet_table
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.header import render_header
from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.jobs_table import render_jobs_rows

__all__ = ["register_callbacks"]


def register_callbacks(app: dash.Dash, data: DataLayer) -> None:
    """Register the Tab A callbacks against the layout's slot ids."""

    @app.callback(
        Output("tab-a-header", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
    )
    def _update_header(_n, dispatch_id):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_header_children(data, dispatch_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a header callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to load dispatch.json", exc)

    @app.callback(
        Output("tab-a-fleet-table", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
    )
    def _update_fleet(_n, dispatch_id):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_fleet_children(data, dispatch_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a fleet callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render fleet table", exc)

    @app.callback(
        Output("tab-a-jobs-rows-content", "children"),
        Input("tab-a-tick", "n_intervals"),
        Input("tab-a-dispatch-id", "data"),
        Input("tab-a-status-filter", "value"),
        Input("tab-a-kind-filter", "value"),
        Input("tab-a-task-text", "value"),
        Input("tab-a-failure-filter", "data"),
        Input("tab-a-expanded-run-ids", "data"),
        Input("tab-a-ssh-tail-store", "data"),
        Input("tab-a-running-tail-shown", "data"),
        Input("tab-a-running-tail-store", "data"),
        Input("tab-a-retry-bump", "data"),
    )
    def _update_jobs(
        _n,
        dispatch_id,
        status_filter,
        kind_filter,
        task_text,
        failure_filter,
        expanded_run_ids,
        ssh_tail_store,
        running_tail_shown,
        running_tail_store,
        _retry_bump,
    ):
        if not dispatch_id:
            return dash.no_update
        try:
            return _compute_jobs_children(
                data,
                dispatch_id=dispatch_id,
                status_filter=status_filter,
                kind_filter=kind_filter,
                task_text=task_text or "",
                failure_filter=failure_filter,
                expanded_run_ids=expanded_run_ids or [],
                ssh_tail_store=ssh_tail_store or {},
                running_tail_shown=running_tail_shown or [],
                running_tail_store=running_tail_store or {},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] tab-a jobs callback: {type(exc).__name__}: {exc}", file=sys.stderr)
            return _error_banner("Failed to render jobs section", exc)

    @app.callback(
        Output("tab-a-failure-filter", "data"),
        Output("tab-a-kind-filter", "value"),
        Input({"type": "tab-a-failure-pill", "kind": ALL}, "n_clicks"),
        State({"type": "tab-a-failure-pill", "kind": ALL}, "id"),
    )
    def _on_failure_pill(n_clicks_list, ids_list):
        if not n_clicks_list or not any(n_clicks_list):
            return dash.no_update, dash.no_update
        # Pick the most-recently clicked pill: any with n_clicks > 0; the
        # last one in the list wins under Dash's pattern-matching event order.
        for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
            if n and n > 0:
                return _handle_pill_click(ident["kind"])
        return dash.no_update, dash.no_update

    @app.callback(
        Output("tab-a-expanded-run-ids", "data"),
        Input({"type": "tab-a-expand-toggle", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-expand-toggle", "run_id": ALL}, "id"),
        State("tab-a-expanded-run-ids", "data"),
    )
    def _on_expand_toggle(n_clicks_list, ids_list, current):
        return _on_expand_toggle_handler(n_clicks_list, ids_list, current=current)

    @app.callback(
        Output("tab-a-ssh-tail-store", "data"),
        Input({"type": "tab-a-ssh-tail-button", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-ssh-tail-button", "run_id": ALL}, "id"),
        State("tab-a-dispatch-id", "data"),
        State("tab-a-ssh-tail-store", "data"),
    )
    def _on_ssh_tail(n_clicks_list, ids_list, dispatch_id, store):
        return _on_ssh_tail_handler(
            n_clicks_list, ids_list, data=dispatch_id, current_store=store, runs_root=data._runs_root
        )

    @app.callback(
        Output("tab-a-running-tail-shown", "data"),
        Input({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "id"),
        State("tab-a-running-tail-shown", "data"),
    )
    def _on_running_tail_toggle(n_clicks_list, ids_list, current):
        return _on_running_tail_toggle_handler(
            n_clicks_list, ids_list, current=current, triggered_id=dash.ctx.triggered_id
        )

    @app.callback(
        Output("tab-a-running-tail-store", "data"),
        Input({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "n_clicks"),
        Input({"type": "tab-a-running-tail-refresh", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-running-tail-toggle", "run_id": ALL}, "id"),
        State({"type": "tab-a-running-tail-refresh", "run_id": ALL}, "id"),
        State("tab-a-dispatch-id", "data"),
        State("tab-a-running-tail-shown", "data"),
        State("tab-a-running-tail-store", "data"),
        prevent_initial_call=True,
    )
    def _on_running_tail_fetch(
        toggle_clicks,
        refresh_clicks,
        toggle_ids,
        refresh_ids,
        dispatch_id,
        current_shown,
        current_store,
    ):
        return _on_running_tail_fetch_handler(
            toggle_clicks,
            refresh_clicks,
            toggle_ids,
            refresh_ids,
            dispatch_id=dispatch_id,
            current_shown=current_shown,
            current_store=current_store,
            data=data,
            triggered_id=dash.ctx.triggered_id,
        )

    @app.callback(
        Output("tab-a-retry-bump", "data"),
        Input({"type": "tab-a-retry-toggle", "run_id": ALL}, "n_clicks"),
        State({"type": "tab-a-retry-toggle", "run_id": ALL}, "id"),
        State("tab-a-dispatch-id", "data"),
        State("tab-a-retry-bump", "data"),
    )
    def _on_retry_toggle(n_clicks_list, ids_list, dispatch_id, bump):
        return _on_retry_toggle_handler(n_clicks_list, ids_list, dispatch_id=dispatch_id, bump=bump, data=data)

    @app.callback(
        Output("tab-a-cancel-pending", "data"),
        Input({"type": "tab-a-cancel-toggle", "run_id": ALL}, "n_clicks"),
        Input("tab-a-cancel-revert", "n_intervals"),
        State({"type": "tab-a-cancel-toggle", "run_id": ALL}, "id"),
        State("tab-a-dispatch-id", "data"),
        State("tab-a-cancel-pending", "data"),
    )
    def _on_cancel_toggle(
        n_clicks_list,
        _n_intervals,
        ids_list,
        dispatch_id,
        pending_store,
    ):
        from time import monotonic

        now_ms = int(monotonic() * 1000)
        triggered = dash.ctx.triggered_id
        if triggered == "tab-a-cancel-revert":
            return _on_cancel_revert_handler(pending_store=pending_store, now_ms=now_ms)
        # Resolve per-row statuses + dispatch.ended_at from the dispatch payload
        # so we don't need a parallel dcc.Store. data.load_dispatch is cached.
        payload = data.load_dispatch(dispatch_id) if dispatch_id else {}
        status_by_run = {j.get("run_id"): j.get("status", "") for j in payload.get("jobs", []) or []}
        statuses = [status_by_run.get(ident["run_id"], "") for ident in (ids_list or [])]
        dispatch_ended = bool(payload.get("ended_at"))
        return _on_cancel_toggle_handler(
            n_clicks_list,
            ids_list,
            statuses=statuses,
            dispatch_id=dispatch_id,
            dispatch_ended=dispatch_ended,
            pending_store=pending_store,
            data=data,
            now_ms=now_ms,
        )


# -- pure helpers (testable without the Dash callback graph) ----------------------


def _compute_header_children(data: DataLayer, dispatch_id: str):
    payload = data.load_dispatch(dispatch_id)
    return render_header(payload)


def _compute_fleet_children(data: DataLayer, dispatch_id: str):
    payload = data.load_dispatch(dispatch_id)
    hardware = data.load_hardware(dispatch_id)
    return render_fleet_table(payload, hardware, data.lookup_hardware)


def _compute_jobs_children(
    data: DataLayer,
    *,
    dispatch_id: str,
    status_filter: list[str] | None,
    kind_filter: list[str] | None,
    task_text: str,
    failure_filter: str | None,
    expanded_run_ids: list[str],
    ssh_tail_store: dict[str, list[str]],
    running_tail_shown: list[str] | None = None,
    running_tail_store: dict[str, dict] | None = None,
):
    payload = data.load_dispatch(dispatch_id)
    effective_kind = list(kind_filter or [])
    if failure_filter and failure_filter not in effective_kind:
        effective_kind.append(failure_filter)
    retry_queue = data.read_retry_queue(dispatch_id)
    return render_jobs_rows(
        payload,
        status_filter=status_filter or None,
        kind_filter=effective_kind or None,
        task_text=task_text or "",
        expanded_run_ids=set(expanded_run_ids or []),
        ssh_tail_store=ssh_tail_store or {},
        running_tail_shown=set(running_tail_shown or []),
        running_tail_store=running_tail_store or {},
        retry_queue=retry_queue,
    )


def _handle_pill_click(kind: str) -> tuple[str, list[str]]:
    """Return (failure-filter store value, kind-dropdown value) for a pill click."""
    return kind, [kind]


def _error_banner(message: str, exc: Exception):
    from dash import html

    return html.Div(
        className="tab-a-error-banner",
        children=[html.Strong(message), f": {type(exc).__name__}: {exc}"],
    )


def _on_expand_toggle_handler(n_clicks_list, ids_list, *, current):
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update
    # Find the latest non-zero click; toggle that run_id.
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            return _toggle_run_id(current or [], ident["run_id"])
    return dash.no_update


def _on_retry_toggle_handler(n_clicks_list, ids_list, *, dispatch_id, bump, data):
    """Mark / un-mark a run_id in the retry DB.

    Side effect: ``data.toggle_retry_queue(dispatch_id, run_id)`` writes the
    new state to ``odin_runs/.retry.sqlite`` atomically.

    Returns the next bump value so the jobs-poll callback re-fires; the actual
    queue contents come from re-reading SQLite in the poll callback.
    """
    if not n_clicks_list or not any(n_clicks_list) or not dispatch_id:
        return dash.no_update
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            data.toggle_retry_queue(dispatch_id, ident["run_id"])
            return (bump or 0) + 1
    return dash.no_update


_CANCEL_CONFIRM_WINDOW_MS = 5000


def _on_cancel_toggle_handler(
    n_clicks_list,
    ids_list,
    *,
    statuses: list[str],
    dispatch_id: str,
    dispatch_ended: bool,
    pending_store: dict | None,
    data,
    now_ms: int,
) -> dict:
    """Two-click confirm flow for the per-row cancel button.

    First click on a row flips the store entry to ``{run_id: expires_at_ms}``
    (5 s out). Second click within the window writes the DB row via
    ``data.request_cancel`` and clears the entry. Clicks for finished
    dispatches are dropped.
    """
    if dispatch_ended:
        return {}
    pending_store = dict(pending_store or {})
    if not n_clicks_list or not any(n_clicks_list):
        return pending_store
    for n, ident, status in zip(n_clicks_list, ids_list, statuses):
        if not n or n <= 0:
            continue
        run_id = ident["run_id"]
        existing = pending_store.get(run_id)
        if existing is not None and existing > now_ms:
            # Second click inside the confirm window → write the DB row.
            kind = "kill" if status == "running" else "skip"
            data.request_cancel(dispatch_id, run_id, kind=kind)
            pending_store.pop(run_id, None)
        elif status in {"pending", "running"}:
            pending_store[run_id] = now_ms + _CANCEL_CONFIRM_WINDOW_MS
    return pending_store


def _on_cancel_revert_handler(*, pending_store: dict | None, now_ms: int) -> dict:
    """Drain expired entries from the pending-confirm store. Run by a 500 ms interval."""
    pending_store = dict(pending_store or {})
    return {run_id: ts for run_id, ts in pending_store.items() if ts > now_ms}


def _on_running_tail_toggle_handler(n_clicks_list, ids_list, *, current, triggered_id=None):
    ident = _clicked_triggered_id(n_clicks_list, ids_list, triggered_id)
    if ident is None and not (isinstance(triggered_id, dict) and triggered_id.get("run_id")):
        ident = _last_clicked_id(n_clicks_list, ids_list)
    if ident is None:
        return dash.no_update
    return _toggle_run_id(current or [], ident["run_id"])


def _on_running_tail_fetch_handler(
    toggle_clicks,
    refresh_clicks,
    toggle_ids,
    refresh_ids,
    *,
    dispatch_id,
    current_shown,
    current_store,
    data,
    triggered_id,
):
    if triggered_id is None or not dispatch_id:
        return dash.no_update
    run_id = triggered_id.get("run_id") if isinstance(triggered_id, dict) else None
    if not run_id:
        return dash.no_update

    current_shown = current_shown or []
    current_store = current_store or {}
    triggered_type = triggered_id.get("type")
    if triggered_type == "tab-a-running-tail-toggle":
        if run_id in set(current_shown):
            return dash.no_update
        if run_id in current_store:
            return dash.no_update
        if _clicked_triggered_id(toggle_clicks, toggle_ids, triggered_id) is None:
            return dash.no_update
    elif triggered_type == "tab-a-running-tail-refresh":
        if _clicked_triggered_id(refresh_clicks, refresh_ids, triggered_id) is None:
            return dash.no_update
    else:
        return dash.no_update

    return _compute_running_tail_store(data, dispatch_id, run_id, current_store=current_store)


def _last_clicked_id(n_clicks_list, ids_list):
    if not n_clicks_list or not ids_list or not any(n_clicks_list):
        return None
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            return ident
    return None


def _clicked_triggered_id(n_clicks_list, ids_list, triggered_id):
    if not isinstance(triggered_id, dict) or not triggered_id.get("run_id"):
        return None
    for n, ident in zip(n_clicks_list or [], ids_list or []):
        if ident == triggered_id and n and n > 0:
            return triggered_id
    return None


def _compute_running_tail_store(data, dispatch_id: str, run_id: str, *, current_store: dict):
    payload = data.load_dispatch(dispatch_id)
    job = _find_job(payload, run_id)
    host = job.get("assigned_to") if job else None
    patch = dash.Patch()
    if not job or not host:
        patch[run_id] = {"source": None, "lines": [], "fetched_at": _utc_now_iso()}
        return patch

    host_config = data.lookup_fleet_host_config(dispatch_id, host)
    ssh_key = host_config.get("ssh_key") if host_config else None
    tail_payload = data.read_running_job_tail_payload(
        dispatch_id,
        run_id,
        host=host,
        ssh_user=(host_config or {}).get("ssh_user") or "horde",
        ssh_key=Path(ssh_key) if ssh_key else None,
        container_name=(host_config or {}).get("container_name") or "isaac-lab-base",
        n=50,
    )
    patch[run_id] = {
        "source": tail_payload.get("source"),
        "lines": list(tail_payload.get("lines") or []),
        "warning": tail_payload.get("warning"),
        "fetched_at": _utc_now_iso(),
    }
    return patch


def _find_job(dispatch_payload: dict, run_id: str) -> dict | None:
    for job in dispatch_payload.get("jobs", []) or []:
        if job.get("run_id") == run_id:
            return job
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _toggle_run_id(current: list[str], run_id: str) -> list[str]:
    """Add ``run_id`` to ``current`` (a list) if absent; remove if present."""
    s = set(current)
    if run_id in s:
        s.remove(run_id)
    else:
        s.add(run_id)
    return sorted(s)


def _on_ssh_tail_handler(n_clicks_list, ids_list, *, data, current_store, runs_root=None):
    if not n_clicks_list or not any(n_clicks_list):
        return dash.no_update
    for n, ident in zip(reversed(n_clicks_list), reversed(ids_list)):
        if n and n > 0:
            run_id = ident["run_id"]
            from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.ssh_tail import load_ssh_tail

            new_store = dict(current_store or {})
            if runs_root is None:
                # Test-mode shortcut for the phantom-click test.
                return new_store
            new_store[run_id] = load_ssh_tail(runs_root, data, run_id)
            return new_store
    return dash.no_update


def _compute_ssh_tail_store(data, dispatch_id: str, run_id: str, *, current_store: dict):
    """Test-friendly helper: load the tail and return the new store dict.

    `data` is duck-typed — only `_runs_root` is required.
    """
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.ssh_tail import load_ssh_tail

    new_store = dict(current_store or {})
    new_store[run_id] = load_ssh_tail(data._runs_root, dispatch_id, run_id)
    return new_store
