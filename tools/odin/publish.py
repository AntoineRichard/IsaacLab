# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map fetched benchmark bundles onto the OSMO results table.

The table is ``omni_runtime_isaac_lab_osmo_v3``: one row per benchmark run, with
``result``, ``kpis``, ``startup`` and ``meta`` as jsonb. Its shape is the OmniPerf
legacy shape, so metric names are prose (``"Mean Total FPS"``) and are read by
dashboard SQL as literal paths -- renaming one empties a panel rather than raising.

Everything above :func:`insert_rows` is pure: it maps a schema bundle to the row
dict and is testable offline against ``isaaclab_v3_sample.json``. Only the insert
needs the database, which needs corpnet, which is why publishing runs on the
dispatching machine after ``odin fetch`` rather than inside an OSMO task.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    "TABLE",
    "PublishError",
    "build_row",
    "bundle_to_kpi_phases",
    "comparison_group_for",
    "collect_rows",
    "insert_rows",
    "preset_slug",
    "run_key_for",
    "scope_of",
]

TABLE = "omni_runtime_isaac_lab_osmo_v3"

# The thermal phase the dashboards read. Odin runs each row once, so there is no
# cold/warm pair to report; the single run is the warm one.
_PHASE = "WARM"


class PublishError(RuntimeError):
    """Raised when a bundle cannot be mapped or the insert fails."""


def _mean(node: Any) -> float | None:
    return node.get("mean") if isinstance(node, dict) else None


def _stat(node: Any, key: str) -> float | None:
    return node.get(key) if isinstance(node, dict) else None


def _ms(value: float | None) -> float | None:
    return value * 1000.0 if isinstance(value, (int, float)) else None


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    """Assign only real values, so absent metrics stay absent rather than null."""
    if value is not None:
        target[key] = value


def _spread(target: dict[str, Any], name: str, node: Any, *, scale: float = 1.0) -> None:
    """Write a ``{mean,std,peak}`` triple as ``X`` / ``X std`` / ``X peak``.

    The table names the mean without a suffix, which is why this is not a loop over
    the three keys.
    """
    if not isinstance(node, dict):
        return
    for suffix, key in (("", "mean"), (" std", "std"), (" peak", "peak")):
        value = _stat(node, key)
        if isinstance(value, (int, float)):
            target[f"{name}{suffix}"] = value * scale


def _renderer(cfg: dict[str, Any]) -> str | None:
    """Return the renderer token, treating the literal ``"none"`` as absent.

    Bundles spell "ran headless" as ``rendering_backend: "none"`` rather than null, so
    a naive read puts ``none`` into the run key and the preset string.
    """
    value = cfg.get("rendering_backend")
    return value if value not in (None, "", "none") else None


def scope_of(task_id: str) -> str:
    """Return ``core`` or ``contrib`` for a task id.

    Derived from the id rather than carried in the bundle, the same way discovery
    decides it. Redundant with the ``IsaacContrib-`` prefix, but naming it explicitly
    lets a dashboard group or filter on maturity without pattern-matching the task.
    """
    return "contrib" if str(task_id).startswith("IsaacContrib-") else "core"


def preset_slug(cfg: dict[str, Any]) -> str:
    """Return the run's presets as one sorted, dash-separated token.

    Every axis is a preset -- physics, renderer and domain alike, which is how the
    existing ``OMNIPERF_ISAACLAB_PRESET`` encoding treats them. Naming only the
    physics backend was enough while every task was state-only, but a camera task
    varies by renderer and by domain preset, and those rows would otherwise collide
    under one key.

    Sorted so a configuration yields the same slug however the tokens were ordered in
    the row: the key has to be stable for a trend line to stay continuous. Dashes
    separate presets because the names themselves contain underscores.
    """
    tokens = {t for t in (cfg.get("physics_backend"), _renderer(cfg), *(cfg.get("presets") or [])) if t}
    return "-".join(sorted(tokens)) or "default"


def run_key_for(bundle: dict[str, Any], kind: str) -> str:
    """Return the ``kpis`` level-2 key identifying one run.

    The sample uses ``benchmark_non_rl_<task>_r<res>_<envs>``: the producing workflow,
    the task, and the parameters that make two runs of it different. Odin's equivalent
    discriminators are the RL library, the physics backend and the environment count --
    without the backend, every row of a task collides under one key.

    Note:
        Unconfirmed against the dashboard, which indexes this by literal path. Agree the
        spelling with the table owner before treating published rows as visible.
    """
    run = bundle.get("run") or {}
    cfg = run.get("config") or {}
    task = run.get("task", "unknown")
    parts = [f"benchmark_{kind}", scope_of(task), task, run.get("framework", "unknown"), preset_slug(cfg)]
    if run.get("num_envs") is not None:
        parts.append(f"n{run['num_envs']}")
    return "_".join(str(p) for p in parts)


def comparison_group_for(bundle: dict[str, Any], kind: str) -> str:
    """Return the dashboard group name identifying this configuration.

    A group is a configuration tracked over time, not a single run, so the seed is
    deliberately absent: seeds of one config share a group and their spread is the
    signal. Mirrors the existing convention, which is uppercase and composite --
    ``ISAAC_LAB_SINGLE_GPU_GTL_PRESET_NEWTON_ISAACSIMRTX_RGB_E8192_R64``.
    """
    run = bundle.get("run") or {}
    cfg = run.get("config") or {}
    task = str(run.get("task", "UNKNOWN"))
    parts = ["ISAAC_LAB", "OSMO", kind, scope_of(task), task, str(run.get("framework", "UNKNOWN"))]
    if cfg.get("physics_backend"):
        parts.append(str(cfg["physics_backend"]))
    if _renderer(cfg):
        parts.append(str(_renderer(cfg)))
    for preset in cfg.get("presets") or []:
        if preset not in parts:
            parts.append(str(preset))
    if run.get("num_envs") is not None:
        parts.append(f"E{run['num_envs']}")
    return "_".join(parts).upper().replace("-", "_")


def bundle_to_kpi_phases(bundle: dict[str, Any], *, workflow_name: str) -> dict[str, Any]:
    """Convert one schema bundle into the table's per-run KPI phases.

    Returns the ``{startup, runtime, version_info, hardware_info}`` mapping that sits
    under ``kpis.<phase>.<run_key>``. ``sim_runtime``/``frametime`` are Kit-profiler
    phases and are omitted: most Odin rows are kitless and have no such data, and an
    empty phase is a worse answer than an absent one.

    Args:
        bundle: A schema bundle as written by ``isaaclab benchmark``.
        workflow_name: Value for the ``workflow_name`` field each phase carries.

    Returns:
        Phase name to metric mapping, with units matching the table: times in
        milliseconds, memory in GB, utilisation in percent.
    """
    runtime = bundle.get("runtime") or {}
    resources = bundle.get("resources") or {}
    run = bundle.get("run") or {}
    startup_s = runtime.get("startup_time_s") or {}

    startup: dict[str, Any] = {"phase": "startup", "workflow_name": workflow_name}
    for label, field in (
        ("App Launch Time", "app_launch"),
        ("Python Imports Time", "python_imports"),
        ("Task Creation and Start Time", "task_config"),
        ("Scene Creation Time", "env_creation"),
        ("Simulation Start Time", "first_step"),
    ):
        _put(startup, label, _ms(startup_s.get(field)))
    components = [v for v in startup_s.values() if isinstance(v, (int, float))]
    if components:
        startup["Total Start Time (Launch to Train)"] = _ms(sum(components))

    rt: dict[str, Any] = {"phase": "runtime", "workflow_name": workflow_name}
    _put(rt, "num_envs", run.get("num_envs"))
    _spread(rt, "Mean Total FPS", runtime.get("total_fps"))
    _spread(rt, "Mean Collection FPS", runtime.get("collection_fps"))
    _spread(rt, "Mean Iterations per Second", runtime.get("iterations_per_s"))
    _spread(rt, "Mean Iteration Time", runtime.get("iteration_time_s"), scale=1000.0)
    _put(rt, "Iterations Completed", runtime.get("iterations_completed"))
    _put(rt, "Steps per Iteration", runtime.get("steps_per_iteration"))
    _put(rt, "Total Wall Time", runtime.get("total_wall_time_s"))
    # Recorder telemetry. The table splits each into value/std/peak the same way.
    _spread(rt, "GPU Memory Used", resources.get("gpu_mem_gb"))
    _spread(rt, "GPU Utilization", resources.get("gpu_util_pct"))
    _spread(rt, "CPU Utilization", resources.get("cpu_util_pct"))
    _spread(rt, "System Memory RSS", resources.get("ram_gb"))
    step = runtime.get("environment_step_timing") or {}
    _spread(rt, "Mean Environment step FPS", step.get("environment_step_fps"))
    _spread(rt, "Mean Environment step times", step.get("environment_step_time_s"), scale=1000.0)

    phases: dict[str, Any] = {"startup": startup, "runtime": rt}

    learning = bundle.get("learning") or {}
    success = bundle.get("success_rate")
    reward = (learning.get("reward") or {}).get("final_ema")
    if learning or success is not None:
        train: dict[str, Any] = {"phase": "train", "workflow_name": workflow_name}
        _put(train, "Success Rate (tail mean)", success)
        _put(train, "Max Rewards", reward)
        _put(train, "Max Episode Lengths", (learning.get("ep_length") or {}).get("final_ema"))
        phases["train"] = train

    versions = bundle.get("versions") or {}
    version_info: dict[str, Any] = {"phase": "version_info", "workflow_name": workflow_name}
    for label, field in (
        ("kit_version", "kit"),
        ("isaacsim_version", "isaacsim"),
        ("isaaclab_version", "isaaclab"),
        ("warp_version", "warp"),
        ("torch_version", "torch"),
        ("numpy_version", "numpy"),
        ("newton_version", "newton"),
        ("skrl_version", "skrl"),
        ("dev", "git_commit"),
    ):
        _put(version_info, label, versions.get(field))
    phases["version_info"] = version_info

    hardware = bundle.get("hardware") or {}
    hardware_info: dict[str, Any] = {"phase": "hardware_info", "workflow_name": workflow_name}
    _put(hardware_info, "cpu_name", hardware.get("cpu_name"))
    _put(hardware_info, "physical_cores", hardware.get("cpu_count"))
    _put(hardware_info, "total_ram_gb", hardware.get("ram_gb"))
    gpus = hardware.get("gpu_devices") or []
    if gpus:
        hardware_info["gpu_devices"] = gpus
        hardware_info["gpu_device_count"] = len(gpus)
    phases["hardware_info"] = hardware_info

    return phases


def build_row(
    bundle: dict[str, Any],
    *,
    kind: str,
    dispatch_id: str,
    row_key: str,
    image_ref: str,
    pool: str,
    workflow_id: str | None = None,
    submitter: str | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one table row from one bundle.

    Args:
        bundle: Schema bundle from a completed benchmark.
        kind: ``training`` or ``play``; distinguishes the two bundles a row emits.
        dispatch_id: Odin dispatch id, ``YYYYMMDD-HHMMSS``.
        row_key: Odin row identity, unique within a dispatch.
        image_ref: Digest-pinned image the run used.
        pool: OSMO pool the run landed on.
        workflow_id: OSMO workflow the task belonged to.
        submitter: Who dispatched; defaults to ``$USER``.
        artifacts: Files that landed alongside the bundle, as ``{"checkpoint": name,
            "video": name, "results_uri": uri}``. Recorded in ``meta`` only: the table
            has no artifact columns, and video/checkpoint support is post-GA. Keeping
            the reference now means the rows are already linked when it arrives.

    Returns:
        Column name to value, ready for :func:`insert_rows`. ``result``, ``kpis``,
        ``startup`` and ``meta`` are plain dicts; the caller adapts them to jsonb.
    """
    run = bundle.get("run") or {}
    cfg = run.get("config") or {}
    versions = bundle.get("versions") or {}
    hardware = bundle.get("hardware") or {}
    workflow_name = f"benchmark_{kind}"
    key = run_key_for(bundle, kind)

    gpu = (hardware.get("gpu_devices") or [{}])[0].get("name", "unknown")
    # Mirrors the sample's OMNIPERF_ISAACLAB_PRESET encoding so existing dashboard
    # filters keep working; meta carries the same values as queryable fields.
    # Backends are mirrored into ``presets`` on some tasks, so the naive concatenation
    # yields ``newton_mjwarp,newton_mjwarp``. Order is kept: physics, renderer, domain.
    preset_tokens: list[str] = []
    for token in (cfg.get("physics_backend"), _renderer(cfg), *(cfg.get("presets") or [])):
        if token and token not in preset_tokens:
            preset_tokens.append(token)

    session = {
        # Includes the kind: a row emits a training and a play bundle, and without it
        # the two collide on one id.
        "job_id": f"{dispatch_id}:{row_key}:{kind}",
        "app_type": "isaaclab",
        # The sample reports an Isaac Lab commit here, not an Isaac Sim version, which
        # is what makes this field meaningful for kitless runs.
        "app_version": versions.get("git_commit") or "",
        "app_creation_time": run.get("start_time_utc") or "",
        "timestamp": run.get("start_time_utc") or "",
        "benchmark_type": kind.upper(),
        "image_url": image_ref,
        "kit_envs": f"OMNIPERF_ISAACLAB_PRESET={','.join(preset_tokens)}" if preset_tokens else "",
        "kit_args": "",
        "macro_file_path": "",
        "platform_type": f"{pool}:{gpu}",
        "profiler_type": "none",
        "stage": "single_gpu",
        "submitter": submitter or os.environ.get("USER", "odin"),
        "viewport_resolution": "",
        "meta": {
            "HOSTNAME": hardware.get("hostname", ""),
            "num_gpus": str(len(hardware.get("gpu_devices") or [])),
        },
    }

    group_name = comparison_group_for(bundle, kind)
    session["meta"]["group_name"] = group_name

    result = {
        "version": 1,
        "comparison_group_name": group_name,
        "comparison_group_index": 0,
        "omniperf_target": "",
        # The dashboards filter on this; a row that is not "success" is invisible.
        "omniperf_type": "success" if run.get("status") == "completed" else "failure",
        "stage_info": {},
        "benchmark_result": {"event_list": [], "metric_info": []},
        "benchmark_storage": {},
        "benchmark_session": session,
        "benchmark_execution_env": {},
        "meta": {},
    }

    phases = bundle_to_kpi_phases(bundle, workflow_name=workflow_name)
    meta = {
        "row_key": row_key,
        "dispatch_id": dispatch_id,
        "kind": kind,
        "task": run.get("task"),
        "scope": scope_of(run.get("task", "")),
        "rl_library": run.get("framework"),
        "physics_backend": cfg.get("physics_backend"),
        "rendering_backend": _renderer(cfg),
        "presets": list(cfg.get("presets") or []),
        "seed": run.get("seed"),
        "num_envs": run.get("num_envs"),
        "max_iterations": run.get("max_iterations"),
        "status": run.get("status"),
        "pool": pool,
        "image_ref": image_ref,
        "osmo_workflow": workflow_id,
        "git_commit": versions.get("git_commit"),
        "git_branch": versions.get("git_branch"),
        "git_dirty": versions.get("git_dirty"),
        "schema_version": bundle.get("schema_version"),
        # Where the artifacts actually are. The bundle's own checkpoint_path is the
        # in-container path and does not resolve anywhere else.
        "artifacts": artifacts or {},
    }

    return {
        "job_id": session["job_id"],
        "benchmark_type": kind.upper(),
        "result": result,
        "kpis": {_PHASE: {key: phases}},
        "startup": {},
        "meta": meta,
    }


def _artifacts_beside(bundle_path: Path, *, dispatch_id: str, results_uri: str | None) -> dict[str, Any]:
    """Return the artifacts sitting next to a bundle, with their storage URIs.

    Reads the fetched tree rather than the bundle, because the bundle records the
    in-container path of a checkpoint, which resolves nowhere once the task is gone.
    """
    row_dir = bundle_path.parent
    found: dict[str, Any] = {}
    checkpoints = sorted(p.name for p in (row_dir / "checkpoint").glob("*") if p.is_file())
    if checkpoints:
        found["checkpoint"] = checkpoints[-1]
    videos = sorted(str(p.relative_to(row_dir)) for p in row_dir.rglob("*.mp4"))
    if videos:
        found["video"] = videos[-1]
    if found and results_uri:
        found["prefix"] = f"{results_uri.rstrip('/')}/{dispatch_id}/{row_dir.name}"
    return found


def collect_rows(
    dispatch_dir: Path,
    *,
    image_ref: str,
    pool: str,
    workflow_id: str | None = None,
    results_uri: str | None = None,
) -> list[dict[str, Any]]:
    """Build a table row for every readable bundle under a fetched dispatch.

    Args:
        dispatch_dir: ``odin_runs/<dispatch_id>``, as written by ``odin fetch``.
        image_ref: Digest-pinned image the dispatch ran.
        pool: OSMO pool the dispatch targeted.
        workflow_id: OSMO workflow id, when a single one covers the dispatch.
        results_uri: Storage prefix the dispatch uploaded to, used to build artifact
            URIs. Omitted leaves ``meta.artifacts`` with names but no location.

    Returns:
        Rows sorted by ``job_id``. Unreadable bundles are skipped rather than
        aborting the batch, so one corrupt file does not cost the whole publish.
    """
    dispatch_id = dispatch_dir.name
    rows: list[dict[str, Any]] = []
    for path in sorted(dispatch_dir.rglob("benchmark_*.json")):
        kind = "play" if path.name.startswith("benchmark_play") else "training"
        try:
            bundle = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(bundle, dict) or "run" not in bundle:
            continue
        rows.append(
            build_row(
                bundle,
                artifacts=_artifacts_beside(path, dispatch_id=dispatch_id, results_uri=results_uri),
                kind=kind,
                dispatch_id=dispatch_id,
                row_key=path.parent.name if path.parent.name != dispatch_id else path.stem,
                image_ref=image_ref,
                pool=pool,
                workflow_id=workflow_id,
            )
        )
    rows.sort(key=lambda row: row["job_id"])
    return rows


def insert_rows(rows: list[dict[str, Any]], *, dsn: str, table: str = TABLE, verify: bool = True) -> int:
    """Insert rows and return how many the database confirms.

    Args:
        rows: Rows from :func:`collect_rows`.
        dsn: libpq connection string. Read it from the environment; never commit it.
        table: Target table.
        verify: Re-count the inserted ``job_id`` values after commit. A silent
            partial write is the failure mode worth paying a query to rule out.

    Returns:
        Number of rows the database reports holding.

    Raises:
        PublishError: If psycopg is unavailable, or the insert fails.
    """
    try:
        import psycopg2
        from psycopg2.extras import Json, execute_batch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PublishError("psycopg2 is required to publish; install it or use --dry_run") from exc

    if not rows:
        return 0

    statement = (
        f"INSERT INTO {table} (job_id, benchmark_type, result, kpis, startup, meta) VALUES (%s, %s, %s, %s, %s, %s)"
    )
    payload = [
        (r["job_id"], r["benchmark_type"], Json(r["result"]), Json(r["kpis"]), Json(r["startup"]), Json(r["meta"]))
        for r in rows
    ]
    try:
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                execute_batch(cur, statement, payload)
            if not verify:
                return len(rows)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT count(*) FROM {table} WHERE job_id = ANY(%s)",
                    ([r["job_id"] for r in rows],),
                )
                return int(cur.fetchone()[0])
    except Exception as exc:  # noqa: BLE001 - psycopg raises a wide family
        raise PublishError(f"insert into {table} failed: {exc}") from exc
