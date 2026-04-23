# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core Valhalla aggregator — rolls bundle JSONs into a single dispatch aggregate.

Input: an ``odin_runs/<dispatch_id>/`` directory produced by T3.1's
dispatcher. The directory holds one ``dispatch.json`` (source of truth
for which jobs existed) plus one ``<run_id>/`` subdirectory per job.

Output: a plain Python ``dict`` matching the T4.1 ``aggregate.json``
schema v1.0. The dict is returned — writing it to disk is
:func:`valhalla.writer.write_aggregate`'s job.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from tools.odin.valhalla.stats import is_divergent, stats_over

__all__ = ["AggregateOptions", "aggregate_dispatch"]

SCHEMA_VERSION = "1.0"

# Metrics aggregated into Stats blobs. Each entry is
# (per-seed field name, training.json dotted path).
_AGGREGATED_METRICS: list[tuple[str, str]] = [
    ("reward_final_ema",      "learning.reward.final_ema"),
    ("ep_length_final_ema",   "learning.ep_length.final_ema"),
    ("iter_time_s_mean",      "runtime.iteration_time_s.mean"),
    ("env_steps_per_s_mean",  "runtime.env_steps_per_s.mean"),
    ("ram_gb_peak",           "resources.ram_gb.peak"),
    ("gpu_mem_gb_peak",       "resources.gpu_mem_gb.peak"),
]


@dataclass
class AggregateOptions:
    """Options for :func:`aggregate_dispatch`.

    Attributes:
        divergence_z: Z-score threshold for :attr:`divergent_seeds`; a seed
            is flagged when ``|value - mean| > divergence_z * std``.
            Default ``2.0``.
    """

    divergence_z: float = 2.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_bundle_jsons(bundle_dir: Path) -> tuple[dict, dict] | None:
    """Return ``(manifest, training)`` if both parse; ``None`` on any error."""
    try:
        with (bundle_dir / "manifest.json").open("r") as fh:
            manifest = json.load(fh)
        with (bundle_dir / "training.json").open("r") as fh:
            training = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return manifest, training


def _bundle_is_completed(manifest: dict, training: dict) -> bool:
    try:
        tr = manifest["phases"]["training"]
        if tr.get("status") != "completed" or tr.get("exit_code") != 0:
            return False
        ver = str(training.get("schema_version", ""))
        return ver.startswith("1.")
    except (KeyError, TypeError):
        return False


def _extract_seed_payload(training: dict, manifest: dict, run_id: str, assigned_to: str | None) -> dict:
    """Build one ``seeds[<N>]`` sub-object from a completed bundle."""
    del manifest  # Only used by callers for commit_sha/hostname collection.
    rt = training["runtime"]
    rew = training["learning"]["reward"]
    ep = training["learning"]["ep_length"]
    res = training["resources"]
    startup_s = rt.get("startup_phase_times_s", {}) or {}
    return {
        "run_id":                  run_id,
        "status":                  "completed",
        "assigned_to":             assigned_to,
        "reward_final_ema":        float(rew["final_ema"]),
        "ep_length_final_ema":     float(ep["final_ema"]),
        "iter_time_s_mean":        float(rt["iteration_time_s"]["mean"]),
        "iter_time_s_std":         float(rt["iteration_time_s"]["std"]),
        "env_steps_per_s_mean":    float(rt["env_steps_per_s"]["mean"]),
        "iterations_completed":    int(rt["iterations_completed"]),
        "total_wall_time_s":       float(rt["total_wall_time_s"]),
        "ram_gb_peak":             float(res["ram_gb"]["peak"]),
        "gpu_mem_gb_peak":         float(res["gpu_mem_gb"]["peak"]),
        "startup_app_launch_s":    float(startup_s.get("app_launch") or 0.0),
        "startup_env_creation_s":  float(startup_s.get("env_creation") or 0.0),
        "startup_first_step_s":    float(startup_s.get("first_step") or 0.0),
    }


def _compute_aggregate(seeds: dict[str, dict], divergence_z: float) -> dict:
    """Compute the ``aggregate`` block + ``divergent_seeds`` for one row."""
    del divergence_z  # divergent_seeds is computed separately.
    if not seeds:
        return {"n_seeds_completed": 0, "n_seeds_failed": 0}
    result: dict = {"n_seeds_completed": len(seeds), "n_seeds_failed": 0}
    for field_name, _ in _AGGREGATED_METRICS:
        values = [s[field_name] for s in seeds.values()]
        s = stats_over(values)
        result[field_name] = {
            "mean":   s.mean,
            "std":    s.std,
            "min":    s.min,
            "max":    s.max,
            "cv_pct": s.cv_pct,
        }
    return result


def _divergent_seed_keys(seeds: dict[str, dict], divergence_z: float) -> list[str]:
    """Return sorted seed-string keys flagged on ``reward_final_ema``."""
    if len(seeds) < 3:
        return []
    ordered = sorted(seeds.keys(), key=int)
    values = [seeds[k]["reward_final_ema"] for k in ordered]
    flagged_idx = is_divergent(values, z=divergence_z)
    return [ordered[i] for i in flagged_idx]


def _classify_failure(
    job: dict,
    bundle_dir: Path,
    bundle_jsons: tuple[dict, dict] | None,
) -> str:
    """Return the ``failure_kind`` string for a non-completed seed."""
    job_status = job.get("status")
    job_failure = job.get("failure") or {}
    job_failure_kind = job_failure.get("kind")

    if job_status == "failed" and job_failure_kind:
        return str(job_failure_kind)
    if not bundle_dir.exists():
        return "missing_bundle"
    if bundle_jsons is None:
        return "malformed_bundle"
    manifest, training = bundle_jsons
    if not _bundle_is_completed(manifest, training):
        return "malformed_bundle"
    # Shouldn't reach here — this job should have been classified completed.
    return "malformed_bundle"


def _classify_failure_message(job: dict) -> str:
    failure = job.get("failure") or {}
    return str(failure.get("message", ""))


def aggregate_dispatch(
    dispatch_dir: Path,
    options: AggregateOptions | None = None,
) -> dict:
    """Read ``dispatch_dir/dispatch.json`` + every ``<run_id>/`` bundle, return aggregate dict.

    Args:
        dispatch_dir: Path to ``odin_runs/<dispatch_id>/``.
        options: Optional :class:`AggregateOptions`; defaults to
            ``AggregateOptions()``.

    Returns:
        Aggregate dict matching T4.1 schema v1.0.

    Raises:
        FileNotFoundError: When ``dispatch_dir/dispatch.json`` is absent.
    """
    options = options or AggregateOptions()
    dispatch_path = dispatch_dir / "dispatch.json"
    if not dispatch_path.exists():
        raise FileNotFoundError(f"{dispatch_path} does not exist")
    with dispatch_path.open("r") as fh:
        dispatch = json.load(fh)

    rows_by_key: dict[tuple[str, str, str], dict] = {}
    failures: list[dict] = []
    commit_shas: list[str] = []
    hostnames: set[str] = set()

    for job in dispatch.get("jobs", []):
        run_id = job["run_id"]
        task = job["task_id"]
        framework = job["framework"]
        backend = job["backend"]
        seed = int(job["seed"])
        assigned_to = job.get("assigned_to")
        key = (task, framework, backend)
        if key not in rows_by_key:
            rows_by_key[key] = {
                "task":      task,
                "framework": framework,
                "backend":   backend,
                "seeds":     {},
            }

        bundle_dir = dispatch_dir / run_id
        bundle_jsons = _load_bundle_jsons(bundle_dir)
        is_completed = (
            bundle_dir.exists()
            and bundle_jsons is not None
            and _bundle_is_completed(*bundle_jsons)
        )
        if is_completed:
            manifest, training = bundle_jsons  # type: ignore[misc]
            rows_by_key[key]["seeds"][str(seed)] = _extract_seed_payload(
                training, manifest, run_id, assigned_to
            )
            commit_shas.append(str(manifest.get("machine", {}).get("git_commit") or ""))
            hostname = manifest.get("machine", {}).get("hostname")
            if hostname:
                hostnames.add(str(hostname))
        else:
            failures.append(
                {
                    "run_id":          run_id,
                    "task":            task,
                    "framework":       framework,
                    "backend":         backend,
                    "seed":            seed,
                    "assigned_to":     assigned_to,
                    "failure_kind":    _classify_failure(job, bundle_dir, bundle_jsons),
                    "failure_message": _classify_failure_message(job),
                }
            )

    rows: list[dict] = []
    for key in sorted(rows_by_key):
        r = rows_by_key[key]
        n_failed_for_row = sum(
            1
            for f in failures
            if (f["task"], f["framework"], f["backend"]) == key
        )
        agg = _compute_aggregate(r["seeds"], options.divergence_z)
        agg["n_seeds_failed"] = n_failed_for_row
        r["aggregate"] = agg if r["seeds"] else None
        r["divergent_seeds"] = _divergent_seed_keys(r["seeds"], options.divergence_z)
        rows.append(r)

    commit_sha = ""
    if commit_shas:
        counter = Counter(s for s in commit_shas if s)
        if counter:
            commit_sha, _ = counter.most_common(1)[0]
            minority = {s for s in counter if s != commit_sha}
            if minority:
                print(
                    f"[WARNING] bundles carried mixed commit SHAs; majority={commit_sha!r}, "
                    f"minority={sorted(minority)!r}"
                )

    totals = {
        "tasks":     len(rows),
        "runs":      sum(len(r["seeds"]) for r in rows) + len(failures),
        "completed": sum(len(r["seeds"]) for r in rows),
        "failed":    len(failures),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "dispatch_id":    str(dispatch.get("dispatch_id", dispatch_dir.name)),
        "generated_at":   _utc_now_iso(),
        "commit_sha":     commit_sha,
        "hostnames":      sorted(hostnames),
        "totals":         totals,
        "rows":           rows,
        "failures":       failures,
    }
