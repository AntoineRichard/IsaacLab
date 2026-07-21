# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate tuning evidence and write deterministic adaptive decisions."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .manifests import command_hash, sha256_file, write_json_atomic
from .matrix import DEFAULT_MATRIX_PATH, load_matrix
from .models import TerminalState
from .parsing import parse_training_trace
from .statistics import mean_ci95
from .tune import ResolvedTuningCandidate, TuningIdentity, _command_for_candidate, read_tuning_manifest
from .tuning import (
    DEFAULT_TUNING_MATRIX_PATH,
    FinalQualification,
    TuningCandidate,
    TuningMatrix,
    TuningRunMetrics,
    config_hash,
    load_tuning_matrix,
    promote_finalists,
    promote_stage2,
    qualify_finalists,
    resolve_config,
    resolve_wave2,
    select_winner,
)


@dataclass(frozen=True)
class TuningRecord:
    """A terminal tuning metric record with immutable evidence references."""

    metrics: TuningRunMetrics
    run_id: str
    manifest_path: Path
    manifest_hash: str
    event_path: Path | None
    event_hash: str | None
    config_hash: str
    resolved_config: dict[str, str | int | float | bool]
    completed_at_utc: str | None = None

    source_head: str | None = None


def _read_mapping(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON root must be a mapping")
    return data


def _repo_root(command: Sequence[str], manifest_path: Path) -> Path:
    scripts = [Path(item) for item in command if isinstance(item, str) and Path(item).name == "isaaclab.sh"]
    if len(scripts) != 1:
        raise ValueError(f"{manifest_path}: command does not identify exactly one IsaacLab root")
    return scripts[0].parent.resolve()


def _bundle_field(data: Mapping[str, Any], *path: str) -> Any:
    value: Any = data
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"bundle is missing {'.'.join(path)}")
        value = value[component]
    return value


def _candidate_for_manifest(
    matrix: TuningMatrix, identity: TuningIdentity, resolved: dict[str, Any]
) -> ResolvedTuningCandidate:
    if set(resolved) != set(matrix.baseline):
        raise ValueError(f"candidate {identity.candidate} does not contain the complete resolved configuration")
    overrides = {name: value for name, value in resolved.items() if value != matrix.baseline[name]}
    if identity.stage == "wave1":
        declared = matrix.candidate(identity.candidate)
        if overrides != declared.overrides:
            raise ValueError(f"candidate {identity.candidate} resolved config does not match its Wave 1 declaration")
    if identity.stage == "baseline" and (identity.candidate != "baseline" or overrides):
        raise ValueError("baseline manifest does not contain the clean baseline configuration")
    if identity.stage == "canonical":
        overrides = {}
    return ResolvedTuningCandidate(identity.candidate, overrides, resolved, config_hash(resolved))


def _validate_series(metrics: TuningRunMetrics, iterations: int, manifest_path: Path) -> None:
    for name in ("iteration_time_s", "reward", "success_rate", "ep_length"):
        values = getattr(metrics, name)
        if len(values) != iterations:
            raise ValueError(f"{manifest_path}: {name} requires exactly {iterations} aligned values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{manifest_path}: {name} contains nonfinite values")


def _failed_record(manifest_path: Path, manifest, resolved: dict[str, Any]) -> TuningRecord:
    identity = manifest.identity
    reason = manifest.failure_category.value if manifest.failure_category is not None else "failed without category"
    metrics = TuningRunMetrics(
        identity.candidate, identity.stage, identity.seed, identity.num_envs, (), (), (), (), reason
    )
    return TuningRecord(
        metrics=metrics,
        run_id=manifest.run_id,
        manifest_path=manifest_path,
        manifest_hash=sha256_file(manifest_path),
        event_path=Path(manifest.tensorboard_event_path) if manifest.tensorboard_event_path else None,
        event_hash=manifest.tensorboard_event_hash,
        config_hash=manifest.config_hash,
        resolved_config=resolved,
        source_head=manifest.isaaclab_head,
    )


def load_tuning_records(  # noqa: C901
    artifact_root: Path,
    logs_root: Path,
    expected_stage: str | None = None,
    *,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    tuning_matrix_path: Path = DEFAULT_TUNING_MATRIX_PATH,
) -> list[TuningRecord]:
    """Load terminal tuning attempts only after exact evidence validation.

    Args:
        artifact_root: Root containing immutable tuning attempt directories.
        logs_root: Root containing retained TensorBoard event files.
        expected_stage: Optional measured stage to select.
        matrix_path: Locked benchmark matrix path.
        tuning_matrix_path: Locked tuning matrix path.

    Returns:
        Validated completed records and explicit failed rejection records.

    Raises:
        ValueError: If any selected artifact violates identity or provenance.
    """
    benchmark_matrix = load_matrix(matrix_path)
    tuning_matrix = load_tuning_matrix(tuning_matrix_path)
    records: list[TuningRecord] = []
    resolved_artifact_root = artifact_root.resolve()
    seen: set[tuple[str, str, int, int]] = set()
    stage_protocol = {
        "baseline": (matrix_seeds := set(tuning_matrix.seeds), tuning_matrix.final_iterations),
        "wave1": ({42}, tuning_matrix.screen_iterations),
        "wave2": ({42}, tuning_matrix.screen_iterations),
        "halve": ({42, 43}, tuning_matrix.halve_iterations),
        "final": (matrix_seeds, tuning_matrix.final_iterations),
        "canonical": (matrix_seeds, tuning_matrix.final_iterations),
    }
    directories = sorted(path for path in artifact_root.iterdir() if path.is_dir()) if artifact_root.exists() else ()
    for run_dir in directories:
        try:
            run_dir.resolve().relative_to(resolved_artifact_root)
        except ValueError as error:
            raise ValueError(f"tuning run directory escapes artifact root: {run_dir}") from error
        components = run_dir.name.split("__")
        if len(components) != 6 or components[0] not in {*stage_protocol, "preflight"}:
            raise ValueError(f"undeclared tuning directory: {run_dir}")
        if components[0] == "preflight":
            continue
        if expected_stage is not None and components[0] != expected_stage:
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"undeclared tuning directory has no manifest: {run_dir}")
        try:
            manifest = read_tuning_manifest(manifest_path)
        except Exception as error:
            raise ValueError(f"{manifest_path}: invalid typed tuning manifest: {error}") from error
        identity = manifest.identity
        key = (identity.stage, identity.candidate, identity.seed, identity.attempt)
        if key in seen:
            raise ValueError(f"duplicate tuning record: {key}")
        seen.add(key)
        if expected_stage is not None and identity.stage != expected_stage:
            raise ValueError(f"{manifest_path}: expected stage {expected_stage}")
        if identity.stage not in stage_protocol:
            raise ValueError(f"{manifest_path}: unsupported measured stage {identity.stage}")
        allowed_seeds, required_iterations = stage_protocol[identity.stage]
        if identity.seed not in allowed_seeds:
            raise ValueError(f"{identity.stage} uses an invalid seed {identity.seed}")
        if identity.max_iterations != required_iterations:
            raise ValueError(f"{identity.stage} requires exactly {required_iterations} iterations")
        if manifest.run_id != identity.run_id or run_dir.name != identity.run_id:
            raise ValueError(f"{manifest_path}: run identity does not match its directory")
        if Path(manifest.artifact_root).resolve() != run_dir.resolve():
            raise ValueError(f"{manifest_path}: artifact root does not match its directory")
        if identity.num_envs != tuning_matrix.num_envs:
            raise ValueError(f"{manifest_path}: reduced environment count is forbidden")
        if manifest.revisions != benchmark_matrix.revisions or manifest.schema_version != "1.1":
            raise ValueError(f"{manifest_path}: locked revisions or schema mismatch")
        resolved = dict(manifest.resolved_config)
        if manifest.config_hash != config_hash(resolved):
            raise ValueError(f"{manifest_path}: config hash mismatch")
        candidate = _candidate_for_manifest(tuning_matrix, identity, resolved)
        if manifest.command_hash != command_hash(manifest.command):
            raise ValueError(f"{manifest_path}: command hash mismatch")
        repo_root = _repo_root(manifest.command, manifest_path)
        expected_command = tuple(
            _command_for_candidate(benchmark_matrix, tuning_matrix, candidate, identity, repo_root, run_dir)
        )
        if manifest.command != expected_command:
            raise ValueError(f"{manifest_path}: command does not match exact reconstruction")
        for name, expected_hash in manifest.artifact_hashes.items():
            artifact = run_dir / name
            try:
                artifact.resolve().relative_to(run_dir.resolve())
            except ValueError as error:
                raise ValueError(f"{manifest_path}: artifact path escapes its run directory") from error
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"{manifest_path}: artifact hash mismatch for {name}")
        if manifest.state is TerminalState.FAILED:
            records.append(_failed_record(manifest_path, manifest, resolved))
            continue
        if manifest.state is not TerminalState.COMPLETED:
            raise ValueError(f"{manifest_path}: manifest is incomplete ({manifest.state.value})")
        bundles = tuple(sorted(run_dir.glob("benchmark_training_*.json")))
        if len(bundles) != 1 or bundles[0].name not in manifest.artifact_hashes:
            raise ValueError(f"{manifest_path}: requires exactly one hashed schema bundle")
        bundle_path = bundles[0]
        bundle = _read_mapping(bundle_path)
        if _bundle_field(bundle, "versions", "git_dirty") is not False:
            raise ValueError(f"{bundle_path}: source tree was dirty")
        if _bundle_field(bundle, "versions", "git_commit") != manifest.isaaclab_head:
            raise ValueError(f"{manifest_path}: source HEAD does not match bundle provenance")
        actual_identity = (
            _bundle_field(bundle, "run", "task"),
            _bundle_field(bundle, "run", "seed"),
            _bundle_field(bundle, "run", "num_envs"),
            _bundle_field(bundle, "run", "max_iterations"),
            _bundle_field(bundle, "runtime", "iterations_completed"),
        )
        expected_identity = (
            tuning_matrix.task,
            identity.seed,
            identity.num_envs,
            identity.max_iterations,
            identity.max_iterations,
        )
        if actual_identity != expected_identity or _bundle_field(bundle, "run", "status") != "completed":
            raise ValueError(f"{bundle_path}: bundle identity or completion mismatch")
        if not isinstance(manifest.tensorboard_event_path, str) or not isinstance(manifest.tensorboard_event_hash, str):
            raise ValueError(f"{manifest_path}: missing event hash or path")
        event_path = Path(manifest.tensorboard_event_path)
        try:
            event_path.resolve().relative_to(logs_root.resolve())
        except ValueError as error:
            raise ValueError(f"{manifest_path}: event path is outside the declared logs root") from error
        if not event_path.is_file() or sha256_file(event_path) != manifest.tensorboard_event_hash:
            raise ValueError(f"{manifest_path}: event hash mismatch")
        trace = parse_training_trace(bundle_path, event_path)
        trace_identity = (trace.task, trace.seed, trace.num_envs, trace.iterations)
        if trace_identity != expected_identity[:4]:
            raise ValueError(f"{bundle_path}: parsed trace identity mismatch")
        metrics = TuningRunMetrics(
            identity.candidate,
            identity.stage,
            identity.seed,
            identity.num_envs,
            trace.iteration_time_s,
            trace.reward,
            trace.success_rate,
            trace.ep_length,
        )
        _validate_series(metrics, identity.max_iterations, manifest_path)
        completed_at = str(_bundle_field(bundle, "run", "end_time_utc"))
        source_head = manifest.isaaclab_head
        records.append(
            TuningRecord(
                metrics,
                manifest.run_id,
                manifest_path,
                sha256_file(manifest_path),
                event_path,
                manifest.tensorboard_event_hash,
                manifest.config_hash,
                resolved,
                completed_at,
                source_head,
            )
        )
    return records


def validate_tuning_records(records: Sequence[TuningRecord], expected: Iterable[tuple[str, int]]) -> None:
    """Require exactly one terminal record for every candidate and seed."""
    expected_set = set(expected)
    identities = [(record.metrics.candidate, record.metrics.seed) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate tuning record")
    unexpected = set(identities) - expected_set
    if unexpected:
        raise ValueError(f"unexpected tuning record: {sorted(unexpected)[0]}")
    missing = expected_set - set(identities)
    if missing:
        raise ValueError(f"missing tuning record: {sorted(missing)[0]}")


def derive_stage2_baseline(
    records: Sequence[TuningRecord], iterations: int
) -> tuple[tuple[TuningRunMetrics, ...], list[dict[str, Any]]]:
    """Derive a first-iteration prefix view for Stage 2 guardrails."""
    derived: list[TuningRunMetrics] = []
    provenance: list[dict[str, Any]] = []
    for record in records:
        source = record.metrics
        if source.stage != "baseline" or source.failure is not None or len(source.reward) < iterations:
            raise ValueError("Stage 2 baseline derivation requires complete clean baseline records")
        derived.append(
            TuningRunMetrics(
                source.candidate,
                "halve",
                source.seed,
                source.num_envs,
                source.iteration_time_s[:iterations],
                source.reward[:iterations],
                source.success_rate[:iterations],
                source.ep_length[:iterations],
            )
        )
        provenance.append(
            {
                "seed": source.seed,
                "source_run_id": record.run_id,
                "source_manifest_path": str(record.manifest_path),
                "source_manifest_hash": record.manifest_hash,
                "source_event_path": str(record.event_path) if record.event_path else None,
                "source_event_hash": record.event_hash,
                "source_config_hash": record.config_hash,
                "derivation": f"first {iterations} aligned iterations of clean 300-iteration baseline",
            }
        )
    return tuple(derived), provenance


def _resolved_records(matrix: TuningMatrix, candidates: Sequence[TuningCandidate]) -> list[dict[str, Any]]:
    output = []
    for candidate in candidates:
        resolved = resolve_config(matrix, candidate)
        output.append(
            {
                "name": candidate.name,
                "overrides": dict(candidate.overrides),
                "resolved_config": resolved,
                "config_hash": config_hash(resolved),
            }
        )
    return output


def _timestamp(records: Sequence[TuningRecord]) -> str:
    values = sorted(record.completed_at_utc for record in records if record.completed_at_utc)
    return values[-1] if values else "1970-01-01T00:00:00+00:00"


def _base_decision(source_stage: str, artifact_root: Path, records: Sequence[TuningRecord]) -> dict[str, Any]:
    matrix = load_matrix(DEFAULT_MATRIX_PATH)
    return {
        "source_stage": source_stage,
        "source_artifact_root": str(artifact_root.resolve()),
        "source_manifests": [
            {"run_id": record.run_id, "path": str(record.manifest_path), "sha256": record.manifest_hash}
            for record in sorted(records, key=lambda item: item.run_id)
        ],
        "timestamp_utc": _timestamp(records),
        "schema_version": "1.0",
        "action": {
            "wave1": "resolve-wave2",
            "stage1": "promote-stage2",
            "stage2": "promote-finalists",
            "stage3": "select-winner",
        }[source_stage],
        "revisions": dataclasses.asdict(matrix.revisions),
    }


def load_decision(
    path: Path,
    expected_action: str,
    matrix: TuningMatrix,
    *,
    minimum_count: int,
    maximum_count: int,
) -> dict[str, Any]:
    """Strictly parse and recompute every persisted decision candidate."""
    data = _read_mapping(path)
    if data.get("schema_version") != "1.0" or data.get("action") != expected_action:
        raise ValueError(f"{path}: decision schema/action mismatch")
    if data.get("revisions") != dataclasses.asdict(load_matrix(DEFAULT_MATRIX_PATH).revisions):
        raise ValueError(f"{path}: decision revisions mismatch")
    if not isinstance(data.get("source_artifact_root"), str):
        raise ValueError(f"{path}: decision artifact root is invalid")
    try:
        timestamp = datetime.fromisoformat(str(data["timestamp_utc"]).replace("Z", "+00:00"))
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: decision timestamp is invalid") from error
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise ValueError(f"{path}: decision timestamp must be UTC")
    manifests = data.get("source_manifests")
    if not isinstance(manifests, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("run_id"), str)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("sha256"), str)
        for item in manifests
    ):
        raise ValueError(f"{path}: decision source manifests are invalid")
    records = data.get("resolved_candidates")
    if not isinstance(records, list) or not minimum_count <= len(records) <= maximum_count:
        raise ValueError(f"{path}: decision candidate count must be between {minimum_count} and {maximum_count}")
    names: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError(f"{path}: decision candidate name is invalid")
        if not isinstance(record.get("overrides"), dict) or not isinstance(record.get("resolved_config"), dict):
            raise ValueError(f"{path}: decision candidate provenance is invalid")
        candidate = TuningCandidate(record["name"], dict(record["overrides"]))
        resolved = resolve_config(matrix, candidate)
        if record["resolved_config"] != resolved or record.get("config_hash") != config_hash(resolved):
            raise ValueError(f"{path}: decision candidate provenance mismatch for {candidate.name}")
        names.append(candidate.name)
    if len(names) != len(set(names)) or data.get("selected") != names:
        raise ValueError(f"{path}: decision selected candidates mismatch")
    if not isinstance(data.get("rejected"), dict):
        raise ValueError(f"{path}: decision rejection mapping is invalid")
    return data


def reconcile_decision_candidates(decision: Mapping[str, Any], records: Sequence[TuningRecord]) -> None:
    """Require every downstream manifest to match its decision candidate config."""
    candidates = {item["name"]: item for item in decision["resolved_candidates"]}
    for record in records:
        expected = candidates.get(record.metrics.candidate)
        if expected is None:
            raise ValueError(f"unexpected downstream manifest candidate {record.metrics.candidate}")
        if record.resolved_config != expected["resolved_config"] or record.config_hash != expected["config_hash"]:
            raise ValueError(f"downstream manifest config mismatch for {record.metrics.candidate}")


def _metric_estimates(matrix: TuningMatrix, records: Sequence[TuningRunMetrics]) -> dict[str, Any]:
    return {
        "runtime": mean_ci95([record.steady_time(matrix.warmup_iterations) for record in records]),
        "reward": mean_ci95([record.final_mean(record.reward, matrix.learning_window) for record in records]),
        "success": mean_ci95([record.final_mean(record.success_rate, matrix.learning_window) for record in records]),
        "episode_length": mean_ci95(
            [record.final_mean(record.ep_length, matrix.learning_window) for record in records]
        ),
    }


def _intervals_overlap(left, right) -> bool:
    return (
        left.mean - left.half_width <= right.mean + right.half_width
        and right.mean - right.half_width <= left.mean + left.half_width
    )


def _canonical_comparison(
    matrix: TuningMatrix,
    final_records: Sequence[TuningRunMetrics],
    canonical_records: Sequence[TuningRunMetrics],
) -> dict[str, Any]:
    """Require canonical runtime, reward, and success confidence intervals to overlap."""
    final = _metric_estimates(matrix, final_records)
    canonical = _metric_estimates(matrix, canonical_records)
    for metric in ("runtime", "reward", "success"):
        if not _intervals_overlap(final[metric], canonical[metric]):
            raise ValueError(f"canonical {metric} interval does not overlap override-based winner")
    return {
        "qualified": True,
        "override_final": {name: dataclasses.asdict(value) for name, value in final.items()},
        "canonical": {name: dataclasses.asdict(value) for name, value in canonical.items()},
    }


def write_decision(path: Path, decision: Mapping[str, Any]) -> None:
    """Atomically write a canonical decision mapping."""
    write_json_atomic(path, dict(decision))


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    records = _read_mapping(path).get("resolved_candidates")
    if not isinstance(records, list):
        raise ValueError(f"{path}: missing resolved_candidates")
    return [dict(record) for record in records]


def _records(args: argparse.Namespace, stage: str) -> list[TuningRecord]:
    return load_tuning_records(args.artifact_root, args.logs_root, expected_stage=stage)


def _compute_wave2(matrix: TuningMatrix, artifact_root: Path, wave1: Sequence[TuningRecord]) -> dict[str, Any]:
    validate_tuning_records(wave1, ((candidate.name, 42) for candidate in matrix.wave1))
    resolved = resolve_wave2(matrix, [record.metrics for record in wave1])
    decision = _base_decision("wave1", artifact_root, wave1)
    decision.update(
        selected=[candidate.name for candidate in resolved],
        rejected={record.metrics.candidate: record.metrics.failure for record in wave1 if record.metrics.failure},
        resolved_candidates=_resolved_records(matrix, resolved),
    )
    return decision


def _compute_stage2(
    matrix: TuningMatrix,
    artifact_root: Path,
    wave1: Sequence[TuningRecord],
    wave2: Sequence[TuningRecord],
    wave2_decision: Mapping[str, Any],
) -> dict[str, Any]:
    validate_tuning_records(wave2, ((item["name"], 42) for item in wave2_decision["resolved_candidates"]))
    reconcile_decision_candidates(wave2_decision, wave2)
    records = [*wave1, *wave2]
    value = promote_stage2(matrix, [record.metrics for record in records])
    if not value.selected:
        raise ValueError("Stage 2 requires at least one valid survivor")
    by_name = {record.metrics.candidate: record for record in records}
    candidates = [
        TuningCandidate(
            name,
            {key: item for key, item in by_name[name].resolved_config.items() if item != matrix.baseline[key]},
        )
        for name in value.selected
    ]
    decision = _base_decision("stage1", artifact_root, records)
    decision.update(
        selected=list(value.selected),
        rejected=value.rejected,
        resolved_candidates=_resolved_records(matrix, candidates),
    )
    return decision


def _compute_finalists(
    matrix: TuningMatrix,
    artifact_root: Path,
    baseline: Sequence[TuningRecord],
    halve: Sequence[TuningRecord],
    stage2_decision: Mapping[str, Any],
) -> dict[str, Any]:
    validate_tuning_records(baseline, (("baseline", seed) for seed in matrix.seeds))
    validate_tuning_records(
        halve,
        ((item["name"], seed) for item in stage2_decision["resolved_candidates"] for seed in (42, 43)),
    )
    reconcile_decision_candidates(stage2_decision, halve)
    derived, provenance = derive_stage2_baseline(baseline, matrix.halve_iterations)
    value = promote_finalists(
        matrix,
        [record for record in derived if record.seed in (42, 43)],
        [record.metrics for record in halve],
    )
    if not value.selected:
        raise ValueError("final stage requires at least one valid survivor")
    selected = set(value.selected)
    candidates = [
        TuningCandidate(item["name"], dict(item["overrides"]))
        for item in stage2_decision["resolved_candidates"]
        if item["name"] in selected
    ]
    decision = _base_decision("stage2", artifact_root, [*baseline, *halve])
    decision.update(
        selected=list(value.selected),
        rejected=value.rejected,
        resolved_candidates=_resolved_records(matrix, candidates),
        baseline_prefix_provenance=provenance,
        methodology=(
            "Stage 2 uses the first 100 aligned samples of each clean 300-iteration "
            "baseline; its final-20 window is iterations 81-100."
        ),
    )
    return decision


def _qualification_dict(values: Mapping[str, FinalQualification]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in sorted(values.items()):
        record = dataclasses.asdict(value)
        for metric in ("runtime", "reward", "success_rate", "ep_length"):
            estimate = record[metric]
            if not math.isfinite(estimate["mean"]) or not math.isfinite(estimate["half_width"]):
                record[metric] = None
        output[name] = record
    return output


def _compute_winner(
    matrix: TuningMatrix,
    artifact_root: Path,
    baseline: Sequence[TuningRecord],
    final: Sequence[TuningRecord],
    finalists_decision: Mapping[str, Any],
) -> dict[str, Any]:
    validate_tuning_records(
        final,
        ((item["name"], seed) for item in finalists_decision["resolved_candidates"] for seed in matrix.seeds),
    )
    reconcile_decision_candidates(finalists_decision, final)
    qualifications = qualify_finalists(
        matrix, [record.metrics for record in baseline], [record.metrics for record in final]
    )
    configs = {item["name"]: item["resolved_config"] for item in finalists_decision["resolved_candidates"]}
    winner = select_winner(matrix, qualifications, configs)
    selected = next(item for item in finalists_decision["resolved_candidates"] if item["name"] == winner)
    rejected = {
        name: value.reason or "qualified but not selected by deterministic tie-break"
        for name, value in qualifications.items()
        if name != winner
    }
    decision = _base_decision("stage3", artifact_root, [*baseline, *final])
    decision.update(
        candidate=winner,
        selected=[winner],
        rejected=rejected,
        qualifications=_qualification_dict(qualifications),
        overrides=selected["overrides"],
        resolved_config=selected["resolved_config"],
        config_hash=config_hash(selected["resolved_config"]),
        resolved_candidates=[selected],
    )
    return decision


def _require_persisted(
    path: Path, action: str, matrix: TuningMatrix, expected: Mapping[str, Any], minimum: int, maximum: int
) -> dict[str, Any]:
    persisted = load_decision(path, action, matrix, minimum_count=minimum, maximum_count=maximum)
    if persisted != expected:
        raise ValueError(f"{path}: persisted decision does not match recomputed raw evidence")
    return persisted


def _recompute_chain(args: argparse.Namespace, through: str) -> dict[str, Any]:
    matrix = load_tuning_matrix(args.tuning_matrix)
    wave1 = _records(args, "wave1")
    wave2_expected = _compute_wave2(matrix, args.artifact_root, wave1)
    if through == "wave2":
        return {"matrix": matrix, "wave1": wave1, "wave2_decision": wave2_expected}
    wave2_decision = _require_persisted(
        args.decision_root / "wave2.json", "resolve-wave2", matrix, wave2_expected, 6, 6
    )
    wave2 = _records(args, "wave2")
    stage2_expected = _compute_stage2(matrix, args.artifact_root, wave1, wave2, wave2_decision)
    if through == "stage2":
        return {
            "matrix": matrix,
            "wave1": wave1,
            "wave2": wave2,
            "wave2_decision": wave2_decision,
            "stage2_decision": stage2_expected,
        }
    stage2_decision = _require_persisted(
        args.decision_root / "stage2.json", "promote-stage2", matrix, stage2_expected, 1, 8
    )
    baseline = _records(args, "baseline")
    halve = _records(args, "halve")
    finalists_expected = _compute_finalists(matrix, args.artifact_root, baseline, halve, stage2_decision)
    if through == "finalists":
        return {
            "matrix": matrix,
            "wave1": wave1,
            "wave2": wave2,
            "baseline": baseline,
            "halve": halve,
            "wave2_decision": wave2_decision,
            "stage2_decision": stage2_decision,
            "finalists_decision": finalists_expected,
        }
    finalists_decision = _require_persisted(
        args.decision_root / "finalists.json", "promote-finalists", matrix, finalists_expected, 1, 3
    )
    final = _records(args, "final")
    winner_expected = _compute_winner(matrix, args.artifact_root, baseline, final, finalists_decision)
    return {
        "matrix": matrix,
        "wave1": wave1,
        "wave2": wave2,
        "baseline": baseline,
        "halve": halve,
        "final": final,
        "wave2_decision": wave2_decision,
        "stage2_decision": stage2_decision,
        "finalists_decision": finalists_decision,
        "winner_decision": winner_expected,
    }


def _resolve_wave2_action(args: argparse.Namespace) -> None:
    chain = _recompute_chain(args, "wave2")
    write_decision(args.output, chain["wave2_decision"])


def _promote_stage2_action(args: argparse.Namespace) -> None:
    chain = _recompute_chain(args, "stage2")
    write_decision(args.output, chain["stage2_decision"])


def _promote_finalists_action(args: argparse.Namespace) -> None:
    chain = _recompute_chain(args, "finalists")
    write_decision(args.output, chain["finalists_decision"])


def _select_winner_action(args: argparse.Namespace) -> None:
    chain = _recompute_chain(args, "winner")
    write_decision(args.output, chain["winner_decision"])


def _report_action(args: argparse.Namespace) -> None:
    from .tuning_reporting import write_tuning_report

    chain = _recompute_chain(args, "winner")
    matrix: TuningMatrix = chain["matrix"]
    winner_expected = chain["winner_decision"]
    winner = _require_persisted(args.decision_root / "winner.json", "select-winner", matrix, winner_expected, 1, 1)
    canonical = _records(args, "canonical")
    validate_tuning_records(canonical, (("canonical_winner", seed) for seed in matrix.seeds))
    for record in canonical:
        if record.resolved_config != winner["resolved_config"] or record.config_hash != winner["config_hash"]:
            raise ValueError("canonical manifest config does not match recomputed winner")
    canonical_heads = {record.source_head for record in canonical}
    if None in canonical_heads or len(canonical_heads) != 1:
        raise ValueError("canonical runs require one exact clean committed source HEAD")
    final_winner = [record.metrics for record in chain["final"] if record.metrics.candidate == winner["candidate"]]
    canonical_metrics = [record.metrics for record in canonical]
    canonical_comparison = _canonical_comparison(matrix, final_winner, canonical_metrics)
    qualifications = winner["qualifications"]
    winner_runtime = float(qualifications[winner["candidate"]]["runtime"]["mean"])
    clean_runtime = mean_ci95([record.metrics.steady_time(matrix.warmup_iterations) for record in chain["baseline"]])
    comparisons = json.loads(args.comparison_summary.read_text(encoding="utf-8"))
    legacy_rows = [
        row for row in comparisons if row.get("task") == matrix.task and row.get("variant") in {"mjwarp", "physx"}
    ]
    legacy_runtime = {row["variant"]: float(row["iteration_time_s"]["mean"]) for row in legacy_rows}
    all_records = [
        *chain["baseline"],
        *chain["wave1"],
        *chain["wave2"],
        *chain["halve"],
        *chain["final"],
        *canonical,
    ]
    decisions = {
        "wave2": chain["wave2_decision"],
        "stage2": chain["stage2_decision"],
        "finalists": chain["finalists_decision"],
        "winner": winner,
    }
    rejection_map: dict[str, str] = {}
    for decision in decisions.values():
        rejection_map.update(decision.get("rejected", {}))
    for record in all_records:
        if record.metrics.failure:
            rejection_map[record.run_id] = record.metrics.failure
    derived_baseline, _ = derive_stage2_baseline(chain["baseline"], matrix.halve_iterations)
    canonical_estimates = canonical_comparison["canonical"]
    report = {
        "schema_version": "1.0",
        "winner": winner["candidate"],
        "winner_config": winner["resolved_config"],
        "winner_config_hash": winner["config_hash"],
        "environment_count": matrix.num_envs,
        "decisions": decisions,
        "canonical_comparison": canonical_comparison,
        "canonical_provenance": {
            "source_head": next(iter(canonical_heads)),
            "records": [
                {
                    "run_id": record.run_id,
                    "manifest_path": str(record.manifest_path),
                    "manifest_hash": record.manifest_hash,
                    "event_path": str(record.event_path),
                    "event_hash": record.event_hash,
                    "config_hash": record.config_hash,
                }
                for record in canonical
            ],
        },
        "funnel": [
            {"stage": "baseline", "attempted": len(chain["baseline"]), "valid": 3, "rejected": 0, "promoted": 1},
            {
                "stage": "Wave 1",
                "attempted": len(chain["wave1"]),
                "valid": sum(item.metrics.failure is None for item in chain["wave1"]),
                "rejected": len(chain["wave2_decision"]["rejected"]),
                "promoted": len(chain["wave2_decision"]["selected"]),
            },
            {
                "stage": "Wave 2",
                "attempted": len(chain["wave2"]),
                "valid": sum(item.metrics.failure is None for item in chain["wave2"]),
                "rejected": 0,
                "promoted": len(chain["stage2_decision"]["selected"]),
            },
            {
                "stage": "halve",
                "attempted": len(chain["halve"]),
                "valid": sum(item.metrics.failure is None for item in chain["halve"]),
                "rejected": len(chain["finalists_decision"]["rejected"]),
                "promoted": len(chain["finalists_decision"]["selected"]),
            },
            {
                "stage": "final",
                "attempted": len(chain["final"]),
                "valid": sum(item.metrics.failure is None for item in chain["final"]),
                "rejected": len(winner["rejected"]),
                "promoted": 1,
            },
            {"stage": "canonical", "attempted": len(canonical), "valid": len(canonical), "rejected": 0, "promoted": 1},
        ],
        "runtime_rows": [
            {
                "stage": record.metrics.stage,
                "candidate": record.metrics.candidate,
                "mean": record.metrics.steady_time(matrix.warmup_iterations),
                "half_width": 0.0,
                "n": 1,
            }
            for record in [*chain["wave1"], *chain["wave2"]]
            if record.metrics.failure is None
        ],
        "final_rows": [
            {
                "candidate": name,
                "runtime": value["runtime"],
                "reward": value["reward"],
                "success": value["success_rate"],
                "episode_length": value["ep_length"],
            }
            for name, value in sorted(qualifications.items())
        ]
        + [
            {
                "candidate": "canonical_winner",
                "runtime": canonical_estimates["runtime"],
                "reward": canonical_estimates["reward"],
                "success": canonical_estimates["success"],
                "episode_length": canonical_estimates["episode_length"],
            }
        ],
        "learning_traces": [
            {
                "candidate": f"derived baseline seed {record.seed}",
                "reward": record.reward,
                "success": record.success_rate,
                "episode_length": record.ep_length,
            }
            for record in derived_baseline
            if record.seed in (42, 43)
        ]
        + [
            {
                "candidate": f"{record.metrics.candidate} seed {record.metrics.seed}",
                "reward": record.metrics.reward,
                "success": record.metrics.success_rate,
                "episode_length": record.metrics.ep_length,
            }
            for record in chain["halve"]
            if record.metrics.failure is None
        ],
        "speedups": {
            "clean DVI": clean_runtime.mean / winner_runtime,
            "legacy MJWarp": legacy_runtime["mjwarp"] / winner_runtime,
            "legacy PhysX": legacy_runtime["physx"] / winner_runtime,
        },
        "rejections": [f"{name}: {reason}" for name, reason in sorted(rejection_map.items())],
        "coverage": {
            stage: [
                {
                    "candidate": record.metrics.candidate,
                    "seed": record.metrics.seed,
                    "iterations": len(record.metrics.reward),
                    "num_envs": record.metrics.num_envs,
                }
                for record in records
            ]
            for stage, records in (
                ("baseline", chain["baseline"]),
                ("wave1", chain["wave1"]),
                ("wave2", chain["wave2"]),
                ("halve", chain["halve"]),
                ("final", chain["final"]),
                ("canonical", canonical),
            )
        },
        "source_hashes": [
            {"run_id": record.run_id, "manifest_hash": record.manifest_hash, "event_hash": record.event_hash}
            for record in sorted(all_records, key=lambda item: item.run_id)
        ],
        "legacy_comparison": legacy_rows,
        "seed_iteration_coverage": (
            "4096 environments; baseline/final/canonical seeds 42--44 at 300 iterations; "
            "Wave 1/2 seed 42 at 40; halve seeds 42--43 at 100"
        ),
        "stage2_baseline_derivation": (
            "first 100 aligned iterations of clean 300-iteration baseline; final-20 is "
            "iterations 81--100; source manifest/event/config hashes are retained in finalists.json"
        ),
        "legacy_limitations": (
            "MJWarp/PhysX values come from the existing five-variant campaign whose manifests "
            "lack current exact source-HEAD and retained TensorBoard-event hashes; speedups are contextual only."
        ),
    }
    write_tuning_report(report, args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the staged tuning analysis parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action, default_name in (
        ("resolve-wave2", "wave2.json"),
        ("promote-stage2", "stage2.json"),
        ("promote-finalists", "finalists.json"),
        ("select-winner", "winner.json"),
    ):
        subparser = subparsers.add_parser(action)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--logs-root", type=Path, default=Path("logs"))
        subparser.add_argument("--decision-root", type=Path, default=Path("benchmark_artifacts/kamino_dvi/decisions"))
        subparser.add_argument("--tuning-matrix", type=Path, default=DEFAULT_TUNING_MATRIX_PATH)
        subparser.add_argument("--output", type=Path, default=Path(default_name))
    report = subparsers.add_parser("report")
    report.add_argument("--artifact-root", type=Path, required=True)
    report.add_argument("--logs-root", type=Path, default=Path("logs"))
    report.add_argument("--decision-root", type=Path, required=True)
    report.add_argument("--tuning-matrix", type=Path, default=DEFAULT_TUNING_MATRIX_PATH)
    report.add_argument(
        "--comparison-summary",
        type=Path,
        default=Path("benchmarks/kamino_dvi/results/summary.json"),
    )
    report.add_argument("--output-dir", type=Path, default=Path("benchmarks/kamino_dvi/results/anymal_d_tuning"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute one deterministic analysis or report action."""
    args = build_parser().parse_args(argv)
    actions = {
        "resolve-wave2": _resolve_wave2_action,
        "promote-stage2": _promote_stage2_action,
        "promote-finalists": _promote_finalists_action,
        "select-winner": _select_winner_action,
        "report": _report_action,
    }
    actions[args.action](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
