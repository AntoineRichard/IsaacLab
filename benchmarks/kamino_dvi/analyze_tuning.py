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
from pathlib import Path
from typing import Any

from .manifests import command_hash, sha256_file, write_json_atomic
from .matrix import DEFAULT_MATRIX_PATH, load_matrix
from .models import TerminalState
from .parsing import parse_training_trace
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
    )


def load_tuning_records(
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
    seen: set[tuple[str, str, int, int]] = set()
    for run_dir in sorted(path for path in artifact_root.iterdir() if path.is_dir()) if artifact_root.exists() else ():
        components = run_dir.name.split("__")
        if len(components) != 6 or components[0] not in {
            "baseline",
            "wave1",
            "wave2",
            "halve",
            "final",
            "canonical",
            "preflight",
        }:
            raise ValueError(f"undeclared tuning directory: {run_dir}")
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
        "revisions": dataclasses.asdict(matrix.revisions),
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


def _resolve_wave2_action(args: argparse.Namespace) -> None:
    matrix = load_tuning_matrix(args.tuning_matrix)
    records = _records(args, "wave1")
    validate_tuning_records(records, ((candidate.name, 42) for candidate in matrix.wave1))
    resolved = resolve_wave2(matrix, [record.metrics for record in records])
    selected = [candidate.name for candidate in resolved]
    decision = _base_decision("wave1", args.artifact_root, records)
    decision.update(
        selected=selected,
        rejected={record.metrics.candidate: record.metrics.failure for record in records if record.metrics.failure},
        resolved_candidates=_resolved_records(matrix, resolved),
    )
    write_decision(args.output, decision)


def _promote_stage2_action(args: argparse.Namespace) -> None:
    matrix = load_tuning_matrix(args.tuning_matrix)
    wave1 = _records(args, "wave1")
    wave2 = _records(args, "wave2")
    validate_tuning_records(wave1, ((candidate.name, 42) for candidate in matrix.wave1))
    wave2_candidates = _load_candidates(args.decision_root / "wave2.json")
    validate_tuning_records(wave2, ((item["name"], 42) for item in wave2_candidates))
    records = wave1 + wave2
    decision_value = promote_stage2(matrix, [record.metrics for record in records])
    by_name = {record.metrics.candidate: record for record in records}
    candidates = [
        TuningCandidate(
            name, {key: value for key, value in by_name[name].resolved_config.items() if value != matrix.baseline[key]}
        )
        for name in decision_value.selected
    ]
    decision = _base_decision("stage1", args.artifact_root, records)
    decision.update(
        selected=list(decision_value.selected),
        rejected=decision_value.rejected,
        resolved_candidates=_resolved_records(matrix, candidates),
    )
    write_decision(args.output, decision)


def _promote_finalists_action(args: argparse.Namespace) -> None:
    matrix = load_tuning_matrix(args.tuning_matrix)
    baseline = _records(args, "baseline")
    halve = _records(args, "halve")
    validate_tuning_records(baseline, (("baseline", seed) for seed in matrix.seeds))
    stage2_candidates = _load_candidates(args.decision_root / "stage2.json")
    validate_tuning_records(halve, ((item["name"], seed) for item in stage2_candidates for seed in (42, 43)))
    derived, provenance = derive_stage2_baseline(baseline, matrix.halve_iterations)
    decision_value = promote_finalists(
        matrix, [record for record in derived if record.seed in (42, 43)], [record.metrics for record in halve]
    )
    selected = set(decision_value.selected)
    candidates = [
        TuningCandidate(item["name"], dict(item["overrides"])) for item in stage2_candidates if item["name"] in selected
    ]
    decision = _base_decision("stage2", args.artifact_root, baseline + halve)
    decision.update(
        selected=list(decision_value.selected),
        rejected=decision_value.rejected,
        resolved_candidates=_resolved_records(matrix, candidates),
        baseline_prefix_provenance=provenance,
        methodology=(
            "Stage 2 uses the first 100 aligned samples of each clean 300-iteration "
            "baseline; its final-20 window is iterations 81-100."
        ),
    )
    write_decision(args.output, decision)


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


def _select_winner_action(args: argparse.Namespace) -> None:
    matrix = load_tuning_matrix(args.tuning_matrix)
    baseline = _records(args, "baseline")
    final = _records(args, "final")
    validate_tuning_records(baseline, (("baseline", seed) for seed in matrix.seeds))
    finalists = _load_candidates(args.decision_root / "finalists.json")
    validate_tuning_records(final, ((item["name"], seed) for item in finalists for seed in matrix.seeds))
    qualifications = qualify_finalists(
        matrix, [record.metrics for record in baseline], [record.metrics for record in final]
    )
    configs = {item["name"]: item["resolved_config"] for item in finalists}
    winner = select_winner(matrix, qualifications, configs)
    selected = next(item for item in finalists if item["name"] == winner)
    decision = _base_decision("stage3", args.artifact_root, baseline + final)
    decision.update(
        candidate=winner,
        selected=[winner],
        rejected={name: value.reason for name, value in qualifications.items() if not value.qualified},
        qualifications=_qualification_dict(qualifications),
        overrides=selected["overrides"],
        resolved_config=selected["resolved_config"],
        config_hash=config_hash(selected["resolved_config"]),
    )
    write_decision(args.output, decision)


def _report_action(args: argparse.Namespace) -> None:
    from .statistics import mean_ci95
    from .tuning_reporting import write_tuning_report

    decisions = {
        name: _read_mapping(args.decision_root / filename)
        for name, filename in (
            ("wave2", "wave2.json"),
            ("stage2", "stage2.json"),
            ("finalists", "finalists.json"),
            ("winner", "winner.json"),
        )
    }
    matrix = load_tuning_matrix(DEFAULT_TUNING_MATRIX_PATH)
    baseline = _records(args, "baseline")
    wave1 = _records(args, "wave1")
    wave2 = _records(args, "wave2")
    halve = _records(args, "halve")
    final = _records(args, "final")
    validate_tuning_records(baseline, (("baseline", seed) for seed in matrix.seeds))
    validate_tuning_records(wave1, ((candidate.name, 42) for candidate in matrix.wave1))
    validate_tuning_records(
        wave2,
        ((item["name"], 42) for item in decisions["wave2"]["resolved_candidates"]),
    )
    validate_tuning_records(
        halve,
        ((item["name"], seed) for item in decisions["stage2"]["resolved_candidates"] for seed in (42, 43)),
    )
    validate_tuning_records(
        final,
        ((item["name"], seed) for item in decisions["finalists"]["resolved_candidates"] for seed in matrix.seeds),
    )
    winner = decisions["winner"]
    qualifications = winner["qualifications"]
    winner_runtime = float(qualifications[winner["candidate"]]["runtime"]["mean"])
    clean_runtime = mean_ci95([record.metrics.steady_time(matrix.warmup_iterations) for record in baseline])
    comparisons = json.loads(args.comparison_summary.read_text(encoding="utf-8"))
    legacy_runtime = {
        row["variant"]: float(row["iteration_time_s"]["mean"])
        for row in comparisons
        if row.get("task") == matrix.task and row.get("variant") in {"mjwarp", "physx"}
    }
    all_records = baseline + wave1 + wave2 + halve + final
    rejection_map: dict[str, str] = {}
    for decision in decisions.values():
        rejection_map.update(decision.get("rejected", {}))
    for record in all_records:
        if record.metrics.failure:
            rejection_map[record.run_id] = record.metrics.failure
    stage_rows = (
        ("Wave 1", wave1, decisions["wave2"]),
        ("Wave 1/2", wave1 + wave2, decisions["stage2"]),
        ("Stage 2", halve, decisions["finalists"]),
        ("Stage 3", final, decisions["winner"]),
    )
    report = {
        "winner": winner["candidate"],
        "winner_config": winner["resolved_config"],
        "environment_count": matrix.num_envs,
        "funnel": [
            {
                "stage": label,
                "attempted": len(records),
                "valid": sum(record.metrics.failure is None for record in records),
                "rejected": len(decision.get("rejected", {})),
                "promoted": len(decision.get("selected", [])),
            }
            for label, records, decision in stage_rows
        ],
        "runtime_rows": [
            {
                "stage": record.metrics.stage,
                "candidate": record.metrics.candidate,
                "mean": record.metrics.steady_time(matrix.warmup_iterations),
                "half_width": 0.0,
                "n": 1,
            }
            for record in wave1 + wave2
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
        ],
        "learning_traces": [
            {
                "candidate": f"{record.metrics.candidate} seed {record.metrics.seed}",
                "reward": record.metrics.reward,
                "success": record.metrics.success_rate,
                "episode_length": record.metrics.ep_length,
            }
            for record in halve
            if record.metrics.failure is None
        ],
        "speedups": {
            "clean DVI": clean_runtime.mean / winner_runtime,
            "legacy MJWarp": legacy_runtime["mjwarp"] / winner_runtime,
            "legacy PhysX": legacy_runtime["physx"] / winner_runtime,
        },
        "rejections": [f"{name}: {reason}" for name, reason in sorted(rejection_map.items())],
        "seed_iteration_coverage": (
            "4096 environments; baseline/final seeds 42--44 at 300 iterations; "
            "Wave 1/2 seed 42 at 40; Stage 2 seeds 42--43 at 100"
        ),
        "stage2_baseline_derivation": (
            "first 100 aligned iterations of clean 300-iteration baseline; "
            "final-20 is iterations 81--100; source manifest/event/config hashes "
            "are retained in finalists.json"
        ),
        "legacy_limitations": (
            "MJWarp/PhysX values come from the existing five-variant campaign. "
            "Its manifests lack the current exact source-HEAD and retained "
            "TensorBoard-event hashes, so the speedups are contextual only."
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
