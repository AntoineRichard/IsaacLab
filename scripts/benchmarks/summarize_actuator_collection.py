# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict reader and reporter for final actuator-collection benchmark evidence.

Only documents named by the immutable selection manifest are numerical input.
Everything else is retained solely as rejected-evidence accounting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ATTEMPT_SCHEMA = "actuator_collection_attempt/v1"
_SELECTION_SCHEMA = "actuator_collection_selection/v1"
_SUMMARY_SCHEMA = "actuator_collection_summary/v1"
_REVISIONS = ("develop", "current", "global")


@dataclass(frozen=True)
class LoadedAttempts:
    """Selected immutable documents and non-selected rejection accounting."""

    manifest: dict[str, Any]
    selected: list[dict[str, Any]]
    rejected_reasons: dict[str, int]
    unselected_attempt_count: int


def _driver() -> Any:
    """Load the adjacent schema and schedule authority without target imports."""
    path = Path(__file__).with_name("benchmark_actuator_collection.py")
    spec = importlib.util.spec_from_file_location("_actuator_collection_driver", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {label}: {path}: object required")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_attempt_path(run_root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("selected attempt path must be a nonempty relative path")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("selected attempt path escapes run root")
    candidate = run_root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(run_root)
    except (OSError, ValueError) as error:
        raise ValueError(f"selected attempt path escapes or is missing: {raw}") from error
    cursor = run_root
    for part in path.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"selected attempt path contains symlink: {raw}")
    if not resolved.is_dir():
        raise ValueError(f"selected attempt path is not a directory: {raw}")
    return resolved


def _probe_worktree(path: Path) -> tuple[str, bool]:
    """Return the immutable HEAD and cleanliness of one selected worktree."""
    head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    if head.returncode != 0 or status.returncode != 0:
        raise ValueError(f"cannot probe worktree: {path}")
    return head.stdout.strip(), bool(status.stdout.strip())


def _validate_manifest(manifest: dict[str, Any], candidate_sha: str) -> dict[str, str]:
    if manifest.get("schema") != _SELECTION_SCHEMA:
        raise ValueError("selection manifest schema")
    if manifest.get("candidate_sha") != candidate_sha:
        raise ValueError("candidate SHA mismatch")
    revisions = manifest.get("revision_shas")
    if not isinstance(revisions, dict) or set(revisions) != set(_REVISIONS):
        raise ValueError("selection manifest exact revision SHA map")
    if revisions["global"] != candidate_sha:
        raise ValueError("candidate SHA must equal global revision SHA")
    if any(not isinstance(value, str) or len(value) != 40 for value in revisions.values()):
        raise ValueError("selection manifest revision SHA")
    digest = manifest.get("harness_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("selection manifest harness SHA")
    attempts = manifest.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("selection manifest attempts")
    if len(attempts) != len(set(attempts)):
        raise ValueError("duplicate selected attempt path")
    return revisions


def _validate_harness(record: dict[str, Any], run_root: Path, digest: str) -> None:
    raw = (record.get("paths") or {}).get("harness")
    if not isinstance(raw, str):
        raise ValueError("selected attempt harness path")
    harness = Path(raw)
    try:
        resolved = harness.resolve(strict=True)
        resolved.relative_to(run_root)
    except (OSError, ValueError) as error:
        raise ValueError("selected harness escapes run root") from error
    if resolved != (run_root / "harness" / "benchmark_actuator_collection.py").resolve():
        raise ValueError("selected harness path is not immutable harness")
    if harness.is_symlink():
        raise ValueError("selected harness is symlinked")
    if _sha256(resolved) != digest:
        raise ValueError("harness bytes differ from manifest")
    digest_file = resolved.with_name("benchmark_actuator_collection.sha256")
    if not digest_file.is_file() or digest_file.read_text(encoding="utf-8").strip() != digest:
        raise ValueError("harness digest file differs")


def _validate_worktrees(record: dict[str, Any], revisions: dict[str, str]) -> None:
    paths = (record.get("paths") or {}).get("worktrees")
    if not isinstance(paths, dict) or set(paths) != set(_REVISIONS):
        raise ValueError("selected worktree path map")
    for revision in _REVISIONS:
        raw = paths[revision]
        if not isinstance(raw, str) or not Path(raw).is_dir():
            raise ValueError(f"missing {revision} worktree")
        head, dirty = _probe_worktree(Path(raw).resolve())
        if head != revisions[revision] or dirty:
            raise ValueError(f"dirty or mismatched {revision} worktree")


def _rejection_reason(record: dict[str, Any]) -> str:
    reasons = (record.get("process") or {}).get("rejection_reasons")
    if isinstance(reasons, list) and reasons:
        return str(reasons[0])
    for member in record.get("members", []):
        failure = member.get("failure") if isinstance(member, dict) else None
        if isinstance(failure, dict) and failure.get("reason"):
            return str(failure["reason"])
    return "rejected"


def load_selected_attempts(run_root: Path, candidate_sha: str) -> LoadedAttempts:
    """Load exactly manifest-selected final evidence and rejected-attempt counts."""
    run_root = run_root.resolve()
    manifest = _read_json(run_root / "accepted-attempts.json", "selection manifest")
    revisions = _validate_manifest(manifest, candidate_sha)
    digest = manifest["harness_sha256"]
    selected: list[dict[str, Any]] = []
    selected_paths: set[Path] = set()
    keys: set[str] = set()
    driver = _driver()
    for raw in manifest["attempts"]:
        attempt = _relative_attempt_path(run_root, raw)
        if attempt in selected_paths:
            raise ValueError("duplicate selected attempt path")
        selected_paths.add(attempt)
        document = attempt / "attempt.json"
        if document.is_symlink() or not document.is_file():
            raise ValueError("selected attempt document missing or symlinked")
        record = _read_json(document, "selected attempt")
        if record.get("schema") != _ATTEMPT_SCHEMA:
            raise ValueError("selected document is not an attempt")
        driver.validate_attempt(record)
        if record.get("status") == "rejected":
            raise ValueError("selected rejected attempt")
        identity = record["identity"]
        if attempt.name != identity.get("attempt_id"):
            raise ValueError("attempt directory identity mismatch")
        if identity.get("candidate_sha") != candidate_sha or identity.get("revision_shas") != revisions:
            raise ValueError("selected candidate or revision identity mismatch")
        if identity.get("harness_sha256") != digest:
            raise ValueError("selected harness identity mismatch")
        observation = identity.get("observation_key")
        if not isinstance(observation, str) or observation in keys:
            raise ValueError("duplicate selected observation key")
        keys.add(observation)
        _validate_harness(record, run_root, digest)
        _validate_worktrees(record, revisions)
        selected.append(record)

    rejection_counts: Counter[str] = Counter()
    all_attempts = sorted(run_root.rglob("attempt.json"))
    for document in all_attempts:
        if document.resolve().parent in selected_paths:
            continue
        record = _read_json(document, "unselected attempt")
        if record.get("schema") == _ATTEMPT_SCHEMA and record.get("status") == "rejected":
            rejection_counts[_rejection_reason(record)] += 1
    return LoadedAttempts(manifest, selected, dict(sorted(rejection_counts.items())), len(all_attempts) - len(selected))


def _schedule_for(matrix: str) -> dict[str, Any]:
    driver = _driver()
    observations = (
        driver.build_coordinate_schedule(6, 6) if matrix == "build" else driver.runtime_coordinate_schedule(6)
    )
    return {item.observation_key: item for item in observations}


def _finite_samples(timing: Any) -> bool:
    samples = timing.get("samples_ms") if isinstance(timing, dict) else None
    return (
        isinstance(samples, list)
        and bool(samples)
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
            for value in samples
        )
    )


def _command_int(record: dict[str, Any], flag: str) -> int:
    command = record.get("command")
    if not isinstance(command, list) or flag not in command:
        raise ValueError(f"missing {flag} in coordinate command")
    try:
        value = int(command[command.index(flag) + 1])
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"invalid {flag} in coordinate command") from error
    if value <= 0:
        raise ValueError(f"invalid {flag} in coordinate command")
    return value


def _matches_total(timing: dict[str, Any], per_field: str, count: int) -> bool:
    """Require a reported total to agree with its named process-level value."""
    try:
        total = float(timing["total_ms"])
        per_value = float(timing[per_field])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(total)
        and math.isfinite(per_value)
        and math.isclose(total, per_value * count, rel_tol=1e-8, abs_tol=1e-9)
    )


def _validate_telemetry(record: dict[str, Any]) -> None:
    telemetry = record.get("telemetry")
    if not isinstance(telemetry, dict):
        raise ValueError("telemetry")
    gpu_pair = record.get("kind") == "pair" and str(record.get("device", "")).startswith("cuda")
    if gpu_pair:
        samples = telemetry.get("samples")
        if (
            telemetry.get("required") is not True
            or telemetry.get("available") is not True
            or telemetry.get("rejection_reasons")
            or not isinstance(samples, dict)
            or len(samples.get("pre", [])) != 20
            or len(samples.get("post", [])) != 20
        ):
            raise ValueError("incomplete accepted GPU telemetry")


def _expected_source_emulation(revision: str, row: dict[str, Any]) -> bool:
    return revision != "global" and row.get("case") == "B3"


def _validate_member_timing(record: dict[str, Any], observation: Any, member: dict[str, Any]) -> None:
    if not _finite_samples(member.get("timing")):
        raise ValueError("missing supported schedule row timing")
    timing = member["timing"]
    iterations = _command_int(record, "--num_iterations")
    if observation.matrix == "runtime":
        if timing.get("application_count") != iterations or timing.get("per_application_ms") is None:
            raise ValueError("runtime total/per-application identity")
        if not _matches_total(timing, "per_application_ms", iterations):
            raise ValueError("runtime total/per-application value mismatch")
    elif observation.phase == "warm":
        if timing.get("construction_count") != iterations or timing.get("per_construction_ms") is None:
            raise ValueError("construction count identity")
        if not _matches_total(timing, "per_construction_ms", iterations):
            raise ValueError("construction total/per-construction value mismatch")
    elif observation.phase == "cold" and timing.get("construction_count") != 1:
        raise ValueError("cold construction count identity")


def _validate_member(record: dict[str, Any], observation: Any, member: dict[str, Any], index: int) -> None:
    revision = observation.revisions[index]
    requested = observation.requested_executions[index]
    row = observation.child_rows[index]
    if member.get("revision") != revision or member.get("requested_execution") != requested:
        raise ValueError("wrong member order or requested mode")
    if member.get("resolved_row") != row or member.get("revision_sha") != record["identity"]["revision_shas"][revision]:
        raise ValueError("wrong member row or revision SHA")
    if member.get("source_emulation") is not _expected_source_emulation(revision, row):
        raise ValueError("source-emulation identity mismatch")
    capability = member.get("capability")
    if not isinstance(capability, dict) or "supported" not in capability:
        raise ValueError("member capability")
    if observation.unsupported_reason is not None:
        if capability.get("supported") is not False or not capability.get("reason"):
            raise ValueError("unsupported capability/reason")
        if member.get("effective_execution") is not None or member.get("timing") is not None:
            raise ValueError("unsupported member timing")
        return
    if capability.get("supported") is not True or member.get("process", {}).get("returncode") != 0:
        raise ValueError("selected child or process rejection")
    if member.get("effective_execution") != row.get("effective_execution", requested):
        raise ValueError("effective execution mismatch")
    if requested == "graph" and member.get("execution", {}).get("graph_capture_live") is not True:
        raise ValueError("graph request was not a live graph")
    _validate_member_timing(record, observation, member)


def _validate_observation(record: dict[str, Any], observation: Any) -> None:
    metadata = record.get("metadata") or {}
    for name in ("matrix", "row_key", "comparison", "mode_pair", "pair_id", "order", "phase"):
        if metadata.get(name) != getattr(observation, name):
            raise ValueError(f"schedule {name} mismatch")
    if record.get("kind") != observation.kind or record.get("boundary") != observation.boundary:
        raise ValueError("schedule kind or boundary mismatch")
    if record.get("pair_id") != observation.pair_id or record.get("pair_order") != observation.order:
        raise ValueError("schedule pair identity mismatch")
    if observation.unsupported_reason is not None:
        if record.get("status") != "unsupported" or len(record["members"]) != 1:
            raise ValueError("unsupported schedule status")
    elif record.get("status") != "accepted":
        raise ValueError("selected child or process rejection")
    _validate_telemetry(record)
    if record.get("status") == "accepted" and (record.get("process") or {}).get("rejection_reasons"):
        raise ValueError("selected process rejection")
    if len(record["members"]) != len(observation.revisions):
        raise ValueError("wrong scheduled member count")
    for index, member in enumerate(record["members"]):
        _validate_member(record, observation, member, index)


def _validate_schedule(records: list[dict[str, Any]]) -> None:
    matrices = {record.get("metadata", {}).get("matrix") for record in records}
    if len(matrices) != 1 or matrices - {"build", "runtime"}:
        raise ValueError("mixed or missing matrix identity")
    expected = _schedule_for(next(iter(matrices)))
    actual = {record["identity"]["observation_key"]: record for record in records}
    if set(actual) != set(expected):
        missing, extra = set(expected) - set(actual), set(actual) - set(expected)
        raise ValueError(f"final schedule mismatch: missing={len(missing)}, extra={len(extra)}")
    for key, record in actual.items():
        _validate_observation(record, expected[key])


def validate_pair_ids(pair_ids: list[str]) -> None:
    """Require precisely the six immutable pair identifiers."""
    if sorted(pair_ids) != [f"{number:02}" for number in range(1, 7)]:
        raise ValueError("pair IDs must be exactly 01 through 06")


def validate_orders(orders: list[str]) -> None:
    """Require the frozen three-then-three counterbalanced orders."""
    if len(orders) != 6 or len(set(orders[:3])) != 1 or len(set(orders[3:])) != 1 or orders[0] == orders[3]:
        raise ValueError("unbalanced pair order")


def validate_records(records: list[dict[str, Any]], candidate_sha: str) -> None:
    """Keep the original direct-record validation contract for unit fixtures."""
    driver = _driver()
    harnesses, global_shas, keys = set(), set(), set()
    for record in records:
        driver.validate_attempt(record)
        identity = record["identity"]
        if identity["candidate_sha"] != candidate_sha:
            raise ValueError("candidate SHA mismatch")
        harnesses.add(identity["harness_sha256"])
        key = (identity["observation_key"], identity["attempt_id"])
        if key in keys:
            raise ValueError("duplicate complete key")
        keys.add(key)
        for member in record["members"]:
            if member["revision"] == "global":
                global_shas.add(member["revision_sha"])
    if len(harnesses) != 1:
        raise ValueError("harness identity mismatch")
    if len(global_shas) > 1:
        raise ValueError("mixed global SHA")


def _process_value(record: dict[str, Any], member: dict[str, Any]) -> float:
    timing = member["timing"]
    if record.get("boundary") == "runtime_application":
        value = timing.get("per_application_ms")
        if value is None:
            value = float(np.median(timing["samples_ms"]))
    else:
        value = timing.get("per_construction_ms")
        if value is None:
            value = float(np.median(timing["samples_ms"]))
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid process-level timing statistic")
    return float(value)


def _pair_values(record: dict[str, Any]) -> tuple[float, float]:
    members = record["members"]
    if len(members) != 2 or not all(member["capability"]["supported"] for member in members):
        raise ValueError("unsupported member in accepted comparison")
    baseline, global_member = (
        (members[0], members[1]) if members[0]["revision"] != "global" else (members[1], members[0])
    )
    return _process_value(record, baseline), _process_value(record, global_member)


def _bootstrap(ratios: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(ratios, size=(10_000, len(ratios)), replace=True), axis=1)
    return [float(value) for value in np.percentile(samples, (2.5, 97.5))]


def _comparison_report(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["kind"] == "pair":
            metadata = record["metadata"]
            identity = "|".join((metadata["row_key"], metadata["mode_pair"], metadata["phase"]))
            grouped.setdefault(identity, []).append(record)
    comparisons = []
    for identity, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["metadata"]["pair_id"])
        validate_pair_ids([item["metadata"]["pair_id"] for item in items])
        validate_orders([item["metadata"]["order"] for item in items])
        values = np.asarray([_pair_values(item) for item in items], dtype=float)
        ratios = values[:, 1] / values[:, 0]
        comparisons.append(
            {
                "comparison_identity": identity,
                "observation_key": items[0]["metadata"].get("row_key", identity),
                "pair_ids": [item["metadata"]["pair_id"] for item in items],
                "orders": [item["metadata"]["order"] for item in items],
                "accepted_pair_count": len(items),
                "baseline_process_medians_ms": values[:, 0].tolist(),
                "global_process_medians_ms": values[:, 1].tolist(),
                "ratio_median": float(np.median(ratios)),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_p95": float(np.percentile(ratios, 95)),
                "ratio_dispersion": float(np.std(ratios)),
                "ratio_bootstrap_95": _bootstrap(ratios, seed),
            }
        )
    return comparisons


def summarize_records(records: list[dict[str, Any]], candidate_sha: str, bootstrap_seed: int = 42) -> dict[str, Any]:
    """Return process-level paired statistics for direct legacy unit fixtures."""
    validate_records(records, candidate_sha)
    accepted = [record for record in records if record["status"] == "accepted"]
    rejected = [record for record in records if record["status"] == "rejected"]
    paired = []
    for record in accepted:
        if record["kind"] == "pair" and "/pair-" in record["identity"]["observation_key"]:
            record.setdefault("metadata", {}).setdefault(
                "mode_pair", record["identity"]["observation_key"].rsplit("/pair-", 1)[0]
            )
            record["metadata"].setdefault("pair_id", record["identity"]["observation_key"].rsplit("pair-", 1)[-1])
            record["metadata"].setdefault("order", record.get("pair_order"))
            record["metadata"].setdefault("row_key", record["identity"]["observation_key"].rsplit("/pair-", 1)[0])
            record["metadata"].setdefault("phase", "legacy")
            paired.append(record)
    return {
        "schema": _SUMMARY_SCHEMA,
        "candidate_sha": candidate_sha,
        "bootstrap_seed": bootstrap_seed,
        "accepted_attempt_count": len(accepted),
        "rejected_attempt_count": len(rejected),
        "unsupported_attempt_count": len(records) - len(accepted) - len(rejected),
        "comparisons": _comparison_report(paired, bootstrap_seed),
    }


def summarize_run(run_root: Path, candidate_sha: str, bootstrap_seed: int = 42) -> dict[str, Any]:
    """Strictly validate a final run root before calculating any statistic."""
    loaded = load_selected_attempts(run_root, candidate_sha)
    _validate_schedule(loaded.selected)
    report = {
        "schema": _SUMMARY_SCHEMA,
        "candidate_sha": candidate_sha,
        "bootstrap_seed": bootstrap_seed,
        "selection_identity": {
            "revision_shas": loaded.manifest["revision_shas"],
            "harness_sha256": loaded.manifest["harness_sha256"],
            "selected_attempt_paths": loaded.manifest["attempts"],
        },
        "accepted_attempt_count": sum(record["status"] == "accepted" for record in loaded.selected),
        "unsupported_attempt_count": sum(record["status"] == "unsupported" for record in loaded.selected),
        "unselected_attempt_count": loaded.unselected_attempt_count,
        "immutable_rejected_attempts": loaded.rejected_reasons,
        "comparisons": _comparison_report(loaded.selected, bootstrap_seed),
        "capabilities": [
            {
                "observation_key": record["identity"]["observation_key"],
                "status": record["status"],
                "members": [
                    {
                        "revision": member["revision"],
                        "supported": member["capability"]["supported"],
                        "reason": member["capability"]["reason"],
                    }
                    for member in record["members"]
                ],
                "telemetry": record["telemetry"],
                "process": record.get("process", {}),
                "structural": [member.get("structural") for member in record["members"]],
                "counters": [member.get("counters") for member in record["members"]],
            }
            for record in loaded.selected
        ],
    }
    return report


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    """Atomically write identity-parallel JSON, CSV and Markdown reports."""
    _atomic_write(output_dir / "benchmark-summary.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows = [
        {
            "record_type": "comparison",
            "identity": item["comparison_identity"] if "comparison_identity" in item else item["observation_key"],
            "status": "accepted",
            "ratio_median": item["ratio_median"],
            "ratio_bootstrap_95": json.dumps(item["ratio_bootstrap_95"]),
        }
        for item in report.get("comparisons", [])
    ]
    rows.extend(
        {
            "record_type": "capability",
            "identity": item["observation_key"],
            "status": item["status"],
            "ratio_median": "",
            "ratio_bootstrap_95": "",
        }
        for item in report.get("capabilities", [])
    )
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=("record_type", "identity", "status", "ratio_median", "ratio_bootstrap_95")
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(output_dir / "benchmark-summary.csv", buffer.getvalue())
    lines = ["# Actuator collection benchmark summary", "", f"Candidate SHA: `{report['candidate_sha']}`", ""]
    lines.extend(
        f"- Comparison `{row['identity']}`: median ratio {row['ratio_median']:.6g}"
        for row in rows
        if row["record_type"] == "comparison"
    )
    lines.extend(
        f"- Capability `{row['identity']}`: {row['status']}" for row in rows if row["record_type"] == "capability"
    )
    _atomic_write(output_dir / "benchmark-summary.md", "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Validate one final run root and write its three reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--candidate_sha", required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    write_outputs(summarize_run(args.run_root, args.candidate_sha, args.bootstrap_seed), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
