# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Strict reader and reporter for ``actuator_collection_attempt/v1`` documents."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _driver():
    path = Path(__file__).with_name("benchmark_actuator_collection.py")
    spec = importlib.util.spec_from_file_location("_actuator_collection_driver", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_pair_ids(pair_ids: list[str]) -> None:
    """Require precisely the six immutable pair identifiers."""
    if sorted(pair_ids) != [f"{number:02}" for number in range(1, 7)]:
        raise ValueError("pair IDs must be exactly 01 through 06")


def validate_orders(orders: list[str]) -> None:
    """Require the frozen three-then-three counterbalanced orders."""
    if len(orders) != 6 or len(set(orders[:3])) != 1 or len(set(orders[3:])) != 1 or orders[0] == orders[3]:
        raise ValueError("unbalanced pair order")


def validate_records(records: list[dict[str, Any]], candidate_sha: str) -> None:
    """Validate identity and narrow attempt invariants before statistics."""
    driver = _driver()
    harnesses: set[str] = set()
    keys: set[tuple[str, str]] = set()
    global_shas: set[str] = set()
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


def _pair_medians(record: dict[str, Any]) -> tuple[float, float]:
    members = record["members"]
    if len(members) != 2:
        raise ValueError("wrong pair member count")
    first, second = members
    if not first["capability"]["supported"] or not second["capability"]["supported"]:
        raise ValueError("unsupported member in accepted comparison")
    first_samples = first.get("timing", {}).get("samples_ms")
    second_samples = second.get("timing", {}).get("samples_ms")
    if not first_samples or not second_samples:
        raise ValueError("missing supported row timing")
    baseline, global_member = (first, second) if first["revision"] != "global" else (second, first)
    return float(np.median(baseline["timing"]["samples_ms"])), float(np.median(global_member["timing"]["samples_ms"]))


def _bootstrap(ratios: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(ratios, size=(10_000, len(ratios)), replace=True), axis=1)
    return tuple(float(value) for value in np.percentile(samples, (2.5, 97.5)))


def summarize_records(records: list[dict[str, Any]], candidate_sha: str, bootstrap_seed: int = 42) -> dict[str, Any]:
    """Return accepted paired statistics while retaining rejected evidence counts."""
    validate_records(records, candidate_sha)
    accepted = [record for record in records if record["status"] == "accepted"]
    rejected = [record for record in records if record["status"] == "rejected"]
    comparisons: list[dict[str, Any]] = []
    by_key: dict[str, list[dict[str, Any]]] = {}
    for record in accepted:
        if record["kind"] == "pair" and "/pair-" in record["identity"]["observation_key"]:
            key = record["identity"]["observation_key"].rsplit("/pair-", 1)[0]
            by_key.setdefault(key, []).append(record)
    for key, items in sorted(by_key.items()):
        pair_ids = [item["identity"]["observation_key"].rsplit("pair-", 1)[-1] for item in items]
        validate_pair_ids(pair_ids)
        orders = [
            item.get("pair_order") for item in sorted(items, key=lambda item: item["identity"]["observation_key"])
        ]
        if any(order is None for order in orders):
            raise ValueError("missing pair order")
        validate_orders(orders)
        medians = np.array([_pair_medians(item) for item in items], dtype=float)
        ratios = medians[:, 1] / medians[:, 0]
        comparisons.append(
            {
                "observation_key": key,
                "accepted_pair_count": len(items),
                "baseline_process_medians_ms": medians[:, 0].tolist(),
                "global_process_medians_ms": medians[:, 1].tolist(),
                "ratio_median": float(np.median(ratios)),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_p95": float(np.percentile(ratios, 95)),
                "ratio_dispersion": float(np.std(ratios)),
                "ratio_bootstrap_95": list(_bootstrap(ratios, bootstrap_seed)),
            }
        )
    return {
        "schema": "actuator_collection_summary/v1",
        "candidate_sha": candidate_sha,
        "bootstrap_seed": bootstrap_seed,
        "accepted_attempt_count": len(accepted),
        "rejected_attempt_count": len(rejected),
        "unsupported_attempt_count": len(records) - len(accepted) - len(rejected),
        "comparisons": comparisons,
    }


def _load_attempts(run_root: Path) -> list[dict[str, Any]]:
    """Load immutable final documents only."""
    return [json.loads(path.read_text()) for path in sorted(run_root.rglob("attempt.json"))]


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    """Write JSON, CSV and Markdown output derived from one validated report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark-summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "benchmark-summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "observation_key",
                "accepted_pair_count",
                "ratio_median",
                "ratio_mean",
                "ratio_p95",
                "ratio_dispersion",
                "ratio_bootstrap_95",
            ),
        )
        writer.writeheader()
        writer.writerows(report["comparisons"])
    lines = [
        "# Actuator collection benchmark summary",
        "",
        f"Candidate SHA: `{report['candidate_sha']}`",
        "",
        "| Observation | Pairs | Median ratio | 95% bootstrap interval |",
        "| --- | ---: | ---: | --- |",
    ]
    lines.extend(
        "| {observation} | {pairs} | {median:.6g} | {interval} |".format(
            observation=item["observation_key"],
            pairs=item["accepted_pair_count"],
            median=item["ratio_median"],
            interval=item["ratio_bootstrap_95"],
        )
        for item in report["comparisons"]
    )
    (output_dir / "benchmark-summary.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Validate a run root and write its three strict reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--candidate_sha", required=True)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize_records(_load_attempts(args.run_root), args.candidate_sha, args.bootstrap_seed)
    write_outputs(report, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
