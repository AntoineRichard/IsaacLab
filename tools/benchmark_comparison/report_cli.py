# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate and audit benchmark reports without starting either simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from .manifest import manifest_path, read_manifest
from .matrix import expand_canary_matrix, expand_final_matrix, load_matrix
from .models import RunSet
from .normalize import normalize_run_set, write_normalized_outputs
from .plot import generate_plots
from .report import write_markdown_report

_NORMALIZED_FILES = ("raw_runs.csv", "paired_summary.csv", "failures.csv")


def main(argv: list[str] | None = None) -> int:
    """Run deterministic report-only processing for one completed run set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_root", type=Path, required=True)
    parser.add_argument("--run_set", choices=[value.value for value in RunSet], required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)

    artifact_root = args.artifact_root.resolve()
    run_set = RunSet(args.run_set)
    output_directory = args.output_dir.resolve()
    _validate_output_directory(artifact_root, run_set, output_directory)
    manifest = read_manifest(manifest_path(artifact_root, run_set))
    if manifest.run_set is not run_set:
        raise ValueError("manifest run_set does not match requested run set")
    if manifest.phase != args.phase:
        raise ValueError("manifest phase does not match requested phase")

    raw_hash_contents, raw_file_count = _raw_hash_manifest(artifact_root, run_set, output_directory)
    existing_raw_hashes = output_directory / "raw_artifact_hashes.sha256"
    if existing_raw_hashes.exists() and existing_raw_hashes.read_text(encoding="utf-8") != raw_hash_contents:
        raise ValueError("raw artifact hashes differ from the previously published report")

    expansion = expand_canary_matrix(load_matrix()) if run_set is RunSet.CANARY else expand_final_matrix(load_matrix())
    runs, failures = normalize_run_set(artifact_root, expansion, manifest)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent))
    try:
        normalized = write_normalized_outputs(staging, runs, failures)
        write_markdown_report(
            normalized["raw_runs"],
            normalized["paired_summary"],
            normalized["failures"],
            staging / "report.md",
            manifest=manifest,
            artifact_root=artifact_root,
        )
        generate_plots(normalized["raw_runs"], staging)
        _write_text(staging / "raw_artifact_hashes.sha256", raw_hash_contents)

        generated = tuple(
            sorted(
                (
                    *[staging / name for name in _NORMALIZED_FILES],
                    staging / "report.md",
                    *staging.glob("*.png"),
                    *staging.glob("*.svg"),
                ),
                key=lambda path: path.name,
            )
        )
        generated_hash_contents = _hash_lines(generated, staging)
        _write_text(staging / "generated_hashes.sha256", generated_hash_contents)
        audit = {
            "schema_version": "1.0",
            "run_set": run_set.value,
            "phase": manifest.phase,
            "successful_attempts": len(runs),
            "failed_or_missing_attempts": len(failures),
            "raw_file_count": raw_file_count,
            "generated_file_count": len(generated),
            "manifest_sha256": _sha256(manifest_path(artifact_root, run_set)),
            "raw_hash_manifest_sha256": hashlib.sha256(raw_hash_contents.encode()).hexdigest(),
            "generated_hash_manifest_sha256": hashlib.sha256(generated_hash_contents.encode()).hexdigest(),
        }
        _write_text(
            staging / "audit_summary.json",
            json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        )
        final_raw_hash_contents, final_raw_file_count = _raw_hash_manifest(
            artifact_root, run_set, output_directory, excluded_roots=(staging,)
        )
        if final_raw_hash_contents != raw_hash_contents or final_raw_file_count != raw_file_count:
            raise ValueError("raw artifacts changed during normalization")
        _publish(staging, output_directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return 0


def _validate_output_directory(artifact_root: Path, run_set: RunSet, output_directory: Path) -> None:
    resolved_root = artifact_root.resolve()
    canonical_report = resolved_root / run_set.value / "report"
    if output_directory == canonical_report:
        return
    overlaps_artifacts = (
        output_directory == resolved_root
        or output_directory in resolved_root.parents
        or resolved_root in output_directory.parents
    )
    if overlaps_artifacts:
        raise ValueError("output directory overlaps benchmark artifact root")


def _raw_hash_manifest(
    artifact_root: Path,
    run_set: RunSet,
    output_directory: Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
) -> tuple[str, int]:
    run_set_root = artifact_root / run_set.value
    excluded = (output_directory.resolve(), *(path.resolve() for path in excluded_roots))
    files: list[Path] = []
    for path in run_set_root.rglob("*"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(resolved == root or root in resolved.parents for root in excluded):
            continue
        if path.name in {"manifest.json", "manifest.json.lock"}:
            continue
        files.append(path)
    files.sort(key=lambda path: path.relative_to(artifact_root).as_posix())
    return _hash_lines(files, artifact_root), len(files)


def _hash_lines(paths: tuple[Path, ...] | list[Path], root: Path) -> str:
    return "".join(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, contents: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def _publish(staging: Path, output_directory: Path) -> None:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    backup = output_directory.parent / f".{output_directory.name}.backup-{uuid.uuid4().hex}"
    had_previous = output_directory.exists()
    if had_previous:
        os.replace(output_directory, backup)
    try:
        os.replace(staging, output_directory)
    except BaseException:
        if had_previous:
            os.replace(backup, output_directory)
        raise
    if had_previous:
        shutil.rmtree(backup)


if __name__ == "__main__":
    raise SystemExit(main())
