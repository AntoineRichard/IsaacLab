# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic-success end-to-end coverage for report-only processing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.benchmark_comparison.artifacts import finalize_attempt
from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity, write_manifest
from tools.benchmark_comparison.matrix import expand_canary_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.report_cli import main
from tools.benchmark_comparison.validate import attempt_identity

_FIXTURES = Path(__file__).parent / "fixtures"


def test_report_only_cli_normalizes_a_synthetic_raw_success(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    provenance = ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64)
    software = SoftwareIdentity("2.3.2", "5.1.0", "3.11.13", "2.7.0", "5.0.1")
    expansion = expand_canary_matrix(load_matrix())
    manifest = RunSetManifest(
        "2.0",
        RunSet.CANARY,
        "measured",
        provenance,
        HostIdentity(
            "fixture-host",
            "Fixture OS",
            "Fixture CPU",
            32,
            "Fixture GPU",
            "590.48.01",
            "12.8",
            gpu_index=0,
            gpu_uuid="GPU-TEST-0000",
        ),
        software,
        SoftwareIdentity("3.0.0", "6.0.0", "3.12.13", "2.11.0", "5.4.1"),
        expansion=expansion,
    )
    write_manifest(root / "canary" / "manifest.json", manifest)
    attempt = expansion.attempts[0]
    schema = json.loads((_FIXTURES / "schema_runtime.json").read_text(encoding="utf-8"))
    schema["run"].update(task=attempt.concrete_task, seed=attempt.seed, num_envs=attempt.num_envs)
    schema["runtime"]["iterations_completed"] = attempt.bound.value
    schema["versions"].update(
        isaaclab_release=software.isaac_lab,
        isaacsim=software.isaac_sim,
        torch=software.pytorch,
        rsl_rl=software.rsl_rl,
    )
    schema["hardware"].update(cpu_name="Fixture CPU", cpu_count=32)
    identity = attempt_identity(attempt)
    finalize_attempt(
        root,
        attempt,
        command={"identity": identity, "argv": ["synthetic-benchmark"]},
        environment={
            "identity": identity,
            "environment_identity": provenance.environment_identity(attempt.version),
            **provenance.to_json(),
            "values": {},
        },
        stdout="| Driver Version: 590.48.01 | Graphics API: Vulkan\n",
        stderr="",
        exit_status={
            "exit_code": 0,
            "failure_stage": None,
            "timed_out": False,
            "interrupted": False,
            "out_of_memory": False,
            "wall_time_s": 1.0,
        },
        schema=schema,
        measurements=json.loads((_FIXTURES / "generic_runtime.json").read_text(encoding="utf-8")),
    )
    output = root / "canary" / "report"

    assert (
        main(
            [
                "--artifact_root",
                str(root),
                "--run_set",
                "canary",
                "--phase",
                "measured",
                "--output_dir",
                str(output),
            ]
        )
        == 0
    )

    with (output / "raw_runs.csv").open(newline="", encoding="utf-8") as file:
        rows = tuple(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["isaac_sim_version"] == "5.1.0"
