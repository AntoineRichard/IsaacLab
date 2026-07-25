# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused manifest-to-schema consistency tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tools.benchmark_comparison.manifest import HostIdentity, RunSetManifest, SoftwareIdentity
from tools.benchmark_comparison.matrix import expand_final_matrix, load_matrix
from tools.benchmark_comparison.models import ExecutionProvenance, RunSet
from tools.benchmark_comparison.normalize import _validate_schema_identity


def _manifest() -> RunSetManifest:
    return RunSetManifest(
        schema_version="1.0",
        run_set=RunSet.FINAL,
        phase="measured",
        provenance=ExecutionProvenance("a" * 40, "b" * 40, "sha256:" + "c" * 64, "d" * 64),
        host=HostIdentity("host", "os", "cpu", 32, "gpu", "590.48.01", "12.8"),
        lab2=SoftwareIdentity("2.3.2", "5.1.0", "3.11.13", "2.7.0", "5.0.1"),
        lab3=SoftwareIdentity("3.0.0", "6.0.0", "3.12.13", "2.11.0", "5.4.1"),
    )


def _schema() -> dict[str, object]:
    return {
        "versions": {
            "isaaclab_release": "2.3.2",
            "isaacsim": "5.1.0",
            "torch": "2.7.0",
            "rsl_rl": "5.0.1",
        },
        "hardware": {
            "hostname": "host",
            "cpu_name": "cpu",
            "cpu_count": 32,
            "gpu_devices": [{"name": "gpu"}],
        },
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("version", "schema versions.pytorch"),
        ("driver", "GPU driver"),
        ("gpu", "schema hardware.gpu_model"),
    ],
)
def test_schema_identity_mismatch_is_rejected(field: str, message: str) -> None:
    attempt = next(item for item in expand_final_matrix(load_matrix()).attempts if item.version.value == "lab2")
    schema = _schema()
    stdout = "| Driver Version: 590.48.01 | Graphics API: Vulkan\n"
    if field == "version":
        schema["versions"]["torch"] = "wrong"
    elif field == "driver":
        stdout = "| Driver Version: 999.0 | Graphics API: Vulkan\n"
    else:
        schema["hardware"]["gpu_devices"][0]["name"] = "other"

    with pytest.raises(ValueError, match=message):
        _validate_schema_identity(schema, stdout, attempt, _manifest())


def test_schema_cpu_brand_may_differ_from_manifest_processor_architecture() -> None:
    attempt = next(item for item in expand_final_matrix(load_matrix()).attempts if item.version.value == "lab2")
    schema = _schema()
    schema["hardware"]["cpu_name"] = "Intel(R) Core(TM) i9-14900K"

    _validate_schema_identity(
        schema,
        "| Driver Version: 590.48.01 | Graphics API: Vulkan\n",
        attempt,
        _manifest(),
    )


def test_manifest_run_set_mismatch_is_not_interchangeable() -> None:
    manifest = replace(_manifest(), run_set=RunSet.CANARY)
    assert manifest.run_set is not RunSet.FINAL
