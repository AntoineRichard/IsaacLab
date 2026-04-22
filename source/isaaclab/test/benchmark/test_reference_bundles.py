# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate committed Odin reference bundles against the v1.0 schema."""

import json
from pathlib import Path

import pytest

from isaaclab.test.benchmark.standard_schema import SCHEMA_VERSION

_REFERENCE_ROOT = Path(__file__).resolve().parents[4] / "docs" / "odin" / "reference_runs"


def _iter_bundles():
    if not _REFERENCE_ROOT.exists():
        return
    for entry in sorted(_REFERENCE_ROOT.iterdir()):
        if entry.is_dir() and (entry / "manifest.json").exists():
            yield entry


@pytest.mark.parametrize(
    "bundle",
    list(_iter_bundles()),
    ids=lambda b: b.name if b is not None else "none",
)
def test_bundle_schema_v1(bundle):
    with open(bundle / "manifest.json") as fh:
        manifest = json.load(fh)
    assert manifest["schema_version"] == SCHEMA_VERSION
    for key in ("run_id", "config", "machine", "phases", "artifacts"):
        assert key in manifest, f"{bundle.name}: missing manifest key {key}"

    with open(bundle / "training.json") as fh:
        training = json.load(fh)
    assert training["schema_version"] == SCHEMA_VERSION
    for key in ("run", "versions", "hardware", "runtime", "resources", "learning"):
        assert key in training, f"{bundle.name}: missing training key {key}"
    assert "env_steps_per_s" in training["runtime"]
    assert "final_ema" in training["learning"]["reward"]

    with open(bundle / "startup.json") as fh:
        startup = json.load(fh)
    assert startup["schema_version"] == SCHEMA_VERSION
    for key in ("run", "versions", "hardware", "phases", "config"):
        assert key in startup, f"{bundle.name}: missing startup key {key}"
