# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for structured preflight metadata probes."""

from tools.benchmark_comparison.preflight_metadata import software_metadata_probe


def test_software_probe_uses_public_release_version_file() -> None:
    probe = software_metadata_probe()

    assert "open('VERSION'" in probe
    assert "'isaac_lab': dist('isaaclab')" not in probe
