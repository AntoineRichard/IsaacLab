# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for structured preflight metadata probes."""

import pytest

from tools.benchmark_comparison.preflight_metadata import parse_nvidia_identity, software_metadata_probe


def test_software_probe_uses_public_release_version_file() -> None:
    probe = software_metadata_probe()

    assert "open('VERSION'" in probe
    assert "'isaac_lab': dist('isaaclab')" not in probe


def test_nvidia_identity_requires_one_physical_gpu_zero_row_with_uuid() -> None:
    value = parse_nvidia_identity("Fixture GPU, 590.48.01, 100, 0, GPU-01234567-89ab-cdef-0123-456789abcdef\n")

    assert value == (
        "Fixture GPU",
        "590.48.01",
        100,
        "GPU-01234567-89ab-cdef-0123-456789abcdef",
    )


def test_nvidia_identity_rejects_multiple_rows_even_for_matching_models() -> None:
    stdout = "Fixture GPU, 590.48.01, 100, 0, GPU-ZERO\nFixture GPU, 590.48.01, 20, 0, GPU-ONE\n"

    with pytest.raises(ValueError, match="malformed"):
        parse_nvidia_identity(stdout)
