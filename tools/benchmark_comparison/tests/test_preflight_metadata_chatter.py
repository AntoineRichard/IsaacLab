# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for exact metadata extraction from wrapper output."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tools.benchmark_comparison.preflight_metadata import metadata_object, software_metadata_probe


def test_metadata_parser_accepts_one_marked_object_amid_wrapper_chatter() -> None:
    value = metadata_object(
        "[INFO] Using python from: /workspace/isaaclab/_isaac_sim/python.sh\n"
        '__ISAACLAB_BENCHMARK_METADATA__{"hostname":"host"}\n',
        "host",
    )

    assert value == {"hostname": "host"}


def test_metadata_parser_rejects_multiple_marked_objects() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        metadata_object(
            '__ISAACLAB_BENCHMARK_METADATA__{"hostname":"first"}\n'
            '__ISAACLAB_BENCHMARK_METADATA__{"hostname":"second"}\n',
            "host",
        )


def test_software_probe_reads_installed_isaac_sim_build_version() -> None:
    probe = software_metadata_probe()

    assert "find_spec('isaacsim')" in probe
    assert "submodule_search_locations" in probe
    assert "'isaac_sim': dist('isaacsim')" not in probe


def test_software_probe_reads_standalone_build_from_isaacsim_path(tmp_path) -> None:
    packages = tmp_path / "packages"
    (packages / "isaacsim").mkdir(parents=True)
    torch = packages / "torch"
    torch.mkdir()
    (torch / "__init__.py").write_text(
        "__version__ = '2.7.0+cu128'\nclass version:\n    cuda = '12.8'\n",
        encoding="utf-8",
    )
    distribution = packages / "rsl_rl_lib-5.0.1.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text("Name: rsl-rl-lib\nVersion: 5.0.1\n", encoding="utf-8")
    standalone = tmp_path / "isaac-sim"
    standalone.mkdir()
    exact_build = "5.1.0-rc.19+release.26219.9c81211b.gl"
    (standalone / "VERSION").write_text(exact_build + "\n", encoding="utf-8")
    (tmp_path / "VERSION").write_text("2.3.2\n", encoding="utf-8")
    environment = {
        **os.environ,
        "ISAACSIM_PATH": str(standalone),
        "PYTHONPATH": str(packages),
    }

    result = subprocess.run(
        [sys.executable, "-c", software_metadata_probe()],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert metadata_object(result.stdout, "lab2")["isaac_sim"] == exact_build
