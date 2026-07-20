# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for process-based environment provenance capture."""

import json
from pathlib import Path
from subprocess import CompletedProcess

from benchmarks.kamino_dvi.environment import probe_environment


def test_probe_environment_captures_packages_newton_path_and_git_state():
    """The environment probe must retain package and source provenance."""
    calls: list[list[str]] = []
    probe_output = "Warp 1.12.0 initialized:\n   CUDA Toolkit 12.9\n" + json.dumps(
        {
            "packages": {"newton": "0.1.0", "torch": "2.11.0"},
            "newton_path": "/venv/site-packages/newton/__init__.py",
            "newton_revision": "c" * 40,
        }
    )

    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == "/repo/.venv-current/bin/python":
            assert command[1] == "-c"
            assert "importlib.metadata" in command[2]
            return CompletedProcess(command, 0, stdout=probe_output, stderr="")
        if command[-2:] == ["rev-parse", "HEAD"]:
            return CompletedProcess(command, 0, stdout="f" * 40 + "\n", stderr="")
        if command[-2:] == ["rev-list", "HEAD"]:
            return CompletedProcess(command, 0, stdout="f" * 40 + "\n" + "a" * 40 + "\n", stderr="")
        if command[-2:] == ["status", "--porcelain"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    provenance = probe_environment(Path("/repo/.venv-current/bin/python"), Path("/repo"), runner=runner)

    assert provenance.packages == {"newton": "0.1.0", "torch": "2.11.0"}
    assert provenance.newton_path == Path("/venv/site-packages/newton/__init__.py")
    assert provenance.newton_revision == "c" * 40
    assert provenance.isaaclab.head == "f" * 40
    assert provenance.isaaclab.ancestors == frozenset({"a" * 40})
    assert provenance.isaaclab.dirty is False
    assert len(calls) == 4
