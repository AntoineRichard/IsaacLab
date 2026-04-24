# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the odin-bootstrap CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.bootstrap import BootstrapResult
from tools.odin.asgard.bootstrap_cli import main, parse_args


def _write_fleet_yaml(tmp_path: Path) -> Path:
    content = """\
fleet_name: test
default_ssh_user: odin
default_ssh_key: ~/.ssh/id_ed25519
hosts:
  - host: v1.internal
  - host: v2.internal
"""
    path = tmp_path / "fleet.yaml"
    path.write_text(content)
    return path


def test_parse_args_minimal(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["--fleet", str(fleet_path)])
    assert args.fleet == fleet_path
    assert args.build_timeout == 1800
    assert args.sequential is False
    assert args.verbose is False


def test_parse_args_all_flags(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(
        [
            "--fleet",
            str(fleet_path),
            "--build-timeout",
            "3600",
            "--sequential",
            "--verbose",
        ]
    )
    assert args.build_timeout == 3600
    assert args.sequential is True
    assert args.verbose is True


def test_main_exit_zero_when_all_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        return [
            BootstrapResult(host="v1.internal", ok=True),
            BootstrapResult(host="v2.internal", ok=True),
        ]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    exit_code = main(["--fleet", str(fleet_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "bootstrap complete: 2/2 hosts ok" in out


def test_main_exit_one_when_any_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        return [
            BootstrapResult(host="v1.internal", ok=True),
            BootstrapResult(host="v2.internal", ok=False, message="ssh unreachable: conn refused"),
        ]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    exit_code = main(["--fleet", str(fleet_path)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "bootstrap complete: 1/2 hosts ok" in out
    assert "v2.internal" in out
    assert "ssh unreachable" in out


def test_main_sequential_flag_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fleet_path = _write_fleet_yaml(tmp_path)
    recorded: dict = {}

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        recorded["parallel"] = parallel
        return [BootstrapResult(host="v1.internal", ok=True), BootstrapResult(host="v2.internal", ok=True)]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    main(["--fleet", str(fleet_path), "--sequential"])
    assert recorded["parallel"] is False


def test_main_build_timeout_passes_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fleet_path = _write_fleet_yaml(tmp_path)
    recorded: dict = {}

    def _fake_bootstrap_fleet(fleet, working_tree, *, ssh, rsync, build_timeout_s, parallel, verbose):
        recorded["build_timeout_s"] = build_timeout_s
        return [BootstrapResult(host="v1.internal", ok=True), BootstrapResult(host="v2.internal", ok=True)]

    monkeypatch.setattr(
        "tools.odin.asgard.bootstrap_cli.bootstrap_fleet",
        _fake_bootstrap_fleet,
    )
    main(["--fleet", str(fleet_path), "--build-timeout", "3600"])
    assert recorded["build_timeout_s"] == 3600
