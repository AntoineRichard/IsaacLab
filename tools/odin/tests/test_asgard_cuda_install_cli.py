# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the odin-cuda CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.cuda_install import CheckResult
from tools.odin.asgard.cuda_install_cli import main, parse_args


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


def test_parse_args_check_minimal(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["check", "--fleet", str(fleet_path)])
    assert args.subcommand == "check"
    assert args.fleet == fleet_path
    assert args.floor == "12.4"
    assert args.verbose is False


def test_parse_args_check_custom_floor(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["check", "--fleet", str(fleet_path), "--floor", "12.6", "--verbose"])
    assert args.floor == "12.6"
    assert args.verbose is True


def test_parse_args_no_subcommand_errors(tmp_path: Path):
    with pytest.raises(SystemExit):
        parse_args([])


def test_main_check_exit_zero_when_all_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_check_fleet(fleet, *, ssh, floor, parallel):
        return [
            CheckResult(host="v1.internal", status="ok", driver="575.1", cuda="12.9"),
            CheckResult(host="v2.internal", status="ok", driver="575.1", cuda="12.9"),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.check_fleet", _fake_check_fleet)
    code = main(["check", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "v1.internal" in out
    assert "12.9" in out
    assert "ok" in out


def test_main_check_exit_one_when_any_below_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _fake_check_fleet(fleet, *, ssh, floor, parallel):
        return [
            CheckResult(host="v1", status="ok", driver="575.1", cuda="12.9"),
            CheckResult(host="v2", status="needs-upgrade", driver="535.1", cuda="12.2"),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.check_fleet", _fake_check_fleet)
    code = main(["check", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "needs-upgrade" in out
    assert "v2" in out


import json

from tools.odin.asgard.cuda_install import CudaInstallResult


def _write_running_dispatch(runs_root: Path, dispatch_id: str) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    d = runs_root / dispatch_id
    d.mkdir()
    (d / "dispatch.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "dispatch_id": dispatch_id,
                "started_at": "2026-04-27T10:00:00+00:00",
                "ended_at": None,
                "seeds": [42],
                "commit_sha": "abc1234",
                "fleet": [],
                "jobs": [],
                "skipped": [],
            }
        )
    )


def test_parse_args_install_minimal(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(["install", "--fleet", str(fleet_path)])
    assert args.subcommand == "install"
    assert args.target == "12.9"
    assert args.floor == "12.4"
    assert args.sequential is False
    assert args.yes is False
    assert args.force is False
    assert args.reboot_timeout == 600
    assert args.runs_root == Path("odin_runs")


def test_parse_args_install_all_flags(tmp_path: Path):
    fleet_path = _write_fleet_yaml(tmp_path)
    args = parse_args(
        [
            "install",
            "--fleet",
            str(fleet_path),
            "--floor",
            "12.6",
            "--target",
            "12.8",
            "--sequential",
            "--yes",
            "--force",
            "--reboot-timeout",
            "900",
            "--runs-root",
            "/tmp/runs",
            "--verbose",
        ]
    )
    assert args.target == "12.8"
    assert args.floor == "12.6"
    assert args.sequential is True
    assert args.yes is True
    assert args.force is True
    assert args.reboot_timeout == 900
    assert args.runs_root == Path("/tmp/runs")


def test_main_install_refuses_with_active_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)
    runs_root = tmp_path / "odin_runs"
    _write_running_dispatch(runs_root, "20260427-active")

    # install_fleet must NOT be called.
    def _explode(*args, **kwargs):
        raise AssertionError("install_fleet must not run when dispatch is active")

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.install_fleet", _explode)
    code = main(
        [
            "install",
            "--fleet",
            str(fleet_path),
            "--runs-root",
            str(runs_root),
            "--yes",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2
    assert "20260427-active" in out
    assert "--force" in out  # tells the user how to override


def test_main_install_force_overrides_dispatch_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fleet_path = _write_fleet_yaml(tmp_path)
    runs_root = tmp_path / "odin_runs"
    _write_running_dispatch(runs_root, "20260427-active")

    captured = {}

    def _fake_install_fleet(fleet, *, ssh, floor, target, reboot_timeout_s, parallel, verbose):
        captured["called"] = True
        return [
            CudaInstallResult(host="v1.internal", ok=True, skipped=True),
            CudaInstallResult(host="v2.internal", ok=True, skipped=True),
        ]

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.install_fleet", _fake_install_fleet)
    code = main(
        [
            "install",
            "--fleet",
            str(fleet_path),
            "--runs-root",
            str(runs_root),
            "--yes",
            "--force",
        ]
    )
    assert code == 0
    assert captured.get("called") is True


def test_main_install_yes_skips_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    monkeypatch.setattr(
        "tools.odin.asgard.cuda_install_cli.install_fleet",
        lambda fleet, **kw: [
            CudaInstallResult(host="v1.internal", ok=True),
            CudaInstallResult(host="v2.internal", ok=True),
        ],
    )
    code = main(["install", "--fleet", str(fleet_path), "--yes"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Proceed? [y/N]" not in out


def test_main_install_prompt_no_aborts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)

    def _explode(*args, **kwargs):
        raise AssertionError("install_fleet must not run after a 'no' answer")

    monkeypatch.setattr("tools.odin.asgard.cuda_install_cli.install_fleet", _explode)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    code = main(["install", "--fleet", str(fleet_path)])
    out = capsys.readouterr().out
    assert code == 3
    assert "aborted" in out.lower()


def test_main_install_exit_one_when_any_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    fleet_path = _write_fleet_yaml(tmp_path)
    monkeypatch.setattr(
        "tools.odin.asgard.cuda_install_cli.install_fleet",
        lambda fleet, **kw: [
            CudaInstallResult(host="v1.internal", ok=True),
            CudaInstallResult(host="v2.internal", ok=False, message="apt-get install failed"),
        ],
    )
    code = main(["install", "--fleet", str(fleet_path), "--yes"])
    assert code == 1
    out = capsys.readouterr().out
    assert "1/2" in out
    assert "v2.internal" in out
