# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.cli import main, parse_args, resolve_dispatch_dir


def _mkdir_dispatch(parent: Path, name: str) -> Path:
    d = parent / name
    d.mkdir()
    return d


def test_parse_args_minimal():
    args = parse_args(["20260423-100000"])
    assert args.dispatch_id == "20260423-100000"
    assert str(args.runs_root) == "odin_runs"
    assert args.divergence_z == 2.0
    assert args.overwrite is True
    assert args.quiet is False


def test_parse_args_all_flags():
    args = parse_args(
        [
            "LATEST",
            "--runs-root",
            "/tmp/runs",
            "--divergence-z",
            "3.5",
            "--no-overwrite",
            "--quiet",
        ]
    )
    assert args.dispatch_id == "LATEST"
    assert str(args.runs_root) == "/tmp/runs"
    assert args.divergence_z == 3.5
    assert args.overwrite is False
    assert args.quiet is True


def test_resolve_dispatch_dir_exact_name(tmp_path: Path):
    d = _mkdir_dispatch(tmp_path, "20260423-100000")
    assert resolve_dispatch_dir(tmp_path, "20260423-100000") == d


def test_resolve_dispatch_dir_latest(tmp_path: Path):
    _mkdir_dispatch(tmp_path, "20260422-120000")
    newest = _mkdir_dispatch(tmp_path, "20260423-150000")
    _mkdir_dispatch(tmp_path, "20260423-100000")
    assert resolve_dispatch_dir(tmp_path, "LATEST") == newest


def test_resolve_dispatch_dir_latest_empty_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No prior dispatch"):
        resolve_dispatch_dir(tmp_path, "LATEST")


def test_resolve_dispatch_dir_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_dispatch_dir(tmp_path, "does-not-exist")


def test_cli_main_writes_aggregate(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dispatch = _mkdir_dispatch(tmp_path, "20260423-100000")
    (dispatch / "dispatch.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dispatch_id": "20260423-100000",
                "started_at": "2026-04-23T09:59:00Z",
                "ended_at": "2026-04-23T10:10:00Z",
                "seeds": [42],
                "commit_sha": "",
                "fleet": [],
                "jobs": [],
            }
        )
    )
    exit_code = main(["20260423-100000", "--runs-root", str(tmp_path)])
    assert exit_code == 0
    assert (dispatch / "aggregate.json").exists()
    out = capsys.readouterr().out
    assert "aggregate.json" in out or "rows" in out  # summary line printed


def test_cli_main_quiet_suppresses_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dispatch = _mkdir_dispatch(tmp_path, "20260423-100000")
    (dispatch / "dispatch.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dispatch_id": "20260423-100000",
                "started_at": "",
                "ended_at": "",
                "seeds": [42],
                "commit_sha": "",
                "fleet": [],
                "jobs": [],
            }
        )
    )
    exit_code = main(["20260423-100000", "--runs-root", str(tmp_path), "--quiet"])
    assert exit_code == 0
    assert capsys.readouterr().out == ""
