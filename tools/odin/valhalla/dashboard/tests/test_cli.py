# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the odin-dashboard CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.dashboard import cli as cli_mod


def _write_dispatch(runs_root: Path, dispatch_id: str) -> Path:
    d = runs_root / dispatch_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.3",
        "dispatch_id": dispatch_id,
        "started_at": "2026-04-27T14:13:02Z",
        "ended_at": None,
        "seeds": [42],
        "commit_sha": "abc",
        "fleet": [],
        "jobs": [],
        "skipped": [],
    }
    (d / "dispatch.json").write_text(json.dumps(payload))
    return d


def test_parse_args_defaults(tmp_path):
    ns = cli_mod.parse_args(["--runs-root", str(tmp_path)])
    assert ns.port == 8050
    assert ns.host == "127.0.0.1"
    assert ns.runs_root == tmp_path
    assert ns.dispatch is None
    assert ns.no_browser is False
    assert ns.debug is False


def test_parse_args_explicit(tmp_path):
    ns = cli_mod.parse_args([
        "20260427-141302",
        "--runs-root", str(tmp_path),
        "--port", "9000",
        "--host", "0.0.0.0",
        "--no-browser",
        "--debug",
    ])
    assert ns.dispatch == "20260427-141302"
    assert ns.port == 9000
    assert ns.host == "0.0.0.0"
    assert ns.no_browser is True
    assert ns.debug is True


def test_main_invalid_runs_root_exits_2(capsys):
    rc = cli_mod.main(["--runs-root", "/nonexistent/odin_runs_dir"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "runs-root" in err.lower() or "/nonexistent" in err


def test_main_unknown_dispatch_exits_2(tmp_path, capsys):
    rc = cli_mod.main(["does-not-exist", "--runs-root", str(tmp_path)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "does-not-exist" in err


def test_main_no_browser_suppresses_open(tmp_path, monkeypatch):
    """--no-browser must not call webbrowser.open."""
    _write_dispatch(tmp_path, "20260427-141302")
    open_calls = []

    def _fake_open(url):
        open_calls.append(url)

    def _stub_run_server(self, host=None, port=None, debug=None):
        pass

    monkeypatch.setattr("webbrowser.open", _fake_open)
    monkeypatch.setattr("dash.Dash.run_server", _stub_run_server, raising=False)
    monkeypatch.setattr("dash.Dash.run", _stub_run_server, raising=False)

    rc = cli_mod.main(["20260427-141302", "--runs-root", str(tmp_path), "--no-browser"])
    assert rc == 0
    assert open_calls == []


def test_main_default_calls_browser_open(tmp_path, monkeypatch):
    """Without --no-browser, webbrowser.open is called once."""
    _write_dispatch(tmp_path, "20260427-141302")
    open_calls = []

    def _fake_open(url):
        open_calls.append(url)

    def _stub_run_server(self, host=None, port=None, debug=None):
        pass

    monkeypatch.setattr("webbrowser.open", _fake_open)
    monkeypatch.setattr("dash.Dash.run_server", _stub_run_server, raising=False)
    monkeypatch.setattr("dash.Dash.run", _stub_run_server, raising=False)

    rc = cli_mod.main(["20260427-141302", "--runs-root", str(tmp_path)])
    assert rc == 0
    assert len(open_calls) == 1
    assert open_calls[0].startswith("http://127.0.0.1:8050")


def test_main_port_in_use_exits_4(tmp_path, monkeypatch, capsys):
    _write_dispatch(tmp_path, "20260427-141302")

    def _raise_in_use(self, host=None, port=None, debug=None):
        raise OSError("Address already in use")

    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr("dash.Dash.run_server", _raise_in_use, raising=False)
    monkeypatch.setattr("dash.Dash.run", _raise_in_use, raising=False)

    rc = cli_mod.main(["20260427-141302", "--runs-root", str(tmp_path), "--no-browser"])
    assert rc == 4
    err = capsys.readouterr().err
    assert "in use" in err.lower() or "8050" in err


def test_main_latest_resolves_to_newest(tmp_path, monkeypatch):
    _write_dispatch(tmp_path, "20260424-160119")
    _write_dispatch(tmp_path, "20260427-141302")

    captured: dict = {}

    def _stub_run_server(self, host=None, port=None, debug=None):
        captured["pathname"] = self.layout.children[0].pathname

    monkeypatch.setattr("webbrowser.open", lambda url: None)
    monkeypatch.setattr("dash.Dash.run_server", _stub_run_server, raising=False)
    monkeypatch.setattr("dash.Dash.run", _stub_run_server, raising=False)

    rc = cli_mod.main(["LATEST", "--runs-root", str(tmp_path), "--no-browser"])
    assert rc == 0
    assert captured["pathname"] == "/20260427-141302/dispatch-fleet"
