# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for DataLayer.read_running_job_tail."""

from __future__ import annotations

import json
import subprocess

from tools.odin.valhalla.dashboard import data as data_mod
from tools.odin.valhalla.dashboard.data import DataLayer


class _Result:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_run(monkeypatch, result: _Result, calls: list[list[str]]):
    def _fake_run(argv, *, capture_output, timeout, check):
        calls.append(list(argv))
        assert capture_output is True
        # Mirrors data._RUNNING_TAIL_TIMEOUT_S — sized for slow ARM (DGX Spark)
        # SSH handshakes; bump in lockstep if the constant moves.
        assert timeout == 60
        assert check is False
        return result

    monkeypatch.setattr(data_mod, "_subprocess_run", _fake_run)


def _marked(source: str, lines: list[str]) -> bytes:
    return (f"__odin_tail_source__:{source}\n" + "\n".join(lines) + "\n").encode()


def test_read_running_tail_returns_lines_from_training_log_when_present(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    expected = [f"line-{i}" for i in range(50)]
    _install_fake_run(monkeypatch, _Result(stdout=_marked("training.stdout.log", expected)), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-1", host="v1")

    assert lines == expected
    assert calls[0][0] == "ssh"
    assert "-o" in calls[0]
    assert "StrictHostKeyChecking=accept-new" in calls[0]
    assert "ConnectTimeout=5" in calls[0]
    assert "BatchMode=yes" in calls[0]
    assert calls[0][-2] == "horde@v1"
    assert "docker exec isaac-lab-base bash -c" in calls[0][-1]
    assert "training.stdout.log" in calls[0][-1]


def test_read_running_tail_falls_back_to_startup_when_training_empty(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    _install_fake_run(monkeypatch, _Result(stdout=_marked("startup.stdout.log", ["startup ready"])), calls)

    entry = DataLayer(tmp_path).read_running_job_tail_payload("20260430-110509", "run-2", host="v1")

    assert entry == {"source": "startup.stdout.log", "lines": ["startup ready"], "warning": None}
    remote_cmd = calls[0][-1]
    assert remote_cmd.index("training.stdout.log") < remote_cmd.index("startup.stdout.log")


def test_read_running_tail_returns_empty_list_when_no_logs_yet(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    _install_fake_run(monkeypatch, _Result(stdout=b""), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-3", host="v1")

    assert lines == []


def test_read_running_tail_caps_at_n_lines(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    all_lines = [f"line-{i}" for i in range(200)]
    _install_fake_run(monkeypatch, _Result(stdout=_marked("training.stdout.log", all_lines)), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-4", host="v1", n=50)

    assert lines == all_lines[-50:]


def test_read_running_tail_uses_custom_ssh_key_user_container_and_line_count(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    key = tmp_path / "id_ed25519"
    key.write_text("fake")
    _install_fake_run(monkeypatch, _Result(stdout=_marked("training.stdout.log", ["x"])), calls)

    DataLayer(tmp_path).read_running_job_tail(
        "20260430-110509",
        "run-5",
        host="10.0.0.5",
        ssh_user="odin",
        ssh_key=key,
        container_name="custom-container",
        n=17,
    )

    argv = calls[0]
    assert "-i" in argv
    assert argv[argv.index("-i") + 1] == str(key)
    assert argv[-2] == "odin@10.0.0.5"
    assert "docker exec custom-container bash -c" in argv[-1]
    assert "tail -n 17" in argv[-1]


def test_read_running_tail_ssh_failure_returns_empty_with_warning(tmp_path, monkeypatch, capsys):
    calls: list[list[str]] = []
    _install_fake_run(monkeypatch, _Result(stderr=b"connection timed out", returncode=255), calls)

    data = DataLayer(tmp_path)
    lines = data.read_running_job_tail("20260430-110509", "run-6", host="v1")
    payload = data.read_running_job_tail_payload("20260430-110509", "run-6", host="v1")

    assert lines == []
    assert payload == {"source": None, "lines": [], "warning": "connection timed out"}
    captured = capsys.readouterr()
    assert "[WARNING] read_running_job_tail 20260430-110509/run-6" in captured.err
    assert "connection timed out" in captured.err


def test_read_running_tail_timeout_returns_empty_with_warning(tmp_path, monkeypatch, capsys):
    def _timeout(argv, *, capture_output, timeout, check):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(data_mod, "_subprocess_run", _timeout)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-7", host="v1")

    assert lines == []
    captured = capsys.readouterr()
    assert "TimeoutExpired" in captured.err


def test_read_running_tail_handles_binary_garbage_gracefully(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    payload = b"__odin_tail_source__:training.stdout.log\nok\nbad:\xff\n"
    _install_fake_run(monkeypatch, _Result(stdout=payload), calls)

    lines = DataLayer(tmp_path).read_running_job_tail("20260430-110509", "run-8", host="v1")

    assert lines[0] == "ok"
    assert lines[1].startswith("bad:")


def test_lookup_fleet_host_config_reads_snapshot(tmp_path):
    dispatch_dir = tmp_path / "20260430-110509"
    dispatch_dir.mkdir()
    (dispatch_dir / "fleet.yaml.snapshot").write_text(
        json.dumps(
            {
                "fleet_name": "test",
                "hosts": [
                    {
                        "host": "v1",
                        "ssh_user": "odin",
                        "ssh_key": "/keys/id_ed25519",
                        "container_name": "custom-container",
                    }
                ],
            }
        )
    )

    config = DataLayer(tmp_path).lookup_fleet_host_config("20260430-110509", "v1")

    assert config == {
        "host": "v1",
        "ssh_user": "odin",
        "ssh_key": "/keys/id_ed25519",
        "container_name": "custom-container",
    }
