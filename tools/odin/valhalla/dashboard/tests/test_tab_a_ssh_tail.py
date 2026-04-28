# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tab_a.ssh_tail.load_ssh_tail."""

from __future__ import annotations

from pathlib import Path

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.ssh_tail import (
    SSH_TAIL_DEFAULT_LINES,
    SSH_TAIL_MAX_BYTES,
    load_ssh_tail,
)


def _write_log(runs_root: Path, dispatch_id: str, run_id: str, content: str) -> Path:
    log_dir = runs_root / dispatch_id / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ssh-tail.log"
    log_path.write_text(content)
    return log_path


def test_load_ssh_tail_full_file_under_threshold(tmp_path):
    _write_log(tmp_path, "d", "r", "\n".join(f"line {i}" for i in range(10)) + "\n")
    lines = load_ssh_tail(tmp_path, "d", "r", lines=SSH_TAIL_DEFAULT_LINES)
    assert lines == [f"line {i}" for i in range(10)]


def test_load_ssh_tail_returns_last_n_lines(tmp_path):
    _write_log(tmp_path, "d", "r", "\n".join(f"line {i}" for i in range(100)) + "\n")
    lines = load_ssh_tail(tmp_path, "d", "r", lines=10)
    assert lines == [f"line {i}" for i in range(90, 100)]


def test_load_ssh_tail_returns_empty_when_file_missing(tmp_path):
    assert load_ssh_tail(tmp_path, "d", "r-missing") == []


def test_load_ssh_tail_truncates_huge_file(tmp_path):
    # Write 200 KB so we exceed the 64 KB cap.
    chunk = "abcdefghij" * 100  # 1000 bytes per line
    payload = "\n".join([f"{i:05d} {chunk}" for i in range(200)]) + "\n"
    _write_log(tmp_path, "d", "r", payload)
    lines = load_ssh_tail(tmp_path, "d", "r", lines=20)
    assert len(lines) == 20
    assert lines[0].startswith("…")
    assert "truncated" in lines[0].lower()


def test_load_ssh_tail_handles_partial_first_line_when_seeking(tmp_path):
    # Construct a file where the truncation point is mid-line; the partial first line should be dropped.
    chunk = "x" * 70_000  # bigger than SSH_TAIL_MAX_BYTES
    payload = chunk + "\nfinal-line\n"
    _write_log(tmp_path, "d", "r", payload)
    lines = load_ssh_tail(tmp_path, "d", "r", lines=5)
    # The truncation marker is line[0]; the partial line is dropped; "final-line" is the only real line.
    assert lines[0].startswith("…")
    assert any("final-line" in s for s in lines)


def test_load_ssh_tail_returns_empty_on_permission_error(tmp_path, monkeypatch):
    log_path = _write_log(tmp_path, "d", "r", "hello\n")

    real_open = open

    def _raise(path, *args, **kwargs):
        if str(path) == str(log_path):
            raise PermissionError("denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _raise)
    assert load_ssh_tail(tmp_path, "d", "r") == []


def test_load_ssh_tail_max_bytes_constant_is_64kb():
    assert SSH_TAIL_MAX_BYTES == 64 * 1024
