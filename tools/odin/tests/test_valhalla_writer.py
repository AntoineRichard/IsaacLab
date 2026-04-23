# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla.writer — atomic write of aggregate.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.odin.valhalla.writer import write_aggregate


def test_write_creates_file_with_expected_content(tmp_path: Path):
    agg = {"schema_version": "1.0", "dispatch_id": "20260423-100000", "rows": []}
    path = write_aggregate(tmp_path, agg)
    assert path == tmp_path / "aggregate.json"
    assert path.exists()
    with path.open("r") as fh:
        loaded = json.load(fh)
    assert loaded == agg


def test_write_overwrites_by_default(tmp_path: Path):
    (tmp_path / "aggregate.json").write_text('{"stale": true}')
    agg = {"schema_version": "1.0", "rows": [], "new": True}
    write_aggregate(tmp_path, agg)
    with (tmp_path / "aggregate.json").open("r") as fh:
        loaded = json.load(fh)
    assert loaded == agg


def test_write_no_overwrite_raises_on_existing(tmp_path: Path):
    (tmp_path / "aggregate.json").write_text('{"existing": true}')
    agg = {"schema_version": "1.0", "rows": []}
    with pytest.raises(FileExistsError):
        write_aggregate(tmp_path, agg, overwrite=False)


def test_write_cleans_up_temp_on_dump_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Sabotage json.dump so the writer has to clean up the tempfile.
    import tools.odin.valhalla.writer as mod

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated dump failure")

    monkeypatch.setattr(mod.json, "dump", _boom)
    with pytest.raises(RuntimeError, match="simulated"):
        write_aggregate(tmp_path, {"x": 1})

    leftover = [p for p in tmp_path.iterdir() if p.name.startswith(".aggregate_")]
    assert leftover == [], f"tempfile leaked: {leftover}"
    assert not (tmp_path / "aggregate.json").exists()
