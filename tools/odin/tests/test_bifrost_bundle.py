# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
from pathlib import Path
from unittest.mock import MagicMock

from tools.odin.bifrost.bundle import download_and_validate_bundle


def _client_writing_manifest(content: dict, run_subdir: str = "rsl-rl_physx_X_seed42"):
    """Make a MagicMock OsmoClient whose dataset_download writes a manifest.json."""
    client = MagicMock()

    def fake_download(name: str, dest_dir: Path) -> None:
        run_dir = Path(dest_dir) / run_subdir
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(content))

    client.dataset_download.side_effect = fake_download
    return client


def test_download_writes_to_expected_path(tmp_path: Path):
    valid = lambda p: True  # noqa: E731
    client = _client_writing_manifest({"schema": "v1"})
    res = download_and_validate_bundle(
        client=client,
        dataset_name="odin-disp1-rsl-rl_physx_X_seed42",
        dispatch_dir=tmp_path,
        run_id="rsl-rl_physx_X_seed42",
        validator=valid,
    )
    expected_dir = tmp_path / "rsl-rl_physx_X_seed42"
    assert res.bundle_dir == expected_dir
    assert (expected_dir / "manifest.json").exists()
    assert res.is_valid


def test_invalid_manifest_marks_malformed(tmp_path: Path):
    client = _client_writing_manifest({"missing": "fields"})
    res = download_and_validate_bundle(
        client=client,
        dataset_name="odin-disp1-rsl-rl_physx_X_seed42",
        dispatch_dir=tmp_path,
        run_id="rsl-rl_physx_X_seed42",
        validator=lambda p: False,
    )
    assert not res.is_valid


def test_idempotent_skips_redownload_when_manifest_present(tmp_path: Path):
    client = _client_writing_manifest({"schema": "v1"})
    # Pre-populate the bundle dir with a valid manifest.
    bundle_dir = tmp_path / "rsl-rl_physx_X_seed42"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(json.dumps({"schema": "v1"}))
    res = download_and_validate_bundle(
        client=client,
        dataset_name="odin-disp1-rsl-rl_physx_X_seed42",
        dispatch_dir=tmp_path,
        run_id="rsl-rl_physx_X_seed42",
        validator=lambda p: True,
    )
    client.dataset_download.assert_not_called()
    assert res.is_valid
