# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The publish/fetch seam and bundle validation."""

import json
from pathlib import Path

import pytest

from tools.odin.results import dispatch_output_uri, fetch_results, read_bundle, results_uri_for, validate_bundle


def _bundle(directory: Path, *, schema_version: str = "1.2", task: str = "Isaac-Ant") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"benchmark_training_{task}_2026-07-28_12-00-00.json"
    path.write_text(json.dumps({"schema_version": schema_version, "run": {"task": task}}))
    return path


def test_uri_is_namespaced_by_dispatch_and_row() -> None:
    uri = results_uri_for("s3://isaac-odin/results", "20260728-120000", "rsl_rl_physx_Ant_seed42")
    assert uri == "s3://isaac-odin/results/20260728-120000/rsl_rl_physx_Ant_seed42"


def test_trailing_slash_in_base_uri_does_not_double_up() -> None:
    uri = results_uri_for("s3://isaac-odin/results/", "20260728-120000", "row")
    assert "//20260728" not in uri.removeprefix("s3://")


def test_dispatch_output_uri_is_the_shared_prefix() -> None:
    uri = dispatch_output_uri("swift://h/AUTH_x/results", "20260728-120000")
    assert uri == "swift://h/AUTH_x/results/20260728-120000/"


def test_dispatch_output_uri_prefixes_every_row_uri() -> None:
    # OSMO uploads {{output}} into the dispatch prefix, and each task writes
    # into {{output}}/<row_key>/, so the fetch URI must sit under it.
    base, dispatch = "swift://h/AUTH_x/results", "20260728-120000"
    assert results_uri_for(base, dispatch, "row").startswith(dispatch_output_uri(base, dispatch))


def test_valid_bundle_passes(tmp_path: Path) -> None:
    _bundle(tmp_path / "row")
    assert validate_bundle(tmp_path / "row") is True


def test_read_bundle_returns_the_parsed_payload(tmp_path: Path) -> None:
    _bundle(tmp_path / "row")
    bundle = read_bundle(tmp_path / "row")
    assert bundle is not None
    assert bundle["run"]["task"] == "Isaac-Ant"


def test_missing_directory_fails(tmp_path: Path) -> None:
    assert validate_bundle(tmp_path / "absent") is False


def test_directory_without_a_bundle_fails(tmp_path: Path) -> None:
    (tmp_path / "row").mkdir()
    (tmp_path / "row" / "training.stdout.log").write_text("noise")
    assert validate_bundle(tmp_path / "row") is False


def test_malformed_json_fails(tmp_path: Path) -> None:
    (tmp_path / "row").mkdir()
    (tmp_path / "row" / "benchmark_training_x_2026-07-28_12-00-00.json").write_text("{not json")
    assert validate_bundle(tmp_path / "row") is False


def test_bundle_without_schema_version_fails(tmp_path: Path) -> None:
    (tmp_path / "row").mkdir()
    (tmp_path / "row" / "benchmark_training_x_2026-07-28_12-00-00.json").write_text('{"run": {}}')
    assert validate_bundle(tmp_path / "row") is False


def test_one_good_bundle_among_broken_ones_passes(tmp_path: Path) -> None:
    row = tmp_path / "row"
    row.mkdir()
    (row / "benchmark_training_a_2026-07-28_11-00-00.json").write_text("{broken")
    _bundle(row)
    assert validate_bundle(row) is True


def test_fetch_downloads_into_the_row_directory(tmp_path: Path) -> None:
    calls: list[tuple[str, Path]] = []

    class _Client:
        def data_download(self, remote_uri: str, dest_dir: Path) -> None:
            calls.append((remote_uri, dest_dir))
            _bundle(dest_dir)

    dest = fetch_results(
        client=_Client(),
        base_uri="s3://b/results",
        dispatch_id="20260728-120000",
        row_key="row",
        dest_dir=tmp_path,
    )

    assert dest == tmp_path / "row"
    assert calls[0][0] == "s3://b/results/20260728-120000/row"
    assert validate_bundle(dest) is True


def test_fetch_is_idempotent_when_a_valid_bundle_exists(tmp_path: Path) -> None:
    _bundle(tmp_path / "row")

    class _Client:
        def data_download(self, remote_uri: str, dest_dir: Path) -> None:
            raise AssertionError("should not re-download a valid bundle")

    assert fetch_results(client=_Client(), base_uri="s3://b", dispatch_id="d", row_key="row", dest_dir=tmp_path) == (
        tmp_path / "row"
    )


def test_fetch_returns_the_row_dir_even_when_the_bundle_is_bad(tmp_path: Path) -> None:
    # The caller reclassifies this as malformed_bundle; fetch itself must not
    # raise, or one bad row would abort the whole dispatch.
    class _Client:
        def data_download(self, remote_uri: str, dest_dir: Path) -> None:
            dest_dir.mkdir(parents=True, exist_ok=True)

    dest = fetch_results(client=_Client(), base_uri="s3://b", dispatch_id="d", row_key="row", dest_dir=tmp_path)
    assert dest == tmp_path / "row"
    assert validate_bundle(dest) is False


def test_the_dataset_is_pulled_once_per_dispatch(tmp_path, monkeypatch) -> None:
    # DSS has no per-row download: --filter matches nothing and --snapshot-name
    # returns the whole dataset. Pulling per row would re-download everything
    # once per row, so the pull is cached by dispatch directory.
    import subprocess

    from tools.odin import results

    results._DATASET_PULLED.clear()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("ODIN_NVDATASET", "/usr/bin/true")
    for row in ("row_a", "row_b", "row_c"):
        results.fetch_results(
            client=None, base_uri="swift://unused", dispatch_id="d",
            row_key=row, dest_dir=tmp_path, dataset="odin-results",
        )

    assert len(calls) == 1
    assert calls[0][1:3] == ["download", "odin-results"]


def test_the_swift_download_is_untouched_without_a_dataset(tmp_path) -> None:
    seen = []

    class _Client:
        def data_download(self, uri, dest):
            seen.append(uri)

    from tools.odin.results import fetch_results

    fetch_results(client=_Client(), base_uri="swift://b/odin", dispatch_id="d",
                  row_key="row_a", dest_dir=tmp_path)

    assert seen == ["swift://b/odin/d/row_a"]


def test_a_missing_cli_is_reported_not_crashed(tmp_path, monkeypatch) -> None:
    # The CLI is on PATH inside the image but not necessarily on the host, where
    # fetching runs. A bare FileNotFoundError killed a live 801-row dispatch.
    import shutil

    from tools.odin.results import ResultsError, _DATASET_PULLED, fetch_results

    _DATASET_PULLED.clear()
    monkeypatch.delenv("ODIN_NVDATASET", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _name: None)

    with pytest.raises(ResultsError, match="ODIN_NVDATASET"):
        fetch_results(client=None, base_uri="swift://unused", dispatch_id="d",
                      row_key="row_a", dest_dir=tmp_path, dataset="odin-results")
