# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish and fetch benchmark bundles.

OSMO datasets were retired — ``outputs: - dataset:`` is rejected server-side with
*"Bucket isaac mode is read-only"*. Publishing is therefore declarative: each
task carries an ``outputs: - url: <dispatch prefix>`` block and OSMO uploads the
task's output directory itself.

That is deliberately not a command spliced into the entry script. OSMO's
``uploadOutputs()`` runs unconditionally after exec, including when the
benchmark fails, so a crashed run still returns whatever it produced and the
benchmark's exit code cannot be masked by a publishing step. It also keeps
storage credentials out of the container entirely.

Fetching stays client-side via ``osmo data download``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

__all__ = ["dispatch_output_uri", "fetch_results", "read_bundle", "results_uri_for", "validate_bundle"]

# Upstream's SchemaBundleFile writes "<output_prefix>_<timestamp>.json" into
# --output_path, where output_prefix is "benchmark_<workflow>_<task>".
_BUNDLE_GLOB = "benchmark_*.json"


class ResultsError(RuntimeError):
    """Raised when results cannot be retrieved."""


class _DownloaderProto(Protocol):
    def data_download(self, remote_uri: str, dest_dir: Path) -> None: ...


def results_uri_for(base_uri: str, dispatch_id: str, row_key: str) -> str:
    """Return the storage URI for one row's results.

    Args:
        base_uri: Bucket URI prefix from ``odin.yaml``.
        dispatch_id: Dispatch identifier.
        row_key: Row identity.

    Returns:
        ``<base_uri>/<dispatch_id>/<row_key>``, without a doubled separator.
    """
    return f"{base_uri.rstrip('/')}/{dispatch_id}/{row_key}"


def dispatch_output_uri(base_uri: str, dispatch_id: str) -> str:
    """Return the storage prefix an OSMO ``outputs:`` block uploads into.

    Every task in a dispatch shares this prefix. Each task writes into
    ``{{output}}/<row_key>/``, so the uploaded tree lands at
    ``<base_uri>/<dispatch_id>/<row_key>/`` and rows never collide.

    Args:
        base_uri: Bucket URI prefix from ``odin.yaml``.
        dispatch_id: Dispatch identifier.

    Returns:
        ``<base_uri>/<dispatch_id>/``, with a trailing slash.
    """
    return f"{base_uri.rstrip('/')}/{dispatch_id}/"


def read_bundle(bundle_dir: Path) -> dict[str, Any] | None:
    """Return the parsed schema bundle in *bundle_dir*, or ``None``.

    Args:
        bundle_dir: Directory holding one row's results.

    Returns:
        The first readable bundle mapping that declares a ``schema_version``, or
        ``None`` when the directory holds none.
    """
    if not bundle_dir.is_dir():
        return None
    for candidate in sorted(bundle_dir.glob(_BUNDLE_GLOB)):
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("schema_version"):
            return payload
    return None


def validate_bundle(bundle_dir: Path) -> bool:
    """Return ``True`` iff *bundle_dir* holds a readable schema bundle.

    A ``COMPLETED`` OSMO task that produced no parseable bundle is classified
    ``malformed_bundle`` by the caller.

    Args:
        bundle_dir: Directory the row's results were fetched into.

    Returns:
        Whether :func:`read_bundle` finds anything.
    """
    return read_bundle(bundle_dir) is not None


# Dispatch directories whose dataset snapshot has already been pulled. DSS has no
# per-row download: ``--filter`` matched nothing in testing and ``--snapshot-name``
# returns the whole dataset, so the pull is once per dispatch and rows are read
# out of the resulting tree.
_DATASET_PULLED: set[Path] = set()


def download_dataset_once(dataset: str, dest_dir: Path, *, run: Any = None) -> None:
    """Pull *dataset* into *dest_dir*, at most once per directory.

    Rows upload under ``<row_key>/...``, so the downloaded tree already has the
    shape :func:`fetch_results` expects and each row is a subdirectory.

    Args:
        dataset: DSS dataset name.
        dest_dir: Dispatch directory to download into.
        run: Subprocess runner, injected for testing.

    Raises:
        ResultsError: If the download fails.
    """
    import subprocess

    if dest_dir in _DATASET_PULLED:
        return
    runner = run or subprocess.run
    dest_dir.mkdir(parents=True, exist_ok=True)
    completed = runner(
        ["nvdataset", "download", dataset, str(dest_dir), "-y"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ResultsError(f"`nvdataset download {dataset}` failed: {(completed.stderr or '').strip()}")
    _DATASET_PULLED.add(dest_dir)


def fetch_results(
    *,
    client: _DownloaderProto,
    base_uri: str,
    dispatch_id: str,
    row_key: str,
    dest_dir: Path,
    dataset: str | None = None,
) -> Path:
    """Download one row's results into ``<dest_dir>/<row_key>``.

    Idempotent: a directory that already holds a valid bundle is not
    re-downloaded.

    Args:
        client: Provides ``data_download(remote_uri, dest_dir)``.
        base_uri: Bucket URI prefix from ``odin.yaml``.
        dispatch_id: Dispatch identifier.
        row_key: Row identity.
        dest_dir: Dispatch directory the row directory is created under.
        dataset: DSS dataset holding the results. When set, the dispatch's
            snapshot is pulled once and rows are read from it; *base_uri* is
            unused, because nothing was written there.

    Returns:
        The row directory, whether or not the bundle within it validates.
    """
    row_dir = dest_dir / row_key
    if validate_bundle(row_dir):
        return row_dir
    if dataset:
        download_dataset_once(dataset, dest_dir)
        return row_dir
    client.data_download(results_uri_for(base_uri, dispatch_id, row_key), row_dir)
    return row_dir
