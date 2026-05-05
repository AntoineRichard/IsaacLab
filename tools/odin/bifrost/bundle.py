# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bundle pull-back for Bifrost: ``osmo dataset download`` + manifest validation.

The OSMO task writes ``{{output}}/<run_id>/manifest.json`` (and friends);
the dataset uploaded to OSMO therefore contains ``<run_id>/manifest.json``
at the top level. Downloading the dataset into ``<dispatch_dir>/`` lands
the bundle at ``<dispatch_dir>/<run_id>/manifest.json`` — exactly Odin's
canonical layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

__all__ = ["BundleResult", "download_and_validate_bundle"]


class _DownloaderProto(Protocol):
    def dataset_download(self, name: str, dest_dir: Path) -> None: ...


@dataclass(frozen=True)
class BundleResult:
    """Result of a bundle download + validation pass."""

    bundle_dir: Path
    is_valid: bool


def download_and_validate_bundle(
    *,
    client: _DownloaderProto,
    dataset_name: str,
    dispatch_dir: Path,
    run_id: str,
    validator: Callable[[Path], bool],
) -> BundleResult:
    """Download a dataset into ``<dispatch_dir>/<run_id>/`` and validate the manifest.

    Idempotent: if ``<dispatch_dir>/<run_id>/manifest.json`` already exists
    AND the validator accepts it, the download is skipped.

    Args:
        client: An object with a ``dataset_download(name, dest)`` method.
        dataset_name: OSMO dataset name to download.
        dispatch_dir: Local directory containing the dispatch (e.g.
            ``odin_runs/<dispatch_id>``).
        run_id: Odin run_id; the bundle will land at ``dispatch_dir / run_id``.
        validator: A callable taking the bundle directory and returning
            ``True`` iff the manifest passes validation.

    Returns:
        :class:`BundleResult` with the bundle directory and validation outcome.
    """
    bundle_dir = dispatch_dir / run_id
    manifest = bundle_dir / "manifest.json"
    if manifest.exists() and validator(bundle_dir):
        return BundleResult(bundle_dir=bundle_dir, is_valid=True)
    client.dataset_download(dataset_name, dispatch_dir)
    is_valid = manifest.exists() and validator(bundle_dir)
    return BundleResult(bundle_dir=bundle_dir, is_valid=is_valid)
