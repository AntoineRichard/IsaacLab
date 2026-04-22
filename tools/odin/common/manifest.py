# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write Odin ``manifest.json`` bundles.

Manifests are thin navigational indexes over a bundle directory — see the
Odin T1 spec for the full schema.
"""

from __future__ import annotations

import os
import socket
import subprocess
from datetime import datetime, timezone

from isaaclab.test.benchmark.standard_schema import (
    Manifest,
    ManifestConfig,
    ManifestMachine,
    ManifestPhase,
    write_bundle_file,
)


def _get_git_info(repo_root: str) -> tuple[str | None, str | None]:
    """Return ``(commit, branch)`` for ``repo_root`` or ``(None, None)`` if unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit, branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None


def write_manifest(
    bundle_dir: str,
    run_id: str,
    framework: str,
    backend: str,
    task: str,
    seed: int,
    num_envs: int,
    max_iterations: int,
    run_start_dt: datetime,
    run_end_dt: datetime,
    startup_phase: ManifestPhase,
    training_phase: ManifestPhase,
    repo_root: str,
) -> str:
    """Write ``manifest.json`` to ``<bundle_dir>/manifest.json`` and return the path."""
    git_commit, git_branch = _get_git_info(repo_root)
    artifacts = sorted(os.listdir(bundle_dir)) if os.path.isdir(bundle_dir) else []
    manifest = Manifest(
        run_id=run_id,
        run_start_time_utc=run_start_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        run_end_time_utc=run_end_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        run_duration_s=(run_end_dt - run_start_dt).total_seconds(),
        config=ManifestConfig(
            framework=framework,
            backend=backend,
            task=task,
            seed=seed,
            num_envs=num_envs,
            max_iterations=max_iterations,
        ),
        machine=ManifestMachine(hostname=socket.gethostname(), git_commit=git_commit, git_branch=git_branch),
        phases={"startup": startup_phase, "training": training_phase},
        artifacts=artifacts,
    )
    path = os.path.join(bundle_dir, "manifest.json")
    write_bundle_file(manifest, path)
    return path
