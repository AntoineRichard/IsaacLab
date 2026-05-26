# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-shot reconciliation of ``dispatch.json`` from OSMO state.

The Bifrost poller drives a dispatch to completion in-process. If it
dies (Ctrl+C, SIGTERM, OOM, machine reboot) or just falls behind,
``dispatch.json`` is left out of sync with what OSMO sees. This script
replays the same per-tick logic the poller uses, but as a single pass:
walks every workflow id, applies status diffs through ``transition_to``,
downloads bundles for newly-completed tasks, then writes
``dispatch.json``.

Idempotent: re-running with no remote changes is a no-op. Safe to run
while the live poller is also active (last writer wins on
``dispatch.json``; both write atomically).

Usage::

    # Sync the most recent dispatch under tools/odin/runs/
    python3 tools/odin/scripts/osmo_sync.py

    # Specify dispatch id
    python3 tools/odin/scripts/osmo_sync.py --dispatch-id 20260522-093457

    # Different runs root
    python3 tools/odin/scripts/osmo_sync.py --runs-root /path/to/odin_runs

    # Skip bundle downloads (just update statuses)
    python3 tools/odin/scripts/osmo_sync.py --no-download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make tools/ importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.odin.asgard.jobs import JobEntry  # noqa: E402
from tools.odin.asgard.state import read_dispatch_state, write_dispatch_state  # noqa: E402
from tools.odin.bifrost.bundle import download_and_validate_bundle  # noqa: E402
from tools.odin.bifrost.cli import _manifest_validator  # noqa: E402
from tools.odin.bifrost.client import OsmoClient  # noqa: E402
from tools.odin.bifrost.config import load_bifrost_config  # noqa: E402
from tools.odin.bifrost.poller import sync_once  # noqa: E402


_DEFAULT_RUNS_ROOT = Path("odin_runs")
_DEFAULT_CONFIG = Path("tools/odin/config/bifrost-osmo.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dispatch-id",
        default="LATEST",
        help=(
            "Dispatch id (matches a directory under ``--runs-root``). "
            "Default: ``LATEST`` (most recently modified subdirectory)."
        ),
    )
    parser.add_argument("--runs-root", type=Path, default=_DEFAULT_RUNS_ROOT, help="Root containing dispatch directories.")
    parser.add_argument("--osmo-config", type=Path, default=_DEFAULT_CONFIG, help="Path to bifrost-osmo.yaml.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip bundle downloads for newly-completed tasks. Status fields are still updated.",
    )
    parser.add_argument(
        "--rehydrate",
        action="store_true",
        help=(
            "After the sync pass, re-fire the bundle-download hook for every"
            " task already marked ``completed`` whose local manifest.json"
            " is missing. Use this to backfill bundles after an earlier"
            " ``--no-download`` sync, or when the live poller died before"
            " its on-completed hook ran."
        ),
    )
    parser.add_argument(
        "--keep-osmo-datasets",
        action="store_true",
        help="Do not delete OSMO datasets after successful bundle download.",
    )
    return parser.parse_args()


def _resolve_dispatch_dir(runs_root: Path, dispatch_id: str) -> Path:
    if dispatch_id == "LATEST":
        candidates = [p for p in runs_root.iterdir() if p.is_dir() and (p / "dispatch.json").exists()]
        if not candidates:
            print(f"error: no dispatches under {runs_root}", file=sys.stderr)
            sys.exit(1)
        return max(candidates, key=lambda p: p.stat().st_mtime)
    p = runs_root / dispatch_id
    if not (p / "dispatch.json").exists():
        print(f"error: {p / 'dispatch.json'} not found", file=sys.stderr)
        sys.exit(1)
    return p


def main() -> int:
    args = _parse_args()
    dispatch_dir = _resolve_dispatch_dir(args.runs_root, args.dispatch_id)
    print(f"[osmo-sync] dispatch: {dispatch_dir.name}")

    state = read_dispatch_state(dispatch_dir)
    if not state.osmo_workflow_ids and state.osmo_workflow_id is None:
        print(f"error: dispatch {state.dispatch_id} has no OSMO workflow ids", file=sys.stderr)
        return 1

    cfg = load_bifrost_config(args.osmo_config)
    client = OsmoClient(profile=cfg.osmo_profile)

    validator = _manifest_validator()

    if args.no_download:
        def on_completed(job: JobEntry) -> None:
            return
    else:
        def on_completed(job: JobEntry) -> None:
            dataset_name = f"{cfg.bundle_dataset_prefix}-{state.dispatch_id}-{job.run_id}"
            try:
                result = download_and_validate_bundle(
                    client=client,
                    dataset_name=dataset_name,
                    dispatch_dir=dispatch_dir,
                    run_id=job.run_id,
                    validator=validator,
                )
            except Exception as exc:
                print(f"[osmo-sync] bundle download for {job.run_id} skipped: {exc}", file=sys.stderr)
                return
            if result.is_valid and not args.keep_osmo_datasets:
                try:
                    client.dataset_delete(dataset_name)
                except Exception as exc:
                    print(f"[osmo-sync] dataset cleanup for {dataset_name} skipped: {exc}", file=sys.stderr)

    # Snapshot pre-sync counts for a useful diff report.
    pre = _count_by_status(state)
    sync_once(client=client, state=state, dispatch_dir=dispatch_dir, on_task_completed=on_completed)
    post = _count_by_status(state)
    write_dispatch_state(dispatch_dir, state)

    print(f"[osmo-sync] before: {pre}")
    print(f"[osmo-sync] after:  {post}")
    delta = {k: post.get(k, 0) - pre.get(k, 0) for k in set(pre) | set(post)}
    delta = {k: v for k, v in delta.items() if v}
    if delta:
        print(f"[osmo-sync] delta:  {delta}")
    else:
        print("[osmo-sync] no changes")

    if args.rehydrate and not args.no_download:
        backfilled = 0
        missing = [j for j in state.jobs if j.status == "completed" and not (dispatch_dir / j.run_id / "manifest.json").exists()]
        print(f"[osmo-sync] rehydrate: {len(missing)} completed jobs missing local bundle")
        for job in missing:
            on_completed(job)
            backfilled += 1
        print(f"[osmo-sync] rehydrate done: {backfilled} download attempts")
    return 0


def _count_by_status(state) -> dict[str, int]:
    out: dict[str, int] = {}
    for j in state.jobs:
        out[j.status] = out.get(j.status, 0) + 1
    return out


if __name__ == "__main__":
    sys.exit(main())
