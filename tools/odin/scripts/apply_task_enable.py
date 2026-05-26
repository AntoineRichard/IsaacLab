# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Propagate ``task_enable.txt`` edits back into ``physx_envs.yaml`` / ``newton_envs.yaml``.

Reads ``tools/odin/config/task_enable.txt`` (the human-edited summary) and
writes ``keep: true``/``false`` for each task in the two env yamls. The
``-`` cell (task does not exist for that backend) is left alone.

Run from the repo root::

    python3 tools/odin/scripts/apply_task_enable.py
    python3 tools/odin/scripts/apply_task_enable.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TXT_PATH = REPO_ROOT / "tools/odin/config/task_enable.txt"
ENV_YAMLS = {
    "physx": REPO_ROOT / "tools/odin/config/physx_envs.yaml",
    "newton": REPO_ROOT / "tools/odin/config/newton_envs.yaml",
}


def _parse_txt(path: Path) -> dict[str, dict[str, bool | None]]:
    """Return ``{task_id: {"physx": bool|None, "newton": bool|None}}``."""
    out: dict[str, dict[str, bool | None]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("task_id") or line.startswith("---"):
            continue
        parts = line.split()
        if len(parts) < 3:
            print(f"skipping malformed line: {raw!r}", file=sys.stderr)
            continue
        tid, px, nw = parts[0], parts[1], parts[2]
        out[tid] = {
            "physx": None if px == "-" else px == "on",
            "newton": None if nw == "-" else nw == "on",
        }
    return out


def _apply_to_yaml(yaml_path: Path, backend: str, desired: dict[str, dict[str, bool | None]]) -> tuple[int, int]:
    """Write ``keep`` according to ``desired``. Returns ``(changed, total)``."""
    data = yaml.safe_load(yaml_path.read_text())
    changed = 0
    total = 0
    for group_rows in (data.get("groups") or {}).values():
        for row in group_rows or []:
            total += 1
            tid = row["task_id"]
            want = desired.get(tid, {}).get(backend)
            if want is None:
                continue
            if bool(row.get("keep")) != want:
                row["keep"] = want
                changed += 1
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, width=200))
    return changed, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    args = parser.parse_args()

    if not TXT_PATH.exists():
        print(f"error: {TXT_PATH} does not exist. Regenerate it with the helper script.", file=sys.stderr)
        return 1

    desired = _parse_txt(TXT_PATH)
    print(f"parsed {len(desired)} task rows from {TXT_PATH.relative_to(REPO_ROOT)}")

    for backend, yaml_path in ENV_YAMLS.items():
        if args.dry_run:
            data = yaml.safe_load(yaml_path.read_text())
            flips = []
            for group_rows in (data.get("groups") or {}).values():
                for row in group_rows or []:
                    tid = row["task_id"]
                    want = desired.get(tid, {}).get(backend)
                    if want is None:
                        continue
                    if bool(row.get("keep")) != want:
                        flips.append((tid, bool(row.get("keep")), want))
            print(f"[{backend}] would flip {len(flips)} entries:")
            for tid, was, now in flips[:20]:
                print(f"  {tid}: {was} -> {now}")
            if len(flips) > 20:
                print(f"  ... ({len(flips) - 20} more)")
        else:
            changed, total = _apply_to_yaml(yaml_path, backend, desired)
            print(f"[{backend}] flipped {changed}/{total} entries in {yaml_path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
