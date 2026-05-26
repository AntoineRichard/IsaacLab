# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regenerate ``tools/odin/config/task_enable.txt`` from the env yamls.

The summary file is a human-editable view of the ``keep`` flag in
``physx_envs.yaml`` and ``newton_envs.yaml``. Run this whenever the env
yamls are re-enumerated to pick up newly discovered tasks.

Run from the repo root::

    python3 tools/odin/scripts/regen_task_enable.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TXT_PATH = REPO_ROOT / "tools/odin/config/task_enable.txt"
ENV_YAMLS = {
    "physx": REPO_ROOT / "tools/odin/config/physx_envs.yaml",
    "newton": REPO_ROOT / "tools/odin/config/newton_envs.yaml",
}


def main() -> int:
    matrix: dict[str, dict[str, bool | None]] = defaultdict(lambda: {"physx": None, "newton": None})
    for backend, path in ENV_YAMLS.items():
        data = yaml.safe_load(path.read_text())
        for group_rows in (data.get("groups") or {}).values():
            for row in group_rows or []:
                matrix[row["task_id"]][backend] = bool(row.get("keep"))

    rows = sorted(matrix.items())
    name_w = max(len(t) for t, _ in rows)

    lines = [
        "# Odin task enable/disable matrix.",
        "# Edit the on/off values; '-' means the task does not exist for that backend.",
        "# Run `tools/odin/scripts/apply_task_enable.py` to write changes back to *_envs.yaml.",
        "",
        f"{'task_id'.ljust(name_w)}  physx  newton",
        f"{'-' * name_w}  -----  ------",
    ]
    for tid, backends in rows:
        px = "-" if backends["physx"] is None else ("on" if backends["physx"] else "off")
        nw = "-" if backends["newton"] is None else ("on" if backends["newton"] else "off")
        lines.append(f"{tid.ljust(name_w)}  {px.ljust(5)}  {nw.ljust(6)}")

    TXT_PATH.write_text("\n".join(lines) + "\n")
    kept_px = sum(1 for _, b in rows if b["physx"] is True)
    kept_nw = sum(1 for _, b in rows if b["newton"] is True)
    print(f"wrote {TXT_PATH.relative_to(REPO_ROOT)} — {len(rows)} tasks ({kept_px} physx-on, {kept_nw} newton-on)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
