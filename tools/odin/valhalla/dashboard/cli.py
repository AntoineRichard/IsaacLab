# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""odin-dashboard CLI — spins a local Dash server for an Odin runs_root.

Invoke from the repo root with ``PYTHONPATH=.``:

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/dashboard/cli.py \\
        --runs-root odin_runs

Or, jumping directly to a dispatch::

    PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/dashboard/cli.py \\
        20260427-141302 --runs-root odin_runs
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

__all__ = ["main", "parse_args"]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="odin-dashboard",
        description="Browser-based dashboard over odin_runs/.",
    )
    parser.add_argument(
        "dispatch",
        nargs="?",
        default=None,
        help=(
            "Optional dispatch_id (e.g. 20260427-141302) or 'LATEST'. "
            "If set, the dashboard opens directly on Tab A of that dispatch."
        ),
    )
    parser.add_argument(
        "--dispatch",
        dest="dispatch_flag",
        default=None,
        help="Same as the positional dispatch arg; flag form for clarity.",
    )
    parser.add_argument("--runs-root", type=Path, default=Path("odin_runs"))
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no-browser", action="store_true", default=False)
    ns = parser.parse_args(argv)
    if ns.dispatch_flag and not ns.dispatch:
        ns.dispatch = ns.dispatch_flag
    delattr(ns, "dispatch_flag")
    return ns


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    runs_root = ns.runs_root.resolve()
    if not runs_root.exists():
        print(f"odin-dashboard: --runs-root {runs_root} does not exist", file=sys.stderr)
        return 2
    initial_dispatch: Path | None = None
    if ns.dispatch:
        initial_dispatch = _resolve_dispatch_dir(runs_root, ns.dispatch)
        if initial_dispatch is None:
            print(
                f"odin-dashboard: dispatch {ns.dispatch!r} not found under {runs_root}",
                file=sys.stderr,
            )
            return 2
    # Imported here so a missing `dash` install yields a clean exit-3 above the framework.
    try:
        from tools.odin.valhalla.dashboard.app import create_app
    except ModuleNotFoundError as exc:
        if "dash" in str(exc):
            print(
                "odin-dashboard: dash not installed; run `pip install dash plotly pandas`",
                file=sys.stderr,
            )
            return 3
        raise
    app = create_app(runs_root, initial_dispatch=initial_dispatch)
    url = f"http://{ns.host}:{ns.port}"
    print(f"odin-dashboard: serving {url}/ runs_root={runs_root}")
    if not ns.no_browser:
        webbrowser.open(url)
    try:
        # dash 4.x renamed `run_server` to `run` and raises
        # ObsoleteAttributeException (not AttributeError) on the old name,
        # so a plain getattr() with a default still raises.  Catch broadly
        # to support both 2.x and 4.x.
        try:
            run = app.run_server  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            run = app.run
        run(host=ns.host, port=ns.port, debug=ns.debug)
    except OSError as exc:
        if "in use" in str(exc).lower() or "Address already in use" in str(exc):
            print(
                f"odin-dashboard: port {ns.port} is in use; try --port",
                file=sys.stderr,
            )
            return 4
        raise
    return 0


def _resolve_dispatch_dir(runs_root: Path, spec: str) -> Path | None:
    """Resolve `spec` (a dispatch_id or 'LATEST') to an absolute dispatch path.

    Returns None if not found. Mirrors the dispatcher's resolution rules.
    """
    if spec == "LATEST":
        candidates = sorted(
            (p for p in runs_root.iterdir() if p.is_dir() and (p / "dispatch.json").exists()),
            key=lambda p: p.name,
            reverse=True,
        )
        return candidates[0] if candidates else None
    candidate = runs_root / spec
    if (candidate / "dispatch.json").exists():
        return candidate
    return None


if __name__ == "__main__":
    sys.exit(main())
