# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Look at loading screen greetings without starting a simulator.

    uv run python tools/anim_view.py                    # list what is shipped
    uv run python tools/anim_view.py walker-wide        # play it
    uv run python tools/anim_view.py walker-wide --frame 6
    uv run python tools/anim_view.py out/new.anim       # a file that is not installed yet
    uv run python tools/anim_view.py --all              # one frame of each, side by side
    uv run python tools/anim_view.py walker-wide --beside   # as the loading screen shows it

Playback runs on the alternate screen buffer and draws every line at an absolute position.
Both matter: the terminal's scrollback is left untouched, and a resize mid-playback needs no
correction because each frame is drawn fresh at whatever size the terminal now is. Relative
cursor movement -- draw, move back up, redraw -- looks equivalent until the window changes
size, at which point the "move back up" is measured against a geometry that no longer exists.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), str(_ROOT / "source" / "isaaclab")]

from isaaclab.app.anims import Animation, available, load, unpack  # noqa: E402

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Text with its colour escapes removed, for measuring."""
    return _ANSI.sub("", text)


def resolve(target: str) -> Animation:
    """Load a greeting by shipped name or by path to a container file."""
    path = Path(target)
    if path.is_file():
        return unpack(path.read_bytes())
    return load(target)


def summarise() -> int:
    """List the shipped greetings with their size and frame count."""
    names = available()
    if not names:
        print("no greetings are shipped yet")
        return 1
    print(f"{'name':<16} {'size':>8} {'frames':>7} {'bytes':>7}")
    for name in names:
        animation = load(name)
        path = _ROOT / "source/isaaclab/isaaclab/app/anims" / f"{name}.anim"
        size = path.stat().st_size if path.is_file() else 0
        kind = "still" if len(animation.frames) == 1 else f"{len(animation.frames)}"
        print(f"{name:<16} {animation.cols:>4}x{animation.rows:<3} {kind:>7} {size:>7}")
    return 0


def contact_sheet() -> int:
    """Print one frame of every greeting, labelled, so they can be compared at a glance."""
    for name in available():
        animation = load(name)
        print(f"\x1b[1m{name}\x1b[0m  {animation.cols}x{animation.rows}, {len(animation.frames)} frame(s)")
        print(animation.frames[0])
        print()
    return 0


def beside_summary(animation: Animation, frame: str) -> str:
    """Lay a frame out beside a mock run summary, as the loading screen composes it.

    A greeting can look fine alone and wrong in place -- too tall for the box, or so narrow
    that the gap swallows it -- so this is the view that actually predicts the result.
    """
    fields = [
        ("Task", "Isaac-Velocity-Flat-Anymal-D-v0"),
        ("Backend", "newton"),
        ("Device", "cuda:0"),
        ("Num envs", "4096"),
        ("Library", "rsl_rl"),
    ]
    width = 50  # the merged loading screen locks its summary box to this
    title = "─ Isaac Lab "
    box = ["╭" + title + "─" * (width - 2 - len(title)) + "╮"]
    box += [f"│ {label:<14}{value}".ljust(width - 1) + "│" for label, value in fields]
    box.append("╰" + "─" * (width - 2) + "╯")

    art = frame.splitlines()
    out = []
    for row in range(max(len(box), len(art))):
        left = box[row] if row < len(box) else " " * width
        right = art[row] if row < len(art) else ""
        out.append(f"{left}      {right}")
    return "\n".join(out)


def play(animation: Animation, fps: float, beside: bool) -> int:
    """Loop a greeting on the alternate screen until interrupted."""
    write = sys.stdout.write
    write("\x1b[?1049h\x1b[?25l")
    try:
        index = 0
        while True:
            frame = animation.frames[index % len(animation.frames)]
            body = beside_summary(animation, frame) if beside else frame
            size = shutil.get_terminal_size((80, 24))
            lines = body.splitlines()
            shown = index % len(animation.frames) + 1
            header = (
                f"{animation.name}  {animation.cols}x{animation.rows}  "
                f"frame {shown}/{len(animation.frames)}  (Ctrl-C to stop)"
            )
            out = ["\x1b[H\x1b[J", f"\x1b[1;1H\x1b[2K{header[: size.columns]}"]
            for row, line in enumerate(lines[: size.lines - 2], start=3):
                out.append(f"\x1b[{row};1H\x1b[2K{line}")
            write("".join(out))
            sys.stdout.flush()
            time.sleep(1 / fps)
            index += 1
    except KeyboardInterrupt:
        pass
    finally:
        write("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and show the requested greeting."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", help="a shipped greeting name, or a path to a .anim file")
    parser.add_argument("--frame", type=int, help="print one frame and exit, 0-based")
    parser.add_argument("--all", action="store_true", help="print one frame of every shipped greeting")
    parser.add_argument("--beside", action="store_true", help="lay it out beside a mock run summary")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.all:
        return contact_sheet()
    if not args.target:
        return summarise()

    try:
        animation = resolve(args.target)
    except (KeyError, OSError) as error:
        print(error, file=sys.stderr)
        return 1

    if args.frame is not None:
        frame = animation.frames[args.frame % len(animation.frames)]
        print(beside_summary(animation, frame) if args.beside else frame)
        return 0
    if len(animation.frames) == 1:
        print(beside_summary(animation, animation.frames[0]) if args.beside else animation.frames[0])
        return 0
    return play(animation, args.fps, args.beside)


if __name__ == "__main__":
    raise SystemExit(main())
