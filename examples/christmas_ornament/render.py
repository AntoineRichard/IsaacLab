# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the ornament demo headless to MP4 (OpenGL through EGL, frames piped to ffmpeg).

Run with ``PYGLET_HEADLESS=1`` on a machine without a display::

    PYGLET_HEADLESS=1 python -m examples.christmas_ornament.render --cam close --out /tmp/close.mp4
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import newton
import warp as wp

from examples.christmas_ornament.run import FPS, OrnamentScene
from examples.christmas_ornament.run import parser as run_parser
from examples.christmas_ornament.tree import BRANCH_YOUNG_MODULUS

# (position, look-at) per camera; the robot stands at the origin facing +x, the hang point is
# near (0.40, -0.10, 1.12) and the tree trunk ~0.4 m beyond it.
CAMERAS = {
    "wide": ((-0.9, -2.3, 1.35), (0.55, 0.0, 0.85)),
    "close": ((0.42, -0.62, 1.17), (0.40, -0.10, 1.11)),
    "hand": ((-0.30, -0.42, 1.45), (0.40, -0.10, 1.10)),
}


def _yaw_pitch(position, target) -> tuple[float, float]:
    dx, dy, dz = (t - p for t, p in zip(target, position, strict=True))
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return yaw, pitch


def _ffmpeg(out: Path, width: int, height: int) -> subprocess.Popen:
    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
        + ["-s", f"{width}x{height}", "-r", str(FPS), "-i", "-"]
        + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow", str(out)],
        stdin=subprocess.PIPE,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cam", choices=tuple(CAMERAS), default="wide")
    p.add_argument("--mode", choices=("probe", "hang"), default="hang")
    p.add_argument("--out", type=Path, default=Path("/tmp/christmas_ornament.mp4"))
    p.add_argument("--frames", type=int, default=0, help="0 renders the whole sequence.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    cli = p.parse_args()

    args = run_parser().parse_args(["--mode", cli.mode, "--viewer", "null"])
    args.branch_modulus = BRANCH_YOUNG_MODULUS
    scene = OrnamentScene(args)
    viewer = newton.viewer.ViewerGL(width=cli.width, height=cli.height, headless=True)
    viewer.set_model(scene.model)
    position, target = CAMERAS[cli.cam]
    yaw, pitch = _yaw_pitch(position, target)
    viewer.set_camera(pos=wp.vec3(*position), pitch=pitch, yaw=yaw)

    frames = cli.frames or int(round(scene.total_time * FPS))
    ffmpeg = _ffmpeg(cli.out, cli.width, cli.height)
    buffer = None
    while scene.frame < frames:
        scene.step()
        scene.render(viewer)
        buffer = viewer.get_frame(output=buffer)
        ffmpeg.stdin.write(buffer.numpy().tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    viewer.close()
    print(f"wrote {cli.out} ({frames} frames)")


if __name__ == "__main__":
    main()
