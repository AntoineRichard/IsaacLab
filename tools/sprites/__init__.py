# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural sprites for loading screen greetings.

Each module exposes ``frames(cols, rows)`` returning rendered terminal frames at exactly that
size. Drawing onto the target grid rather than scaling to it is deliberate: pixel art does not
survive resampling, so a sprite is authored once per size it ships at.

``tools/gif2anim.py --from-module`` will only import from this package. :mod:`.probe` is the
worked example; :mod:`tools.anim_encode` provides the canvas to draw on.

The sprites that produced the shipped greetings are not kept here -- the art they generate is
what ships, and the generators would otherwise be a second copy of it to maintain. They live
on the ``antoiner/loading-screen-sprites`` branch if a greeting needs redrawing.
"""
