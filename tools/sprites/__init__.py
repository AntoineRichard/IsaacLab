# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Procedural sprites for loading screen greetings.

Each module exposes ``frames(cols, rows)`` returning rendered terminal frames at exactly that
size. Drawing onto the target grid rather than scaling to it is deliberate: pixel art does not
survive resampling, so a sprite is authored once per size it ships at.

``tools/gif2anim.py --from-module`` will only import from this package.
"""
