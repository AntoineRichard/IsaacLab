# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab A — Dispatch & Fleet — for the Odin dashboard."""

from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.layout import build_layout

__all__ = ["render", "register"]


def render(dispatch_id: str, tab_id: str):
    """Spec 0 registry hook — return the static layout for this tab."""
    return build_layout(dispatch_id)


def register(app, data):
    """Spec 0 registry hook — wire Tab A's callbacks at app startup.

    Lazy-imported to avoid pulling Dash callbacks into the module graph at
    test collection time when only `render` is needed.
    """
    from tools.odin.valhalla.dashboard.tabs.dispatch_fleet.callbacks import register_callbacks

    register_callbacks(app, data)
