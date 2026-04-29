# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tab B — Task drill-down — for the Odin dashboard."""

__all__ = ["render", "register"]


def render(dispatch_id: str, tab_id: str):
    """Spec 0 registry hook — return the static layout for this tab."""
    from tools.odin.valhalla.dashboard.tabs.task_drilldown.layout import build_layout

    return build_layout(dispatch_id)


def register(app, data):
    """Spec 0 registry hook — wire Tab B's callbacks at app startup.

    Until T9 lands the callbacks module, this is a no-op so the
    Spec 0 registry's startup walk doesn't crash.
    """
    try:
        from tools.odin.valhalla.dashboard.tabs.task_drilldown.callbacks import (
            register_callbacks,
        )
    except ModuleNotFoundError:
        return
    register_callbacks(app, data)
