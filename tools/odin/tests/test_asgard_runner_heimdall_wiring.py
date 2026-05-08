# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DispatchOptions Heimdall fields + CLI flag wiring."""

from __future__ import annotations

from tools.odin.asgard.runner import DispatchOptions


def test_dispatch_options_heimdall_defaults():
    opts = DispatchOptions(seeds=[0])
    assert opts.no_heimdall is False
    assert opts.heimdall_probe_interval_s == 300
    assert opts.heimdall_stale_threshold_s == 180


def test_dispatch_options_heimdall_overrides():
    opts = DispatchOptions(
        seeds=[0],
        no_heimdall=True,
        heimdall_probe_interval_s=60,
        heimdall_stale_threshold_s=45,
    )
    assert opts.no_heimdall is True
    assert opts.heimdall_probe_interval_s == 60
    assert opts.heimdall_stale_threshold_s == 45
