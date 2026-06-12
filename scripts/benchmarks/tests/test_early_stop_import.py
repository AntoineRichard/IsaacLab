# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""early_stop uses the core SuccessRateTracker (Isaac-Sim-free import check)."""

import argparse

from isaaclab.test.benchmark.metrics import SuccessRateTracker as CoreTracker

from scripts.benchmarks import early_stop


def test_uses_core_tracker():
    assert early_stop.SuccessRateTracker is CoreTracker


def test_success_cli_args_roundtrip():
    p = argparse.ArgumentParser()
    early_stop.add_success_cli_args(p)
    ns = p.parse_args([])
    kw = early_stop.build_success_kwargs(ns)
    assert set(kw) == {"threshold", "window", "stop_on_convergence"}
    assert kw["stop_on_convergence"] is False
