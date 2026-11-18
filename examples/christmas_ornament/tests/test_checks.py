# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The --check gates must be able to fail on the failure modes they guard."""

import types

import pytest

from examples.christmas_ornament.motion import PHASES
from examples.christmas_ornament.run import CREEP_GATE, TIP_MARGIN_GATE, OrnamentScene, clearance_window_start

pytestmark = [pytest.mark.kitless, pytest.mark.unit]


def _metrics(**overrides):
    base = {
        "finite": True,
        "complete": True,
        "max_hook_error_m": 0.001,
        "hung": True,
        "hung_fraction": 1.0,
        "released_clearance_m": 0.05,
        "dip_m": 0.01,
        "ornament_speed_end": 0.001,
        "trunk_top_disp_m": None,
        "admire_creep_m": 0.0005,
        "tip_margin_m": 0.03,
    }
    base.update(overrides)
    return base


def test_clearance_is_measured_from_the_retreat_on():
    """A finger that drags the ring during withdraw is still near it when retreat starts."""
    ends = {
        name: end for (name, _), end in zip(PHASES, __import__("numpy").cumsum([d for _, d in PHASES]), strict=True)
    }
    assert clearance_window_start() == pytest.approx(ends["withdraw"])


def test_creep_window_is_the_last_two_seconds_of_admire():
    """Settling after the drop is fine; drift still going at the end is the ring walking off."""
    from examples.christmas_ornament.run import creep_window_start

    total = sum(d for _, d in PHASES)
    assert creep_window_start() == pytest.approx(total - 2.0)


def test_creep_and_tip_margin_are_gated():
    check = OrnamentScene.check
    scene = types.SimpleNamespace()
    assert check(scene, _metrics()) == []
    assert any("creep" in f for f in check(scene, _metrics(admire_creep_m=CREEP_GATE + 0.001)))
    assert any("tip" in f for f in check(scene, _metrics(tip_margin_m=TIP_MARGIN_GATE - 0.001)))
