# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest config for Odin tests.

Asserts that no Kit / IsaacSim modules are loaded at test-session start —
the entire Odin test suite is pure-Python and should be runnable with
plain ``python3 -m pytest`` (not ``./isaaclab.sh -p``). Importing Kit
loads ``LD_PRELOAD=libcarb.so`` and triggers the Rerun viewer, which is
both slow and inappropriate for unit tests.
"""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def _no_kit_imported() -> None:
    """Fail fast if any test pulls in Kit / IsaacSim modules."""
    forbidden_prefixes = ("omni.kit", "isaacsim", "carb")
    leaked = sorted(m for m in sys.modules if m.startswith(forbidden_prefixes))
    assert not leaked, (
        f"Kit modules imported during Odin tests: {leaked}. "
        f"Odin tests are pure-Python; do not import Kit / IsaacSim."
    )
