# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.common.env_list.suggest_framework`."""

import pytest

from tools.odin.common.env_list import suggest_framework


@pytest.mark.parametrize(
    "has_rsl_rl, has_skrl, expected",
    [
        (True,  True,  "rsl_rl"),   # both → prefer rsl_rl
        (True,  False, "rsl_rl"),
        (False, True,  "skrl"),
        (False, False, None),        # neither → caller should set keep=False
    ],
)
def test_suggest_framework_decision_table(has_rsl_rl, has_skrl, expected):
    assert suggest_framework(has_rsl_rl, has_skrl) == expected
