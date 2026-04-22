# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.common.env_list.derive_group`."""

import pytest

from tools.odin.common.env_list import derive_group


@pytest.mark.parametrize(
    "entry_point, expected_group",
    [
        ("isaaclab_tasks.direct.ant:AntEnv", "direct/ant"),
        ("isaaclab_tasks.direct.anymal_c.flat_env:FlatEnv", "direct/anymal_c"),
        (
            "isaaclab_tasks.manager_based.locomotion.velocity.config.anymal_c:EnvCfg",
            "manager_based/locomotion/velocity",
        ),
        (
            "isaaclab_tasks.manager_based.manipulation.lift:Env",
            "manager_based/manipulation/lift",
        ),
        ("isaaclab_tasks.direct.factory.factory_env:FactoryEnv", "direct/factory"),
    ],
)
def test_derive_group_known_shapes(entry_point, expected_group):
    assert derive_group(entry_point) == expected_group


def test_derive_group_missing_colon_returns_unknown():
    # Malformed: no class reference; can't derive meaningfully.
    assert derive_group("isaaclab_tasks.direct.ant") == "unknown"


def test_derive_group_not_isaaclab_tasks_returns_unknown():
    # Third-party task registration: don't attempt derivation.
    assert derive_group("some_other_pkg.envs.my_env:MyEnv") == "unknown"


def test_derive_group_empty_string():
    assert derive_group("") == "unknown"
