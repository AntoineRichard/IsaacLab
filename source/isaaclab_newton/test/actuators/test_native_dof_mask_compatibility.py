# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility coverage for the legacy native-DOF ownership helper."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import warp as wp


def test_build_native_dof_mask_keeps_legacy_signature_and_effort_result():
    """The public helper keeps its four arguments and ignores later Lab groups."""
    from isaaclab_newton.actuators import build_native_dof_mask
    from isaaclab_newton.actuators.kernels import _build_native_dof_masks

    assert tuple(inspect.signature(build_native_dof_mask).parameters) == (
        "actuators",
        "native_group_names",
        "num_joints",
        "device",
    )

    actuators = {
        "native": SimpleNamespace(joint_indices=torch.tensor([2, 0])),
        "later_lab": SimpleNamespace(joint_indices=torch.tensor([2])),
    }
    legacy_mask, legacy_owner = build_native_dof_mask(actuators, frozenset({"native"}), 4, "cpu")

    assert legacy_owner.tolist() == [1, 0, 1, 0]
    assert wp.to_torch(legacy_mask).tolist() == [1, 0, 1, 0]

    masks, owners = _build_native_dof_masks(actuators, frozenset({"native"}), 4, "cpu")
    assert owners["effort"] is owners["computed_effort"] is owners["applied_effort"]
    assert masks["effort"] is masks["computed_effort"] is masks["applied_effort"]
