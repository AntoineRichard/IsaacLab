# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for private typed actuator parameter storage."""

from __future__ import annotations

import warnings

import pytest
import torch

from isaaclab.actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg
from isaaclab.actuators.actuator_storage import _ActuatorStorage, _GroupBinding


def _make_bound_group(
    actuator_type: type,
    *,
    num_worlds: int = 2,
    group_dofs: int = 2,
    type_dofs: int = 2,
    offset: int = 0,
):
    """Create a private, bound actuator group without constructor-only state."""
    group = object.__new__(actuator_type)
    group._num_envs = num_worlds
    group._device = "cpu"
    group._joint_names = [f"joint_{index}" for index in range(group_dofs)]
    group.__dict__.update(
        {
            "effort_limit_sim": torch.full((num_worlds, group_dofs), 101.0),
            "velocity_limit_sim": torch.full((num_worlds, group_dofs), 102.0),
            "armature": torch.full((num_worlds, group_dofs), 103.0),
            "friction": torch.full((num_worlds, group_dofs), 104.0),
            "dynamic_friction": torch.full((num_worlds, group_dofs), 105.0),
            "viscous_friction": torch.full((num_worlds, group_dofs), 106.0),
            "stiffness": torch.full((num_worlds, group_dofs), 11.0),
            "damping": torch.full((num_worlds, group_dofs), 12.0),
        }
    )
    store = _ActuatorStorage(num_worlds=num_worlds, device="cpu")
    arrays = store.allocate(actuator_type, type_dofs)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.arange(offset, offset + group_dofs, dtype=torch.int32),
        joint_names=tuple(group._joint_names),
        type_slice=slice(offset, offset + group_dofs),
        arrays=arrays,
    )
    group._bind_parameter_storage(binding)
    return group, store


def make_bound_ideal_pd_group(*, num_worlds: int, group_dofs: int, type_dofs: int, offset: int):
    """Create an ideal-PD group and its canonical stiffness array."""
    group, store = _make_bound_group(
        IdealPDActuator,
        num_worlds=num_worlds,
        group_dofs=group_dofs,
        type_dofs=type_dofs,
        offset=offset,
    )
    return group, store.array(IdealPDActuator, "stiffness").torch


def make_bound_neural_group():
    """Create a neural actuator group without loading a network checkpoint."""
    group, _ = _make_bound_group(ActuatorNetMLP)
    return group


def make_bound_dc_group():
    """Create a DC motor group and its typed store."""
    return _make_bound_group(DCMotor)


@pytest.mark.parametrize(
    ("actuator_type", "expected"),
    [
        (ImplicitActuator, frozenset({"stiffness", "damping", "effort_limit", "velocity_limit"})),
        (IdealPDActuator, frozenset({"stiffness", "damping", "effort_limit", "velocity_limit"})),
        (
            DCMotor,
            frozenset({"stiffness", "damping", "effort_limit", "velocity_limit", "saturation_effort"}),
        ),
        (ActuatorNetMLP, frozenset({"effort_limit", "velocity_limit", "saturation_effort"})),
        (ActuatorNetLSTM, frozenset({"effort_limit", "velocity_limit", "saturation_effort"})),
    ],
)
def test_exact_class_parameter_schema_excludes_solver_properties(actuator_type, expected) -> None:
    """Schemas only describe typed actuator inputs, not solver compatibility values."""
    assert actuator_type._parameter_schema().parameter_names == expected
    assert "effort_limit_sim" not in expected
    assert "velocity_limit_sim" not in expected
    assert "armature" not in expected
    assert "friction" not in expected
    assert "dynamic_friction" not in expected
    assert "viscous_friction" not in expected


def test_managed_group_assignment_copies_without_rebinding() -> None:
    """Whole-attribute writes preserve the canonical strided parameter view."""
    group, canonical = make_bound_ideal_pd_group(num_worlds=3, group_dofs=2, type_dofs=5, offset=1)
    held = group.stiffness
    group.stiffness = torch.full((3, 2), 7.0, device=held.device)
    assert group.stiffness.data_ptr() == held.data_ptr()
    assert group.stiffness.stride() == (5, 1)
    torch.testing.assert_close(canonical[:, 1:3], held, rtol=0.0, atol=0.0)


def test_neural_gains_are_lazy_deprecated_sidecars() -> None:
    """Neural compatibility gains stay group-local and warn only when requested."""
    group = make_bound_neural_group()
    assert group._deprecated_sidecars == {}
    with pytest.warns(DeprecationWarning, match="stiffness"):
        stiffness = group.stiffness
    assert stiffness.shape == (group.num_envs, group.num_joints)
    assert set(group._deprecated_sidecars) == {"stiffness"}
    with warnings.catch_warnings(record=True) as warnings_record:
        warnings.simplefilter("always")
        assert group.stiffness is stiffness
    assert warnings_record == []


def test_solver_compatibility_fields_are_lazy_and_not_typed() -> None:
    """Solver values remain lazy group-local buffers outside the typed store."""
    group, store = make_bound_dc_group()
    assert "armature" not in store.allocated_fields(DCMotor)
    assert group._solver_compatibility_sidecars == {}
    armature = group.armature
    assert isinstance(armature, torch.Tensor)
    torch.testing.assert_close(armature, torch.full((2, 2), 103.0))
    assert set(group._solver_compatibility_sidecars) == {"armature"}


def test_dc_motor_resolves_mapping_saturation_effort_per_joint() -> None:
    """DC-motor saturation effort remains a per-world, per-joint tensor."""
    actuator = DCMotor(
        DCMotorCfg(
            joint_names_expr=["left", "right"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=20.0,
            velocity_limit=10.0,
            saturation_effort={"left": 40.0, "right": 60.0},
        ),
        joint_names=["left", "right"],
        joint_ids=slice(None),
        num_envs=2,
        device="cpu",
    )

    torch.testing.assert_close(actuator.saturation_effort, torch.tensor([[40.0, 60.0], [40.0, 60.0]]))


def test_opaque_subclass_does_not_inherit_managed_storage_opt_in() -> None:
    """A custom subclass remains unbound until it declares its own schema hook."""

    class CustomDCMotor(DCMotor):
        pass

    store = _ActuatorStorage(num_worlds=1, device="cpu")
    with pytest.raises(TypeError, match="does not opt into"):
        store.allocate(CustomDCMotor, 1)
