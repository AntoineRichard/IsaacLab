# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for private typed actuator parameter storage."""

from __future__ import annotations

import warnings

import pytest
import torch
import warp as wp

from isaaclab.actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from isaaclab.actuators.actuator_pd import (
    DCMotor,
    DelayedPDActuator,
    IdealPDActuator,
    ImplicitActuator,
    RemotizedPDActuator,
)
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, DelayedPDActuatorCfg, RemotizedPDActuatorCfg
from isaaclab.actuators.actuator_storage import _GroupBinding
from isaaclab.utils.warp import ProxyArray


class _TestActuatorStorage:
    """Minimal canonical-array fixture for managed-parameter descriptor tests."""

    def __init__(self, *, num_worlds: int, device: str) -> None:
        self._num_worlds = num_worlds
        self._device = device
        self._arrays: dict[type, dict[str, ProxyArray]] = {}

    def allocate(self, actuator_type: type, num_slots: int) -> dict[str, ProxyArray]:
        """Allocate canonical arrays for one exact actuator schema."""
        if actuator_type.__dict__.get("_parameter_schema") is None:
            raise TypeError(f"{actuator_type.__name__} does not opt into managed parameter storage.")
        arrays = {
            field.name: ProxyArray(
                wp.from_torch(
                    torch.full((self._num_worlds, num_slots), field.fill, dtype=torch.float32, device=self._device),
                    dtype=wp.float32,
                )
            )
            for field in actuator_type._parameter_schema().fields
        }
        self._arrays[actuator_type] = arrays
        return arrays

    def array(self, actuator_type: type, name: str) -> ProxyArray:
        """Return one allocated canonical array."""
        return self._arrays[actuator_type][name]

    def allocated_fields(self, actuator_type: type) -> frozenset[str]:
        """Return allocated typed fields for an exact actuator type."""
        return frozenset(self._arrays.get(actuator_type, {}))


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
    store = _TestActuatorStorage(num_worlds=num_worlds, device="cpu")
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


def _bind_existing_group(group) -> _TestActuatorStorage:
    """Bind a fully initialized actuator while retaining its eager structural state."""
    store = _TestActuatorStorage(num_worlds=group.num_envs, device=group._device)
    arrays = store.allocate(type(group), group.num_joints)
    group._bind_parameter_storage(
        _GroupBinding(
            generation=0,
            joint_indices=torch.arange(group.num_joints, dtype=torch.int32),
            joint_names=tuple(group.joint_names),
            type_slice=slice(None),
            arrays=arrays,
        )
    )
    return store


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

    store = _TestActuatorStorage(num_worlds=1, device="cpu")
    with pytest.raises(TypeError, match="does not opt into"):
        store.allocate(CustomDCMotor, 1)


def test_delayed_pd_structural_signature_ignores_numeric_pd_parameters() -> None:
    """Delay state shape, not PD values, determines delayed actuator structure."""
    low_gains = DelayedPDActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=1.0,
        damping=2.0,
        effort_limit=3.0,
        velocity_limit=4.0,
        min_delay=1,
        max_delay=3,
    )
    high_gains = DelayedPDActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=10.0,
        damping=20.0,
        effort_limit=30.0,
        velocity_limit=40.0,
        min_delay=1,
        max_delay=3,
    )

    assert DelayedPDActuator._structural_signature(low_gains) == DelayedPDActuator._structural_signature(high_gains)


def test_delayed_pd_delay_buffers_remain_group_owned_state() -> None:
    """Delay buffers are owned by the eager group and excluded from typed fields."""
    cfg = DelayedPDActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=1.0,
        damping=2.0,
        effort_limit=3.0,
        velocity_limit=4.0,
        min_delay=1,
        max_delay=3,
    )
    actuator = DelayedPDActuator(cfg, ["joint"], slice(None), num_envs=2, device="cpu")
    delay_buffers = tuple(
        getattr(actuator, name)
        for name in (
            "positions_delay_buffer",
            "velocities_delay_buffer",
            "efforts_delay_buffer",
        )
    )
    _bind_existing_group(actuator)

    typed_fields = {field.name for field in actuator._parameter_schema().fields}
    for name, buffer in zip(
        ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"), delay_buffers
    ):
        assert name in actuator.__dict__
        assert name not in typed_fields
        assert getattr(actuator, name) is buffer


def test_remotized_pd_structural_signature_tracks_lookup_shape_not_values() -> None:
    """Lookup storage structure is independent of PD values and lookup samples."""
    low_values = RemotizedPDActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=1.0,
        damping=2.0,
        min_delay=0,
        max_delay=2,
        joint_parameter_lookup=[[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]],
    )
    high_values = RemotizedPDActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=10.0,
        damping=20.0,
        min_delay=0,
        max_delay=2,
        joint_parameter_lookup=[[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
    )

    assert RemotizedPDActuator._structural_signature(low_values) == RemotizedPDActuator._structural_signature(
        high_values
    )


def test_remotized_pd_lookup_remains_group_owned_state() -> None:
    """Lookup data remains on the eager group instead of canonical typed storage."""
    cfg = RemotizedPDActuatorCfg(
        joint_names_expr=["joint"],
        stiffness=1.0,
        damping=2.0,
        min_delay=0,
        max_delay=2,
        joint_parameter_lookup=[[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]],
    )
    actuator = RemotizedPDActuator(cfg, ["joint"], slice(None), num_envs=2, device="cpu")
    lookup = actuator._joint_parameter_lookup
    _bind_existing_group(actuator)

    typed_fields = {field.name for field in actuator._parameter_schema().fields}
    assert "_joint_parameter_lookup" in actuator.__dict__
    assert "_joint_parameter_lookup" not in typed_fields
    assert actuator._joint_parameter_lookup is lookup
