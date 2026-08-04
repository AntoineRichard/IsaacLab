# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for private typed actuator parameter storage."""

from __future__ import annotations

import copy
import warnings
from types import SimpleNamespace

import pytest
import torch
import warp as wp

import isaaclab.actuators.actuator_storage as actuator_storage
import isaaclab.assets.asset_base as asset_base_module
from isaaclab.actuators.actuator_base import _ResolvedManagedRegistration
from isaaclab.actuators.actuator_control import ActuatorJointProperties, _ActuatorParameterWrite
from isaaclab.actuators.actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from isaaclab.actuators.actuator_pd import (
    DCMotor,
    DelayedPDActuator,
    IdealPDActuator,
    ImplicitActuator,
    RemotizedPDActuator,
)
from isaaclab.actuators.actuator_pd_cfg import (
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    RemotizedPDActuatorCfg,
)
from isaaclab.actuators.actuator_storage import _GroupBinding
from isaaclab.assets.asset_base import AssetBase
from isaaclab.cloner import ClonePlan
from isaaclab.utils.warp import ProxyArray


def _build_articulation_layout(**kwargs):
    """Call the production layout builder without making test collection depend on its existence."""
    return actuator_storage._build_articulation_layout(**kwargs)


def _group_registration(name, joint_indices, stiffness):
    """Build one prototype-local resolved group record."""
    return actuator_storage._GroupRegistration(
        name=name,
        actuator_type=IdealPDActuator,
        joint_indices=tuple(joint_indices),
        values={
            "stiffness": tuple(stiffness),
            "damping": tuple(value + 0.5 for value in stiffness),
            "effort_limit": tuple(value + 10.0 for value in stiffness),
            "velocity_limit": tuple(value + 20.0 for value in stiffness),
        },
    )


def _prototype_registration(key, num_joints, groups):
    """Build one source-prototype record."""
    return actuator_storage._PrototypeRegistration(
        registration_key=key,
        num_joints=num_joints,
        groups=tuple(groups),
    )


def _make_clone_plan(cfg, clone_mask):
    """Build a clone plan whose rows all belong to ``cfg``."""
    num_rows = clone_mask.shape[0]
    return ClonePlan(
        sources=tuple(f"/World/envs/env_{row}/Robot" for row in range(num_rows)),
        destinations=("/World/envs/env_{}/Robot",) * num_rows,
        clone_mask=clone_mask,
        cfg_rows={id(cfg): tuple(range(num_rows))},
    )


def make_variant_registrations(*, key="robot", num_prototypes=4):
    """Build three group records for every source prototype."""
    return tuple(
        _prototype_registration(
            key,
            6,
            (
                _group_registration("hip", (0, 2), (prototype + 1.0, prototype + 2.0)),
                _group_registration("knee", (1,), (prototype + 3.0,)),
                _group_registration("ankle", (4, 5), (prototype + 4.0, prototype + 5.0)),
            ),
        )
        for prototype in range(num_prototypes)
    )


def build_instrumented_layout(*, num_worlds, num_prototypes):
    """Build a layout while counting source-level records created by the fixture."""
    cfg = object()
    clone_mask = torch.arange(num_worlds).remainder(num_prototypes) == torch.arange(num_prototypes)[:, None]
    registrations = make_variant_registrations(num_prototypes=num_prototypes)
    counters = SimpleNamespace(
        group_records=sum(len(registration.groups) for registration in registrations),
        prototype_records=len(registrations),
    )
    return (
        _build_articulation_layout(
            replication_cfg_id=id(cfg),
            clone_plan=_make_clone_plan(cfg, clone_mask),
            registrations=registrations,
        ),
        counters,
    )


class _ArticulationLayoutFixture:
    """Small test adapter exposing scoped proxies from one immutable layout."""

    def __init__(self, store, layout):
        self._store = store
        self._layout = layout
        self.stiffness = None

    def type_proxy(self, actuator_type, field):
        return self._store.type_proxy(self._layout.type_layouts[actuator_type], field)

    def type_dofs(self, actuator_type):
        return self._layout.type_layouts[actuator_type].num_dofs


class _GroupLayoutFixture:
    """Small test adapter exposing the logical group's canonical stiffness view."""

    def __init__(self, store, layout):
        self.stiffness = store.group_proxy(layout, "stiffness").torch


def make_two_articulation_store(*, device="cpu"):
    """Allocate two articulation ranges in one exact-type flat store."""
    type_offsets = {}
    layouts = []
    for index, (num_worlds, group_joint_ids) in enumerate(((2, ((0, 2), (1,))), (3, ((0,), (2,))))):
        cfg = object()
        registrations = (
            _prototype_registration(
                f"robot_{index}",
                3,
                tuple(
                    _group_registration(f"group_{group}", joint_ids, (1.0,) * len(joint_ids))
                    for group, joint_ids in enumerate(group_joint_ids)
                ),
            ),
        )
        layouts.append(
            _build_articulation_layout(
                replication_cfg_id=id(cfg),
                clone_plan=_make_clone_plan(cfg, torch.ones((1, num_worlds), dtype=torch.bool, device=device)),
                registrations=registrations,
                type_offsets=type_offsets,
            )
        )
    store = actuator_storage._TypedStore(IdealPDActuator)
    store.allocate(layouts, device=device)
    first = _ArticulationLayoutFixture(store, layouts[0])
    second = _ArticulationLayoutFixture(store, layouts[1])
    second.stiffness = _GroupLayoutFixture(store, layouts[0].group_layouts[0]).stiffness
    return store, first, second


def make_type_layout(*, group_joint_ids):
    """Build the exact-type layout for overlapping logical groups."""
    cfg = object()
    registration = _prototype_registration(
        "robot",
        max(joint_id for joint_ids in group_joint_ids for joint_id in joint_ids) + 1,
        tuple(
            _group_registration(f"group_{index}", joint_ids, (1.0,) * len(joint_ids))
            for index, joint_ids in enumerate(group_joint_ids)
        ),
    )
    layout = _build_articulation_layout(
        replication_cfg_id=id(cfg),
        clone_plan=_make_clone_plan(cfg, torch.ones((1, 1), dtype=torch.bool)),
        registrations=(registration,),
    )
    return layout.type_layouts[IdealPDActuator]


def csr_slots(layout, *, articulation_joint_id):
    """Read one articulation joint's compact slots from the immutable CSR table."""
    start = layout.articulation_to_compact_offsets[articulation_joint_id]
    stop = layout.articulation_to_compact_offsets[articulation_joint_id + 1]
    return layout.articulation_to_compact_slots[start:stop]


class _MinimalAsset(AssetBase):
    """AssetBase test double that keeps constructor behavior and removes simulator callbacks."""

    @property
    def data(self):
        return None

    @property
    def num_instances(self):
        return 1

    def reset(self, env_ids=None):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass

    def _initialize_impl(self):
        pass

    def _register_callbacks(self):
        pass

    def _clear_callbacks(self):
        pass

    def set_debug_vis(self, debug_vis):
        return False


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


class _TracedLSTMNetwork(torch.nn.Module):
    """Tiny traceable LSTM matching the actuator network's public shape."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=3, num_layers=1)

    def forward(
        self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(inputs, state)


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


@pytest.mark.parametrize("num_worlds", [1, 64, 4096])
def test_layout_python_object_count_is_clone_count_independent(num_worlds: int) -> None:
    layout, counters = build_instrumented_layout(num_worlds=num_worlds, num_prototypes=4)
    assert layout.num_worlds == num_worlds
    assert counters.group_records == 12
    assert counters.prototype_records == 4


def test_type_block_is_contiguous_and_group_block_is_strided_zero_copy() -> None:
    store, articulation, group = make_two_articulation_store()
    type_stiffness = articulation.type_proxy(IdealPDActuator, "stiffness").torch
    assert type_stiffness.is_contiguous()
    assert group.stiffness.stride() == (articulation.type_dofs(IdealPDActuator), 1)
    assert group.stiffness.untyped_storage().data_ptr() == store.stiffness.torch.untyped_storage().data_ptr()


def test_multiple_articulation_type_ranges_are_disjoint() -> None:
    store, first, second = make_two_articulation_store()
    first_stiffness = first.type_proxy(IdealPDActuator, "stiffness").torch
    second_stiffness = second.type_proxy(IdealPDActuator, "stiffness").torch
    first_stiffness.fill_(3.0)
    second_stiffness.fill_(9.0)
    torch.testing.assert_close(first_stiffness, torch.full_like(first_stiffness, 3.0))
    torch.testing.assert_close(second_stiffness, torch.full_like(second_stiffness, 9.0))


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_joint_domain_views_are_disjoint_live_and_pointer_stable(device: str) -> None:
    """Joint-domain articulation aliases are cached, contiguous, and isolated."""
    _, first, second = make_two_articulation_store(device=device)
    joint_store = actuator_storage._JointDomainStore()
    joint_store.allocate((first._layout, second._layout), device=device)

    first_position = joint_store.articulation_proxy("raw_position", first._layout)
    second_position = joint_store.articulation_proxy("raw_position", second._layout)
    first_pointer = first_position.torch.data_ptr()

    assert first_position is joint_store.articulation_proxy("raw_position", first._layout)
    assert first_position.torch.is_contiguous()
    assert second_position.torch.is_contiguous()
    second_position.torch.fill_(8.0)

    assert first_position.torch.data_ptr() == first_pointer
    assert torch.count_nonzero(first_position.torch) == 0
    assert torch.all(second_position.torch == 8.0)


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_joint_domain_store_eagerly_caches_one_alias_for_each_field_and_articulation(device: str) -> None:
    """Eight flat fields prebuild stable aliases before facade installation."""
    _, first, second = make_two_articulation_store()
    joint_store = actuator_storage._JointDomainStore()
    joint_store.allocate((first._layout, second._layout), device=device)

    assert set(joint_store._fields) == {
        "raw_position",
        "raw_velocity",
        "raw_effort",
        "processed_position",
        "processed_velocity",
        "processed_effort",
        "computed_effort",
        "applied_effort",
    }
    assert len({field.torch.data_ptr() for field in joint_store._fields.values()}) == 8
    assert len(joint_store._articulation_proxies) == 16
    for layout in (first._layout, second._layout):
        for field in joint_store._fields:
            assert joint_store.articulation_proxy(field, layout) is joint_store.articulation_proxy(field, layout)


def test_joint_domain_store_metadata_is_bounded_for_4096_worlds() -> None:
    """Joint-domain construction stores articulation prefixes rather than per-world objects."""
    layout, _ = build_instrumented_layout(num_worlds=4096, num_prototypes=4)
    joint_store = actuator_storage._JointDomainStore()
    joint_store.allocate((layout,), device="cpu")

    assert len(joint_store._fields) == 8
    assert len(joint_store._articulation_offsets) == 1
    assert len(joint_store._articulation_proxies) == 8
    assert joint_store.articulation_proxy("raw_position", layout).shape == (4096, layout.num_joints)


def test_overlapping_type_layout_builds_one_to_many_joint_fanout() -> None:
    layout = make_type_layout(group_joint_ids=((0, 2), (2, 3), (2,)))
    assert layout.compact_joint_indices == (0, 2, 2, 3, 2)
    assert csr_slots(layout, articulation_joint_id=2) == (1, 2, 4)


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_heterogeneous_layout_expands_four_prototypes_to_4096_worlds(device: str) -> None:
    num_worlds = 4096
    num_prototypes = 4
    cfg = object()
    clone_mask = (
        torch.arange(num_worlds, device=device).remainder(num_prototypes)
        == torch.arange(num_prototypes, device=device)[:, None]
    )
    registrations = tuple(
        _prototype_registration(
            "robot",
            3,
            (
                _group_registration("first", (0, 2), (10.0 + prototype, 20.0 + prototype)),
                _group_registration("second", (1,), (30.0 + prototype,)),
            ),
        )
        for prototype in range(num_prototypes)
    )
    layout = _build_articulation_layout(
        replication_cfg_id=id(cfg),
        clone_plan=_make_clone_plan(cfg, clone_mask),
        registrations=registrations,
    )
    store = actuator_storage._TypedStore(IdealPDActuator)
    store.allocate((layout,), device=device)

    actual = store.type_proxy(layout.type_layouts[IdealPDActuator], "stiffness").torch
    prototype_values = torch.tensor(
        [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0], [13.0, 23.0, 33.0]],
        device=device,
    )
    expected = prototype_values[torch.arange(num_worlds, device=device).remainder(num_prototypes)]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_layout_proxies_cache_one_torch_alias_and_device_csr_copy() -> None:
    store, articulation, group = make_two_articulation_store()
    layout = articulation._layout.type_layouts[IdealPDActuator]
    type_proxy = articulation.type_proxy(IdealPDActuator, "stiffness")
    offsets, slots = store.mapping_proxies(layout)
    assert type_proxy is articulation.type_proxy(IdealPDActuator, "stiffness")
    assert type_proxy.torch is type_proxy.torch
    assert group.stiffness is not None
    torch.testing.assert_close(offsets.torch, torch.tensor(layout.articulation_to_compact_offsets, dtype=torch.int32))
    torch.testing.assert_close(slots.torch, torch.tensor(layout.articulation_to_compact_slots, dtype=torch.int32))


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_layout_alias_and_csr_semantics_match_on_cpu_and_cuda(device: str) -> None:
    store, first, group = make_two_articulation_store(device=device)
    layout = first._layout.type_layouts[IdealPDActuator]
    type_proxy = first.type_proxy(IdealPDActuator, "stiffness")
    offsets, slots = store.mapping_proxies(layout)
    assert type_proxy.torch.is_contiguous()
    assert type_proxy.torch is type_proxy.torch
    assert group.stiffness.stride() == (layout.num_dofs, 1)
    assert group.stiffness.untyped_storage().data_ptr() == store.stiffness.torch.untyped_storage().data_ptr()
    second_type = group.type_proxy(IdealPDActuator, "stiffness").torch
    type_proxy.torch.fill_(3.0)
    second_type.fill_(9.0)
    torch.testing.assert_close(type_proxy.torch, torch.full_like(type_proxy.torch, 3.0))
    torch.testing.assert_close(second_type, torch.full_like(second_type, 9.0))
    torch.testing.assert_close(
        offsets.torch,
        torch.tensor(layout.articulation_to_compact_offsets, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(
        slots.torch,
        torch.tensor(layout.articulation_to_compact_slots, dtype=torch.int32, device=device),
    )


def test_layout_rejects_unknown_exact_schema_field() -> None:
    cfg = object()
    registration = actuator_storage._PrototypeRegistration(
        registration_key="robot",
        num_joints=1,
        groups=(
            actuator_storage._GroupRegistration(
                name="joint",
                actuator_type=IdealPDActuator,
                joint_indices=(0,),
                values={"stifness": (1.0,)},
            ),
        ),
    )
    with pytest.raises(ValueError, match="stifness.*IdealPDActuator.*joint"):
        _build_articulation_layout(
            replication_cfg_id=id(cfg),
            clone_plan=_make_clone_plan(cfg, torch.ones((1, 1), dtype=torch.bool)),
            registrations=(registration,),
        )


def test_asset_records_original_cfg_identity_before_copy(monkeypatch) -> None:
    events = []

    class _Cfg:
        disable_shape_checks = None
        spawn = None
        prim_path = "/World/Robot"
        debug_vis = False

        def validate(self):
            pass

        def copy(self):
            events.append(("copy", id(self)))
            return copy.copy(self)

    cfg = _Cfg()
    monkeypatch.setattr(asset_base_module, "queue_replication", lambda value: events.append(("queue", id(value))))
    monkeypatch.setattr(asset_base_module, "get_current_stage", object)

    asset = _MinimalAsset(cfg)

    assert events == [("queue", id(cfg)), ("copy", id(cfg))]
    assert asset._replication_cfg_id == id(cfg)
    assert id(asset.cfg) != id(cfg)


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_backend_parameter_staging_patches_only_owned_indexed_joints_without_reallocation(device: str) -> None:
    """Stage canonical compact values into their configuration-owned backend joints.

    This fails if backend staging allocates a compact selector, treats an
    articulation joint as a compact slot, or rewrites joints outside the
    selected Cartesian slice.
    """
    staging = actuator_storage._BackendParameterStaging(
        num_worlds=2,
        num_joints=4,
        device=device,
        owner_slots={(IdealPDActuator, "stiffness"): torch.tensor([1, -1, 0, 2], dtype=torch.int32, device=device)},
    )
    compact = ProxyArray(
        wp.from_torch(
            torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=torch.float32, device=device),
            dtype=wp.float32,
        )
    )
    target = staging.target(IdealPDActuator, "stiffness")
    target.torch.fill_(-5.0)
    pointer = target.torch.data_ptr()

    staging.patch_index(
        actuator_type=IdealPDActuator,
        name="stiffness",
        canonical=compact,
        env_ids=torch.tensor([1], dtype=torch.int32, device=device),
        joint_ids=torch.tensor([0, 1, 3], dtype=torch.int64, device=device),
    )

    assert target.torch.data_ptr() == pointer
    torch.testing.assert_close(
        target.torch,
        torch.tensor([[-5.0, -5.0, -5.0, -5.0], [21.0, -5.0, -5.0, 22.0]], device=device),
    )


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_backend_parameter_staging_respects_full_masks_and_releases_candidate_storage(device: str) -> None:
    """Stage a full mask without retaining caller-owned values after candidate close.

    This fails if mask staging does not use canonical compact slots, updates a
    disabled world or joint, or leaves candidate-owned storage live on close.
    """
    staging = actuator_storage._BackendParameterStaging(
        num_worlds=2,
        num_joints=3,
        device=device,
        owner_slots={(IdealPDActuator, "damping"): torch.tensor([0, -1, 1], dtype=torch.int32, device=device)},
    )
    compact = ProxyArray(
        wp.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device), dtype=wp.float32)
    )
    target = staging.target(IdealPDActuator, "damping")
    target.torch.fill_(-7.0)

    staging.patch_mask(
        actuator_type=IdealPDActuator,
        name="damping",
        canonical=compact,
        env_mask=torch.tensor([False, True], dtype=torch.bool, device=device),
        joint_mask=torch.tensor([True, True, False], dtype=torch.bool, device=device),
    )

    torch.testing.assert_close(
        target.torch,
        torch.tensor([[-7.0, -7.0, -7.0], [3.0, -7.0, -7.0]], device=device),
    )
    staging.close()
    assert staging._targets == {}


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_backend_parameter_staging_uses_type_canonical_values_for_group_writes(device: str) -> None:
    """Use the type-wide canonical field when a group is only a strided subview.

    This fails if a group side effect feeds its local ``value`` view to the
    backend path: owner slot 2 is outside the group's single local column.
    """
    canonical = ProxyArray(
        wp.from_torch(
            torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=torch.float32, device=device),
            dtype=wp.float32,
        )
    )
    group_value = torch.full((2, 1), 99.0, dtype=torch.float32, device=device)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.tensor([0], dtype=torch.int32, device=device),
        joint_names=("joint_0",),
        type_slice=slice(0, 1),
        arrays={"stiffness": canonical},
    )
    write = _ActuatorParameterWrite(
        value=group_value,
        env_ids=torch.tensor([0, 1], dtype=torch.int32, device=device),
        joint_ids=torch.tensor([0], dtype=torch.int32, device=device),
        group_binding=binding,
        canonical=canonical,
    )
    staging = actuator_storage._BackendParameterStaging(
        num_worlds=2,
        num_joints=1,
        device=device,
        owner_slots={(IdealPDActuator, "stiffness"): torch.tensor([2], dtype=torch.int32, device=device)},
    )

    staging.patch_write(actuator_type=IdealPDActuator, name="stiffness", write=write)

    torch.testing.assert_close(
        staging.target(IdealPDActuator, "stiffness").torch,
        torch.tensor([[12.0], [22.0]], device=device),
    )


@pytest.mark.parametrize("device", ["cpu", *(str(device) for device in wp.get_cuda_devices())])
def test_backend_parameter_staging_reuses_candidate_owned_default_selectors(device: str) -> None:
    """Use persistent all-world/all-joint selectors when a write omits both selectors.

    This fails if the steady path synthesizes temporary selector tensors or
    leaves unselected owner slots stale when the public API passes ``None``.
    """
    canonical = ProxyArray(
        wp.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device), dtype=wp.float32)
    )
    staging = actuator_storage._BackendParameterStaging(
        num_worlds=2,
        num_joints=3,
        device=device,
        owner_slots={(IdealPDActuator, "stiffness"): torch.tensor([1, -1, 0], dtype=torch.int32, device=device)},
    )
    write = _ActuatorParameterWrite(value=canonical.torch, canonical=canonical)
    env_ids = staging._all_env_ids
    joint_ids = staging._all_joint_ids

    staging.patch_write(actuator_type=IdealPDActuator, name="stiffness", write=write)

    assert staging._all_env_ids is env_ids
    assert staging._all_joint_ids is joint_ids
    torch.testing.assert_close(
        staging.target(IdealPDActuator, "stiffness").torch,
        torch.tensor([[2.0, 0.0, 1.0], [4.0, 0.0, 3.0]], device=device),
    )


@pytest.mark.skipif(not wp.get_cuda_devices(), reason="CUDA is required to validate cross-device backend staging.")
def test_backend_parameter_staging_rejects_a_canonical_array_on_the_wrong_device() -> None:
    """Reject a cross-device canonical source before the backend patch launch."""
    staging = actuator_storage._BackendParameterStaging(
        num_worlds=1,
        num_joints=1,
        device="cuda:0",
        owner_slots={(IdealPDActuator, "stiffness"): torch.tensor([0], dtype=torch.int32, device="cuda:0")},
    )
    canonical = ProxyArray(wp.from_torch(torch.ones((1, 1), dtype=torch.float32), dtype=wp.float32))
    write = _ActuatorParameterWrite(value=canonical.torch, canonical=canonical)

    with pytest.raises(ValueError, match="canonical.*candidate device"):
        staging.patch_write(actuator_type=IdealPDActuator, name="stiffness", write=write)


def test_backend_parameter_staging_shares_one_solver_property_target_across_exact_types() -> None:
    """Cross-type partial patches retain earlier winners and untouched solver defaults."""
    initial = ProxyArray(wp.from_torch(torch.full((1, 3), 9.0), dtype=wp.float32))
    staging = actuator_storage._BackendParameterStaging(
        num_worlds=1,
        num_joints=3,
        device="cpu",
        owner_slots={
            (IdealPDActuator, "stiffness"): torch.tensor([0, -1, -1], dtype=torch.int32),
            (ImplicitActuator, "stiffness"): torch.tensor([-1, 0, -1], dtype=torch.int32),
        },
        initial_values={"stiffness": initial},
    )
    ideal = ProxyArray(wp.from_torch(torch.tensor([[1.0]], dtype=torch.float32), dtype=wp.float32))
    implicit = ProxyArray(wp.from_torch(torch.tensor([[2.0]], dtype=torch.float32), dtype=wp.float32))
    env_ids = torch.tensor([0], dtype=torch.int32)

    staging.patch_index(
        actuator_type=IdealPDActuator,
        name="stiffness",
        canonical=ideal,
        env_ids=env_ids,
        joint_ids=torch.tensor([0], dtype=torch.int32),
    )
    staging.patch_index(
        actuator_type=ImplicitActuator,
        name="stiffness",
        canonical=implicit,
        env_ids=env_ids,
        joint_ids=torch.tensor([1], dtype=torch.int32),
    )

    assert staging.target(IdealPDActuator, "stiffness") is staging.target(ImplicitActuator, "stiffness")
    torch.testing.assert_close(staging.target(IdealPDActuator, "stiffness").torch, torch.tensor([[1.0, 2.0, 9.0]]))


def _source_defaults(stiffness: float, damping: float, *, device: str = "cpu") -> ActuatorJointProperties:
    """Build one source-prototype row of articulation solver defaults."""
    values = torch.tensor([[stiffness, stiffness + 1.0]], dtype=torch.float32, device=device)
    return ActuatorJointProperties(
        stiffness=values,
        damping=torch.full_like(values, damping),
        armature=torch.full_like(values, 0.1),
        friction=torch.full_like(values, 0.2),
        dynamic_friction=torch.full_like(values, 0.3),
        viscous_friction=torch.full_like(values, 0.4),
        effort_limit=torch.full_like(values, 100.0),
        velocity_limit=torch.full_like(values, 20.0),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for source-resolution sync coverage")
def test_source_registration_skips_debug_scalar_resolution_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source construction must not read CUDA values on the host when diagnostics are disabled."""
    defaults = ActuatorJointProperties(
        stiffness=torch.tensor([[1.0, 2.0], [11.0, 12.0]], dtype=torch.float32, device="cuda"),
        damping=torch.full((2, 2), 3.0, dtype=torch.float32, device="cuda"),
        armature=torch.full((2, 2), 0.1, dtype=torch.float32, device="cuda"),
        friction=torch.full((2, 2), 0.2, dtype=torch.float32, device="cuda"),
        dynamic_friction=torch.full((2, 2), 0.3, dtype=torch.float32, device="cuda"),
        viscous_friction=torch.full((2, 2), 0.4, dtype=torch.float32, device="cuda"),
        effort_limit=torch.full((2, 2), 100.0, dtype=torch.float32, device="cuda"),
        velocity_limit=torch.full((2, 2), 20.0, dtype=torch.float32, device="cuda"),
    )
    cfg = IdealPDActuatorCfg(
        joint_names_expr=["joint_0", "joint_1"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )

    def _host_read_forbidden(*_args, **_kwargs):
        raise AssertionError("source registration attempted a CUDA host read")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "allclose", _host_read_forbidden)
        patch.setattr(torch, "all", _host_read_forbidden)
        patch.setattr(torch.Tensor, "item", _host_read_forbidden)
        patch.setattr(torch.Tensor, "tolist", _host_read_forbidden)
        patch.setattr(torch.Tensor, "__float__", _host_read_forbidden)
        patch.setattr(torch.Tensor, "__int__", _host_read_forbidden)
        resolved = IdealPDActuator._resolve_managed_registration(
            cfg=cfg,
            joint_names=["joint_0", "joint_1"],
            joint_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            defaults_by_source=defaults,
            debug_value_resolution=False,
        )

    assert resolved.source_shell.num_envs == 2
    assert resolved.source_values["stiffness"].device.type == "cuda"


def test_managed_registration_resolves_exact_class_parameters_per_source_prototype() -> None:
    """Resolve one exact class over source rows without expanding clone worlds.

    This fails if registration skips the exact class's existing parsing rules,
    combines prototype defaults incorrectly, or materializes the final world
    count before canonical storage exists.
    """
    resolved = IdealPDActuator._resolve_managed_registration(
        cfg=IdealPDActuatorCfg(
            joint_names_expr=["joint_0", "joint_1"],
            stiffness=None,
            damping=None,
            effort_limit=None,
            velocity_limit=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        joint_names=["joint_0", "joint_1"],
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        defaults_by_source=(_source_defaults(1.0, 3.0), _source_defaults(11.0, 13.0)),
    )

    assert resolved.actuator_type is IdealPDActuator
    assert resolved.source_shell.num_envs == 2
    torch.testing.assert_close(resolved.source_values["stiffness"], torch.tensor([[1.0, 2.0], [11.0, 12.0]]))
    torch.testing.assert_close(resolved.source_values["damping"], torch.tensor([[3.0, 3.0], [13.0, 13.0]]))


def test_managed_runtime_shell_rebinds_ideal_pd_to_world_sized_canonical_storage() -> None:
    """Bind one exact logical shell after source-only parameter resolution.

    This fails if runtime construction allocates a second world-sized parameter
    tensor or leaves the group attached to source-prototype parameter rows.
    """
    resolved = IdealPDActuator._resolve_managed_registration(
        cfg=IdealPDActuatorCfg(
            joint_names_expr=["joint_0", "joint_1"],
            stiffness=None,
            damping=None,
            effort_limit=None,
            velocity_limit=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        joint_names=["joint_0", "joint_1"],
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        defaults_by_source=(_source_defaults(1.0, 3.0), _source_defaults(11.0, 13.0)),
    )
    store = _TestActuatorStorage(num_worlds=4, device="cpu")
    arrays = store.allocate(IdealPDActuator, 2)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        joint_names=("joint_0", "joint_1"),
        type_slice=slice(0, 2),
        arrays=arrays,
    )

    runtime = IdealPDActuator._build_managed_runtime_shell(
        resolved=resolved,
        binding=binding,
        num_envs=4,
        device="cpu",
        joint_indices=binding.joint_indices,
    )

    assert type(runtime) is IdealPDActuator
    assert runtime.num_envs == 4
    assert runtime.stiffness.data_ptr() == arrays["stiffness"].torch.data_ptr()


def test_dc_motor_runtime_shell_reallocates_only_its_world_sized_structural_state() -> None:
    """Keep DC motor scratch state aligned with final worlds after source resolution.

    This fails when the generic source-shell copy leaves the motor's private
    state at the number of source prototypes rather than the final world count.
    """
    resolved = DCMotor._resolve_managed_registration(
        cfg=DCMotorCfg(
            joint_names_expr=["joint_0", "joint_1"],
            stiffness=None,
            damping=None,
            effort_limit=None,
            velocity_limit=20.0,
            effort_limit_sim=None,
            velocity_limit_sim=None,
            saturation_effort=100.0,
        ),
        joint_names=["joint_0", "joint_1"],
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        defaults_by_source=(_source_defaults(1.0, 3.0), _source_defaults(11.0, 13.0)),
    )
    store = _TestActuatorStorage(num_worlds=4, device="cpu")
    arrays = store.allocate(DCMotor, 2)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        joint_names=("joint_0", "joint_1"),
        type_slice=slice(0, 2),
        arrays=arrays,
    )

    runtime = DCMotor._build_managed_runtime_shell(
        resolved=resolved,
        binding=binding,
        num_envs=4,
        device="cpu",
        joint_indices=binding.joint_indices,
    )

    assert runtime._joint_vel.shape == (4, 2)
    assert runtime._zeros_effort.shape == (4, 2)


@pytest.mark.parametrize(
    ("actuator_type", "cfg"),
    [
        (
            DelayedPDActuator,
            DelayedPDActuatorCfg(
                joint_names_expr=["joint_0", "joint_1"],
                stiffness=None,
                damping=None,
                effort_limit=None,
                velocity_limit=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
                min_delay=0,
                max_delay=2,
            ),
        ),
        (
            RemotizedPDActuator,
            RemotizedPDActuatorCfg(
                joint_names_expr=["joint_0", "joint_1"],
                stiffness=None,
                damping=None,
                effort_limit=None,
                velocity_limit=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
                min_delay=0,
                max_delay=2,
                joint_parameter_lookup=[[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ),
        ),
    ],
)
def test_delayed_runtime_shell_reallocates_delay_state_for_final_worlds(actuator_type, cfg) -> None:
    """Rebuild delay state at final world count while retaining the exact class.

    This fails when either delayed class keeps a two-source delay buffer after
    its canonical parameter storage has been expanded to four worlds.
    """
    resolved = actuator_type._resolve_managed_registration(
        cfg=cfg,
        joint_names=["joint_0", "joint_1"],
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        defaults_by_source=(_source_defaults(1.0, 3.0), _source_defaults(11.0, 13.0)),
    )
    store = _TestActuatorStorage(num_worlds=4, device="cpu")
    arrays = store.allocate(actuator_type, 2)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        joint_names=("joint_0", "joint_1"),
        type_slice=slice(0, 2),
        arrays=arrays,
    )

    runtime = actuator_type._build_managed_runtime_shell(
        resolved=resolved,
        binding=binding,
        num_envs=4,
        device="cpu",
        joint_indices=binding.joint_indices,
    )

    assert type(runtime) is actuator_type
    assert runtime.positions_delay_buffer.batch_size == 4
    assert runtime.velocities_delay_buffer.batch_size == 4
    assert runtime.efforts_delay_buffer.batch_size == 4
    assert runtime._ALL_INDICES.shape == (4,)
    if actuator_type is DelayedPDActuator:
        assert ("delay_history_length", 2) in resolved.structural_signature[-1]
    else:
        assert ("delay_history_length", 2) in resolved.structural_signature[-1]
        assert ("joint_parameter_lookup_shape", (2, 3)) in resolved.structural_signature[-1]


@pytest.mark.parametrize("actuator_type", [ActuatorNetMLP, ActuatorNetLSTM])
def test_neural_runtime_shell_reallocates_world_sized_state_without_reloading_the_network(actuator_type) -> None:
    """Rebuild neural state for final worlds while reusing one loaded network object.

    This fails if the generic shell copy retains state for two source rows, or
    if a runtime rebuild replaces the already loaded immutable TorchScript model.
    """
    source = object.__new__(actuator_type)
    source._num_envs = 2
    source._device = "cpu"
    source._joint_names = ["joint_0", "joint_1"]
    source._joint_indices = torch.tensor([0, 1], dtype=torch.int32)
    source.network = SimpleNamespace(
        lstm=SimpleNamespace(
            state_dict=lambda: {
                "weight_ih_l0": torch.empty((8, 3)),
                "weight_hh_l0": torch.empty((8, 3)),
                "bias_ih_l0": torch.empty(8),
                "bias_hh_l0": torch.empty(8),
            }
        )
    )
    if actuator_type is ActuatorNetMLP:
        source.cfg = SimpleNamespace(input_idx=(0, 2))
        source._joint_pos_error_history = torch.zeros((2, 3, 2))
        source._joint_vel_history = torch.zeros((2, 3, 2))
    else:
        source.cfg = SimpleNamespace()
        source.sea_input = torch.zeros((4, 1, 2))
        source.sea_hidden_state = torch.zeros((1, 4, 3))
        source.sea_cell_state = torch.zeros((1, 4, 3))
        source.sea_hidden_state_per_env = source.sea_hidden_state.view((1, 2, 2, 3))
        source.sea_cell_state_per_env = source.sea_cell_state.view((1, 2, 2, 3))
    source._joint_vel = torch.zeros((2, 2))
    source._zeros_effort = torch.zeros((2, 2))
    source._vel_at_effort_lim = torch.zeros((2, 2))

    resolved = _ResolvedManagedRegistration(
        cfg=source.cfg,
        actuator_type=actuator_type,
        joint_names=("joint_0", "joint_1"),
        joint_indices=source._joint_indices,
        source_shell=source,
        source_values={},
        structural_signature=(actuator_type,),
    )
    store = _TestActuatorStorage(num_worlds=4, device="cpu")
    arrays = store.allocate(actuator_type, 2)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        joint_names=("joint_0", "joint_1"),
        type_slice=slice(0, 2),
        arrays=arrays,
    )

    runtime = actuator_type._build_managed_runtime_shell(
        resolved=resolved,
        binding=binding,
        num_envs=4,
        device="cpu",
        joint_indices=binding.joint_indices,
    )

    assert runtime.network is source.network
    assert runtime._joint_vel.shape == (4, 2)
    if actuator_type is ActuatorNetMLP:
        assert runtime._joint_pos_error_history.shape == (4, 3, 2)
        assert runtime._joint_vel_history.shape == (4, 3, 2)
    else:
        assert runtime.sea_input.shape == (8, 1, 2)
        assert runtime.sea_hidden_state_per_env.shape == (1, 4, 2, 3)
        assert runtime.sea_cell_state_per_env.shape == (1, 4, 2, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for runtime-device structural coverage")
@pytest.mark.parametrize("actuator_type", [ActuatorNetMLP, ActuatorNetLSTM])
def test_neural_runtime_shell_moves_cpu_source_structure_to_cuda(actuator_type) -> None:
    """Move one CPU-resolved source network to its sole CUDA runtime consumer."""
    source = object.__new__(actuator_type)
    source._num_envs = 2
    source._device = "cpu"
    source._joint_names = ["joint_0", "joint_1"]
    source._joint_indices = torch.tensor([0, 1], dtype=torch.int32)
    source._joint_vel = torch.zeros((2, 2))
    source._zeros_effort = torch.zeros((2, 2))
    source._vel_at_effort_lim = torch.zeros((2, 2))
    if actuator_type is ActuatorNetMLP:
        source.cfg = SimpleNamespace(input_idx=(0, 2))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated", category=DeprecationWarning)
            source.network = torch.jit.script(torch.nn.Linear(6, 1))
        source._joint_pos_error_history = torch.zeros((2, 3, 2))
        source._joint_vel_history = torch.zeros((2, 3, 2))
    else:
        source.cfg = SimpleNamespace()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`torch.jit.script` is deprecated", category=DeprecationWarning)
            source.network = torch.jit.script(_TracedLSTMNetwork())
        source.sea_input = torch.zeros((4, 1, 2))
        source.sea_hidden_state = torch.zeros((1, 4, 3))
        source.sea_cell_state = torch.zeros((1, 4, 3))
        source.sea_hidden_state_per_env = source.sea_hidden_state.view((1, 2, 2, 3))
        source.sea_cell_state_per_env = source.sea_cell_state.view((1, 2, 2, 3))

    resolved = _ResolvedManagedRegistration(
        cfg=source.cfg,
        actuator_type=actuator_type,
        joint_names=("joint_0", "joint_1"),
        joint_indices=source._joint_indices,
        source_shell=source,
        source_values={},
        structural_signature=(actuator_type,),
    )
    store = _TestActuatorStorage(num_worlds=4, device="cuda")
    arrays = store.allocate(actuator_type, 2)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        joint_names=("joint_0", "joint_1"),
        type_slice=slice(0, 2),
        arrays=arrays,
    )
    assert all(parameter.device.type == "cpu" for parameter in source.network.parameters())

    runtime = actuator_type._build_managed_runtime_shell(
        resolved=resolved,
        binding=binding,
        num_envs=4,
        device="cuda",
        joint_indices=binding.joint_indices,
    )

    assert runtime.network is source.network
    assert all(parameter.device.type == "cuda" for parameter in runtime.network.parameters())
    assert all(parameter.device.type == "cuda" for parameter in source.network.parameters())
    if actuator_type is ActuatorNetMLP:
        assert runtime._joint_pos_error_history.device.type == "cuda"
        assert runtime._joint_vel_history.device.type == "cuda"
    else:
        assert runtime.sea_input.device.type == "cuda"
        assert runtime.sea_hidden_state.device.type == "cuda"
        assert runtime.sea_cell_state.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for runtime-device structural coverage")
def test_remotized_runtime_shell_moves_cpu_source_lookup_to_cuda() -> None:
    """Rebuild the remotized lookup interpolator on the runtime device."""
    cfg = RemotizedPDActuatorCfg(
        joint_names_expr=["joint_0", "joint_1"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
        min_delay=0,
        max_delay=2,
        joint_parameter_lookup=[[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    )
    resolved = RemotizedPDActuator._resolve_managed_registration(
        cfg=cfg,
        joint_names=["joint_0", "joint_1"],
        joint_indices=torch.tensor([0, 1], dtype=torch.int32),
        defaults_by_source=(_source_defaults(1.0, 3.0), _source_defaults(11.0, 13.0)),
    )
    store = _TestActuatorStorage(num_worlds=4, device="cuda")
    arrays = store.allocate(RemotizedPDActuator, 2)
    binding = _GroupBinding(
        generation=0,
        joint_indices=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        joint_names=("joint_0", "joint_1"),
        type_slice=slice(0, 2),
        arrays=arrays,
    )

    runtime = RemotizedPDActuator._build_managed_runtime_shell(
        resolved=resolved,
        binding=binding,
        num_envs=4,
        device="cuda",
        joint_indices=binding.joint_indices,
    )

    assert resolved.source_shell._joint_parameter_lookup.device.type == "cpu"
    assert runtime._joint_parameter_lookup.device.type == "cuda"
    assert runtime._torque_limit._x.device.type == "cuda"
    assert runtime._torque_limit._y.device.type == "cuda"
