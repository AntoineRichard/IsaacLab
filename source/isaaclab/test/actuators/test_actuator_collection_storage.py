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
