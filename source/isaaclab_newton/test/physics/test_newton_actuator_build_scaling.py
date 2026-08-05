# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic construction-scaling coverage for hosted Newton actuators."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import warp as wp
from newton.actuators import ClampingPositionBased, ControllerPD

from isaaclab.utils.warp import ProxyArray


@dataclass(frozen=True)
class _BuildAudit:
    """Deterministic call, allocation, and retained-storage census."""

    allocation_calls: Counter
    alias_calls: Counter
    host_constructions: Counter
    launch_calls: Counter
    resolver_calls: Counter
    scalar_conversions: Counter
    retained_torch_owner_sizes: tuple[int, ...]
    direct_pointer_registration_counts: tuple[int, ...]
    direct_pointer_original_allocations: tuple[tuple[int, int], ...]
    descriptor_sizes: tuple[int, ...]
    live_bytes: Counter
    clone_calls: int
    adapter: Any
    native_binding: Any
    binding: Any


def _make_stage(monkeypatch, parsed_actuators: list[object]):
    """Create an in-memory USD traversal whose actuator parser is controlled."""
    import newton.actuators as newton_actuators

    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot")
    parsed_by_path = {}
    for index, parsed in enumerate(parsed_actuators):
        prim = stage.DefinePrim(f"/Robot/Actuator_{index}")
        parsed_by_path[str(prim.GetPath())] = parsed
    monkeypatch.setattr(
        newton_actuators,
        "parse_actuator_prim",
        lambda prim: parsed_by_path.get(str(prim.GetPath())),
    )
    return stage


def _parsed_actuator(joint_name: str, signature_id: int) -> SimpleNamespace:
    """Return one PD recipe with a shared lookup table that identifies its signature."""
    return SimpleNamespace(
        target_path=f"/Robot/{joint_name}",
        controller_class=ControllerPD,
        controller_kwargs={"kp": 100.0 + signature_id, "kd": 10.0 + signature_id},
        component_specs=[
            (
                ClampingPositionBased,
                {
                    "lookup_positions": (0.0, 1.0),
                    "lookup_efforts": (0.0, float(signature_id + 1)),
                },
            )
        ],
    )


def _canonical_arrays(num_worlds: int, num_dofs: int) -> dict[str, ProxyArray]:
    """Allocate one exact-type canonical block outside the construction census."""
    return {
        name: ProxyArray(wp.zeros((num_worlds, num_dofs), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "effort_limit", "velocity_limit", "computed_effort", "applied_effort")
    }


def _build_case(monkeypatch, *, mode: str, num_worlds: int, num_signatures: int):
    """Build one direct or deliberately ambiguous hosted layout."""
    parsed = []
    groups = {}
    group_layouts = []
    type_layouts = {}
    joint_names = []
    native_group_names = set()
    joint_index = 0
    repetitions = 1 if mode == "direct" else 2
    for signature_id in range(num_signatures):
        for copy_id in range(repetitions):
            joint_name = f"joint_{joint_index}"
            joint_names.append(joint_name)
            parsed.append(_parsed_actuator(joint_name, signature_id))
            actuator_type = type(f"_{mode.title()}Type_{signature_id}_{copy_id}", (), {})
            group_name = f"group_{joint_index}"
            arrays = _canonical_arrays(num_worlds, 1)
            groups[group_name] = SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays))
            group_layouts.append(
                SimpleNamespace(
                    name=group_name,
                    actuator_type=actuator_type,
                    joint_indices=(joint_index,),
                    joint_names=(joint_name,),
                    type_slice=slice(0, 1),
                )
            )
            type_layouts[actuator_type] = SimpleNamespace(
                num_worlds=num_worlds,
                num_dofs=1,
                compact_joint_indices=(joint_index,),
            )
            native_group_names.add(group_name)
            joint_index += 1
    binding = SimpleNamespace(
        groups=groups,
        native_group_names=frozenset(native_group_names),
        computed_effort=ProxyArray(wp.zeros((num_worlds, joint_index), dtype=wp.float32, device="cpu")),
        applied_effort=ProxyArray(wp.zeros((num_worlds, joint_index), dtype=wp.float32, device="cpu")),
        layout=SimpleNamespace(
            num_joints=joint_index,
            type_layouts=type_layouts,
            group_layouts=tuple(group_layouts),
        ),
    )
    return _make_stage(monkeypatch, parsed), joint_names, binding


def _caller_role() -> str:
    """Name the first IsaacLab adapter or Newton actuator construction frame."""
    frame = sys._getframe(2)
    for _ in range(12):
        filename = frame.f_code.co_filename.replace("\\", "/")
        if filename.endswith("isaaclab_newton/actuators/adapter.py"):
            return f"adapter:{frame.f_code.co_name}"
        marker = "/newton/_src/actuators/"
        if marker in filename:
            component = filename.split(marker, 1)[1].rsplit(".", 1)[0]
            return f"newton:{component}:{frame.f_code.co_name}"
        if frame.f_back is None:
            break
        frame = frame.f_back
    return "external"


def _numel(value: Any) -> int:
    """Return the element count of a Torch or Warp array."""
    return int(value.numel()) if isinstance(value, torch.Tensor) else int(value.size)


def _install_build_spies(monkeypatch):  # noqa: C901
    """Install deterministic spies around explicit allocation and construction boundaries."""
    import isaaclab_newton.actuators.adapter as adapter_module

    records = SimpleNamespace(
        allocation_calls=Counter(),
        alias_calls=Counter(),
        host_constructions=Counter(),
        launch_calls=Counter(),
        resolver_calls=Counter(),
        scalar_conversions=Counter(),
        clone_calls=0,
    )
    original_array = wp.array
    original_empty = wp.empty
    original_zeros = wp.zeros
    original_full = wp.full
    original_ones = wp.ones
    original_from_torch = wp.from_torch
    original_clone = wp.clone
    original_launch = wp.launch
    original_torch_tensor = torch.tensor
    original_numpy_arange = np.arange
    original_controller_resolve = ControllerPD.resolve_arguments
    original_clamping_resolve = ClampingPositionBased.resolve_arguments
    original_signature = adapter_module._actuator_signature
    warp_array_type = type(wp.zeros(0, dtype=wp.float32, device="cpu"))
    original_warp_numpy = warp_array_type.numpy
    original_torch_numpy = torch.Tensor.numpy
    original_torch_item = torch.Tensor.item

    def record_allocation(api, function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            records.allocation_calls[(api, _caller_role(), str(result.dtype), tuple(result.shape))] += 1
            return result

        return wrapper

    class _WarpArraySpyMeta(type):
        def __call__(cls, *args, **kwargs):
            del cls
            result = original_array(*args, **kwargs)
            records.allocation_calls[("array", _caller_role(), str(result.dtype), tuple(result.shape))] += 1
            return result

        def __instancecheck__(cls, instance):
            del cls
            return isinstance(instance, original_array)

    class _WarpArraySpy(metaclass=_WarpArraySpyMeta):
        pass

    def from_torch(*args, **kwargs):
        result = original_from_torch(*args, **kwargs)
        records.alias_calls[("from_torch", _caller_role(), str(result.dtype), tuple(result.shape))] += 1
        return result

    def clone(*args, **kwargs):
        records.clone_calls += 1
        return original_clone(*args, **kwargs)

    def launch(kernel, *args, **kwargs):
        kernel_name = getattr(kernel, "key", getattr(kernel, "__name__", type(kernel).__name__)).rsplit(".", 1)[-1]
        records.launch_calls[kernel_name] += 1
        return original_launch(kernel, *args, **kwargs)

    def tensor(*args, **kwargs):
        result = original_torch_tensor(*args, **kwargs)
        records.allocation_calls[("torch.tensor", _caller_role(), str(result.dtype), tuple(result.shape))] += 1
        return result

    def numpy_arange(*args, **kwargs):
        result = original_numpy_arange(*args, **kwargs)
        records.host_constructions[("numpy.arange", _caller_role(), result.size)] += 1
        return result

    def resolve_controller(cls, arguments):
        del cls
        records.resolver_calls["controller"] += 1
        return original_controller_resolve(arguments)

    def resolve_clamping(cls, arguments):
        del cls
        records.resolver_calls["component"] += 1
        return original_clamping_resolve(arguments)

    def signature(*args, **kwargs):
        records.resolver_calls["signature"] += 1
        return original_signature(*args, **kwargs)

    def warp_numpy(array):
        records.scalar_conversions["warp.numpy"] += 1
        return original_warp_numpy(array)

    def torch_numpy(tensor):
        records.scalar_conversions["torch.numpy"] += 1
        return original_torch_numpy(tensor)

    def torch_item(tensor):
        records.scalar_conversions["torch.item"] += 1
        return original_torch_item(tensor)

    monkeypatch.setattr(wp, "array", _WarpArraySpy)
    monkeypatch.setattr(wp, "empty", record_allocation("empty", original_empty))
    monkeypatch.setattr(wp, "zeros", record_allocation("zeros", original_zeros))
    monkeypatch.setattr(wp, "full", record_allocation("full", original_full))
    monkeypatch.setattr(wp, "ones", record_allocation("ones", original_ones))
    monkeypatch.setattr(wp, "from_torch", from_torch)
    monkeypatch.setattr(wp, "clone", clone)
    monkeypatch.setattr(wp, "launch", launch)
    monkeypatch.setattr(torch, "tensor", tensor)
    monkeypatch.setattr(np, "arange", numpy_arange)
    monkeypatch.setattr(ControllerPD, "resolve_arguments", classmethod(resolve_controller))
    monkeypatch.setattr(ClampingPositionBased, "resolve_arguments", classmethod(resolve_clamping))
    monkeypatch.setattr(adapter_module, "_actuator_signature", signature)
    monkeypatch.setattr(warp_array_type, "numpy", warp_numpy)
    monkeypatch.setattr(torch.Tensor, "numpy", torch_numpy)
    monkeypatch.setattr(torch.Tensor, "item", torch_item)
    return records, original_array


def _array_items(component: Any, warp_array_type: type) -> list[tuple[str, Any]]:
    """Return all direct Warp-array attributes retained by one component."""
    return [(name, value) for name, value in vars(component).items() if isinstance(value, warp_array_type)]


def _direct_pointer_log_census(adapter: Any) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Return registrations and every original Warp allocation reachable from the reverse log."""
    registrations = []
    originals = []
    for direct_binding in getattr(adapter, "_direct_pointer_bindings", {}).values():
        registrations.append(len(direct_binding.registrations))
        originals.extend((mutation.original.ptr, mutation.original.capacity) for mutation in direct_binding.mutations)
    return tuple(sorted(registrations)), tuple(sorted(originals))


def _audit_build(monkeypatch, *, mode: str, num_worlds: int, num_signatures: int) -> _BuildAudit:
    """Construct and bind one case under deterministic spies."""
    import isaaclab_newton.actuators.adapter as adapter_module

    stage, joint_names, binding = _build_case(
        monkeypatch,
        mode=mode,
        num_worlds=num_worlds,
        num_signatures=num_signatures,
    )
    with monkeypatch.context() as scoped:
        records, warp_array_type = _install_build_spies(scoped)
        adapter = adapter_module.NewtonActuatorAdapter._from_usd_binding(
            binding,
            stage=stage,
            joint_names=joint_names,
            num_envs=num_worlds,
            num_joints=len(joint_names),
            device="cpu",
            articulation_prim_path="/Robot",
        )
        native_binding = adapter.bind_articulation(binding, dof_offset=0)

    canonical_pointers = {
        array.warp.ptr for group in binding.groups.values() for array in group._parameter_binding.arrays.values()
    }
    live_bytes = Counter()
    for actuator in adapter.actuators:
        live_bytes["physical_indices"] += actuator.indices.capacity
        live_bytes["newton_sequential_indices"] += actuator._sequential_indices.capacity
        for name, array in _array_items(actuator.controller, warp_array_type):
            if array.ptr in canonical_pointers:
                role = "canonical"
            elif name in {"kp", "kd"}:
                role = "staged_parameter"
            else:
                role = "newton_parameter"
            live_bytes[role] += array.capacity
        for clamping in actuator.clamping:
            for name, array in _array_items(clamping, warp_array_type):
                if array.ptr in canonical_pointers:
                    role = "canonical"
                elif name in {"lookup_positions", "lookup_efforts"}:
                    role = "component_lookup"
                else:
                    role = "component_scratch"
                live_bytes[role] += array.capacity
        for name in ("_computed_forces", "_applied_forces"):
            array = getattr(actuator, name)
            if array is not None:
                role = "canonical" if array.ptr in canonical_pointers else "staged_output"
                live_bytes[role] += array.capacity

    descriptor_arrays = {}
    for range_binding in native_binding.ranges:
        for name in (
            "compact_joint_ids",
            "canonical_slots",
            "effort_owner_slots",
            "computed_owner_slots",
            "applied_owner_slots",
            "controller_local_slots",
            "user_to_backend",
        ):
            array = getattr(range_binding, name)
            if array is not None:
                descriptor_arrays[array.ptr] = array.size
    retained_torch_owner_sizes = tuple(
        sorted(
            owner.numel()
            for actuator in adapter.actuators
            for owner in getattr(actuator, "_isaaclab_adapter_torch_owners", ())
        )
    )
    direct_pointer_registration_counts, direct_pointer_original_allocations = _direct_pointer_log_census(adapter)
    return _BuildAudit(
        allocation_calls=records.allocation_calls,
        alias_calls=records.alias_calls,
        host_constructions=records.host_constructions,
        launch_calls=records.launch_calls,
        resolver_calls=records.resolver_calls,
        scalar_conversions=records.scalar_conversions,
        retained_torch_owner_sizes=retained_torch_owner_sizes,
        direct_pointer_registration_counts=direct_pointer_registration_counts,
        direct_pointer_original_allocations=direct_pointer_original_allocations,
        descriptor_sizes=tuple(sorted(descriptor_arrays.values())),
        live_bytes=live_bytes,
        clone_calls=records.clone_calls,
        adapter=adapter,
        native_binding=native_binding,
        binding=binding,
    )


def _call_roles(counter: Counter) -> Counter:
    """Drop shapes from allocation records while retaining explicit API/owner roles."""
    roles = Counter()
    for key, count in counter.items():
        roles[key[:2]] += count
    return roles


def test_native_masks_pack_equivalent_fields_into_three_rows(monkeypatch) -> None:
    """Target fields, force fields, and native-touched each own one uploaded row."""
    from isaaclab_newton.actuators.kernels import _build_native_dof_masks, build_implicit_dof_mask

    from isaaclab.actuators import ImplicitActuator

    class _NativeType:
        pass

    class _LabType:
        pass

    group_layouts = (
        SimpleNamespace(name="native", actuator_type=_NativeType, joint_indices=(0, 1)),
        SimpleNamespace(name="lab", actuator_type=_LabType, joint_indices=(1,)),
    )
    uploads = []
    original_tensor = torch.tensor

    def record_upload(*args, **kwargs):
        result = original_tensor(*args, **kwargs)
        uploads.append(tuple(result.shape))
        return result

    monkeypatch.setattr(torch, "tensor", record_upload)
    masks, owners = _build_native_dof_masks(
        {},
        frozenset({"native"}),
        2,
        "cpu",
        group_layouts=group_layouts,
    )

    assert masks["position"] is masks["velocity"]
    assert masks["effort"] is masks["computed_effort"] is masks["applied_effort"]
    assert masks["touched"].ptr not in {masks["position"].ptr, masks["effort"].ptr}
    assert owners["position"] is owners["velocity"]
    assert owners["effort"] is owners["computed_effort"] is owners["applied_effort"]
    assert masks["position"].numpy().tolist() == [1, 1]
    assert masks["effort"].numpy().tolist() == [1, 0]
    assert masks["touched"].numpy().tolist() == [1, 1]
    assert uploads == [(3, 2)]

    class _ImplicitType(ImplicitActuator):
        pass

    uploads.clear()
    implicit_mask, implicit_owner = build_implicit_dof_mask(
        {},
        2,
        "cpu",
        group_layouts=(SimpleNamespace(name="implicit", actuator_type=_ImplicitType, joint_indices=(1,)),),
    )
    assert uploads == [(2,)]
    assert implicit_mask.numpy().tolist() == [0, 1]
    assert implicit_mask.ptr == wp.from_torch(implicit_owner, dtype=wp.int32).ptr


@pytest.mark.parametrize("mode", ["direct", "staged"])
def test_hosted_build_roles_are_world_invariant(monkeypatch, mode: str) -> None:
    """World count changes bytes only in explicitly allowlisted live Newton arrays."""
    audits = {
        worlds: _audit_build(monkeypatch, mode=mode, num_worlds=worlds, num_signatures=2) for worlds in (1, 2, 4096)
    }
    reference = audits[1]
    for worlds, audit in audits.items():
        assert _call_roles(audit.allocation_calls) == _call_roles(reference.allocation_calls)
        assert audit.alias_calls == reference.alias_calls
        assert _call_roles(audit.host_constructions) == _call_roles(reference.host_constructions)
        assert audit.launch_calls == reference.launch_calls
        assert audit.resolver_calls == reference.resolver_calls
        assert audit.scalar_conversions == Counter()
        assert audit.clone_calls == 0
        assert audit.retained_torch_owner_sizes == reference.retained_torch_owner_sizes
        # Hosted direct construction owns its Newton objects.  A successful
        # alias must therefore commit and drop the reverse log rather than
        # retaining the constructor's throwaway controller/output buffers.
        assert audit.direct_pointer_registration_counts == ()
        assert audit.direct_pointer_original_allocations == ()
        assert audit.descriptor_sizes == reference.descriptor_sizes
        assert max(audit.retained_torch_owner_sizes, default=0) <= 2
        assert max(audit.descriptor_sizes, default=0) <= audit.binding.layout.num_joints
        model_dofs = audit.binding.layout.num_joints
        if worlds > 1:
            assert not any(
                role.startswith("adapter:") and "int32" in dtype and shape and shape[0] == worlds * model_dofs
                for (_, role, dtype, shape) in audit.allocation_calls
            )
        expected_segment_dofs = 1 if mode == "direct" else 2
        assert sum(audit.host_constructions.values()) == len(audit.adapter.actuators)
        assert all(
            api == "numpy.arange" and role == "newton:actuator:__init__" and size == worlds * expected_segment_dofs
            for (api, role, size) in audit.host_constructions
        )
        assert all(
            range_binding.effort_owner_slots.ptr
            == range_binding.computed_owner_slots.ptr
            == range_binding.applied_owner_slots.ptr
            for range_binding in audit.native_binding.ranges
        )
        assert len({range_binding.user_to_backend.ptr for range_binding in audit.native_binding.ranges}) == 1
        descriptor_count = 1 + (3 if mode == "direct" else 8) * 2
        assert len(audit.descriptor_sizes) == descriptor_count

    varying_live_roles = {
        role
        for role in set().union(*(audit.live_bytes for audit in audits.values()))
        if len({audit.live_bytes[role] for audit in audits.values()}) > 1
    }
    allowed = {"physical_indices", "newton_sequential_indices", "canonical", "newton_parameter"}
    if mode == "staged":
        allowed.update({"staged_parameter", "staged_output"})
    assert varying_live_roles <= allowed
    if mode == "direct":
        assert reference.launch_calls["_expand_env_major_values"] == 2
        assert "staged_parameter" not in reference.live_bytes
        assert "staged_output" not in reference.live_bytes
    else:
        assert reference.launch_calls["_expand_env_major_values"] == 6
        assert all(not range_binding.direct for range_binding in reference.native_binding.ranges)
        for worlds, audit in audits.items():
            assert audit.live_bytes["staged_parameter"] == 32 * worlds
            assert audit.live_bytes["newton_parameter"] == 16 * worlds
            assert audit.live_bytes["staged_output"] == 32 * worlds


@pytest.mark.parametrize("mode", ["direct", "staged"])
def test_hosted_build_roles_scale_affinely_with_signatures(monkeypatch, mode: str) -> None:
    """Fixed-world construction work is affine in signatures, not parameter-field cross products."""
    audits = {
        signatures: _audit_build(monkeypatch, mode=mode, num_worlds=2, num_signatures=signatures)
        for signatures in (1, 2, 8)
    }
    parsed_per_signature = 1 if mode == "direct" else 2
    for signatures, audit in audits.items():
        expected_parsed = parsed_per_signature * signatures
        assert audit.resolver_calls == Counter(
            controller=expected_parsed,
            component=expected_parsed,
            signature=expected_parsed,
        )
        assert audit.clone_calls == 0
        assert max(audit.retained_torch_owner_sizes, default=0) <= parsed_per_signature
        assert audit.direct_pointer_registration_counts == ()
        assert audit.direct_pointer_original_allocations == ()
        assert max(audit.descriptor_sizes, default=0) <= audit.binding.layout.num_joints
        expected_descriptor_uploads = 1 + (3 if mode == "direct" else 8) * signatures
        assert len(audit.descriptor_sizes) == expected_descriptor_uploads
        assert len({range_binding.user_to_backend.ptr for range_binding in audit.native_binding.ranges}) == 1
        assert all(
            range_binding.effort_owner_slots.ptr
            == range_binding.computed_owner_slots.ptr
            == range_binding.applied_owner_slots.ptr
            for range_binding in audit.native_binding.ranges
        )

    for extractor in (
        lambda audit: _call_roles(audit.allocation_calls),
        lambda audit: _call_roles(audit.alias_calls),
        lambda audit: _call_roles(audit.host_constructions),
        lambda audit: audit.launch_calls,
    ):
        values = {signatures: extractor(audit) for signatures, audit in audits.items()}
        keys = set().union(*values.values())
        for key in keys:
            per_signature = values[2][key] - values[1][key]
            assert values[8][key] == values[1][key] + 7 * per_signature


def test_native_shared_direct_binding_retains_only_live_originals_until_last_unregister() -> None:
    """A borrowed direct controller keeps exactly its originals for every live registration."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter

    class _NativeType:
        pass

    signature = ("native",)
    originals = {
        "kp": wp.full(2, 1.0, dtype=wp.float32, device="cpu"),
        "kd": wp.full(2, 2.0, dtype=wp.float32, device="cpu"),
        "computed": wp.full(2, 3.0, dtype=wp.float32, device="cpu"),
        "applied": wp.full(2, 4.0, dtype=wp.float32, device="cpu"),
    }
    actuator = SimpleNamespace(
        indices=wp.array([0, 1], dtype=wp.uint32, device="cpu"),
        controller=SimpleNamespace(kp=originals["kp"], kd=originals["kd"]),
        clamping=(),
        _computed_forces=originals["computed"],
        _applied_forces=originals["applied"],
    )
    canonical = {
        name: ProxyArray(wp.zeros((2, 1), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    binding = SimpleNamespace(
        groups={"native": SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=canonical))},
        native_group_names=frozenset({"native"}),
        computed_effort=canonical["computed_effort"],
        layout=SimpleNamespace(
            num_joints=1,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=2, num_dofs=1, compact_joint_indices=(0,))},
            group_layouts=(
                SimpleNamespace(
                    name="native",
                    actuator_type=_NativeType,
                    joint_indices=(0,),
                    joint_names=("joint",),
                    type_slice=slice(0, 1),
                ),
            ),
        ),
    )
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter.actuators = [actuator]
    adapter._device = "cpu"
    adapter.num_joints = 1
    adapter._num_envs = 2
    adapter._global_native_bindings = {}
    adapter._direct_pointer_bindings = {}
    adapter._owns_actuators = False
    adapter._actuators_by_signature = {signature: actuator}
    adapter._joint_signatures = {"joint": signature}
    adapter._dof_signatures = {}
    adapter._actuator_dof_indices = {signature: (0,)}

    first = adapter.bind_articulation(binding, dof_offset=0)
    expected_originals = tuple(sorted((array.ptr, array.capacity) for array in originals.values()))
    assert _direct_pointer_log_census(adapter) == ((1,), expected_originals)
    assert actuator.controller.kp.ptr == canonical["stiffness"].warp.ptr
    assert actuator.controller.kd.ptr == canonical["damping"].warp.ptr
    assert actuator._computed_forces.ptr == canonical["computed_effort"].warp.ptr
    assert actuator._applied_forces.ptr == canonical["applied_effort"].warp.ptr

    second = adapter.bind_articulation(binding, dof_offset=0)
    assert _direct_pointer_log_census(adapter) == ((2,), expected_originals)

    adapter.unregister_articulation_ranges(first.ranges)
    assert _direct_pointer_log_census(adapter) == ((1,), expected_originals)
    assert actuator.controller.kp.ptr == canonical["stiffness"].warp.ptr

    adapter.unregister_articulation_ranges(second.ranges)
    assert _direct_pointer_log_census(adapter) == ((), ())
    assert actuator.controller.kp is originals["kp"]
    assert actuator.controller.kd is originals["kd"]
    assert actuator._computed_forces is originals["computed"]
    assert actuator._applied_forces is originals["applied"]


def test_warm_staged_actuator_path_has_no_allocation_or_host_conversion(monkeypatch) -> None:
    """Warm gather, actuator step, and publish use only cached launches and persistent arrays."""
    import isaaclab_newton.actuators.adapter as adapter_module

    audit = _audit_build(monkeypatch, mode="staged", num_worlds=2, num_signatures=2)
    adapter = audit.adapter
    ranges = audit.native_binding.ranges
    num_dofs = audit.binding.layout.num_joints
    state = SimpleNamespace(
        joint_q=wp.zeros(2 * num_dofs, dtype=wp.float32, device="cpu"),
        joint_qd=wp.zeros(2 * num_dofs, dtype=wp.float32, device="cpu"),
    )
    control = SimpleNamespace(
        joint_target_pos=wp.zeros(2 * num_dofs, dtype=wp.float32, device="cpu"),
        joint_target_vel=wp.zeros(2 * num_dofs, dtype=wp.float32, device="cpu"),
        joint_act=wp.zeros(2 * num_dofs, dtype=wp.float32, device="cpu"),
        joint_f=wp.zeros(2 * num_dofs, dtype=wp.float32, device="cpu"),
    )
    joint_computed = wp.zeros((2, num_dofs), dtype=wp.float32, device="cpu")
    joint_applied = wp.zeros((2, num_dofs), dtype=wp.float32, device="cpu")

    def run_warm_path() -> None:
        adapter.gather_staged_ranges(ranges)
        adapter.step(state, control, dt=0.01)
        adapter.publish_outputs(ranges, joint_computed, joint_applied)

    run_warm_path()

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("warm actuator path attempted allocation or host/scalar conversion")

    warp_array_type = type(control.joint_f)
    for name in ("array", "empty", "zeros", "full", "ones", "clone", "from_torch", "to_torch"):
        monkeypatch.setattr(wp, name, forbidden)
    monkeypatch.setattr(np, "arange", forbidden)
    monkeypatch.setattr(warp_array_type, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(adapter_module, "_NativeRangeBinding", forbidden)

    run_warm_path()
