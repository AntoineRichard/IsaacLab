# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scoped actuator group and exact-type facade views."""

from __future__ import annotations

import dis
import gc
import warnings
import weakref
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.actuators import ActuatorCollection, actuator_collection, actuator_kernels
from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_control import ActuatorJointProperties
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.actuators.actuator_storage import _ACTUATOR_NET_MLP_SCHEMA
from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray


def _available_devices() -> tuple[str, ...]:
    return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


class _CustomIdealPD(IdealPDActuator):
    """Opaque custom actuator that deliberately has no exact managed schema."""


class _NeuralLikeDC(DCMotor):
    """DC motor shell that declares the neural parameter contract for capability tests."""

    @classmethod
    def _parameter_schema(cls):
        """Return a schema with no meaningful stiffness or damping parameters."""
        return _ACTUATOR_NET_MLP_SCHEMA


class _Simulation:
    def __init__(self, device: str = "cpu", num_worlds: int = 2) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, num_worlds), dtype=torch.bool, device=device),
            cfg_rows={1: (0,)},
        )

    def get_clone_plan(self) -> ClonePlan:
        return self._clone_plan


class _Control:
    def __init__(self, device: str = "cpu", num_worlds: int = 2) -> None:
        self._joint_names = ("hip", "knee", "ankle")
        self._device = device
        self._num_worlds = num_worlds
        self._joint_pos = ProxyArray(wp.zeros((num_worlds, len(self._joint_names)), dtype=wp.float32, device=device))
        self._joint_vel = ProxyArray(wp.zeros((num_worlds, len(self._joint_names)), dtype=wp.float32, device=device))
        self.parameter_writes = []

    @property
    def num_instances(self) -> int:
        return self._num_worlds

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def device(self) -> str:
        return self._device

    @property
    def joint_pos(self) -> ProxyArray:
        return self._joint_pos

    @property
    def joint_vel(self) -> ProxyArray:
        return self._joint_vel

    def discover_native_actuators(self, cfgs) -> set[str]:
        del cfgs
        return set()

    def find_joints(self, names):
        requested = set(names)
        indices = [index for index, name in enumerate(self._joint_names) if name in requested]
        return indices, [self._joint_names[index] for index in indices]

    def get_default_joint_properties(self, joint_ids):
        count = self.num_joints if isinstance(joint_ids, slice) else len(joint_ids)
        values = torch.zeros((self.num_instances, count), device=self.device)
        return ActuatorJointProperties(
            stiffness=values,
            damping=values,
            armature=values,
            friction=values,
            dynamic_friction=values,
            viscous_friction=values,
            effort_limit=torch.full_like(values, torch.inf),
            velocity_limit=torch.full_like(values, torch.inf),
        )

    def prepare_actuator_binding(self, binding) -> None:
        del binding

    def bind_actuator_view(self, view) -> None:
        del view

    def complete_articulation_initialization(self) -> None:
        pass

    def invalidate_actuator_view(self) -> None:
        pass

    def write_actuator_parameter(self, name, write) -> None:
        self.parameter_writes.append((name, write))


def _ideal_cfg(joint_names: list[str]) -> IdealPDActuatorCfg:
    cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=1.0,
        damping=2.0,
        effort_limit=3.0,
        velocity_limit=4.0,
    )
    cfg.class_type = IdealPDActuator
    return cfg


def _custom_cfg(joint_names: list[str]) -> IdealPDActuatorCfg:
    cfg = _ideal_cfg(joint_names)
    cfg.class_type = _CustomIdealPD
    return cfg


def _dc_cfg(joint_names: list[str]) -> DCMotorCfg:
    cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=1.0,
        damping=2.0,
        effort_limit=3.0,
        velocity_limit=4.0,
        saturation_effort=5.0,
    )
    cfg.class_type = DCMotor
    return cfg


def _neural_like_cfg(joint_names: list[str]) -> DCMotorCfg:
    cfg = _dc_cfg(joint_names)
    cfg.class_type = _NeuralLikeDC
    return cfg


def _implicit_cfg(joint_names: list[str]) -> ImplicitActuatorCfg:
    cfg = ImplicitActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=1.0,
        damping=2.0,
        effort_limit=3.0,
        velocity_limit=4.0,
    )
    cfg.class_type = ImplicitActuator
    return cfg


def make_finalized_robot(*, groups=None, device: str = "cpu", debug_validation: bool = False):
    collection = ActuatorCollection(_Simulation(device))
    cfgs = groups or {"hip": _ideal_cfg(["hip", "knee"]), "knee": _dc_cfg(["knee"]), "ankle": _ideal_cfg(["ankle"])}
    view = collection.register_articulation(
        key="robot",
        cfgs=cfgs,
        control=_Control(device),
        replication_cfg_id=1,
        debug_validation=debug_validation,
        debug_value_resolution=False,
    )
    collection.finalize()
    return SimpleNamespace(collection=collection, actuators=view)


@pytest.mark.parametrize("device", _available_devices())
def test_compatibility_projections_are_lazy_stable_and_refresh_after_parameter_write(device: str) -> None:
    """Legacy dense projections allocate only on access and retain their pointer."""
    robot = make_finalized_robot(device=device)

    assert robot.actuators._compatibility_allocations == {}
    assert robot.collection._active_generation.joint_store._compatibility_projections == {}
    robot.actuators.compute()
    assert robot.actuators._compatibility_allocations == {}
    assert robot.collection._active_generation.joint_store._compatibility_projections == {}
    velocity_limits = robot.actuators._get_compatibility_projection("soft_joint_vel_limits")
    gear_ratio = robot.actuators._get_compatibility_projection("gear_ratio")

    assert velocity_limits.torch.device.type == torch.device(device).type
    assert gear_ratio.torch.device.type == torch.device(device).type
    assert robot.actuators._compatibility_allocations == {
        "soft_joint_vel_limits": velocity_limits,
        "gear_ratio": gear_ratio,
    }
    assert (
        velocity_limits.torch.data_ptr()
        == robot.actuators._get_compatibility_projection("soft_joint_vel_limits").torch.data_ptr()
    )
    assert gear_ratio.torch.data_ptr() == robot.actuators._get_compatibility_projection("gear_ratio").torch.data_ptr()
    torch.testing.assert_close(velocity_limits.torch, torch.full_like(velocity_limits.torch, 4.0))
    torch.testing.assert_close(gear_ratio.torch, torch.ones_like(gear_ratio.torch))

    robot.actuators["hip"].set_parameter_index(
        "velocity_limit",
        torch.full((2, 2), 9.0, device=device),
        joint_ids=torch.tensor([0, 1], device=device),
    )

    torch.testing.assert_close(velocity_limits.torch[:, 0], torch.full((2,), 9.0, device=device))
    torch.testing.assert_close(velocity_limits.torch[:, 1], torch.full((2,), 4.0, device=device))


@pytest.mark.parametrize("device", _available_devices())
def test_deprecated_gain_writer_routes_only_capable_exact_type_views_and_warns_once(device: str) -> None:
    """The compatibility writer uses canonical type setters without sidecars."""
    robot = make_finalized_robot(device=device)
    values = torch.tensor([[11.0, 13.0, 17.0]], device=device)
    env_ids = torch.tensor([0], device=device)
    joint_ids = torch.tensor([0, 1, 2], device=device)

    with pytest.warns(DeprecationWarning, match="set_parameter_index"):
        robot.actuators.write_actuator_stiffness_to_sim(stiffness=values, env_ids=env_ids, joint_ids=joint_ids)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        robot.actuators.write_actuator_stiffness_to_sim(stiffness=values, env_ids=env_ids, joint_ids=joint_ids)

    assert caught == []
    assert robot.actuators["hip"].stiffness.device.type == torch.device(device).type
    torch.testing.assert_close(robot.actuators["hip"].stiffness[:1], values[:, :2])
    torch.testing.assert_close(robot.actuators["ankle"].stiffness[:1], values[:, 2:])


@pytest.mark.parametrize("device", _available_devices())
def test_deprecated_gain_writer_skips_neural_capability_sidecars(device: str) -> None:
    """A neural-style exact type is skipped without materializing meaningless gains."""
    robot = make_finalized_robot(
        groups={"hip": _ideal_cfg(["hip"]), "neural": _neural_like_cfg(["knee", "ankle"])}, device=device
    )

    with pytest.warns(DeprecationWarning, match="set_parameter_index"):
        robot.actuators.write_actuator_stiffness_to_sim(
            stiffness=torch.tensor([[11.0, 13.0, 17.0]], device=device),
            env_ids=torch.tensor([0], device=device),
            joint_ids=torch.tensor([0, 1, 2], device=device),
        )

    assert robot.actuators["neural"]._deprecated_sidecars == {}
    assert "stiffness" not in robot.actuators.by_type[_NeuralLikeDC].parameter_names


@pytest.mark.parametrize("device", _available_devices())
def test_neural_gain_sidecar_uses_the_facade_warning_registry(device: str) -> None:
    """Neural gain compatibility warns once per published articulation facade."""
    robot = make_finalized_robot(groups={"neural": _neural_like_cfg(["hip", "knee", "ankle"])}, device=device)

    with pytest.warns(DeprecationWarning, match="neural-actuator compatibility sidecar"):
        sidecar = robot.actuators["neural"].stiffness
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert robot.actuators["neural"].stiffness.data_ptr() == sidecar.data_ptr()

    assert caught == []
    assert sidecar.device.type == torch.device(device).type


@pytest.mark.parametrize("device", _available_devices())
def test_solver_sidecar_refresh_updates_only_an_already_materialized_field(device: str) -> None:
    """Solver writer hooks can refresh held sidecars without allocating untouched fields."""
    robot = make_finalized_robot(device=device)
    hip = robot.actuators["hip"]
    armature = hip.armature
    solver_sidecars = hip._solver_compatibility_sidecars
    assert set(solver_sidecars) == {"armature"}

    values = torch.full((2, 3), 8.0, device=device)
    hip._refresh_solver_compatibility_sidecar("armature", values)
    hip._refresh_solver_compatibility_sidecar("friction", values)

    assert armature.device.type == torch.device(device).type
    torch.testing.assert_close(armature, torch.full_like(armature, 8.0))
    assert set(solver_sidecars) == {"armature"}


@pytest.mark.parametrize("device", _available_devices())
def test_solver_sidecar_presence_check_avoids_inactive_temporary_containers(device: str) -> None:
    """Check held-sidecar presence without constructing a temporary container."""
    robot = make_finalized_robot(device=device)
    instructions = {
        instruction.opname
        for instruction in dis.get_instructions(ActuatorCollection.ArticulationView._has_solver_compatibility_sidecar)
    }

    assert "BUILD_LIST" not in instructions
    assert "BUILD_TUPLE" not in instructions
    assert "BUILD_MAP" not in instructions
    assert "MAKE_FUNCTION" not in instructions
    assert not robot.actuators._has_solver_compatibility_sidecar("armature")
    assert robot.actuators["hip"]._solver_compatibility_sidecars == {}
    assert robot.actuators["ankle"]._solver_compatibility_sidecars == {}


@pytest.mark.parametrize("device", _available_devices())
def test_compatibility_projections_use_legacy_fills_and_refresh_held_values_at_compute(device: str) -> None:
    """Unsupported joints retain fills while direct parameter mutation waits for compute."""
    robot = make_finalized_robot(
        groups={"hip": _ideal_cfg(["hip", "knee"]), "ankle": _custom_cfg(["ankle"])}, device=device
    )

    assert robot.actuators._compatibility_allocations == {}
    velocity_limits = robot.actuators._get_compatibility_projection("soft_joint_vel_limits")
    gear_ratio = robot.actuators._get_compatibility_projection("gear_ratio")
    torch.testing.assert_close(velocity_limits.torch, torch.tensor([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0]], device=device))
    torch.testing.assert_close(gear_ratio.torch, torch.ones_like(gear_ratio.torch))

    robot.actuators["hip"].velocity_limit.fill_(12.0)
    assert velocity_limits.torch[0, 0].item() == 4.0

    robot.actuators.compute()

    assert (
        velocity_limits.torch.data_ptr()
        == robot.actuators._get_compatibility_projection("soft_joint_vel_limits").torch.data_ptr()
    )
    torch.testing.assert_close(
        velocity_limits.torch, torch.tensor([[12.0, 12.0, 0.0], [12.0, 12.0, 0.0]], device=device)
    )


@pytest.mark.parametrize("device", _available_devices())
def test_compatibility_projection_fails_through_a_stale_facade(device: str) -> None:
    """A retained facade cannot allocate or refresh compatibility storage after STOP."""
    robot = make_finalized_robot(device=device)
    robot.actuators._get_compatibility_projection("soft_joint_vel_limits")

    robot.collection.clear_generation()

    with pytest.raises(RuntimeError, match="stale actuator view"):
        robot.actuators._get_compatibility_projection("soft_joint_vel_limits")


@pytest.mark.parametrize("device", _available_devices())
def test_base_data_compatibility_projection_helper_is_warning_free(device: str) -> None:
    """First-party consumers can retain dense compatibility reads without deprecation warnings."""
    robot = make_finalized_robot(device=device)
    data = SimpleNamespace(_actuator_view=robot.actuators)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projection = BaseArticulationData._get_actuator_compatibility_projection(data, "soft_joint_vel_limits")

    assert caught == []
    assert (
        projection.torch.data_ptr()
        == robot.actuators._get_compatibility_projection("soft_joint_vel_limits").torch.data_ptr()
    )
    assert projection.torch.device.type == torch.device(device).type


def test_facade_preserves_mapping_and_exact_group_classes() -> None:
    robot = make_finalized_robot(
        groups={"hip": _ideal_cfg(["hip"]), "knee": _dc_cfg(["knee"]), "ankle": _ideal_cfg(["ankle"])}
    )

    assert list(robot.actuators) == ["hip", "knee", "ankle"]
    assert isinstance(robot.actuators["hip"], IdealPDActuator)
    assert type(robot.actuators["knee"]) is DCMotor
    assert list(robot.actuators.keys()) == ["hip", "knee", "ankle"]
    assert len(robot.actuators.items()) == 3
    assert isinstance(robot.actuators, dict)
    assert type(robot.actuators) is not dict
    assert type(robot.actuators.copy()) is dict


def test_facade_preserves_dict_copy_union_reverse_and_fromkeys_behavior() -> None:
    actuators = make_finalized_robot().actuators
    extra = object()

    assert actuators.copy() == dict(actuators)
    assert actuators | {"extra": extra} == dict(actuators) | {"extra": extra}
    assert {"extra": extra} | actuators == {"extra": extra} | dict(actuators)
    assert list(reversed(actuators)) == list(reversed(dict(actuators)))
    assert list(reversed(actuators.items())) == list(reversed(dict(actuators).items()))
    assert list(reversed(actuators.values())) == list(reversed(dict(actuators).values()))
    assert repr(actuators.keys()) == repr(dict(actuators).keys())
    assert repr(actuators.items()) == repr(dict(actuators).items())
    assert repr(actuators.values()) == repr(dict(actuators).values())
    assert dict(actuators.keys().mapping) == dict(actuators)
    assert dict(actuators.items().mapping) == dict(actuators)
    assert dict(actuators.values().mapping) == dict(actuators)
    with pytest.raises(TypeError):
        actuators.keys().mapping["extra"] = extra
    assert type(actuators.fromkeys(("a", "b"), 1)) is dict
    assert type(ActuatorCollection.ArticulationView.fromkeys(("a", "b"), 1)) is dict


def test_by_type_uses_exact_classes_and_returns_compact_contiguous_views() -> None:
    robot = make_finalized_robot()
    pd = robot.actuators.by_type[IdealPDActuator]
    dc = robot.actuators.by_type[DCMotor]

    assert pd.joint_names == ("hip", "knee", "ankle")
    assert pd.joint_indices.tolist() == [0, 1, 2]
    assert pd.parameters["stiffness"].torch.is_contiguous()
    assert "saturation_effort" not in pd.parameter_names
    assert "saturation_effort" in dc.parameter_names
    with pytest.raises(KeyError):
        _ = robot.actuators.by_type[object]
    with pytest.raises(KeyError):
        _ = robot.actuators.by_type["ideal_pd"]


def test_type_view_exposes_group_slices_without_dense_projection() -> None:
    view = make_finalized_robot().actuators.by_type[IdealPDActuator]

    assert view.group_slices == {"hip": slice(0, 2), "ankle": slice(2, 3)}
    assert view.parameters["damping"].torch.shape == (view.num_instances, 3)
    assert not hasattr(view, "compact")


def test_type_view_preserves_repeated_dofs_in_configuration_order() -> None:
    view = make_finalized_robot(
        groups={"first": _ideal_cfg(["hip", "knee"]), "second": _ideal_cfg(["knee", "ankle"])}
    ).actuators.by_type[IdealPDActuator]

    assert view.joint_names == ("hip", "knee", "knee", "ankle")
    assert view.joint_indices.tolist() == [0, 1, 1, 2]
    assert view.group_slices == {"first": slice(0, 2), "second": slice(2, 4)}
    assert view.parameters["stiffness"].torch.shape == (2, 4)


def test_group_views_expose_articulation_joint_metadata_and_proxy_parameters() -> None:
    group = make_finalized_robot().actuators["hip"]

    assert group.joint_names == ["hip", "knee"]
    assert group.joint_indices.tolist() == [0, 1]
    assert group.parameters["stiffness"].torch.stride() == (3, 1)
    with pytest.raises(TypeError):
        group.parameters["stiffness"] = group.parameters["damping"]


def test_group_and_type_parameter_lookups_are_stable_bidirectional_aliases() -> None:
    actuators = make_finalized_robot().actuators
    group = actuators["hip"]
    group_parameters = group.parameters
    type_parameters = actuators.by_type[IdealPDActuator].parameters

    assert group.parameters is group_parameters
    assert group_parameters["stiffness"] is group_parameters["stiffness"]
    assert type_parameters["stiffness"] is type_parameters["stiffness"]
    group.stiffness.fill_(7.0)
    torch.testing.assert_close(group_parameters["stiffness"].torch, torch.full((2, 2), 7.0))
    group_parameters["stiffness"].torch.fill_(9.0)
    torch.testing.assert_close(group.stiffness, torch.full((2, 2), 9.0))


def test_all_joint_group_preserves_slice_none_metadata() -> None:
    group = make_finalized_robot(groups={"all": _ideal_cfg(["hip", "knee", "ankle"])}).actuators["all"]

    assert group.joint_indices == slice(None)


def test_multiple_articulations_publish_isolated_views_and_arrays() -> None:
    collection = ActuatorCollection(_Simulation())
    first = collection.register_articulation(
        key="first",
        cfgs={"hip": _ideal_cfg(["hip"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    second = collection.register_articulation(
        key="second",
        cfgs={"ankle": _ideal_cfg(["ankle"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )

    collection.finalize()

    assert list(first) == ["hip"]
    assert list(second) == ["ankle"]
    assert first.by_type[IdealPDActuator] is not second.by_type[IdealPDActuator]
    assert (
        first.by_type[IdealPDActuator].parameters["stiffness"].torch.data_ptr()
        != second.by_type[IdealPDActuator].parameters["stiffness"].torch.data_ptr()
    )
    first.by_type[IdealPDActuator].parameters["stiffness"].torch.fill_(11.0)
    torch.testing.assert_close(second.by_type[IdealPDActuator].parameters["stiffness"].torch, torch.ones((2, 1)))


def test_unknown_groups_and_opaque_exact_classes_are_not_typed() -> None:
    robot = make_finalized_robot(groups={"managed": _ideal_cfg(["hip"]), "opaque": _custom_cfg(["ankle"])})

    assert type(robot.actuators["opaque"]) is _CustomIdealPD
    with pytest.raises(KeyError, match="missing"):
        _ = robot.actuators["missing"]
    with pytest.raises(KeyError, match="DCMotor"):
        _ = robot.actuators.by_type[DCMotor]
    with pytest.raises(KeyError, match="_CustomIdealPD"):
        _ = robot.actuators.by_type[_CustomIdealPD]
    with pytest.raises(KeyError, match="ActuatorBase"):
        _ = robot.actuators.by_type[ActuatorBase]


def test_two_opaque_groups_remain_independent_executable_exact_objects() -> None:
    robot = make_finalized_robot(groups={"first": _custom_cfg(["hip"]), "second": _custom_cfg(["ankle"])})

    def make_action() -> ArticulationActions:
        return ArticulationActions(
            joint_positions=torch.ones((2, 1)),
            joint_velocities=torch.zeros((2, 1)),
            joint_efforts=torch.zeros((2, 1)),
        )

    first = robot.actuators["first"]
    second = robot.actuators["second"]
    assert type(first) is type(second) is _CustomIdealPD
    assert first is not second
    first_action = make_action()
    second_action = make_action()
    assert first.compute(first_action, torch.zeros((2, 1)), torch.zeros((2, 1))) is first_action
    assert second.compute(second_action, torch.zeros((2, 1)), torch.zeros((2, 1))) is second_action


def test_candidate_reuses_the_single_resolved_joint_name_snapshot() -> None:
    class _StatefulControl(_Control):
        def __init__(self) -> None:
            super().__init__()
            self.find_count = 0

        def find_joints(self, names):
            self.find_count += 1
            indices, joint_names = super().find_joints(names)
            if self.find_count > 1:
                joint_names = [f"changed_{name}" for name in joint_names]
            return indices, joint_names

    collection = ActuatorCollection(_Simulation())
    control = _StatefulControl()
    view = collection.register_articulation(
        key="robot",
        cfgs={"hip": _ideal_cfg(["hip"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )

    collection.finalize()

    assert control.find_count == 1
    assert view["hip"].joint_names == ["hip"]
    assert view.by_type[IdealPDActuator].joint_names == ("hip",)


@pytest.mark.parametrize(
    "operation",
    ["setitem", "delitem", "clear", "pop", "popitem", "setdefault", "update", "ior"],
)
def test_every_dict_mutator_stages_copied_cfgs_warns_once_and_dirties(operation: str) -> None:
    robot = make_finalized_robot()
    actuators = robot.actuators
    hip = actuators["hip"]
    expected_result = None

    with pytest.warns(DeprecationWarning, match="Mutating Articulation.actuators") as caught:
        if operation == "setitem":
            result = actuators.__setitem__("extra", hip)
        elif operation == "delitem":
            result = actuators.__delitem__("knee")
        elif operation == "clear":
            result = actuators.clear()
        elif operation == "pop":
            expected_result = actuators["knee"]
            result = actuators.pop("knee")
        elif operation == "popitem":
            expected_result = ("ankle", actuators["ankle"])
            result = actuators.popitem()
        elif operation == "setdefault":
            expected_result = hip
            result = actuators.setdefault("extra", hip)
        elif operation == "update":
            result = actuators.update({"extra": hip})
        else:
            result = actuators.__ior__({"extra": hip})
            expected_result = actuators

    assert len(caught) == 1
    if operation == "popitem":
        assert result == expected_result
    else:
        assert result is expected_result
    assert robot.collection._dirty
    staged = robot.collection._deprecated_staged_topology_overrides["robot"]
    expected_keys = {
        "setitem": ["hip", "knee", "ankle", "extra"],
        "delitem": ["hip", "ankle"],
        "clear": [],
        "pop": ["hip", "ankle"],
        "popitem": ["hip", "knee"],
        "setdefault": ["hip", "knee", "ankle", "extra"],
        "update": ["hip", "knee", "ankle", "extra"],
        "ior": ["hip", "knee", "ankle", "extra"],
    }[operation]
    assert list(staged) == expected_keys
    assert all(staged[name] is not dict.__getitem__(actuators, name).__dict__["cfg"] for name in expected_keys)
    assert list(actuators) == expected_keys
    assert list(actuators.keys()) == expected_keys

    robot.collection.clear_generation()
    replayed = robot.collection.register_articulation(
        key="robot",
        cfgs={"ignored": _dc_cfg(["knee"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    robot.collection.finalize()
    assert list(replayed) == expected_keys
    assert robot.collection._deprecated_staged_topology_overrides == {}


@pytest.mark.parametrize("operation", ["pop", "setdefault", "update", "ior", "clear"])
def test_noop_dict_mutators_preserve_ready_state_without_warning(operation: str) -> None:
    if operation == "clear":
        collection = ActuatorCollection(_Simulation())
        actuators = collection.register_articulation(
            key="robot",
            cfgs={},
            control=_Control(),
            replication_cfg_id=1,
            debug_validation=False,
            debug_value_resolution=False,
        )
        collection.finalize()
    else:
        robot = make_finalized_robot()
        collection = robot.collection
        actuators = robot.actuators

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if operation == "pop":
            marker = object()
            assert actuators.pop("missing", marker) is marker
        elif operation == "setdefault":
            assert actuators.setdefault("hip", actuators["hip"]) is actuators["hip"]
        elif operation == "update":
            assert actuators.update({}) is None
        elif operation == "ior":
            assert actuators.__ior__({}) is actuators
        else:
            assert actuators.clear() is None

    assert caught == []
    assert collection.is_finalized
    assert collection._deprecated_staged_topology_overrides == {}


def test_invalid_or_uncopyable_mutation_leaves_facade_order_and_manager_state_unchanged() -> None:
    robot = make_finalized_robot()
    actuators = robot.actuators
    original_items = list(actuators.items())

    with warnings.catch_warnings(record=True) as invalid_warnings:
        warnings.simplefilter("always")
        with pytest.raises(TypeError, match="ActuatorBase"):
            actuators["invalid"] = object()
    assert invalid_warnings == []
    assert list(actuators.items()) == original_items
    assert robot.collection.is_finalized

    class _UncopyableCfg:
        def copy(self):
            raise RuntimeError("cannot copy config")

    original_cfg = original_items[0][1].__dict__["cfg"]
    original_items[0][1].__dict__["cfg"] = _UncopyableCfg()
    try:
        with warnings.catch_warnings(record=True) as copy_warnings:
            warnings.simplefilter("always")
            with pytest.raises(RuntimeError, match="cannot copy config"):
                actuators["extra"] = original_items[0][1]
        assert copy_warnings == []
        assert list(actuators.items()) == original_items
        assert robot.collection.is_finalized
        assert robot.collection._deprecated_staged_topology_overrides == {}
    finally:
        original_items[0][1].__dict__["cfg"] = original_cfg


def test_topology_mutation_warning_is_once_only_and_staged_override_replays_after_stop() -> None:
    robot = make_finalized_robot()
    old = robot.actuators
    hip = old["hip"]
    hip_cfg = hip.cfg

    with pytest.warns(DeprecationWarning) as caught:
        old["extra"] = hip
        old.update({"other": hip})
    assert len(caught) == 1

    robot.collection.clear_generation()
    replacement = robot.collection.register_articulation(
        key="robot",
        cfgs={"ignored": _dc_cfg(["knee"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    robot.collection.finalize()

    assert list(replacement) == ["hip", "knee", "ankle", "extra", "other"]
    assert robot.collection._deprecated_staged_topology_overrides == {}
    assert replacement is not old
    assert replacement["extra"] is not hip
    assert replacement["extra"].cfg is not hip_cfg


def test_staged_override_replay_is_scoped_to_the_mutated_articulation_key() -> None:
    collection = ActuatorCollection(_Simulation())
    first = collection.register_articulation(
        key="first",
        cfgs={"hip": _ideal_cfg(["hip"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.register_articulation(
        key="second",
        cfgs={"knee": _dc_cfg(["knee"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()

    with pytest.warns(DeprecationWarning):
        first["extra"] = first["hip"]
    assert set(collection._deprecated_staged_topology_overrides) == {"first"}

    collection.clear_generation()
    replayed_first = collection.register_articulation(
        key="first",
        cfgs={"ignored": _dc_cfg(["knee"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    replayed_second = collection.register_articulation(
        key="second",
        cfgs={"ankle": _ideal_cfg(["ankle"])},
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()

    assert list(replayed_first) == ["hip", "extra"]
    assert list(replayed_second) == ["ankle"]


def test_dirty_topology_keeps_current_generation_reads_visible_but_blocks_writes_and_execution() -> None:
    robot = make_finalized_robot()
    facade = robot.actuators
    group = facade["hip"]
    type_view = facade.by_type[IdealPDActuator]
    live_keys = facade.keys()
    live_items = facade.items()
    live_values = facade.values()
    replacement_stiffness = group.stiffness.clone()
    replacement_cfg = group.cfg.copy()

    with pytest.warns(DeprecationWarning):
        facade["extra"] = group

    assert list(facade) == ["hip", "knee", "ankle", "extra"]
    assert facade["extra"] is group
    assert list(live_keys) == ["hip", "knee", "ankle", "extra"]
    assert list(live_items)[-1] == ("extra", group)
    assert list(live_values)[-1] is group
    assert live_keys & {"hip", "missing"} == {"hip"}
    assert list(reversed(live_keys)) == ["extra", "ankle", "knee", "hip"]
    assert group.joint_names == ["hip", "knee"]
    assert group.cfg is not None
    assert group.parameters["stiffness"].torch.shape == (2, 2)
    assert type_view.joint_names == ("hip", "knee", "ankle")
    assert type_view.parameters["stiffness"].torch.shape == (2, 3)
    assert facade.by_type[IdealPDActuator] is type_view

    with pytest.raises(RuntimeError, match="rebuild"):
        facade.compute()
    with pytest.raises(RuntimeError, match="rebuild"):
        _ = facade.command
    with pytest.raises(RuntimeError, match="rebuild"):
        _ = facade.joint_command
    with pytest.raises(RuntimeError, match="rebuild"):
        group.stiffness = replacement_stiffness
    with pytest.raises(RuntimeError, match="rebuild"):
        group.cfg = replacement_cfg
    with pytest.raises(RuntimeError, match="rebuild"):
        group.custom_state = 1


def test_retained_facade_iterators_and_views_recheck_generation_when_consumed() -> None:
    robot = make_finalized_robot()
    facade = robot.actuators
    iterator = iter(facade)
    reverse_iterator = reversed(facade)
    keys = facade.keys()
    items = facade.items()
    values = facade.values()
    key_iterator = iter(keys)
    item_iterator = iter(items)
    value_iterator = iter(values)
    reverse_key_iterator = reversed(keys)
    reverse_item_iterator = reversed(items)
    reverse_value_iterator = reversed(values)
    key_mapping = keys.mapping
    item_mapping = items.mapping
    value_mapping = values.mapping
    key_mapping_iterator = iter(key_mapping)
    item_mapping_iterator = iter(item_mapping)
    value_mapping_iterator = iter(value_mapping)

    robot.collection.clear_generation()

    operations = (
        lambda: next(iterator),
        lambda: next(reverse_iterator),
        lambda: list(keys),
        lambda: len(keys),
        lambda: "hip" in keys,
        lambda: repr(keys),
        lambda: keys.isdisjoint(()),
        lambda: keys & set(),
        lambda: set() & keys,
        lambda: list(items),
        lambda: len(items),
        lambda: ("hip", object()) in items,
        lambda: repr(items),
        lambda: items.isdisjoint(()),
        lambda: items & set(),
        lambda: set() & items,
        lambda: list(values),
        lambda: len(values),
        lambda: object() in values,
        lambda: repr(values),
        lambda: next(key_iterator),
        lambda: next(item_iterator),
        lambda: next(value_iterator),
        lambda: next(reverse_key_iterator),
        lambda: next(reverse_item_iterator),
        lambda: next(reverse_value_iterator),
        lambda: list(key_mapping),
        lambda: list(item_mapping),
        lambda: list(value_mapping),
        lambda: next(key_mapping_iterator),
        lambda: next(item_mapping_iterator),
        lambda: next(value_mapping_iterator),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="stale actuator view"):
            operation()


def test_retained_readonly_mapping_iterators_and_views_recheck_generation_when_consumed() -> None:
    robot = make_finalized_robot()
    facade = robot.actuators
    group = facade["hip"]
    mappings = (
        facade.by_type,
        group.parameters,
        facade.by_type[IdealPDActuator].parameters,
    )
    operations = []
    for mapping in mappings:
        iterator = iter(mapping)
        keys = mapping.keys()
        items = mapping.items()
        values = mapping.values()
        key_iterator = iter(keys)
        item_iterator = iter(items)
        value_iterator = iter(values)
        assert list(keys) == list(mapping)
        assert list(items) == [(key, mapping[key]) for key in mapping]
        assert list(values) == [mapping[key] for key in mapping]
        operations.extend(
            (
                lambda iterator=iterator: next(iterator),
                lambda keys=keys: list(keys),
                lambda keys=keys: len(keys),
                lambda keys=keys: repr(keys),
                lambda items=items: list(items),
                lambda items=items: len(items),
                lambda items=items: repr(items),
                lambda values=values: list(values),
                lambda values=values: len(values),
                lambda values=values: repr(values),
                lambda key_iterator=key_iterator: next(key_iterator),
                lambda item_iterator=item_iterator: next(item_iterator),
                lambda value_iterator=value_iterator: next(value_iterator),
            )
        )

    robot.collection.clear_generation()

    for operation in operations:
        with pytest.raises(RuntimeError, match="stale actuator view"):
            operation()


def test_retained_facade_group_and_type_views_reject_a_stale_generation() -> None:
    robot = make_finalized_robot()
    facade = robot.actuators
    group = facade["hip"]
    by_type = facade.by_type
    type_view = by_type[IdealPDActuator]
    group_parameters = group.parameters

    robot.collection.clear_generation()

    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = facade.keys()
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = group.joint_names
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = group.cfg
    with pytest.raises(RuntimeError, match="stale actuator view"):
        group.cfg = group.__dict__["cfg"]
    with pytest.raises(RuntimeError, match="stale actuator view"):
        group.custom_state = 1
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = group.compute
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = group.stiffness
    with pytest.raises(RuntimeError, match="stale actuator view"):
        group.stiffness = torch.zeros((2, 2))
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = group_parameters["stiffness"]
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = by_type[IdealPDActuator]
    for attribute in (
        "actuator_type",
        "num_instances",
        "num_joints",
        "joint_names",
        "joint_indices",
        "group_slices",
        "parameter_names",
        "parameters",
    ):
        with pytest.raises(RuntimeError, match="stale actuator view"):
            getattr(type_view, attribute)


@pytest.mark.parametrize(
    "read",
    [
        lambda value: value["hip"],
        iter,
        len,
        lambda value: "hip" in value,
        lambda value: value.get("hip"),
        lambda value: value.keys(),
        lambda value: value.items(),
        lambda value: value.values(),
        lambda value: value.copy(),
        lambda value: dict(value),
        lambda value: value | {},
        lambda value: {} | value,
        lambda value: value == {},
        lambda value: value != {},
        reversed,
        lambda value: value.fromkeys(("a", "b"), 1),
        repr,
    ],
)
def test_every_facade_read_and_snapshot_route_checks_generation(read) -> None:
    robot = make_finalized_robot()
    facade = robot.actuators
    robot.collection.clear_generation()

    with pytest.raises(RuntimeError, match="stale actuator view"):
        read(facade)


@pytest.mark.parametrize("device", _available_devices())
def test_opaque_group_joint_metadata_uses_packed_selector_storage(device: str) -> None:
    """Catch opaque exact groups receiving empty joint-id slices from the packed selector layout."""
    robot = make_finalized_robot(
        device=device,
        groups={"first": _custom_cfg(["hip", "knee"]), "second": _custom_cfg(["ankle"])},
    )
    selector_state = robot.actuators._selector_state
    assert selector_state is not None
    first = robot.actuators["first"]
    second = robot.actuators["second"]

    assert first.joint_indices.tolist() == [0, 1]
    assert second.joint_indices.tolist() == [2]
    int_storage = selector_state._int_slab.untyped_storage().data_ptr()
    assert first.joint_indices.untyped_storage().data_ptr() == int_storage
    assert second.joint_indices.untyped_storage().data_ptr() == int_storage


def _parameter_scope(robot, scope: str):
    """Return the requested public parameter facade."""
    return robot.actuators["hip"] if scope == "group" else robot.actuators.by_type[IdealPDActuator]


@pytest.mark.parametrize("scope", ("group", "type"))
@pytest.mark.parametrize("method", ("set_parameter_index", "set_parameter_mask"))
@pytest.mark.parametrize("state", ("dirty", "stale"))
def test_retained_parameter_setters_reject_dirty_and_stale_views_before_launch(
    monkeypatch: pytest.MonkeyPatch, scope: str, method: str, state: str
) -> None:
    """Catch retained setters that launch after their generation is no longer executable."""
    robot = make_finalized_robot()
    view = _parameter_scope(robot, scope)
    group = robot.actuators["hip"]

    if state == "dirty":
        with pytest.warns(DeprecationWarning):
            robot.actuators["extra"] = group
    else:
        robot.collection.clear_generation()

    def _launch_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("generation guard must run before wp.launch")

    monkeypatch.setattr(wp, "launch", _launch_forbidden)
    kwargs = (
        {"joint_ids": torch.tensor([0])}
        if method.endswith("index")
        else {"joint_mask": torch.tensor([True, False, False])}
    )
    with pytest.raises(RuntimeError, match="rebuild|stale actuator view"):
        getattr(view, method)("stiffness", 1.0, **kwargs)


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_setters_accept_all_value_forms_and_default_compact_slots(scope: str, device: str) -> None:
    """Catch value normalization that loses compact ordering or rejects supported scalar/vector/matrix forms."""
    robot = make_finalized_robot(
        device=device,
        groups={"first": _ideal_cfg(["hip", "knee"]), "second": _ideal_cfg(["knee", "ankle"])},
    )
    view = robot.actuators["first"] if scope == "group" else robot.actuators.by_type[IdealPDActuator]
    slots = 2 if scope == "group" else 4
    expected = torch.arange(slots, dtype=torch.float32, device=device).reshape(1, -1).expand(2, -1)

    view.set_parameter_index("stiffness", 3.0, joint_ids=None)
    torch.testing.assert_close(view.parameters["stiffness"].torch, torch.full_like(expected, 3.0))
    view.set_parameter_index("stiffness", torch.arange(slots, dtype=torch.float32, device=device), joint_ids=None)
    torch.testing.assert_close(view.parameters["stiffness"].torch, expected)
    world_values = expected + 10.0
    view.set_parameter_index("stiffness", world_values, joint_ids=None)
    torch.testing.assert_close(view.parameters["stiffness"].torch, world_values)

    view.set_parameter_mask("damping", 4.0)
    torch.testing.assert_close(view.parameters["damping"].torch, torch.full_like(expected, 4.0))
    view.set_parameter_mask("damping", torch.arange(slots, dtype=torch.float32, device=device))
    torch.testing.assert_close(view.parameters["damping"].torch, expected)
    view.set_parameter_mask("damping", world_values + 10.0)
    torch.testing.assert_close(view.parameters["damping"].torch, world_values + 10.0)


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_default_parameter_index_write_uses_single_kernel_without_duplicate_staging(
    monkeypatch: pytest.MonkeyPatch, device: str, scope: str
) -> None:
    """Catch default selectors that clear or record duplicate scratch despite needing no deduplication."""
    collection = ActuatorCollection(_Simulation(device))
    control = _Control(device)
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"] if scope == "group" else facade.by_type[ImplicitActuator]
    selector_state = facade._selector_state
    assert selector_state is not None
    selector_state._last_env_positions_wp.fill_(17)
    selector_state._last_joint_positions_wp.fill_(19)

    launches = []
    original_launch = wp.launch

    def _record_launch(*args, **kwargs):
        launches.append(args[0] if args else kwargs["kernel"])
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(wp, "launch", _record_launch)
    view.set_parameter_index("stiffness", 5.0)

    assert launches == [actuator_kernels.write_scoped_parameter_index]
    torch.testing.assert_close(
        selector_state._last_env_positions,
        torch.full_like(selector_state._last_env_positions, 17),
    )
    torch.testing.assert_close(
        selector_state._last_joint_positions,
        torch.full_like(selector_state._last_joint_positions, 19),
    )
    write = control.parameter_writes[-1][1]
    assert write.env_ids is None
    assert write.joint_ids is None
    assert write.env_mask is None
    assert write.joint_mask is None


@pytest.mark.parametrize("device", _available_devices())
def test_parameter_preflight_rejects_before_canonical_or_backend_mutation(device: str) -> None:
    """A capture guard must run before the canonical setter launch and backend route."""

    class _CaptureGuardControl(_Control):
        def __init__(self, device: str) -> None:
            super().__init__(device)
            self.preflight_writes = []

        def preflight_actuator_parameter_write(self, name, write) -> None:
            self.preflight_writes.append((name, write))
            raise RuntimeError("parameter write is not capture-safe")

    collection = ActuatorCollection(_Simulation(device))
    control = _CaptureGuardControl(device)
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"]
    canonical_before = view.parameters["stiffness"].torch.clone()
    staging = facade._backend_parameter_staging
    assert staging is not None
    staged_before = staging.target(ImplicitActuator, "stiffness").torch.clone()

    with pytest.raises(RuntimeError, match="not capture-safe"):
        view.set_parameter_index("stiffness", 7.0)

    assert len(control.preflight_writes) == 1
    assert control.parameter_writes == []
    torch.testing.assert_close(view.parameters["stiffness"].torch, canonical_before)
    torch.testing.assert_close(staging.target(ImplicitActuator, "stiffness").torch, staged_before)


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_setters_accept_python_and_warp_inputs_with_signed_indices(scope: str, device: str) -> None:
    """Catch compatibility paths that only accept contiguous Torch inputs or one signed index width."""
    view = _parameter_scope(make_finalized_robot(device=device), scope)
    joint_count = 2 if scope == "group" else 3
    selected_joint = 1

    view.set_parameter_index("stiffness", [5], env_ids=[0], joint_ids=[selected_joint])
    assert view.parameters["stiffness"].torch[0, selected_joint] == 5.0
    view.set_parameter_index(
        "stiffness",
        wp.array([7.0], dtype=wp.float32, device=device),
        env_ids=wp.array([1], dtype=wp.int64, device=device),
        joint_ids=wp.array([selected_joint], dtype=wp.int32, device=device),
    )
    assert view.parameters["stiffness"].torch[1, selected_joint] == 7.0
    view.set_parameter_mask(
        "damping",
        wp.array([float(index) for index in range(joint_count)], dtype=wp.float32, device=device),
        env_mask=wp.array([True, False], dtype=wp.bool, device=device),
        joint_mask=wp.array([True, True, True], dtype=wp.bool, device=device),
    )
    torch.testing.assert_close(
        view.parameters["damping"].torch[0, :joint_count], torch.arange(joint_count, dtype=torch.float32, device=device)
    )


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_setters_reuse_selector_allocations_after_warmup(
    monkeypatch: pytest.MonkeyPatch, scope: str, device: str
) -> None:
    """Catch steady-state setters that recreate selector or scratch storage from noncontiguous inputs."""
    robot = make_finalized_robot(device=device)
    view = _parameter_scope(robot, scope)
    facade = robot.actuators
    binding = robot.actuators["hip"].__dict__["_parameter_binding"]
    selector_state = facade._selector_state
    assert selector_state is not None
    scope_joint_ids = binding.joint_indices if scope == "group" else view._joint_indices
    slot_count = scope_joint_ids.shape[0]
    group_inverse_pointer = selector_state.group_inverse_wp(binding).ptr if scope == "group" else None
    warm_value = torch.arange(1, slot_count + 1, dtype=torch.float32, device=device)
    view.set_parameter_index("stiffness", warm_value, joint_ids=None)
    view.set_parameter_mask("damping", warm_value)
    pointers = (
        selector_state._int_slab.data_ptr(),
        selector_state._bool_slab.data_ptr(),
        selector_state._float_slab.data_ptr(),
        selector_state._identity_ids_wp.ptr,
        selector_state._all_env_mask_wp.ptr,
        selector_state._all_joint_mask_wp.ptr,
        group_inverse_pointer,
    )
    noncontiguous_value = (
        torch.arange(2 * slot_count, dtype=torch.float32, device=device).reshape(slot_count, 2).transpose(0, 1)
    )
    noncontiguous_explicit_value = torch.arange(4, dtype=torch.float32, device=device).reshape(2, 2).transpose(0, 1)
    noncontiguous_ids = torch.tensor([0, 9, 1, 9], dtype=torch.int64, device=device)[::2]
    noncontiguous_joint_ids = torch.tensor([0, 9, 1, 9], dtype=torch.int32, device=device)[::2]
    noncontiguous_env_mask = torch.tensor([True, False, False, False], dtype=torch.bool, device=device)[::2]
    noncontiguous_joint_mask = torch.tensor([True, False, False, False, False, False], dtype=torch.bool, device=device)[
        ::2
    ]

    def _allocation_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("steady-state setter allocated host-visible selector storage")

    monkeypatch.setattr(torch, "arange", _allocation_forbidden)
    monkeypatch.setattr(torch, "full", _allocation_forbidden)
    monkeypatch.setattr(torch, "tensor", _allocation_forbidden)
    monkeypatch.setattr(torch, "as_tensor", _allocation_forbidden)
    monkeypatch.setattr(torch.Tensor, "contiguous", _allocation_forbidden)
    view.set_parameter_index("stiffness", noncontiguous_value, joint_ids=None)
    view.set_parameter_index("stiffness", noncontiguous_value, env_ids=noncontiguous_ids, joint_ids=None)
    view.set_parameter_index(
        "stiffness", noncontiguous_explicit_value, env_ids=noncontiguous_ids, joint_ids=noncontiguous_joint_ids
    )
    view.set_parameter_mask("damping", noncontiguous_value)
    view.set_parameter_mask(
        "damping", noncontiguous_value, env_mask=noncontiguous_env_mask, joint_mask=noncontiguous_joint_mask
    )
    torch.testing.assert_close(view.parameters["stiffness"].torch, noncontiguous_value)
    torch.testing.assert_close(view.parameters["damping"].torch, noncontiguous_value)
    assert pointers == (
        selector_state._int_slab.data_ptr(),
        selector_state._bool_slab.data_ptr(),
        selector_state._float_slab.data_ptr(),
        selector_state._identity_ids_wp.ptr,
        selector_state._all_env_mask_wp.ptr,
        selector_state._all_joint_mask_wp.ptr,
        selector_state.group_inverse_wp(binding).ptr if scope == "group" else None,
    )
    monkeypatch.undo()
    robot.collection.clear_generation()
    replacement = robot.collection.register_articulation(
        key="robot",
        cfgs={"hip": _ideal_cfg(["ankle"])},
        control=_Control(device),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    robot.collection.finalize()
    replacement_view = replacement["hip"] if scope == "group" else replacement.by_type[IdealPDActuator]
    assert replacement_view.joint_indices.tolist() == [2]
    replacement_view.set_parameter_index("stiffness", torch.tensor([8.0], device=device), joint_ids=None)
    torch.testing.assert_close(replacement_view.parameters["stiffness"].torch, torch.full((2, 1), 8.0, device=device))


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_index_uses_cartesian_articulation_selectors(scope: str, device: str) -> None:
    """Catch a selector implementation that treats selected joint ids as compact columns."""
    robot = make_finalized_robot(device=device)
    view = robot.actuators["hip"] if scope == "group" else robot.actuators.by_type[IdealPDActuator]
    before = view.parameters["stiffness"].torch.clone()

    view.set_parameter_index(
        "stiffness",
        torch.tensor([[10.0, 11.0], [20.0, 21.0]], device=device),
        env_ids=torch.tensor([1, 0], device=device),
        joint_ids=torch.tensor([1, 0], device=device),
    )

    expected = before
    expected[1, 1] = 10.0
    expected[1, 0] = 11.0
    expected[0, 1] = 20.0
    expected[0, 0] = 21.0
    torch.testing.assert_close(view.parameters["stiffness"].torch[:, :2], expected[:, :2])


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_index_stages_aliased_source_before_reordered_write(scope: str, device: str) -> None:
    """Catch an indexed write that reads a target alias after an earlier destination overwrote it."""
    view = _parameter_scope(make_finalized_robot(device=device), scope)
    target = view.parameters["stiffness"].torch
    slots = target.shape[1]
    initial = torch.arange(1, 2 * slots + 1, dtype=torch.float32, device=device).reshape(2, slots)
    target.copy_(initial)

    view.set_parameter_index("stiffness", target, env_ids=torch.tensor([1, 0], device=device), joint_ids=None)

    torch.testing.assert_close(target, initial.flip(0))


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_mask_stages_a_strided_raw_warp_alias_before_write(scope: str, device: str) -> None:
    """Catch a masked write that reads a transposed raw Warp alias after overwriting it."""
    groups = (
        {"first": _ideal_cfg(["hip", "knee"]), "second": _ideal_cfg(["ankle"])}
        if scope == "group"
        else {"first": _ideal_cfg(["hip", "knee"])}
    )
    robot = make_finalized_robot(device=device, groups=groups)
    view = robot.actuators["first"] if scope == "group" else robot.actuators.by_type[IdealPDActuator]
    target = view.parameters["stiffness"].torch
    target.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device))
    target_warp = view.parameters["stiffness"].warp
    raw_warp_transpose = wp.array(
        ptr=target_warp.ptr,
        dtype=wp.float32,
        shape=target_warp.shape,
        strides=(target_warp.strides[1], target_warp.strides[0]),
        capacity=target_warp.capacity,
        device=target_warp.device,
        copy=False,
    )

    view.set_parameter_mask(
        "stiffness",
        raw_warp_transpose,
        env_mask=torch.tensor([True, True], device=device),
        joint_mask=torch.tensor([True, True, False], device=device),
    )

    torch.testing.assert_close(target, torch.tensor([[1.0, 3.0], [2.0, 4.0]], device=device))


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_index_stages_a_raw_warp_alias_before_reordered_write(scope: str, device: str) -> None:
    """Catch alias detection that misses a distinct Torch wrapper over the same Warp allocation."""
    view = _parameter_scope(make_finalized_robot(device=device), scope)
    target = view.parameters["stiffness"].torch
    slots = target.shape[1]
    initial = torch.arange(1, 2 * slots + 1, dtype=torch.float32, device=device).reshape(2, slots)
    target.copy_(initial)
    raw_warp_alias = view.parameters["stiffness"].warp

    view.set_parameter_index("stiffness", raw_warp_alias, env_ids=torch.tensor([1, 0], device=device), joint_ids=None)

    torch.testing.assert_close(target, initial.flip(0))


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_alias_staging_rejects_oversized_stride_zero_source_before_launch(
    monkeypatch: pytest.MonkeyPatch, device: str, scope: str
) -> None:
    """Catch overlapping stride-zero values that exceed the bounded selector staging slab."""
    collection = ActuatorCollection(_Simulation(device))
    control = _Control(device)
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"] if scope == "group" else facade.by_type[ImplicitActuator]
    target = view.parameters["stiffness"].torch
    source = target[:1].expand(3, target.shape[1])

    def _launch_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("oversized aliased values launched a kernel")

    def _backend_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("oversized aliased values reached the backend bridge")

    monkeypatch.setattr(wp, "launch", _launch_forbidden)
    monkeypatch.setattr(control, "write_actuator_parameter", _backend_forbidden)
    with pytest.raises(ValueError, match="exceed"):
        view.set_parameter_index(
            "stiffness",
            source,
            env_ids=torch.tensor([0, 1, 0], dtype=torch.int32, device=device),
            joint_ids=None,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_parameter_alias_staging_is_cuda_graph_safe() -> None:
    """Catch Torch-stream alias staging that cannot capture or replay on Warp's stream."""
    view = make_finalized_robot(device="cuda").actuators["hip"]
    target = view.parameters["stiffness"].torch
    slots = target.shape[1]
    initial = torch.arange(1, 2 * slots + 1, dtype=torch.float32, device="cuda").reshape(2, slots)
    target.copy_(initial)
    raw_warp_alias = view.parameters["stiffness"].warp
    env_ids = torch.tensor([1, 0], device="cuda")

    with wp.ScopedCapture(device="cuda") as capture:
        view.set_parameter_index("stiffness", raw_warp_alias, env_ids=env_ids, joint_ids=None)

    replay_source = initial + 10.0
    target.copy_(replay_source)
    torch.cuda.synchronize()
    wp.capture_launch(capture.graph)
    torch.cuda.synchronize()

    torch.testing.assert_close(target, replay_source.flip(0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph capture")
def test_explicit_duplicate_parameter_indices_are_cuda_graph_safe() -> None:
    """Catch Torch-stream duplicate scratch clearing during CUDA graph capture."""
    view = make_finalized_robot(device="cuda").actuators["hip"]
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    env_ids = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    joint_ids = torch.tensor([0, 0], dtype=torch.int32, device="cuda")

    with wp.ScopedCapture(device="cuda") as capture:
        view.set_parameter_index("stiffness", values, env_ids=env_ids, joint_ids=joint_ids)

    values.copy_(torch.tensor([[11.0, 12.0], [13.0, 14.0]], device="cuda"))
    env_ids.copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    joint_ids.copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    view.parameters["stiffness"].torch.zero_()
    torch.cuda.synchronize()
    wp.capture_launch(capture.graph)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        view.parameters["stiffness"].torch,
        torch.tensor([[11.0, 12.0], [13.0, 14.0]], device="cuda"),
    )


@pytest.mark.parametrize("device", _available_devices())
def test_selector_metadata_owning_storage_count_does_not_scale_with_groups(device: str) -> None:
    """Catch per-group selector tensors instead of slices into three flat slabs."""
    owner_counts = []
    for group_count in (1, 3, 12):
        groups = {f"group_{index}": _ideal_cfg(["hip"]) for index in range(group_count)}
        facade = make_finalized_robot(groups=groups, device=device).actuators
        selector_state = facade._selector_state
        assert selector_state is not None
        type_view = facade.by_type[IdealPDActuator]
        slab_storages = {
            selector_state._int_slab.untyped_storage().data_ptr(),
            selector_state._bool_slab.untyped_storage().data_ptr(),
            selector_state._float_slab.untyped_storage().data_ptr(),
        }
        int_storage = selector_state._int_slab.untyped_storage().data_ptr()
        selector_tensors = [value for value in vars(selector_state).values() if isinstance(value, torch.Tensor)]
        for value in vars(selector_state).values():
            if isinstance(value, dict):
                selector_tensors.extend(item for item in value.values() if isinstance(item, torch.Tensor))
        owner_counts.append({tensor.untyped_storage().data_ptr() for tensor in selector_tensors})
        assert type_view.joint_indices.untyped_storage().data_ptr() == int_storage
        assert selector_state.default_joint_ids(type_view.num_joints).untyped_storage().data_ptr() == int_storage
        for group in facade.values():
            binding = group.__dict__["_parameter_binding"]
            assert binding.joint_indices.untyped_storage().data_ptr() == int_storage
            assert selector_state.group_inverse_wp(binding).ptr != 0
            assert (
                int_storage
                <= selector_state.group_inverse_wp(binding).ptr
                < int_storage + selector_state._int_slab.nbytes
            )
        for owner_slots in selector_state._backend_owner_slots.values():
            assert owner_slots.untyped_storage().data_ptr() == int_storage
        for alias in (
            selector_state._identity_ids_wp,
            selector_state._all_env_ids_wp,
            selector_state._all_env_mask_wp,
            selector_state._all_joint_mask_wp,
            selector_state._last_env_positions_wp,
            selector_state._last_joint_positions_wp,
            *selector_state._type_joint_ids_wp.values(),
            *selector_state._group_joint_ids_wp.values(),
            *selector_state._group_inverse_wp.values(),
        ):
            assert any(
                slab.data_ptr() <= alias.ptr < slab.data_ptr() + slab.nbytes
                for slab in (selector_state._int_slab, selector_state._bool_slab)
            )
        assert slab_storages == owner_counts[-1]
        for old_attribute in (
            "_parameter_default_joint_ids",
            "_scope_joint_ids_wp",
            "_group_inverse_lookups",
            "_group_inverse_lookups_wp",
        ):
            assert not hasattr(facade, old_attribute)

    assert [len(count) for count in owner_counts] == [3, 3, 3]


@pytest.mark.parametrize("device", _available_devices())
def test_selector_metadata_build_avoids_python_world_sized_sequences(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Catch selector construction that materializes Python objects once per cloned world."""
    num_worlds = 4096
    original_range = range
    original_tensor = torch.tensor

    def _reject_world_range(*args):
        generated = original_range(*args)
        if len(generated) >= num_worlds:
            raise AssertionError("selector metadata iterated once per cloned world")
        return generated

    def _reject_world_sized_tensor(data, *args, **kwargs):
        if isinstance(data, (list, tuple)) and len(data) >= num_worlds:
            raise AssertionError("selector metadata materialized a world-sized Python sequence")
        return original_tensor(data, *args, **kwargs)

    monkeypatch.setattr(actuator_collection, "range", _reject_world_range, raising=False)
    monkeypatch.setattr(torch, "tensor", _reject_world_sized_tensor)
    collection = ActuatorCollection(_Simulation(device, num_worlds))
    facade = collection.register_articulation(
        key="robot",
        cfgs={"hip": _ideal_cfg(["hip", "knee"])},
        control=_Control(device, num_worlds),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )

    collection.finalize()

    assert facade["hip"].parameters["stiffness"].torch.shape == (num_worlds, 2)


def test_clear_generation_releases_selector_state_from_retained_public_views() -> None:
    """Catch stale facade objects retaining selector slabs or canonical proxy aliases."""
    robot = make_finalized_robot()
    facade = robot.actuators
    group = facade["hip"]
    group_parameters = group.parameters
    by_type = facade.by_type
    type_view = by_type[IdealPDActuator]
    type_parameters = type_view.parameters
    selector_state = facade._selector_state
    assert selector_state is not None
    int_ref = weakref.ref(selector_state._int_slab)
    bool_ref = weakref.ref(selector_state._bool_slab)
    float_ref = weakref.ref(selector_state._float_slab)
    active_generation = robot.collection._active_generation
    assert active_generation is not None
    field_proxy = active_generation.stores[IdealPDActuator]._fields["stiffness"]
    field_proxy_ref = weakref.ref(field_proxy)
    field_warp_ref = weakref.ref(field_proxy.warp)
    field_torch_ref = weakref.ref(field_proxy.torch)
    type_proxy = type_parameters["stiffness"]
    type_proxy_ref = weakref.ref(type_proxy)
    del active_generation, field_proxy, type_proxy
    del selector_state

    robot.collection.clear_generation()
    gc.collect()

    assert int_ref() is None
    assert bool_ref() is None
    assert float_ref() is None
    assert field_proxy_ref() is None
    assert field_warp_ref() is None
    assert field_torch_ref() is None
    assert type_proxy_ref() is None
    for operation in (
        lambda: group.parameters,
        lambda: group_parameters["stiffness"],
        lambda: by_type[IdealPDActuator],
        lambda: type_view.parameters,
        lambda: type_parameters["stiffness"],
    ):
        with pytest.raises(RuntimeError, match="stale actuator view"):
            operation()


def test_failed_finalization_does_not_revive_retained_selector_children() -> None:
    """Catch ABA reuse when a retry republishes the same facade generation number."""

    class _CompletionControl(_Control):
        def __init__(self, *, fail: bool) -> None:
            super().__init__()
            self.fail = fail
            self.view = None
            self.children = None

        def bind_actuator_view(self, view) -> None:
            self.view = view

        def complete_articulation_initialization(self) -> None:
            if self.children is None:
                group = self.view["hip"]
                by_type = self.view.by_type
                type_view = by_type[IdealPDActuator]
                self.children = (
                    group,
                    by_type,
                    type_view,
                    group.parameters,
                    type_view.parameters,
                    by_type.keys(),
                    by_type.items(),
                )
            if self.fail:
                raise RuntimeError("intentional completion failure")

    collection = ActuatorCollection(_Simulation())
    first_control = _CompletionControl(fail=False)
    second_control = _CompletionControl(fail=True)
    first_facade = collection.register_articulation(
        key="first",
        cfgs={"hip": _ideal_cfg(["hip"])},
        control=first_control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    second_facade = collection.register_articulation(
        key="second",
        cfgs={"hip": _ideal_cfg(["knee"])},
        control=second_control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    with pytest.raises(RuntimeError, match="intentional completion failure"):
        collection.finalize()
    (
        old_group,
        old_by_type,
        old_type,
        old_group_parameters,
        old_type_parameters,
        old_by_type_keys,
        old_by_type_items,
    ) = first_control.children

    second_control.fail = False
    collection.finalize()
    first_facade["hip"].set_parameter_index("stiffness", 7.0)
    second_facade["hip"].set_parameter_index("stiffness", 11.0)

    for operation in (
        lambda: old_group.parameters,
        lambda: old_group_parameters["stiffness"],
        lambda: old_by_type[IdealPDActuator],
        lambda: old_type.parameters,
        lambda: old_type_parameters["stiffness"],
        lambda: list(old_by_type_keys),
        lambda: list(old_by_type_items),
    ):
        with pytest.raises(RuntimeError, match="stale actuator view"):
            operation()


def test_parameter_index_filters_out_of_scope_joint_columns() -> None:
    """Catch an implementation that shifts values after filtering an out-of-scope joint."""
    group = make_finalized_robot(groups={"hip": _ideal_cfg(["hip", "knee"])}).actuators["hip"]

    group.set_parameter_index("damping", torch.tensor([10.0, 20.0, 30.0]), joint_ids=torch.tensor([1, 2, 0]))

    torch.testing.assert_close(group.parameters["damping"].torch, torch.tensor([[30.0, 10.0], [30.0, 10.0]]))


def test_parameter_index_duplicate_ids_use_last_cartesian_value() -> None:
    """Catch a parallel write race that does not give duplicate selectors last-write semantics."""
    group = make_finalized_robot(groups={"hip": _ideal_cfg(["hip", "knee"])}).actuators["hip"]

    group.set_parameter_index(
        "damping",
        torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
        env_ids=torch.tensor([0, 0]),
        joint_ids=torch.tensor([1, 1]),
    )

    assert group.parameters["damping"].torch[0, 1] == 7.0


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_index_duplicate_environment_and_joint_ids_use_last_positions(scope: str, device: str) -> None:
    """Catch duplicate filtering that resolves one selector dimension but not the other."""
    view = _parameter_scope(make_finalized_robot(device=device), scope)

    view.set_parameter_index(
        "stiffness",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
        env_ids=torch.tensor([0, 0], device=device),
        joint_ids=torch.tensor([0, 1], device=device),
    )
    torch.testing.assert_close(view.parameters["stiffness"].torch[0, :2], torch.tensor([3.0, 4.0], device=device))
    view.set_parameter_index(
        "stiffness",
        torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device),
        env_ids=torch.tensor([0, 1], device=device),
        joint_ids=torch.tensor([1, 1], device=device),
    )
    torch.testing.assert_close(view.parameters["stiffness"].torch[:, 1], torch.tensor([6.0, 8.0], device=device))


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_index_filters_invalid_environment_and_joint_ids_together(scope: str, device: str) -> None:
    """Catch normal-mode filtering that corrupts the source Cartesian value association."""
    view = _parameter_scope(make_finalized_robot(device=device), scope)

    view.set_parameter_index(
        "stiffness",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
        env_ids=torch.tensor([9, 0], device=device),
        joint_ids=torch.tensor([9, 0], device=device),
    )

    assert view.parameters["stiffness"].torch[0, 0] == 4.0


def test_type_parameter_index_fans_out_overlapping_articulation_dofs() -> None:
    """Catch a type selector that updates only the first compact occurrence of a joint."""
    view = make_finalized_robot(
        groups={"first": _ideal_cfg(["hip", "knee"]), "second": _ideal_cfg(["knee", "ankle"])}
    ).actuators.by_type[IdealPDActuator]

    view.set_parameter_index("stiffness", torch.tensor([13.0]), joint_ids=torch.tensor([1]))

    torch.testing.assert_close(view.parameters["stiffness"].torch[:, [1, 2]], torch.full((2, 2), 13.0))


@pytest.mark.parametrize("device", _available_devices())
def test_type_parameter_selectors_fan_out_three_way_overlaps_in_configuration_order(device: str) -> None:
    """Catch type routing that handles two duplicate slots but drops a third configuration occurrence."""

    class _FourJointControl(_Control):
        def __init__(self, device: str) -> None:
            super().__init__(device)
            self._joint_names = ("hip", "knee", "ankle", "toe")

    collection = ActuatorCollection(_Simulation(device))
    facade = collection.register_articulation(
        key="robot",
        cfgs={
            "first": _ideal_cfg(["hip", "ankle"]),
            "second": _ideal_cfg(["ankle", "toe"]),
            "third": _ideal_cfg(["ankle"]),
        },
        control=_FourJointControl(device),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade.by_type[IdealPDActuator]

    assert view.joint_indices.tolist() == [0, 2, 2, 3, 2]
    view.set_parameter_index(
        "stiffness", torch.tensor([17.0], device=device), joint_ids=torch.tensor([2], device=device)
    )
    torch.testing.assert_close(
        view.parameters["stiffness"].torch[:, [1, 2, 4]], torch.full((2, 3), 17.0, device=device)
    )
    view.set_parameter_mask(
        "damping",
        torch.tensor([0.0, 11.0, 12.0, 0.0, 14.0], device=device),
        joint_mask=torch.tensor([False, False, True, False], device=device),
    )
    torch.testing.assert_close(
        view.parameters["damping"].torch[:, [1, 2, 4]], torch.tensor([[11.0, 12.0, 14.0]] * 2, device=device)
    )


@pytest.mark.parametrize("device", _available_devices())
def test_type_parameter_mask_keeps_values_in_compact_slot_order(device: str) -> None:
    """Catch a mask selector that compacts values by the number of true articulation joints."""
    view = make_finalized_robot(
        device=device, groups={"first": _ideal_cfg(["hip", "knee"]), "second": _ideal_cfg(["knee", "ankle"])}
    ).actuators.by_type[IdealPDActuator]

    view.set_parameter_mask(
        "stiffness",
        torch.tensor([1.0, 11.0, 22.0, 3.0], device=device),
        joint_mask=torch.tensor([False, True, False], device=device),
    )

    torch.testing.assert_close(
        view.parameters["stiffness"].torch[:, [1, 2]], torch.tensor([[11.0, 22.0]] * 2, device=device)
    )


def test_unknown_parameter_fails_before_a_parameter_write() -> None:
    """Catch an unknown-field path that silently writes or launches before validation."""
    group = make_finalized_robot().actuators["hip"]
    before = group.parameters["stiffness"].torch.clone()

    with pytest.raises(KeyError, match="made_up"):
        group.set_parameter_index("made_up", 1.0)

    torch.testing.assert_close(group.parameters["stiffness"].torch, before)


def test_parameter_index_invalid_ids_are_ignored_without_debug_validation() -> None:
    """Catch normal-mode validation that reads device selector values or rejects ignored ids."""
    group = make_finalized_robot().actuators["hip"]
    before = group.parameters["stiffness"].torch.clone()

    group.set_parameter_index("stiffness", torch.tensor([17.0, 19.0]), joint_ids=torch.tensor([-1, 9]))

    torch.testing.assert_close(group.parameters["stiffness"].torch, before)


def test_parameter_index_invalid_ids_raise_with_debug_validation() -> None:
    """Catch the absence of opt-in, value-dependent selector validation."""
    group = make_finalized_robot(debug_validation=True).actuators["hip"]

    with pytest.raises(ValueError, match="stiffness.*group"):
        group.set_parameter_index("stiffness", torch.tensor([-1.0]), joint_ids=torch.tensor([-1]))


@pytest.mark.parametrize("scope", ("group", "type"))
@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize(
    ("env_values", "joint_values", "ownership_case"),
    (
        ([-1], [0], False),
        ([2], [0], False),
        ([0], [3], False),
        ([0, 0], [0], False),
        ([0], [0, 0], False),
        ([0], [2], True),
    ),
)
def test_debug_parameter_index_rejects_invalid_contents_before_launch(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    device: str,
    env_values: list[int],
    joint_values: list[int],
    ownership_case: bool,
) -> None:
    """Catch debug validation that lets invalid selectors reach the device writer."""
    groups = {"hip": _ideal_cfg(["hip", "knee"])}
    robot = make_finalized_robot(device=device, groups=groups, debug_validation=True)
    view = robot.actuators["hip"] if scope == "group" else robot.actuators.by_type[IdealPDActuator]
    if scope == "type" and ownership_case:
        robot = make_finalized_robot(device=device, groups={"hip": _ideal_cfg(["hip"])}, debug_validation=True)
        view = robot.actuators.by_type[IdealPDActuator]
    env_ids = torch.tensor(env_values, device=device)
    joint_ids = torch.tensor(joint_values, device=device)

    def _launch_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("debug validation must reject before wp.launch")

    monkeypatch.setattr(wp, "launch", _launch_forbidden)
    with pytest.raises(ValueError, match="stiffness"):
        view.set_parameter_index("stiffness", 1.0, env_ids=env_ids, joint_ids=joint_ids)


@pytest.mark.parametrize("scope", ("group", "type"))
@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize(
    "operation",
    (
        lambda view, device: view.set_parameter_index("unknown", 1.0),
        lambda view, device: view.set_parameter_mask("unknown", 1.0),
        lambda view, device: view.set_parameter_index("stiffness", torch.ones((1, 1, 1), device=device)),
        lambda view, device: view.set_parameter_mask("stiffness", torch.ones((1, 1, 1), device=device)),
        lambda view, device: view.set_parameter_index("stiffness", torch.ones(1, device=device)),
        lambda view, device: view.set_parameter_mask("stiffness", torch.ones(1, device=device)),
        lambda view, device: view.set_parameter_index("stiffness", torch.ones(2, dtype=torch.float64, device=device)),
        lambda view, device: view.set_parameter_mask("stiffness", torch.ones(2, dtype=torch.float64, device=device)),
        lambda view, device: view.set_parameter_index("stiffness", torch.ones((2, 1), device=device)),
        lambda view, device: view.set_parameter_mask("stiffness", torch.ones((2, 1), device=device)),
        lambda view, device: view.set_parameter_index(
            "stiffness", 1.0, env_ids=torch.ones((1, 1), dtype=torch.int32, device=device)
        ),
        lambda view, device: view.set_parameter_index("stiffness", 1.0, joint_ids=torch.tensor([0.0], device=device)),
        lambda view, device: view.set_parameter_mask("stiffness", 1.0, env_mask=torch.tensor([1, 0], device=device)),
        lambda view, device: view.set_parameter_mask(
            "stiffness", 1.0, joint_mask=torch.ones((1, 1), dtype=torch.bool, device=device)
        ),
        lambda view, device: view.set_parameter_mask("stiffness", 1.0, env_mask=torch.tensor([True], device=device)),
        lambda view, device: view.set_parameter_mask(
            "stiffness", 1.0, joint_mask=torch.tensor([True, False], device=device)
        ),
    ),
)
def test_malformed_parameter_inputs_fail_before_launch_or_backend_hook(
    monkeypatch: pytest.MonkeyPatch, scope: str, device: str, operation
) -> None:
    """Catch malformed paths that launch or notify the backend before rejecting input."""
    collection = ActuatorCollection(_Simulation(device))
    control = _Control(device)
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"] if scope == "group" else facade.by_type[ImplicitActuator]

    def _launch_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid input launched a kernel")

    def _backend_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("invalid input reached the backend bridge")

    monkeypatch.setattr(wp, "launch", _launch_forbidden)
    monkeypatch.setattr(control, "write_actuator_parameter", _backend_forbidden)
    with pytest.raises((KeyError, TypeError, ValueError)):
        operation(view, device)


@pytest.mark.parametrize("scope", ("group", "type"))
def test_parameter_setters_reject_cross_device_inputs_before_launch(
    monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    """Catch device checks deferred until a kernel has already been prepared."""
    if not torch.cuda.is_available():
        pytest.skip("cross-device input validation requires CUDA")
    collection = ActuatorCollection(_Simulation("cuda"))
    control = _Control("cuda")
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"] if scope == "group" else facade.by_type[ImplicitActuator]

    def _launch_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cross-device input launched a kernel")

    def _backend_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cross-device input reached the backend bridge")

    monkeypatch.setattr(wp, "launch", _launch_forbidden)
    monkeypatch.setattr(control, "write_actuator_parameter", _backend_forbidden)
    with pytest.raises(ValueError, match="facade device"):
        view.set_parameter_index("stiffness", torch.tensor([1.0]), joint_ids=torch.tensor([0]))
    with pytest.raises(ValueError, match="same device"):
        view.set_parameter_mask("stiffness", torch.tensor([1.0]))
    with pytest.raises(ValueError, match="facade device"):
        view.set_parameter_index(
            "stiffness",
            torch.tensor([1.0], device="cuda"),
            env_ids=torch.tensor([0]),
            joint_ids=torch.tensor([0], device="cuda"),
        )
    with pytest.raises(ValueError, match="facade device"):
        view.set_parameter_index(
            "stiffness",
            torch.tensor([1.0], device="cuda"),
            env_ids=torch.tensor([0], device="cuda"),
            joint_ids=torch.tensor([0]),
        )
    with pytest.raises(ValueError, match="facade device"):
        view.set_parameter_mask("stiffness", 1.0, env_mask=torch.tensor([True, False]))
    with pytest.raises(ValueError, match="facade device"):
        view.set_parameter_mask("stiffness", 1.0, joint_mask=torch.tensor([True, False, False]))


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
@pytest.mark.parametrize("method", ("index", "mask"))
def test_parameter_selectors_do_not_read_device_values_in_normal_mode(
    monkeypatch: pytest.MonkeyPatch, device: str, scope: str, method: str
) -> None:
    """Catch a normal selector path that synchronizes to inspect device-resident selector contents."""
    view = _parameter_scope(make_finalized_robot(device=device), scope)

    def _read_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("normal selector path read a device tensor")

    monkeypatch.setattr(torch.Tensor, "cpu", _read_forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", _read_forbidden)
    monkeypatch.setattr(torch.Tensor, "item", _read_forbidden)
    if device == "cuda":
        monkeypatch.setattr(torch.cuda, "synchronize", _read_forbidden)
    for name in ("synchronize", "synchronize_device", "synchronize_stream"):
        if hasattr(wp, name):
            monkeypatch.setattr(wp, name, _read_forbidden)

    if method == "index":
        view.set_parameter_index(
            "stiffness",
            torch.tensor([[5.0], [7.0]], device=device),
            env_ids=torch.tensor([0, 1], device=device),
            joint_ids=torch.tensor([1], device=device),
        )
        view.set_parameter_index(
            "stiffness",
            torch.tensor([[5.0], [7.0]], device=device),
            env_ids=torch.tensor([-1, 9], device=device),
            joint_ids=torch.tensor([-1], device=device),
        )
    else:
        view.set_parameter_mask(
            "stiffness",
            torch.arange(view.num_joints, dtype=torch.float32, device=device),
            env_mask=torch.tensor([True, False], device=device),
            joint_mask=torch.tensor([False, True, False], device=device),
        )


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
@pytest.mark.parametrize("method", ("index", "mask"))
def test_backend_parameter_router_does_not_read_device_values_in_normal_mode(
    monkeypatch: pytest.MonkeyPatch, device: str, scope: str, method: str
) -> None:
    """Catch a canonical backend route that synchronizes after an otherwise fast selector write."""
    collection = ActuatorCollection(_Simulation(device))
    control = _Control(device)
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"] if scope == "group" else facade.by_type[ImplicitActuator]

    def _read_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("backend parameter routing read a device tensor")

    monkeypatch.setattr(torch.Tensor, "cpu", _read_forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", _read_forbidden)
    monkeypatch.setattr(torch.Tensor, "item", _read_forbidden)
    if device == "cuda":
        monkeypatch.setattr(torch.cuda, "synchronize", _read_forbidden)
    for name in ("synchronize", "synchronize_device", "synchronize_stream"):
        if hasattr(wp, name):
            monkeypatch.setattr(wp, name, _read_forbidden)

    if method == "index":
        view.set_parameter_index(
            "stiffness",
            torch.tensor([7.0], device=device),
            env_ids=torch.tensor([0], device=device),
            joint_ids=torch.tensor([1], device=device),
        )
    else:
        view.set_parameter_mask(
            "stiffness",
            torch.tensor([5.0, 7.0], device=device),
            env_mask=torch.tensor([True, False], device=device),
            joint_mask=torch.tensor([False, True, False], device=device),
        )
    assert control.parameter_writes


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
@pytest.mark.parametrize("method", ("index", "mask"))
def test_backend_parameter_router_forwards_normalized_explicit_selectors(device: str, scope: str, method: str) -> None:
    """Catch backend writes that retain Python or Warp selector wrappers instead of canonical device tensors."""
    collection = ActuatorCollection(_Simulation(device))
    control = _Control(device)
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"] if scope == "group" else facade.by_type[ImplicitActuator]

    if method == "index":
        view.set_parameter_index("stiffness", [7.0], env_ids=[0], joint_ids=[1])
        write = control.parameter_writes[-1][1]
        assert isinstance(write.env_ids, torch.Tensor)
        assert isinstance(write.joint_ids, torch.Tensor)
        assert write.env_ids.dtype is torch.int32
        assert write.joint_ids.dtype is torch.int32
        assert write.env_ids.device.type == torch.device(device).type
        assert write.joint_ids.device.type == torch.device(device).type
        return

    view.set_parameter_mask(
        "stiffness",
        wp.array([5.0, 7.0], dtype=wp.float32, device=device),
        env_mask=wp.array([True, False], dtype=wp.bool, device=device),
        joint_mask=wp.array([False, True, False], dtype=wp.bool, device=device),
    )
    write = control.parameter_writes[-1][1]
    assert isinstance(write.env_mask, torch.Tensor)
    assert isinstance(write.joint_mask, torch.Tensor)
    assert write.env_mask.dtype is torch.bool
    assert write.joint_mask.dtype is torch.bool
    assert write.env_mask.device.type == torch.device(device).type
    assert write.joint_mask.device.type == torch.device(device).type


@pytest.mark.parametrize("device", _available_devices())
def test_parameter_empty_and_all_false_selections_leave_values_unchanged(device: str) -> None:
    """Catch selector kernels that mutate canonical storage for no-op selections."""
    view = make_finalized_robot(device=device).actuators.by_type[IdealPDActuator]
    before = view.parameters["stiffness"].torch.clone()

    view.set_parameter_index(
        "stiffness",
        torch.empty((0, 0), dtype=torch.float32, device=device),
        env_ids=torch.empty(0, dtype=torch.int32, device=device),
        joint_ids=torch.empty(0, dtype=torch.int32, device=device),
    )
    view.set_parameter_mask(
        "stiffness",
        11.0,
        env_mask=torch.zeros(2, dtype=torch.bool, device=device),
        joint_mask=torch.zeros(3, dtype=torch.bool, device=device),
    )

    torch.testing.assert_close(view.parameters["stiffness"].torch, before)


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("scope", ("group", "type"))
def test_empty_selectors_and_all_false_masks_are_backend_noops(scope: str, device: str) -> None:
    """Catch empty or masked scoped writes that still mutate a backend-owned implicit drive."""

    class _NoOpControl(_Control):
        def __init__(self, device: str) -> None:
            super().__init__(device)
            self.backend_stiffness = torch.ones((2, 3), device=device)

        def write_actuator_parameter(self, name, write) -> None:
            super().write_actuator_parameter(name, write)
            if name != "stiffness" or write.backend_owner_slots is None:
                return
            env_mask = torch.ones(2, dtype=torch.bool, device=self.device)
            joint_mask = torch.ones(3, dtype=torch.bool, device=self.device)
            if write.env_ids is not None:
                env_mask[:] = False
                env_mask[write.env_ids] = True
            if write.joint_ids is not None:
                joint_mask[:] = False
                joint_mask[write.joint_ids] = True
            if write.env_mask is not None:
                env_mask &= write.env_mask
            if write.joint_mask is not None:
                joint_mask &= write.joint_mask
            for articulation_joint, compact_slot in enumerate(write.backend_owner_slots.tolist()):
                if compact_slot >= 0 and joint_mask[articulation_joint]:
                    self.backend_stiffness[env_mask, articulation_joint] = write.value[env_mask, compact_slot]

    collection = ActuatorCollection(_Simulation(device))
    control = _NoOpControl(device)
    facade = collection.register_articulation(
        key="robot",
        cfgs={"implicit": _implicit_cfg(["hip", "knee"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    view = facade["implicit"] if scope == "group" else facade.by_type[ImplicitActuator]
    before_canonical = view.parameters["stiffness"].torch.clone()
    before_backend = control.backend_stiffness.clone()

    view.set_parameter_index(
        "stiffness",
        torch.empty((0, 1), dtype=torch.float32, device=device),
        env_ids=torch.empty(0, dtype=torch.int32, device=device),
        joint_ids=torch.tensor([0], dtype=torch.int32, device=device),
    )
    view.set_parameter_index(
        "stiffness",
        torch.empty((1, 0), dtype=torch.float32, device=device),
        env_ids=torch.tensor([0], dtype=torch.int32, device=device),
        joint_ids=torch.empty(0, dtype=torch.int32, device=device),
    )
    view.set_parameter_mask("stiffness", 7.0, env_mask=torch.zeros(2, dtype=torch.bool, device=device))
    view.set_parameter_mask("stiffness", 7.0, joint_mask=torch.zeros(3, dtype=torch.bool, device=device))

    torch.testing.assert_close(view.parameters["stiffness"].torch, before_canonical)
    torch.testing.assert_close(control.backend_stiffness, before_backend)


def test_group_parameter_mask_forwards_its_owner_binding_to_the_backend() -> None:
    """Catch mask side effects that cannot distinguish a non-owner group from its type view."""
    collection = ActuatorCollection(_Simulation())
    control = _Control()
    view = collection.register_articulation(
        key="robot",
        cfgs={"first": _implicit_cfg(["hip", "knee"]), "last": _implicit_cfg(["knee", "ankle"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()

    view["first"].set_parameter_mask("stiffness", 7.0, joint_mask=torch.tensor([False, True, False], dtype=torch.bool))

    assert control.parameter_writes[-1][1].group_binding is view["first"].__dict__["_parameter_binding"]


@pytest.mark.parametrize("device", _available_devices())
def test_native_group_parameter_write_uses_native_owner_for_overlapping_types(device: str) -> None:
    """Catch cross-type backend routing that selects an ordinary group over its native owner."""

    class _NativeControl(_Control):
        def __init__(self, device: str) -> None:
            super().__init__(device)
            self.backend_stiffness = torch.ones((2, 3), device=device)

        def discover_native_actuators(self, cfgs) -> set[str]:
            assert tuple(cfgs) == ("ordinary", "native")
            return {"native"}

        def write_actuator_parameter(self, name, write) -> None:
            super().write_actuator_parameter(name, write)
            if name != "stiffness" or write.backend_owner_slots is None:
                return
            owners = write.backend_owner_slots.tolist()
            if write.group_binding is not None:
                binding = write.group_binding
                for local_slot, articulation_slot in enumerate(binding.joint_indices.tolist()):
                    if owners[articulation_slot] == binding.type_slice.start + local_slot:
                        self.backend_stiffness[:, articulation_slot] = write.value[:, local_slot]
                return
            for articulation_slot, owner_slot in enumerate(owners):
                if owner_slot >= 0:
                    self.backend_stiffness[:, articulation_slot] = write.value[:, owner_slot]

    collection = ActuatorCollection(_Simulation(device))
    control = _NativeControl(device)
    view = collection.register_articulation(
        key="robot",
        cfgs={"ordinary": _ideal_cfg(["hip"]), "native": _dc_cfg(["hip"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()

    view["ordinary"].set_parameter_index("stiffness", 3.0)
    assert control.parameter_writes == []
    view["native"].set_parameter_index("stiffness", 7.0)
    assert control.parameter_writes[-1][0] == "stiffness"
    assert control.parameter_writes[-1][1].backend_owner_slots is not None
    torch.testing.assert_close(
        control.parameter_writes[-1][1].backend_owner_slots,
        torch.tensor([0, -1, -1], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(control.backend_stiffness[:, 0], torch.full((2,), 7.0, device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.ones(2, device=device))
    view["native"].set_parameter_mask("stiffness", 9.0)
    torch.testing.assert_close(control.backend_stiffness[:, 0], torch.full((2,), 9.0, device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.ones(2, device=device))
    view.by_type[IdealPDActuator].set_parameter_index("stiffness", 11.0, joint_ids=torch.tensor([0], device=device))
    assert control.parameter_writes[-1][0] == "stiffness"
    assert control.parameter_writes[-1][1].backend_owner_slots is not None
    torch.testing.assert_close(control.backend_stiffness[:, 0], torch.full((2,), 9.0, device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.ones(2, device=device))
    view.by_type[DCMotor].set_parameter_mask("stiffness", 13.0)
    torch.testing.assert_close(control.backend_stiffness[:, 0], torch.full((2,), 13.0, device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.ones(2, device=device))


def test_native_discovery_rejects_unknown_actuator_group() -> None:
    """Catch control bridges that return native actuator names absent from the configuration."""

    class _InvalidNativeControl(_Control):
        def discover_native_actuators(self, cfgs) -> set[str]:
            del cfgs
            return {"missing"}

    collection = ActuatorCollection(_Simulation())
    with pytest.raises(ValueError, match="unknown group.*'missing'"):
        collection.register_articulation(
            key="robot",
            cfgs={"hip": _ideal_cfg(["hip"])},
            control=_InvalidNativeControl(),
            replication_cfg_id=1,
            debug_validation=False,
            debug_value_resolution=False,
        )


@pytest.mark.parametrize("device", _available_devices())
def test_type_debug_validation_rejects_in_range_joint_outside_its_scope(device: str) -> None:
    """Catch type debug validation that checks bounds but not type ownership."""
    normal = make_finalized_robot(device=device, groups={"hip": _ideal_cfg(["hip"])}).actuators.by_type[IdealPDActuator]
    before = normal.parameters["stiffness"].torch.clone()
    normal.set_parameter_index(
        "stiffness", torch.tensor([9.0], device=device), joint_ids=torch.tensor([2], device=device)
    )
    torch.testing.assert_close(normal.parameters["stiffness"].torch, before)

    debug = make_finalized_robot(
        device=device, groups={"hip": _ideal_cfg(["hip"])}, debug_validation=True
    ).actuators.by_type[IdealPDActuator]
    with pytest.raises(ValueError, match="stiffness.*type"):
        debug.set_parameter_index(
            "stiffness", torch.tensor([9.0], device=device), joint_ids=torch.tensor([2], device=device)
        )


def test_type_parameter_mask_forwards_csr_owner_metadata() -> None:
    """Catch type mask writes that cannot route canonical owner slots through a shared backend consumer."""
    collection = ActuatorCollection(_Simulation())
    control = _Control()
    view = collection.register_articulation(
        key="robot",
        cfgs={"first": _implicit_cfg(["hip", "knee"]), "last": _implicit_cfg(["knee", "ankle"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()

    view.by_type[ImplicitActuator].set_parameter_mask("stiffness", 7.0)

    write = control.parameter_writes[-1][1]
    assert write.type_csr_offsets is not None
    assert write.type_csr_slots is not None
    assert write.backend_owner_slots is not None


@pytest.mark.parametrize("device", _available_devices())
def test_overlapping_implicit_parameter_writes_update_only_configuration_owner(device: str) -> None:
    """Catch backend routing that lets a non-owner overlapping implicit group overwrite a solver drive."""

    class _StatefulControl(_Control):
        def __init__(self, device: str) -> None:
            super().__init__(device)
            self.backend_stiffness = torch.ones((2, 3), device=device)

        def write_actuator_parameter(self, name, write) -> None:
            super().write_actuator_parameter(name, write)
            if name != "stiffness" or write.backend_owner_slots is None:
                return
            owners = write.backend_owner_slots.tolist()
            env_selected = torch.ones(2, dtype=torch.bool, device=self.device)
            joint_selected = torch.ones(3, dtype=torch.bool, device=self.device)
            if write.env_ids is not None:
                env_selected[:] = False
                env_selected[write.env_ids] = True
            if write.joint_ids is not None:
                joint_selected[:] = False
                joint_selected[write.joint_ids] = True
            if write.env_mask is not None:
                env_selected &= write.env_mask
            if write.joint_mask is not None:
                joint_selected &= write.joint_mask
            if write.group_binding is not None:
                binding = write.group_binding
                for local_slot, articulation_slot in enumerate(binding.joint_indices.tolist()):
                    if (
                        joint_selected[articulation_slot]
                        and owners[articulation_slot] == binding.type_slice.start + local_slot
                    ):
                        self.backend_stiffness[env_selected, articulation_slot] = write.value[env_selected, local_slot]
                return
            for articulation_slot, owner_slot in enumerate(owners):
                if joint_selected[articulation_slot] and owner_slot >= 0:
                    self.backend_stiffness[env_selected, articulation_slot] = write.value[env_selected, owner_slot]

    collection = ActuatorCollection(_Simulation(device))
    control = _StatefulControl(device)
    view = collection.register_articulation(
        key="robot",
        cfgs={"first": _implicit_cfg(["hip", "knee"]), "last": _implicit_cfg(["knee", "ankle"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()

    view["first"].set_parameter_index("stiffness", 5.0, joint_ids=torch.tensor([1], device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.ones(2, device=device))
    view["last"].set_parameter_index("stiffness", 17.0, joint_ids=torch.tensor([1], device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 17.0, device=device))
    view["first"].set_parameter_mask("stiffness", 5.0, joint_mask=torch.tensor([False, True, False], device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 17.0, device=device))
    view["last"].set_parameter_mask("stiffness", 19.0, joint_mask=torch.tensor([False, True, False], device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 19.0, device=device))
    view.by_type[ImplicitActuator].set_parameter_index("stiffness", 21.0, joint_ids=torch.tensor([1], device=device))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 21.0, device=device))
    view.by_type[ImplicitActuator].set_parameter_mask(
        "stiffness",
        torch.tensor([1.0, 9.0, 23.0, 3.0], device=device),
        joint_mask=torch.tensor([False, True, False], device=device),
    )
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 23.0, device=device))
    canonical_before = view.by_type[ImplicitActuator].parameters["stiffness"].torch.clone()
    backend_before = control.backend_stiffness.clone()
    view.by_type[ImplicitActuator].set_parameter_index(
        "stiffness",
        torch.empty((0, 0), dtype=torch.float32, device=device),
        env_ids=torch.empty(0, dtype=torch.int32, device=device),
        joint_ids=torch.empty(0, dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(view.by_type[ImplicitActuator].parameters["stiffness"].torch, canonical_before)
    torch.testing.assert_close(control.backend_stiffness, backend_before)
    view.by_type[ImplicitActuator].set_parameter_mask(
        "stiffness",
        31.0,
        env_mask=torch.zeros(2, dtype=torch.bool, device=device),
        joint_mask=torch.zeros(3, dtype=torch.bool, device=device),
    )
    torch.testing.assert_close(view.by_type[ImplicitActuator].parameters["stiffness"].torch, canonical_before)
    torch.testing.assert_close(control.backend_stiffness, backend_before)
