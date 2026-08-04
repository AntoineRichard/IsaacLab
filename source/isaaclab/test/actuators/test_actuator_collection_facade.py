# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for scoped actuator group and exact-type facade views."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest
import torch

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_control import ActuatorJointProperties
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, IdealPDActuatorCfg
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.utils.types import ArticulationActions


class _CustomIdealPD(IdealPDActuator):
    """Opaque custom actuator that deliberately has no exact managed schema."""


class _Simulation:
    def __init__(self) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool),
            cfg_rows={1: (0,)},
        )

    def get_clone_plan(self) -> ClonePlan:
        return self._clone_plan


class _Control:
    def __init__(self) -> None:
        self._joint_names = ("hip", "knee", "ankle")

    @property
    def num_instances(self) -> int:
        return 2

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def device(self) -> str:
        return "cpu"

    def discover_native_actuators(self, cfgs) -> set[str]:
        del cfgs
        return set()

    def find_joints(self, names):
        requested = set(names)
        indices = [index for index, name in enumerate(self._joint_names) if name in requested]
        return indices, [self._joint_names[index] for index in indices]

    def get_default_joint_properties(self, joint_ids):
        count = self.num_joints if isinstance(joint_ids, slice) else len(joint_ids)
        values = torch.zeros((self.num_instances, count))
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


def make_finalized_robot(*, groups=None):
    collection = ActuatorCollection(_Simulation())
    cfgs = groups or {"hip": _ideal_cfg(["hip", "knee"]), "knee": _dc_cfg(["knee"]), "ankle": _ideal_cfg(["ankle"])}
    view = collection.register_articulation(
        key="robot",
        cfgs=cfgs,
        control=_Control(),
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()
    return SimpleNamespace(collection=collection, actuators=view)


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
    with pytest.raises(KeyError):
        _ = robot.actuators["missing"]
    with pytest.raises(KeyError):
        _ = robot.actuators.by_type[_CustomIdealPD]
    with pytest.raises(KeyError):
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
    with pytest.raises(RuntimeError, match="rebuild"):
        _ = actuators.keys()

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
