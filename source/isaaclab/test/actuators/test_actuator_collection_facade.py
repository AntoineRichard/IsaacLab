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
import warp as wp

from isaaclab.actuators import ActuatorCollection
from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_control import ActuatorJointProperties
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.utils.types import ArticulationActions


class _CustomIdealPD(IdealPDActuator):
    """Opaque custom actuator that deliberately has no exact managed schema."""


class _Simulation:
    def __init__(self, device: str = "cpu") -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool, device=device),
            cfg_rows={1: (0,)},
        )

    def get_clone_plan(self) -> ClonePlan:
        return self._clone_plan


class _Control:
    def __init__(self, device: str = "cpu") -> None:
        self._joint_names = ("hip", "knee", "ankle")
        self._device = device
        self.parameter_writes = []

    @property
    def num_instances(self) -> int:
        return 2

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def device(self) -> str:
        return self._device

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


def _available_devices() -> tuple[str, ...]:
    return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)


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


def test_type_parameter_index_fans_out_overlapping_articulation_dofs() -> None:
    """Catch a type selector that updates only the first compact occurrence of a joint."""
    view = make_finalized_robot(
        groups={"first": _ideal_cfg(["hip", "knee"]), "second": _ideal_cfg(["knee", "ankle"])}
    ).actuators.by_type[IdealPDActuator]

    view.set_parameter_index("stiffness", torch.tensor([13.0]), joint_ids=torch.tensor([1]))

    torch.testing.assert_close(view.parameters["stiffness"].torch[:, [1, 2]], torch.full((2, 2), 13.0))


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

    with pytest.raises(KeyError, match="unknown"):
        group.set_parameter_index("unknown", 1.0)

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


@pytest.mark.parametrize("device", _available_devices())
def test_parameter_selectors_do_not_read_device_values_in_normal_mode(
    monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    """Catch a normal selector path that synchronizes to inspect device-resident selector contents."""
    view = make_finalized_robot(device=device).actuators.by_type[IdealPDActuator]

    def _read_forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("normal selector path read a device tensor")

    monkeypatch.setattr(torch.Tensor, "cpu", _read_forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", _read_forbidden)
    if device == "cuda":
        monkeypatch.setattr(torch.cuda, "synchronize", _read_forbidden)
        monkeypatch.setattr(wp, "synchronize_device", _read_forbidden)

    view.set_parameter_index(
        "stiffness",
        torch.tensor([[5.0], [7.0]], device=device),
        env_ids=torch.tensor([0, 1], device=device),
        joint_ids=torch.tensor([1], device=device),
    )


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


def test_native_group_parameter_write_delegates_without_affecting_ordinary_groups() -> None:
    """Catch native discovery metadata that is discarded before scoped writes reach the control bridge."""

    class _NativeControl(_Control):
        def discover_native_actuators(self, cfgs) -> set[str]:
            assert tuple(cfgs) == ("native", "ordinary")
            return {"native"}

    collection = ActuatorCollection(_Simulation())
    control = _NativeControl()
    view = collection.register_articulation(
        key="robot",
        cfgs={"native": _ideal_cfg(["hip"]), "ordinary": _ideal_cfg(["knee"])},
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
    view.by_type[IdealPDActuator].set_parameter_mask("stiffness", 11.0)
    assert control.parameter_writes[-1][0] == "stiffness"


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


def test_overlapping_implicit_parameter_writes_update_only_configuration_owner() -> None:
    """Catch backend routing that lets a non-owner overlapping implicit group overwrite a solver drive."""

    class _StatefulControl(_Control):
        def __init__(self) -> None:
            super().__init__()
            self.backend_stiffness = torch.ones((2, 3))

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

    collection = ActuatorCollection(_Simulation())
    control = _StatefulControl()
    view = collection.register_articulation(
        key="robot",
        cfgs={"first": _implicit_cfg(["hip", "knee"]), "last": _implicit_cfg(["knee", "ankle"])},
        control=control,
        replication_cfg_id=1,
        debug_validation=False,
        debug_value_resolution=False,
    )
    collection.finalize()

    view["first"].set_parameter_index("stiffness", 5.0, joint_ids=torch.tensor([1]))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.ones(2))
    view["last"].set_parameter_index("stiffness", 17.0, joint_ids=torch.tensor([1]))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 17.0))
    view["first"].set_parameter_mask("stiffness", 5.0, joint_mask=torch.tensor([False, True, False]))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 17.0))
    view["last"].set_parameter_mask("stiffness", 19.0, joint_mask=torch.tensor([False, True, False]))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 19.0))
    view.by_type[ImplicitActuator].set_parameter_index("stiffness", 21.0, joint_ids=torch.tensor([1]))
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 21.0))
    view.by_type[ImplicitActuator].set_parameter_mask(
        "stiffness", torch.tensor([1.0, 9.0, 23.0, 3.0]), joint_mask=torch.tensor([False, True, False])
    )
    torch.testing.assert_close(control.backend_stiffness[:, 1], torch.full((2,), 23.0))
