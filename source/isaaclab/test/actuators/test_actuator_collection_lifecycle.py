# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the transactional simulation-scoped actuator lifecycle.

Setup:
    - ./isaaclab.sh -p -m pytest source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py
Tests:
    - ./isaaclab.sh -p -m pytest
      source/isaaclab/test/actuators/test_actuator_collection_lifecycle.py
      -> verify failed completion rolls every registration back.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch

from isaaclab.actuators import ActuatorCollection, actuator_collection
from isaaclab.actuators.actuator_control import ActuatorJointProperties
from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator
from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.asset_base import AssetBase
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.service_locator import ServiceLocator
from isaaclab.sim.simulation_context import SimulationContext


@dataclass
class _Data:
    is_primed: bool = False


class _Control:
    def __init__(
        self,
        *,
        prepare_error: Exception | None = None,
        complete_error: Exception | None = None,
        invalidate_error: Exception | None = None,
        num_joints: int = 0,
        num_instances: int = 1,
        device: str = "cpu",
    ) -> None:
        self.prepare_error = prepare_error
        self.complete_error = complete_error
        self.invalidate_error = invalidate_error
        self.bind_count = 0
        self.bound_ready_states = []
        self.bound_alias_pointers = []
        self.bind_setter_rejected = False
        self.bind_compute_rejected = False
        self.staged_commands = []
        self.bound_staging = None
        self.staging_alive_during_invalidation = False
        self.complete_count = 0
        self.invalidate_count = 0
        self._num_joints = num_joints
        self._num_instances = num_instances
        self._device = device
        self.asset = type("_Asset", (), {"_is_initialized": False, "data": _Data()})()

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def device(self) -> str:
        return self._device

    def discover_native_actuators(self, cfgs) -> set[str]:
        return set()

    def find_joints(self, name_keys):
        del name_keys
        return list(range(self.num_joints)), ["wheel"] * self.num_joints

    def prepare_actuator_binding(self, binding) -> None:
        self.bound_staging = binding.backend_parameter_staging
        if self.prepare_error is not None:
            raise self.prepare_error

    def write_resolved_joint_properties_staged(self, properties) -> None:
        del properties

    def validate_resolved_joint_properties(self) -> None:
        pass

    def restore_resolved_joint_properties(self) -> None:
        pass

    def stage_user_command(self, command_name, collection, env_ids, joint_ids, env_mask, joint_mask) -> None:
        self.staged_commands.append((command_name, collection, env_ids, joint_ids, env_mask, joint_mask))

    def bind_actuator_view(self, view) -> None:
        self.bind_count += 1
        self.bound_ready_states.append(view.is_ready)
        self.bound_alias_pointers.append(
            (
                view.command.position.torch.data_ptr(),
                view.joint_command.position.torch.data_ptr(),
                view.computed_effort.torch.data_ptr(),
            )
        )
        with pytest.raises(RuntimeError, match="finalization is incomplete"):
            view.command.set_effort_index(value=torch.zeros(view.command.effort.torch.shape))
        self.bind_setter_rejected = True
        with pytest.raises(RuntimeError, match="finalization is incomplete"):
            view.compute()
        self.bind_compute_rejected = True

    def complete_articulation_initialization(self) -> None:
        self.complete_count += 1
        if self.complete_error is not None:
            raise self.complete_error
        self.asset._is_initialized = True
        self.asset.data.is_primed = True

    def invalidate_actuator_view(self) -> None:
        self.invalidate_count += 1
        if self.bound_staging is not None:
            self.staging_alive_during_invalidation = self.bound_staging._all_env_ids is not None
        self.asset._is_initialized = False
        self.asset.data.is_primed = False
        if self.invalidate_error is not None:
            raise self.invalidate_error


class _StructuredError(RuntimeError):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(code, message)
        self.code = code
        self.message = message


class _Simulation:
    def __init__(self) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 1), dtype=torch.bool),
            cfg_rows={1: (0,), 2: (0,)},
        )

    def get_clone_plan(self) -> ClonePlan:
        return self._clone_plan


class _VariantSimulation:
    """Two-source, four-world clone plan for source-resolution coverage."""

    def __init__(self) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0", "/World/envs/env_1"),
            destinations=("/World/envs/env_{}", "/World/envs/env_{}"),
            clone_mask=torch.tensor([[True, False, True, False], [False, True, False, True]]),
            cfg_rows={1: (0, 1)},
        )

    def get_clone_plan(self) -> ClonePlan:
        return self._clone_plan


class _PartialPresenceSimulation:
    """Five global clone columns with one articulation present in three backend rows."""

    def __init__(self) -> None:
        self._clone_plan = ClonePlan(
            sources=("/World/envs/env_0", "/World/envs/env_1"),
            destinations=("/World/envs/env_{}", "/World/envs/env_{}"),
            clone_mask=torch.tensor([[False, False, True, False, False], [True, False, False, False, True]]),
            cfg_rows={1: (0, 1)},
        )

    def get_clone_plan(self) -> ClonePlan:
        return self._clone_plan


class _DeferredAsset(AssetBase):
    @property
    def num_instances(self) -> int:
        return 0

    @property
    def data(self):
        return None

    def reset(self, env_ids=None) -> None:
        del env_ids

    def write_data_to_sim(self) -> None:
        pass

    def update(self, dt: float) -> None:
        del dt

    def _initialize_impl(self) -> None:
        pass


class _SourceResolvingControl(_Control):
    """Control double that supplies only source-prototype solver rows."""

    def __init__(self, *, num_instances: int = 4, device: str = "cpu") -> None:
        super().__init__(num_joints=2, num_instances=num_instances, device=device)
        self.source_resolution_count = 0
        self.requested_source_env_ids = None
        self.solver_property_write_count = 0
        self.saw_unready_solver_write = False
        self.solver_properties = {}
        self.native_groups = set()

    def discover_native_actuators(self, cfgs) -> set[str]:
        return self.native_groups.intersection(cfgs)

    def find_joints(self, name_keys):
        del name_keys
        return [0, 1], ["joint_0", "joint_1"]

    def get_source_joint_properties(self, joint_ids, source_env_ids):
        assert joint_ids.shape == (2,)
        assert source_env_ids.shape == (2,)
        self.source_resolution_count += 1
        self.requested_source_env_ids = source_env_ids

        def _row(stiffness: float) -> ActuatorJointProperties:
            values = torch.tensor([[stiffness, stiffness + 1.0]], dtype=torch.float32)
            return ActuatorJointProperties(
                stiffness=values,
                damping=torch.full_like(values, 2.0),
                armature=torch.full_like(values, 0.1),
                friction=torch.full_like(values, 0.2),
                dynamic_friction=torch.full_like(values, 0.3),
                viscous_friction=torch.full_like(values, 0.4),
                effort_limit=torch.full_like(values, 100.0),
                velocity_limit=torch.full_like(values, 20.0),
            )

        rows = tuple(_row(1.0 if source_env_id == 0 else 11.0) for source_env_id in source_env_ids.tolist())
        first, second = rows
        return ActuatorJointProperties(
            stiffness=torch.cat((first.stiffness, second.stiffness)),
            damping=torch.cat((first.damping, second.damping)),
            armature=torch.cat((first.armature, second.armature)),
            friction=torch.cat((first.friction, second.friction)),
            dynamic_friction=torch.cat((first.dynamic_friction, second.dynamic_friction)),
            viscous_friction=torch.cat((first.viscous_friction, second.viscous_friction)),
            effort_limit=torch.cat((first.effort_limit, second.effort_limit)),
            velocity_limit=torch.cat((first.velocity_limit, second.velocity_limit)),
        )

    def get_default_joint_properties(self, joint_ids):
        del joint_ids
        raise AssertionError("candidate construction must resolve source rows before world-sized defaults")

    def write_resolved_joint_properties_staged(self, properties) -> None:
        self.solver_property_write_count += len(properties.properties)
        self.saw_unready_solver_write = not self.asset._is_initialized and not self.asset.data.is_primed
        self.solver_properties = dict(properties.properties)


class _OpaqueIdealPD(IdealPDActuator):
    """Exact third-party-style actuator intentionally outside managed storage."""


class _RoutedSourceControl(_SourceResolvingControl):
    """Source fixture with configurable group joint coverage and backend routing."""

    def __init__(self, joint_groups: dict[str, tuple[int, ...]], *, device: str = "cpu") -> None:
        super().__init__(num_instances=4, device=device)
        self._joint_groups = joint_groups
        self.backend_writes = []

    def find_joints(self, name_keys):
        joint_ids = self._joint_groups[name_keys[0]]
        return list(joint_ids), [f"joint_{joint_id}" for joint_id in joint_ids]

    def get_source_joint_properties(self, joint_ids, source_env_ids):
        self.source_resolution_count += 1
        self.requested_source_env_ids = source_env_ids
        columns = joint_ids.shape[0]
        stiffness = (source_env_ids.to(dtype=torch.float32).reshape(-1, 1) * 10.0 + 1.0).expand(-1, columns)
        return ActuatorJointProperties(
            stiffness=stiffness,
            damping=stiffness + 1.0,
            armature=stiffness + 2.0,
            friction=stiffness + 3.0,
            dynamic_friction=stiffness + 4.0,
            viscous_friction=stiffness + 5.0,
            effort_limit=stiffness + 100.0,
            velocity_limit=stiffness + 20.0,
        )

    def get_default_joint_properties(self, joint_ids):
        columns = joint_ids.shape[0]
        values = torch.full((self.num_instances, columns), 7.0, dtype=torch.float32, device=self.device)
        return ActuatorJointProperties(
            stiffness=values,
            damping=values + 1.0,
            armature=values + 2.0,
            friction=values + 3.0,
            dynamic_friction=values + 4.0,
            viscous_friction=values + 5.0,
            effort_limit=values + 100.0,
            velocity_limit=values + 20.0,
        )

    def write_actuator_parameter(self, name, write) -> None:
        self.backend_writes.append((name, write))
        if write.backend_parameter_staging is not None:
            write.backend_parameter_staging.patch_write(
                actuator_type=ImplicitActuator,
                name=name,
                write=write,
            )


class _FakeCallbackHandle:
    def __init__(self, callbacks, entry) -> None:
        self._callbacks = callbacks
        self._entry = entry

    def deregister(self) -> None:
        self._callbacks.remove(self._entry)


class _FakePhysicsManager:
    callbacks = []

    @classmethod
    def reset(cls) -> None:
        cls.callbacks = []

    @classmethod
    def _prepare_stage_creation(cls) -> None:
        pass

    @classmethod
    def initialize(cls, context) -> None:
        del context

    @classmethod
    def get_scene_data_backend(cls):
        return object()

    @classmethod
    def register_callback(cls, callback, event, order=0, name=None, wrap_weak_ref=True):
        del wrap_weak_ref
        entry = SimpleNamespace(callback=callback, event=event, order=order, name=name)
        cls.callbacks.append(entry)
        return _FakeCallbackHandle(cls.callbacks, entry)

    @classmethod
    def dispatch(cls, event) -> None:
        snapshot = sorted((entry for entry in cls.callbacks if entry.event == event), key=lambda entry: entry.order)
        for entry in snapshot:
            entry.callback(None)

    @classmethod
    def close(cls) -> None:
        cls.callbacks = []


@pytest.fixture(autouse=True)
def _reset_fake_context_singleton():
    """Prevent fake SimulationContext instances from leaking between assertions."""
    yield
    SimulationContext._instance = None
    _FakePhysicsManager.reset()


def _make_fake_context(monkeypatch) -> SimulationContext:
    class _StageCache:
        @staticmethod
        def Get():
            return _StageCache()

        def GetId(self, stage):
            del stage
            return SimpleNamespace(ToLongInt=lambda: 1)

        def Insert(self, stage):
            del stage

        def Size(self):
            return 0

    class _Settings:
        def get(self, name):
            return 0

        def set_bool(self, name, value):
            del name, value

        def set_int(self, name, value):
            del name, value

        def set_float(self, name, value):
            del name, value

        def set_string(self, name, value):
            del name, value

        def set(self, name, value):
            del name, value

    pxr = ModuleType("pxr")
    pxr.UsdUtils = SimpleNamespace(StageCache=_StageCache)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setattr("isaaclab.sim.simulation_context.has_kit", lambda: False)
    monkeypatch.setattr("isaaclab.sim.simulation_context.create_new_stage", lambda: object())
    monkeypatch.setattr("isaaclab.sim.simulation_context.SimulationContext._init_usd_physics_scene", lambda self: None)
    monkeypatch.setattr("isaaclab.sim.simulation_context.SceneDataProvider", lambda backend: object())
    monkeypatch.setattr(
        "isaaclab.sim.simulation_context.RenderContext", lambda: SimpleNamespace(ensure_initialize=lambda: None)
    )
    monkeypatch.setattr("isaaclab.sim.simulation_context.VisMarkerRegistry", lambda: SimpleNamespace())
    monkeypatch.setattr("isaaclab.sim.simulation_context.SettingsManager.instance", lambda: _Settings())
    monkeypatch.setattr("isaaclab.sim.simulation_context._resolve_physics_cfg", lambda physics, use_isaac_sim: physics)
    monkeypatch.setattr("isaaclab.sim.simulation_context.stage_utils.close_stage", lambda: None)
    monkeypatch.setattr("isaaclab.sim.simulation_context.clear_resolve_matching_names_cache", lambda: None)
    _FakePhysicsManager.reset()
    SimulationContext._instance = None
    cfg = SimpleNamespace(
        physics=SimpleNamespace(class_type=_FakePhysicsManager, dt=0.01),
        create_stage_in_memory=True,
        device="cpu",
        dt=0.01,
        render_interval=1,
    )
    return SimulationContext(cfg)


def _register(collection: ActuatorCollection, key: object, control: _Control):
    return collection.register_articulation(
        key=key,
        cfgs={},
        control=control,
        replication_cfg_id=1 if key == "first" else 2,
        debug_validation=False,
        debug_value_resolution=False,
    )


def _managed_cfgs(actuator_type=IdealPDActuator) -> dict[str, SimpleNamespace]:
    return {"wheel": SimpleNamespace(class_type=actuator_type, joint_names_expr=["wheel"])}


def _register_managed(collection: ActuatorCollection, key: object, control: _Control, cfgs=None):
    return collection.register_articulation(
        key=key,
        cfgs=_managed_cfgs() if cfgs is None else cfgs,
        control=control,
        replication_cfg_id=1 if key == "first" else 2,
        debug_validation=False,
        debug_value_resolution=False,
    )


def test_pending_view_rejects_runtime_access_before_publication() -> None:
    collection = ActuatorCollection(_Simulation())
    view = _register(collection, "first", _Control())

    with pytest.raises(RuntimeError, match="pending actuator view"):
        _ = view.command


def test_failed_finalization_publishes_no_partial_generation() -> None:
    collection = ActuatorCollection(_Simulation())
    good = _register(collection, "first", _Control())
    bad = _register(collection, "second", _Control(prepare_error=ValueError("bad IdealPDActuator wheel")))

    with pytest.raises(ValueError, match="bad IdealPDActuator wheel"):
        collection.finalize()

    assert collection.generation is None
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = good.command
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = bad.command


def test_failed_invalidator_does_not_prevent_other_controls_or_candidate_close() -> None:
    """One invalidation failure must not mask the trigger or retain another binding."""
    collection = ActuatorCollection(_Simulation())
    first_control = _Control(invalidate_error=RuntimeError("first invalidation"))
    second_control = _Control(complete_error=RuntimeError("completion trigger"))
    first = _register(collection, "first", first_control)
    second = _register(collection, "second", second_control)

    with pytest.raises(RuntimeError, match="completion trigger") as caught:
        collection.finalize()

    assert first_control.invalidate_count == second_control.invalidate_count == 1
    assert any("first invalidation" in note for note in caught.value.__notes__)
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = first.command
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = second.command


def test_partial_solver_write_enrolls_for_best_effort_restore_before_candidate_close() -> None:
    """A failing staged write must still restore itself and every earlier control."""

    class _PartialSolverWriteControl(_Control):
        def __init__(self, *, write_error: Exception | None = None, restore_error: Exception | None = None) -> None:
            super().__init__()
            self.write_error = write_error
            self.restore_error = restore_error
            self.partial_target = None
            self.restore_count = 0
            self.restore_saw_live_target = False

        def write_resolved_joint_properties_staged(self, properties) -> None:
            self.partial_target = properties.properties["stiffness"].canonical_target
            assert self.partial_target is not None
            self.partial_target.torch.fill_(17.0)
            if self.write_error is not None:
                raise self.write_error

        def restore_resolved_joint_properties(self) -> None:
            self.restore_count += 1
            self.restore_saw_live_target = self.partial_target is not None and self.partial_target.torch.shape == (1, 0)
            if self.restore_error is not None:
                raise self.restore_error

    collection = ActuatorCollection(_Simulation())
    first_control = _PartialSolverWriteControl()
    second_control = _PartialSolverWriteControl(
        write_error=RuntimeError("partial solver write"),
        restore_error=RuntimeError("later restore failure"),
    )
    _register(collection, "first", first_control)
    _register(collection, "second", second_control)

    with pytest.raises(RuntimeError, match="partial solver write") as caught:
        collection.finalize()

    assert first_control.restore_count == second_control.restore_count == 1
    assert first_control.restore_saw_live_target and second_control.restore_saw_live_target
    assert any("later restore failure" in note for note in caught.value.__notes__)


def test_second_completion_failure_rolls_back_first_completed_asset(monkeypatch) -> None:
    collection = ActuatorCollection(_Simulation())
    first_control = _Control(num_joints=1)
    second_control = _Control(complete_error=RuntimeError("second completion"), num_joints=1)
    first = _register_managed(collection, "first", first_control)
    second = _register_managed(collection, "second", second_control)
    allocated_stores = []
    allocate = actuator_collection._TypedStore.allocate

    def record_allocation(store, layouts, *, device):
        allocated_stores.append(store)
        allocate(store, layouts, device=device)

    monkeypatch.setattr(actuator_collection._TypedStore, "allocate", record_allocation)

    with pytest.raises(RuntimeError, match="second completion") as caught:
        collection.finalize()

    assert collection.generation is None
    assert not first_control.asset._is_initialized and not first_control.asset.data.is_primed
    assert not second_control.asset._is_initialized and not second_control.asset.data.is_primed
    assert first_control.invalidate_count == second_control.invalidate_count == 1
    # No actuator parameter has a backend route in this fixture, so the
    # candidate intentionally avoids allocating a staging object at all.
    assert first_control.bound_staging is None
    assert second_control.bound_staging is None
    assert allocated_stores and all(not store._fields for store in allocated_stores)
    assert any("wheel (IdealPDActuator)" in note for note in caught.value.__notes__)
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = first.command
    with pytest.raises(RuntimeError, match="finalization failed"):
        _ = second.command


def test_completion_preserves_structured_exception_and_adds_binding_context() -> None:
    collection = ActuatorCollection(_Simulation())
    error = _StructuredError(17, "structured completion")
    _register_managed(collection, "first", _Control(num_joints=1))
    _register_managed(collection, "second", _Control(complete_error=error, num_joints=1))

    with pytest.raises(_StructuredError) as caught:
        collection.finalize()

    assert caught.value is error
    assert caught.value.args == (17, "structured completion")
    assert any("articulation 'second'" in note for note in caught.value.__notes__)
    assert any("wheel (IdealPDActuator)" in note for note in caught.value.__notes__)


def test_same_type_registrations_use_disjoint_global_storage() -> None:
    collection = ActuatorCollection(_Simulation())
    first = _register_managed(collection, "first", _Control(num_joints=1))
    second = _register_managed(collection, "second", _Control(num_joints=1))

    collection.finalize()

    assert first.is_ready and second.is_ready
    bindings = collection._active_generation.bindings
    first_slice = bindings[0].layout.type_layouts[IdealPDActuator].global_slice
    second_slice = bindings[1].layout.type_layouts[IdealPDActuator].global_slice
    assert first_slice.stop <= second_slice.start


def test_bind_callback_reads_aliases_before_readiness_but_cannot_execute() -> None:
    """Publication exposes aliases before completion without allowing execution."""
    collection = ActuatorCollection(_Simulation())
    control = _Control(num_joints=1)
    view = _register_managed(collection, "first", control)

    collection.finalize()

    assert control.bound_ready_states == [False]
    assert control.bind_setter_rejected and control.bind_compute_rejected
    assert len(control.bound_alias_pointers) == 1
    assert view.is_ready and collection.is_finalized


def test_scoped_command_stages_normalized_empty_index_and_all_false_mask() -> None:
    """Raw command staging follows every successful canonical write attempt exactly once."""
    collection = ActuatorCollection(_Simulation())
    control = _Control(num_joints=1)
    view = _register_managed(collection, "first", control)
    collection.finalize()
    raw_before = view.command.position.torch.clone()

    view.command.set_position_index(
        value=torch.empty((0, 1), dtype=torch.float32),
        env_ids=torch.empty(0, dtype=torch.int32),
        joint_ids=torch.tensor([0], dtype=torch.int32),
    )
    view.command.set_velocity_mask(
        value=torch.full((1, 1), 3.0),
        env_mask=torch.tensor([False]),
        joint_mask=torch.tensor([False]),
    )

    assert [event[0] for event in control.staged_commands] == ["position", "velocity"]
    assert all(event[1] is view for event in control.staged_commands)
    assert control.staged_commands[0][2].shape == (0,)
    assert control.staged_commands[1][4].dtype is torch.bool
    torch.testing.assert_close(view.command.position.torch, raw_before)
    with pytest.raises(ValueError, match="shape"):
        view.command.set_effort_index(value=torch.zeros((2, 1)))
    assert [event[0] for event in control.staged_commands] == ["position", "velocity"]


def test_source_prototype_rows_expand_into_one_exact_runtime_group() -> None:
    """Source-only backend rows must expand before the final group shell is built."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _SourceResolvingControl()
    cfg = IdealPDActuatorCfg(
        class_type=IdealPDActuator,
        joint_names_expr=[".*"],
        stiffness=None,
        damping=None,
    )
    view = _register_managed(collection, "first", control, {"drive": cfg})

    collection.finalize()

    assert control.source_resolution_count == 1
    torch.testing.assert_close(control.requested_source_env_ids, torch.tensor([0, 1]))
    assert isinstance(view["drive"], IdealPDActuator)
    assert view["drive"]._solver_compatibility_sidecars == {}
    torch.testing.assert_close(
        view["drive"].stiffness,
        torch.tensor([[1.0, 2.0], [11.0, 12.0], [1.0, 2.0], [11.0, 12.0]]),
    )
    torch.testing.assert_close(view["drive"].computed_effort, torch.zeros((4, 2)))


def test_successful_finalization_releases_build_only_solver_staging() -> None:
    """Successful publication keeps runtime gain staging but releases solver build state."""

    class _CommitControl(_SourceResolvingControl):
        def __init__(self) -> None:
            super().__init__()
            self.commit_count = 0

        def commit_resolved_joint_properties(self) -> None:
            self.commit_count += 1

    collection = ActuatorCollection(_VariantSimulation())
    control = _CommitControl()
    cfg = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=[".*"],
        stiffness=None,
        damping=None,
    )
    facade = _register_managed(collection, "first", control, {"drive": cfg})

    collection.finalize()

    generation = collection._active_generation
    assert generation is not None
    assert control.commit_count == 1
    assert generation._solver_properties_written == []
    assert generation.managed_group_resolutions == {}
    assert generation.solver_store._fields == {}
    assert generation.solver_store._source_rows == {}
    staging = facade._backend_parameter_staging
    assert staging is not None
    assert staging.target(ImplicitActuator, "stiffness").torch.shape == (4, 2)


def test_source_solver_rowset_writes_once_per_property_with_config_order_overlap() -> None:
    """One solver rowset must use the final config owner before publication."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _SourceResolvingControl()
    explicit = IdealPDActuatorCfg(
        class_type=IdealPDActuator,
        joint_names_expr=[".*"],
        stiffness=None,
        damping=None,
    )
    implicit = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=[".*"],
        stiffness=None,
        damping=None,
    )
    _register_managed(collection, "first", control, {"explicit": explicit, "implicit": implicit})

    collection.finalize()

    assert control.source_resolution_count == 1
    torch.testing.assert_close(control.requested_source_env_ids, torch.tensor([0, 1]))
    assert control.solver_property_write_count == 8
    assert control.saw_unready_solver_write
    stiffness = control.solver_properties["stiffness"]
    assert stiffness.transport == "device"
    assert stiffness.source_slot_by_backend_row is None
    torch.testing.assert_close(stiffness.source_rows, torch.tensor([[1.0, 2.0], [11.0, 12.0]]))
    torch.testing.assert_close(stiffness.source_assignment, torch.tensor([0, 1, 0, 1], dtype=torch.int32))
    torch.testing.assert_close(
        stiffness.canonical_target.torch,
        torch.tensor([[1.0, 2.0], [11.0, 12.0], [1.0, 2.0], [11.0, 12.0]]),
    )


@pytest.mark.parametrize(
    ("cfg", "actuator_type"),
    [
        (
            IdealPDActuatorCfg(
                joint_names_expr=["all"],
                stiffness=None,
                damping=None,
                effort_limit=None,
                velocity_limit=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            ),
            IdealPDActuator,
        ),
        (
            ImplicitActuatorCfg(
                joint_names_expr=["all"],
                stiffness=None,
                damping=None,
                effort_limit=None,
                velocity_limit=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            ),
            ImplicitActuator,
        ),
        (
            DCMotorCfg(
                joint_names_expr=["all"],
                stiffness=None,
                damping=None,
                effort_limit=None,
                velocity_limit=20.0,
                effort_limit_sim=None,
                velocity_limit_sim=None,
                saturation_effort=100.0,
            ),
            DCMotor,
        ),
    ],
)
def test_managed_collection_resolves_builtin_lazy_cfg_classes_without_mutating_configs(cfg, actuator_type) -> None:
    """Use resolved exact classes internally while preserving the caller's lazy config surface."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _RoutedSourceControl({"all": (0, 1)})
    view = _register_managed(collection, "first", control, {"drive": cfg})
    original_class_type = cfg.class_type

    collection.finalize()

    assert type(view["drive"]) is actuator_type
    assert isinstance(cfg.class_type, str)
    assert cfg.class_type is original_class_type
    assert view["drive"].joint_indices == slice(None)


@pytest.mark.parametrize("explicit_type", [IdealPDActuator, _OpaqueIdealPD])
@pytest.mark.parametrize("explicit_after", [False, True])
def test_explicit_config_order_blocks_or_enables_implicit_backend_gain_routes(explicit_type, explicit_after) -> None:
    """Only the final config owner may route an implicit gain write to the backend."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _RoutedSourceControl({"both": (0, 1)})
    implicit = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=["both"],
        stiffness=5.0,
        damping=6.0,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    explicit = IdealPDActuatorCfg(
        class_type=explicit_type,
        joint_names_expr=["both"],
        stiffness=9.0,
        damping=10.0,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    ordered_cfgs = {"explicit": explicit, "implicit": implicit}
    if explicit_after:
        ordered_cfgs = {"implicit": implicit, "explicit": explicit}
    view = _register_managed(collection, "first", control, ordered_cfgs)

    collection.finalize()
    view["implicit"].set_parameter_index("stiffness", 33.0)

    if explicit_after:
        assert control.backend_writes == []
        assert view._backend_parameter_staging is None
    else:
        assert len(control.backend_writes) == 1
        staging = view._backend_parameter_staging
        assert staging is not None
        torch.testing.assert_close(
            staging.target(ImplicitActuator, "stiffness").torch,
            torch.full((4, 2), 33.0),
        )


def test_partial_implicit_gain_write_preserves_later_opaque_solver_defaults() -> None:
    """Persistent gain shadows must start from final opaque-overlaid source rows."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _RoutedSourceControl({"implicit": (0,), "opaque": (1,)})
    implicit = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=["implicit"],
        stiffness=5.0,
        damping=6.0,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    opaque = IdealPDActuatorCfg(
        class_type=_OpaqueIdealPD,
        joint_names_expr=["opaque"],
        stiffness=9.0,
        damping=10.0,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    view = _register_managed(collection, "first", control, {"implicit": implicit, "opaque": opaque})

    collection.finalize()
    view["implicit"].set_parameter_index("stiffness", 33.0)

    staging = view._backend_parameter_staging
    assert staging is not None
    torch.testing.assert_close(
        staging.target(ImplicitActuator, "stiffness").torch,
        torch.tensor([[33.0, 0.0], [33.0, 0.0], [33.0, 0.0], [33.0, 0.0]]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for lazy compatibility-device coverage")
def test_solver_compatibility_fields_expand_lazily_to_runtime_cuda_shape() -> None:
    """Lazy solver fields retain compact CPU rows but expose runtime-shaped CUDA values."""

    class _CpuSourceControl(_RoutedSourceControl):
        def resolved_solver_property_transport(self) -> str:
            return "cpu"

    collection = ActuatorCollection(_VariantSimulation())
    control = _CpuSourceControl({"all": (0, 1)}, device="cuda")
    cfg = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=["all"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    view = _register_managed(collection, "first", control, {"drive": cfg})

    collection.finalize()

    drive = view["drive"]
    expected_bases = {
        "effort_limit_sim": (101.0, 111.0),
        "velocity_limit_sim": (21.0, 31.0),
        "armature": (3.0, 13.0),
        "friction": (4.0, 14.0),
        "dynamic_friction": (5.0, 15.0),
        "viscous_friction": (6.0, 16.0),
    }
    for name, (first_source, second_source) in expected_bases.items():
        value = getattr(drive, name)
        assert value.shape == (4, 2)
        assert value.device.type == "cuda"
        torch.testing.assert_close(
            value,
            torch.tensor(
                [
                    [first_source, first_source],
                    [second_source, second_source],
                    [first_source, first_source],
                    [second_source, second_source],
                ],
                device="cuda",
            ),
        )
    assert set(drive._solver_compatibility_sidecars) == set(drive._SOLVER_COMPATIBILITY_PARAMETER_NAMES)


def test_cpu_solver_transport_keeps_all_eight_properties_compact_during_candidate_build() -> None:
    """CPU transport must not allocate candidate-wide dense solver fields before publication."""

    class _CpuSourceControl(_RoutedSourceControl):
        def resolved_solver_property_transport(self) -> str:
            return "cpu"

    collection = ActuatorCollection(_VariantSimulation())
    control = _CpuSourceControl({"all": (0, 1)})
    cfg = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=["all"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    _register_managed(collection, "first", control, {"drive": cfg})

    candidate = actuator_collection._CollectionGeneration.build(
        tuple(collection._registrations), collection._sim_context, generation=0
    )
    try:
        assert candidate.solver_store._fields == {}
        assert len(candidate.solver_store._source_rows) == 8
    finally:
        candidate.close()


def test_published_generation_reuses_selector_defaults_and_releases_build_only_tensors() -> None:
    """Persistent gain staging reuses selector slabs and drops build-only source buffers."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _RoutedSourceControl({"all": (0, 1)})
    cfg = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=["all"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    view = _register_managed(collection, "first", control, {"drive": cfg})

    collection.finalize()

    generation = collection._active_generation
    assert generation is not None
    selector_state = generation.selector_states["first"]
    staging = view._backend_parameter_staging
    assert staging is not None
    assert staging._all_env_ids.data_ptr() == selector_state._all_env_ids.data_ptr()
    assert staging._all_joint_ids.data_ptr() == selector_state._all_joint_ids.data_ptr()
    assert staging._all_env_mask.data_ptr() == selector_state._all_env_mask.data_ptr()
    assert staging._all_joint_mask.data_ptr() == selector_state._all_joint_mask.data_ptr()
    assert all(store._initialization_buffers == [] for store in generation.stores.values())
    shadowed_names = {
        *(field.name for field in ImplicitActuator._parameter_schema().fields),
        *view["drive"]._SOLVER_COMPATIBILITY_PARAMETER_NAMES,
    }
    assert all(name not in view["drive"].__dict__ for name in shadowed_names)


def test_solver_compatibility_seeds_share_articulation_rows_until_first_access() -> None:
    """Keep source rows shared across groups and consume only the accessed compatibility seed."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _RoutedSourceControl({"first": (0,), "second": (1,)})
    first_cfg = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=["first"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    second_cfg = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=["second"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    view = _register_managed(collection, "first", control, {"first": first_cfg, "second": second_cfg})

    collection.finalize()

    first_seed = view["first"]._solver_compatibility_seeds["armature"]
    second_seed = view["second"]._solver_compatibility_seeds["armature"]
    assert first_seed.source_rows.data_ptr() == second_seed.source_rows.data_ptr()
    assert first_seed.source_joint_indices.data_ptr() != second_seed.source_joint_indices.data_ptr()

    _ = view["first"].armature

    assert "armature" not in view["first"]._solver_compatibility_seeds
    assert "armature" in view["second"]._solver_compatibility_seeds


def test_stateless_execution_signature_ignores_group_joint_names() -> None:
    """Stateless exact classes must not split execution compatibility by group names."""
    control = _RoutedSourceControl({"first": (0,), "second": (1,)})
    defaults = control.get_source_joint_properties(
        torch.tensor([0], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int64)
    )
    cfg = IdealPDActuatorCfg(
        class_type=IdealPDActuator,
        joint_names_expr=["first"],
        stiffness=None,
        damping=None,
        effort_limit=None,
        velocity_limit=None,
        effort_limit_sim=None,
        velocity_limit_sim=None,
    )
    first = IdealPDActuator._resolve_managed_registration(
        cfg=cfg,
        joint_names=["hip"],
        joint_indices=torch.tensor([0], dtype=torch.int32),
        defaults_by_source=defaults,
    )
    second = IdealPDActuator._resolve_managed_registration(
        cfg=cfg,
        joint_names=["knee"],
        joint_indices=torch.tensor([1], dtype=torch.int32),
        defaults_by_source=defaults,
    )

    assert first.structural_signature == second.structural_signature


def test_partial_clone_presence_uses_compact_backend_source_slots() -> None:
    """Source metadata must compact global clone columns before solver-property transport."""

    class _CpuSourceControl(_SourceResolvingControl):
        def resolved_solver_property_transport(self) -> str:
            return "cpu"

    collection = ActuatorCollection(_PartialPresenceSimulation())
    control = _CpuSourceControl(num_instances=3)
    cfg = ImplicitActuatorCfg(
        class_type=ImplicitActuator,
        joint_names_expr=[".*"],
        stiffness=None,
        damping=None,
    )
    _register_managed(collection, "first", control, {"drive": cfg})

    collection.finalize()

    stiffness = control.solver_properties["stiffness"]
    assert stiffness.transport == "cpu"
    torch.testing.assert_close(control.requested_source_env_ids, torch.tensor([1, 0], dtype=torch.int64))
    torch.testing.assert_close(stiffness.source_slot_by_backend_row, torch.tensor([1, 0, 1], dtype=torch.int64))
    torch.testing.assert_close(stiffness.source_rows, torch.tensor([[11.0, 12.0], [1.0, 2.0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for opaque CPU-transport coverage")
def test_opaque_cuda_constructor_uses_one_bounded_cpu_source_row_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opaque CUDA values remain compatible with CPU solver transport through compact source rows."""

    class _CpuOpaqueControl(_SourceResolvingControl):
        def __init__(self) -> None:
            super().__init__(num_instances=3, device="cuda")

        def resolved_solver_property_transport(self) -> str:
            return "cpu"

        def get_default_joint_properties(self, joint_ids):
            del joint_ids
            world_values = torch.tensor([10.0, 20.0, 10.0], dtype=torch.float32, device="cuda").reshape(-1, 1)
            values = world_values.expand(-1, 2)
            return ActuatorJointProperties(
                stiffness=torch.zeros_like(values),
                damping=torch.zeros_like(values),
                armature=torch.zeros_like(values),
                friction=values,
                dynamic_friction=values + 1.0,
                viscous_friction=values + 2.0,
                effort_limit=torch.full_like(values, 100.0),
                velocity_limit=torch.full_like(values, 20.0),
            )

    collection = ActuatorCollection(_PartialPresenceSimulation())
    control = _CpuOpaqueControl()
    cfg = IdealPDActuatorCfg(
        class_type=_OpaqueIdealPD,
        joint_names_expr=[".*"],
        stiffness=None,
        damping=None,
        friction=None,
    )
    _register_managed(collection, "first", control, {"drive": cfg})

    cuda_to_cpu_transfers = []
    original_to = torch.Tensor.to

    def _record_to(self, *args, **kwargs):
        requested_device = kwargs.get("device", args[0] if args else None)
        if requested_device is not None and self.device.type == "cuda" and torch.device(requested_device).type == "cpu":
            cuda_to_cpu_transfers.append(tuple(self.shape))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _record_to)
    collection.finalize()

    friction = control.solver_properties["friction"]
    assert friction.transport == "cpu"
    assert friction.source_rows.device.type == "cpu"
    torch.testing.assert_close(friction.source_rows, torch.tensor([[20.0, 20.0], [10.0, 10.0]]))
    assert cuda_to_cpu_transfers == [(2, 12)]


@pytest.mark.parametrize(
    ("cfgs", "winner_type"),
    [
        (
            ("ideal", "implicit"),
            ImplicitActuator,
        ),
        (
            ("implicit", "ideal"),
            IdealPDActuator,
        ),
    ],
)
def test_backend_owner_slots_use_the_last_cross_type_config_group(cfgs, winner_type) -> None:
    """Cross-type backend routes must clear earlier owners when config order overlaps."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _SourceResolvingControl()
    control.native_groups = {"ideal", "implicit"}
    configs = {
        "ideal": IdealPDActuatorCfg(
            class_type=IdealPDActuator,
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
        "implicit": ImplicitActuatorCfg(
            class_type=ImplicitActuator,
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
        ),
    }
    _register_managed(collection, "first", control, {name: configs[name] for name in cfgs})

    collection.finalize()

    owners = collection._active_generation.selector_states["first"]._backend_owner_slots
    for actuator_type in (IdealPDActuator, ImplicitActuator):
        slots = owners[(actuator_type, "stiffness")]
        if actuator_type is winner_type:
            assert torch.all(slots >= 0)
        else:
            torch.testing.assert_close(slots, torch.full((2,), -1, dtype=torch.int32))


@pytest.mark.parametrize(
    ("first_type", "second_type", "fields"),
    [
        (IdealPDActuator, IdealPDActuator, ("stiffness", "damping", "effort_limit", "velocity_limit")),
        (DCMotor, DCMotor, ("saturation_effort",)),
    ],
)
def test_later_non_native_group_blocks_every_semantically_owned_backend_field(first_type, second_type, fields) -> None:
    """A later non-native group owns its schema fields even without a backend route."""
    collection = ActuatorCollection(_VariantSimulation())
    control = _SourceResolvingControl()
    control.native_groups = {"native"}
    cfg_type = DCMotorCfg if first_type is DCMotor else IdealPDActuatorCfg
    config_kwargs = {"stiffness": None, "damping": None}
    if first_type is DCMotor:
        config_kwargs.update(effort_limit=100.0, velocity_limit=20.0, saturation_effort=10.0)
    cfgs = {
        "native": cfg_type(class_type=first_type, joint_names_expr=[".*"], **config_kwargs),
        "later": cfg_type(class_type=second_type, joint_names_expr=[".*"], **config_kwargs),
    }
    _register_managed(collection, "first", control, cfgs)

    collection.finalize()

    owners = collection._active_generation.selector_states["first"]._backend_owner_slots
    for field in fields:
        torch.testing.assert_close(owners[(first_type, field)], torch.full((2,), -1, dtype=torch.int32))


def test_failed_retry_then_stop_invalidates_every_registration() -> None:
    collection = ActuatorCollection(_Simulation())
    first_control = _Control(complete_error=RuntimeError("retry completion"))
    second_control = _Control()
    _register(collection, "first", first_control)
    _register(collection, "second", second_control)

    with pytest.raises(RuntimeError, match="retry completion"):
        collection.finalize()
    first_control.complete_error = None
    collection.finalize()
    collection.clear_generation()

    assert first_control.invalidate_count == second_control.invalidate_count == 2
    assert not first_control.asset._is_initialized and not first_control.asset.data.is_primed
    assert not second_control.asset._is_initialized and not second_control.asset.data.is_primed


def test_partial_store_allocation_failure_releases_earlier_store(monkeypatch) -> None:
    collection = ActuatorCollection(_Simulation())
    _register_managed(collection, "first", _Control(num_joints=1), _managed_cfgs(IdealPDActuator))
    _register_managed(collection, "second", _Control(num_joints=1), _managed_cfgs(ImplicitActuator))
    created_stores = []
    allocate = actuator_collection._TypedStore.allocate

    def fail_after_first_allocation(store, layouts, *, device):
        created_stores.append(store)
        if len(created_stores) == 2:
            raise RuntimeError("second store allocation")
        allocate(store, layouts, device=device)

    monkeypatch.setattr(actuator_collection._TypedStore, "allocate", fail_after_first_allocation)

    with pytest.raises(RuntimeError, match="second store allocation"):
        collection.finalize()

    assert len(created_stores) == 2
    assert all(not store._fields for store in created_stores)


def test_late_registration_marks_active_generation_dirty_until_replay() -> None:
    collection = ActuatorCollection(_Simulation())
    first = _register(collection, "first", _Control())
    collection.finalize()
    _register(collection, "second", _Control())

    with pytest.raises(RuntimeError, match="late registration.*rebuild"):
        first.compute()


def test_clear_generation_invalidates_old_facade_before_next_generation() -> None:
    collection = ActuatorCollection(_Simulation())
    first = _register(collection, "first", _Control())
    collection.finalize()
    old_generation = first.generation
    collection.clear_generation()

    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = first.command

    replacement = _register(collection, "first", _Control())
    collection.finalize()
    assert replacement.generation == old_generation + 1


def test_stop_replay_invalidates_old_facade_and_builds_new_generation() -> None:
    collection = ActuatorCollection(_Simulation())
    old = _register(collection, "first", _Control())
    collection.finalize()
    old_generation = old.generation

    collection.clear_generation()
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = old.command

    replacement = _register(collection, "first", _Control())
    collection.finalize()
    assert replacement.generation == old_generation + 1
    assert replacement is not old


def test_close_invalidates_views_and_permanently_rejects_registration() -> None:
    collection = ActuatorCollection(_Simulation())
    view = _register(collection, "first", _Control())
    collection.finalize()
    collection.close()
    collection.close()

    with pytest.raises(RuntimeError, match="closed"):
        _ = view.command
    with pytest.raises(RuntimeError, match="closed"):
        _register(collection, "second", _Control())


def test_context_lifecycle_bridges_keep_the_service_lazy() -> None:
    context = object.__new__(SimulationContext)
    context._services = ServiceLocator()

    SimulationContext._finalize_actuator_collection(context, None)
    SimulationContext._clear_actuator_collection_generation(context, None)
    assert context.services[ActuatorCollection] is None

    collection = SimulationContext._get_actuator_collection(context)
    assert context.services[ActuatorCollection] is collection


def test_context_eagerly_registers_named_actuator_bridges_before_ready_dispatch(monkeypatch) -> None:
    context = _make_fake_context(monkeypatch)
    ready_callbacks = sorted(
        (entry for entry in _FakePhysicsManager.callbacks if entry.event == PhysicsEvent.PHYSICS_READY),
        key=lambda entry: entry.order,
    )
    stop_callbacks = [entry for entry in _FakePhysicsManager.callbacks if entry.event == PhysicsEvent.STOP]

    assert [(entry.order, entry.name) for entry in ready_callbacks] == [
        (5, "render_context_initialize"),
        (20, "actuator_collection_finalize"),
    ]
    assert [(entry.order, entry.name) for entry in stop_callbacks] == [(20, "actuator_collection_stop")]
    assert context.services[ActuatorCollection] is None


def test_context_ready_snapshot_finalizes_order_ten_registration_and_stop_replays(monkeypatch) -> None:
    context = _make_fake_context(monkeypatch)
    context.set_clone_plan(
        ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=torch.ones((1, 1), dtype=torch.bool),
            cfg_rows={1: (0,), 2: (0,)},
        )
    )
    registered = []

    def register_order_ten(_payload) -> None:
        collection = context._get_actuator_collection()
        control = _Control()
        registered.append(_register(collection, "first" if not registered else "second", control))

    _FakePhysicsManager.register_callback(
        register_order_ten, PhysicsEvent.PHYSICS_READY, order=10, name="asset_initialize"
    )
    _FakePhysicsManager.dispatch(PhysicsEvent.PHYSICS_READY)
    collection = context.services[ActuatorCollection]
    assert collection.is_finalized
    assert registered[0].is_ready

    old = registered[0]
    _FakePhysicsManager.dispatch(PhysicsEvent.STOP)
    with pytest.raises(RuntimeError, match="stale actuator view"):
        _ = old.command
    _FakePhysicsManager.dispatch(PhysicsEvent.PHYSICS_READY)
    assert registered[1].is_ready
    assert registered[1] is not old


def test_context_clear_instance_closes_collection_before_a_second_context(monkeypatch) -> None:
    first = _make_fake_context(monkeypatch)
    collection = first._get_actuator_collection()
    view = _register(collection, "first", _Control())

    SimulationContext.clear_instance()
    with pytest.raises(RuntimeError, match="closed"):
        _ = view.command
    second = _make_fake_context(monkeypatch)
    assert second.services[ActuatorCollection] is None
    SimulationContext.clear_instance()


def test_deferred_asset_completion_requires_an_explicit_deferral() -> None:
    asset = object.__new__(_DeferredAsset)
    asset._is_initialized = False
    asset._initialization_deferred = False
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None

    with pytest.raises(RuntimeError, match="not deferred"):
        asset._complete_deferred_initialization()
    asset._defer_initialization()
    asset._complete_deferred_initialization()
    assert asset._is_initialized
    assert not asset._initialization_deferred
