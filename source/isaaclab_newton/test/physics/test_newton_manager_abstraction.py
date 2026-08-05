# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-solver :class:`NewtonManager` abstraction.

Covers:

* :attr:`NewtonSolverCfg.class_type` resolves to the matching manager subclass.
* :meth:`NewtonCfg.__post_init__` propagates ``solver_cfg.class_type`` onto
  :attr:`NewtonCfg.class_type` so that ``SimulationContext`` picks the right
  manager.
* Each leaf manager subclasses :class:`NewtonManager` and implements
  :meth:`_build_solver` (with the abstract base raising ``NotImplementedError``).
* The cross-config validation in :meth:`NewtonMJWarpManager._build_solver`
  rejects the ``MJWarp + use_mujoco_contacts=True + collision_cfg`` combination.
* Manager name dispatch (used by :class:`InteractiveScene` and the various
  factory dispatchers) still starts with ``"newton"``.
* End-to-end: spinning up a simulation with each solver builds the correct
  solver, sets the right ``_use_single_state`` / ``_needs_collision_pipeline``
  flags, and lands canonical state on :class:`NewtonManager` so that external
  ``NewtonManager._foo`` reads keep working.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonFeatherstoneManager,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
    NewtonMPMManager,
    NewtonShapeCfg,
    NewtonSolverCfg,
    NewtonXPBDManager,
    XPBDSolverCfg,
)
from isaaclab_newton.physics.mpm_manager import _make_solver_config
from newton.solvers import SolverFeatherstone, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverXPBD

from isaaclab.sim import SimulationCfg, build_simulation_context

# ---------------------------------------------------------------------------
# Lightweight (no sim) parametrisation
# ---------------------------------------------------------------------------

# (solver_cfg_factory, expected_manager, expected_solver_cls,
#  expected_use_single_state, expected_needs_collision_pipeline)
SOLVER_MATRIX = [
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=True),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        False,
        id="mjwarp_internal_contacts",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=False),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        True,
        id="mjwarp_newton_pipeline",
    ),
    pytest.param(
        lambda: XPBDSolverCfg(),
        NewtonXPBDManager,
        SolverXPBD,
        False,
        True,
        id="xpbd",
    ),
    pytest.param(
        lambda: FeatherstoneSolverCfg(),
        NewtonFeatherstoneManager,
        SolverFeatherstone,
        False,
        True,
        id="featherstone",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=True),
        NewtonKaminoManager,
        SolverKamino,
        False,
        False,
        id="kamino_internal_contacts",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=False),
        NewtonKaminoManager,
        SolverKamino,
        False,
        True,
        id="kamino_newton_pipeline",
    ),
    pytest.param(
        lambda: MPMSolverCfg(max_iterations=2, voxel_size=0.05),
        NewtonMPMManager,
        SolverImplicitMPM,
        True,
        False,
        id="implicit_mpm",
    ),
]


def test_unregister_post_actuator_callback_removes_exact_callback_idempotently(monkeypatch) -> None:
    """Removing a rolled-back telemetry callback preserves equal registrations."""
    callbacks: list[object] = []
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", callbacks)

    class _EqualCallback:
        def __call__(self) -> None:
            pass

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _EqualCallback)

    first = _EqualCallback()
    equal_but_distinct = _EqualCallback()
    NewtonManager.register_post_actuator_callback(first)
    NewtonManager.register_post_actuator_callback(equal_but_distinct)

    NewtonManager.unregister_post_actuator_callback(first)
    NewtonManager.unregister_post_actuator_callback(first)

    assert len(callbacks) == 1
    assert callbacks[0] is equal_but_distinct


def test_control_invalidation_deregisters_telemetry_before_releasing_candidate(monkeypatch) -> None:
    """A rollback cannot leave telemetry able to touch candidate-owned state."""
    from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl

    from isaaclab.actuators.actuator_control import ArticulationActuatorControl

    callbacks: list[object] = []
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", callbacks)
    candidate = {"open": True}

    def _telemetry() -> None:
        if not candidate["open"]:
            raise RuntimeError("telemetry touched a closed candidate")

    control = object.__new__(NewtonActuatorControl)
    control._post_actuator_callback = _telemetry
    callbacks.append(_telemetry)

    def _release_binding(self) -> None:
        assert _telemetry not in callbacks
        candidate["open"] = False

    monkeypatch.setattr(ArticulationActuatorControl, "invalidate_actuator_view", _release_binding)
    control.invalidate_actuator_view()

    assert callbacks == []
    assert not candidate["open"]


def test_native_actuator_discovery_does_not_mutate_solver_properties(monkeypatch) -> None:
    """Discovery classifies native groups without changing defaults before source capture."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module

    from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg

    class _Articulation:
        _sim_cfg = SimpleNamespace(use_newton_actuators=True)
        device = "cpu"

        def write_joint_stiffness_to_sim_index(self, **kwargs) -> None:
            raise AssertionError("native discovery wrote stiffness before source capture")

        def write_joint_damping_to_sim_index(self, **kwargs) -> None:
            raise AssertionError("native discovery wrote damping before source capture")

        def find_joints(self, joint_names_expr) -> tuple[list[int], list[str]]:
            return [0], ["joint"]

    monkeypatch.setattr(control_module, "_HAS_NEWTON_ACTUATORS", True)
    monkeypatch.setattr(NewtonManager, "activate_newton_actuator_path", classmethod(lambda cls: None))
    control = control_module.NewtonActuatorControl(_Articulation())

    native_groups = control.discover_native_actuators(
        {
            "implicit": ImplicitActuatorCfg(joint_names_expr=[".*"]),
            "explicit": IdealPDActuatorCfg(joint_names_expr=[".*"]),
        }
    )

    assert native_groups == {"explicit"}


def test_staged_solver_properties_write_each_newton_binding_once(monkeypatch) -> None:
    """The final merged rowset reaches each supported Newton property binding exactly once."""
    from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
    from isaaclab_newton.benchmark.assets import runtime

    from isaaclab.actuators.actuator_control import _ResolvedSolverProperties, _ResolvedSolverProperty
    from isaaclab.utils.warp import ProxyArray

    runtime._load_runtime_symbols()
    with ExitStack() as stack:
        for module in (
            "isaaclab_newton.assets.articulation.articulation_data.SimulationManager",
            "isaaclab_newton.assets.articulation.articulation.SimulationManager",
        ):
            stack.enter_context(runtime.create_mock_newton_manager(module, num_instances=2, num_bodies=3, num_joints=2))
        articulation, _ = runtime.create_test_articulation(num_instances=2, num_bodies=3, num_joints=2, device="cpu")

        property_bindings = {
            "stiffness": ("write_joint_stiffness_to_sim_mask", "_sim_bind_joint_stiffness_sim"),
            "damping": ("write_joint_damping_to_sim_mask", "_sim_bind_joint_damping_sim"),
            "effort_limit_sim": ("write_joint_effort_limit_to_sim_mask", "_sim_bind_joint_effort_limits_sim"),
            "velocity_limit_sim": ("write_joint_velocity_limit_to_sim_mask", "_sim_bind_joint_vel_limits_sim"),
            "armature": ("write_joint_armature_to_sim_mask", "_sim_bind_joint_armature"),
            "friction": (
                "write_joint_friction_coefficient_to_sim_mask",
                "_sim_bind_joint_friction_coeff",
            ),
        }
        values = {
            name: np.arange(4, dtype=np.float32).reshape(2, 2) + offset
            for offset, name in enumerate((*property_bindings, "dynamic_friction", "viscous_friction"), start=1)
        }
        writes = dict.fromkeys(property_bindings, 0)
        for property_name, (method_name, _) in property_bindings.items():
            original = getattr(articulation, method_name)

            def counted_write(*, _name=property_name, _original=original, **kwargs) -> None:
                writes[_name] += 1
                _original(**kwargs)

            monkeypatch.setattr(articulation, method_name, counted_write)

        properties = {
            name: _ResolvedSolverProperty(
                transport="device",
                source_rows=torch.from_numpy(value[:1].copy()),
                source_slot_by_backend_row=None,
                source_assignment=torch.zeros(2, dtype=torch.int32),
                canonical_target=ProxyArray(wp.array(value, dtype=wp.float32, device="cpu")),
            )
            for name, value in values.items()
        }

        NewtonActuatorControl(articulation).write_resolved_joint_properties_staged(
            _ResolvedSolverProperties(properties=properties)
        )

        assert writes == dict.fromkeys(property_bindings, 1)
        for name, (_, binding_name) in property_bindings.items():
            np.testing.assert_array_equal(getattr(articulation.data, binding_name).numpy(), values[name])


def test_staged_solver_properties_restore_and_commit_actual_newton_bindings() -> None:
    """Rollback restores exact Newton arrays, while commit makes the staged values durable."""
    from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
    from isaaclab_newton.benchmark.assets import runtime

    from isaaclab.actuators.actuator_control import _ResolvedSolverProperties, _ResolvedSolverProperty
    from isaaclab.utils.warp import ProxyArray

    runtime._load_runtime_symbols()
    with ExitStack() as stack:
        for module in (
            "isaaclab_newton.assets.articulation.articulation_data.SimulationManager",
            "isaaclab_newton.assets.articulation.articulation.SimulationManager",
        ):
            stack.enter_context(runtime.create_mock_newton_manager(module, num_instances=2, num_bodies=3, num_joints=2))
        articulation, _ = runtime.create_test_articulation(num_instances=2, num_bodies=3, num_joints=2, device="cpu")
        binding_names = {
            "stiffness": "_sim_bind_joint_stiffness_sim",
            "damping": "_sim_bind_joint_damping_sim",
            "effort_limit_sim": "_sim_bind_joint_effort_limits_sim",
            "velocity_limit_sim": "_sim_bind_joint_vel_limits_sim",
            "armature": "_sim_bind_joint_armature",
            "friction": "_sim_bind_joint_friction_coeff",
        }
        baseline = {name: getattr(articulation.data, binding).numpy().copy() for name, binding in binding_names.items()}
        values = {
            name: np.arange(4, dtype=np.float32).reshape(2, 2) + 20 + offset
            for offset, name in enumerate((*binding_names, "dynamic_friction", "viscous_friction"))
        }
        properties = {
            name: _ResolvedSolverProperty(
                transport="device",
                source_rows=torch.from_numpy(value[:1].copy()),
                source_slot_by_backend_row=None,
                source_assignment=torch.zeros(2, dtype=torch.int32),
                canonical_target=ProxyArray(wp.array(value, dtype=wp.float32, device="cpu")),
            )
            for name, value in values.items()
        }
        payload = _ResolvedSolverProperties(properties=properties)
        control = NewtonActuatorControl(articulation)

        control.write_resolved_joint_properties_staged(payload)
        control.restore_resolved_joint_properties()

        for name, binding_name in binding_names.items():
            np.testing.assert_array_equal(getattr(articulation.data, binding_name).numpy(), baseline[name])

        control.write_resolved_joint_properties_staged(payload)
        control.commit_resolved_joint_properties()
        control.restore_resolved_joint_properties()

        for name, binding_name in binding_names.items():
            np.testing.assert_array_equal(getattr(articulation.data, binding_name).numpy(), values[name])


def test_runtime_native_parameter_routes_patch_controller_and_clamping_in_place(monkeypatch) -> None:
    """Group/type index/mask writes refresh ordered native parameters without runtime allocation."""
    from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl

    from isaaclab.actuators import DCMotor
    from isaaclab.actuators.actuator_control import _ActuatorParameterWrite
    from isaaclab.utils.warp import ProxyArray

    controller = SimpleNamespace(
        kp=wp.array([30.0, 10.0, 31.0, 11.0], dtype=wp.float32, device="cpu"),
        kd=wp.array([3.0, 1.0, 3.1, 1.1], dtype=wp.float32, device="cpu"),
    )
    dc_clamping = SimpleNamespace(
        max_motor_effort=wp.array([60.0, 40.0, 61.0, 41.0], dtype=wp.float32, device="cpu"),
        velocity_limit=wp.array([6.0, 8.0, 6.1, 8.1], dtype=wp.float32, device="cpu"),
        saturation_effort=wp.array([120.0, 100.0, 121.0, 110.0], dtype=wp.float32, device="cpu"),
        corner_velocity=wp.zeros(4, dtype=wp.float32, device="cpu"),
    )
    ideal_clamping = SimpleNamespace(max_effort=wp.array([60.0, 40.0, 61.0, 41.0], dtype=wp.float32, device="cpu"))
    actuator = SimpleNamespace(
        indices=wp.array([1, 3, 7, 9], dtype=wp.uint32, device="cpu"),
        controller=controller,
        clamping=[dc_clamping, ideal_clamping],
    )
    adapter = SimpleNamespace(actuators=[actuator], num_joints=6)
    backend_to_user = wp.array([2, 1, 0], dtype=wp.int32, device="cpu")
    articulation = SimpleNamespace(
        num_instances=2,
        num_joints=3,
        num_fixed_tendons=0,
        device="cpu",
        data=SimpleNamespace(has_joint_ordering=True),
        newton_actuator_adapter=adapter,
        _joint_backend_to_user_map=lambda: backend_to_user,
    )
    control = NewtonActuatorControl(articulation)
    control._native_dof_offset = 1
    owner_slots = torch.tensor([0, -1, 1], dtype=torch.int32)
    parameter_cases = (
        (
            "stiffness",
            controller.kp,
            [[10.0, 30.0], [11.0, 99.0]],
            [30.0, 10.0, 99.0, 11.0],
            {"scope": "group", "env_ids": torch.tensor([1]), "joint_ids": torch.tensor([2])},
        ),
        (
            "damping",
            controller.kd,
            [[88.0, 3.0], [1.1, 3.1]],
            [3.0, 88.0, 3.1, 1.1],
            {
                "scope": "group",
                "env_mask": torch.tensor([True, False]),
                "joint_mask": torch.tensor([True, False, False]),
            },
        ),
        (
            "effort_limit",
            dc_clamping.max_motor_effort,
            [[40.0, 60.0], [77.0, 61.0]],
            [60.0, 40.0, 61.0, 77.0],
            {"scope": "type", "env_ids": torch.tensor([1]), "joint_ids": torch.tensor([0])},
        ),
        (
            "velocity_limit",
            dc_clamping.velocity_limit,
            [[8.0, 66.0], [8.1, 6.1]],
            [66.0, 8.0, 6.1, 8.1],
            {
                "scope": "type",
                "env_mask": torch.tensor([True, False]),
                "joint_mask": torch.tensor([False, False, True]),
            },
        ),
        (
            "saturation_effort",
            dc_clamping.saturation_effort,
            [[100.0, 120.0], [110.0, 200.0]],
            [120.0, 100.0, 200.0, 110.0],
            {
                "scope": "group",
                "env_mask": torch.tensor([False, True]),
                "joint_mask": torch.tensor([False, False, True]),
            },
        ),
    )
    pointers = {id(array): int(array.ptr) for _, array, *_ in parameter_cases}

    def fail_allocation(*args, **kwargs):
        raise AssertionError("runtime parameter routing allocated a new buffer")

    monkeypatch.setattr(wp, "zeros", fail_allocation)
    monkeypatch.setattr(torch, "zeros", fail_allocation)
    for name, target, canonical_values, expected, selectors in parameter_cases:
        canonical = ProxyArray(wp.array(canonical_values, dtype=wp.float32, device="cpu"))
        control.write_actuator_parameter(
            name,
            _ActuatorParameterWrite(
                value=canonical.torch,
                actuator_type=DCMotor,
                canonical=canonical,
                backend_owner_slots=owner_slots,
                **selectors,
            ),
        )
        np.testing.assert_allclose(target.numpy(), expected)
        assert int(target.ptr) == pointers[id(target)]

    np.testing.assert_allclose(ideal_clamping.max_effort.numpy(), [60.0, 40.0, 61.0, 77.0])
    expected_corner_velocity = dc_clamping.velocity_limit.numpy() * (
        1.0 + dc_clamping.max_motor_effort.numpy() / dc_clamping.saturation_effort.numpy()
    )
    np.testing.assert_allclose(dc_clamping.corner_velocity.numpy(), expected_corner_velocity)


def test_runtime_implicit_gain_routes_patch_actual_newton_solver_bindings(monkeypatch) -> None:
    """Canonical group-index and type-mask gain writes update Newton drives in place."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module
    from isaaclab_newton.benchmark.assets import runtime

    from isaaclab.actuators import ImplicitActuator
    from isaaclab.actuators.actuator_control import _ActuatorParameterWrite
    from isaaclab.utils.warp import ProxyArray

    runtime._load_runtime_symbols()
    with ExitStack() as stack:
        for module in (
            "isaaclab_newton.assets.articulation.articulation_data.SimulationManager",
            "isaaclab_newton.assets.articulation.articulation.SimulationManager",
        ):
            stack.enter_context(runtime.create_mock_newton_manager(module, num_instances=2, num_bodies=4, num_joints=3))
        articulation, _ = runtime.create_test_articulation(num_instances=2, num_bodies=4, num_joints=3, device="cpu")
        articulation.newton_actuator_adapter = None
        control = control_module.NewtonActuatorControl(articulation)
        stiffness_binding = articulation.data._sim_bind_joint_stiffness_sim
        damping_binding = articulation.data._sim_bind_joint_damping_sim
        stiffness_baseline = stiffness_binding.numpy().copy()
        damping_baseline = damping_binding.numpy().copy()
        stiffness_values = stiffness_baseline[:, [0, 2]].copy()
        damping_values = damping_baseline[:, [0, 2]].copy()
        stiffness_values[1, 1] = 99.0
        damping_values[0, 0] = 88.0
        owner_slots = torch.tensor([0, -1, 1], dtype=torch.int32)
        changes = []
        monkeypatch.setattr(
            control_module.SimulationManager,
            "add_model_change",
            classmethod(lambda cls, change: changes.append(change)),
        )
        stiffness = ProxyArray(wp.array(stiffness_values, dtype=wp.float32, device="cpu"))
        damping = ProxyArray(wp.array(damping_values, dtype=wp.float32, device="cpu"))
        stiffness_ptr = int(stiffness_binding.ptr)
        damping_ptr = int(damping_binding.ptr)

        def fail_allocation(*args, **kwargs):
            raise AssertionError("runtime implicit gain routing allocated a new buffer")

        monkeypatch.setattr(wp, "zeros", fail_allocation)
        monkeypatch.setattr(torch, "zeros", fail_allocation)
        control.write_actuator_parameter(
            "stiffness",
            _ActuatorParameterWrite(
                value=stiffness.torch,
                actuator_type=ImplicitActuator,
                canonical=stiffness,
                env_ids=torch.tensor([1]),
                joint_ids=torch.tensor([2]),
                backend_owner_slots=owner_slots,
                scope="group",
            ),
        )
        control.write_actuator_parameter(
            "damping",
            _ActuatorParameterWrite(
                value=damping.torch,
                actuator_type=ImplicitActuator,
                canonical=damping,
                env_mask=torch.tensor([True, False]),
                joint_mask=torch.tensor([True, False, False]),
                backend_owner_slots=owner_slots,
                scope="type",
            ),
        )

        expected_stiffness = stiffness_baseline.copy()
        expected_stiffness[1, 2] = 99.0
        expected_damping = damping_baseline.copy()
        expected_damping[0, 0] = 88.0
        np.testing.assert_array_equal(stiffness_binding.numpy(), expected_stiffness)
        np.testing.assert_array_equal(damping_binding.numpy(), expected_damping)
        assert int(stiffness_binding.ptr) == stiffness_ptr
        assert int(damping_binding.ptr) == damping_ptr
        assert len(changes) == 2


def test_failed_native_prepare_restores_all_candidate_specific_state(monkeypatch) -> None:
    """A prepare failure restores every adapter field and removes its exact callback."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module
    from newton import Model as NewtonModel

    original = {
        name: object()
        for name in (
            "newton_actuator_adapter",
            "newton_default_stiffness",
            "newton_default_damping",
            "newton_managed_local_joints",
            "_implicit_dof_mask",
            "_implicit_dof_mask_owner",
        )
    }
    original_computed_effort = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    data = SimpleNamespace(
        joint_ordering=None,
        _sim_bind_joint_computed_effort=original_computed_effort,
        _rollback_actuator_initialization=lambda: None,
    )
    articulation = SimpleNamespace(
        num_instances=2,
        num_joints=3,
        num_fixed_tendons=0,
        device="cpu",
        data=data,
        _data=data,
        _root_view=SimpleNamespace(
            frequency_layouts={
                NewtonModel.AttributeFrequency.JOINT_DOF: SimpleNamespace(slice=SimpleNamespace(start=1), indices=None)
            }
        ),
        _rollback_deferred_initialization=lambda: None,
        **original,
    )
    candidate_binding = SimpleNamespace(
        stiffness=torch.full((2, 3), 10.0),
        damping=torch.full((2, 3), 2.0),
        joint_indices=torch.tensor([0, 2], dtype=torch.int32),
        implicit_dof_mask=wp.zeros(3, dtype=wp.int32, device="cpu"),
        implicit_dof_mask_owner=torch.zeros(3, dtype=torch.int32),
        computed_effort_view=wp.zeros((2, 3), dtype=wp.float32, device="cpu"),
    )
    adapter = SimpleNamespace(bind_articulation=lambda **kwargs: candidate_binding)
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", adapter)
    callbacks = []
    monkeypatch.setattr(control_module.SimulationManager, "_post_actuator_callbacks", callbacks)

    def fail_registration(cls, callback) -> None:
        callbacks.append(callback)
        raise RuntimeError("post-actuator registration failed")

    monkeypatch.setattr(
        control_module.SimulationManager,
        "register_post_actuator_callback",
        classmethod(fail_registration),
    )
    control = control_module.NewtonActuatorControl(articulation)
    control._native_active = True
    binding = SimpleNamespace(
        groups={},
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        computed_effort=SimpleNamespace(warp=object()),
        applied_effort=SimpleNamespace(warp=object()),
    )

    with pytest.raises(RuntimeError, match="post-actuator registration failed"):
        control.prepare_actuator_binding(binding)

    for name, value in original.items():
        assert getattr(articulation, name) is value
    assert data._sim_bind_joint_computed_effort is original_computed_effort
    assert control._native_dof_offset is None
    assert control._post_actuator_callback is None
    assert callbacks == []


def test_failed_finalize_invalidation_clears_fallback_state_before_binding_release(monkeypatch) -> None:
    """A facade-bind failure removes callback/fallback pointers before shared binding release."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module

    callbacks = []
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", None)
    monkeypatch.setattr(control_module.SimulationManager, "_post_actuator_callbacks", callbacks)

    def build_mask(groups, num_joints, device):
        return wp.zeros(num_joints, dtype=wp.int32, device=device), torch.zeros(num_joints, dtype=torch.int32)

    monkeypatch.setattr("isaaclab_newton.actuators.build_implicit_dof_mask", build_mask)
    data = SimpleNamespace(
        joint_ordering=None,
        _sim_bind_joint_computed_effort=None,
        _rollback_actuator_initialization=lambda: None,
        bind_actuator_collection=lambda view: (_ for _ in ()).throw(RuntimeError("facade bind failed")),
    )
    articulation = SimpleNamespace(
        num_instances=2,
        num_joints=3,
        num_fixed_tendons=0,
        device="cpu",
        data=data,
        _data=data,
        _rollback_deferred_initialization=lambda: None,
        newton_actuator_adapter=None,
        newton_default_stiffness=None,
        newton_default_damping=None,
        newton_managed_local_joints=None,
        _implicit_dof_mask=None,
        _implicit_dof_mask_owner=None,
    )
    control = control_module.NewtonActuatorControl(articulation)
    control._native_active = True
    binding = SimpleNamespace(
        groups={},
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        computed_effort=SimpleNamespace(warp=object()),
        applied_effort=SimpleNamespace(warp=object()),
    )
    control.prepare_actuator_binding(binding)
    assert len(callbacks) == 1

    with pytest.raises(RuntimeError, match="facade bind failed"):
        control.bind_actuator_view(object())
    control.invalidate_actuator_view()

    assert callbacks == []
    assert articulation._implicit_dof_mask is None
    assert articulation._implicit_dof_mask_owner is None
    assert data._sim_bind_joint_computed_effort is None
    assert control._actuator_binding is None


def test_stop_ready_rebuild_uses_fresh_fallback_state_and_callback(monkeypatch) -> None:
    """STOP clears fallback allocations so READY installs fresh pointers and one callback."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module

    callbacks = []
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", None)
    monkeypatch.setattr(control_module.SimulationManager, "_post_actuator_callbacks", callbacks)

    def build_mask(groups, num_joints, device):
        return wp.zeros(num_joints, dtype=wp.int32, device=device), torch.zeros(num_joints, dtype=torch.int32)

    monkeypatch.setattr("isaaclab_newton.actuators.build_implicit_dof_mask", build_mask)
    data = SimpleNamespace(
        joint_ordering=None,
        _sim_bind_joint_computed_effort=None,
        _rollback_actuator_initialization=lambda: None,
    )
    articulation = SimpleNamespace(
        num_instances=2,
        num_joints=3,
        num_fixed_tendons=0,
        device="cpu",
        data=data,
        _data=data,
        _rollback_deferred_initialization=lambda: None,
        newton_actuator_adapter=None,
        newton_default_stiffness=None,
        newton_default_damping=None,
        newton_managed_local_joints=None,
        _implicit_dof_mask=None,
        _implicit_dof_mask_owner=None,
    )
    control = control_module.NewtonActuatorControl(articulation)
    binding = SimpleNamespace(
        groups={},
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        computed_effort=SimpleNamespace(warp=object()),
        applied_effort=SimpleNamespace(warp=object()),
    )

    control._native_active = True
    control.prepare_actuator_binding(binding)
    first_mask = articulation._implicit_dof_mask
    first_computed_effort = data._sim_bind_joint_computed_effort
    first_callback = control._post_actuator_callback
    control.invalidate_actuator_view()

    assert callbacks == []
    assert articulation._implicit_dof_mask is None
    assert articulation._implicit_dof_mask_owner is None
    assert data._sim_bind_joint_computed_effort is None

    control._native_active = True
    control.prepare_actuator_binding(binding)

    assert articulation._implicit_dof_mask is not None
    assert articulation._implicit_dof_mask is not first_mask
    assert data._sim_bind_joint_computed_effort is not None
    assert data._sim_bind_joint_computed_effort is not first_computed_effort
    assert control._post_actuator_callback is not first_callback
    assert callbacks == [control._post_actuator_callback]


def test_articulation_callback_teardown_does_not_partially_invalidate_collection_control(monkeypatch) -> None:
    """Asset callback teardown leaves atomic generation invalidation to the collection manager."""
    from isaaclab_newton.assets.articulation.articulation import Articulation

    from isaaclab.assets.asset_base import AssetBase

    invalidations = []
    monkeypatch.setattr(AssetBase, "_clear_callbacks", lambda self: None)
    articulation = object.__new__(Articulation)
    articulation._physics_ready_handle = None
    articulation._post_step_callback = None
    articulation._actuator_control = SimpleNamespace(invalidate_actuator_view=lambda: invalidations.append(True))

    articulation._clear_callbacks()

    assert invalidations == []


# ---------------------------------------------------------------------------
# class_type wiring (no SimulationContext required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_solver_cfg_class_type_resolves_to_subclass(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """Each ``*SolverCfg.class_type`` resolves to its matching manager subclass."""
    solver_cfg = solver_cfg_factory()
    # ``class_type`` is a lazy ``"module:Class"`` reference; calling its
    # ``_resolve()`` returns the actual class. ``__name__`` works without
    # forcing import (LazyType caches metadata) and is sufficient identity.
    assert solver_cfg.class_type.__name__ == expected_manager.__name__


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_newton_cfg_post_init_propagates_class_type(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """``NewtonCfg.__post_init__`` lifts ``solver_cfg.class_type`` onto ``NewtonCfg.class_type``."""
    cfg = NewtonCfg(solver_cfg=solver_cfg_factory())
    assert cfg.class_type.__name__ == expected_manager.__name__


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, should_warn",
    [
        (8, 0, False),  # Default: feature disabled, no warning.
        (8, 1, False),  # Valid: re-collide every substep.
        (8, 2, False),  # Valid: re-collide every 2 substeps.
        (8, 7, False),  # Valid edge: one mid-loop re-collide at i=6.
        (8, 8, True),  # Equal to num_substeps: gate never fires.
        (8, 16, True),  # Larger than num_substeps: gate never fires.
    ],
)
def test_newton_cfg_collision_decimation_warning(num_substeps, collision_decimation, should_warn, caplog):
    """``NewtonCfg.__post_init__`` warns when ``collision_decimation >= num_substeps``."""
    import logging

    with caplog.at_level(logging.WARNING, logger="isaaclab_newton.physics.newton_manager_cfg"):
        cfg = NewtonCfg(num_substeps=num_substeps, collision_decimation=collision_decimation)
    warned = any("collision_decimation" in rec.getMessage() for rec in caplog.records)
    assert warned is should_warn
    # Cfg field round-trips regardless of warning.
    assert cfg.collision_decimation == collision_decimation


def test_refit_sensor_bvh_rejects_missing_sensor_state(monkeypatch):
    """BVH refitting raises when a particle BVH exists without an initialized sensor state."""
    model = SimpleNamespace(shape_count=0, particle_count=1, bvh_particles=object())
    monkeypatch.setattr(NewtonManager, "_model", model, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_state", None, raising=False)

    with pytest.raises(RuntimeError, match="requires an initialized sensor state"):
        NewtonManager._refit_sensor_bvh()


def test_sensor_task_builds_and_refits_bvhs_before_rendering(monkeypatch):
    """Shape and particle BVHs are built and refit before a render task runs."""
    from isaaclab.physics import PhysicsManager

    state = object()
    status = {"shape_refit": False, "particle_refit": False, "rendered": False}

    class FakeModel:
        shape_count = 1
        particle_count = 1
        bvh_shapes = None
        bvh_particles = None

        def bvh_build_shapes(self, current_state):
            assert current_state is state
            self.bvh_shapes = object()

        def bvh_build_particles(self, current_state):
            assert current_state is state
            self.bvh_particles = object()

        def bvh_refit_shapes(self, current_state):
            assert current_state is state
            status["shape_refit"] = True

        def bvh_refit_particles(self, current_state):
            assert current_state is state
            status["particle_refit"] = True

    model = FakeModel()

    def render():
        assert model.bvh_shapes is not None
        assert model.bvh_particles is not None
        assert status["shape_refit"]
        assert status["particle_refit"]
        status["rendered"] = True

    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))
    monkeypatch.setattr(NewtonManager, "_model", model, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_tasks", {}, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_state", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_state_dirty", True, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_graph", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_flags", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_flags_host", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_sensor_graph_capture_failed", False, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=False), raising=False)

    NewtonManager._register_sensor_task("render", render)
    NewtonManager._update_sensor_tasks("render")

    assert status["rendered"]


def test_newton_shape_cfg_defaults_match_newton_shape_config():
    """``NewtonShapeCfg`` contact defaults mirror Newton's ``ShapeConfig``.

    Guards the invariant that keeps ``checked_apply`` a no-op for envs that do
    not override ``ke``/``kd``/``mu``: if Newton's upstream defaults drift, this
    fails instead of silently clobbering every Newton scene's shape materials.
    """
    import newton

    upstream = newton.ModelBuilder().default_shape_cfg
    shape_cfg = NewtonShapeCfg()
    assert shape_cfg.ke == upstream.ke
    assert shape_cfg.kd == upstream.kd
    assert shape_cfg.mu == upstream.mu


def test_mpm_solver_cfg_maps_only_newton_solver_fields():
    """MPM config forwarding ignores Isaac Lab metadata fields explicitly."""

    solver_cfg = MPMSolverCfg(
        max_iterations=7,
        voxel_size=0.04,
        solver_type="isaaclab_metadata_should_not_forward",
    )

    newton_cfg = _make_solver_config(solver_cfg)

    assert newton_cfg.max_iterations == 7
    assert newton_cfg.voxel_size == 0.04
    assert not hasattr(newton_cfg, "class_type")
    assert not hasattr(newton_cfg, "solver_type")
    # Manager-level stepping option must not leak into the Newton solver config.
    assert not hasattr(newton_cfg, "project_outside_colliders")


# Tuples of ``(field_name, non_default_value)`` covering every solver-tunable
# field on :class:`MPMSolverCfg`. Each entry exercises the implementation-side
# SolverImplicitMPM.Config construction so a Newton field rename or accidental
# drop is caught here instead of silently producing wrong-physics runs.
_MPM_FIELD_VALUES = [
    ("max_iterations", 13),
    ("tolerance", 5.0e-5),
    ("solver", "gauss-seidel"),
    ("warmstart_mode", "particles"),
    ("collider_velocity_mode", "backward"),
    ("voxel_size", 0.0375),
    ("grid_type", "dense"),
    ("grid_padding", 4),
    ("max_active_cell_count", 1024),
    ("transfer_scheme", "pic"),
    ("integration_scheme", "gimp"),
    ("critical_fraction", 0.25),
    ("air_drag", 0.5),
    ("collider_normal_from_sdf_gradient", True),
    ("collider_basis", "Q1"),
    ("strain_basis", "P1d"),
    ("velocity_basis", "B2"),
]


@pytest.mark.parametrize("field_name, value", _MPM_FIELD_VALUES)
def test_mpm_solver_cfg_forwards_every_solver_field(field_name, value):
    """Every tunable MPM cfg field round-trips into ``SolverImplicitMPM.Config``.

    Guards against MPM manager construction dropping or mis-naming a field if
    Newton's config surface changes.
    """
    solver_cfg = MPMSolverCfg(**{field_name: value})
    newton_cfg = _make_solver_config(solver_cfg)
    assert hasattr(newton_cfg, field_name), (
        f"{field_name!r} disappeared from SolverImplicitMPM.Config — MPMSolverCfg needs to drop or rename it."
    )
    assert getattr(newton_cfg, field_name) == value


def test_mpm_register_builder_attributes_is_idempotent():
    """The MPM custom-attribute hook is a no-op when attributes are already registered."""
    import newton

    builder = newton.ModelBuilder()
    assert not builder.has_custom_attribute("mpm:young_modulus")

    NewtonMPMManager._register_builder_attributes(builder)
    assert builder.has_custom_attribute("mpm:young_modulus")

    # Second call must be a no-op (no exceptions, attribute still present).
    NewtonMPMManager._register_builder_attributes(builder)
    assert builder.has_custom_attribute("mpm:young_modulus")


def test_mpm_prepare_builder_makes_kinematic_bodies_massless():
    """Kinematic bodies must be massless so MPM treats them as kinematic colliders."""
    import newton

    builder = newton.ModelBuilder()
    kinematic_body = builder.add_body(
        mass=0.35,
        inertia=wp.mat33(1.0),
        is_kinematic=True,
        label="kinematic_collider",
    )
    dynamic_body = builder.add_body(
        mass=1.2,
        inertia=wp.mat33(2.0),
        is_kinematic=False,
        label="dynamic_body",
    )

    NewtonMPMManager._prepare_builder_for_finalize(builder)

    assert builder.body_flags[kinematic_body] & int(newton.BodyFlags.KINEMATIC)
    assert builder.body_mass[kinematic_body] == 0.0
    assert builder.body_inv_mass[kinematic_body] == 0.0
    assert np.allclose(np.array(builder.body_inertia[kinematic_body]), 0.0)
    assert np.allclose(np.array(builder.body_inv_inertia[kinematic_body]), 0.0)

    assert builder.body_mass[dynamic_body] == pytest.approx(1.2)
    assert builder.body_inv_mass[dynamic_body] == pytest.approx(1.0 / 1.2)
    assert np.allclose(np.array(builder.body_inertia[dynamic_body]), 2.0)


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_mpm_prepare_builder_converts_convex_mesh_before_solver_construction():
    """Convex meshes must become triangle meshes before implicit MPM consumes the model."""
    import newton

    builder = newton.ModelBuilder()
    NewtonMPMManager._register_builder_attributes(builder)
    body = builder.add_body(label="convex_mesh_collider")
    mesh = newton.Mesh(
        vertices=[(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        indices=[0, 1, 2],
    )
    shape = builder.add_shape_mesh(body, mesh=mesh)
    builder.shape_type[shape] = newton.GeoType.CONVEX_MESH
    builder.add_particles(
        pos=[(0.0, 0.0, 0.1)],
        vel=[(0.0, 0.0, 0.0)],
        mass=[0.01],
        radius=[0.02],
        custom_attributes={
            "mpm:viscosity": 50.0,
            "mpm:friction": 0.0,
            "mpm:tensile_yield_ratio": 1.0,
            "mpm:yield_pressure": 1.0e15,
            "mpm:yield_stress": 0.0,
            "mpm:young_modulus": 1.0e15,
            "mpm:damping": 0.0,
        },
    )

    NewtonMPMManager._prepare_builder_for_finalize(builder)
    model = builder.finalize(device="cuda:0")
    solver = NewtonMPMManager._create_solver(model, MPMSolverCfg(max_iterations=2, voxel_size=0.05))

    assert builder.shape_type[shape] == newton.GeoType.MESH
    assert isinstance(solver, SolverImplicitMPM)


def test_active_manager_create_builder_registers_mpm_attributes():
    """The active MPM manager registers solver-specific builder attributes."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()

    assert builder.has_custom_attribute("mpm:young_modulus")


def test_mpm_end_to_end_with_particle_custom_attributes():
    """End-to-end MPM step using ``add_particles(custom_attributes=...)`` — the production path."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        # MPM custom attrs must exist on the builder before particles use them.
        assert builder.has_custom_attribute("mpm:young_modulus")

        positions = [(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)]
        builder.add_particles(
            pos=positions,
            vel=[(0.0, 0.0, 0.0)] * len(positions),
            mass=[0.01] * len(positions),
            radius=[0.02] * len(positions),
            custom_attributes={
                "mpm:viscosity": 50.0,
                "mpm:friction": 0.0,
                "mpm:tensile_yield_ratio": 1.0,
                "mpm:yield_pressure": 1.0e15,
                "mpm:yield_stress": 0.0,
                "mpm:young_modulus": 1.0e15,
                "mpm:damping": 0.0,
            },
        )
        NewtonManager.set_builder(builder)

        sim.reset()
        assert isinstance(NewtonManager._solver, SolverImplicitMPM)
        sim.step(render=False)


@pytest.mark.parametrize("project_outside", [True, False])
def test_mpm_project_outside_colliders_gates_projection(project_outside):
    """``project_outside_colliders`` controls whether ``project_outside`` runs per substep.

    Wraps the solver's ``project_outside`` with a counter after ``sim.reset()``
    (``use_cuda_graph=False`` keeps the Python callable on the step path) and
    runs one tick. The call count is positive only when the flag is set.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05, project_outside_colliders=project_outside),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        builder.add_particles(
            pos=[(0.0, 0.0, 0.10), (0.05, 0.0, 0.10), (0.0, 0.05, 0.10)],
            vel=[(0.0, 0.0, 0.0)] * 3,
            mass=[0.01] * 3,
            radius=[0.02] * 3,
            custom_attributes={
                "mpm:viscosity": 50.0,
                "mpm:friction": 0.0,
                "mpm:tensile_yield_ratio": 1.0,
                "mpm:yield_pressure": 1.0e15,
                "mpm:yield_stress": 0.0,
                "mpm:young_modulus": 1.0e15,
                "mpm:damping": 0.0,
            },
        )
        NewtonManager.set_builder(builder)
        sim.reset()

        calls = {"n": 0}
        original_project = NewtonManager._solver.project_outside

        def counting_project(*args, **kwargs):
            calls["n"] += 1
            return original_project(*args, **kwargs)

        NewtonManager._solver.project_outside = counting_project
        try:
            sim.step(render=False)
        finally:
            NewtonManager._solver.project_outside = original_project

        if project_outside:
            assert calls["n"] >= 1
        else:
            assert calls["n"] == 0


@pytest.mark.parametrize(
    "grid_type, expected",
    [
        ("fixed", True),
        ("sparse", False),
        ("dense", False),
    ],
)
def test_mpm_cuda_graph_capture_supports_only_fixed_grid(monkeypatch, grid_type, expected):
    """Newton implicit MPM is CUDA-graph capturable only with a fixed grid."""

    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(grid_type=grid_type), raising=False)

    assert NewtonMPMManager._supports_cuda_graph_capture() is expected


def test_mpm_unsupported_cuda_graph_capture_uses_eager_execution(monkeypatch):
    """Sparse/dense MPM should not enter a CUDA graph capture window."""
    from isaaclab.physics import PhysicsManager

    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=MPMSolverCfg(grid_type="sparse"), use_cuda_graph=True),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", SimpleNamespace(grid_type="sparse"), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)

    NewtonMPMManager._capture_or_defer_graph()

    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is False


def test_cuda_graph_capture_uses_simulation_device(monkeypatch):
    """CUDA graph capture should use the simulation device instead of Warp's default device."""
    from isaaclab.physics import PhysicsManager

    captured_devices = []
    captured_graph = object()

    class FakeScopedCapture:
        def __init__(self, device=None):
            captured_devices.append(device)
            self.graph = captured_graph

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:1", raising=False)
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", None, raising=False)
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: False))
    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(lambda cls: None))
    monkeypatch.setattr(wp, "ScopedCapture", FakeScopedCapture)

    NewtonManager._capture_or_defer_graph()

    assert captured_devices == ["cuda:1"]
    assert NewtonManager._graph is captured_graph


# ---------------------------------------------------------------------------
# Manager state-refresh boundaries (no SimulationContext required)
# ---------------------------------------------------------------------------


def test_forward_consumes_existing_reset_masks(monkeypatch):
    """The existing device masks are the complete input to masked FK and the solver reset hook."""
    world_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    observed: list[tuple[list[bool], list[bool]]] = []
    solver_resets: list[list[bool]] = []

    def record_fk(worlds, articulations):
        observed.append((worlds.numpy().tolist(), articulations.numpy().tolist()))

    class _RecordingSolver:
        def reset(self, state, world_mask=None, flags=0):
            solver_resets.append(world_mask.numpy().tolist())

    monkeypatch.setattr(NewtonManager, "_world_reset_mask", world_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", fk_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_eval_fk", record_fk, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _RecordingSolver(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_reset_solver_internals_delegate",
        NewtonManager._reset_solver_internals,
        raising=False,
    )

    NewtonManager.forward()

    assert observed == [([False, True], [True, False])]
    assert solver_resets == [[False, True]]
    assert world_mask.numpy().tolist() == [False, False]
    assert fk_mask.numpy().tolist() == [False, False]


def test_forward_dispatches_active_mpm_reset_hook_through_base_manager(monkeypatch):
    """Base-class state reads must use the active MPM manager's reset behavior."""
    world_mask = wp.array([True], dtype=wp.bool, device="cpu")
    fk_mask = wp.array([], dtype=wp.bool, device="cpu")

    class _RejectingSolver:
        def reset(self, state, world_mask=None, flags=0):
            raise AssertionError("the base reset hook must not run for implicit MPM")

    monkeypatch.setattr(NewtonManager, "_world_reset_mask", world_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_fk_reset_mask", fk_mask, raising=False)
    monkeypatch.setattr(NewtonManager, "_eval_fk", lambda worlds, articulations: None, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", _RejectingSolver(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_reset_solver_internals_delegate",
        NewtonMPMManager._reset_solver_internals,
        raising=False,
    )

    NewtonManager.forward()

    assert world_mask.numpy().tolist() == [False]


# ---------------------------------------------------------------------------
# Manager class hierarchy and factory contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manager",
    [NewtonMJWarpManager, NewtonXPBDManager, NewtonFeatherstoneManager, NewtonKaminoManager, NewtonMPMManager],
)
def test_subclass_of_newton_manager(manager):
    """All concrete managers inherit from :class:`NewtonManager`."""
    assert issubclass(manager, NewtonManager)
    # Subclasses must override the abstract factory.
    assert manager._build_solver is not NewtonManager._build_solver
    assert manager._create_solver is not NewtonManager._create_solver


def test_abstract_build_solver_raises():
    """Calling :meth:`_build_solver` on the abstract base raises."""
    with pytest.raises(NotImplementedError):
        NewtonManager._build_solver(model=None, solver_cfg=NewtonSolverCfg())


def test_abstract_create_solver_raises():
    """Calling :meth:`_create_solver` on the base manager raises."""
    with pytest.raises(NotImplementedError):
        NewtonManager._create_solver(model=None, solver_cfg=NewtonSolverCfg())


@pytest.mark.parametrize(
    "manager",
    [NewtonMJWarpManager, NewtonXPBDManager, NewtonFeatherstoneManager, NewtonKaminoManager, NewtonMPMManager],
)
def test_manager_name_starts_with_newton(manager):
    """The ``"newton"`` prefix is required by :class:`InteractiveScene` and the
    various backend factories that dispatch on ``physics_manager.__name__.lower()``.
    """
    assert manager.__name__.lower().startswith("newton")


# ---------------------------------------------------------------------------
# End-to-end: build each solver via SimulationContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, expected_solver_cls,"
    " expected_use_single_state, expected_needs_collision_pipeline",
    SOLVER_MATRIX,
)
def test_initialize_solver_populates_canonical_state(
    solver_cfg_factory,
    expected_manager,
    expected_solver_cls,
    expected_use_single_state,
    expected_needs_collision_pipeline,
):
    """End-to-end: ``SimulationContext`` resolves the right manager subclass and
    ``initialize_solver`` lands the right solver + flags on :class:`NewtonManager`.

    External code reads :class:`NewtonManager` attributes directly (``_solver``,
    ``_use_single_state``, ``_needs_collision_pipeline``).  Even though dispatch
    runs through a leaf subclass (e.g. :class:`NewtonMJWarpManager`), shared
    state is assigned through the explicit base class so that those reads keep
    working regardless of which leaf is active.  This test is the regression
    guard for that contract.

    The builder is pre-populated directly (instead of relying on a USD stage)
    with either a minimal particle grid for MPM or a one-body / one-joint scene
    for rigid/articulation solvers:

    1. :class:`SolverImplicitMPM` requires particles and MPM custom attributes
       registered on the builder before particle creation.
    2. :class:`SolverMuJoCo` requires at least one joint to convert the model
       to MJCF; a ground-plane-only scene fails MJCF conversion.
    3. Kamino's internal collision detector requires collidable geometry to
       construct its collision pipeline.
    4. Pre-populating ``NewtonManager._builder`` causes
       :meth:`NewtonManager.start_simulation` to skip
       :meth:`instantiate_builder_from_stage`, so the test does not depend on
       USD asset packages.
    """
    solver_cfg = solver_cfg_factory()
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        # Resolved manager class matches the expected leaf.
        resolved_manager = sim.physics_manager
        # ``physics_manager`` is a LazyType proxy — compare by ``__name__`` to
        # avoid forcing identity-by-id checks against the unresolved proxy.
        assert resolved_manager.__name__ == expected_manager.__name__
        assert resolved_manager.__name__.lower().startswith("newton")

        builder = resolved_manager.create_builder()
        if expected_solver_cls is SolverImplicitMPM:
            assert builder.has_custom_attribute("mpm:young_modulus")
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.10),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                mass=0.01,
                jitter=0.0,
                radius_mean=0.02,
            )
        else:
            # Pre-populate the builder with a minimal scene so MJCF conversion has
            # something to work with.
            body = builder.add_body(mass=1.0)
            builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
            if isinstance(solver_cfg, KaminoSolverCfg) and solver_cfg.use_collision_detector:
                builder.add_shape_sphere(body=body, radius=0.05)
                builder.add_ground_plane()
        NewtonManager.set_builder(builder)

        # Force resolution and bring up the solver.
        sim.reset()

        # Canonical state lives on the base class.
        assert NewtonManager._solver is not None
        assert isinstance(NewtonManager._solver, expected_solver_cls)
        assert NewtonManager._use_single_state is expected_use_single_state
        assert NewtonManager._needs_collision_pipeline is expected_needs_collision_pipeline
        assert NewtonManager._reset_solver_internals_delegate.__self__ is expected_manager
        assert (
            NewtonManager._reset_solver_internals_delegate.__func__ is expected_manager._reset_solver_internals.__func__
        )

        # ``_contacts`` is allocated whichever way contacts are handled
        # (MuJoCo internal buffer or Newton pipeline output).
        # Kamino with internal contacts and MPM do not currently set NewtonManager._contacts.
        if expected_solver_cls not in (SolverKamino, SolverImplicitMPM):
            assert NewtonManager._contacts is not None

        # One step should not raise — proves the dispatch wiring lines up
        # end-to-end.  (We do not assert physics; that's covered by the
        # asset/sensor test suites.)
        sim.step(render=False)


def test_mjwarp_internal_contacts_with_collision_cfg_raises():
    """Combining ``use_mujoco_contacts=True`` with a ``collision_cfg`` is rejected.

    The check lives in :meth:`NewtonMJWarpManager._build_solver` because it
    needs both the solver cfg subtype and the parent :class:`NewtonCfg`, so it
    fires during :meth:`NewtonManager.initialize_solver` (i.e. on
    ``sim.reset()``) rather than at cfg construction time.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True),
            collision_cfg=NewtonCollisionPipelineCfg(),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        with pytest.raises(ValueError, match="collision_cfg cannot be set"):
            sim.reset()


@pytest.mark.parametrize(
    "num_substeps, collision_decimation, expected_mid_loop_collides",
    [
        (8, 0, 0),  # Feature disabled.
        (8, 2, 3),  # Re-collide after substeps 2, 4, 6 (skip last).
        (8, 4, 1),  # Re-collide after substep 4 only.
        (8, 7, 1),  # Re-collide after substep 7 only.
        (8, 8, 0),  # Gated off (>= num_substeps).
    ],
)
def test_collision_decimation_invokes_mid_loop_collide(num_substeps, collision_decimation, expected_mid_loop_collides):
    """``_run_solver_substeps`` re-invokes ``collide`` at the expected substeps.

    Wraps :attr:`NewtonManager._collision_pipeline.collide` with a counter and
    runs one physics tick. The collide-call count is ``1`` (top-of-tick) plus
    one per matching mid-loop substep, excluding the last substep.

    The scene has a free-joint sphere falling onto a ground plane so the
    broadphase actually generates pairs — guards against a future change
    that skips ``collide()`` when there are no collidable shapes.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            num_substeps=num_substeps,
            collision_decimation=collision_decimation,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = sim.physics_manager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_sphere(body=body, radius=0.05)
        builder.add_ground_plane()
        # Lift the sphere to 0.5 m above the plane so the scene is non-degenerate.
        # joint_q for a free joint is [tx, ty, tz, qx, qy, qz, qw].
        builder.joint_q[-7:] = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        NewtonManager.set_builder(builder)
        sim.reset()

        # Wrap collide() with a counter — must run after sim.reset() so the
        # pipeline is allocated, and use_cuda_graph=False so the wrapped
        # Python callable isn't bypassed by a captured graph.
        calls = {"n": 0}
        original_collide = NewtonManager._collision_pipeline.collide

        def counting_collide(state, contacts):
            calls["n"] += 1
            return original_collide(state, contacts)

        NewtonManager._collision_pipeline.collide = counting_collide
        try:
            sim.step(render=False)
        finally:
            NewtonManager._collision_pipeline.collide = original_collide

        # Expect: 1 (top-of-tick) + expected_mid_loop_collides.
        assert calls["n"] == 1 + expected_mid_loop_collides


# ---------------------------------------------------------------------------
# Regression: an env reset written through the data layer must land in the
# manager's canonical _state_0 after an odd number of steps when CUDA graphs
# are disabled (the use_cuda_graph state-swap gating bug).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 3])
def test_reset_lands_in_state_0_after_odd_kamino_steps_without_cuda_graph(num_steps):
    """An env reset written through the data-layer binding lands in ``_state_0``.

    Kamino is double-buffered (``_use_single_state=False``), so each substep
    ping-pongs ``_state_0`` / ``_state_1``. With a single substep the loop must
    copy the result back into ``_state_0`` instead of swapping, otherwise after
    an *odd* number of steps the canonical ``_state_0`` ends up on the other
    buffer. This copy-on-last was previously gated on ``use_cuda_graph``, so with
    CUDA graphs disabled ``_state_0`` flipped buffers and env-reset writes landed
    in the stale buffer.

    :class:`~isaaclab_newton.assets.ArticulationData` binds its joint-state write
    target to ``_state_0.joint_q`` once at setup (``_sim_bind_joint_pos``) and
    never re-binds on env resets, so a flipped ``_state_0`` makes reset writes
    miss the live state. This test reproduces that contract without a full USD
    articulation: it caches the same ``_state_0.joint_q`` binding, steps Kamino an
    odd number of times, writes a sentinel through the cached binding (mimicking
    the reset write), and asserts the manager's ``_state_0`` observes it.

    Without the fix the swap-on-last flips ``_state_0`` for odd ``num_steps`` and
    the sentinel lands in ``_state_1`` instead, so the final assertion fails.
    """
    sentinel = 1.2345
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=KaminoSolverCfg(),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = NewtonManager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)
        sim.reset()

        # Kamino keeps separate input/output states; the bug only exists there.
        assert NewtonManager._use_single_state is False
        # The data layer binds its joint-state write target to _state_0 at setup.
        reset_target = NewtonManager._state_0.joint_q
        assert reset_target.shape[0] > 0  # guard against a vacuous assertion

        for _ in range(num_steps):
            sim.step(render=False)

        # An env reset writes joint state through the (still bound) target.
        reset_target.fill_(sentinel)

        # The reset must be visible in the manager's canonical _state_0; if the
        # buffer flipped it landed in _state_1 instead.
        canonical_joint_q = NewtonManager._state_0.joint_q.numpy()
        assert np.allclose(canonical_joint_q, sentinel), (
            f"reset write did not land in _state_0 after {num_steps} steps: {canonical_joint_q}"
        )


def _build_collision_scene(sim, num_boxes=8):
    """Add ``num_boxes`` free-falling boxes over a ground plane.

    Uses ``MJWarpSolverCfg(use_mujoco_contacts=False)`` so the Newton collision
    pipeline / contacts are allocated on ``sim.reset()``.
    """
    builder = sim.physics_manager.create_builder()
    for _ in range(num_boxes):
        body = builder.add_body(mass=1.0)
        builder.add_joint_free(child=body)
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    NewtonManager.set_builder(builder)


# Model device arrays ``CollisionPipeline.collide()`` reads off its cached model.
_COLLIDE_MODEL_ARRAYS = (
    "shape_transform",
    "shape_body",
    "shape_type",
    "shape_scale",
    "shape_collision_radius",
    "shape_source_ptr",
    "shape_margin",
    "shape_gap",
    "shape_collision_aabb_lower",
    "shape_collision_aabb_upper",
)


def _free_model_collide_arrays_and_churn(model, device):
    """Free the arrays ``collide()`` reads off ``model``, then churn the allocator.

    Reusing the freed blocks mimics the GPU memory pressure a real workload
    applies between resets, so a stale pipeline still pointing at ``model``
    would read overwritten memory on its next ``collide()``.
    """
    import gc

    for attr in _COLLIDE_MODEL_ARRAYS:
        arr = getattr(model, attr, None)
        if isinstance(arr, wp.array) and arr.device.is_cuda:
            setattr(model, attr, None)
    gc.collect()
    wp.synchronize_device(device)
    _churn = [wp.zeros(1 << 16, dtype=wp.float32, device=device) for _ in range(128)]  # noqa: F841
    wp.synchronize_device(device)


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_hard_reset_then_step_runs(use_cuda_graph):
    """A step after a second (hard) ``sim.reset()`` runs without a CUDA error.

    Drives reset -> step -> hard reset, frees the old model's collide arrays and
    churns the allocator to mimic GPU memory pressure, then steps and syncs.
    Without the fix the stale pipeline reads the freed buffers and faults
    (CUDA 700). Run with CUDA graphs off and on.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
            num_substeps=2,
            use_cuda_graph=use_cuda_graph,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        _build_collision_scene(sim)

        sim.reset()
        assert NewtonManager._needs_collision_pipeline is True
        old_model = NewtonManager._collision_pipeline.model
        sim.step(render=False)

        sim.reset()

        _free_model_collide_arrays_and_churn(old_model, "cuda:0")

        # A hard device sync surfaces any deferred illegal access as an exception.
        sim.step(render=False)
        wp.synchronize_device("cuda:0")
