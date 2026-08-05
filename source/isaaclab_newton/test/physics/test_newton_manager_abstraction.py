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

import gc
import inspect
import warnings
import weakref
from contextlib import ExitStack, nullcontext
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


def test_unregister_pre_actuator_callback_removes_exact_callback_idempotently(monkeypatch) -> None:
    """Removing a staged-gather callback preserves equal registrations."""
    callbacks: list[object] = []
    monkeypatch.setattr(NewtonManager, "_pre_actuator_callbacks", callbacks, raising=False)

    class _EqualCallback:
        def __call__(self) -> None:
            pass

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _EqualCallback)

    first = _EqualCallback()
    equal_but_distinct = _EqualCallback()
    NewtonManager.register_pre_actuator_callback(first)
    NewtonManager.register_pre_actuator_callback(equal_but_distinct)

    NewtonManager.unregister_pre_actuator_callback(first)
    NewtonManager.unregister_pre_actuator_callback(first)

    assert len(callbacks) == 1
    assert callbacks[0] is equal_but_distinct


def test_simulate_full_runs_pre_actuator_callbacks_before_every_actuator_step(monkeypatch) -> None:
    """A staged gather occurs before each decimated native actuator computation."""
    events: list[str] = []

    class _Adapter:
        gathered_ranges: list[object] = []

        def gather_staged_ranges(self, ranges: object) -> None:
            self.gathered_ranges.append(ranges)

        def step(self, *_args) -> None:
            assert self.gathered_ranges[-1] == "canonical"
            events.append("actuator")

    adapter = _Adapter()
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.01, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_decimation", 2, raising=False)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_adapter", adapter, raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_pre_actuator_callbacks",
        [lambda: (adapter.gather_staged_ranges("canonical"), events.append("pre"))],
        raising=False,
    )
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [lambda: events.append("post")], raising=False)
    monkeypatch.setattr(NewtonManager, "_post_step_callbacks", [lambda: events.append("post_step")], raising=False)
    monkeypatch.setattr(
        NewtonManager, "_run_solver_substeps", classmethod(lambda cls, contacts: events.append("solver"))
    )
    monkeypatch.setattr(NewtonManager, "_update_sensors", classmethod(lambda cls, contacts: events.append("sensors")))

    NewtonManager._simulate_full()

    assert events == ["pre", "actuator", "post", "solver", "pre", "actuator", "post", "solver", "post_step", "sensors"]
    assert adapter.gathered_ranges == ["canonical", "canonical"]


def test_eager_step_runs_pre_actuator_callbacks_before_native_computation(monkeypatch) -> None:
    """The non-graphable fallback gathers staged inputs before its single actuator step."""
    from isaaclab.physics import PhysicsManager

    events: list[str] = []

    class _Adapter:
        def step(self, *_args) -> None:
            events.append("actuator")

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", SimpleNamespace(use_cuda_graph=False), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)
    monkeypatch.setattr(PhysicsManager, "_sim_time", 0.0, raising=False)
    monkeypatch.setattr(NewtonManager, "_reset_solver_internals_delegate", lambda _mask: None, raising=False)
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver_dt", 0.01, raising=False)
    monkeypatch.setattr(NewtonManager, "_num_substeps", 1, raising=False)
    monkeypatch.setattr(NewtonManager, "_adapter", _Adapter(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_control", object(), raising=False)
    monkeypatch.setattr(NewtonManager, "_pre_actuator_callbacks", [lambda: events.append("pre")], raising=False)
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [lambda: events.append("post")], raising=False)
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: False))
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(lambda cls: events.append("physics")))
    monkeypatch.setattr(NewtonManager, "_log_solver_debug", classmethod(lambda cls: None))

    NewtonManager.step()

    assert events == ["pre", "actuator", "post", "physics"]


def test_env_zero_structural_mapping_preserves_unsorted_authored_joint_order() -> None:
    """Builder structural matching retains every unsorted env-0 DOF."""
    from isaaclab_newton.physics.newton_manager import _build_env_zero_actuator_metadata

    first, second = ("first",), ("second",)
    entries = (
        (first, SimpleNamespace(indices=[2, 0, 5, 3])),
        (second, SimpleNamespace(indices=[1, 4])),
    )

    signatures, local_indices = _build_env_zero_actuator_metadata(entries, dofs_per_env=3)

    assert signatures == {2: (first,), 0: (first,), 1: (second,)}
    assert local_indices == {first: (2, 0), second: (1,)}


def test_build_newton_actuator_defaults_warns_once_and_preserves_empty_result(monkeypatch) -> None:
    """The deprecated compatibility helper retains its historical empty result."""
    import isaaclab_newton.actuators.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_DEFAULTS_DEPRECATION_EMITTED", False)
    with pytest.warns(DeprecationWarning, match="ActuatorCollection"):
        stiffness, damping, joint_indices = adapter_module.build_newton_actuator_defaults(
            [], num_envs=2, num_joints=3, dof_offset=0, env_stride=3, device="cpu"
        )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        repeated = adapter_module.build_newton_actuator_defaults(
            [], num_envs=2, num_joints=3, dof_offset=0, env_stride=3, device="cpu"
        )

    assert not recorded
    assert stiffness.shape == damping.shape == (2, 3)
    assert torch.count_nonzero(stiffness) == torch.count_nonzero(damping) == 0
    assert torch.equal(joint_indices, torch.empty(0, dtype=torch.int32))
    assert torch.equal(repeated[2], joint_indices)


def _legacy_newton_actuator_defaults_reference(
    actuators: list[object],
    num_envs: int,
    num_joints: int,
    dof_offset: int,
    env_stride: int,
    joint_user_to_backend_indices: tuple[int, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | slice]:
    """Literal pre-collection implementation used to pin compatibility behavior."""
    articulation_actuators = [
        actuator for actuator in actuators if dof_offset <= int(actuator.indices.numpy()[0]) < dof_offset + num_joints
    ]
    managed_local: set[int] = set()
    for actuator in articulation_actuators:
        per_actuator = actuator.indices.shape[0] // num_envs
        for global_dof in actuator.indices.numpy()[:per_actuator]:
            local_dof = int(global_dof) - dof_offset
            if 0 <= local_dof < num_joints:
                managed_local.add(local_dof)

    stiffness = torch.zeros((num_envs, num_joints), dtype=torch.float32)
    damping = torch.zeros_like(stiffness)
    for actuator in articulation_actuators:
        for value_index, global_dof in enumerate(actuator.indices.numpy()):
            relative_dof = int(global_dof) - dof_offset
            env_index = relative_dof // env_stride
            local_dof = relative_dof - env_index * env_stride
            if 0 <= env_index < num_envs and 0 <= local_dof < num_joints:
                if hasattr(actuator.controller, "kp"):
                    stiffness[env_index, local_dof] = float(actuator.controller.kp.numpy()[value_index])
                if hasattr(actuator.controller, "kd"):
                    damping[env_index, local_dof] = float(actuator.controller.kd.numpy()[value_index])

    if len(managed_local) == num_joints:
        joint_indices: torch.Tensor | slice = slice(None)
    else:
        joint_indices = torch.tensor(sorted(managed_local), dtype=torch.int32)
    if joint_user_to_backend_indices is not None:
        stiffness = stiffness[:, list(joint_user_to_backend_indices)]
        damping = damping[:, list(joint_user_to_backend_indices)]
        if not isinstance(joint_indices, slice):
            backend_to_user = [0] * num_joints
            for user_index, backend_index in enumerate(joint_user_to_backend_indices):
                backend_to_user[backend_index] = user_index
            joint_indices = torch.tensor(sorted(backend_to_user[index] for index in managed_local), dtype=torch.int32)
    return stiffness, damping, joint_indices


def test_build_newton_actuator_defaults_matches_legacy_nonempty_reversed_layout(monkeypatch) -> None:
    """The deprecated helper preserves values, types, ordering, offsets, and permutations."""
    import isaaclab_newton.actuators.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_DEFAULTS_DEPRECATION_EMITTED", True)
    offset, stride, num_envs, num_joints = 2, 7, 2, 4
    first = SimpleNamespace(
        controller=SimpleNamespace(
            kp=wp.array((120.0, 100.0, 121.0, 101.0), dtype=wp.float32, device="cpu"),
            kd=wp.array((12.0, 10.0, 12.1, 10.1), dtype=wp.float32, device="cpu"),
        ),
        indices=wp.array((4, 2, 11, 9), dtype=wp.uint32, device="cpu"),
    )
    second = SimpleNamespace(
        controller=SimpleNamespace(
            kp=wp.array((130.0, 131.0), dtype=wp.float32, device="cpu"),
            kd=wp.array((13.0, 13.1), dtype=wp.float32, device="cpu"),
        ),
        indices=wp.array((5, 12), dtype=wp.uint32, device="cpu"),
    )
    unrelated = SimpleNamespace(
        controller=SimpleNamespace(
            kp=wp.array((999.0, 999.0), dtype=wp.float32, device="cpu"),
            kd=wp.array((999.0, 999.0), dtype=wp.float32, device="cpu"),
        ),
        indices=wp.array((0, 7), dtype=wp.uint32, device="cpu"),
    )
    # Deliberately reverse the authored groups and request reverse public order.
    actuators = [unrelated, second, first]
    user_to_backend = (3, 2, 1, 0)
    expected = _legacy_newton_actuator_defaults_reference(
        actuators, num_envs, num_joints, offset, stride, user_to_backend
    )
    actual = adapter_module.build_newton_actuator_defaults(
        actuators, num_envs, num_joints, offset, stride, "cpu", user_to_backend
    )

    for actual_value, expected_value in zip(actual[:2], expected[:2], strict=True):
        assert type(actual_value) is torch.Tensor
        assert actual_value.dtype is torch.float32
        assert actual_value.device.type == "cpu"
        assert actual_value.shape == (num_envs, num_joints)
        torch.testing.assert_close(actual_value, expected_value)
    assert type(actual[2]) is torch.Tensor
    assert actual[2].dtype is torch.int32
    assert actual[2].shape == expected[2].shape
    torch.testing.assert_close(actual[2], expected[2])


def test_build_newton_actuator_defaults_preserves_slice_for_full_coverage(monkeypatch) -> None:
    """The legacy all-managed sentinel remains ``slice(None)`` under a permutation."""
    import isaaclab_newton.actuators.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_DEFAULTS_DEPRECATION_EMITTED", True)
    actuator = SimpleNamespace(
        controller=SimpleNamespace(
            kp=wp.array((10.0, 20.0, 30.0, 40.0), dtype=wp.float32, device="cpu"),
            kd=wp.array((1.0, 2.0, 3.0, 4.0), dtype=wp.float32, device="cpu"),
        ),
        indices=wp.array((0, 1, 2, 3), dtype=wp.uint32, device="cpu"),
    )

    expected = _legacy_newton_actuator_defaults_reference([actuator], 1, 4, 0, 4, (3, 2, 1, 0))
    actual = adapter_module.build_newton_actuator_defaults(
        [actuator], 1, 4, 0, 4, "cpu", joint_user_to_backend_indices=(3, 2, 1, 0)
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert actual[2] == expected[2] == slice(None)


def test_build_newton_actuator_defaults_remains_public_with_legacy_signature(monkeypatch) -> None:
    """Package import and positional argument order remain available during deprecation."""
    import isaaclab_newton.actuators as public_actuators
    import isaaclab_newton.actuators.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_DEFAULTS_DEPRECATION_EMITTED", True)
    assert public_actuators.build_newton_actuator_defaults is adapter_module.build_newton_actuator_defaults
    assert "build_newton_actuator_defaults" in public_actuators.__all__
    signature = inspect.signature(public_actuators.build_newton_actuator_defaults)
    assert list(signature.parameters) == [
        "actuators",
        "num_envs",
        "num_joints",
        "dof_offset",
        "env_stride",
        "device",
        "joint_user_to_backend_indices",
    ]
    assert signature.parameters["joint_user_to_backend_indices"].default is None


def test_build_newton_actuator_defaults_warns_once_at_the_direct_caller(monkeypatch) -> None:
    """The deprecation points at its caller and is emitted only once per process."""
    import isaaclab_newton.actuators.adapter as adapter_module

    monkeypatch.setattr(adapter_module, "_DEFAULTS_DEPRECATION_EMITTED", False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warning_line = inspect.currentframe().f_lineno + 1
        adapter_module.build_newton_actuator_defaults([], 1, 0, 0, 0, "cpu")
        adapter_module.build_newton_actuator_defaults([], 1, 0, 0, 0, "cpu")

    assert len(recorded) == 1
    assert recorded[0].category is DeprecationWarning
    assert recorded[0].filename == __file__
    assert recorded[0].lineno == warning_line


def _make_hosted_actuator_stage(monkeypatch, parsed_actuators: list[object]):
    """Create a real in-memory USD traversal with controlled parsed actuator results."""
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


def test_usd_structural_occurrences_preserve_same_joint_interleaving(monkeypatch) -> None:
    """USD parsing retains ``[A, B, A]`` before Newton groups compatible entries."""
    import isaaclab_newton.actuators.adapter as adapter_module
    from newton.actuators import ControllerPD, Delay

    stage = _make_hosted_actuator_stage(
        monkeypatch,
        [
            SimpleNamespace(
                target_path="/Robot/joint",
                controller_class=ControllerPD,
                controller_kwargs={"kp": 2.0, "kd": 0.0},
                component_specs=[],
            ),
            SimpleNamespace(
                target_path="/Robot/joint",
                controller_class=ControllerPD,
                controller_kwargs={"kp": 11.0, "kd": 0.0},
                component_specs=[(Delay, {"delay_steps": 1})],
            ),
            SimpleNamespace(
                target_path="/Robot/joint",
                controller_class=ControllerPD,
                controller_kwargs={"kp": 7.0, "kd": 0.0},
                component_specs=[],
            ),
        ],
    )

    occurrences = adapter_module._structural_occurrences_from_usd(stage, ["joint"], "/Robot")

    assert len(occurrences[0]) == 3
    assert occurrences[0][0] == occurrences[0][2]
    assert occurrences[0][0] != occurrences[0][1]
    assert occurrences[0][1][1] is True


def test_usd_structural_occurrences_allow_articulation_without_actuators() -> None:
    """An uncovered articulation provides an explicit all-empty occurrence ledger."""
    import isaaclab_newton.actuators.adapter as adapter_module

    from pxr import Usd

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot")

    occurrences = adapter_module._structural_occurrences_from_usd(stage, ["joint_0", "joint_1"], "/Robot")

    assert occurrences == {0: (), 1: ()}


def test_hosted_direct_materialization_skips_mapped_parameter_expansion(monkeypatch) -> None:
    """A whole native exact type reads mapped parameters from canonical storage at construction."""
    import isaaclab_newton.actuators.adapter as adapter_module
    from newton.actuators import ClampingMaxEffort, ControllerPD

    from isaaclab.actuators import IdealPDActuator
    from isaaclab.utils.warp import ProxyArray

    parsed = [
        SimpleNamespace(
            target_path=f"/Robot/joint_{index}",
            controller_class=ControllerPD,
            controller_kwargs={"kp": 100.0 + index, "kd": 10.0 + index},
            component_specs=[(ClampingMaxEffort, {"max_effort": 50.0 + index})],
        )
        for index in range(2)
    ]
    stage = _make_hosted_actuator_stage(monkeypatch, parsed)
    canonical = {
        name: ProxyArray(wp.array(values, dtype=wp.float32, device="cpu"))
        for name, values in {
            "stiffness": [[11.0, 12.0], [21.0, 22.0]],
            "damping": [[1.1, 1.2], [2.1, 2.2]],
            "effort_limit": [[31.0, 32.0], [41.0, 42.0]],
            "computed_effort": [[0.0, 0.0], [0.0, 0.0]],
            "applied_effort": [[0.0, 0.0], [0.0, 0.0]],
        }.items()
    }
    group = SimpleNamespace(
        joint_indices=(0, 1),
        joint_names=("joint_0", "joint_1"),
        _parameter_binding=SimpleNamespace(arrays=canonical),
    )
    binding = SimpleNamespace(
        groups={"native": group},
        native_group_names=frozenset({"native"}),
        layout=SimpleNamespace(
            num_joints=2,
            type_layouts={IdealPDActuator: SimpleNamespace(num_worlds=2, num_dofs=2, compact_joint_indices=(0, 1))},
            group_layouts=(
                SimpleNamespace(
                    name="native",
                    actuator_type=IdealPDActuator,
                    joint_indices=(0, 1),
                    joint_names=("joint_0", "joint_1"),
                    type_slice=slice(0, 2),
                ),
            ),
        ),
    )
    expanded_sources = []
    original_launch = wp.launch

    def record_expansions(kernel, *args, **kwargs):
        if kernel is adapter_module._expand_env_major_values:
            expanded_sources.append(kwargs["inputs"][0].numpy().tolist())
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", record_expansions)

    adapter_module.NewtonActuatorAdapter._from_usd_binding(
        binding,
        stage=stage,
        joint_names=["joint_0", "joint_1"],
        num_envs=2,
        num_joints=2,
        device="cpu",
        articulation_prim_path="/Robot",
    )

    assert expanded_sources == [[0.0, 0.0]]


def test_hosted_direct_materialization_injects_canonical_pointers(monkeypatch) -> None:
    """A whole native exact type installs canonical parameters and clamped outputs before binding."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ClampingDCMotor, ControllerPD

    from isaaclab.actuators import DCMotor
    from isaaclab.utils.warp import ProxyArray

    parsed = [
        SimpleNamespace(
            target_path=f"/Robot/joint_{index}",
            controller_class=ControllerPD,
            controller_kwargs={"kp": 100.0 + index, "kd": 10.0 + index},
            component_specs=[
                (
                    ClampingDCMotor,
                    {
                        "max_motor_effort": 50.0 + index,
                        "velocity_limit": 20.0 + index,
                        "saturation_effort": 80.0 + index,
                    },
                )
            ],
        )
        for index in range(2)
    ]
    stage = _make_hosted_actuator_stage(monkeypatch, parsed)
    canonical = {
        name: ProxyArray(wp.array(values, dtype=wp.float32, device="cpu"))
        for name, values in {
            "stiffness": [[11.0, 12.0], [21.0, 22.0]],
            "damping": [[1.1, 1.2], [2.1, 2.2]],
            "effort_limit": [[31.0, 32.0], [41.0, 42.0]],
            "velocity_limit": [[51.0, 52.0], [61.0, 62.0]],
            "saturation_effort": [[71.0, 72.0], [81.0, 82.0]],
            "computed_effort": [[0.0, 0.0], [0.0, 0.0]],
            "applied_effort": [[0.0, 0.0], [0.0, 0.0]],
        }.items()
    }
    group = SimpleNamespace(
        joint_indices=(0, 1),
        joint_names=("joint_0", "joint_1"),
        _parameter_binding=SimpleNamespace(arrays=canonical),
    )
    binding = SimpleNamespace(
        groups={"native": group},
        native_group_names=frozenset({"native"}),
        layout=SimpleNamespace(
            num_joints=2,
            type_layouts={DCMotor: SimpleNamespace(num_worlds=2, num_dofs=2, compact_joint_indices=(0, 1))},
            group_layouts=(
                SimpleNamespace(
                    name="native",
                    actuator_type=DCMotor,
                    joint_indices=(0, 1),
                    joint_names=("joint_0", "joint_1"),
                    type_slice=slice(0, 2),
                ),
            ),
        ),
    )

    adapter = NewtonActuatorAdapter._from_usd_binding(
        binding,
        stage=stage,
        joint_names=["joint_0", "joint_1"],
        num_envs=2,
        num_joints=2,
        device="cpu",
        articulation_prim_path="/Robot",
    )

    actuator = adapter.actuators[0]
    assert actuator.controller.kp.ptr == canonical["stiffness"].warp.ptr
    assert actuator.controller.kd.ptr == canonical["damping"].warp.ptr
    assert actuator.clamping[0].max_motor_effort.ptr == canonical["effort_limit"].warp.ptr
    assert actuator.clamping[0].velocity_limit.ptr == canonical["velocity_limit"].warp.ptr
    assert actuator.clamping[0].saturation_effort.ptr == canonical["saturation_effort"].warp.ptr
    assert actuator._computed_forces.ptr == canonical["computed_effort"].warp.ptr
    assert actuator._applied_forces.ptr == canonical["applied_effort"].warp.ptr


def test_hosted_same_signature_across_two_lab_types_uses_staging(monkeypatch) -> None:
    """One Newton signature shared by two Lab exact types cannot alias either canonical type."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ClampingMaxEffort, ControllerPD

    from isaaclab.utils.warp import ProxyArray

    class _FirstType:
        pass

    class _SecondType:
        pass

    parsed = [
        SimpleNamespace(
            target_path=f"/Robot/joint_{index}",
            controller_class=ControllerPD,
            controller_kwargs={"kp": 100.0 + index, "kd": 10.0 + index},
            component_specs=[(ClampingMaxEffort, {"max_effort": 50.0 + index})],
        )
        for index in range(2)
    ]
    stage = _make_hosted_actuator_stage(monkeypatch, parsed)

    def make_canonical(seed: float) -> dict[str, ProxyArray]:
        return {
            name: ProxyArray(wp.array([[seed], [seed + 1.0]], dtype=wp.float32, device="cpu"))
            for name in ("stiffness", "damping", "effort_limit", "computed_effort", "applied_effort")
        }

    first_arrays = make_canonical(10.0)
    second_arrays = make_canonical(20.0)
    first_group = SimpleNamespace(
        joint_indices=(0,),
        joint_names=("joint_0",),
        _parameter_binding=SimpleNamespace(arrays=first_arrays),
    )
    second_group = SimpleNamespace(
        joint_indices=(1,),
        joint_names=("joint_1",),
        _parameter_binding=SimpleNamespace(arrays=second_arrays),
    )
    group_layouts = (
        SimpleNamespace(
            name="first",
            actuator_type=_FirstType,
            joint_indices=(0,),
            joint_names=("joint_0",),
            type_slice=slice(0, 1),
        ),
        SimpleNamespace(
            name="second",
            actuator_type=_SecondType,
            joint_indices=(1,),
            joint_names=("joint_1",),
            type_slice=slice(0, 1),
        ),
    )
    binding = SimpleNamespace(
        groups={"first": first_group, "second": second_group},
        native_group_names=frozenset({"first", "second"}),
        computed_effort=ProxyArray(wp.zeros((2, 2), dtype=wp.float32, device="cpu")),
        layout=SimpleNamespace(
            num_joints=2,
            type_layouts={
                _FirstType: SimpleNamespace(num_worlds=2, num_dofs=1, compact_joint_indices=(0,)),
                _SecondType: SimpleNamespace(num_worlds=2, num_dofs=1, compact_joint_indices=(1,)),
            },
            group_layouts=group_layouts,
        ),
    )

    adapter = NewtonActuatorAdapter._from_usd_binding(
        binding,
        stage=stage,
        joint_names=["joint_0", "joint_1"],
        num_envs=2,
        num_joints=2,
        device="cpu",
        articulation_prim_path="/Robot",
    )
    native_binding = adapter.bind_articulation(binding, dof_offset=0)

    actuator = adapter.actuators[0]
    assert len(native_binding.ranges) == 2
    assert all(not range_binding.direct for range_binding in native_binding.ranges)
    assert actuator.controller.kp.ptr not in {first_arrays["stiffness"].warp.ptr, second_arrays["stiffness"].warp.ptr}
    assert actuator._computed_forces.ptr not in {
        first_arrays["computed_effort"].warp.ptr,
        second_arrays["computed_effort"].warp.ptr,
    }


def test_hosted_materialization_resolves_each_joint_recipe_once(monkeypatch) -> None:
    """Parsing caches one controller, component, and signature resolution per authored joint."""
    import isaaclab_newton.actuators.adapter as adapter_module
    from newton.actuators import ClampingMaxEffort, ControllerPD

    from isaaclab.actuators import IdealPDActuator
    from isaaclab.utils.warp import ProxyArray

    parsed = [
        SimpleNamespace(
            target_path=f"/Robot/joint_{index}",
            controller_class=ControllerPD,
            controller_kwargs={"kp": 100.0 + index, "kd": 10.0 + index},
            component_specs=[(ClampingMaxEffort, {"max_effort": 50.0 + index})],
        )
        for index in range(2)
    ]
    stage = _make_hosted_actuator_stage(monkeypatch, parsed)
    canonical = {
        name: ProxyArray(wp.zeros((1, 2), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "effort_limit", "computed_effort", "applied_effort")
    }
    group = SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=canonical))
    binding = SimpleNamespace(
        groups={"native": group},
        native_group_names=frozenset({"native"}),
        layout=SimpleNamespace(
            num_joints=2,
            type_layouts={IdealPDActuator: SimpleNamespace(num_worlds=1, num_dofs=2, compact_joint_indices=(0, 1))},
            group_layouts=(
                SimpleNamespace(
                    name="native",
                    actuator_type=IdealPDActuator,
                    joint_indices=(0, 1),
                    joint_names=("joint_0", "joint_1"),
                    type_slice=slice(0, 2),
                ),
            ),
        ),
    )
    counts = {"controller": 0, "clamping": 0, "signature": 0}
    original_controller_resolve = ControllerPD.resolve_arguments
    original_clamping_resolve = ClampingMaxEffort.resolve_arguments
    original_signature = adapter_module._actuator_signature

    def resolve_controller(cls, arguments):
        del cls
        counts["controller"] += 1
        return original_controller_resolve(arguments)

    def resolve_clamping(cls, arguments):
        del cls
        counts["clamping"] += 1
        return original_clamping_resolve(arguments)

    def build_signature(*args, **kwargs):
        counts["signature"] += 1
        return original_signature(*args, **kwargs)

    monkeypatch.setattr(ControllerPD, "resolve_arguments", classmethod(resolve_controller))
    monkeypatch.setattr(ClampingMaxEffort, "resolve_arguments", classmethod(resolve_clamping))
    monkeypatch.setattr(adapter_module, "_actuator_signature", build_signature)

    adapter_module.NewtonActuatorAdapter._from_usd_binding(
        binding,
        stage=stage,
        joint_names=["joint_0", "joint_1"],
        num_envs=1,
        num_joints=2,
        device="cpu",
        articulation_prim_path="/Robot",
    )

    assert counts == {"controller": 2, "clamping": 2, "signature": 2}


@pytest.mark.parametrize("device", ("cpu", *(str(device) for device in wp.get_cuda_devices())))
def test_native_adapter_direct_binding_keeps_canonical_pointers_without_copy_or_sync(monkeypatch, device: str) -> None:
    """Compatible native ranges bind canonical pointers without copying or synchronizing."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter

    from isaaclab.utils.warp import ProxyArray

    class _NativeType:
        pass

    canonical = {
        name: ProxyArray(wp.zeros((2, 2), dtype=wp.float32, device=device))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    controller = SimpleNamespace(
        kp=wp.zeros(4, dtype=wp.float32, device=device), kd=wp.zeros(4, dtype=wp.float32, device=device)
    )
    actuator = SimpleNamespace(
        indices=wp.array([0, 1, 2, 3], dtype=wp.uint32, device=device),
        controller=controller,
        clamping=(),
        _computed_forces=wp.zeros(4, dtype=wp.float32, device=device),
        _applied_forces=wp.zeros(4, dtype=wp.float32, device=device),
    )
    group = SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=canonical))
    binding = SimpleNamespace(
        groups={"native": group},
        native_group_names=frozenset({"native"}),
        computed_effort=canonical["computed_effort"],
        layout=SimpleNamespace(
            num_joints=2,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=2, num_dofs=2, compact_joint_indices=(0, 1))},
            group_layouts=(
                SimpleNamespace(
                    name="native",
                    actuator_type=_NativeType,
                    joint_indices=(0, 1),
                    joint_names=("joint_0", "joint_1"),
                    type_slice=slice(0, 2),
                ),
            ),
        ),
    )
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter.actuators = [actuator]
    adapter._device = device
    adapter.num_joints = 2
    signature = ("native",)
    adapter._actuators_by_signature = {signature: actuator}
    adapter._joint_signatures = {"joint_0": signature, "joint_1": signature}
    adapter._dof_signatures = {}
    adapter._actuator_dof_indices = {signature: (0, 1)}

    def _forbid(*args, **kwargs):
        raise AssertionError("direct canonical binding must not copy, synchronize, or read host values")

    canonical_pointers = {value.warp.ptr for value in canonical.values()}
    original_copy = wp.copy
    original_to = torch.Tensor.to

    def _forbid_canonical_copy(destination, source, *args, **kwargs):
        # Warp initializes small routing-metadata arrays through ``wp.copy``.
        # Only canonical value storage must remain copy-free during binding.
        arrays = (destination, source)
        if any(isinstance(array, wp.array) and array.ptr in canonical_pointers for array in arrays):
            _forbid()
        return original_copy(destination, source, *args, **kwargs)

    def _forbid_host_to(tensor, *args, **kwargs):
        target = kwargs.get("device")
        if target is None and args:
            if isinstance(args[0], torch.Tensor):
                target = args[0].device
            elif isinstance(args[0], (str, torch.device, int)):
                target = args[0]
        if target is not None:
            target_device = torch.device(target)
            if target_device.type == "cpu" and tensor.device.type != "cpu":
                _forbid()
        return original_to(tensor, *args, **kwargs)

    with monkeypatch.context() as guarded:
        for method in ("clone", "cpu", "numpy", "tolist", "item", "copy_"):
            guarded.setattr(torch.Tensor, method, _forbid)
        guarded.setattr(torch.Tensor, "to", _forbid_host_to)
        guarded.setattr(wp, "copy", _forbid_canonical_copy)
        guarded.setattr(wp, "clone", _forbid)
        guarded.setattr(wp.array, "numpy", _forbid)
        for method in ("synchronize", "synchronize_device", "synchronize_event", "synchronize_stream"):
            if hasattr(wp, method):
                guarded.setattr(wp, method, _forbid)

        native_binding = adapter.bind_articulation(binding, dof_offset=0)

    (native_range,) = native_binding.ranges
    assert native_range.direct
    assert controller.kp.ptr == canonical["stiffness"].warp.ptr
    assert controller.kd.ptr == canonical["damping"].warp.ptr
    assert actuator._computed_forces.ptr == canonical["computed_effort"].warp.ptr
    assert actuator._applied_forces.ptr == canonical["applied_effort"].warp.ptr


@pytest.mark.parametrize("clamped", [False, True], ids=["unclamped", "dc_motor"])
def test_real_newton_actuator_two_step_output_transport(clamped: bool) -> None:
    """Real Newton controllers publish fresh computed and applied force every step."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ClampingDCMotor, ControllerPD

    from isaaclab.utils.warp import ProxyArray

    class _NativeType:
        pass

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    clamping = (
        [(ClampingDCMotor, {"saturation_effort": 1.0, "velocity_limit": 2.0, "max_motor_effort": 0.75})]
        if clamped
        else None
    )
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.5, clamping=clamping)
    model = builder.finalize(device="cpu")
    state, control = model.state(), model.control()
    (structural_key,) = builder.actuator_entries
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=1,
        dof_offset=0,
        device="cpu",
        actuator_keys=(structural_key,),
        actuator_dof_indices={structural_key: (0,)},
    )
    adapter._joint_signatures = {"joint": structural_key}
    arrays = {
        name: ProxyArray(wp.zeros((1, 1), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    arrays["stiffness"].warp.fill_(2.0)
    binding = SimpleNamespace(
        groups={"native": SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays))},
        native_group_names=frozenset({"native"}),
        computed_effort=arrays["computed_effort"],
        layout=SimpleNamespace(
            num_joints=1,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=1, num_dofs=1, compact_joint_indices=(0,))},
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
    native_binding = adapter.bind_articulation(binding, dof_offset=0)
    joint_computed = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    joint_applied = wp.zeros((1, 1), dtype=wp.float32, device="cpu")

    # The adapter owns output zeroing; feedforward stays on Newton's distinct
    # joint_act command field. Poison canonical outputs before the second step
    # so a stale output alias cannot accidentally pass this regression.
    for target, expected_computed in ((1.0, 2.25), (0.5, 1.25)):
        arrays["computed_effort"].warp.fill_(999.0)
        arrays["applied_effort"].warp.fill_(999.0)
        control.joint_target_q.fill_(target)
        control.joint_target_qd.zero_()
        control.joint_act.fill_(0.25)
        adapter.step(state, control, dt=0.01)
        adapter.publish_outputs(native_binding.ranges, joint_computed, joint_applied)

        expected_applied = min(expected_computed, 0.75) if clamped else expected_computed
        assert control.joint_f.numpy()[0] == pytest.approx(expected_applied)
        assert arrays["computed_effort"].warp.numpy()[0, 0] == pytest.approx(expected_computed)
        assert arrays["applied_effort"].warp.numpy()[0, 0] == pytest.approx(expected_applied)
        assert joint_computed.numpy()[0, 0] == pytest.approx(expected_computed)
        assert joint_applied.numpy()[0, 0] == pytest.approx(expected_applied)


def test_global_native_actuator_staging_survives_two_articulation_registrations() -> None:
    """A globally merged controller never rebinds one articulation over another."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter

    from isaaclab.utils.warp import ProxyArray

    class _NativeType:
        pass

    controller = SimpleNamespace(
        kp=wp.zeros(4, dtype=wp.float32, device="cpu"), kd=wp.zeros(4, dtype=wp.float32, device="cpu")
    )
    actuator = SimpleNamespace(
        indices=wp.array([0, 1, 2, 3], dtype=wp.uint32, device="cpu"),
        controller=controller,
        clamping=(),
        _computed_forces=wp.zeros(4, dtype=wp.float32, device="cpu"),
        _applied_forces=wp.zeros(4, dtype=wp.float32, device="cpu"),
    )
    original_kp = controller.kp

    def binding(key: str) -> SimpleNamespace:
        arrays = {
            name: ProxyArray(wp.zeros((2, 1), dtype=wp.float32, device="cpu"))
            for name in ("stiffness", "damping", "computed_effort", "applied_effort")
        }
        return SimpleNamespace(
            groups={"native": SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays))},
            native_group_names=frozenset({"native"}),
            computed_effort=arrays["computed_effort"],
            registration=SimpleNamespace(key=key),
            layout=SimpleNamespace(
                num_joints=1,
                type_layouts={_NativeType: SimpleNamespace(num_worlds=2, num_dofs=1, compact_joint_indices=(0,))},
                group_layouts=(
                    SimpleNamespace(
                        name="native",
                        actuator_type=_NativeType,
                        joint_indices=(0,),
                        joint_names=("joint_0",),
                        type_slice=slice(0, 1),
                    ),
                ),
            ),
        )

    adapter = object.__new__(NewtonActuatorAdapter)
    adapter.actuators = [actuator]
    adapter._device = "cpu"
    adapter.num_joints = 2
    adapter._num_envs = 2
    adapter._global_native_bindings = {}
    signature = ("native",)
    adapter._actuators_by_signature = {signature: actuator}
    adapter._joint_signatures = {"joint_0": signature}
    adapter._dof_signatures = {}
    adapter._actuator_dof_indices = {signature: (0, 1)}

    first = adapter.bind_articulation(binding("first"), dof_offset=0)
    first_kp = controller.kp
    second = adapter.bind_articulation(binding("second"), dof_offset=1)

    assert not first.ranges[0].direct
    assert not second.ranges[0].direct
    assert controller.kp is first_kp
    adapter.unregister_articulation_ranges(first.ranges)
    assert controller.kp is first_kp
    adapter.unregister_articulation_ranges(second.ranges)
    assert controller.kp is original_kp


def test_native_adapter_segments_a_b_a_exact_type_by_structural_key() -> None:
    """One exact type binds [A, B, A] through two persistent global controllers."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter

    from isaaclab.utils.warp import ProxyArray

    class _NativeType:
        pass

    def make_actuator(indices: list[int]) -> SimpleNamespace:
        size = len(indices)
        return SimpleNamespace(
            indices=wp.array(indices, dtype=wp.uint32, device="cpu"),
            controller=SimpleNamespace(kp=wp.zeros(size, dtype=wp.float32), kd=wp.zeros(size, dtype=wp.float32)),
            clamping=(),
            _computed_forces=wp.zeros(size, dtype=wp.float32),
            _applied_forces=wp.zeros(size, dtype=wp.float32),
        )

    actuator_a = make_actuator([0, 2, 3, 5])
    actuator_b = make_actuator([1, 4])
    canonical = {
        name: ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    groups = (
        SimpleNamespace(
            name="a0", actuator_type=_NativeType, joint_indices=(0,), joint_names=("j0",), type_slice=slice(0, 1)
        ),
        SimpleNamespace(
            name="b", actuator_type=_NativeType, joint_indices=(1,), joint_names=("j1",), type_slice=slice(1, 2)
        ),
        SimpleNamespace(
            name="a1", actuator_type=_NativeType, joint_indices=(2,), joint_names=("j2",), type_slice=slice(2, 3)
        ),
    )
    binding = SimpleNamespace(
        groups={
            name: SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=canonical)) for name in ("a0", "b", "a1")
        },
        native_group_names=frozenset(("a0", "b", "a1")),
        computed_effort=canonical["computed_effort"],
        layout=SimpleNamespace(
            num_joints=3,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=2, num_dofs=3, compact_joint_indices=(0, 1, 2))},
            group_layouts=groups,
        ),
    )
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter.actuators = [actuator_a, actuator_b]
    adapter._device = "cpu"
    adapter.num_joints = 3
    adapter._num_envs = 2
    adapter._global_native_bindings = {}
    signature_a, signature_b = ("A",), ("B",)
    adapter._actuators_by_signature = {signature_a: actuator_a, signature_b: actuator_b}
    adapter._joint_signatures = {"j0": signature_a, "j1": signature_b, "j2": signature_a}
    adapter._dof_signatures = {}
    adapter._actuator_dof_indices = {signature_a: (0, 2), signature_b: (1,)}

    native_binding = adapter.bind_articulation(binding, dof_offset=0)

    assert [range_binding.actuator for range_binding in native_binding.ranges] == [actuator_a, actuator_b]
    assert all(not range_binding.direct for range_binding in native_binding.ranges)
    assert native_binding.ranges[0].canonical_slots.numpy().tolist() == [0, 2]
    assert native_binding.ranges[1].canonical_slots.numpy().tolist() == [1]


def test_staged_registration_keeps_native_pointers() -> None:
    """Staging uses Newton's persistent arrays without rebinding or cloning."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter

    controller = SimpleNamespace(
        kp=wp.zeros(2, dtype=wp.float32, device="cpu"),
        kd=wp.zeros(2, dtype=wp.float32, device="cpu"),
    )
    actuator = SimpleNamespace(
        indices=wp.array([0, 1], dtype=wp.uint32, device="cpu"),
        controller=controller,
        clamping=(),
        _computed_forces=wp.zeros(2, dtype=wp.float32, device="cpu"),
        _applied_forces=None,
    )
    original_kp = controller.kp
    original_kd = controller.kd
    original_computed = actuator._computed_forces
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter._device = "cpu"
    adapter.num_joints = 2
    adapter._num_envs = 1
    adapter._global_native_bindings = {}

    registration = adapter._register_staged_actuator(actuator, object())

    assert controller.kp is original_kp
    assert controller.kd is original_kd
    assert actuator._computed_forces is original_computed
    assert actuator._applied_forces is None
    assert registration.parameters[0][2] is original_kp
    assert registration.computed_effort is original_computed


def test_range_binding_failure_restores_prior_hosted_direct_alias(monkeypatch) -> None:
    """A later direct install failure restores an earlier hosted direct binding."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter

    from isaaclab.utils.warp import ProxyArray

    class _FirstType:
        pass

    class _SecondType:
        pass

    def make_actuator(index: int) -> SimpleNamespace:
        return SimpleNamespace(
            indices=wp.array([index], dtype=wp.uint32, device="cpu"),
            controller=SimpleNamespace(
                kp=wp.zeros(1, dtype=wp.float32, device="cpu"),
                kd=wp.zeros(1, dtype=wp.float32, device="cpu"),
            ),
            clamping=(),
            _computed_forces=wp.zeros(1, dtype=wp.float32, device="cpu"),
            _applied_forces=wp.zeros(1, dtype=wp.float32, device="cpu"),
        )

    first, second = make_actuator(0), make_actuator(1)
    original_kp = first.controller.kp
    arrays = {
        name: ProxyArray(wp.zeros((1, 1), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    groups = (
        SimpleNamespace(
            name="first", actuator_type=_FirstType, joint_indices=(0,), joint_names=("j0",), type_slice=slice(0, 1)
        ),
        SimpleNamespace(
            name="second", actuator_type=_SecondType, joint_indices=(1,), joint_names=("j1",), type_slice=slice(0, 1)
        ),
    )
    binding = SimpleNamespace(
        groups={
            name: SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays)) for name in ("first", "second")
        },
        native_group_names=frozenset(("first", "second")),
        computed_effort=arrays["computed_effort"],
        layout=SimpleNamespace(
            num_joints=2,
            type_layouts={
                _FirstType: SimpleNamespace(num_worlds=1, num_dofs=1, compact_joint_indices=(0,)),
                _SecondType: SimpleNamespace(num_worlds=1, num_dofs=1, compact_joint_indices=(1,)),
            },
            group_layouts=groups,
        ),
    )
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter.actuators = [first, second]
    adapter._device = "cpu"
    adapter.num_joints = 2
    adapter._num_envs = 1
    adapter._global_native_bindings = {}
    first_signature, second_signature = ("first",), ("second",)
    adapter._actuators_by_signature = {first_signature: first, second_signature: second}
    adapter._joint_signatures = {"j0": first_signature, "j1": second_signature}
    adapter._dof_signatures = {}
    adapter._actuator_dof_indices = {first_signature: (0,), second_signature: (1,)}
    adapter._direct_pointer_bindings = {}
    adapter._owns_actuators = True

    original_install = adapter._install_direct_pointer_binding
    calls = 0

    def fail_on_second_install(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second direct install failed")
        return original_install(*args, **kwargs)

    monkeypatch.setattr(adapter, "_install_direct_pointer_binding", fail_on_second_install)

    with pytest.raises(RuntimeError, match="second direct install failed"):
        adapter.bind_articulation(binding, dof_offset=0)

    assert first.controller.kp is original_kp
    assert second.controller.kp is not arrays["stiffness"].warp
    assert adapter._direct_pointer_bindings == {}


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


def test_native_rebuild_gathers_new_canonical_ranges_only_at_first_step(monkeypatch) -> None:
    """Rebuilding against one model defers each generation's canonical gather to its step."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module
    from newton import Model as NewtonModel

    first_ranges = (SimpleNamespace(direct=False),)
    second_ranges = (SimpleNamespace(direct=False),)
    direct_ranges = (SimpleNamespace(direct=True),)
    candidate_bindings = iter(
        (
            SimpleNamespace(
                implicit_dof_mask=wp.zeros(3, dtype=wp.int32, device="cpu"),
                implicit_dof_mask_owner=torch.zeros(3, dtype=torch.int32),
                ranges=first_ranges,
            ),
            SimpleNamespace(
                implicit_dof_mask=wp.zeros(3, dtype=wp.int32, device="cpu"),
                implicit_dof_mask_owner=torch.zeros(3, dtype=torch.int32),
                ranges=second_ranges,
            ),
            SimpleNamespace(
                implicit_dof_mask=wp.zeros(3, dtype=wp.int32, device="cpu"),
                implicit_dof_mask_owner=torch.zeros(3, dtype=torch.int32),
                ranges=direct_ranges,
            ),
        )
    )
    gathered_ranges: list[tuple[object, ...]] = []
    released_ranges: list[tuple[object, ...]] = []
    adapter = SimpleNamespace(
        bind_articulation=lambda *_args, **_kwargs: next(candidate_bindings),
        gather_staged_ranges=lambda ranges: gathered_ranges.append(ranges),
        unregister_articulation_ranges=lambda ranges: released_ranges.append(ranges),
    )
    identity_map = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    data = SimpleNamespace(
        joint_ordering=None,
        has_joint_ordering=False,
        _sim_bind_joint_effort=wp.zeros((2, 3), dtype=wp.float32, device="cpu"),
        _sim_bind_joint_computed_effort=None,
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
                NewtonModel.AttributeFrequency.JOINT_DOF: SimpleNamespace(offset=0, slice=None, indices=None)
            }
        ),
        newton_actuator_adapter=None,
        _newton_native_ranges=None,
        _native_dof_masks=None,
        _native_dof_mask_owners=None,
        _native_dof_mask=None,
        _native_dof_mask_owner=None,
        _implicit_dof_mask=None,
        _implicit_dof_mask_owner=None,
        _joint_user_to_backend_map=lambda: identity_map,
    )
    binding = SimpleNamespace(
        groups={},
        layout=SimpleNamespace(group_layouts=()),
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        joint_command=SimpleNamespace(effort=SimpleNamespace(warp=object())),
        computed_effort=SimpleNamespace(warp=object()),
        applied_effort=SimpleNamespace(warp=object()),
    )
    pre_callbacks = []
    post_callbacks = []
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", adapter)
    monkeypatch.setattr(control_module.SimulationManager, "_pre_actuator_callbacks", pre_callbacks, raising=False)
    monkeypatch.setattr(control_module.SimulationManager, "_post_actuator_callbacks", post_callbacks)

    control = control_module.NewtonActuatorControl(articulation)
    control._prepare_native_actuator_binding(binding)

    assert gathered_ranges == []
    assert len(pre_callbacks) == 1
    pre_callbacks[0]()
    assert gathered_ranges == [first_ranges]

    control._unregister_pre_actuator_callback()
    control._unregister_post_actuator_callback()
    control._unregister_native_ranges()
    control._prepare_native_actuator_binding(binding)

    assert gathered_ranges == [first_ranges]
    assert released_ranges == [first_ranges]
    assert len(pre_callbacks) == 1
    pre_callbacks[0]()
    assert gathered_ranges == [first_ranges, second_ranges]

    control._unregister_pre_actuator_callback()
    control._unregister_post_actuator_callback()
    control._unregister_native_ranges()
    control._prepare_native_actuator_binding(binding)

    assert gathered_ranges == [first_ranges, second_ranges]
    assert pre_callbacks == []


def test_failed_native_prepare_restores_all_candidate_specific_state(monkeypatch) -> None:
    """A prepare failure restores fields without gathering into borrowed model arrays."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module
    from newton import Model as NewtonModel

    original = {
        name: object()
        for name in (
            "newton_actuator_adapter",
            "_newton_native_ranges",
            "_implicit_dof_mask",
            "_implicit_dof_mask_owner",
        )
    }
    original_computed_effort = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    identity_map = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    data = SimpleNamespace(
        joint_ordering=None,
        has_joint_ordering=False,
        _sim_bind_joint_effort=wp.zeros((2, 3), dtype=wp.float32, device="cpu"),
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
                NewtonModel.AttributeFrequency.JOINT_DOF: SimpleNamespace(
                    offset=7, slice=SimpleNamespace(start=1), indices=None
                )
            }
        ),
        _rollback_deferred_initialization=lambda: None,
        _joint_user_to_backend_map=lambda: identity_map,
        **original,
    )
    candidate_binding = SimpleNamespace(
        implicit_dof_mask=wp.zeros(3, dtype=wp.int32, device="cpu"),
        implicit_dof_mask_owner=torch.zeros(3, dtype=torch.int32),
        ranges=(SimpleNamespace(direct=False),),
    )
    bound_offsets = []

    def bind_articulation(*_args, **kwargs):
        bound_offsets.append(kwargs["dof_offset"])
        return candidate_binding

    borrowed_model_parameters: list[str] = []
    adapter = SimpleNamespace(
        bind_articulation=bind_articulation,
        gather_staged_ranges=lambda _ranges: borrowed_model_parameters.append("gathered"),
        unregister_articulation_ranges=lambda _ranges: None,
    )
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", adapter)
    pre_callbacks = []
    monkeypatch.setattr(control_module.SimulationManager, "_pre_actuator_callbacks", pre_callbacks, raising=False)

    def register_pre_callback(cls, callback) -> None:
        pre_callbacks.append(callback)

    monkeypatch.setattr(
        control_module.SimulationManager,
        "register_pre_actuator_callback",
        classmethod(register_pre_callback),
        raising=False,
    )
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
        layout=SimpleNamespace(group_layouts=()),
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        joint_command=SimpleNamespace(effort=SimpleNamespace(warp=object())),
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
    assert borrowed_model_parameters == []
    assert pre_callbacks == []
    assert callbacks == []
    assert bound_offsets == [8]


def test_native_prepare_rejects_indexed_frequency_layout_before_mutation(monkeypatch) -> None:
    """Indexed native placement fails contextually before candidate state changes."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module
    from newton import Model as NewtonModel

    sentinels = {
        "_native_dof_masks": object(),
        "_native_dof_mask_owners": object(),
        "_native_dof_mask": object(),
        "_native_dof_mask_owner": object(),
        "newton_actuator_adapter": object(),
        "_newton_native_ranges": object(),
    }
    data = SimpleNamespace(joint_ordering=None, _sim_bind_joint_computed_effort=object())
    articulation = SimpleNamespace(
        num_instances=2,
        num_joints=3,
        num_fixed_tendons=0,
        device="cpu",
        data=data,
        _data=data,
        _root_view=SimpleNamespace(
            frequency_layouts={
                NewtonModel.AttributeFrequency.JOINT_DOF: SimpleNamespace(
                    offset=4,
                    slice=None,
                    indices=wp.array([2, 1, 0], dtype=wp.int32, device="cpu"),
                )
            }
        ),
        **sentinels,
    )
    calls = []
    monkeypatch.setattr(
        control_module.SimulationManager,
        "_adapter",
        SimpleNamespace(bind_articulation=lambda *_args, **_kwargs: calls.append("bind")),
    )
    binding = SimpleNamespace(
        groups={},
        native_group_names=frozenset(),
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        computed_effort=SimpleNamespace(warp=object()),
        applied_effort=SimpleNamespace(warp=object()),
    )

    with pytest.raises(RuntimeError, match="Indexed Newton JOINT_DOF layouts are not supported"):
        control_module.NewtonActuatorControl(articulation)._prepare_native_actuator_binding(binding)

    assert calls == []
    for name, value in sentinels.items():
        assert getattr(articulation, name) is value


def test_failed_finalize_invalidation_clears_fallback_state_before_binding_release(monkeypatch) -> None:
    """A facade-bind failure removes callback/fallback pointers before shared binding release."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module

    callbacks = []
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", None)
    monkeypatch.setattr(control_module.SimulationManager, "_post_actuator_callbacks", callbacks)

    def build_mask(groups, num_joints, device, *, group_layouts=None):
        del group_layouts
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
        layout=SimpleNamespace(group_layouts=()),
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        computed_effort=SimpleNamespace(warp=object()),
        applied_effort=SimpleNamespace(warp=object()),
    )
    control.prepare_actuator_binding(binding)
    assert callbacks == []
    assert control._post_actuator_callback is None

    with pytest.raises(RuntimeError, match="facade bind failed"):
        control.bind_actuator_view(object())
    control.invalidate_actuator_view()

    assert callbacks == []
    assert articulation._implicit_dof_mask is None
    assert articulation._implicit_dof_mask_owner is None
    assert data._sim_bind_joint_computed_effort is None
    assert control._actuator_binding is None


def test_stop_ready_rebuild_uses_fresh_fallback_state_without_callback(monkeypatch) -> None:
    """STOP clears fallback allocations so READY installs fresh pointers without a no-op callback."""
    import isaaclab_newton.assets.articulation.actuator_control as control_module

    callbacks = []
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", None)
    monkeypatch.setattr(control_module.SimulationManager, "_post_actuator_callbacks", callbacks)

    def build_mask(groups, num_joints, device, *, group_layouts=None):
        del group_layouts
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
        layout=SimpleNamespace(group_layouts=()),
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
    assert first_callback is None
    assert control._post_actuator_callback is None
    assert callbacks == []


def test_failed_graph_revoke_pins_native_callback_owners_until_retry(monkeypatch) -> None:
    """Native callbacks retain captured raw owners after their facade fields are cleared."""
    import isaaclab_newton.actuators._graph as graph_module
    import isaaclab_newton.actuators.kernels as actuator_kernels
    import isaaclab_newton.assets.articulation.actuator_control as control_module
    from isaaclab_newton.actuators._graph import _CapturedGraphLease
    from newton import Model as NewtonModel

    class Owner:
        def __init__(self, **attributes) -> None:
            self.__dict__.update(attributes)

    class Adapter(Owner):
        def bind_articulation(self, *_args, **_kwargs):
            candidate = self.candidate
            self.candidate = None
            return candidate

        def gather_staged_ranges(self, _ranges) -> None:
            pass

        def publish_outputs(self, *_args) -> None:
            pass

        def unregister_articulation_ranges(self, _ranges) -> None:
            pass

    command_effort_raw = wp.zeros((1, 2), dtype=wp.float32, device="cpu")
    computed_effort_raw = wp.zeros((1, 2), dtype=wp.float32, device="cpu")
    applied_effort_raw = wp.zeros((1, 2), dtype=wp.float32, device="cpu")
    command_effort_proxy = Owner(warp=command_effort_raw)
    computed_effort_proxy = Owner(warp=computed_effort_raw)
    applied_effort_proxy = Owner(warp=applied_effort_raw)
    binding = Owner(
        groups={},
        native_group_names=frozenset(),
        layout=SimpleNamespace(group_layouts=()),
        command=SimpleNamespace(position=SimpleNamespace(warp=object()), velocity=SimpleNamespace(warp=object())),
        joint_command=SimpleNamespace(effort=command_effort_proxy),
        computed_effort=computed_effort_proxy,
        applied_effort=applied_effort_proxy,
    )

    range_route_owner = wp.zeros(2, dtype=wp.int32, device="cpu")
    range_binding = Owner(direct=False, route_owner=range_route_owner)
    native_binding = Owner(
        ranges=(range_binding,),
        implicit_dof_mask=wp.zeros(2, dtype=wp.int32, device="cpu"),
        implicit_dof_mask_owner=torch.zeros(2, dtype=torch.int32),
    )
    adapter = Adapter(candidate=native_binding)

    touched_mask_owner = torch.ones(2, dtype=torch.int32)
    touched_mask = wp.from_torch(touched_mask_owner, dtype=wp.int32)

    class MaskBuilder:
        def __init__(self, mask, owner) -> None:
            self.mask = mask
            self.owner = owner

        def __call__(self, *_args, **_kwargs):
            mask = self.mask
            owner = self.owner
            self.mask = None
            self.owner = None
            fields = ("position", "velocity", "effort", "computed_effort", "applied_effort", "touched")
            return ({field: mask for field in fields}, {field: owner for field in fields})

    mask_builder = MaskBuilder(touched_mask, touched_mask_owner)
    monkeypatch.setattr(actuator_kernels, "_build_native_dof_masks", mask_builder)

    user_to_backend = wp.array([1, 0], dtype=wp.int32, device="cpu")
    backend_to_user = wp.array([1, 0], dtype=wp.int32, device="cpu")
    joint_ordering = Owner(
        user_to_backend_indices=(1, 0),
        backend_to_user_indices=(1, 0),
        user_to_backend=user_to_backend,
        backend_to_user=backend_to_user,
    )
    backend_effort = wp.zeros((1, 2), dtype=wp.float32, device="cpu")
    data = Owner(
        joint_ordering=joint_ordering,
        has_joint_ordering=True,
        _sim_bind_joint_effort=backend_effort,
        _sim_bind_joint_computed_effort=None,
        _rollback_actuator_initialization=lambda: None,
    )
    articulation = Owner(
        num_instances=1,
        num_joints=2,
        num_fixed_tendons=0,
        device="cpu",
        data=data,
        _data=data,
        _root_view=SimpleNamespace(
            frequency_layouts={
                NewtonModel.AttributeFrequency.JOINT_DOF: SimpleNamespace(offset=0, slice=None, indices=None)
            }
        ),
        newton_actuator_adapter=None,
        _newton_native_ranges=None,
        _native_dof_masks=None,
        _native_dof_mask_owners=None,
        _native_dof_mask=None,
        _native_dof_mask_owner=None,
        _implicit_dof_mask=None,
        _implicit_dof_mask_owner=None,
        _rollback_deferred_initialization=lambda: None,
    )
    articulation._joint_user_to_backend_map = lambda: data.joint_ordering.user_to_backend

    monkeypatch.setattr(control_module.SimulationManager, "_graph", None)
    monkeypatch.setattr(control_module.SimulationManager, "_graph_capture_pending", False)
    monkeypatch.setattr(control_module.SimulationManager, "_adapter", adapter)
    monkeypatch.setattr(control_module.SimulationManager, "_pre_actuator_callbacks", [])
    monkeypatch.setattr(control_module.SimulationManager, "_post_actuator_callbacks", [])
    monkeypatch.setattr(control_module.SimulationManager, "_post_step_callbacks", [])

    control = control_module.NewtonActuatorControl(articulation)
    control._native_active = True
    control.prepare_actuator_binding(binding)
    backend_computed_effort = data._sim_bind_joint_computed_effort

    tracked_owners = (
        adapter,
        range_binding,
        range_route_owner,
        touched_mask_owner,
        touched_mask,
        joint_ordering,
        user_to_backend,
        command_effort_proxy,
        command_effort_raw,
        computed_effort_proxy,
        computed_effort_raw,
        applied_effort_proxy,
        applied_effort_raw,
        backend_effort,
        backend_computed_effort,
    )
    owner_refs = tuple(weakref.ref(owner) for owner in tracked_owners)

    graph = SimpleNamespace(
        device=SimpleNamespace(context=object(), context_guard=nullcontext()),
        graph=object(),
        graph_exec=object(),
    )
    synchronization_attempts = 0

    def synchronize(_device) -> None:
        nonlocal synchronization_attempts
        synchronization_attempts += 1
        if synchronization_attempts == 1:
            raise RuntimeError("injected native callback graph synchronization failure")

    monkeypatch.setattr(graph_module.wp, "synchronize_device", synchronize)
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=lambda *_args: True,
                wp_cuda_graph_exec_destroy=lambda *_args: True,
            )
        ),
    )
    lease = _CapturedGraphLease(
        graph,
        generation=control_module.SimulationManager._make_actuator_graph_generation(),
        label="native callback owner lifetime graph",
    )
    control_module.SimulationManager._graph = lease

    with pytest.raises(RuntimeError, match="injected native callback graph synchronization failure"):
        control.invalidate_actuator_graphs()

    control.invalidate_actuator_view()
    binding.command = None
    binding.joint_command = None
    binding.computed_effort = None
    binding.applied_effort = None
    data.joint_ordering = None
    data.has_joint_ordering = False
    data._sim_bind_joint_effort = None
    control_module.SimulationManager._adapter = None
    del tracked_owners
    del adapter, range_binding, range_route_owner, native_binding
    del touched_mask_owner, touched_mask, joint_ordering, user_to_backend, backend_to_user
    del command_effort_proxy, command_effort_raw, computed_effort_proxy, computed_effort_raw
    del applied_effort_proxy, applied_effort_raw, backend_effort, backend_computed_effort, binding

    gc.collect()
    assert control_module.SimulationManager._graph is lease
    assert lease.is_live is False
    assert all(owner_ref() is not None for owner_ref in owner_refs)

    control.invalidate_actuator_graphs()

    gc.collect()
    assert control_module.SimulationManager._graph is None
    assert all(owner_ref() is None for owner_ref in owner_refs)


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


@pytest.mark.parametrize(
    "stateful, decimation, expected",
    [
        (False, 1, True),
        (True, 1, False),
        (True, 2, True),
        (True, 3, False),
    ],
)
def test_native_actuator_graph_capture_requires_stable_state_parity(
    monkeypatch, stateful: bool, decimation: int, expected: bool
) -> None:
    """One replayable graph cannot advance a stateful actuator by an odd buffer parity."""
    monkeypatch.setattr(NewtonManager, "_adapter", SimpleNamespace(is_stateful=stateful), raising=False)
    monkeypatch.setattr(NewtonManager, "_decimation", decimation, raising=False)

    assert NewtonManager._supports_cuda_graph_capture() is expected


def test_stateful_relaxed_full_capture_skips_warmup_before_first_eager_step(monkeypatch, caplog) -> None:
    """Declining a stateful full capture leaves its first eager state advance untouched."""
    import isaaclab_newton.physics.newton_manager as manager_module

    events: list[str] = []

    class RejectingCudart:
        def cudaStreamCreateWithFlags(self, *_args) -> int:
            events.append("create_stream")
            return 1

    monkeypatch.setattr(manager_module, "_cudart", RejectingCudart())
    monkeypatch.setattr(NewtonManager, "_adapter", SimpleNamespace(is_stateful=True), raising=False)
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: True))
    monkeypatch.setattr(NewtonManager, "_simulate_full", classmethod(lambda cls: events.append("full_step")))
    monkeypatch.setattr(wp, "ScopedDevice", lambda _device: nullcontext())
    monkeypatch.setattr(wp, "get_stream", lambda _device: object())
    monkeypatch.setattr(wp, "synchronize_stream", lambda _stream: events.append("synchronize"))

    with caplog.at_level("WARNING", logger=manager_module.__name__):
        graph = NewtonManager._capture_relaxed_graph("cuda:0")

    assert graph is None
    assert events == []
    assert sum("stateful" in record.getMessage() for record in caplog.records) == 1

    NewtonManager._simulate_full()

    assert events == ["full_step"]


@pytest.mark.parametrize(
    ("stateful", "all_graphable", "use_capture_target", "expected_warmup"),
    [
        pytest.param(False, True, False, "full", id="stateless_full"),
        pytest.param(True, False, False, "physics", id="stateful_physics_only"),
        pytest.param(True, True, True, "sensor", id="stateful_sensor_target"),
    ],
)
def test_relaxed_capture_preserves_safe_warmup_paths(
    monkeypatch,
    stateful: bool,
    all_graphable: bool,
    use_capture_target: bool,
    expected_warmup: str,
) -> None:
    """Stateless full, mixed physics-only, and sensor captures still warm once."""
    import isaaclab_newton.physics.newton_manager as manager_module

    events: list[str] = []

    class RejectingCudart:
        def cudaStreamCreateWithFlags(self, *_args) -> int:
            events.append("create_stream")
            return 1

    monkeypatch.setattr(manager_module, "_cudart", RejectingCudart())
    monkeypatch.setattr(NewtonManager, "_adapter", SimpleNamespace(is_stateful=stateful), raising=False)
    monkeypatch.setattr(NewtonManager, "_is_all_graphable", classmethod(lambda cls: all_graphable))
    monkeypatch.setattr(NewtonManager, "_simulate_full", classmethod(lambda cls: events.append("full")))
    monkeypatch.setattr(NewtonManager, "_simulate_physics_only", classmethod(lambda cls: events.append("physics")))
    monkeypatch.setattr(wp, "ScopedDevice", lambda _device: nullcontext())
    monkeypatch.setattr(wp, "get_stream", lambda _device: object())
    monkeypatch.setattr(wp, "synchronize_stream", lambda _stream: events.append("synchronize"))
    capture_target = (lambda: events.append("sensor")) if use_capture_target else None

    graph = NewtonManager._capture_relaxed_graph("cuda:0", capture_target=capture_target)

    assert graph is None
    assert events == [expected_warmup, "synchronize", "create_stream"]


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
    revocations = []
    lease = SimpleNamespace(revoke=lambda: revocations.append("revoke"), is_live=True)
    monkeypatch.setattr(NewtonManager, "_graph", lease, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)

    NewtonMPMManager._capture_or_defer_graph()

    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is False
    assert revocations == ["revoke"]


def test_manager_stop_revokes_retained_graph_lease(monkeypatch) -> None:
    """A retained manager lease rejects replay after the STOP invalidation seam."""

    class RetainedLease:
        def __init__(self) -> None:
            self.is_live = True
            self.revoke_count = 0

        def launch(self) -> None:
            if not self.is_live:
                raise RuntimeError("manager actuator graph was revoked")

        def revoke(self) -> None:
            if not self.is_live:
                return
            self.is_live = False
            self.revoke_count += 1

    lease = RetainedLease()
    monkeypatch.setattr(NewtonManager, "_graph", lease, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)

    NewtonManager.invalidate_actuator_graphs()
    NewtonManager.invalidate_actuator_graphs()

    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is False
    assert lease.revoke_count == 1
    with pytest.raises(RuntimeError, match="manager actuator graph was revoked"):
        lease.launch()


def test_manager_clear_retains_captured_owners_until_failed_graph_cleanup_retries(monkeypatch) -> None:
    """A failed graph cleanup pins every captured manager owner until a later successful retry."""
    import isaaclab_newton.actuators._graph as graph_module
    from isaaclab_newton.actuators._graph import _CapturedGraphLease

    class Owner:
        pass

    owners = {
        "_model": Owner(),
        "_adapter": Owner(),
        "_solver": Owner(),
        "_state_0": Owner(),
        "_state_1": Owner(),
        "_control": Owner(),
        "_contacts": Owner(),
        "_collision_pipeline": Owner(),
        "_world_reset_mask": Owner(),
        "_fk_reset_mask": Owner(),
        "_pre_actuator_callbacks": [Owner()],
        "_post_actuator_callbacks": [Owner()],
        "_post_step_callbacks": [Owner()],
        "_newton_contact_sensors": {"sensor": Owner()},
        "_newton_frame_transform_sensors": [Owner()],
        "_newton_imu_sensors": [Owner()],
    }
    owner_refs = [
        weakref.ref(owner)
        for value in owners.values()
        for owner in (value if isinstance(value, list) else value.values() if isinstance(value, dict) else (value,))
    ]
    for name, owner in owners.items():
        monkeypatch.setattr(NewtonManager, name, owner, raising=False)
    del owner

    graph = SimpleNamespace(
        device=SimpleNamespace(context=object(), context_guard=nullcontext()),
        graph=object(),
        graph_exec=object(),
    )
    attempts = 0

    def synchronize(_device) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected graph synchronization failure")

    monkeypatch.setattr(graph_module.wp, "synchronize_device", synchronize)
    monkeypatch.setattr(
        graph_module,
        "runtime",
        SimpleNamespace(
            core=SimpleNamespace(
                wp_cuda_graph_destroy=lambda *_args: True,
                wp_cuda_graph_exec_destroy=lambda *_args: True,
            )
        ),
    )
    lease = _CapturedGraphLease(
        graph,
        generation=NewtonManager._make_actuator_graph_generation(),
        label="manager owner lifetime graph",
    )
    monkeypatch.setattr(NewtonManager, "_graph", lease, raising=False)
    pre_callback = owners["_pre_actuator_callbacks"][0]
    NewtonManager.unregister_pre_actuator_callback(pre_callback)
    assert NewtonManager._pre_actuator_callbacks == []
    del pre_callback
    del owners

    with pytest.raises(RuntimeError, match="injected graph synchronization failure"):
        NewtonManager.clear()

    gc.collect()
    assert NewtonManager._graph is lease
    assert lease.is_live is False
    assert all(owner_ref() is not None for owner_ref in owner_refs)
    with pytest.raises(RuntimeError, match="manager owner lifetime graph.*revoked"):
        lease.launch()

    NewtonManager.clear()

    gc.collect()
    assert NewtonManager._graph is None
    assert all(owner_ref() is None for owner_ref in owner_refs)


def test_graph_capture_without_active_configuration_revokes_old_lease(monkeypatch) -> None:
    """Losing the active configuration still revokes a previously captured graph."""
    from isaaclab.physics import PhysicsManager

    revocations = []
    lease = SimpleNamespace(revoke=lambda: revocations.append("revoke"), is_live=True)
    monkeypatch.setattr(NewtonManager, "_graph", lease, raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", None, raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", None, raising=False)

    NewtonManager._capture_or_defer_graph()

    assert NewtonManager._graph is None
    assert NewtonManager._graph_capture_pending is False
    assert revocations == ["revoke"]


def test_cuda_graph_capture_uses_simulation_device(monkeypatch):
    """CUDA graph capture should use the simulation device instead of Warp's default device."""
    from isaaclab.physics import PhysicsManager

    captured_devices = []
    captured_graph = object()
    wrapped = []
    lease = SimpleNamespace(is_live=True, launch=lambda: None, revoke=lambda: None)

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
    monkeypatch.setattr(
        "isaaclab_newton.physics.newton_manager._CapturedGraphLease",
        lambda graph, **kwargs: wrapped.append((graph, kwargs)) or lease,
    )

    NewtonManager._capture_or_defer_graph()

    assert captured_devices == ["cuda:1"]
    assert wrapped[0][0] is captured_graph
    assert wrapped[0][1]["label"] == "Newton manager actuator graph"
    assert NewtonManager._graph is lease


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
