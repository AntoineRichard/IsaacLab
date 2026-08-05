# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-Newton value-flow regressions for globally aggregated actuator ranges."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray


class _NativeType:
    """Private exact type used to build one canonical actuator layout."""


def _make_three_group_binding(arrays: dict[str, ProxyArray]) -> SimpleNamespace:
    """Return one two-world canonical layout with structural order ``[A, B, A]``."""
    groups = (
        SimpleNamespace(
            name="a_first",
            actuator_type=_NativeType,
            joint_indices=(0,),
            joint_names=("joint_0",),
            type_slice=slice(0, 1),
        ),
        SimpleNamespace(
            name="b",
            actuator_type=_NativeType,
            joint_indices=(1,),
            joint_names=("joint_1",),
            type_slice=slice(1, 2),
        ),
        SimpleNamespace(
            name="a_last",
            actuator_type=_NativeType,
            joint_indices=(2,),
            joint_names=("joint_2",),
            type_slice=slice(2, 3),
        ),
    )
    return SimpleNamespace(
        groups={group.name: SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays)) for group in groups},
        native_group_names=frozenset(group.name for group in groups),
        computed_effort=arrays["computed_effort"],
        layout=SimpleNamespace(
            num_joints=3,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=2, num_dofs=3, compact_joint_indices=(0, 1, 2))},
            group_layouts=groups,
        ),
    )


def _make_real_two_world_model() -> tuple[object, tuple[tuple, ...]]:
    """Build real Newton entries whose physical controller order is ``[A, B, A]``."""
    import newton
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(6)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation(joints)

    # The numeric gains differ per DOF but Newton merges all A DOFs. B is
    # deliberately stateful (one-step delay), which must remain separate.
    for index, stiffness in zip((0, 2, 3, 5), (2.0, 4.0, 5.0, 7.0), strict=True):
        builder.add_actuator(ControllerPD, index=index, kp=stiffness, kd=0.0)
    for index, stiffness in zip((1, 4), (11.0, 14.0), strict=True):
        builder.add_actuator(ControllerPD, index=index, kp=stiffness, kd=0.0, delay_steps=1)

    entries = tuple(builder.actuator_entries)
    return builder.finalize(device="cpu"), entries


def test_real_newton_a_b_a_staging_preserves_world_parameters_and_outputs() -> None:
    """Globally merged A columns gather/scatter without leaking delayed B values."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ControllerPD

    model, keys = _make_real_two_world_model()
    state = model.state()
    control = model.control()
    assert len(model.actuators) == 2
    assert keys[0][0] is ControllerPD
    assert keys[0][1] is False
    assert keys[1][0] is ControllerPD
    assert keys[1][1] is True
    assert model.actuators[0].indices.numpy().tolist() == [0, 2, 3, 5]
    assert model.actuators[1].indices.numpy().tolist() == [1, 4]
    assert model.actuators[0].controller.kp.numpy().tolist() == [2.0, 4.0, 5.0, 7.0]
    assert model.actuators[1].controller.kp.numpy().tolist() == [11.0, 14.0]

    arrays = {
        name: ProxyArray(wp.zeros((2, 3), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    # World-major, canonical [A, B, A]. Every value is intentionally unique.
    stiffness = torch.tensor([[2.0, 11.0, 4.0], [5.0, 14.0, 7.0]])
    wp.to_torch(arrays["stiffness"].warp).copy_(stiffness)
    binding = _make_three_group_binding(arrays)
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=2,
        num_joints=3,
        dof_offset=0,
        device="cpu",
        actuator_keys=keys,
        actuator_dof_indices={keys[0]: (0, 2), keys[1]: (1,)},
    )
    adapter._joint_signatures = {"joint_0": keys[0], "joint_1": keys[1], "joint_2": keys[0]}
    native_binding = adapter.bind_articulation(binding, dof_offset=0)

    # A is one persistent staged controller spanning disjoint canonical slots;
    # B remains a separately staged delayed controller.
    assert len(native_binding.ranges) == 2
    assert all(not range_binding.direct for range_binding in native_binding.ranges)
    assert native_binding.ranges[0].canonical_slots.numpy().tolist() == [0, 2]
    assert native_binding.ranges[1].canonical_slots.numpy().tolist() == [1]

    joint_computed = wp.full((2, 3), 999.0, dtype=wp.float32, device="cpu")
    joint_applied = wp.full((2, 3), 999.0, dtype=wp.float32, device="cpu")
    target_cycles = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
    )
    # These hand-derived A values expose an A-column swap, an env swap, and
    # any contamination by B. B is checked against its own delayed history.
    expected_a = (
        torch.tensor([[2.0, 12.0], [20.0, 42.0]]),
        torch.tensor([[14.0, 36.0], [50.0, 84.0]]),
    )
    # A delayed controller uses its first target again on cycle two. These
    # values prove B's state did not leak into either disjoint A segment.
    expected_b = (torch.tensor([22.0, 70.0]), torch.tensor([22.0, 70.0]))

    for target, expected_a_cycle, expected_b_cycle in zip(target_cycles, expected_a, expected_b, strict=True):
        arrays["computed_effort"].warp.fill_(999.0)
        arrays["applied_effort"].warp.fill_(999.0)
        joint_computed.fill_(999.0)
        joint_applied.fill_(999.0)
        wp.to_torch(control.joint_target_q)[:6].copy_(target.reshape(-1))
        wp.to_torch(control.joint_target_qd).zero_()
        wp.to_torch(control.joint_act).zero_()

        adapter.gather_staged_ranges(native_binding.ranges)
        adapter.step(state, control, dt=0.01)
        adapter.publish_outputs(native_binding.ranges, joint_computed, joint_applied)

        computed = wp.to_torch(arrays["computed_effort"].warp).clone()
        applied = wp.to_torch(arrays["applied_effort"].warp).clone()
        physical = wp.to_torch(control.joint_f)[:6].reshape(2, 3).clone()
        assert torch.equal(computed[:, (0, 2)], expected_a_cycle)
        assert torch.equal(computed[:, 1], expected_b_cycle)
        assert torch.equal(applied, computed)
        assert torch.equal(wp.to_torch(joint_computed), computed)
        assert torch.equal(wp.to_torch(joint_applied), computed)
        assert torch.equal(physical, computed)
    # The delayed B column has a distinct state history, while every A output
    # stays tied to its own world and its own canonical [0, 2] slot.


def test_real_newton_same_dof_a_b_a_uses_binding_local_authored_occurrences() -> None:
    """A grouped builder must not replace one joint's authored ``[A, B, A]`` stream."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from isaaclab_newton.physics.newton_manager import _build_env_zero_actuator_metadata
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.0)
    builder.add_actuator(ControllerPD, index=0, kp=11.0, kd=0.0, delay_steps=1)
    builder.add_actuator(ControllerPD, index=0, kp=7.0, kd=0.0)
    keys = tuple(builder.actuator_entries)
    model = builder.finalize(device="cpu")
    control = model.control()
    state, control = model.state(), model.control()

    key_a, key_b = keys
    assert key_a[0] is ControllerPD and key_a[1] is False
    assert key_b[0] is ControllerPD and key_b[1] is True
    assert model.actuators[0].indices.numpy().tolist() == [0, 0]
    assert model.actuators[1].indices.numpy().tolist() == [0]
    grouped_signatures, grouped_indices = _build_env_zero_actuator_metadata(
        tuple(builder.actuator_entries.items()), dofs_per_env=1
    )
    assert grouped_signatures == {0: (key_a, key_a, key_b)}
    assert grouped_indices == {key_a: (0, 0), key_b: (0,)}

    arrays = {
        name: ProxyArray(wp.zeros((1, 3), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    arrays["stiffness"].torch.copy_(torch.tensor([[2.0, 11.0, 7.0]]))
    groups = tuple(
        SimpleNamespace(
            name=name,
            actuator_type=_NativeType,
            joint_indices=(0,),
            joint_names=("joint",),
            type_slice=slice(index, index + 1),
        )
        for index, name in enumerate(("a_first", "b", "a_last"))
    )
    binding = SimpleNamespace(
        groups={group.name: SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays)) for group in groups},
        native_group_names=frozenset(group.name for group in groups),
        computed_effort=arrays["computed_effort"],
        layout=SimpleNamespace(
            num_joints=1,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=1, num_dofs=3, compact_joint_indices=(0, 0, 0))},
            group_layouts=groups,
        ),
    )
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=1,
        dof_offset=0,
        device="cpu",
        actuator_keys=keys,
        dof_signatures=grouped_signatures,
        actuator_dof_indices=grouped_indices,
    )

    native = adapter.bind_articulation(
        binding,
        dof_offset=0,
        structural_occurrences={0: (key_a, key_b, key_a)},
    )

    assert [range_binding.actuator for range_binding in native.ranges] == [model.actuators[0], model.actuators[1]]
    assert native.ranges[0].controller_local_slots.numpy().tolist() == [0, 1]
    assert native.ranges[1].controller_local_slots.numpy().tolist() == [0]

    joint_computed = wp.full((1, 1), -1.0, dtype=wp.float32, device="cpu")
    joint_applied = wp.full((1, 1), -1.0, dtype=wp.float32, device="cpu")
    for target, expected in ((1.0, torch.tensor([[2.0, 11.0, 7.0]])), (3.0, torch.tensor([[6.0, 11.0, 21.0]]))):
        control.joint_target_q.fill_(target)
        control.joint_target_qd.zero_()
        control.joint_act.zero_()
        adapter.gather_staged_ranges(native.ranges)
        adapter.step(state, control, dt=0.01)
        adapter.publish_outputs(
            native.ranges,
            joint_computed,
            joint_applied,
            backend_effort=wp.from_torch(wp.to_torch(control.joint_f)[:1].reshape(1, 1), dtype=wp.float32),
            user_to_backend=wp.array([0], dtype=wp.int32, device="cpu"),
        )

        torch.testing.assert_close(arrays["computed_effort"].torch, expected)
        torch.testing.assert_close(arrays["applied_effort"].torch, expected)
        torch.testing.assert_close(wp.to_torch(joint_computed), expected[:, 2:])
        torch.testing.assert_close(wp.to_torch(joint_applied), expected[:, 2:])
        torch.testing.assert_close(wp.to_torch(control.joint_f)[:1].reshape(1, 1), expected[:, 2:])


def test_native_bindings_with_shared_joint_names_keep_distinct_authored_streams() -> None:
    """Two articulations named ``joint`` bind their own pre-grouping occurrence stream."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from isaaclab_newton.physics.newton_manager import _build_env_zero_actuator_metadata
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(2)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation([joints[0]])
    builder.add_articulation([joints[1]])
    for index, delay_steps in ((0, 0), (0, 1), (0, 0), (1, 1), (1, 0), (1, 1)):
        delay_kwargs = {"delay_steps": delay_steps} if delay_steps else {}
        builder.add_actuator(ControllerPD, index=index, kp=2.0, kd=0.0, **delay_kwargs)
    keys = tuple(builder.actuator_entries)
    model = builder.finalize(device="cpu")
    key_a, key_b = keys
    grouped_signatures, grouped_indices = _build_env_zero_actuator_metadata(
        tuple(builder.actuator_entries.items()), dofs_per_env=2
    )

    def binding(name: str) -> SimpleNamespace:
        arrays = {
            field: ProxyArray(wp.zeros((1, 3), dtype=wp.float32, device="cpu"))
            for field in ("stiffness", "damping", "computed_effort", "applied_effort")
        }
        groups = tuple(
            SimpleNamespace(
                name=f"{name}_{suffix}",
                actuator_type=_NativeType,
                joint_indices=(0,),
                joint_names=("joint",),
                type_slice=slice(slot, slot + 1),
            )
            for slot, suffix in enumerate(("first", "middle", "last"))
        )
        return SimpleNamespace(
            groups={group.name: SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays)) for group in groups},
            native_group_names=frozenset(group.name for group in groups),
            computed_effort=arrays["computed_effort"],
            layout=SimpleNamespace(
                num_joints=1,
                type_layouts={_NativeType: SimpleNamespace(num_worlds=1, num_dofs=3, compact_joint_indices=(0, 0, 0))},
                group_layouts=groups,
            ),
        )

    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=2,
        dof_offset=0,
        device="cpu",
        actuator_keys=keys,
        dof_signatures=grouped_signatures,
        actuator_dof_indices=grouped_indices,
    )
    first = adapter.bind_articulation(binding("first"), dof_offset=0, structural_occurrences={0: (key_a, key_b, key_a)})
    second = adapter.bind_articulation(
        binding("second"), dof_offset=1, structural_occurrences={0: (key_b, key_a, key_b)}
    )

    assert first.ranges[0].controller_local_slots.numpy().tolist() == [0, 1]
    assert first.ranges[1].controller_local_slots.numpy().tolist() == [0]
    assert second.ranges[0].controller_local_slots.numpy().tolist() == [1, 2]
    assert second.ranges[1].controller_local_slots.numpy().tolist() == [2]


def test_real_newton_actuator_clears_effort_indices_each_step() -> None:
    """A real actuator does not accumulate output when effort and input slots differ."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import Actuator, ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(2)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation(joints)
    model = builder.finalize(device="cpu")
    state, control = model.state(), model.control()
    actuator = Actuator(
        indices=wp.array([0], dtype=wp.uint32, device="cpu"),
        effort_indices=wp.array([1], dtype=wp.uint32, device="cpu"),
        controller=ControllerPD(
            kp=wp.array([2.0], dtype=wp.float32, device="cpu"),
            kd=wp.array([0.0], dtype=wp.float32, device="cpu"),
        ),
        control_target_pos_attr="joint_target_q",
        control_target_vel_attr="joint_target_qd",
    )
    adapter = object.__new__(NewtonActuatorAdapter)
    adapter.actuators = [actuator]
    adapter._states_a = [actuator.state()]
    adapter._states_b = [actuator.state()]
    adapter._device = "cpu"

    control.joint_target_q.fill_(1.0)
    control.joint_target_qd.zero_()
    control.joint_act.zero_()
    control.joint_f.zero_()
    for _ in range(2):
        adapter.step(state, control, dt=0.01)
        torch.testing.assert_close(wp.to_torch(control.joint_f)[:2], torch.tensor([0.0, 2.0]))


def test_native_binding_rejects_programmatic_actuator_on_implicit_lab_dof_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prebuilt poison actuator cannot coexist with an implicit Lab-covered DOF."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(2)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation(joints)
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.0)
    builder.add_actuator(ControllerPD, index=1, kp=100.0, kd=0.0, delay_steps=1)
    keys = tuple(builder.actuator_entries)
    model = builder.finalize(device="cpu")
    control = model.control()
    native_key, poison_key = keys
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
            num_joints=2,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=1, num_dofs=1, compact_joint_indices=(0,))},
            group_layouts=(
                SimpleNamespace(
                    name="native",
                    actuator_type=_NativeType,
                    joint_indices=(0,),
                    joint_names=("native_joint",),
                    type_slice=slice(0, 1),
                ),
                SimpleNamespace(
                    name="implicit",
                    actuator_type=object,
                    joint_indices=(1,),
                    joint_names=("implicit_joint",),
                    type_slice=slice(0, 1),
                ),
            ),
        ),
    )
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=2,
        dof_offset=0,
        device="cpu",
        actuator_keys=keys,
        dof_signatures={0: (native_key,), 1: (poison_key,)},
        actuator_dof_indices={native_key: (0,), poison_key: (1,)},
    )
    original_kp = model.actuators[0].controller.kp
    control.joint_f[:2].fill_(17.0)
    monkeypatch.setattr(
        adapter,
        "_register_staged_actuator",
        lambda *_args: pytest.fail("uncovered actuator registered staging"),
    )
    with pytest.raises(RuntimeError, match="occurrence.*implicit_joint"):
        adapter.bind_articulation(
            binding,
            dof_offset=0,
            structural_occurrences={0: (native_key,), 1: ()},
        )

    assert model.actuators[0].controller.kp is original_kp
    assert adapter._direct_pointer_bindings == {}
    assert adapter._global_native_bindings == {}
    torch.testing.assert_close(wp.to_torch(control.joint_f)[:2], torch.full((2,), 17.0))


def test_native_binding_allows_programmatic_actuator_on_uncovered_dof() -> None:
    """A Newton-only DOF remains valid when no Lab group claims that joint."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(2)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation(joints)
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.0)
    builder.add_actuator(ControllerPD, index=1, kp=100.0, kd=0.0, delay_steps=1)
    keys = tuple(builder.actuator_entries)
    model = builder.finalize(device="cpu")
    native_key, uncovered_key = keys
    arrays = {
        name: ProxyArray(wp.zeros((1, 1), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
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
                    joint_names=("native_joint",),
                    type_slice=slice(0, 1),
                ),
            ),
        ),
    )
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=2,
        dof_offset=0,
        device="cpu",
        actuator_keys=keys,
        dof_signatures={0: (native_key,), 1: (uncovered_key,)},
        actuator_dof_indices={native_key: (0,), uncovered_key: (1,)},
    )

    native = adapter.bind_articulation(binding, dof_offset=0, structural_occurrences={0: (native_key,)})

    assert len(native.ranges) == 1


def test_global_adapter_accepts_zero_occurrences_for_second_lab_articulation() -> None:
    """A zero-actuator articulation can share a global adapter with a native one."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(2)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation(joints)
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.0)
    (key,) = tuple(builder.actuator_entries)
    model = builder.finalize(device="cpu")
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=2,
        dof_offset=0,
        device="cpu",
        actuator_keys=(key,),
        dof_signatures={0: (key,)},
        actuator_dof_indices={key: (0,)},
    )
    implicit_only_binding = SimpleNamespace(
        native_group_names=frozenset(),
        layout=SimpleNamespace(
            group_layouts=(
                SimpleNamespace(
                    name="implicit",
                    joint_indices=(0,),
                    joint_names=("joint",),
                ),
            )
        ),
    )

    assignments = adapter._plan_structural_occurrences(
        implicit_only_binding,
        dof_offset=1,
        joint_user_to_backend_indices=None,
        structural_occurrences={0: ()},
    )

    assert assignments == {}


def test_native_binding_rejects_nonidentity_effort_indices_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coupled effort routing is rejected before a native pointer can be rebound."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.0)
    model = builder.finalize(device="cpu")
    (key,) = builder.actuator_entries
    actuator = model.actuators[0]
    actuator.effort_indices = wp.array([0], dtype=wp.uint32, device="cpu")
    arrays = {
        name: ProxyArray(wp.zeros((1, 1), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
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
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=1,
        dof_offset=0,
        device="cpu",
        actuator_keys=(key,),
        dof_signatures={0: (key,)},
        actuator_dof_indices={key: (0,)},
    )
    original_kp = actuator.controller.kp
    monkeypatch.setattr(
        adapter,
        "_register_staged_actuator",
        lambda *_args: pytest.fail("coupled effort routing registered staging"),
    )

    with pytest.raises(NotImplementedError, match="effort_indices"):
        adapter.bind_articulation(binding, dof_offset=0, structural_occurrences={0: (key,)})

    assert actuator.controller.kp is original_kp
    assert adapter._direct_pointer_bindings == {}
    assert adapter._global_native_bindings == {}


def test_real_newton_duplicate_dof_occurrences_keep_distinct_controller_slots() -> None:
    """Bind duplicate authored Newton DOFs to their ordered controller occurrences."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=0.0)
    builder.add_actuator(ControllerPD, index=0, kp=7.0, kd=0.0)
    model = builder.finalize(device="cpu")
    (signature,) = builder.actuator_entries
    assert model.actuators[0].indices.numpy().tolist() == [0, 0]
    assert model.actuators[0].controller.kp.numpy().tolist() == [2.0, 7.0]

    arrays = {
        name: ProxyArray(wp.zeros((1, 2), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    binding = SimpleNamespace(
        groups={
            name: SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays)) for name in ("first", "second")
        },
        native_group_names=frozenset(("first", "second")),
        computed_effort=arrays["computed_effort"],
        layout=SimpleNamespace(
            num_joints=1,
            type_layouts={_NativeType: SimpleNamespace(num_worlds=1, num_dofs=2, compact_joint_indices=(0, 0))},
            group_layouts=(
                SimpleNamespace(
                    name="first",
                    actuator_type=_NativeType,
                    joint_indices=(0,),
                    joint_names=("joint",),
                    type_slice=slice(0, 1),
                ),
                SimpleNamespace(
                    name="second",
                    actuator_type=_NativeType,
                    joint_indices=(0,),
                    joint_names=("joint",),
                    type_slice=slice(1, 2),
                ),
            ),
        ),
    )
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=1,
        dof_offset=0,
        device="cpu",
        actuator_keys=(signature,),
        actuator_dof_indices={signature: (0, 0)},
    )
    adapter._joint_signatures = {"joint": (signature, signature)}

    native = adapter.bind_articulation(binding, dof_offset=0)

    assert len(native.ranges) == 1
    assert native.ranges[0].direct is False
    assert native.ranges[0].controller_local_slots.numpy().tolist() == [0, 1]
    arrays["stiffness"].torch.copy_(torch.tensor([[2.0, 7.0]]))
    adapter.gather_staged_ranges(native.ranges)
    control = model.control()
    state = model.state()
    wp.to_torch(control.joint_target_q)[0] = 3.0
    wp.to_torch(control.joint_target_qd).zero_()
    wp.to_torch(control.joint_act).zero_()
    joint_computed = wp.full((1, 1), -1.0, dtype=wp.float32, device="cpu")
    joint_applied = wp.full((1, 1), -1.0, dtype=wp.float32, device="cpu")
    physical_effort = wp.from_torch(wp.to_torch(control.joint_f)[:1].reshape(1, 1), dtype=wp.float32)
    adapter.step(state, control, dt=0.01)
    adapter.publish_outputs(
        native.ranges,
        joint_computed,
        joint_applied,
        backend_effort=physical_effort,
        user_to_backend=wp.array([0], dtype=wp.int32, device="cpu"),
    )

    torch.testing.assert_close(arrays["computed_effort"].torch, torch.tensor([[6.0, 21.0]]))
    torch.testing.assert_close(wp.to_torch(joint_computed), torch.tensor([[21.0]]))
    torch.testing.assert_close(wp.to_torch(physical_effort), torch.tensor([[21.0]]))


def _duplicate_occurrence_binding(count: int) -> SimpleNamespace:
    """Build canonical duplicate-DOF groups for cardinality transaction tests."""
    arrays = {
        name: ProxyArray(wp.zeros((1, count), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    groups = tuple(
        SimpleNamespace(
            name=f"native_{index}",
            actuator_type=_NativeType,
            joint_indices=(0,),
            joint_names=("joint",),
            type_slice=slice(index, index + 1),
        )
        for index in range(count)
    )
    return SimpleNamespace(
        groups={group.name: SimpleNamespace(_parameter_binding=SimpleNamespace(arrays=arrays)) for group in groups},
        native_group_names=frozenset(group.name for group in groups),
        computed_effort=arrays["computed_effort"],
        layout=SimpleNamespace(
            num_joints=1,
            type_layouts={
                _NativeType: SimpleNamespace(num_worlds=1, num_dofs=count, compact_joint_indices=(0,) * count)
            },
            group_layouts=groups,
        ),
    )


@pytest.mark.parametrize(("canonical_count", "signature_count"), ((2, 1), (1, 2)))
def test_duplicate_occurrence_cardinality_rejects_before_binding_mutation(
    monkeypatch: pytest.MonkeyPatch, canonical_count: int, signature_count: int
) -> None:
    """Insufficient and extra occurrences fail before aliases or staging registrations."""
    import newton
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    for stiffness in (2.0, 7.0):
        builder.add_actuator(ControllerPD, index=0, kp=stiffness, kd=0.0)
    model = builder.finalize(device="cpu")
    (signature,) = builder.actuator_entries
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=1,
        dof_offset=0,
        device="cpu",
        actuator_keys=(signature,),
        dof_signatures={0: (signature,) * signature_count},
        actuator_dof_indices={signature: (0, 0)},
    )
    original_kp = model.actuators[0].controller.kp
    monkeypatch.setattr(
        adapter,
        "_register_staged_actuator",
        lambda *_args: pytest.fail("cardinality validation registered staging"),
    )

    with pytest.raises(RuntimeError, match="occurrence cardinality"):
        adapter.bind_articulation(_duplicate_occurrence_binding(canonical_count), dof_offset=0)

    assert model.actuators[0].controller.kp is original_kp
    assert adapter._direct_pointer_bindings == {}
    assert adapter._global_native_bindings == {}


def _make_real_native_four_joint_model() -> tuple[object, object, object, tuple]:
    """Build a real four-DOF Newton PD actuator for control-path ownership tests."""
    import newton
    from newton.actuators import ControllerPD

    builder = newton.ModelBuilder()
    bodies = [builder.add_body(mass=1.0) for _ in range(4)]
    joints = [builder.add_joint_revolute(parent=-1, child=body) for body in bodies]
    builder.add_articulation(joints)
    for index in range(4):
        builder.add_actuator(ControllerPD, index=index, kp=10.0, kd=0.0)
    model = builder.finalize(device="cpu")
    (key,) = builder.actuator_entries
    return model, model.state(), model.control(), key


def _command_binding(*, num_joints: int, fill: float = 0.0) -> SimpleNamespace:
    """Create one pointer-stable three-field command domain."""
    return SimpleNamespace(
        position=ProxyArray(wp.full((1, num_joints), fill, dtype=wp.float32, device="cpu")),
        velocity=ProxyArray(wp.full((1, num_joints), fill, dtype=wp.float32, device="cpu")),
        effort=ProxyArray(wp.full((1, num_joints), fill, dtype=wp.float32, device="cpu")),
    )


def _run_native_lab_native_control_cycle(
    monkeypatch: pytest.MonkeyPatch, order: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Run two real cycles through the Lab executor, Newton control, and callback."""
    from isaaclab_newton.actuators.adapter import NewtonActuatorAdapter
    from isaaclab_newton.assets.articulation.actuator_control import NewtonActuatorControl
    from isaaclab_newton.physics import NewtonManager
    from newton import Model as NewtonModel

    from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg, IdealPDActuator, IdealPDActuatorCfg
    from isaaclab.actuators.actuator_collection import _SelectorState
    from isaaclab.actuators.actuator_execution import _ArticulationExecutionPlan

    model, state, newton_control, key = _make_real_native_four_joint_model()
    native_actuator = model.actuators[0]
    canonical = {
        name: ProxyArray(wp.zeros((1, 4), dtype=wp.float32, device="cpu"))
        for name in ("stiffness", "damping", "computed_effort", "applied_effort")
    }
    canonical["stiffness"].warp.fill_(10.0)
    native_groups = {
        "native_first": SimpleNamespace(
            joint_indices=torch.tensor([0, 1]), _parameter_binding=SimpleNamespace(arrays=canonical)
        ),
        "native_last": SimpleNamespace(
            joint_indices=torch.tensor([2, 3]), _parameter_binding=SimpleNamespace(arrays=canonical)
        ),
    }
    lab = DelayedPDActuator(
        DelayedPDActuatorCfg(
            joint_names_expr=["joint_1", "joint_2"],
            stiffness=0.0,
            damping=0.0,
            effort_limit=10_000.0,
            velocity_limit=10_000.0,
            min_delay=1,
            max_delay=1,
        ),
        joint_names=["joint_1", "joint_2"],
        joint_ids=torch.tensor([1, 2]),
        num_envs=1,
        device="cpu",
    )
    groups = {"native_first": native_groups["native_first"], "lab": lab, "native_last": native_groups["native_last"]}
    groups = {name: groups[name] for name in order}
    native_type_layout = SimpleNamespace(num_worlds=1, num_dofs=4, compact_joint_indices=(0, 1, 2, 3))
    layouts = {
        "native_first": SimpleNamespace(
            name="native_first",
            actuator_type=IdealPDActuator,
            joint_indices=(0, 1),
            joint_names=("joint_0", "joint_1"),
            type_slice=slice(0, 2),
            num_worlds=1,
            num_dofs=2,
        ),
        "lab": SimpleNamespace(
            name="lab",
            actuator_type=DelayedPDActuator,
            joint_indices=(1, 2),
            joint_names=("joint_1", "joint_2"),
            type_slice=slice(0, 2),
            num_worlds=1,
            num_dofs=2,
        ),
        "native_last": SimpleNamespace(
            name="native_last",
            actuator_type=IdealPDActuator,
            joint_indices=(2, 3),
            joint_names=("joint_2", "joint_3"),
            type_slice=slice(2, 4),
            num_worlds=1,
            num_dofs=2,
        ),
    }
    command = _command_binding(num_joints=4)
    joint_command = _command_binding(num_joints=4, fill=-999.0)
    computed = ProxyArray(wp.full((1, 4), -999.0, dtype=wp.float32, device="cpu"))
    applied = ProxyArray(wp.full((1, 4), -999.0, dtype=wp.float32, device="cpu"))
    identity = wp.array([0, 1, 2, 3], dtype=wp.int32, device="cpu")
    # These Warp aliases directly own the portion of the real Newton Control
    # that its four actuator DOFs use. submit_commands therefore has no shim.
    backend_views = [
        wp.from_torch(wp.to_torch(field)[:4].reshape(1, 4), dtype=wp.float32)
        for field in (
            newton_control.joint_target_q,
            newton_control.joint_target_qd,
            newton_control.joint_act,
            newton_control.joint_f,
        )
    ]
    data = SimpleNamespace(
        joint_ordering=None,
        has_joint_ordering=False,
        joint_pos=ProxyArray(wp.zeros((1, 4), dtype=wp.float32, device="cpu")),
        joint_vel=ProxyArray(wp.zeros((1, 4), dtype=wp.float32, device="cpu")),
        _sim_bind_joint_position_target=backend_views[0],
        _sim_bind_joint_velocity_target=backend_views[1],
        _sim_bind_joint_act=backend_views[2],
        _sim_bind_joint_effort=backend_views[3],
        _sim_bind_joint_computed_effort=wp.zeros((1, 4), dtype=wp.float32, device="cpu"),
    )
    articulation = SimpleNamespace(
        num_instances=1,
        num_joints=4,
        num_fixed_tendons=0,
        device="cpu",
        data=data,
        _data=data,
        _root_view=SimpleNamespace(
            frequency_layouts={
                NewtonModel.AttributeFrequency.JOINT_DOF: SimpleNamespace(
                    slice=SimpleNamespace(start=0), indices=None, offset=0
                )
            }
        ),
        newton_actuator_adapter=None,
        _newton_native_ranges=None,
        _native_dof_mask=None,
        _native_dof_mask_owner=None,
        _native_dof_masks=None,
        _native_dof_mask_owners=None,
        _implicit_dof_mask=None,
        _implicit_dof_mask_owner=None,
        _joint_user_to_backend_map=lambda: identity,
        _joint_backend_to_user_map=lambda: identity,
    )
    control = NewtonActuatorControl(articulation)
    control._native_active = True
    registration = SimpleNamespace(
        key=f"native-lab-native-{order}",
        control=control,
        native_group_names=frozenset(("native_first", "native_last")),
        cfgs={
            "native_first": IdealPDActuatorCfg(joint_names_expr=["joint_0", "joint_1"]),
            "lab": lab.cfg,
            "native_last": IdealPDActuatorCfg(joint_names_expr=["joint_2", "joint_3"]),
        },
    )
    binding = SimpleNamespace(
        registration=registration,
        groups=groups,
        command=command,
        joint_command=joint_command,
        computed_effort=computed,
        applied_effort=applied,
        native_group_names=registration.native_group_names,
        layout=SimpleNamespace(
            num_worlds=1,
            num_joints=4,
            type_layouts={IdealPDActuator: native_type_layout},
            group_layouts=tuple(layouts[name] for name in order),
        ),
    )
    adapter = NewtonActuatorAdapter(
        model.actuators,
        num_envs=1,
        num_joints=4,
        dof_offset=0,
        device="cpu",
        actuator_keys=(key,),
        actuator_dof_indices={key: (0, 1, 2, 3)},
    )
    adapter._joint_signatures = {f"joint_{index}": key for index in range(4)}
    monkeypatch.setattr(NewtonManager, "_adapter", adapter)
    monkeypatch.setattr(NewtonManager, "_post_actuator_callbacks", [])

    native_step_count = 0
    original_native_step = native_actuator.step

    def counted_native_step(*args, **kwargs):
        nonlocal native_step_count
        native_step_count += 1
        return original_native_step(*args, **kwargs)

    monkeypatch.setattr(native_actuator, "step", counted_native_step)
    lab_compute_count = 0
    original_lab_compute = lab.compute

    def counted_lab_compute(*args, **kwargs):
        nonlocal lab_compute_count
        lab_compute_count += 1
        return original_lab_compute(*args, **kwargs)

    monkeypatch.setattr(lab, "compute", counted_lab_compute)
    control._prepare_native_actuator_binding(binding)
    selector_state = _SelectorState(binding)
    plan = _ArticulationExecutionPlan.build(binding=binding, control=control, selector_state=selector_state)
    plan.set_runtime_hooks(validate_execution=lambda: None, native_compute=None)
    plan.reset(None)

    lab_pre_native: torch.Tensor | None = None
    try:
        for cycle, values in enumerate(
            (
                (
                    torch.tensor([[1.0, 21.0, 31.0, 4.0]]),
                    torch.tensor([[10.0, 210.0, 310.0, 40.0]]),
                    torch.tensor([[100.0, 2100.0, 3100.0, 400.0]]),
                ),
                (
                    torch.tensor([[2.0, 22.0, 32.0, 5.0]]),
                    torch.tensor([[20.0, 220.0, 320.0, 50.0]]),
                    torch.tensor([[200.0, 2200.0, 3200.0, 500.0]]),
                ),
            )
        ):
            for target, value in zip((command.position, command.velocity, command.effort), values, strict=True):
                target.torch.copy_(value)
            joint_command.position.warp.fill_(-999.0)
            joint_command.velocity.warp.fill_(-999.0)
            joint_command.effort.warp.fill_(-999.0)
            computed.warp.fill_(-999.0)
            applied.warp.fill_(-999.0)

            # The real opaque Lab executor is the only caller of the delayed
            # model. Its delayed values are observable before native merge.
            plan.compute()
            if cycle == 1:
                lab_pre_native = torch.stack(
                    (
                        joint_command.position.torch[0, 1:3],
                        joint_command.velocity.torch[0, 1:3],
                        joint_command.effort.torch[0, 1:3],
                    )
                )
            control.compute_native_actuators(binding, dt=0.01)
            control.submit_commands(binding)
            adapter.gather_staged_ranges(articulation._newton_native_ranges)
            adapter.step(state, newton_control, dt=0.01)
            assert len(NewtonManager._post_actuator_callbacks) == 1
            NewtonManager._post_actuator_callbacks[0]()

        assert lab_pre_native is not None
        return {
            "pre_native_lab": lab_pre_native,
            "position": joint_command.position.torch.clone(),
            "velocity": joint_command.velocity.torch.clone(),
            "effort": joint_command.effort.torch.clone(),
            "computed": computed.torch.clone(),
            "applied": applied.torch.clone(),
            "physical": wp.to_torch(newton_control.joint_f)[:4].reshape(1, 4).clone(),
            "native_steps": torch.tensor(native_step_count),
            "lab_computes": torch.tensor(lab_compute_count),
        }
    finally:
        plan.invalidate()
        control._unregister_post_actuator_callback()
        adapter.unregister_articulation_ranges(articulation._newton_native_ranges or ())
        selector_state.close()


def test_native_lab_native_control_uses_field_specific_winners(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native/Lab/native ordering independently routes targets, torque, telemetry, and physical force."""
    forward = _run_native_lab_native_control_cycle(monkeypatch, ("native_first", "lab", "native_last"))
    reverse = _run_native_lab_native_control_cycle(monkeypatch, ("native_last", "lab", "native_first"))

    # Explicit Lab PD clears processed position/velocity, so those poison
    # values prove it did not claim the target fields. Its one-step delayed
    # effort must still be the unique cycle-one sentinel; a second accidental
    # Lab compute would instead emit the cycle-two values.
    expected_lab = torch.tensor([[-999.0, -999.0], [-999.0, -999.0], [2100.0, 3100.0]])
    torch.testing.assert_close(forward["pre_native_lab"], expected_lab)
    torch.testing.assert_close(reverse["pre_native_lab"], expected_lab)
    assert int(forward["lab_computes"]) == 2
    assert int(reverse["lab_computes"]) == 2
    assert int(forward["native_steps"]) == 2
    assert int(reverse["native_steps"]) == 2

    expected_position = torch.tensor([[2.0, 22.0, 32.0, 5.0]])
    expected_velocity = torch.tensor([[20.0, 220.0, 320.0, 50.0]])
    torch.testing.assert_close(forward["position"], expected_position)
    torch.testing.assert_close(forward["velocity"], expected_velocity)
    torch.testing.assert_close(reverse["position"], expected_position)
    torch.testing.assert_close(reverse["velocity"], expected_velocity)

    # Forward: the middle Lab group owns effort/telemetry on joint 1, but
    # native still owns position/velocity there. Reverse: it owns joint 2.
    expected_forward = torch.tensor([[220.0, 2100.0, 3520.0, 550.0]])
    expected_reverse = torch.tensor([[220.0, 2420.0, 3100.0, 550.0]])
    for field in ("effort", "computed", "applied", "physical"):
        torch.testing.assert_close(forward[field], expected_forward)
        torch.testing.assert_close(reverse[field], expected_reverse)
