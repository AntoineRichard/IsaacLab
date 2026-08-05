# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.integration
_EXECUTION_DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
]


class _ZeroClipIdealPDActuator(IdealPDActuator):
    """Ideal PD test actuator whose protected clip seam suppresses all effort."""

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(effort)


def _tensor_fingerprint(
    tensor: torch.Tensor,
) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]:
    """Return the complete fixed-view identity required by private execution."""
    return (
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


class _NoNewTensorStorage(TorchDispatchMode):
    """Reject dispatcher results whose storage was not finalized before execution."""

    def __init__(self, tensors: tuple[torch.Tensor, ...]):
        super().__init__()
        self._storage_pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **({} if kwargs is None else kwargs))
        leaves, _ = torch.utils._pytree.tree_flatten(result)
        for tensor in leaves:
            if isinstance(tensor, torch.Tensor):
                assert tensor.untyped_storage().data_ptr() in self._storage_pointers, func
        return result


def _make_ideal_pd_execution_fixture(device: str):
    """Create a fixed-shape IdealPD private-execution fixture."""
    num_envs, num_joints = 3, 4
    cfg = IdealPDActuatorCfg(
        joint_names_expr=[f"joint_{index}" for index in range(num_joints)],
        stiffness=17.0,
        damping=3.0,
        effort_limit=23.0,
    )
    actuator = cfg.class_type(
        cfg,
        joint_names=[f"joint_{index}" for index in range(num_joints)],
        joint_ids=list(range(num_joints)),
        num_envs=num_envs,
        device=device,
    )
    action = ArticulationActions(
        joint_positions=torch.empty(num_envs, num_joints, device=device),
        joint_velocities=torch.empty(num_envs, num_joints, device=device),
        joint_efforts=torch.empty(num_envs, num_joints, device=device),
        joint_indices=torch.arange(num_joints, device=device),
    )
    joint_pos = torch.empty(num_envs, num_joints, device=device)
    joint_vel = torch.empty(num_envs, num_joints, device=device)
    return actuator, action, joint_pos, joint_vel


def _ideal_pd_execution_tensors(actuator, action: ArticulationActions) -> tuple[torch.Tensor, ...]:
    """Return all private execution tensors that must retain their exact views."""
    return (
        action.joint_positions,
        action.joint_velocities,
        action.joint_efforts,
        action.joint_indices,
        actuator.computed_effort,
        actuator.applied_effort,
        actuator.stiffness,
        actuator.damping,
        actuator.effort_limit,
        actuator._effort_limit_lower,
    )


def _literal_ideal_pd_reference(
    action: ArticulationActions,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    effort_limit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the pre-change IdealPD equations in their original operation order."""
    computed_effort = torch.empty_like(joint_pos)
    velocity_error = torch.empty_like(joint_vel)
    torch.sub(action.joint_positions, joint_pos, out=computed_effort)
    torch.sub(action.joint_velocities, joint_vel, out=velocity_error)
    computed_effort.mul_(stiffness)
    computed_effort.addcmul_(damping, velocity_error)
    computed_effort.add_(action.joint_efforts)
    applied_effort = torch.clip(computed_effort, min=-effort_limit, max=effort_limit)
    return computed_effort, applied_effort


def _assert_tensor_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert zero-tolerance equality, including the sign of zero values."""
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
    zero_mask = (actual == 0.0) & (expected == 0.0)
    assert torch.equal(torch.signbit(actual[zero_mask]), torch.signbit(expected[zero_mask]))


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("usd_default", [False, True])
def test_ideal_pd_actuator_init_minimum(num_envs, num_joints, device, usd_default):
    """Test initialization of ideal pd actuator with minimum configuration."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = None if usd_default else 200
    damping = None if usd_default else 10
    friction = None if usd_default else 0.1
    armature = None if usd_default else 0.2

    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        armature=armature,
        friction=friction,
    )
    # assume Articulation class:
    #   - finds joints (names and ids) associate with the provided joint_names_expr

    # faux usd defaults
    stiffness_default = 300
    damping_default = 20
    friction_default = 0.0
    armature_default = 0.0

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=stiffness_default,
        damping=damping_default,
        friction=friction_default,
        armature=armature_default,
    )

    # check initialized actuator
    assert actuator.is_implicit_model is False
    # check device and shape
    torch.testing.assert_close(actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device))

    torch.testing.assert_close(actuator.effort_limit, torch.inf * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(
        actuator.effort_limit_sim, actuator._DEFAULT_MAX_EFFORT_SIM * torch.ones(num_envs, num_joints, device=device)
    )
    torch.testing.assert_close(actuator.velocity_limit, torch.inf * torch.ones(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.velocity_limit_sim, torch.inf * torch.ones(num_envs, num_joints, device=device))

    if not usd_default:
        torch.testing.assert_close(actuator.stiffness, stiffness * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.damping, damping * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.armature, armature * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(actuator.friction, friction * torch.ones(num_envs, num_joints, device=device))
    else:
        torch.testing.assert_close(
            actuator.stiffness, stiffness_default * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(actuator.damping, damping_default * torch.ones(num_envs, num_joints, device=device))
        torch.testing.assert_close(
            actuator.armature, armature_default * torch.ones(num_envs, num_joints, device=device)
        )
        torch.testing.assert_close(
            actuator.friction, friction_default * torch.ones(num_envs, num_joints, device=device)
        )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 300])
@pytest.mark.parametrize("effort_lim_sim", [None, 400])
def test_ideal_pd_actuator_init_effort_limits(num_envs, num_joints, device, effort_lim, effort_lim_sim):
    """Test initialization of ideal pd actuator with effort limits."""
    # used as a standin for the usd default value read in by articulation.
    # This value should not be propagated for ideal pd actuators
    effort_lim_default = 5000

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        effort_limit=effort_lim,
        effort_limit_sim=effort_lim_sim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        effort_limit=effort_lim_default,
    )

    if effort_lim is not None and effort_lim_sim is None:
        effort_lim_expected = effort_lim
        effort_lim_sim_expected = actuator._DEFAULT_MAX_EFFORT_SIM

    elif effort_lim is None and effort_lim_sim is not None:
        effort_lim_expected = effort_lim_default
        effort_lim_sim_expected = effort_lim_sim

    elif effort_lim is None and effort_lim_sim is None:
        effort_lim_expected = effort_lim_default
        effort_lim_sim_expected = actuator._DEFAULT_MAX_EFFORT_SIM

    elif effort_lim is not None and effort_lim_sim is not None:
        effort_lim_expected = effort_lim
        effort_lim_sim_expected = effort_lim_sim

    torch.testing.assert_close(
        actuator.effort_limit, effort_lim_expected * torch.ones(num_envs, num_joints, device=device)
    )
    torch.testing.assert_close(
        actuator.effort_limit_sim, effort_lim_sim_expected * torch.ones(num_envs, num_joints, device=device)
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("velocity_lim", [None, 300])
@pytest.mark.parametrize("velocity_lim_sim", [None, 400])
def test_ideal_pd_actuator_init_velocity_limits(num_envs, num_joints, device, velocity_lim, velocity_lim_sim):
    """Test initialization of ideal pd actuator with velocity limits.

    Note Ideal PD actuator does not use velocity limits in computation, they are passed to physics via articulations.
    """
    velocity_limit_default = 1000
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]

    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=200,
        damping=10,
        velocity_limit=velocity_lim,
        velocity_limit_sim=velocity_lim_sim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
        velocity_limit=velocity_limit_default,
    )
    if velocity_lim is not None and velocity_lim_sim is None:
        vel_lim_expected = velocity_lim
        vel_lim_sim_expected = velocity_limit_default
    elif velocity_lim is None and velocity_lim_sim is not None:
        vel_lim_expected = velocity_lim_sim
        vel_lim_sim_expected = velocity_lim_sim
    elif velocity_lim is None and velocity_lim_sim is None:
        vel_lim_expected = velocity_limit_default
        vel_lim_sim_expected = velocity_limit_default
    elif velocity_lim is not None and velocity_lim_sim is not None:
        vel_lim_expected = velocity_lim
        vel_lim_sim_expected = velocity_lim_sim

    torch.testing.assert_close(
        actuator.velocity_limit, vel_lim_expected * torch.ones(num_envs, num_joints, device=device)
    )
    torch.testing.assert_close(
        actuator.velocity_limit_sim, vel_lim_sim_expected * torch.ones(num_envs, num_joints, device=device)
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("effort_lim", [None, 300])
def test_ideal_pd_compute(num_envs, num_joints, device, effort_lim):
    """Test the computation of the ideal pd actuator."""

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    actuator_cfg = IdealPDActuatorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_lim,
    )

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
        stiffness=actuator_cfg.stiffness,
        damping=actuator_cfg.damping,
    )
    desired_pos = 10.0
    desired_vel = 0.1
    measured_joint_pos = 1.0
    measured_joint_vel = -0.1

    desired_control_action = ArticulationActions()
    desired_control_action.joint_positions = desired_pos * torch.ones(num_envs, num_joints, device=device)
    desired_control_action.joint_velocities = desired_vel * torch.ones(num_envs, num_joints, device=device)
    desired_control_action.joint_efforts = torch.zeros(num_envs, num_joints, device=device)

    expected_comp_joint_effort = stiffness * (desired_pos - measured_joint_pos) + damping * (
        desired_vel - measured_joint_vel
    )

    computed_control_action = actuator.compute(
        desired_control_action,
        measured_joint_pos * torch.ones(num_envs, num_joints, device=device),
        measured_joint_vel * torch.ones(num_envs, num_joints, device=device),
    )

    torch.testing.assert_close(
        expected_comp_joint_effort * torch.ones(num_envs, num_joints, device=device), actuator.computed_effort
    )

    if effort_lim is None:
        torch.testing.assert_close(
            expected_comp_joint_effort * torch.ones(num_envs, num_joints, device=device), actuator.applied_effort
        )
    else:
        torch.testing.assert_close(
            effort_lim * torch.ones(num_envs, num_joints, device=device), actuator.applied_effort
        )
    torch.testing.assert_close(
        actuator.applied_effort,
        computed_control_action.joint_efforts,
    )


def test_ideal_pd_public_compute_honors_custom_clip_override():
    """Public compute dispatches through the established protected clipping seam."""
    cfg = IdealPDActuatorCfg(
        joint_names_expr=["joint_0"],
        stiffness=1.0,
        damping=0.0,
        effort_limit=10.0,
    )
    actuator = _ZeroClipIdealPDActuator(
        cfg,
        joint_names=["joint_0"],
        joint_ids=[0],
        num_envs=1,
        device="cpu",
    )
    action = ArticulationActions(
        joint_positions=torch.ones(1, 1),
        joint_velocities=torch.zeros(1, 1),
        joint_efforts=torch.zeros(1, 1),
    )

    returned = actuator.compute(action, torch.zeros(1, 1), torch.zeros(1, 1))

    torch.testing.assert_close(actuator.computed_effort, torch.ones(1, 1))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(1, 1))
    assert returned.joint_efforts is actuator.applied_effort


@pytest.mark.parametrize("device", _EXECUTION_DEVICES)
def test_ideal_pd_private_execution_preserves_fixed_tensor_views(device: str):
    """Private execution writes every result into finalized action, output, and scratch tensors."""
    actuator, action, joint_pos, joint_vel = _make_ideal_pd_execution_fixture(device)
    tensors = _ideal_pd_execution_tensors(actuator, action) + (joint_pos, joint_vel)
    fingerprints = tuple(_tensor_fingerprint(tensor) for tensor in tensors)
    objects = (action.joint_positions, action.joint_velocities, action.joint_efforts, action.joint_indices)

    for scale in (1.0, 2.0, -3.0):
        action.joint_positions.fill_(scale)
        action.joint_velocities.fill_(-2.0 * scale)
        action.joint_efforts.fill_(0.5 * scale)
        joint_pos.fill_(0.25 * scale)
        joint_vel.fill_(-0.75 * scale)
        actuator._compute_execution(action, joint_pos, joint_vel)
        assert all(
            actual is expected
            for actual, expected in zip(
                (action.joint_positions, action.joint_velocities, action.joint_efforts, action.joint_indices),
                objects,
                strict=True,
            )
        )
        assert tuple(_tensor_fingerprint(tensor) for tensor in tensors) == fingerprints


@pytest.mark.parametrize("device", _EXECUTION_DEVICES)
def test_ideal_pd_private_execution_has_no_new_tensor_storage(device: str):
    """Private execution rejects hidden dispatcher allocations after the warm-up step."""
    actuator, action, joint_pos, joint_vel = _make_ideal_pd_execution_fixture(device)
    tensors = _ideal_pd_execution_tensors(actuator, action) + (joint_pos, joint_vel)
    action.joint_positions.fill_(1.0)
    action.joint_velocities.fill_(-2.0)
    action.joint_efforts.fill_(0.5)
    joint_pos.fill_(0.25)
    joint_vel.fill_(-0.75)
    actuator._compute_execution(action, joint_pos, joint_vel)

    with _NoNewTensorStorage(tensors):
        actuator._compute_execution(action, joint_pos, joint_vel)


@pytest.mark.parametrize("device", _EXECUTION_DEVICES)
def test_ideal_pd_private_execution_matches_public_compute_exactly(device: str):
    """Public and private execution match independent pre-change equations exactly."""
    actuator, private_action, joint_pos, joint_vel = _make_ideal_pd_execution_fixture(device)
    generator = torch.Generator(device=device).manual_seed(17)

    for _ in range(16):
        command_pos = torch.randn((3, 4), device=device, generator=generator)
        command_vel = torch.randn((3, 4), device=device, generator=generator)
        command_effort = torch.randn((3, 4), device=device, generator=generator)
        joint_pos.copy_(torch.randn((3, 4), device=device, generator=generator))
        joint_vel.copy_(torch.randn((3, 4), device=device, generator=generator))
        public_action = ArticulationActions(command_pos.clone(), command_vel.clone(), command_effort.clone())
        private_action.joint_positions.copy_(command_pos)
        private_action.joint_velocities.copy_(command_vel)
        private_action.joint_efforts.copy_(command_effort)
        private_fields = (private_action.joint_positions, private_action.joint_velocities, private_action.joint_efforts)

        reference_action = ArticulationActions(command_pos, command_vel, command_effort)
        expected_computed, expected_applied = _literal_ideal_pd_reference(
            reference_action,
            joint_pos,
            joint_vel,
            actuator.stiffness,
            actuator.damping,
            actuator.effort_limit,
        )

        returned = actuator.compute(public_action, joint_pos, joint_vel)
        _assert_tensor_exact(actuator.computed_effort, expected_computed)
        _assert_tensor_exact(actuator.applied_effort, expected_applied)
        actuator._compute_execution(private_action, joint_pos, joint_vel)

        assert returned is public_action
        assert public_action.joint_efforts is actuator.applied_effort
        assert public_action.joint_positions is None
        assert public_action.joint_velocities is None
        assert all(
            actual is expected
            for actual, expected in zip(
                (private_action.joint_positions, private_action.joint_velocities, private_action.joint_efforts),
                private_fields,
                strict=True,
            )
        )
        _assert_tensor_exact(actuator.computed_effort, expected_computed)
        _assert_tensor_exact(actuator.applied_effort, expected_applied)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
