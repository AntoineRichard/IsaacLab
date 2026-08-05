# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from isaaclab.actuators import DCMotorCfg
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.integration
_EXECUTION_DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
]


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


def _make_dc_execution_fixture(device: str):
    """Create a fixed-shape DC motor private-execution fixture."""
    num_envs, num_joints = 3, 4
    cfg = DCMotorCfg(
        joint_names_expr=[f"joint_{index}" for index in range(num_joints)],
        stiffness=17.0,
        damping=3.0,
        effort_limit=60.0,
        saturation_effort=100.0,
        velocity_limit=50.0,
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


def _dc_execution_tensors(actuator, action: ArticulationActions) -> tuple[torch.Tensor, ...]:
    """Return every DC private execution tensor that must retain its exact view."""
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
        actuator.velocity_limit,
        actuator.saturation_effort,
        actuator._effort_limit_lower,
        actuator._vel_at_effort_lim,
        actuator._joint_vel,
        actuator._torque_speed_top,
        actuator._torque_speed_bottom,
        actuator._max_effort,
        actuator._min_effort,
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dc_motor_init_minimum(num_envs, num_joints, device):
    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    effort_limit = 60.0
    saturation_effort = 100.0
    velocity_limit = 50

    actuator_cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_limit,
        saturation_effort=saturation_effort,
        velocity_limit=velocity_limit,
    )
    # assume Articulation class:
    #   - finds joints (names and ids) associate with the provided joint_names_expr

    actuator = actuator_cfg.class_type(
        actuator_cfg,
        joint_names=joint_names,
        joint_ids=joint_ids,
        num_envs=num_envs,
        device=device,
    )

    # check device and shape
    torch.testing.assert_close(actuator.computed_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(actuator.applied_effort, torch.zeros(num_envs, num_joints, device=device))
    torch.testing.assert_close(
        actuator.effort_limit,
        effort_limit * torch.ones(num_envs, num_joints, device=device),
    )
    torch.testing.assert_close(
        actuator.velocity_limit, velocity_limit * torch.ones(num_envs, num_joints, device=device)
    )


@pytest.mark.parametrize("num_envs", [1, 2])
@pytest.mark.parametrize("num_joints", [1, 2])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("test_point", range(20))
def test_dc_motor_clip(num_envs, num_joints, device, test_point):
    r"""Test the computation of the dc motor actuator 4 quadrant torque speed curve.
    torque_speed_pairs of interest:

    0 - fully inside torque speed curve and effort limit (quadrant 1)
    1 - greater than effort limit but under torque-speed curve (quadrant 1)
    2 - greater than effort limit and outside torque-speed curve (quadrant 1)
    3 - less than effort limit but outside torque speed curve (quadrant 1)
    4 - less than effort limit but outside torque speed curve and outside corner velocity(quadrant 4)
    5 - fully inside torque speed curve and effort limit (quadrant 4)
    6 - fully outside torque speed curve and -effort limit (quadrant 4)
    7 - fully inside torque speed curve, outside -effort limit, and inside corner velocity (quadrant 4)
    8 - fully inside torque speed curves, outside -effort limit, and outside corner velocity (quadrant 4)
    9 - less than effort limit but outside torque speed curve and inside corner velocity (quadrant 4)
    e - effort_limit
    s - saturation_effort
    v - velocity_limit
    c - corner velocity
    \ - torque-speed linear boundary between v and s
    each torque_speed_point will be tested in quadrant 3 and 4
    ===========================================================
                            Torque
                             \  (+)
                               \ |
                Q2               s                   Q1
                                 | \        2
        \                        | 1 \
          c ---------------------e-----\
            \                    |       \
              \                  |  0      \ 3
                \                |           \
    (-)-----------v -------------o-------------v --------------(+) Speed
                    \            |               \   9    4
                      \          |    5            \
                        \        |                   \
                          \ -----e---------------------c
                            \    |                      \  6
                Q3            \  |              7    Q4   \
                                \s                          \
                                 |\                       8   \
                                (-) \
    ============================================================
    """
    effort_lim = 60
    saturation_effort = 100.0
    velocity_limit = 50

    torque_speed_pairs = [
        (30.0, 10.0),  # 0
        (70.0, 10.0),  # 1
        (80.0, 40.0),  # 2
        (30.0, 40.0),  # 3
        (-20.0, 90.0),  # 4
        (-30.0, 10.0),  # 5
        (-80.0, 110.0),  # 6
        (-80.0, 50.0),  # 7
        (-120.0, 90.0),  # 8
        (-10.0, 70.0),  # 9
        (-30.0, -10.0),  # -0
        (-70.0, -10.0),  # -1
        (-80.0, -40.0),  # -2
        (-30.0, -40.0),  # -3
        (20.0, -90.0),  # -4
        (30.0, -10.0),  # -5
        (80.0, -110.0),  # -6
        (80.0, -50.0),  # -7
        (120.0, -90.0),  # -8
        (10.0, -70.0),  # -9
    ]
    expected_clipped_effort = [
        30.0,  # 0
        60.0,  # 1
        20.0,  # 2
        20.0,  # 3
        -60.0,  # 4
        -30.0,  # 5
        -60.0,  # 6
        -60.0,  # 7
        -60.0,  # 8
        -40.0,  # 9
        -30.0,  # -0
        -60.0,  # -1
        -20,  # -2
        -20,  # -3
        60.0,  # -4
        30.0,  # -5
        60.0,  # -6
        60.0,  # -7
        60.0,  # -8
        40.0,  # -9
    ]

    joint_names = [f"joint_{d}" for d in range(num_joints)]
    joint_ids = [d for d in range(num_joints)]
    stiffness = 200
    damping = 10
    actuator_cfg = DCMotorCfg(
        joint_names_expr=joint_names,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort_lim,
        velocity_limit=velocity_limit,
        saturation_effort=saturation_effort,
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

    ts = torque_speed_pairs[test_point]
    torque = ts[0]
    speed = ts[1]
    actuator._joint_vel[:] = speed * torch.ones(num_envs, num_joints, device=device)
    effort = torque * torch.ones(num_envs, num_joints, device=device)
    clipped_effort = actuator._clip_effort(effort)
    torch.testing.assert_close(
        expected_clipped_effort[test_point] * torch.ones(num_envs, num_joints, device=device),
        clipped_effort,
    )


@pytest.mark.parametrize("device", _EXECUTION_DEVICES)
def test_dc_motor_private_execution_preserves_fixed_tensor_views(device: str):
    """Private execution writes every DC result into finalized action, output, and scratch tensors."""
    actuator, action, joint_pos, joint_vel = _make_dc_execution_fixture(device)
    tensors = _dc_execution_tensors(actuator, action) + (joint_pos, joint_vel)
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
def test_dc_motor_private_execution_has_no_new_tensor_storage(device: str):
    """Private DC execution rejects hidden dispatcher allocations after the warm-up step."""
    actuator, action, joint_pos, joint_vel = _make_dc_execution_fixture(device)
    tensors = _dc_execution_tensors(actuator, action) + (joint_pos, joint_vel)
    action.joint_positions.fill_(1.0)
    action.joint_velocities.fill_(-2.0)
    action.joint_efforts.fill_(0.5)
    joint_pos.fill_(0.25)
    joint_vel.fill_(-0.75)
    actuator._compute_execution(action, joint_pos, joint_vel)

    with _NoNewTensorStorage(tensors):
        actuator._compute_execution(action, joint_pos, joint_vel)


@pytest.mark.parametrize("device", _EXECUTION_DEVICES)
def test_dc_motor_private_execution_matches_public_compute_exactly(device: str):
    """Private DC execution preserves public float32 arithmetic at random and corner velocities."""
    actuator, private_action, joint_pos, joint_vel = _make_dc_execution_fixture(device)
    generator = torch.Generator(device=device).manual_seed(29)
    corner_velocity = 80.0
    corner_offsets = torch.tensor(
        [-corner_velocity - 0.001, -corner_velocity, -corner_velocity + 0.001, corner_velocity - 0.001],
        device=device,
    )

    for iteration in range(16):
        command_pos = torch.randn((3, 4), device=device, generator=generator)
        command_vel = torch.randn((3, 4), device=device, generator=generator)
        command_effort = 100.0 * torch.randn((3, 4), device=device, generator=generator)
        joint_pos.copy_(torch.randn((3, 4), device=device, generator=generator))
        joint_vel.copy_(torch.randn((3, 4), device=device, generator=generator))
        if iteration == 0:
            joint_vel[0].copy_(corner_offsets)
            joint_vel[1].copy_(-corner_offsets)
            command_effort[2].copy_(torch.tensor([-60.0, 60.0, -60.001, 60.001], device=device))
        public_action = ArticulationActions(command_pos.clone(), command_vel.clone(), command_effort.clone())
        private_action.joint_positions.copy_(command_pos)
        private_action.joint_velocities.copy_(command_vel)
        private_action.joint_efforts.copy_(command_effort)
        private_fields = (private_action.joint_positions, private_action.joint_velocities, private_action.joint_efforts)

        returned = actuator.compute(public_action, joint_pos, joint_vel)
        expected_computed = actuator.computed_effort.clone()
        expected_applied = actuator.applied_effort.clone()
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
        torch.testing.assert_close(actuator.computed_effort, expected_computed, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actuator.applied_effort, expected_applied, rtol=0.0, atol=0.0)
