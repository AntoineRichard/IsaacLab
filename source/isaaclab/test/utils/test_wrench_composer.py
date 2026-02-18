# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import warnings as warnings_module

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets import RigidObject
from isaaclab.utils.wrench_composer import WrenchComposer


class MockAssetData:
    """Mock data class that provides body link positions and quaternions."""

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
        link_pos: torch.Tensor | None = None,
        link_quat: torch.Tensor | None = None,
    ):
        """Initialize mock asset data.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of bodies.
            device: Device to use.
            link_pos: Optional link positions (num_envs, num_bodies, 3). Defaults to zeros.
            link_quat: Optional link quaternions in (w, x, y, z) format (num_envs, num_bodies, 4).
                       Defaults to identity quaternion.
        """
        if link_pos is not None:
            self.body_link_pos_w = link_pos.to(device=device, dtype=torch.float32)
        else:
            self.body_link_pos_w = torch.zeros((num_envs, num_bodies, 3), dtype=torch.float32, device=device)

        if link_quat is not None:
            self.body_link_quat_w = link_quat.to(device=device, dtype=torch.float32)
        else:
            # Identity quaternion (w, x, y, z) = (1, 0, 0, 0)
            self.body_link_quat_w = torch.zeros((num_envs, num_bodies, 4), dtype=torch.float32, device=device)
            self.body_link_quat_w[..., 0] = 1.0


class MockRigidObject:
    """Mock RigidObject that provides the minimal interface required by WrenchComposer.

    This mock enables testing WrenchComposer in isolation without requiring a full simulation setup.
    It passes isinstance checks by registering as a virtual subclass of RigidObject.
    """

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
        link_pos: torch.Tensor | None = None,
        link_quat: torch.Tensor | None = None,
    ):
        """Initialize mock rigid object.

        Args:
            num_envs: Number of environments.
            num_bodies: Number of bodies.
            device: Device to use.
            link_pos: Optional link positions (num_envs, num_bodies, 3).
            link_quat: Optional link quaternions in (w, x, y, z) format (num_envs, num_bodies, 4).
        """
        self.num_instances = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.data = MockAssetData(num_envs, num_bodies, device, link_pos, link_quat)


# --- Helper functions for quaternion math ---


def quat_rotate_inv_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by the inverse of a quaternion (numpy).

    Args:
        quat_wxyz: Quaternion in (w, x, y, z) format. Shape: (..., 4)
        vec: Vector to rotate. Shape: (..., 3)

    Returns:
        Rotated vector. Shape: (..., 3)
    """
    # Extract components
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]

    # For inverse rotation, we conjugate the quaternion (negate xyz)
    # q^-1 * v * q = q_conj * v * q_conj^-1 for unit quaternion
    # Using the formula: v' = v + 2*w*(xyz x v) + 2*(xyz x (xyz x v))
    # But for inverse: use -xyz

    # Cross product: xyz x vec
    t = 2.0 * np.cross(-xyz, vec, axis=-1)
    # Result: vec + w*t + xyz x t
    return vec + w * t + np.cross(-xyz, t, axis=-1)


def quat_rotate_np(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion (numpy).

    Args:
        quat_wxyz: Quaternion in (w, x, y, z) format. Shape: (..., 4)
        vec: Vector to rotate. Shape: (..., 3)

    Returns:
        Rotated vector. Shape: (..., 3)
    """
    # Extract components
    w = quat_wxyz[..., 0:1]
    xyz = quat_wxyz[..., 1:4]

    # Using the formula: v' = v + 2*w*(xyz x v) + 2*(xyz x (xyz x v))
    t = 2.0 * np.cross(xyz, vec, axis=-1)
    return vec + w * t + np.cross(xyz, t, axis=-1)


def random_unit_quaternion_np(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """Generate random unit quaternions in (w, x, y, z) format.

    Args:
        rng: Random number generator.
        shape: Output shape, e.g. (num_envs, num_bodies).

    Returns:
        Random unit quaternions. Shape: (*shape, 4)
    """
    # Generate random quaternion components
    q = rng.standard_normal(shape + (4,)).astype(np.float32)
    # Normalize to unit quaternion
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q


# Register MockRigidObject as a virtual subclass of RigidObject
# This allows isinstance(mock, RigidObject) to return True
RigidObject.register(MockRigidObject)


# ============================================================================
# Basic Tests (identity quaternion, is_global=False by default)
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_force(device: str, num_envs: int, num_bodies: int):
    """Test adding local forces (default is_global=False) with identity quaternion."""
    rng = np.random.default_rng(seed=0)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            forces_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            wrench_composer.add_forces_and_torques(forces=forces, body_ids=body_ids, env_ids=env_ids)
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
        # Local forces with identity quat → compose gives local forces unchanged
        wrench_composer.compose_to_body_frame()
        out_force_np = wrench_composer.out_force_b.numpy()
        assert np.allclose(out_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_add_torque(device: str, num_envs: int, num_bodies: int):
    """Test adding local torques (default is_global=False) with identity quaternion."""
    rng = np.random.default_rng(seed=1)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            torques_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
            wrench_composer.add_forces_and_torques(torques=torques, body_ids=body_ids, env_ids=env_ids)
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        wrench_composer.compose_to_body_frame()
        out_torque_np = wrench_composer.out_torque_b.numpy()
        assert np.allclose(out_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_at_positons(device: str, num_envs: int, num_bodies: int):
    """Test adding local forces at local positions (offset from link frame)."""
    rng = np.random.default_rng(seed=2)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            forces_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            positions_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)
            wrench_composer.add_forces_and_torques(
                forces=forces, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            # Local forces accumulate directly
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            # Local torque from position: cross(local_pos, local_force)
            torques_from_forces = np.cross(positions_np, forces_np)
            for i in range(num_envs_np):
                for j in range(num_bodies_np):
                    hand_calculated_composed_torque_np[env_ids_np[i], body_ids_np[j], :] += torques_from_forces[i, j, :]

        wrench_composer.compose_to_body_frame()
        out_force_np = wrench_composer.out_force_b.numpy()
        assert np.allclose(out_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        out_torque_np = wrench_composer.out_torque_b.numpy()
        assert np.allclose(out_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_torques_at_position(device: str, num_envs: int, num_bodies: int):
    """Test that positions don't affect torque-only additions."""
    rng = np.random.default_rng(seed=3)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            torques_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            positions_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)
            wrench_composer.add_forces_and_torques(
                torques=torques, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        wrench_composer.compose_to_body_frame()
        out_torque_np = wrench_composer.out_torque_b.numpy()
        assert np.allclose(out_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_add_forces_and_torques_at_position(device: str, num_envs: int, num_bodies: int):
    """Test adding local forces and torques at local positions."""
    rng = np.random.default_rng(seed=4)

    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        hand_calculated_composed_force_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        hand_calculated_composed_torque_np = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        for _ in range(10):
            num_envs_np = rng.integers(1, num_envs, endpoint=True)
            num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
            env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
            body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
            env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
            body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
            forces_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            torques_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            positions_np = (
                np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
                .reshape(num_envs_np, num_bodies_np, 3)
                .astype(np.float32)
            )
            forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
            torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
            positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)
            wrench_composer.add_forces_and_torques(
                forces=forces, torques=torques, positions=positions, body_ids=body_ids, env_ids=env_ids
            )
            hand_calculated_composed_force_np[env_ids_np[:, None], body_ids_np[None, :], :] += forces_np
            torques_from_forces = np.cross(positions_np, forces_np)
            for i in range(num_envs_np):
                for j in range(num_bodies_np):
                    hand_calculated_composed_torque_np[env_ids_np[i], body_ids_np[j], :] += torques_from_forces[i, j, :]
            hand_calculated_composed_torque_np[env_ids_np[:, None], body_ids_np[None, :], :] += torques_np
        wrench_composer.compose_to_body_frame()
        out_force_np = wrench_composer.out_force_b.numpy()
        assert np.allclose(out_force_np, hand_calculated_composed_force_np, atol=1, rtol=1e-7)
        out_torque_np = wrench_composer.out_torque_b.numpy()
        assert np.allclose(out_torque_np, hand_calculated_composed_torque_np, atol=1, rtol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100, 1000])
@pytest.mark.parametrize("num_bodies", [1, 3, 5, 10])
def test_wrench_composer_reset(device: str, num_envs: int, num_bodies: int):
    """Test that reset zeros all 4 input buffers and 2 output buffers."""
    rng = np.random.default_rng(seed=5)
    for _ in range(10):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)
        num_envs_np = rng.integers(1, num_envs, endpoint=True)
        num_bodies_np = rng.integers(1, num_bodies, endpoint=True)
        env_ids_np = rng.choice(num_envs, size=num_envs_np, replace=False)
        body_ids_np = rng.choice(num_bodies, size=num_bodies_np, replace=False)
        env_ids = wp.from_numpy(env_ids_np, dtype=wp.int32, device=device)
        body_ids = wp.from_numpy(body_ids_np, dtype=wp.int32, device=device)
        forces_np = (
            np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
            .reshape(num_envs_np, num_bodies_np, 3)
            .astype(np.float32)
        )
        torques_np = (
            np.random.uniform(low=-100.0, high=100.0, size=(num_envs_np * num_bodies_np * 3))
            .reshape(num_envs_np, num_bodies_np, 3)
            .astype(np.float32)
        )
        forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
        torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)
        # Add local forces/torques
        wrench_composer.add_forces_and_torques(forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids)
        # Add global forces/torques
        wrench_composer.add_forces_and_torques(
            forces=forces, torques=torques, body_ids=body_ids, env_ids=env_ids, is_global=True
        )
        # Compose to populate output buffers
        wrench_composer.compose_to_body_frame()
        # Reset
        wrench_composer.reset()
        # All 7 buffers should be zero (5 input + 2 output)
        zeros = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        assert np.allclose(wrench_composer.global_force_w.numpy(), zeros, atol=1e-7)
        assert np.allclose(wrench_composer.global_torque_w.numpy(), zeros, atol=1e-7)
        assert np.allclose(wrench_composer.global_force_at_com_w.numpy(), zeros, atol=1e-7)
        assert np.allclose(wrench_composer.local_force_b.numpy(), zeros, atol=1e-7)
        assert np.allclose(wrench_composer.local_torque_b.numpy(), zeros, atol=1e-7)
        # Access _out_force_b directly to avoid triggering warning (dirty=False after reset)
        assert np.allclose(wrench_composer._out_force_b.numpy(), zeros, atol=1e-7)
        assert np.allclose(wrench_composer._out_torque_b.numpy(), zeros, atol=1e-7)


# ============================================================================
# Global Frame Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_global_forces_stored_in_global_buffer(device: str, num_envs: int, num_bodies: int):
    """Test that global forces without positions are stored in the global_force_at_com_w buffer."""
    rng = np.random.default_rng(seed=10)

    for _ in range(5):
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

        # Global forces without positions stored in global_force_at_com_w buffer
        global_force_at_com_np = wrench_composer.global_force_at_com_w.numpy()
        assert np.allclose(global_force_at_com_np, forces_global_np, atol=1e-4, rtol=1e-5)
        # Positional global_force_w should remain zero
        global_force_np = wrench_composer.global_force_w.numpy()
        assert np.allclose(global_force_np, np.zeros_like(forces_global_np), atol=1e-7)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 100])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_global_torques_stored_in_global_buffer(device: str, num_envs: int, num_bodies: int):
    """Test that global torques are stored unchanged in the global buffer."""
    rng = np.random.default_rng(seed=11)

    for _ in range(5):
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        torques_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        torques_global = wp.from_numpy(torques_global_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(torques=torques_global, is_global=True)

        global_torque_np = wrench_composer.global_torque_w.numpy()
        assert np.allclose(global_torque_np, torques_global_np, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_global_forces_at_global_position(device: str, num_envs: int, num_bodies: int):
    """Test global forces at global positions produce correct torque in global buffer.

    Global torque is stored about the world origin: cross(P, F).
    After compose, the correction -cross(link_pos, F) gives torque about CoM:
    cross(P, F) - cross(link_pos, F) = cross(P - link_pos, F).
    """
    rng = np.random.default_rng(seed=12)

    for _ in range(5):
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_global_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)
        positions_global = wp.from_numpy(positions_global_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_global, is_global=True)

        # Global force stored unchanged
        expected_forces = forces_global_np
        # Torque stored about world origin: cross(P, F)
        expected_stored_torques = np.cross(positions_global_np, forces_global_np)

        global_force_np = wrench_composer.global_force_w.numpy()
        assert np.allclose(global_force_np, expected_forces, atol=1e-3, rtol=1e-4)

        global_torque_np = wrench_composer.global_torque_w.numpy()
        assert np.allclose(global_torque_np, expected_stored_torques, atol=1e-3, rtol=1e-4)

        # After compose, output torque should be R^T @ cross(P - link_pos, F)
        wrench_composer.compose_to_body_frame()
        corrected_torque_w = expected_stored_torques - np.cross(link_pos_np, forces_global_np)
        expected_out_torque = quat_rotate_inv_np(link_quat_np, corrected_torque_w)
        out_torque_np = wrench_composer.out_torque_b.numpy()
        assert np.allclose(out_torque_np, expected_out_torque, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_local_vs_global_identity_quaternion(device: str):
    """Test that local and global give same composed result with identity quaternion."""
    rng = np.random.default_rng(seed=13)
    num_envs, num_bodies = 10, 5

    mock_asset_local = MockRigidObject(num_envs, num_bodies, device)
    mock_asset_global = MockRigidObject(num_envs, num_bodies, device)

    wrench_composer_local = WrenchComposer(mock_asset_local)
    wrench_composer_global = WrenchComposer(mock_asset_global)

    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    torques = wp.from_numpy(torques_np, dtype=wp.vec3f, device=device)

    wrench_composer_local.add_forces_and_torques(forces=forces, torques=torques, is_global=False)
    wrench_composer_global.add_forces_and_torques(forces=forces, torques=torques, is_global=True)

    # Both should produce same body-frame output with identity quaternion
    wrench_composer_local.compose_to_body_frame()
    wrench_composer_global.compose_to_body_frame()

    assert np.allclose(
        wrench_composer_local.out_force_b.numpy(),
        wrench_composer_global.out_force_b.numpy(),
        atol=1e-6,
    )
    assert np.allclose(
        wrench_composer_local.out_torque_b.numpy(),
        wrench_composer_global.out_torque_b.numpy(),
        atol=1e-6,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_90_degree_rotation_global_force(device: str):
    """Test global force composed to body frame with a known 90-degree rotation."""
    num_envs, num_bodies = 1, 1

    # 90-degree rotation around Z-axis: body X points along world Y
    angle = np.pi / 2
    link_quat_np = np.array([[[[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]]]], dtype=np.float32).reshape(1, 1, 4)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Apply global force in +X direction
    force_global = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    force_wp = wp.from_numpy(force_global, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=force_wp, is_global=True)

    wrench_composer.compose_to_body_frame()

    # Global +X rotated to body frame: quat_rotate_inv(90deg_Z, (1,0,0)) = (0,-1,0)
    expected_force_b = np.array([[[0.0, -1.0, 0.0]]], dtype=np.float32)
    out_force_np = wrench_composer.out_force_b.numpy()
    assert np.allclose(out_force_np, expected_force_b, atol=1e-5), (
        f"Expected:\n{expected_force_b}\nGot:\n{out_force_np}"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_90_degree_rotation_local_force(device: str):
    """Test local force stays unchanged in body frame after compose."""
    num_envs, num_bodies = 1, 1

    # 90-degree rotation around Z-axis
    angle = np.pi / 2
    link_quat_np = np.array([[[[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]]]], dtype=np.float32).reshape(1, 1, 4)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    # Apply force in local +X direction
    force_local = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    force_wp = wp.from_numpy(force_local, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=force_wp, is_global=False)

    wrench_composer.compose_to_body_frame()

    # Local force stays unchanged in body frame
    expected_force_b = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    out_force_np = wrench_composer.out_force_b.numpy()
    assert np.allclose(out_force_np, expected_force_b, atol=1e-5), (
        f"Expected:\n{expected_force_b}\nGot:\n{out_force_np}"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_composition_local_and_global(device: str):
    """Test that local and global forces compose correctly in body frame."""
    rng = np.random.default_rng(seed=14)
    num_envs, num_bodies = 5, 3

    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)

    forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    wrench_composer.add_forces_and_torques(forces=forces_local, is_global=False)
    wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

    wrench_composer.compose_to_body_frame()

    # out_force_b = quat_rotate_inv(q, global_force) + local_force
    global_in_body = quat_rotate_inv_np(link_quat_np, forces_global_np)
    expected_total = global_in_body + forces_local_np

    out_force_np = wrench_composer.out_force_b.numpy()
    assert np.allclose(out_force_np, expected_total, atol=1e-4, rtol=1e-5), (
        f"local/global composition failed.\nExpected:\n{expected_total}\nGot:\n{out_force_np}"
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 10, 50])
@pytest.mark.parametrize("num_bodies", [1, 3, 5])
def test_local_forces_and_torques_at_local_position(device: str, num_envs: int, num_bodies: int):
    """Test local forces at local positions produce correct body-frame output."""
    rng = np.random.default_rng(seed=15)

    for _ in range(5):
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        torques_local_np = rng.uniform(-50.0, 50.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_local_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
        torques_local = wp.from_numpy(torques_local_np, dtype=wp.vec3f, device=device)
        positions_local = wp.from_numpy(positions_local_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(
            forces=forces_local, torques=torques_local, positions=positions_local, is_global=False
        )

        wrench_composer.compose_to_body_frame()

        # Local forces stay in body frame
        expected_forces = forces_local_np
        # Local torques: cross(local_pos, local_force) + local_torque
        expected_torques = np.cross(positions_local_np, forces_local_np) + torques_local_np

        out_force_np = wrench_composer.out_force_b.numpy()
        out_torque_np = wrench_composer.out_torque_b.numpy()

        assert np.allclose(out_force_np, expected_forces, atol=1e-3, rtol=1e-5)
        assert np.allclose(out_torque_np, expected_torques, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_at_link_origin_no_composed_torque(device: str):
    """Test that a global force applied at the link origin produces zero composed torque.

    The stored global_torque_w is cross(link_pos, F) (about world origin), but after
    compose the correction -cross(link_pos, F) cancels it out, giving zero net torque.
    """
    rng = np.random.default_rng(seed=16)
    num_envs, num_bodies = 5, 3

    link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_torch = torch.from_numpy(link_pos_np)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)

    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)

    # Position = link position
    positions_at_link = wp.from_numpy(link_pos_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces_global, positions=positions_at_link, is_global=True)

    # Global force stored unchanged
    expected_forces = forces_global_np
    global_force_np = wrench_composer.global_force_w.numpy()
    assert np.allclose(global_force_np, expected_forces, atol=1e-4, rtol=1e-5)

    # Stored torque is cross(link_pos, F), NOT zero
    expected_stored_torque = np.cross(link_pos_np, forces_global_np)
    global_torque_np = wrench_composer.global_torque_w.numpy()
    assert np.allclose(global_torque_np, expected_stored_torque, atol=1e-3, rtol=1e-4)

    # But composed output torque should be zero (correction cancels stored torque)
    wrench_composer.compose_to_body_frame()
    out_torque_np = wrench_composer.out_torque_b.numpy()
    expected_zero = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    assert np.allclose(out_torque_np, expected_zero, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_compose_with_changing_pose(device: str):
    """Test that compose_to_body_frame uses current quaternion, not quaternion at set time.

    This verifies the key behavior: global forces are stored in world frame and only
    rotated to body frame at compose time using the current body orientation.
    """
    rng = np.random.default_rng(seed=17)
    num_envs, num_bodies = 5, 3

    # Initial pose
    link_quat_np_1 = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch_1 = torch.from_numpy(link_quat_np_1)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch_1)
    wrench_composer = WrenchComposer(mock_asset)

    # Set global force
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global = wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces_global, is_global=True)

    # Set local force
    forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_local = wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces_local, is_global=False)

    # Compose with initial pose
    wrench_composer.compose_to_body_frame()
    expected_1 = quat_rotate_inv_np(link_quat_np_1, forces_global_np) + forces_local_np
    out_1 = wrench_composer.out_force_b.numpy()
    assert np.allclose(out_1, expected_1, atol=1e-3, rtol=1e-5)

    # Change the body orientation
    link_quat_np_2 = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    mock_asset.data.body_link_quat_w = torch.from_numpy(link_quat_np_2).to(device=device, dtype=torch.float32)

    # Compose again with new pose — should use new quaternion
    wrench_composer.compose_to_body_frame()
    expected_2 = quat_rotate_inv_np(link_quat_np_2, forces_global_np) + forces_local_np
    out_2 = wrench_composer.out_force_b.numpy()
    assert np.allclose(out_2, expected_2, atol=1e-3, rtol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_compose_with_changing_position(device: str):
    """Test that compose_to_body_frame dynamically adjusts torque based on current position.

    A global force with explicit position produces different composed torque when the body translates.
    """
    num_envs, num_bodies = 1, 1

    # Identity quaternion for simplicity
    link_pos_np_1 = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    link_pos_torch_1 = torch.from_numpy(link_pos_np_1)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch_1)
    wrench_composer = WrenchComposer(mock_asset)

    # Global force F = (0, 10, 0) with explicit position at world origin
    force_np = np.array([[[0.0, 10.0, 0.0]]], dtype=np.float32)
    position_np = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
    force_wp = wp.from_numpy(force_np, dtype=wp.vec3f, device=device)
    position_wp = wp.from_numpy(position_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=force_wp, positions=position_wp, is_global=True)

    # Position at origin → stored torque is cross((0,0,0), F) = 0
    stored_torque = wrench_composer.global_torque_w.numpy()
    assert np.allclose(stored_torque, np.zeros((1, 1, 3), dtype=np.float32), atol=1e-6)

    # Compose with link_pos = (1,0,0): correction = -cross((1,0,0), (0,10,0)) = -(0,0,10) = (0,0,-10)
    wrench_composer.compose_to_body_frame()
    out_torque_1 = wrench_composer.out_torque_b.numpy()
    expected_torque_1 = np.array([[[0.0, 0.0, -10.0]]], dtype=np.float32)
    assert np.allclose(out_torque_1, expected_torque_1, atol=1e-4), f"Expected {expected_torque_1}, got {out_torque_1}"

    # Move body to origin: link_pos = (0,0,0)
    mock_asset.data.body_link_pos_w = torch.zeros((1, 1, 3), dtype=torch.float32, device=device)

    # Recompose: correction = -cross((0,0,0), F) = 0, so composed torque = 0
    wrench_composer._dirty = True
    wrench_composer.compose_to_body_frame()
    out_torque_2 = wrench_composer.out_torque_b.numpy()
    expected_torque_2 = np.zeros((1, 1, 3), dtype=np.float32)
    assert np.allclose(out_torque_2, expected_torque_2, atol=1e-4), f"Expected {expected_torque_2}, got {out_torque_2}"


# ============================================================================
# add_raw_buffers_from Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_raw_buffers_from(device: str):
    """Test that add_raw_buffers_from merges all 5 buffers correctly."""
    rng = np.random.default_rng(seed=20)
    num_envs, num_bodies = 5, 3

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    composer_a = WrenchComposer(mock_asset)
    composer_b = WrenchComposer(mock_asset)

    # Add forces to composer_a (local + global without positions → at CoM)
    forces_a_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_a_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_a.add_forces_and_torques(
        forces=wp.from_numpy(forces_a_local_np, dtype=wp.vec3f, device=device), is_global=False
    )
    composer_a.add_forces_and_torques(
        forces=wp.from_numpy(forces_a_global_np, dtype=wp.vec3f, device=device), is_global=True
    )

    # Add forces to composer_b (local + global without positions → at CoM)
    forces_b_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_b_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_b.add_forces_and_torques(
        forces=wp.from_numpy(forces_b_local_np, dtype=wp.vec3f, device=device), is_global=False
    )
    composer_b.add_forces_and_torques(
        forces=wp.from_numpy(forces_b_global_np, dtype=wp.vec3f, device=device), is_global=True
    )

    # Merge b into a
    composer_a.add_raw_buffers_from(composer_b)

    # Global forces without positions go to global_force_at_com_w
    assert np.allclose(composer_a.global_force_at_com_w.numpy(), forces_a_global_np + forces_b_global_np, atol=1e-4)
    assert np.allclose(composer_a.local_force_b.numpy(), forces_a_local_np + forces_b_local_np, atol=1e-4)


# ============================================================================
# Dirty Flag / Warning Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_compose_randomized_cross_product_identity(device: str):
    """Test cross(P,F) - cross(link_pos,F) = cross(P - link_pos, F) with random inputs."""
    rng = np.random.default_rng(seed=30)
    num_envs, num_bodies = 5, 3

    for _ in range(5):
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
        positions = wp.from_numpy(positions_np, dtype=wp.vec3f, device=device)

        wrench_composer.add_forces_and_torques(forces=forces, positions=positions, is_global=True)
        wrench_composer.compose_to_body_frame()

        # Expected: out_torque_b = quat_rotate_inv(q, cross(P - link_pos, F))
        expected_torque_w = np.cross(positions_np - link_pos_np, forces_np)
        expected_torque_b = quat_rotate_inv_np(link_quat_np, expected_torque_w)

        out_torque_np = wrench_composer.out_torque_b.numpy()
        assert np.allclose(out_torque_np, expected_torque_b, atol=1e-3, rtol=1e-4), (
            f"Cross product identity failed.\nExpected:\n{expected_torque_b}\nGot:\n{out_torque_np}"
        )

        # Also verify force output
        expected_force_b = quat_rotate_inv_np(link_quat_np, forces_np)
        out_force_np = wrench_composer.out_force_b.numpy()
        assert np.allclose(out_force_np, expected_force_b, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_multiple_accumulated_global_forces_at_positions(device: str):
    """Test that 2+ global forces at different positions accumulate correctly."""
    rng = np.random.default_rng(seed=31)
    num_envs, num_bodies = 5, 3

    for _ in range(5):
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # First force at position P1
        f1_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        p1_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        wrench_composer.add_forces_and_torques(
            forces=wp.from_numpy(f1_np, dtype=wp.vec3f, device=device),
            positions=wp.from_numpy(p1_np, dtype=wp.vec3f, device=device),
            is_global=True,
        )

        # Second force at position P2
        f2_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        p2_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        wrench_composer.add_forces_and_torques(
            forces=wp.from_numpy(f2_np, dtype=wp.vec3f, device=device),
            positions=wp.from_numpy(p2_np, dtype=wp.vec3f, device=device),
            is_global=True,
        )

        # Verify stored buffers
        expected_stored_force = f1_np + f2_np
        expected_stored_torque = np.cross(p1_np, f1_np) + np.cross(p2_np, f2_np)
        assert np.allclose(wrench_composer.global_force_w.numpy(), expected_stored_force, atol=1e-3)
        assert np.allclose(wrench_composer.global_torque_w.numpy(), expected_stored_torque, atol=1e-3)

        # Compose and verify output
        wrench_composer.compose_to_body_frame()
        total_force = f1_np + f2_np
        corrected_torque_w = expected_stored_torque - np.cross(link_pos_np, total_force)
        expected_torque_b = quat_rotate_inv_np(link_quat_np, corrected_torque_w)
        expected_force_b = quat_rotate_inv_np(link_quat_np, total_force)

        assert np.allclose(wrench_composer.out_torque_b.numpy(), expected_torque_b, atol=1e-3, rtol=1e-4)
        assert np.allclose(wrench_composer.out_force_b.numpy(), expected_force_b, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_overwrites_previous_positional_torque(device: str):
    """Test that set_forces_and_torques replaces (not accumulates) stored positional torque."""
    rng = np.random.default_rng(seed=32)
    num_envs, num_bodies = 5, 3

    for _ in range(5):
        mock_asset = MockRigidObject(num_envs, num_bodies, device)
        wrench_composer = WrenchComposer(mock_asset)

        # First set
        f1_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        p1_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        wrench_composer.set_forces_and_torques(
            forces=wp.from_numpy(f1_np, dtype=wp.vec3f, device=device),
            positions=wp.from_numpy(p1_np, dtype=wp.vec3f, device=device),
            is_global=True,
        )

        # Verify first set stored correctly
        expected_torque_1 = np.cross(p1_np, f1_np)
        assert np.allclose(wrench_composer.global_torque_w.numpy(), expected_torque_1, atol=1e-3)
        assert np.allclose(wrench_composer.global_force_w.numpy(), f1_np, atol=1e-3)

        # Second set should overwrite, not accumulate
        f2_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        p2_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        wrench_composer.set_forces_and_torques(
            forces=wp.from_numpy(f2_np, dtype=wp.vec3f, device=device),
            positions=wp.from_numpy(p2_np, dtype=wp.vec3f, device=device),
            is_global=True,
        )

        # Should be ONLY the second set's values
        expected_torque_2 = np.cross(p2_np, f2_np)
        assert np.allclose(wrench_composer.global_torque_w.numpy(), expected_torque_2, atol=1e-3), (
            "set_forces_and_torques should overwrite, not accumulate positional torque"
        )
        assert np.allclose(wrench_composer.global_force_w.numpy(), f2_np, atol=1e-3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_raw_buffers_from_with_positional_torques(device: str):
    """Test that add_raw_buffers_from correctly merges composers with positional torques."""
    rng = np.random.default_rng(seed=33)
    num_envs, num_bodies = 5, 3

    link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_torch = torch.from_numpy(link_pos_np)
    link_quat_torch = torch.from_numpy(link_quat_np)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
    composer_a = WrenchComposer(mock_asset)
    composer_b = WrenchComposer(mock_asset)

    # Composer A: global force at position
    fa_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    pa_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_a.add_forces_and_torques(
        forces=wp.from_numpy(fa_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(pa_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Composer B: global force at position
    fb_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    pb_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_b.add_forces_and_torques(
        forces=wp.from_numpy(fb_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(pb_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Merge B into A
    composer_a.add_raw_buffers_from(composer_b)

    # Verify merged raw buffers
    expected_force = fa_np + fb_np
    expected_torque = np.cross(pa_np, fa_np) + np.cross(pb_np, fb_np)
    assert np.allclose(composer_a.global_force_w.numpy(), expected_force, atol=1e-3)
    assert np.allclose(composer_a.global_torque_w.numpy(), expected_torque, atol=1e-3)

    # Compose and verify the output uses combined force for correction
    composer_a.compose_to_body_frame()
    total_force = fa_np + fb_np
    corrected_torque_w = expected_torque - np.cross(link_pos_np, total_force)
    expected_torque_b = quat_rotate_inv_np(link_quat_np, corrected_torque_w)
    expected_force_b = quat_rotate_inv_np(link_quat_np, total_force)

    assert np.allclose(composer_a.out_torque_b.numpy(), expected_torque_b, atol=1e-3, rtol=1e-4)
    assert np.allclose(composer_a.out_force_b.numpy(), expected_force_b, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_torque_only_no_correction(device: str):
    """Test that global torque without forces gets no correction (cross(link_pos, 0) = 0)."""
    rng = np.random.default_rng(seed=34)
    num_envs, num_bodies = 5, 3

    for _ in range(5):
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        torques_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        wrench_composer.add_forces_and_torques(
            torques=wp.from_numpy(torques_np, dtype=wp.vec3f, device=device), is_global=True
        )

        wrench_composer.compose_to_body_frame()

        # No forces → correction is -cross(link_pos, 0) = 0
        # Expected: out_torque_b = quat_rotate_inv(q, T)
        expected_torque_b = quat_rotate_inv_np(link_quat_np, torques_np)
        out_torque_np = wrench_composer.out_torque_b.numpy()
        assert np.allclose(out_torque_np, expected_torque_b, atol=1e-3, rtol=1e-4)

        # Forces should be zero
        expected_zero = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        assert np.allclose(wrench_composer.out_force_b.numpy(), expected_zero, atol=1e-6)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_compose_with_changing_position_and_quaternion(device: str):
    """Test that compose adapts when both position and quaternion change simultaneously."""
    rng = np.random.default_rng(seed=35)
    num_envs, num_bodies = 5, 3

    # Initial pose
    link_pos_np_1 = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np_1 = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_pos_torch_1 = torch.from_numpy(link_pos_np_1)
    link_quat_torch_1 = torch.from_numpy(link_quat_np_1)

    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch_1, link_quat=link_quat_torch_1)
    wrench_composer = WrenchComposer(mock_asset)

    # Add global force at position
    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    wrench_composer.add_forces_and_torques(
        forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Compose with pose 1
    wrench_composer.compose_to_body_frame()
    stored_torque = np.cross(positions_np, forces_np)
    corrected_1 = stored_torque - np.cross(link_pos_np_1, forces_np)
    expected_torque_1 = quat_rotate_inv_np(link_quat_np_1, corrected_1)
    expected_force_1 = quat_rotate_inv_np(link_quat_np_1, forces_np)

    assert np.allclose(wrench_composer.out_torque_b.numpy(), expected_torque_1, atol=1e-3, rtol=1e-4)
    assert np.allclose(wrench_composer.out_force_b.numpy(), expected_force_1, atol=1e-3, rtol=1e-4)

    # Change BOTH position and quaternion
    link_pos_np_2 = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    link_quat_np_2 = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    mock_asset.data.body_link_pos_w = torch.from_numpy(link_pos_np_2).to(device=device, dtype=torch.float32)
    mock_asset.data.body_link_quat_w = torch.from_numpy(link_quat_np_2).to(device=device, dtype=torch.float32)

    # Compose with pose 2
    wrench_composer._dirty = True
    wrench_composer.compose_to_body_frame()
    corrected_2 = stored_torque - np.cross(link_pos_np_2, forces_np)
    expected_torque_2 = quat_rotate_inv_np(link_quat_np_2, corrected_2)
    expected_force_2 = quat_rotate_inv_np(link_quat_np_2, forces_np)

    assert np.allclose(wrench_composer.out_torque_b.numpy(), expected_torque_2, atol=1e-3, rtol=1e-4)
    assert np.allclose(wrench_composer.out_force_b.numpy(), expected_force_2, atol=1e-3, rtol=1e-4)

    # Verify they differ (different pose → different output)
    assert not np.allclose(expected_torque_1, expected_torque_2, atol=1e-3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_large_origin_offset_precision(device: str):
    """Test that compose correction doesn't lose unacceptable precision at large world offsets.

    The compose kernel computes: cross(P, F) - cross(link_pos, F) = cross(P - link_pos, F).
    When link_pos is large (e.g., 2000), both cross products are ~O(20000) but nearly cancel
    to ~O(10), risking catastrophic cancellation in float32.

    This test compares the kernel's result against a direct cross(P - link_pos, F) reference
    (which has no cancellation) across increasing world offsets.
    """
    rng = np.random.default_rng(seed=50)
    num_envs, num_bodies = 10, 3

    offsets = [0.0, 100.0, 1000.0, 2000.0, 5000.0]

    for world_offset in offsets:
        # Random small perturbations around the offset
        link_pos_np = rng.uniform(-1.0, 1.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_pos_np[..., 0] += world_offset  # shift along X

        # Force application point: 1m relative offset from link
        relative_offset_np = rng.uniform(-2.0, 2.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions_np = link_pos_np + relative_offset_np

        # Random force and quaternion
        forces_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))

        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        wrench_composer.add_forces_and_torques(
            forces=wp.from_numpy(forces_np, dtype=wp.vec3f, device=device),
            positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
            is_global=True,
        )
        wrench_composer.compose_to_body_frame()
        actual_torque_b = wrench_composer.out_torque_b.numpy()

        # Reference: compute cross(P - link_pos, F) directly (no cancellation)
        reference_torque_w = np.cross(relative_offset_np, forces_np)
        reference_torque_b = quat_rotate_inv_np(link_quat_np, reference_torque_w)

        # With atol=0.1, we accept ~0.01 rad/s error on a 1kg cube — acceptable for robotics
        # but flags gross precision issues from catastrophic cancellation.
        assert np.allclose(actual_torque_b, reference_torque_b, atol=0.1, rtol=0.0), (
            f"Precision loss at world offset {world_offset}:\n"
            f"  max absolute error: {np.max(np.abs(actual_torque_b - reference_torque_b)):.6f}\n"
            f"  mean absolute error: {np.mean(np.abs(actual_torque_b - reference_torque_b)):.6f}"
        )


# ============================================================================
# Dirty Flag / Warning Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dirty_flag_warns_on_stale_output_access(device: str):
    """Test that accessing output properties without calling compose_to_body_frame() emits a warning."""
    num_envs, num_bodies = 2, 1
    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    forces_np = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces, is_global=False)

    # Accessing output without compose_to_body_frame() should warn
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        _ = wrench_composer.out_force_b
        assert len(caught) == 1
        assert "compose_to_body_frame()" in str(caught[0].message)
        assert "raw buffer properties" in str(caught[0].message)

    # Second access should NOT warn (compose was triggered by first access)
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        _ = wrench_composer.out_force_b
        assert len(caught) == 0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_no_warning_after_explicit_compose(device: str):
    """Test that no warning is emitted when compose_to_body_frame() is called before accessing output."""
    num_envs, num_bodies = 2, 1
    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    forces_np = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces, is_global=False)

    wrench_composer.compose_to_body_frame()

    # No warning expected
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        _ = wrench_composer.out_force_b
        _ = wrench_composer.out_torque_b
        _ = wrench_composer.out_force_b_as_torch
        _ = wrench_composer.out_torque_b_as_torch
        assert len(caught) == 0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_dirty_flag_reset_after_reset(device: str):
    """Test that reset clears the dirty flag so no warning is emitted on output access."""
    num_envs, num_bodies = 2, 1
    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    forces_np = np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces, is_global=False)

    wrench_composer.reset()

    # No warning expected after reset (dirty=False, output is zeroed)
    with warnings_module.catch_warnings(record=True) as caught:
        warnings_module.simplefilter("always")
        out = wrench_composer.out_force_b.numpy()
        assert len(caught) == 0
    # Output should be zero after reset
    assert np.allclose(out, np.zeros((num_envs, num_bodies, 3), dtype=np.float32))


# ============================================================================
# Global Force at CoM (No Position) Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_no_position_zero_torque_at_offset(device: str):
    """Test that a global force without positions produces zero torque even at large body offsets.

    This is the key regression test for the bug where global forces without positions
    would produce spurious torque proportional to the body's distance from the world origin.
    """
    rng = np.random.default_rng(seed=40)
    num_envs, num_bodies = 5, 3

    for world_offset in [0.0, 100.0, 1000.0, 2000.0]:
        link_pos_np = rng.uniform(-1.0, 1.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_pos_np[..., 0] += world_offset
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        forces_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
        wrench_composer.add_forces_and_torques(forces=forces, is_global=True)

        wrench_composer.compose_to_body_frame()

        # Force at CoM → no positional torque
        out_torque_np = wrench_composer.out_torque_b.numpy()
        expected_zero = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
        assert np.allclose(out_torque_np, expected_zero, atol=1e-4), (
            f"Spurious torque at world offset {world_offset}:\n"
            f"  max torque magnitude: {np.max(np.abs(out_torque_np)):.6f}\n"
            f"  Expected zero torque for global force without positions."
        )

        # Force should be correctly rotated to body frame
        expected_force_b = quat_rotate_inv_np(link_quat_np, forces_np)
        out_force_np = wrench_composer.out_force_b.numpy()
        assert np.allclose(out_force_np, expected_force_b, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_at_com_mixed_with_positional(device: str):
    """Test that global forces at CoM and global forces with positions compose correctly together."""
    rng = np.random.default_rng(seed=41)
    num_envs, num_bodies = 5, 3

    for _ in range(5):
        link_pos_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
        link_pos_torch = torch.from_numpy(link_pos_np)
        link_quat_torch = torch.from_numpy(link_quat_np)

        mock_asset = MockRigidObject(num_envs, num_bodies, device, link_pos=link_pos_torch, link_quat=link_quat_torch)
        wrench_composer = WrenchComposer(mock_asset)

        # Force 1: global with position (goes to global_force_w)
        f1_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        p1_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        wrench_composer.add_forces_and_torques(
            forces=wp.from_numpy(f1_np, dtype=wp.vec3f, device=device),
            positions=wp.from_numpy(p1_np, dtype=wp.vec3f, device=device),
            is_global=True,
        )

        # Force 2: global without position (goes to global_force_at_com_w)
        f2_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        wrench_composer.add_forces_and_torques(
            forces=wp.from_numpy(f2_np, dtype=wp.vec3f, device=device),
            is_global=True,
        )

        wrench_composer.compose_to_body_frame()

        # Total force in world frame: f1 (positional) + f2 (at CoM)
        total_force_w = f1_np + f2_np
        expected_force_b = quat_rotate_inv_np(link_quat_np, total_force_w)

        # Torque: only f1 participates in correction
        stored_torque = np.cross(p1_np, f1_np)
        corrected_torque_w = stored_torque - np.cross(link_pos_np, f1_np)
        expected_torque_b = quat_rotate_inv_np(link_quat_np, corrected_torque_w)

        out_force_np = wrench_composer.out_force_b.numpy()
        out_torque_np = wrench_composer.out_torque_b.numpy()

        assert np.allclose(out_force_np, expected_force_b, atol=1e-3, rtol=1e-4)
        assert np.allclose(out_torque_np, expected_torque_b, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_add_raw_buffers_from_with_force_at_com(device: str):
    """Test that add_raw_buffers_from correctly merges the global_force_at_com_w buffer."""
    rng = np.random.default_rng(seed=42)
    num_envs, num_bodies = 5, 3

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    composer_a = WrenchComposer(mock_asset)
    composer_b = WrenchComposer(mock_asset)

    # Composer A: global force without position (at CoM)
    fa_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_a.add_forces_and_torques(forces=wp.from_numpy(fa_np, dtype=wp.vec3f, device=device), is_global=True)

    # Composer B: global force without position (at CoM)
    fb_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer_b.add_forces_and_torques(forces=wp.from_numpy(fb_np, dtype=wp.vec3f, device=device), is_global=True)

    # Merge B into A
    composer_a.add_raw_buffers_from(composer_b)

    # Verify merged global_force_at_com_w buffer
    assert np.allclose(composer_a.global_force_at_com_w.numpy(), fa_np + fb_np, atol=1e-4)


# ============================================================================
# API Behavior Tests
# ============================================================================


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_active_property_lifecycle(device: str):
    """Test active property transitions: False → True after add → False after reset."""
    num_envs, num_bodies = 4, 2
    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    # Initially inactive
    assert wrench_composer.active is False

    # Active after adding forces
    forces_np = np.ones((num_envs, num_bodies, 3), dtype=np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)
    wrench_composer.add_forces_and_torques(forces=forces)
    assert wrench_composer.active is True

    # Inactive after full reset
    wrench_composer.reset()
    assert wrench_composer.active is False


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_clears_all_buffers(device: str):
    """Test that set clears all 5 input buffers before writing new values.

    1. set(forces=F1, positions=P1, is_global=True) → writes global_force_w + global_torque_w
    2. set(forces=F2, is_global=True) (no positions) → writes global_force_at_com_w, clears global_force_w
    3. Verify global_force_w is zero (cleared), global_force_at_com_w has F2
    4. Compose and verify output includes only F2
    """
    rng = np.random.default_rng(seed=100)
    num_envs, num_bodies = 4, 2

    link_quat_np = random_unit_quaternion_np(rng, (num_envs, num_bodies))
    link_quat_torch = torch.from_numpy(link_quat_np)
    mock_asset = MockRigidObject(num_envs, num_bodies, device, link_quat=link_quat_torch)
    wrench_composer = WrenchComposer(mock_asset)
    zeros = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)

    # Step 1: set with positions → global_force_w + global_torque_w
    f1_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    p1_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    wrench_composer.set_forces_and_torques(
        forces=wp.from_numpy(f1_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(p1_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )
    assert np.allclose(wrench_composer.global_force_w.numpy(), f1_np, atol=1e-4)

    # Step 2: set without positions → global_force_at_com_w; should clear global_force_w
    f2_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    wrench_composer.set_forces_and_torques(
        forces=wp.from_numpy(f2_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Step 3: global_force_w cleared, only global_force_at_com_w has data
    assert np.allclose(wrench_composer.global_force_w.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.global_torque_w.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.global_force_at_com_w.numpy(), f2_np, atol=1e-4)
    assert np.allclose(wrench_composer.local_force_b.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.local_torque_b.numpy(), zeros, atol=1e-7)

    # Step 4: compose — only F2 contributes (no stale F1)
    wrench_composer.compose_to_body_frame()
    expected_force_b = quat_rotate_inv_np(link_quat_np, f2_np)
    assert np.allclose(wrench_composer.out_force_b.numpy(), expected_force_b, atol=1e-3, rtol=1e-4)
    # No positional torque — force at CoM
    assert np.allclose(wrench_composer.out_torque_b.numpy(), zeros, atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_partial_reset_preserves_other_envs(device: str):
    """Test that reset(env_ids=[0, 2]) zeros envs 0 and 2 but preserves env 1."""
    rng = np.random.default_rng(seed=101)
    num_envs, num_bodies = 3, 2

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    # Add forces to all 3 envs (local + global + global at CoM)
    forces_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_global_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces_at_com_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_local_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)

    wrench_composer.add_forces_and_torques(
        forces=wp.from_numpy(forces_local_np, dtype=wp.vec3f, device=device),
        torques=wp.from_numpy(torques_local_np, dtype=wp.vec3f, device=device),
        is_global=False,
    )
    wrench_composer.add_forces_and_torques(
        forces=wp.from_numpy(forces_global_np, dtype=wp.vec3f, device=device),
        positions=wp.from_numpy(positions_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )
    wrench_composer.add_forces_and_torques(
        forces=wp.from_numpy(forces_at_com_np, dtype=wp.vec3f, device=device),
        is_global=True,
    )

    # Reset envs 0 and 2 only
    reset_ids = wp.array([0, 2], dtype=wp.int32, device=device)
    wrench_composer.reset(env_ids=reset_ids)

    zeros = np.zeros((num_bodies, 3), dtype=np.float32)

    # All 7 buffers: envs 0 and 2 should be zeroed
    for buf in [
        wrench_composer.global_force_w,
        wrench_composer.global_torque_w,
        wrench_composer.global_force_at_com_w,
        wrench_composer.local_force_b,
        wrench_composer.local_torque_b,
        wrench_composer._out_force_b,
        wrench_composer._out_torque_b,
    ]:
        buf_np = buf.numpy()
        assert np.allclose(buf_np[0], zeros, atol=1e-7), "Env 0 not zeroed in buffer"
        assert np.allclose(buf_np[2], zeros, atol=1e-7), "Env 2 not zeroed in buffer"

    # Env 1 should retain its data
    assert np.allclose(wrench_composer.local_force_b.numpy()[1], forces_local_np[1], atol=1e-4)
    assert np.allclose(wrench_composer.local_torque_b.numpy()[1], torques_local_np[1], atol=1e-4)
    assert np.allclose(wrench_composer.global_force_w.numpy()[1], forces_global_np[1], atol=1e-4)
    expected_global_torque_1 = np.cross(positions_np[1], forces_global_np[1])
    assert np.allclose(wrench_composer.global_torque_w.numpy()[1], expected_global_torque_1, atol=1e-3)
    assert np.allclose(wrench_composer.global_force_at_com_w.numpy()[1], forces_at_com_np[1], atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_torch_tensor_input_conversion(device: str):
    """Test that add_forces_and_torques correctly handles torch.Tensor inputs."""
    rng = np.random.default_rng(seed=102)
    num_envs, num_bodies = 4, 2

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    torques_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    positions_np = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)

    # Pass torch tensors instead of warp arrays
    forces_torch = torch.from_numpy(forces_np).to(device=device)
    torques_torch = torch.from_numpy(torques_np).to(device=device)
    positions_torch = torch.from_numpy(positions_np).to(device=device)

    wrench_composer.add_forces_and_torques(
        forces=forces_torch, torques=torques_torch, positions=positions_torch, is_global=False
    )

    # Verify buffers contain correct values
    assert np.allclose(wrench_composer.local_force_b.numpy(), forces_np, atol=1e-4)
    expected_torque = torques_np + np.cross(positions_np, forces_np)
    assert np.allclose(wrench_composer.local_torque_b.numpy(), expected_torque, atol=1e-3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_resolve_indices_slice_none(device: str):
    """Test that slice(None) selects all envs/bodies."""
    rng = np.random.default_rng(seed=103)
    num_envs, num_bodies = 4, 2

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)

    wrench_composer.add_forces_and_torques(forces=forces, env_ids=slice(None), body_ids=slice(None))

    assert np.allclose(wrench_composer.local_force_b.numpy(), forces_np, atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_resolve_indices_invalid_slice_raises(device: str):
    """Test that a non-None slice for env_ids raises ValueError."""
    num_envs, num_bodies = 4, 2
    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    forces_np = np.ones((2, num_bodies, 3), dtype=np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)

    with pytest.raises(ValueError, match="Doesn't support slice input"):
        wrench_composer.add_forces_and_torques(forces=forces, env_ids=slice(0, 5))

    with pytest.raises(ValueError, match="Doesn't support slice input"):
        wrench_composer.add_forces_and_torques(forces=forces, body_ids=slice(0, 2))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_set_local_forces(device: str):
    """Test that set_forces_and_torques with is_global=False writes to local buffers only."""
    rng = np.random.default_rng(seed=104)
    num_envs, num_bodies = 4, 2

    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    forces_np = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
    forces = wp.from_numpy(forces_np, dtype=wp.vec3f, device=device)

    wrench_composer.set_forces_and_torques(forces=forces, is_global=False)

    # Local buffer should have the forces
    assert np.allclose(wrench_composer.local_force_b.numpy(), forces_np, atol=1e-4)

    # Global buffers should remain zero
    zeros = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    assert np.allclose(wrench_composer.global_force_w.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.global_torque_w.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.global_force_at_com_w.numpy(), zeros, atol=1e-7)

    # Compose and verify output equals local force (identity quat)
    wrench_composer.compose_to_body_frame()
    assert np.allclose(wrench_composer.out_force_b.numpy(), forces_np, atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_both_none_is_noop(device: str):
    """Test that add_forces_and_torques with forces=None and torques=None is a no-op."""
    num_envs, num_bodies = 4, 2
    mock_asset = MockRigidObject(num_envs, num_bodies, device)
    wrench_composer = WrenchComposer(mock_asset)

    wrench_composer.add_forces_and_torques(forces=None, torques=None)

    # Should remain inactive
    assert wrench_composer.active is False

    # All buffers should be zero
    zeros = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    assert np.allclose(wrench_composer.local_force_b.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.local_torque_b.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.global_force_w.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.global_torque_w.numpy(), zeros, atol=1e-7)
    assert np.allclose(wrench_composer.global_force_at_com_w.numpy(), zeros, atol=1e-7)
