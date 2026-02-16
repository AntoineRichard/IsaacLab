# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for wrench composer with rigid objects.

These tests validate that global forces/torques remain invariant under body rotation
(PR #4604 fix: store composed wrenches in mixed frame, apply with is_global=True).
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def generate_cubes_scene(
    num_cubes: int = 1,
    height: float = 1.0,
    device: str = "cuda:0",
) -> tuple[RigidObject, torch.Tensor]:
    """Generate a scene with the provided number of cubes."""
    origins = torch.tensor([(i * 1.0, 0, height) for i in range(num_cubes)]).to(device)
    for i, origin in enumerate(origins):
        sim_utils.create_prim(f"/World/Table_{i}", "Xform", translation=origin)

    spawn_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    )

    cube_object_cfg = RigidObjectCfg(
        prim_path="/World/Table_.*/Object",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, height)),
    )
    cube_object = RigidObject(cfg=cube_object_cfg)
    return cube_object, origins


N_STEPS = 100
FORCE_MAGNITUDE = 10.0
TORQUE_MAGNITUDE = 1.0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_invariant_under_rotation(device):
    """Test that a permanent global force produces the same acceleration before and after body rotation.

    A global +X force is applied. After 100 steps the body is rotated 180deg about Z.
    The acceleration (delta_v per phase) should be the same in both phases because the
    force is in the global frame and should not rotate with the body.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")
        mass = cube_object.root_physx_view.get_masses()[0].item()

        # Apply permanent global force along +X at CoM
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=body_ids, is_global=True,
        )

        # Phase 1: run N_STEPS
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase1 = cube_object.data.root_lin_vel_w[0].clone()

        # Rotate body 180deg about Z (quat wxyz = [0, 0, 0, 1]) while keeping velocity
        root_pose = cube_object.data.root_state_w[0, :7].clone().unsqueeze(0)
        root_pose[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # 180deg about Z
        cube_object.write_root_pose_to_sim(root_pose)

        # Phase 2: run N_STEPS more
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase2 = cube_object.data.root_lin_vel_w[0].clone()

        # Acceleration should be same in both phases: delta_v_phase2 ≈ delta_v_phase1
        delta_v_phase1 = vel_after_phase1[0].item()  # vx after phase 1
        delta_v_phase2 = vel_after_phase2[0].item() - vel_after_phase1[0].item()  # vx gained in phase 2

        expected_dv = FORCE_MAGNITUDE / mass * sim.cfg.dt * N_STEPS

        torch.testing.assert_close(
            torch.tensor(delta_v_phase1), torch.tensor(expected_dv), rtol=0.1, atol=0.01,
        )
        torch.testing.assert_close(
            torch.tensor(delta_v_phase2), torch.tensor(expected_dv), rtol=0.1, atol=0.01,
        )

        # Y and Z velocity should remain ~0
        assert abs(vel_after_phase2[1].item()) < 0.5, f"Unexpected Y velocity: {vel_after_phase2[1].item()}"
        assert abs(vel_after_phase2[2].item()) < 0.5, f"Unexpected Z velocity: {vel_after_phase2[2].item()}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_local_force_follows_rotation(device):
    """Test that a permanent local force rotates with the body.

    A local +X force is applied. After 100 steps the body is rotated 180deg about Z.
    Since local +X is now world -X, the force should decelerate the body back towards zero velocity.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Apply permanent local force along body +X
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE
        torques = torch.zeros(1, len(body_ids), 3, device=device)

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=body_ids, is_global=False,
        )

        # Phase 1: run N_STEPS — object accelerates along world +X
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase1 = cube_object.data.root_lin_vel_w[0].clone()
        assert vel_after_phase1[0].item() > 1.0, "Object should be moving in +X"

        # Rotate body 180deg about Z while keeping velocity
        root_pose = cube_object.data.root_state_w[0, :7].clone().unsqueeze(0)
        root_pose[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)  # 180deg about Z
        cube_object.write_root_pose_to_sim(root_pose)

        # Phase 2: run N_STEPS — local +X is now world -X, so force decelerates
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        vel_after_phase2 = cube_object.data.root_lin_vel_w[0].clone()

        # Velocity should be approximately zero: decelerated by the same amount as it accelerated
        torch.testing.assert_close(
            vel_after_phase2[0], torch.tensor(0.0, device=device), atol=0.5, rtol=0.0,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_force_at_offset_generates_torque(device):
    """Test that a global force applied at an offset from CoM generates the expected torque.

    A global +X force applied at +1m Y offset from CoM should produce:
    - Linear acceleration in +X
    - Angular acceleration about -Z (from cross product: (0,1,0) × (10,0,0) = (0,0,-10))
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Force at offset: +1m in Y from CoM (global frame)
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        forces[..., 0] = FORCE_MAGNITUDE  # +X force

        torques = torch.zeros(1, len(body_ids), 3, device=device)

        # Position offset: CoM position + 1m in Y (global frame)
        com_pos = cube_object.data.body_com_pos_w[:, body_ids, :3].clone()
        positions = com_pos.clone()
        positions[..., 1] += 1.0  # +1m Y offset

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, positions=positions, body_ids=body_ids, is_global=True,
        )

        # Run 50 steps
        for _ in range(50):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        lin_vel = cube_object.data.root_lin_vel_w[0]
        ang_vel = cube_object.data.root_ang_vel_w[0]

        # Linear velocity in +X should be positive
        assert lin_vel[0].item() > 0.1, f"Expected positive X velocity, got {lin_vel[0].item()}"

        # Angular velocity about Z should be negative (cross product: r × F, r=(0,1,0), F=(10,0,0) -> (0,0,-10))
        assert ang_vel[2].item() < -0.1, f"Expected negative Z angular velocity, got {ang_vel[2].item()}"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_global_torque_invariant_under_rotation(device):
    """Test that a permanent global torque produces the same angular acceleration before and after rotation.

    A global +Z torque is applied. After 100 steps the body is rotated 90deg about X.
    The angular acceleration (delta_omega per phase) about Z should be the same in both phases
    because the torque is in the global frame.
    """
    with build_simulation_context(device=device, gravity_enabled=False, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        cube_object, _ = generate_cubes_scene(num_cubes=1, device=device)

        sim.reset()

        body_ids, _ = cube_object.find_bodies(".*")

        # Apply permanent global torque about +Z
        forces = torch.zeros(1, len(body_ids), 3, device=device)
        torques = torch.zeros(1, len(body_ids), 3, device=device)
        torques[..., 2] = TORQUE_MAGNITUDE

        cube_object.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=body_ids, is_global=True,
        )

        # Phase 1: run N_STEPS
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        omega_z_after_phase1 = cube_object.data.root_ang_vel_w[0, 2].clone().item()

        # Rotate body 90deg about X and zero out velocities so phase 2 starts from rest
        # (avoids gyroscopic cross-coupling at high omega)
        root_pose = cube_object.data.root_state_w[0, :7].clone().unsqueeze(0)
        root_pose[0, 3:7] = torch.tensor([0.7071, 0.7071, 0.0, 0.0], device=device)
        cube_object.write_root_pose_to_sim(root_pose)
        cube_object.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

        # Phase 2: run N_STEPS from rest with different body orientation
        for _ in range(N_STEPS):
            cube_object.write_data_to_sim()
            sim.step()
            cube_object.update(sim.cfg.dt)

        omega_z_after_phase2 = cube_object.data.root_ang_vel_w[0, 2].clone().item()

        # Both phases start from rest — angular acceleration about Z should be the same
        torch.testing.assert_close(
            torch.tensor(omega_z_after_phase1), torch.tensor(omega_z_after_phase2), rtol=0.1, atol=0.01,
        )
