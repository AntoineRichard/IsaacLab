# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a scripted Franka FR3 drop three wrapped candies into a sewn stocking.

.. code-block:: bash

    uv run isaaclab demo christmas-atelier
    uv run --extra ovrtx isaaclab demo christmas-atelier --viz newton_rtx

Use ``--viz none`` for a headless run. Both visualizers use the same USD scene and Newton VBD simulation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import newton
import newton.ik as ik
import numpy as np
import torch
import warp as wp
from isaaclab_newton.cloner import copy_newton_clone_source, newton_builder_world_hook
from isaaclab_newton.physics import NewtonCfg, NewtonCollisionPipelineCfg, NewtonSoftContactCfg, VBDSolver, VBDSolverCfg
from isaaclab_visualizers.newton import NewtonGLBackendCfg
from newton.solvers import SolverVBD

from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
FPS = 60
SUBSTEPS = 12
PARTICLE_RADIUS = 0.002
TCP_OFFSET = wp.vec3(0.0, 0.0, 0.1034)
FRANKA_ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))
FRANKA_FINGER_JOINTS = ("fr3_finger_joint1", "fr3_finger_joint2")
FRANKA_JOINTS = (*FRANKA_ARM_JOINTS, *FRANKA_FINGER_JOINTS)
FRANKA_HOME = (0.0, -0.5, 0.0, -2.1, 0.0, 1.6, math.pi / 4, 0.04, 0.04)


@configclass
class AtelierSceneCfg(InteractiveSceneCfg):
    """One FR3 and the authored stocking workcell."""

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "christmas_atelier_fr3.usd")),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.72), joint_pos=dict(zip(FRANKA_JOINTS, FRANKA_HOME, strict=True))
        ),
        actuators={"drives": ImplicitActuatorCfg(joint_names_expr=list(FRANKA_JOINTS), stiffness=None, damping=None)},
    )
    workcell = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Atelier",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "christmas_atelier.usda")),
    )
    lighting = AssetBaseCfg(prim_path="/World/StudioLight", spawn=sim_utils.DomeLightCfg(intensity=500.0))


class _AtelierVBDSolver(VBDSolver):
    """Initialize the authored posture and solve the loose candies after the robot."""

    @classmethod
    def prepare_builder(cls, cfg: VBDSolverCfg, builder: newton.ModelBuilder) -> None:
        super().prepare_builder(cfg, builder)
        candy_bodies = {
            index for index, label in enumerate(builder.body_label) if label.rsplit("/", 1)[-1].startswith("candy_")
        }
        # The candy contact solve sees the updated fingertip poses when these groups run separately.
        body_groups = []
        for group in builder.body_color_groups:
            robot_bodies = [body for body in group if body not in candy_bodies]
            if robot_bodies:
                body_groups.append(np.asarray(robot_bodies, dtype=np.int32))
        body_groups.append(np.asarray(sorted(candy_bodies), dtype=np.int32))
        builder.body_color_groups = body_groups

    def build(self, model: newton.Model, mode: wp.DeterministicMode, env_template: str | None) -> SolverVBD:
        # VBD captures joint rest transforms when constructed, before Isaac Lab initializes the live state.
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        return super().build(model, mode, env_template)

    def create_collision_pipeline(self, **kwargs) -> newton.CollisionPipeline:
        # The Newton revision used by this demo names the detection-only distance a gap.
        kwargs["soft_contact_gap"] = kwargs.pop("soft_contact_margin")
        return super().create_collision_pipeline(**kwargs)


@configclass
class AtelierVBDSolverCfg(VBDSolverCfg):
    """VBD settings for the stocking mesh and individual candy bodies."""

    class_type: type = _AtelierVBDSolver
    particle_self_contact_radius: float | None = None
    particle_self_contact_gap: float = 0.003
    particle_collision_detection_interval: int | None = None
    particle_enable_tile_solve: bool = True


def _franka_joint_ids(builder: newton.ModelBuilder) -> list[int]:
    """Resolve joint labels within the imported FR3 and Franka hand."""
    by_name = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(builder.joint_label)}
    missing = set(FRANKA_JOINTS) - by_name.keys()
    if missing:
        raise ValueError(f"The Franka URDF is missing joints: {', '.join(sorted(missing))}")
    return [by_name[name] for name in FRANKA_JOINTS]


def _solve_waypoints(
    builder: newton.ModelBuilder, positions: np.ndarray, fingers: np.ndarray, finger_coords: list[int], device: str
) -> np.ndarray:
    """Solve IK on a separate reference model; never overwrite simulated body poses."""
    model = builder.finalize(device=device)
    tcp = next(index for index, label in enumerate(model.body_label) if label.endswith("fr3_hand"))
    position = ik.IKObjectivePosition(
        link_index=tcp,
        link_offset=TCP_OFFSET,
        target_positions=wp.array([positions[0]], dtype=wp.vec3, device=device),
    )
    rotation = ik.IKObjectiveRotation(
        link_index=tcp,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([[1.0, 0.0, 0.0, 0.0]], dtype=wp.vec4, device=device),
    )
    limits = ik.IKObjectiveJointLimit(
        joint_limit_lower=model.joint_limit_lower, joint_limit_upper=model.joint_limit_upper, weight=10.0
    )
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[position, rotation, limits],
        lambda_initial=0.1,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    q = wp.clone(model.joint_q).reshape((1, model.joint_coord_count))
    state = model.state()
    targets = []
    for target, opening in zip(positions, fingers, strict=True):
        position.set_target_position(0, wp.vec3(*map(float, target)))
        solver.step(q, q, iterations=80)
        values = q.numpy().ravel()
        values[finger_coords] = opening
        q.assign(values.reshape(1, -1))
        newton.eval_fk(model, q.flatten(), model.joint_qd, state)
        error = np.linalg.norm(
            np.array(wp.transform_point(wp.transform(*state.body_q.numpy()[tcp]), TCP_OFFSET)) - target
        )
        if error > 0.003:
            raise RuntimeError(f"Unreachable TCP waypoint {target.tolist()}: IK error {error:.4f} m")
        targets.append(values.copy())
    return np.array(targets, dtype=np.float32)


def _choreography(
    pick_positions: list[tuple[float, float, float]], mouth: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan the three pickups and releases, with a hold before each lift and after each drop."""
    home = (0.38, -0.02, 1.22)
    times, positions, openings = [0.0], [home], [0.04]

    def arrive(duration: float, position: tuple[float, float, float], opening: float) -> None:
        times.append(times[-1] + duration)
        positions.append(position)
        openings.append(opening)

    mouth_x, mouth_y, mouth_z = mouth
    above_stocking = (mouth_x, mouth_y, mouth_z + 0.15)
    release = (mouth_x, mouth_y, mouth_z + 0.075)
    arrive(1.0, home, 0.04)
    for pick in pick_positions:
        above_tray = (pick[0], pick[1], 1.25)
        arrive(2.0, above_tray, 0.04)
        arrive(2.0, pick, 0.04)
        arrive(0.8, pick, 0.010)
        arrive(2.0, above_tray, 0.010)
        arrive(2.5, above_stocking, 0.010)
        arrive(1.0, release, 0.010)
        arrive(0.8, release, 0.04)
        arrive(0.8, release, 0.04)
        arrive(1.0, above_stocking, 0.04)
    arrive(2.0, home, 0.04)
    arrive(3.0, home, 0.04)
    return np.array(times), np.array(positions), np.array(openings)


def _sample_motion(time: float, times: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Interpolate targets with zero velocity and acceleration at each waypoint."""
    segment = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 2))
    alpha = np.clip((time - times[segment]) / (times[segment + 1] - times[segment]), 0.0, 1.0)
    blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
    return targets[segment] + blend * (targets[segment + 1] - targets[segment])


def _collapse_franka_frames(builder: newton.ModelBuilder) -> None:
    # Keep the hand as a body for the tool-center-point objective.
    hand_joints = [label for label in builder.joint_label if label.endswith("/fr3_hand_joint")]
    builder.collapse_fixed_joints(joints_to_keep=hand_joints)


def _prepare_world(builder: newton.ModelBuilder, _world: int, _position: np.ndarray, _orientation: np.ndarray) -> None:
    """Apply the fixed cuff and the collision split between the candy core and wrapper."""
    stage = sim_utils.get_current_stage()
    stocking = next(path for path in builder.surface_label if path.endswith("/Atelier/Stocking"))
    cloth = stage.GetPrimAtPath(stocking)
    cuff = cloth.GetAttribute("primvars:atelier:cuffIndices").Get()
    # The stocking is the scene's only deformable, so its vertex IDs index the particle buffer directly.
    for vertex in cuff:
        builder.particle_mass[vertex] = 0.0

    # The scene-data provider publishes the deforming surface to RTX. Give Newton GL the same face colors.
    colors = UsdGeom.Gprim(stage.GetPrimAtPath(f"{stocking}/Surface")).GetDisplayColorAttr().Get()
    builder.tri_color[:] = [tuple(color) for color in colors]

    rigid_contact = int(newton.ShapeFlags.COLLIDE_SHAPES)
    particle_contact = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    for index, label in enumerate(builder.shape_label):
        if "/Atelier/" in label and builder.shape_body[index] == -1:
            builder.shape_flags[index] |= int(newton.ShapeFlags.VISIBLE)
        elif label.endswith("/WrapperCollision"):
            builder.shape_flags[index] &= ~rigid_contact
        elif label.endswith("/Core") or (
            "/Robot/" in label and not any(finger in label for finger in ("/fr3_leftfinger/", "/fr3_rightfinger/"))
        ):
            builder.shape_flags[index] &= ~particle_contact


def _run(sim: sim_utils.SimulationContext, scene: InteractiveScene, max_steps: int) -> None:
    env = scene.env_prim_paths[0]
    transforms = UsdGeom.XformCache()

    def position(path: str) -> tuple[float, float, float]:
        return tuple(
            transforms.GetLocalToWorldTransform(scene.stage.GetPrimAtPath(f"{env}/Atelier/{path}")).ExtractTranslation()
        )

    times, positions, openings = _choreography([position(f"candy_{index}") for index in range(3)], position("Stocking"))
    reference = copy_newton_clone_source(f"{env}/Robot")
    _collapse_franka_frames(reference)
    reference_coords = [reference.joint_q_start[joint] for joint in _franka_joint_ids(reference)]
    for coord, angle in zip(reference_coords, FRANKA_HOME, strict=True):
        reference.joint_q[coord] = angle
    finger_coords = reference_coords[len(FRANKA_ARM_JOINTS) :]
    targets = _solve_waypoints(reference, positions, openings, finger_coords, sim.device)[:, reference_coords]

    builder = sim.backend_manager.builder
    _collapse_franka_frames(builder)
    for joint, angle in zip(_franka_joint_ids(builder), targets[0], strict=True):
        builder.joint_q[builder.joint_q_start[joint]] = float(angle)
    sim.reset()

    robot = scene["robot"]
    joint_ids, _ = robot.find_joints(FRANKA_JOINTS, preserve_order=True)
    frames = math.ceil(times[-1] * FPS)
    if max_steps:
        frames = min(frames, max_steps)
    sim_time = 0.0
    for _ in range(frames):
        if not sim.is_running():
            break
        target = torch.as_tensor(_sample_motion(sim_time, times, targets), dtype=torch.float32, device=sim.device)
        robot.actuators.target_command.set_position_index(value=target.unsqueeze(0), joint_ids=joint_ids)
        scene.write_data_to_sim()
        for _ in range(SUBSTEPS):
            sim.step(render=False)
        scene.update(1.0 / FPS)
        sim.render()
        sim_time += 1.0 / FPS


def main() -> None:
    """Launch the workcell and run one complete stocking fill."""
    parser = argparse.ArgumentParser(
        description="Fill a Christmas stocking with a scripted Franka FR3 in Isaac Lab.",
        epilog="Use --viz newton_rtx with the ovrtx extra, or --viz none to run headless.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Limit the run to this many frames. Zero runs it in full."
    )
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max_steps must be nonnegative.")

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / FPS / SUBSTEPS,
        render_interval=SUBSTEPS,
        device=args.device,
        physics=NewtonCfg(
            solver_cfg=AtelierVBDSolverCfg(
                iterations=15,
                update_joint_state=False,
                particle_enable_self_contact=True,
                particle_self_contact_margin=2 * PARTICLE_RADIUS,
                particle_enable_tile_solve=args.device.startswith("cuda"),
                rigid_compliant_alm=True,
                rigid_joint_linear_ke=1.0e7,
                rigid_joint_angular_ke=1.0e6,
                rigid_body_contact_buffer_size=256,
                rigid_body_particle_contact_buffer_size=4096,
                friction_epsilon=1.0e-3,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(
                broad_phase="sap", soft_contact_margin=0.004, enable_rigid_soft_full_surface_contact=True
            ),
            soft_contact_cfg=NewtonSoftContactCfg(soft_contact_ke=2000.0, soft_contact_kd=0.1, soft_contact_mu=0.5),
        ),
        visualizer_cfgs=VisualizerCfg(
            eye=(1.65, -2.25, 1.9),
            lookat=(0.34, 0.0, 1.0),
            focal_length=34.0,
            enable_live_plots=False,
            backend=NewtonGLBackendCfg(show_particles=False),
        ),
    )
    scene_cfg = AtelierSceneCfg(num_envs=1, env_spacing=0.0)
    with launch_simulation({"sim": sim_cfg, "scene": scene_cfg}, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        with newton_builder_world_hook(_prepare_world):
            scene = InteractiveScene(scene_cfg)
        _run(sim, scene, args.max_steps)


if __name__ == "__main__":
    main()
