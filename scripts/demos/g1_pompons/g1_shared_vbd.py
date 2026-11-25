# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mount two dynamic pompons in each G1 motion-tracking world.

Train and play a SuperMimic policy with both props in the same Newton model.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_newton.cloner.replicate import newton_builder_world_hook  # noqa: E402
from isaaclab_newton.physics import VBDSolverCfg  # noqa: E402
from pompon import make_pompon_builder  # noqa: E402

from isaaclab.app import launch_simulation  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402
from isaaclab_rl.entrypoints.backends.runner_rsl_rl import create_rsl_rl_runner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.core.motion_tracking.config.g1.env_cfg import G1MotionTrackingEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

TASK = "Isaac-Motion-Tracking-G1"
AGENT = "rsl_rl_supermimic_cfg_entry_point"
DEFAULT_MOTION = Path(__file__).with_name("step_touch_50hz.npz")
FLOOR_CONTACT_KE = 100_000.0


@configclass
class G1PomponEnvCfg(G1MotionTrackingEnvCfg):
    """Use the VBD settings from the mounted-pompon experiment."""

    render_mounted_pompons: bool = False

    def __post_init__(self) -> None:
        """Set the solver, clock, and floor friction used by the experiment."""
        super().__post_init__()
        self.sim.physics.solver_cfg = VBDSolverCfg(iterations=12)
        self.sim.dt = 0.0025
        self.decimation = 8
        self.sim.render_interval = self.decimation
        self.scene.terrain.physics_material.static_friction = 0.6
        self.scene.terrain.physics_material.dynamic_friction = 0.6
        self.sim.physics_material = self.scene.terrain.physics_material


@wp.kernel
def reset_pompons(
    env_ids: wp.array[wp.int32],
    wrists: wp.array2d[wp.int32],
    props: wp.array3d[wp.int32],
    rest: wp.array3d[wp.transform],
    com: wp.array[wp.vec3],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
):
    selected, hand, link = wp.tid()
    world = env_ids[selected]
    wrist = wrists[world, hand]
    body = props[world, hand, link]
    wrist_pose = poses[wrist]
    pose = wp.transform_multiply(wrist_pose, rest[world, hand, link])
    wrist_velocity = velocities[wrist]
    omega = wp.spatial_bottom(wrist_velocity)
    lever = wp.transform_point(pose, com[body]) - wp.transform_point(wrist_pose, com[wrist])
    linear = wp.spatial_top(wrist_velocity) + wp.cross(omega, lever)
    poses[body] = pose
    velocities[body] = wp.spatial_vector(linear, omega)


class WristPompons:
    """Author wrist constraints and reset the props to the current wrist pose."""

    def __init__(self, visual_meshes: bool = False):
        self.template, self.roots = make_pompon_builder(visual_meshes=visual_meshes)
        self.mount = wp.transform(
            wp.vec3(0.22, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 2.0)
        )
        self.inverse_hub = wp.transform_inverse(self.template.body_q[self.roots[0]])
        self.wrist_rows, self.prop_rows, self.rest_rows = [], [], []
        self.device = None

    def populate(self, builder, world: int, _position: np.ndarray, _orientation: np.ndarray) -> None:
        """Add this world's two dynamic prop trees and their fixed wrist joints."""
        current_bodies = {}
        for body in range(builder.body_count - 1, -1, -1):
            if builder.body_world[body] != world:
                break
            current_bodies[builder.body_label[body].rsplit("/", 1)[-1]] = body
        wrists, props, rest = [], [], []
        for side in ("left", "right"):
            wrist = current_bodies[f"{side}_wrist_yaw_link"]
            wrist_pose = builder.body_q[wrist]
            prefix = f"/World/envs/env_{world}/Pompons/{side}"
            xform = wp.transform_multiply(wp.transform_multiply(wrist_pose, self.mount), self.inverse_hub)
            first = builder.body_count
            builder.add_builder(self.template, xform=xform, label_prefix=prefix)
            body_ids = list(range(first, builder.body_count))
            relative = [
                wp.transform_multiply(wp.transform_inverse(wrist_pose), builder.body_q[body]) for body in body_ids
            ]
            for root in self.roots:
                builder.add_joint_fixed(
                    wrist, first + root, parent_xform=relative[root], label=f"{prefix}/mount_{root}"
                )
            wrists.append(wrist)
            props.append(body_ids)
            rest.append(relative)
        if world != len(self.wrist_rows):
            raise RuntimeError(f"Unexpected clone world order: {world}.")
        self.wrist_rows.append(wrists)
        self.prop_rows.append(props)
        self.rest_rows.append(rest)

    def reset(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice) -> None:
        """Rebase each selected set of guides after the G1 reference reset."""
        backend = env.sim.backend_manager.backend
        if self.device is None:
            self.device = backend.device
            self.wrists = wp.array2d(np.asarray(self.wrist_rows, dtype=np.int32), dtype=wp.int32, device=self.device)
            self.props = wp.array3d(np.asarray(self.prop_rows, dtype=np.int32), dtype=wp.int32, device=self.device)
            self.rest = wp.array3d(np.asarray(self.rest_rows, dtype=np.float32), dtype=wp.transform, device=self.device)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)[env_ids]
        else:
            env_ids = env_ids.to(torch.int32).contiguous()
        backend.forward()
        state = backend.state_0
        wp.launch(
            reset_pompons,
            dim=(len(env_ids), 2, self.props.shape[2]),
            inputs=[
                wp.from_torch(env_ids),
                self.wrists,
                self.props,
                self.rest,
                backend.model.body_com,
                state.body_q,
                state.body_qd,
            ],
            device=self.device,
        )
        world_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        world_mask[env_ids.long()] = True
        backend.invalidate_body_state(wp.from_torch(world_mask))


class SharedVBDEnv(ManagerBasedRLEnv):
    """G1 motion tracking with dynamic props in the same Newton model."""

    floor_contact_ke: float = FLOOR_CONTACT_KE

    def __init__(self, cfg: G1PomponEnvCfg, **kwargs):
        """Mount the props during cloning and apply the floor stiffness."""
        if not isinstance(cfg.sim.physics.solver_cfg, VBDSolverCfg):
            raise ValueError("This demo requires the Newton VBD solver.")
        self.wrist_pompons = WristPompons(visual_meshes=cfg.render_mounted_pompons)
        with newton_builder_world_hook(self.wrist_pompons.populate):
            super().__init__(cfg, **kwargs)
        from newton import GeoType

        model = self.sim.backend_manager.backend.model
        planes = (wp.to_torch(model.shape_body) == -1) & (wp.to_torch(model.shape_type) == int(GeoType.PLANE))
        if planes.count_nonzero().item() != 1:
            raise ValueError("Setting floor contact stiffness requires one global plane in the G1 scene.")
        wp.to_torch(model.shape_material_ke)[planes] = self.floor_contact_ke
        print(f"VBD floor contact stiffness: {self.floor_contact_ke:g} N/m")

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.wrist_pompons.reset(self, env_ids)


def check(args: argparse.Namespace) -> None:
    """Roll out a saved actor from the first reference frame."""
    cfg, agent_cfg = resolve_task_config(TASK, AGENT, overrides=["presets=state_obs"])
    cfg.scene.num_envs = 1
    cfg.seed = 42
    cfg.episode_length_s = max(cfg.episode_length_s, (args.steps + 1) * cfg.sim.dt * cfg.decimation)
    cfg.commands.motion.motion_file = str(args.motion)
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0.0, 0.0)
    cfg.events.physics_material = None
    cfg.events.base_mass = None
    cfg.events.push_robot = None
    if args.render_frames:
        cfg.render_mounted_pompons = True
        cfg.sim.physics.load_visual_shapes = True

    with launch_simulation(cfg, {"device": args.device}):
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg))
        viewer = None
        try:
            raw = env.unwrapped
            mount = raw.wrist_pompons
            motion = raw.command_manager.get_term("motion")
            motion._adaptive_sampling = lambda ids: motion.time_steps.__setitem__(ids, 0)
            observations, _ = env.reset()
            backend = raw.sim.backend_manager.backend
            if args.render_frames:
                from newton.geometry import DeformableVisuals
                from newton.viewer import ViewerRTX
                from PIL import Image, ImageDraw
                from pompon_ovrtx import make_batches

                args.render_frames.mkdir(parents=True, exist_ok=True)
                viewer = ViewerRTX(
                    width=960,
                    height=720,
                    fps=30,
                    headless=True,
                    environment="studio",
                    async_rendering=False,
                    render_settings={"omni:rtx:dlss:frameGeneration": ("bool", False)},
                )
                viewer.set_model(backend.model)
                viewer.show_deformable_visual_meshes = False
                viewer.set_camera(pos=wp.vec3(-0.45, -2.5, 1.06))
                viewer.camera.look_at(wp.vec3(0.0, 0.0, 0.81))
                viewer.camera.fov = 46.0
                visuals = DeformableVisuals(backend.model)
                batches = make_batches(backend.model)
                frame = 0
                camera_x = camera_y = 0.0
            runner = create_rsl_rl_runner(env, agent_cfg, inference=True)
            runner.load(str(args.checkpoint), load_cfg={"actor": True}, map_location=raw.device)
            policy = runner.get_inference_policy(device=raw.device)
            elapsed, errors, mount_errors = 0, [], []
            torso_tilts, torso_heights, anchor_errors = [], [], []
            with torch.inference_mode():
                for _ in range(args.steps):
                    state = backend.state_0
                    poses = state.body_q.numpy()
                    if not np.isfinite(poses).all():
                        raise RuntimeError("VBD produced non-finite link poses.")
                    mount_positions = [
                        wp.transform_point(wp.transform(*poses[wrist]), mount.mount.p) for wrist in mount.wrist_rows[0]
                    ]
                    hub_positions = poses[np.asarray(mount.prop_rows[0])[:, 0], :3]
                    mount_errors.append(float(np.linalg.norm(hub_positions - mount_positions, axis=-1).max()))
                    errors.append(
                        torch.linalg.vector_norm(motion.robot_body_pos_w - motion.body_pos_relative_w, dim=-1)
                        .mean()
                        .item()
                    )
                    torso_up = 1 - 2 * motion.robot_anchor_quat_w[0, :2].square().sum().item()
                    torso_tilts.append(math.degrees(math.acos(min(1.0, max(-1.0, torso_up)))))
                    torso_heights.append(motion.robot_anchor_pos_w[0, 2].item())
                    anchor_errors.append(
                        torch.linalg.vector_norm(motion.robot_anchor_pos_w - motion.anchor_pos_w).item()
                    )
                    if viewer is not None and elapsed == round(frame / (30 * raw.step_dt)):
                        if args.follow_camera:
                            x, y = motion.robot_anchor_pos_w[0, :2].tolist()
                            camera_x += 0.15 * (x - camera_x)
                            camera_y += 0.15 * (y - camera_y)
                            viewer.set_camera(pos=wp.vec3(camera_x - 0.45, camera_y - 2.5, 1.06))
                            viewer.camera.look_at(wp.vec3(camera_x, camera_y, 0.81))
                        visuals.update(state)
                        skin = np.concatenate(
                            [visuals.get_points(mesh).numpy() for mesh in backend.model.deformable_visual_meshes]
                        )
                        viewer.begin_frame(frame / 30)
                        viewer.log_state(state)
                        for batch in batches:
                            viewer.log_mesh(
                                batch.name,
                                wp.array(skin[batch.vertices], dtype=wp.vec3, device=backend.device),
                                batch.triangles,
                                color=batch.color,
                                roughness=batch.roughness,
                                metallic=batch.metallic,
                                backface_culling=False,
                            )
                        viewer.end_frame()
                        image = Image.fromarray(viewer.get_frame().numpy())
                        ImageDraw.Draw(image).text(
                            (22, 19), "SuperMimic · one VBD simulation", fill=(233, 242, 248), font_size=21
                        )
                        image.save(args.render_frames / f"frame_{frame:04}.png")
                        frame += 1
                    observations, _, done, _ = env.step(policy(observations))
                    elapsed += 1
                    if done.item():
                        reasons = [
                            name
                            for name in raw.termination_manager.active_terms
                            if raw.termination_manager.get_term(name)[0]
                        ]
                        print(f"CHECK terminated at step {elapsed}: {reasons}")
                        break
            print(
                f"CHECK shared model: {backend.model.body_count} dynamic/rigid links, "
                f"{backend.model.joint_count} joints, {elapsed} steps, mean body error "
                f"{np.mean(errors):.4f} m, max hub anchor error {max(mount_errors):.4f} m"
            )
            print(
                f"CHECK torso tilt (max/final) {max(torso_tilts):.1f}/{torso_tilts[-1]:.1f} deg, "
                f"height (min/final) {min(torso_heights):.3f}/{torso_heights[-1]:.3f} m, "
                f"anchor error (mean/max) {np.mean(anchor_errors):.4f}/{max(anchor_errors):.4f} m"
            )
            if max(mount_errors) > 0.025:
                raise RuntimeError("Wrist anchor error exceeded 2.5 cm.")
        finally:
            if viewer is not None:
                viewer.close()
            env.close()


def positive_stiffness(value: str) -> float:
    """Parse a finite, positive contact stiffness in N/m."""
    stiffness = float(value)
    if not math.isfinite(stiffness) or stiffness <= 0:
        raise argparse.ArgumentTypeError("Floor contact stiffness must be finite and positive.")
    return stiffness


if __name__ == "__main__":
    task_spec = gym.spec(TASK)
    task_spec.entry_point = SharedVBDEnv
    task_spec.kwargs["env_cfg_entry_point"] = G1PomponEnvCfg
    floor_parser = argparse.ArgumentParser(add_help=False)
    floor_parser.add_argument(
        "--floor-contact-ke",
        type=positive_stiffness,
        default=FLOOR_CONTACT_KE,
        help="Ground shape stiffness in N/m. VBD averages it with the foot shape stiffness.",
    )
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        floor_args, train_cli = floor_parser.parse_known_args(sys.argv[2:])
        SharedVBDEnv.floor_contact_ke = floor_args.floor_contact_ke
        if not any(arg.startswith("env.commands.motion.motion_file=") for arg in train_cli):
            train_cli.append(f"env.commands.motion.motion_file={DEFAULT_MOTION}")
        raise SystemExit(run_train_cli(train_cli))
    parser = argparse.ArgumentParser(description=__doc__, parents=[floor_parser])
    parser.add_argument("motion", type=Path, nargs="?", default=DEFAULT_MOTION)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1624)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-frames", type=Path, help="Save OvRTX frames to this directory.")
    parser.add_argument("--follow-camera", action="store_true", help="Pan the render camera with the G1.")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive.")
    if args.follow_camera and not args.render_frames:
        parser.error("--follow-camera requires --render-frames.")
    SharedVBDEnv.floor_contact_ke = args.floor_contact_ke
    check(args)
