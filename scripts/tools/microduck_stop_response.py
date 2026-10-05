# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure how a deployed MicroDuck velocity policy brakes after its command drops to zero.

The script builds ``IsaacContrib-Velocity-Flat-MicroDuck`` or ``IsaacContrib-Velocity-Rough-MicroDuck``
the way ``isaaclab play`` builds it (play mode: observation noise and pushes off), runs an exported
ONNX actor (61 observations, 14 actions, normalization inside the graph) through onnxruntime on the
environment's own policy observation, and drives the velocity command by hand:

1. ``settle``: zero twist for ``--settle-s`` seconds;
2. ``command``: ``(+-speed, 0, 0)`` until the root link has moved ``--distance`` metres along its initial
   heading (``fwd_cl`` / ``rev_cl``), for a fixed number of steps (``fwd_ol`` / ``rev_ol``), or for one
   second (``fwd_1s`` / ``rev_1s``);
3. ``zero``: zero twist for ``--zero-s`` seconds.

``stand`` holds a zero twist for ``--stand-s`` seconds after the settle. Every policy step is written to the
output JSON (observation, actions, root pose and velocity, joint state), and a summary table is printed:
displacement while commanded, displacement after the zero command (along and lateral to the initial
heading), yaw change, time until the planar speed stays below 0.02 m/s for 0.2 s, and falls.

Example::

    uv run python scripts/tools/microduck_stop_response.py --task IsaacContrib-Velocity-Flat-MicroDuck \\
        --onnx /path/to/velocity_flat.onnx --out stop_response.json

``--usd`` points the robot at a local copy of the MicroDuck USD when Nucleus is unavailable, ``--no-randomization``
zeroes the task's randomization ranges, ``--bam-nominal`` replaces the asset's sampled BAM deployment by a
fixed supply without command delay or current limit, and ``--plane`` runs the rough task on the flat
collision plane (MJWarp rejects an all-flat mesh terrain). Requires ``onnxruntime`` (``uv pip install onnxruntime``).
"""

# Warp captures ``enable_backward`` when a module is created, which happens at import
# time, so it has to be set before importing anything that defines Warp kernels.
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils import validate  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

POLICY_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
EPISODE_KINDS = ("stand", "fwd_cl", "rev_cl", "fwd_ol", "rev_ol", "fwd_1s", "rev_1s")
STILL_SPEED = 0.02
"""Planar root speed below which the robot counts as still [m/s]."""
STILL_STEPS = 10
"""Consecutive still policy steps required (0.2 s at 50 Hz)."""


def parse_args(argv=None):
    """Parse the command line; the remainder goes to Hydra."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=str, default="IsaacContrib-Velocity-Flat-MicroDuck", help="MicroDuck task.")
    parser.add_argument("--onnx", type=str, required=True, help="Exported actor (61 obs -> 14 actions).")
    parser.add_argument("--out", type=str, required=True, help="Output JSON with per-step logs and summaries.")
    parser.add_argument(
        "--usd", type=str, default=None, help="Local MicroDuck USD overriding the asset's Nucleus path."
    )
    parser.add_argument("--seed", type=int, default=0, help="Environment seed; episode seeds derive from it.")
    parser.add_argument("--episodes", type=str, default="stand,fwd_cl,rev_cl,fwd_ol,rev_ol", help="Comma list.")
    parser.add_argument("--trials", type=int, default=1, help="Repetitions of the episode list.")
    parser.add_argument("--speed", type=float, default=0.3, help="Commanded |vx| [m/s].")
    parser.add_argument("--distance", type=float, default=0.025, help="Closed-loop displacement target [m].")
    parser.add_argument("--open-loop-steps", type=int, default=15, help="Command steps of the fwd_ol/rev_ol episodes.")
    parser.add_argument("--max-command-s", type=float, default=3.0, help="Cap on the closed-loop command phase [s].")
    parser.add_argument("--settle-s", type=float, default=1.0, help="Zero-twist settle before the command [s].")
    parser.add_argument("--zero-s", type=float, default=3.0, help="Zero-twist hold after the command [s].")
    parser.add_argument("--stand-s", type=float, default=10.0, help="Duration of the stand episode [s].")
    parser.add_argument(
        "--no-randomization", action="store_true", help="Zero all randomization ranges and the IMU lag."
    )
    parser.add_argument(
        "--bam-nominal", action="store_true", help="Fixed supply, no sag range, no delay, no current limit."
    )
    parser.add_argument("--vin", type=float, default=7.4, help="Supply voltage with --bam-nominal [V].")
    parser.add_argument("--sag-gain", type=float, default=0.1, help="Supply sag gain with --bam-nominal [V/(N.m)].")
    parser.add_argument("--plane", action="store_true", help="Run the rough task on the flat collision plane.")
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    unknown = sorted(set(k.strip() for k in args_cli.episodes.split(",") if k.strip()) - set(EPISODE_KINDS))
    if unknown:
        parser.error(f"unknown episodes {unknown}; choose from {EPISODE_KINDS}")
    return args_cli


def sha256_of(path: str) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def yaw_of(q_xyzw: np.ndarray) -> float:
    """Yaw [rad] of an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_of(q_xyzw: np.ndarray) -> float:
    """Angle between the body z axis and world up [rad] for an (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


def wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def configure_env_cfg(env_cfg, args_cli):
    """Apply the script's overrides to the play-mode task configuration."""
    if args_cli.usd:
        env_cfg.scene.robot.spawn.usd_path = args_cli.usd
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1
    # long episodes: time-outs would reset the robot in the middle of a protocol
    env_cfg.episode_length_s = max(60.0, args_cli.settle_s + args_cli.stand_s + 10.0)
    # the script writes the commands; make the task's own sampling inert
    for name in ("base_velocity", "head_pose", "body_pose"):
        getattr(env_cfg.commands, name).resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_forward_envs = 0.0
    env_cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    env_cfg.commands.base_velocity.debug_vis = False
    if args_cli.plane and env_cfg.scene.terrain.terrain_type != "plane":
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        for sensor_name in ("left_foot_height", "right_foot_height"):
            if hasattr(env_cfg.scene, sensor_name):
                setattr(env_cfg.scene, sensor_name, None)
        for term in (
            env_cfg.observations.critic.foot_height,
            env_cfg.rewards.foot_clearance,
            env_cfg.rewards.foot_swing_height,
        ):
            term.params["height_sensor_names"] = ()
        if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
            env_cfg.curriculum.terrain_levels = None
    if args_cli.no_randomization:
        events = env_cfg.events
        events.foot_friction.params["static_friction_range"] = (1.0, 1.0)
        events.foot_friction.params["dynamic_friction_range"] = (1.0, 1.0)
        events.encoder_bias.params["bias_range"] = (0.0, 0.0)
        events.imu_misalignment.params["max_angle_deg"] = 0.0
        events.mass_inertia.params["mass_distribution_params"] = (1.0, 1.0)
        events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
        zero_com = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        events.randomize_com.params["com_range"] = zero_com
        events.randomize_head_com.params["com_range"] = dict(zero_com)
        events.randomize_joint_friction.params["scale_range"] = (1.0, 1.0)
        events.randomize_armature.params["armature_distribution_params"] = (1.0, 1.0)
        env_cfg.observations.policy.base_ang_vel.params["max_lag"] = 0
        env_cfg.observations.policy.projected_gravity.params["max_lag"] = 0
    if args_cli.bam_nominal:
        servos = env_cfg.scene.robot.actuators["servos"]
        servos.vin = args_cli.vin
        servos.vin_range = None
        servos.vin_drop_gain_range = (args_cli.sag_gain, args_cli.sag_gain)
        servos.min_delay = 0
        servos.max_delay = 0
        servos.motor.max_current = 0.0  # zero disables current limiting
    return env_cfg


def summarize_cfg(env_cfg, args_cli) -> dict:
    """Record the settings that matter for the measurement."""
    servos = env_cfg.scene.robot.actuators["servos"]
    solver = env_cfg.sim.physics.solver_cfg
    usd_path = env_cfg.scene.robot.spawn.usd_path
    return {
        "task": args_cli.task,
        "seed": args_cli.seed,
        "no_randomization": args_cli.no_randomization,
        "bam_nominal": args_cli.bam_nominal,
        "plane": args_cli.plane,
        "usd_path": usd_path,
        "usd_sha256": sha256_of(usd_path) if os.path.isfile(usd_path) else None,
        "terrain_type": env_cfg.scene.terrain.terrain_type,
        "physics_dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "solver": {
            "iterations": solver.iterations,
            "ls_iterations": solver.ls_iterations,
            "nconmax": solver.nconmax,
            "njmax": solver.njmax,
            "integrator": solver.integrator,
        },
        "bam": {
            "kp_fw": servos.kp_fw,
            "vin": servos.vin,
            "vin_range": servos.vin_range,
            "vin_drop_gain_range": servos.vin_drop_gain_range,
            "vin_min": servos.vin_min,
            "min_delay": servos.min_delay,
            "max_delay": servos.max_delay,
            "max_current": servos.motor.max_current,
        },
        "imu_obs_lag": [
            env_cfg.observations.policy.base_ang_vel.params["min_lag"],
            env_cfg.observations.policy.base_ang_vel.params["max_lag"],
        ],
        "joint_vel_obs_lag": [
            env_cfg.observations.policy.joint_vel.delay_min_lag,
            env_cfg.observations.policy.joint_vel.delay_max_lag,
        ],
    }


class OnnxActor:
    """Exported actor run with onnxruntime on the CPU."""

    def __init__(self, path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit("onnxruntime is required: uv pip install onnxruntime") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        if list(inp.shape) != [1, 61] or list(out.shape) != [1, 14]:
            raise SystemExit(f"expected obs [1, 61] -> actions [1, 14], got {inp.shape} -> {out.shape}")
        self.input_name = inp.name
        self.output_name = out.name
        self.path = path
        self.sha256 = sha256_of(path)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs.reshape(1, 61).astype(np.float32))
        actions = self.session.run([self.output_name], {self.input_name: obs})[0][0]
        if not np.isfinite(actions).all():
            raise RuntimeError("non-finite policy output")
        return actions.astype(np.float32)


class StopResponseRunner:
    """Drives the environment through the settle / command / zero protocol."""

    def __init__(self, env, actor: OnnxActor, args_cli):
        self.env = env
        self.uenv = env.unwrapped
        self.actor = actor
        self.args = args_cli
        self.robot = self.uenv.scene["robot"]
        self.joint_ids, joint_names = self.robot.find_joints(POLICY_JOINT_NAMES, preserve_order=True)
        if list(joint_names) != POLICY_JOINT_NAMES:
            raise RuntimeError(f"unexpected joint resolution: {joint_names}")
        self.vel_term = self.uenv.command_manager.get_term("base_velocity")
        self.pose_terms = [self.uenv.command_manager.get_term(n) for n in ("head_pose", "body_pose")]
        self.device = self.uenv.device
        self.dt = self.uenv.step_dt
        names = list(self.uenv.observation_manager.active_terms["policy"])
        dims = [int(np.prod(d)) for d in self.uenv.observation_manager.group_obs_term_dim["policy"]]
        offsets = np.cumsum([0] + dims)
        self.layout = {n: (int(offsets[k]), int(offsets[k + 1])) for k, n in enumerate(names)}
        self.command_slice = slice(*self.layout["velocity_commands"])
        self.pose_slices = [slice(*self.layout["head_pose_commands"]), slice(*self.layout["body_pose_commands"])]

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        self.vel_term.vel_command_b[:, 0] = vx
        self.vel_term.vel_command_b[:, 1] = vy
        self.vel_term.vel_command_b[:, 2] = wz
        self.vel_term.is_standing_env[:] = False
        if hasattr(self.vel_term, "is_heading_env"):
            self.vel_term.is_heading_env[:] = False
        for term in self.pose_terms:
            term.command[:] = 0.0

    def state(self) -> dict:
        data = self.robot.data
        quat = data.root_quat_w.torch[0].detach().cpu().numpy().astype(float)
        return {
            "pos": data.root_pos_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "quat_xyzw": quat.tolist(),
            "yaw": yaw_of(quat),
            "tilt": tilt_of(quat),
            "lin_vel_w": data.root_lin_vel_w.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "ang_vel_b": data.root_ang_vel_b.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "lin_vel_b": data.root_lin_vel_b.torch[0].detach().cpu().numpy().astype(float).tolist(),
            "joint_pos": data.joint_pos.torch[0, self.joint_ids].detach().cpu().numpy().astype(float).tolist(),
            "joint_vel": data.joint_vel.torch[0, self.joint_ids].detach().cpu().numpy().astype(float).tolist(),
        }

    def reset(self, seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        np.random.seed(seed)
        obs, _ = self.env.reset(seed=seed)
        self.set_command(0.0, 0.0, 0.0)
        return obs["policy"][0].detach().cpu().numpy().astype(np.float32)

    def step(self, obs: np.ndarray, cmd: tuple, log: list, phase: str, index: int):
        """Apply ``cmd``, run the actor on the observation of the current state and step once."""
        self.set_command(*cmd)
        # the observation returned by the previous step carries the previous command; only its command
        # slices depend on it, so they are replaced instead of recomputing the whole group
        obs_now = obs.copy()
        obs_now[self.command_slice] = np.asarray(cmd, dtype=np.float32)
        for pose_slice in self.pose_slices:
            obs_now[pose_slice] = 0.0
        action = self.actor(obs_now)
        next_obs, _, terminated, truncated, _ = self.env.step(torch.as_tensor(action, device=self.device).unsqueeze(0))
        record = {
            "i": index,
            "phase": phase,
            "t": index * self.dt,
            "cmd": list(cmd),
            "obs": obs_now.astype(float).tolist(),
            "action": action.astype(float).tolist(),
            "terminated": bool(terminated[0].item()),
            "truncated": bool(truncated[0].item()),
            **self.state(),
        }
        log.append(record)
        done = record["terminated"] or record["truncated"]
        return next_obs["policy"][0].detach().cpu().numpy().astype(np.float32), done

    @staticmethod
    def _planar_speed(record: dict) -> float:
        return math.hypot(record["lin_vel_w"][0], record["lin_vel_w"][1])

    def _hold(self, obs, cmd, steps, phase, log, index, stop_fn=None):
        """Hold ``cmd`` for up to ``steps`` policy steps; ``stop_fn(record)`` ends the phase early."""
        done = False
        count = 0
        while count < steps:
            obs, done = self.step(obs, cmd, log, phase, index)
            index += 1
            count += 1
            if done or (stop_fn is not None and stop_fn(log[-1])):
                break
        return obs, done, index, count

    def run_episode(self, kind: str, seed: int) -> dict:
        """Run one episode of ``kind`` and return its summary with the per-step log."""
        log: list[dict] = []
        obs = self.reset(seed)
        settle_steps = int(round(self.args.settle_s / self.dt))
        result = {"kind": kind, "seed": seed}
        obs, done, index, _ = self._hold(obs, (0.0, 0.0, 0.0), settle_steps, "settle", log, 0)
        if done:
            result.update(
                fell=bool(log[-1]["terminated"]), reset_during_episode=True, note="ended during settle", log=log
            )
            return result
        start = log[-1]
        p0 = np.array(start["pos"][:2])
        heading = np.array([math.cos(start["yaw"]), math.sin(start["yaw"])])
        lateral = np.array([-heading[1], heading[0]])

        def still_stats(records):
            run_length = 0
            first = None
            for k, rec in enumerate(records):
                run_length = run_length + 1 if self._planar_speed(rec) < STILL_SPEED else 0
                if run_length >= STILL_STEPS and first is None:
                    first = (k + 1) * self.dt
            return first

        if kind == "stand":
            stand_steps = int(round(self.args.stand_s / self.dt))
            obs, done, index, _ = self._hold(obs, (0.0, 0.0, 0.0), stand_steps, "stand", log, index)
            stand = [r for r in log if r["phase"] == "stand"]
            end = np.array(stand[-1]["pos"][:2])
            result.update(
                stand_duration_s=len(stand) * self.dt,
                stand_drift_mm=1000.0 * float(np.linalg.norm(end - p0)),
                stand_yaw_change_deg=math.degrees(wrap(stand[-1]["yaw"] - start["yaw"])),
                stand_max_planar_speed=max(self._planar_speed(r) for r in stand),
                stand_max_tilt_deg=math.degrees(max(r["tilt"] for r in stand)),
                fell=bool(done and log[-1]["terminated"]),
                reset_during_episode=bool(done),
                log=log,
            )
            return result

        sign = 1.0 if kind.startswith("fwd") else -1.0
        cmd = (sign * self.args.speed, 0.0, 0.0)
        target = self.args.distance

        def reached_target(rec: dict) -> bool:
            return sign * float((np.array(rec["pos"][:2]) - p0) @ heading) >= target

        closed_loop = kind.endswith("_cl")
        if closed_loop:
            max_steps = int(round(self.args.max_command_s / self.dt))
        elif kind.endswith("_1s"):
            max_steps = int(round(1.0 / self.dt))
        else:
            max_steps = self.args.open_loop_steps
        stop_fn = reached_target if closed_loop else None
        obs, done, index, cmd_steps = self._hold(obs, cmd, max_steps, "command", log, index, stop_fn)
        cmd_end = log[-1]
        pc = np.array(cmd_end["pos"][:2])
        result.update(
            command=list(cmd),
            command_steps=cmd_steps,
            command_reached=(closed_loop and reached_target(cmd_end) and not done),
            during_along_mm=1000.0 * float((pc - p0) @ heading),
            during_lateral_mm=1000.0 * float((pc - p0) @ lateral),
            during_yaw_change_deg=math.degrees(wrap(cmd_end["yaw"] - start["yaw"])),
            during_speed_at_end=self._planar_speed(cmd_end),
            during_max_planar_speed=max(self._planar_speed(r) for r in log if r["phase"] == "command"),
        )
        if done:
            result.update(
                fell=bool(log[-1]["terminated"]), reset_during_episode=True, note="ended while commanded", log=log
            )
            return result
        zero_steps = int(round(self.args.zero_s / self.dt))
        obs, done, index, _ = self._hold(obs, (0.0, 0.0, 0.0), zero_steps, "zero", log, index)
        zero = [r for r in log if r["phase"] == "zero"]
        pz = np.array(zero[-1]["pos"][:2])
        last_second = zero[-int(round(1.0 / self.dt)) :]
        result.update(
            zero_duration_s=len(zero) * self.dt,
            post_zero_along_mm=1000.0 * float((pz - pc) @ heading),
            post_zero_lateral_mm=1000.0 * float((pz - pc) @ lateral),
            post_zero_yaw_change_deg=math.degrees(wrap(zero[-1]["yaw"] - cmd_end["yaw"])),
            post_zero_max_planar_speed=max(self._planar_speed(r) for r in zero),
            post_zero_first_still_s=still_stats(zero),
            post_zero_last_1s_displacement_mm=1000.0 * float(np.linalg.norm(pz - np.array(last_second[0]["pos"][:2]))),
            post_zero_max_tilt_deg=math.degrees(max(r["tilt"] for r in zero)),
            total_along_mm=1000.0 * float((pz - p0) @ heading),
            total_lateral_mm=1000.0 * float((pz - p0) @ lateral),
            fell=bool(done and log[-1]["terminated"]),
            reset_during_episode=bool(done),
            log=log,
        )
        return result


def format_summary(results: list[dict]) -> str:
    """Render the episode summaries as a text table."""
    header = (
        f"{'episode':8} {'seed':>6} {'cmd steps':>9} {'reached':>7} {'during along':>12} {'during lat':>10} "
        f"{'post-zero along':>15} {'post-zero lat':>13} {'yaw after':>9} {'still at':>8} {'fell':>5}"
    )
    lines = [header, "-" * len(header)]
    nan = float("nan")
    for r in results:
        if r["kind"] == "stand":
            drift = r.get("stand_drift_mm")
            drift_text = f"{drift:.2f} mm drift" if drift is not None else "-"
            lines.append(
                f"{r['kind']:8} {r['seed']:>6} {'-':>9} {'-':>7} {'-':>12} {'-':>10} {drift_text:>15} {'-':>13} "
                f"{r.get('stand_yaw_change_deg', nan):>8.2f}d {'-':>8} {str(r.get('fell')):>5}"
            )
            continue
        still = r.get("post_zero_first_still_s")
        still_text = f"{still:.2f} s" if still is not None else "never"
        lines.append(
            f"{r['kind']:8} {r['seed']:>6} {r.get('command_steps', 0):>9} {str(r.get('command_reached')):>7} "
            f"{r.get('during_along_mm', nan):>9.1f} mm {r.get('during_lateral_mm', nan):>7.1f} mm "
            f"{r.get('post_zero_along_mm', nan):>12.1f} mm {r.get('post_zero_lateral_mm', nan):>10.1f} mm "
            f"{r.get('post_zero_yaw_change_deg', nan):>8.1f}d {still_text:>8} {str(r.get('fell')):>5}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    """Run the stop-response protocol and write the JSON report."""
    args_cli = parse_args(argv)
    env_cfg, _ = resolve_task_config(args_cli.task, "", play_mode=True)
    env_cfg = configure_env_cfg(env_cfg, args_cli)
    try:
        validate(env_cfg)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid environment configuration: {exc}") from None
    actor = OnnxActor(args_cli.onnx)
    report = {
        "cfg": summarize_cfg(env_cfg, args_cli),
        "policy": {"path": os.path.abspath(args_cli.onnx), "sha256": actor.sha256},
        "protocol": {
            "settle_s": args_cli.settle_s,
            "speed": args_cli.speed,
            "distance": args_cli.distance,
            "open_loop_steps": args_cli.open_loop_steps,
            "zero_s": args_cli.zero_s,
            "stand_s": args_cli.stand_s,
            "still_speed": STILL_SPEED,
            "still_steps": STILL_STEPS,
        },
        "episodes": [],
    }
    kinds = [k.strip() for k in args_cli.episodes.split(",") if k.strip()]
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        env = gym.make(args_cli.task, cfg=env_cfg)
        cleanup.callback(env.close)
        runner = StopResponseRunner(env, actor, args_cli)
        report["policy_obs_layout"] = runner.layout
        report["default_joint_pos"] = (
            runner.robot.data.default_joint_pos.torch[0, runner.joint_ids].detach().cpu().numpy().astype(float).tolist()
        )
        for trial in range(args_cli.trials):
            for kind in kinds:
                seed = args_cli.seed * 1000 + trial
                print(f"[stop-response] episode {kind}, seed {seed}", flush=True)
                result = runner.run_episode(kind, seed)
                result["trial"] = trial
                report["episodes"].append(result)
                with open(args_cli.out, "w") as f:
                    json.dump(report, f)
    print(format_summary(report["episodes"]), flush=True)
    print(f"[stop-response] wrote {args_cli.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
