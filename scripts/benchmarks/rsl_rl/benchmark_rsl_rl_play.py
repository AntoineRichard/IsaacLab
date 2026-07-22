# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL adapter for the unified play benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

from scripts.benchmarks import _compat as _common
from scripts.benchmarks._compat import launch_app, resolve_task_config


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse benchmark arguments and retain PhysX preset tokens.

    Args:
        argv: Command-line arguments after the dispatcher removes ``--rl_library``.

    Returns:
        Parsed arguments and retained preset tokens.
    """
    parser = argparse.ArgumentParser(description="Benchmark RL inference (play) with RSL-RL.")
    help_requested = "-h" in argv or "--help" in argv
    parser.add_argument("--task", type=str, required=not help_requested, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of inference steps to benchmark.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or 'latest'/'best'; uses the published checkpoint when omitted.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="rsl_rl_cfg_entry_point",
        help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--benchmark_formatter",
        type=str,
        default="schema",
        help=(
            "Output format(s): comma-separated list of 'schema' (default, the typed benchmark bundle),"
            " 'omniperf', 'osmo', 'json', 'summary'. Example: 'schema,json'."
        ),
    )
    _common.add_launcher_args(parser)
    return _common.parse_benchmark_args(parser, argv)


def run(argv: list[str]) -> None:
    """Run the RSL-RL play benchmark and write a play bundle.

    Args:
        argv: Command-line arguments after the dispatcher removes ``--rl_library``.
    """
    args, remaining = _parse_args(argv)

    start_utc = datetime.now(timezone.utc).isoformat()
    app_t0 = time.perf_counter_ns()
    app_launcher = launch_app(args)
    app_t1 = time.perf_counter_ns()
    simulation_app = app_launcher.app

    try:
        import contextlib
        import importlib.metadata as metadata
        import os

        from rsl_rl.runners import DistillationRunner, OnPolicyRunner

        from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
        from isaaclab.test.benchmark.schema import StartupTime

        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper,
            handle_deprecated_rsl_rl_cfg,
            handle_deprecated_rsl_rl_checkpoint,
        )

        import isaaclab_tasks  # noqa: F401

        with contextlib.suppress(ImportError):
            import isaaclab_tasks_experimental  # noqa: F401

        env_cfg, agent_cfg = resolve_task_config(args.task, args.agent)
        _common.apply_env_overrides(args, env_cfg)
        if args.seed is not None:
            agent_cfg.seed = args.seed
        env_cfg.seed = agent_cfg.seed

        installed_rsl_rl = metadata.version("rsl-rl-lib")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_rsl_rl)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args.checkpoint in _common.CHECKPOINT_SELECTORS:
            expected_git_shas = _common.resolve_benchmark_git_shas()
            resume_path = _common.resolve_checkpoint_selector(
                log_root_path,
                args.checkpoint,
                library="rsl_rl",
                task=args.task,
                checkpoint_pattern=r"model_.*\.pt",
                metadata={"agent": args.agent},
                expected_git_shas=expected_git_shas,
            )
        else:
            resume_path = _common.resolve_play_checkpoint(args.checkpoint, "rsl_rl", args.task)
        resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_rsl_rl)
        env_cfg.log_dir = os.path.dirname(resume_path)

        cfg = capture.run_config_from_presets(remaining, env_cfg=env_cfg)
        formatter_types = [value.strip() for value in args.benchmark_formatter.split(",") if value.strip()]
        formatter_types = formatter_types or ["omniperf"]
        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_play",
            formatter_type=formatter_types,
            output_path=args.output_path,
            use_recorders=True,
            frametime_recorders=any(value in ("summary", "omniperf") for value in formatter_types),
            output_prefix=f"benchmark_play_{args.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args.task},
                    {"name": "num_envs", "data": args.num_envs},
                    {"name": "num_frames", "data": args.num_frames},
                    {"name": "presets", "data": ",".join(cfg.presets)},
                ]
            },
        )

        env = None
        try:
            env_t0 = time.perf_counter_ns()
            env = _common.create_isaaclab_env(args.task, env_cfg, args, convert_marl_to_single_agent=True)
            env_t1 = time.perf_counter_ns()
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            num_envs = env.unwrapped.num_envs
            runner_types = {"OnPolicyRunner": OnPolicyRunner, "DistillationRunner": DistillationRunner}
            if agent_cfg.class_name not in runner_types:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            runner = runner_types[agent_cfg.class_name](
                env,
                agent_cfg.to_dict(),
                log_dir=None,
                device=agent_cfg.device,
            )
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            with BenchmarkMonitor(benchmark, interval=1.0):
                step_times, reward, ep_length, success_rate = stepping.run_play_loop(env, policy, args.num_frames)

            benchmark.update_manual_recorders()
            startup = StartupTime(
                app_launch=(app_t1 - app_t0) / 1e9,
                env_creation=(env_t1 - env_t0) / 1e9,
                first_step=(step_times[0] if step_times else 0.0),
            )
            fps = [num_envs / step_time for step_time in step_times if step_time > 0]
            runtime = builders.build_runtime(
                startup_time_s=startup,
                iteration_times_s=step_times,
                collection_fps=fps,
                total_fps=fps,
                steps_per_iteration=num_envs,
            )

            versions = capture.capture_versions(benchmark)
            hardware = capture.capture_hardware(benchmark)
            resources = capture.capture_resources(benchmark)
            end_utc = capture.now_utc_iso()
            stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]
            seed = agent_cfg.seed if agent_cfg.seed is not None else 0
            run_identity = builders.build_run_identity(
                run_id=capture.synth_run_id("rsl_rl", cfg.physics_backend, args.task, seed, stamp),
                framework="rsl_rl",
                config=cfg,
                task=args.task,
                seed=seed,
                start_utc=start_utc,
                end_utc=end_utc,
                num_envs=num_envs,
            )
            bundle = builders.build_play_bundle(
                run=run_identity,
                versions=versions,
                hardware=hardware,
                runtime=runtime,
                resources=resources,
                success_rate=success_rate,
                reward=reward,
                ep_length=ep_length,
                checkpoint_path=resume_path,
            )

            benchmark.attach_bundle(bundle)
            benchmark._finalize_impl()
        finally:
            if env is not None:
                env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    run(sys.argv[1:])
