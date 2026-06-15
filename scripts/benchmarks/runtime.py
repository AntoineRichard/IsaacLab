# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark environment runtime (random actions, no policy).

Standalone script that steps an Isaac Lab environment with random actions and
emits a :class:`~isaaclab.test.benchmark.schema.RuntimeBundle` JSON file.
Supports all physics backends (PhysX, Newton/MJWarp, Newton/Kamino, OVPhysX)
via Hydra preset tokens — no ``--rl_library`` dispatch needed.

Usage example::

    ./isaaclab.sh -p scripts/benchmarks/runtime.py \\
        --task Isaac-Cartpole-Direct \\
        --num_envs 16 --num_frames 100 \\
        presets=newton_mjwarp --headless
"""

from __future__ import annotations

import argparse
import sys

from scripts.benchmarks._common import get_backend_type, preset_tokens

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse CLI arguments and fold Hydra preset tokens into ``sys.argv``.

    Builds the parser, appends launcher args via
    :func:`~isaaclab_tasks.utils.add_launcher_args`, then calls
    :func:`~isaaclab_tasks.utils.setup_preset_cli` to split known args from
    remaining Hydra tokens.  The folded tokens are written back to
    ``sys.argv`` so that ``launch_simulation`` finds them.

    Args:
        argv: Raw command-line arguments (``sys.argv[1:]``).

    Returns:
        Tuple of ``(parsed_args, folded_remaining)`` where *folded_remaining*
        are the collapsed Hydra tokens ready for ``set_hydra_args`` /
        ``launch_simulation``.
    """
    from isaaclab_tasks.utils import add_launcher_args, fold_preset_tokens, setup_preset_cli

    parser = argparse.ArgumentParser(description="Benchmark environment runtime (random actions, no policy).")
    parser.add_argument("--task", type=str, required=True, help="Gym task id to benchmark.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of environment steps to benchmark.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
    parser.add_argument("--output_path", type=str, default=".", help="Directory to write the output JSON.")
    parser.add_argument(
        "--benchmark_backend",
        type=str,
        default="omniperf",
        choices=[
            "json",
            "osmo",
            "omniperf",
            "summary",
            "LocalLogMetrics",
            "JSONFileMetrics",
            "OsmoKPIFile",
            "OmniPerfKPIFile",
        ],
        help="Benchmarking backend. Defaults to omniperf.",
    )
    add_launcher_args(parser)

    args, remaining = setup_preset_cli(parser, argv)
    folded = fold_preset_tokens(remaining)
    sys.argv = [sys.argv[0]] + folded
    return args, folded


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(argv: list[str]) -> None:
    """Run the runtime benchmark and write a :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`.

    Args:
        argv: Command-line arguments, excluding the script path (i.e.
            ``sys.argv[1:]``).
    """
    import contextlib
    import os
    import time

    import gymnasium as gym

    from isaaclab.test.benchmark import BaseIsaacLabBenchmark, BenchmarkMonitor, builders, capture, stepping
    from isaaclab.test.benchmark.schema import StartupTime
    from isaaclab.test.benchmark.serialize import write_bundle_file

    # Importing the task packages registers their gym environments so the
    # requested ``--task`` can be resolved.
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import launch_simulation, resolve_task_config

    # PLACEHOLDER: Extension template (do not remove this comment)
    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    args, folded = _parse_args(argv)

    env_cfg, _ = resolve_task_config(args.task, None)

    start_utc = capture.now_utc_iso()
    app_t0 = time.perf_counter_ns()

    with launch_simulation(env_cfg, args):
        app_t1 = time.perf_counter_ns()

        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed

        backend_type = get_backend_type(args.benchmark_backend)
        tokens = preset_tokens(folded)

        benchmark = BaseIsaacLabBenchmark(
            benchmark_name="benchmark_runtime",
            backend_type=backend_type,
            output_path=args.output_path,
            use_recorders=True,
            frametime_recorders=backend_type in ("summary", "omniperf"),
            output_prefix=f"benchmark_runtime_{args.task}",
            workflow_metadata={
                "metadata": [
                    {"name": "task", "data": args.task},
                    {"name": "num_envs", "data": args.num_envs},
                    {"name": "num_frames", "data": args.num_frames},
                    {"name": "presets", "data": ",".join(tokens)},
                ]
            },
        )

        env_t0 = time.perf_counter_ns()
        env = gym.make(args.task, cfg=env_cfg)
        env_t1 = time.perf_counter_ns()

        num_envs = env.unwrapped.num_envs

        with BenchmarkMonitor(benchmark, interval=1.0):
            step_times_s = stepping.run_runtime_loop(env, args.num_frames)

        benchmark.update_manual_recorders()

        startup = StartupTime(
            app_launch=(app_t1 - app_t0) / 1e9,
            env_creation=(env_t1 - env_t0) / 1e9,
            first_step=(step_times_s[0] if step_times_s else 0.0),
        )

        fps = [num_envs / t for t in step_times_s if t > 0]
        runtime = builders.build_runtime(
            startup_time_s=startup,
            iteration_times_s=step_times_s,
            collection_fps=fps,
            total_fps=fps,
            steps_per_iteration=num_envs,
        )

        versions = capture.capture_versions(benchmark)
        hardware = capture.capture_hardware(benchmark)
        resources = capture.capture_resources(benchmark)

        cfg = capture.run_config_from_presets(tokens)

        end_utc = capture.now_utc_iso()
        stamp = end_utc.translate(str.maketrans("", "", ":-"))[:15]

        seed = args.seed if args.seed is not None else 0
        run_id = capture.synth_run_id(None, cfg.physics_backend, args.task, seed, stamp)

        run = builders.build_run_identity(
            run_id=run_id,
            framework=None,
            config=cfg,
            task=args.task,
            seed=seed,
            start_utc=start_utc,
            end_utc=end_utc,
            num_envs=num_envs,
        )

        bundle = builders.build_runtime_bundle(
            run=run,
            versions=versions,
            hardware=hardware,
            runtime=runtime,
            resources=resources,
        )

        out_path = os.path.join(args.output_path, f"runtime_{args.task}.json")
        write_bundle_file(bundle, out_path)
        print(f"[runtime] wrote {out_path}")

        benchmark._finalize_impl()

        env.close()


if __name__ == "__main__":
    run(sys.argv[1:])
