# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private compatibility helpers for running 3.0 benchmark scripts on Isaac Lab 2.x.

This module intentionally imports only the Python standard library at module
load time. Isaac Sim, Gymnasium, and Isaac Lab imports remain deferred until
the corresponding operation is called so 2.x can launch Kit before task
registration and configuration lookup.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

RUN_MANIFEST_FILENAME = "run.json"
RUN_MANIFEST_VERSION = 1
CHECKPOINT_SELECTORS = frozenset({"latest", "best"})

_LAUNCHER_ARGS_MARKER = "_isaaclab_benchmark_launcher_args_added"
_LAB2_SHA_ENV = "ISAACLAB_BENCHMARK_LAB2_SHA"
_LAB3_SHA_ENV = "ISAACLAB_BENCHMARK_LAB3_SHA"
_PHYSX_PRESETS = frozenset({"default", "physx"})
_UNSUPPORTED_SOLVERS = frozenset({"kamino", "newton", "newton_kamino", "newton_mjwarp", "ovphysx"})


def add_launcher_args(parser: argparse.ArgumentParser) -> None:
    """Add 2.x :class:`AppLauncher` arguments to a parser exactly once."""
    if getattr(parser, _LAUNCHER_ARGS_MARKER, False):
        return

    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    setattr(parser, _LAUNCHER_ARGS_MARKER, True)


def parse_benchmark_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> tuple[argparse.Namespace, list[str]]:
    """Parse benchmark arguments and retain supported 2.x Hydra overrides.

    The 3.0 ``presets=physx`` and ``physics=physx`` spellings are accepted as
    no-op compatibility selectors. Other typed presets are rejected before
    :class:`AppLauncher` is constructed because Isaac Lab 2.x only provides
    the PhysX backend used by this benchmark harness.
    """
    args, remaining = parser.parse_known_args(argv)
    hydra_args = _sanitize_hydra_args(remaining)
    sys.argv = [sys.argv[0], *hydra_args]
    return args, remaining


def launch_app(args: argparse.Namespace) -> Any:
    """Construct and return the 2.x :class:`AppLauncher`."""
    AppLauncher = importlib.import_module("isaaclab.app").AppLauncher
    return AppLauncher(args)


def resolve_task_config(
    task_name: str,
    agent_cfg_entry_point: str | None,
    hydra_args: list[str] | None = None,
) -> tuple[Any, Any]:
    """Resolve 2.x environment and agent configs after Kit has launched."""
    if hydra_args is not None:
        sys.argv = [sys.argv[0], *_sanitize_hydra_args(hydra_args)]

    hydra_task_config = importlib.import_module("isaaclab_tasks.utils.hydra").hydra_task_config

    resolved: dict[str, Any] = {}

    @hydra_task_config(task_name, agent_cfg_entry_point)
    def _capture(env_cfg: Any, agent_cfg: Any) -> None:
        resolved["env"] = env_cfg
        resolved["agent"] = agent_cfg

    _capture()
    if "env" not in resolved:
        raise RuntimeError(f"Unable to resolve task configuration for {task_name!r}.")
    return resolved["env"], resolved["agent"]


def apply_env_overrides(args_cli: argparse.Namespace, env_cfg: Any, *, apply_device: bool = True) -> None:
    """Apply common command-line overrides to a 2.x environment config."""
    if getattr(args_cli, "num_envs", None) is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "seed", None) is not None:
        env_cfg.seed = args_cli.seed
    if apply_device and not getattr(args_cli, "distributed", False):
        device = getattr(args_cli, "device", None)
        if device is not None:
            env_cfg.sim.device = device


def add_common_train_args(
    parser: argparse.ArgumentParser,
    *,
    agent_default: str | None,
    agent_help: str,
    include_agent: bool = True,
    include_distributed: bool = True,
) -> None:
    """Add the common 3.0 training CLI schema supported by 2.x scripts."""
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.add_argument(
        "--video_interval", type=int, default=2000, help="Interval between video recordings (in steps)."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    if include_agent:
        parser.add_argument("--agent", type=str, default=agent_default, help=agent_help)
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    if include_distributed:
        parser.add_argument(
            "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
        )
    parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
    parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
    parser.add_argument(
        "--ray-proc-id",
        "-rid",
        type=int,
        default=None,
        help="Automatically configured by Ray integration, otherwise None.",
    )
    parser.add_argument(
        "--capture_env_sensors",
        type=int,
        default=0,
        help="Number of environment views to capture from each image-like scene sensor.",
    )
    parser.add_argument(
        "--capture_env_sensors_length",
        type=int,
        default=200,
        help="Length of each captured sensor frame window (in steps).",
    )
    parser.add_argument(
        "--capture_env_sensors_interval",
        type=int,
        default=2000,
        help="Interval between captured sensor frame windows (in steps).",
    )
    parser.add_argument(
        "--capture_env_sensors_format",
        choices=["tensorboard", "file"],
        default="tensorboard",
        help="Format used to save captured sensor frames.",
    )


def enable_cameras_for_video(args_cli: argparse.Namespace) -> None:
    """Enable camera rendering when video or sensor capture is requested."""
    if getattr(args_cli, "video", False) or getattr(args_cli, "capture_env_sensors", 0) > 0:
        args_cli.enable_cameras = True


def create_isaaclab_env(
    task: str,
    env_cfg: Any,
    args_cli: argparse.Namespace,
    *,
    convert_marl_to_single_agent: bool,
) -> Any:
    """Create a Gymnasium environment with optional 2.x MARL conversion."""
    gym = importlib.import_module("gymnasium")
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array" if getattr(args_cli, "video", False) else None)
    if convert_marl_to_single_agent:
        envs = importlib.import_module("isaaclab.envs")

        if isinstance(env.unwrapped, envs.DirectMARLEnv):
            env = envs.multi_agent_to_single_agent(env)
    return env


def wrap_record_video(env: Any, log_dir: str, args_cli: argparse.Namespace) -> Any:
    """Wrap an environment with Gymnasium video recording when requested."""
    if not getattr(args_cli, "video", False):
        return env

    gym = importlib.import_module("gymnasium")
    print_dict = importlib.import_module("isaaclab.utils.dict").print_dict

    video_kwargs = {
        "video_folder": os.path.join(log_dir, "videos", "train"),
        "step_trigger": lambda step: step % args_cli.video_interval == 0,
        "video_length": args_cli.video_length,
        "disable_logger": True,
    }
    print("[INFO] Recording videos during training.")
    print_dict(video_kwargs, nesting=4)
    return gym.wrappers.RecordVideo(env, **video_kwargs)


def dispatch_library_entrypoint(
    argv: list[str] | None,
    entrypoints: dict[str, Path],
    *,
    action: str,
    description: str,
    library_help: str,
    run_as_script: bool = False,
) -> int:
    """Dispatch a unified benchmark entry point to a library implementation."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rl_library", choices=sorted(entrypoints))
    args_cli, library_args = parser.parse_known_args(argv)
    if args_cli.rl_library is None:
        help_parser = argparse.ArgumentParser(description=description)
        help_parser.add_argument("--rl_library", choices=sorted(entrypoints), required=True, help=library_help)
        help_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected library.")
        help_parser.print_help()
        return 0 if "-h" in argv or "--help" in argv else 2

    module_path = entrypoints[args_cli.rl_library]
    if run_as_script:
        original_argv = sys.argv
        original_path = list(sys.path)
        try:
            sys.argv = [str(module_path), *library_args]
            sys.path.insert(0, str(module_path.parent))
            runpy.run_path(str(module_path), run_name="__main__")
        finally:
            sys.argv = original_argv
            sys.path[:] = original_path
        return 0

    module = _import_local_module(f"isaaclab_benchmark_{action}_{args_cli.rl_library}", module_path)
    module.run(library_args)
    return 0


def resolve_benchmark_git_shas() -> dict[str, str]:
    """Resolve the validated full-SHA identity for both benchmark checkouts."""
    repo_root = Path(__file__).resolve().parents[2]
    sources = (
        ("lab2", "Lab 2", _LAB2_SHA_ENV, repo_root),
        ("lab3", "Lab 3", _LAB3_SHA_ENV, repo_root.parent / "lab3-develop"),
    )

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for key, label, environment_name, checkout_path in sources:
        override_sha = os.environ.get(environment_name)
        checkout_sha = _git_sha(checkout_path)
        if override_sha is not None:
            override_sha = _validate_git_sha(override_sha, f"Environment variable {environment_name}")
        if checkout_sha is not None:
            checkout_sha = _validate_git_sha(checkout_sha, f"{label} checkout at {str(checkout_path)!r}")
        if override_sha is not None and checkout_sha is not None and override_sha != checkout_sha:
            raise RuntimeError(
                f"{environment_name}={override_sha!r} does not match the {label} checkout at "
                f"{str(checkout_path)!r} ({checkout_sha}). Unset the override or set it to the checkout SHA."
            )
        resolved_sha = override_sha or checkout_sha
        if resolved_sha is None:
            missing.append(environment_name)
        else:
            resolved[key] = resolved_sha

    if missing:
        raise RuntimeError(
            "Unable to resolve both benchmark checkout Git SHAs. Set "
            f"{_LAB2_SHA_ENV} and {_LAB3_SHA_ENV} when the checkouts are not available to git."
        )
    return _validate_git_sha_pair(resolved, "Resolved benchmark provenance")


def write_run_manifest(
    log_dir: str,
    *,
    library: str,
    task: str,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write checkpoint-discovery metadata including both comparison SHAs."""
    run_dir = Path(log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    git_shas = resolve_benchmark_git_shas()
    manifest = {
        "version": RUN_MANIFEST_VERSION,
        "library": library,
        "task": _normalize_task_name(task),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "git_shas": git_shas,
    }
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    temporary_path = run_dir / f".{RUN_MANIFEST_FILENAME}.{os.getpid()}.tmp"
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, manifest_path)


def resolve_play_checkpoint(checkpoint: str | None, framework: str, task: str) -> str:
    """Resolve an explicit checkpoint or the 2.x published checkpoint."""
    if checkpoint:
        retrieve_file_path = importlib.import_module("isaaclab.utils.assets").retrieve_file_path
        return retrieve_file_path(checkpoint)

    get_published_pretrained_checkpoint = importlib.import_module(
        "isaaclab_rl.utils.pretrained_checkpoint"
    ).get_published_pretrained_checkpoint

    published_task = _normalize_task_name(task)
    path = get_published_pretrained_checkpoint(framework, published_task)
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint available for framework {framework!r} and task {task!r}; pass --checkpoint"
        )
    return path


def resolve_checkpoint_selector(
    log_root_path: str,
    selector: str,
    *,
    library: str,
    task: str,
    checkpoint_pattern: str,
    other_dirs: list[str] | None = None,
    preferred_checkpoint_pattern: str | None = None,
    metadata: dict[str, str] | None = None,
    expected_git_shas: dict[str, str] | None = None,
) -> str:
    """Resolve ``latest`` or ``best`` from compatible manifested runs, honoring a preferred checkpoint."""
    if selector not in CHECKPOINT_SELECTORS:
        raise ValueError(f"Unknown checkpoint selector {selector!r}. Expected one of: {sorted(CHECKPOINT_SELECTORS)}.")

    if expected_git_shas is None:
        expected_git_shas = resolve_benchmark_git_shas()
    expected_git_shas = _validate_git_sha_pair(expected_git_shas, "Expected checkpoint provenance")

    log_root = Path(log_root_path)
    expected_task = _normalize_task_name(task)
    expected_metadata = metadata or {}
    runs: list[tuple[datetime, Path]] = []
    if log_root.is_dir():
        for run_dir in log_root.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                manifest = json.loads((run_dir / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
                created_at = datetime.fromisoformat(manifest["created_at"])
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if manifest.get("version") != RUN_MANIFEST_VERSION:
                continue
            if manifest.get("library") != library or manifest.get("task") != expected_task:
                continue
            manifest_metadata = manifest.get("metadata", {})
            try:
                manifest_git_shas = _validate_git_sha_pair(manifest.get("git_shas"), f"Manifest in {run_dir}")
            except RuntimeError:
                continue
            if manifest_git_shas != expected_git_shas:
                continue
            if not isinstance(manifest_metadata, dict):
                continue
            if any(manifest_metadata.get(key) != value for key, value in expected_metadata.items()):
                continue
            runs.append((created_at, run_dir))

    for _, run_dir in sorted(runs, reverse=True):
        checkpoint_dir = run_dir.joinpath(*(other_dirs or []))
        if not checkpoint_dir.is_dir():
            continue
        checkpoints = [
            path for path in checkpoint_dir.iterdir() if path.is_file() and re.fullmatch(checkpoint_pattern, path.name)
        ]
        if not checkpoints:
            continue
        if preferred_checkpoint_pattern is not None:
            preferred = [path for path in checkpoints if re.fullmatch(preferred_checkpoint_pattern, path.name)]
            if preferred:
                checkpoints = preferred
        checkpoints.sort(key=lambda path: _natural_sort_key(path.name))
        return str(checkpoints[-1].resolve())

    raise ValueError(
        f"No compatible manifested run with a checkpoint was found in {str(log_root)!r}. "
        f"Run training with the current unified training entrypoint before using '--checkpoint {selector}'."
    )


def _sanitize_hydra_args(hydra_args: list[str]) -> list[str]:
    """Strip the PhysX compatibility selector and reject unsupported presets."""
    sanitized: list[str] = []
    for token in hydra_args:
        selector, separator, raw_value = token.partition("=")
        if separator and selector in {"physics", "presets"}:
            values = [value.strip() for value in raw_value.split(",") if value.strip()]
            if values and all(value in _PHYSX_PRESETS for value in values):
                continue
            unsupported = next((value for value in values if value in _UNSUPPORTED_SOLVERS), raw_value)
            raise ValueError(
                f"Solver preset {unsupported!r} requires Isaac Lab 3.0. "
                "Isaac Lab 2.x benchmark compatibility supports PhysX only; use 'presets=physx' or omit the preset."
            )
        if token in _PHYSX_PRESETS:
            continue
        if token in _UNSUPPORTED_SOLVERS:
            raise ValueError(
                f"Solver preset {token!r} requires Isaac Lab 3.0. "
                "Isaac Lab 2.x benchmark compatibility supports PhysX only; use 'presets=physx' or omit the preset."
            )
        sanitized.append(token)
    return sanitized


def _import_local_module(module_name: str, module_path: Path) -> ModuleType:
    """Import a module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_git_sha(value: object, source: str) -> str:
    """Validate and normalize one full Git SHA."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise RuntimeError(f"{source} must contain a full 40-character Git SHA; got {value!r}.")
    return value.lower()


def _validate_git_sha_pair(git_shas: object, source: str) -> dict[str, str]:
    """Validate and normalize a Lab 2/Lab 3 full-SHA identity pair."""
    if not isinstance(git_shas, dict) or set(git_shas) != {"lab2", "lab3"}:
        raise RuntimeError(f"{source} must contain exactly the 'lab2' and 'lab3' Git SHAs.")

    normalized: dict[str, str] = {}
    for checkout in ("lab2", "lab3"):
        normalized[checkout] = _validate_git_sha(git_shas[checkout], f"{source} entry {checkout!r}")
    return normalized


def _git_sha(path: Path) -> str | None:
    """Return the Git SHA for a checkout, or ``None`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _normalize_task_name(task: str) -> str:
    """Normalize training and play task variants to one manifest identity."""
    return task.split(":")[-1].removesuffix("-Play")


def _natural_sort_key(value: str) -> list[int | str]:
    """Return a key that sorts numeric filename components by value."""
    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", value)]
