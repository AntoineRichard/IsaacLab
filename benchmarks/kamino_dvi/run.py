# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-GPU subprocess execution for the Kamino DVI benchmark."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from .matrix import expand_full_runs, expand_preflights
from .models import BenchmarkMatrix, RunIdentity, TaskName, Variant


@dataclass(frozen=True)
class ProcessOutcome:
    """Terminal subprocess status returned to failure classification."""

    returncode: int | None
    timed_out: bool


def _terminate_process_group(pid: int) -> None:
    os.killpg(pid, signal.SIGTERM)


def execute_command(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    *,
    timeout_s: int,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    kill_process_group: Callable[[int], None] = _terminate_process_group,
) -> ProcessOutcome:
    """Execute one training command with streamed logs and a hard timeout."""
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = popen_factory(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_group(process.pid)
            process.communicate()
    return ProcessOutcome(returncode=process.returncode, timed_out=timed_out)


def execute_sequentially(
    commands: Iterable[list[str]],
    executor: Callable[[list[str]], ProcessOutcome],
) -> tuple[ProcessOutcome, ...]:
    """Execute every command in order, retaining failures without stopping."""
    return tuple(executor(command) for command in commands)


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark-runner command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--preflight-only", action="store_true")
    phase.add_argument("--full-only", action="store_true")
    parser.add_argument("--task", choices=[task.value for task in TaskName])
    parser.add_argument("--variant", choices=[variant.value for variant in Variant])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def select_identities(matrix: BenchmarkMatrix, args: argparse.Namespace) -> tuple[RunIdentity, ...]:
    """Expand and filter run identities without performing side effects."""
    identities: tuple[RunIdentity, ...]
    if args.preflight_only:
        identities = expand_preflights(matrix)
    elif args.full_only:
        identities = expand_full_runs(matrix)
    else:
        identities = expand_preflights(matrix) + expand_full_runs(matrix)
    if args.task is not None:
        task = TaskName(args.task)
        identities = tuple(identity for identity in identities if identity.task is task)
    if args.variant is not None:
        variant = Variant(args.variant)
        identities = tuple(identity for identity in identities if identity.variant is variant)
    if args.seed is not None:
        identities = tuple(identity for identity in identities if identity.seed == args.seed)
    return identities
