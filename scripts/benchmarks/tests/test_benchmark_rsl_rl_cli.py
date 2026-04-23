# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI-level tests for benchmark_rsl_rl.py.

These tests exercise only the argparse layer — they do not import the
whole script (which launches Isaac Sim at import time). A minimal reimport
of the argparse setup is shared via ``_build_parser``.
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Mirror of the parser setup in benchmark_rsl_rl.py.

    Kept in lockstep with the script; when a new flag is added there,
    add it here too.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--backend", choices=["physx", "newton"], default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--schema_v1_output", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser


def test_log_dir_flag_defaults_none():
    args = _build_parser().parse_args([])
    assert args.log_dir is None


def test_log_dir_flag_captured():
    args = _build_parser().parse_args(["--log_dir", "/tmp/bundle/training_data"])
    assert args.log_dir == "/tmp/bundle/training_data"
