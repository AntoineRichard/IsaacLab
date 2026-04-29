# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.cli.parse_args`."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.odin.asgard.cli import parse_args, parse_seed_list


def test_parse_seed_list_single():
    assert parse_seed_list("42") == [42]


def test_parse_seed_list_multiple():
    assert parse_seed_list("42,43,44") == [42, 43, 44]


def test_parse_seed_list_strips_whitespace():
    assert parse_seed_list(" 42 , 43 ") == [42, 43]


def test_parse_seed_list_rejects_non_int():
    with pytest.raises(ValueError):
        parse_seed_list("42,foo,43")


def test_parse_args_minimal():
    args = parse_args(
        [
            "--fleet",
            "fleet.yaml",
            "--physx-yaml",
            "physx.yaml",
            "--seeds",
            "42",
        ]
    )
    assert str(args.fleet) == "fleet.yaml"
    assert args.physx_yaml == Path("physx.yaml")
    assert args.newton_yaml is None
    assert args.seeds == [42]
    assert args.fresh is False
    assert args.skip_preflight is False
    assert args.per_job_timeout == 14400


def test_parse_args_all_flags():
    args = parse_args(
        [
            "--fleet",
            "fleet.yaml",
            "--physx-yaml",
            "physx.yaml",
            "--newton-yaml",
            "newton.yaml",
            "--seeds",
            "42,43",
            "--include",
            "Isaac-Ant-*",
            "Isaac-Humanoid-*",
            "--resume",
            "LATEST",
            "--fresh",
            "--skip-preflight",
            "--per-job-timeout",
            "7200",
            "--max-infrastructure-retries",
            "5",
            "--retry-failed",
            "run1,run2",
            "--verbose",
        ]
    )
    assert args.seeds == [42, 43]
    assert args.include == ["Isaac-Ant-*", "Isaac-Humanoid-*"]
    assert args.resume == "LATEST"
    assert args.fresh is True
    assert args.skip_preflight is True
    assert args.per_job_timeout == 7200
    assert args.max_infrastructure_retries == 5
    assert args.retry_failed == ["run1", "run2"]
    assert args.verbose is True


def test_parse_args_requires_at_least_one_yaml():
    with pytest.raises(SystemExit):
        parse_args(["--fleet", "fleet.yaml", "--seeds", "42"])


def test_parse_args_no_circuit_breaker_default_off():
    args = parse_args(
        [
            "--fleet",
            "fleet.yaml",
            "--physx-yaml",
            "physx.yaml",
            "--seeds",
            "42",
        ]
    )
    assert args.no_circuit_breaker is False


def test_parse_args_no_circuit_breaker_set():
    args = parse_args(
        [
            "--fleet",
            "fleet.yaml",
            "--physx-yaml",
            "physx.yaml",
            "--seeds",
            "42",
            "--no-circuit-breaker",
        ]
    )
    assert args.no_circuit_breaker is True
