# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the contributed MicroDuck velocity-tracking environments.

The environments spawn :data:`~isaaclab_assets.MICRODUCK_CFG`, whose USD is generated rather than
committed, so every test here skips when that asset is absent -- the same condition the asset
fidelity tests skip on. Generate it with
``uv run --extra importers python scripts/tools/convert_microduck.py``.
"""

import os

# TODO: Remove once usd-core>=26.5 is the minimum. Earlier releases can corrupt
# the heap while parsing Newton payloads concurrently, so disable USD concurrency
# before importing modules that may initialize OpenUSD.
os.environ["PXR_WORK_THREAD_LIMIT"] = "1"

import gymnasium as gym
import pytest

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets.robots.microduck import MICRODUCK_REGENERATE_COMMAND, MICRODUCK_USD_PATH

# Local imports should be imported last
from env_test_utils import _run_environments  # isort: skip

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.path.isfile(MICRODUCK_USD_PATH),
        reason=(
            f"MicroDuck USD asset is missing: {MICRODUCK_USD_PATH}. Generate it with '{MICRODUCK_REGENERATE_COMMAND}'."
        ),
    ),
]

TASK_NAMES = ["IsaacContrib-Velocity-Flat-MicroDuck", "IsaacContrib-Velocity-Rough-MicroDuck"]
"""The registered MicroDuck velocity tasks."""

ACTOR_OBSERVATION_DIM = 61
"""Actor observation width the deployed MicroDuck policy expects.

48 proprioception values plus the 13-wide command block ``[twist(3), head_pose(4), body_pose(6)]``;
see ``artifacts/microduck/upstream_reference.md`` section 7. The head and body pose commands are
not part of the task skeleton yet, so this is the contract the port is being built towards.
"""


@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_environment_steps_with_random_actions(task_name):
    """Each registered task builds, resets, and steps without producing invalid signals."""
    _run_environments(task_name, device="cuda", num_envs=2, num_steps=10)


@pytest.mark.xfail(reason="obs contract lands in Task 9", strict=False)
@pytest.mark.parametrize("task_name", TASK_NAMES)
def test_actor_observation_width_matches_the_deploy_contract(task_name):
    """The actor group is as wide as the policy deployed on the robot expects."""
    sim_utils.create_new_stage()
    env = None
    try:
        env_cfg = parse_env_cfg(task_name, device="cuda", num_envs=2)
        env = gym.make(task_name, cfg=env_cfg)
        env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore
        obs, _ = env.reset()

        assert obs["policy"].shape[-1] == ACTOR_OBSERVATION_DIM
    finally:
        if env is not None:
            env.close()
        SimulationContext.clear_instance()
