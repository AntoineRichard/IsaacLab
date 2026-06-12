# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark backend descriptors (pure data, Isaac-Sim-free)."""

from isaaclab.test.benchmark.backend_descriptor import BACKEND_DESCRIPTORS, BackendDescriptor


def test_all_four_backends_present():
    assert set(BACKEND_DESCRIPTORS) == {"rsl_rl", "rl_games", "skrl", "sb3"}
    for d in BACKEND_DESCRIPTORS.values():
        assert isinstance(d, BackendDescriptor)


def test_rsl_rl_tags():
    d = BACKEND_DESCRIPTORS["rsl_rl"]
    assert d.cfg_entry_point == "rsl_rl_cfg_entry_point"
    assert d.reward_tag == "Train/mean_reward"
    assert d.ep_length_tag == "Train/mean_episode_length"
    assert d.tfevents_pattern == "events*"


def test_rl_games_tags():
    d = BACKEND_DESCRIPTORS["rl_games"]
    assert d.cfg_entry_point == "rl_games_cfg_entry_point"
    assert d.reward_tag == "rewards/iter"
    assert d.ep_length_tag == "episode_lengths/iter"
    assert d.tfevents_pattern == "summaries/events*"


def test_sb3_uses_subdir_glob():
    assert BACKEND_DESCRIPTORS["sb3"].tfevents_pattern == "PPO_*/events*"


def test_skrl_space_padded_tags():
    d = BACKEND_DESCRIPTORS["skrl"]
    assert d.cfg_entry_point == "skrl_cfg_entry_point"
    assert d.reward_tag == "Reward / Total reward (mean)"
    assert d.ep_length_tag == "Episode / Total timesteps (mean)"
