# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-backend descriptors for the unified training benchmark.

Pure data — no RL library imports. The thin per-backend adapters under
``scripts/benchmarks/<backend>/`` consume these to know each framework's config
entry point, where its TensorBoard events land, and which scalar tags carry the
reward / episode-length / FPS series. Framework-specific *launch* logic stays in
the adapters; only the declarative metadata lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.test.benchmark.schema import Framework


@dataclass(frozen=True)
class BackendDescriptor:
    """Declarative metadata for one RL backend's benchmark integration.

    Args:
        framework: Schema framework id.
        cfg_entry_point: gym registry key for the agent config (e.g.
            ``"rsl_rl_cfg_entry_point"``); for skrl this is the PPO default and the
            adapter may resolve an algorithm-specific entry point.
        tfevents_pattern: Glob (relative to the run ``log_dir``) matching the
            TensorBoard events file — ``"events*"`` (root), ``"summaries/events*"``
            (rl_games), or ``"PPO_*/events*"`` (sb3).
        reward_tag: TensorBoard scalar tag for mean reward per iteration.
        ep_length_tag: TensorBoard scalar tag for mean episode length per iteration.
        total_fps_tag: TensorBoard scalar tag carrying total/effective FPS, or
            ``None`` when the adapter must derive it (e.g. rsl_rl derives from
            ``Perf/collection_time``).
        collection_fps_tag: TensorBoard scalar tag for environment-stepping FPS, or
            ``None`` when the adapter derives it.
    """

    framework: Framework
    cfg_entry_point: str
    tfevents_pattern: str
    reward_tag: str
    ep_length_tag: str
    total_fps_tag: str | None = None
    collection_fps_tag: str | None = None


BACKEND_DESCRIPTORS: dict[str, BackendDescriptor] = {
    "rsl_rl": BackendDescriptor(
        framework="rsl_rl",
        cfg_entry_point="rsl_rl_cfg_entry_point",
        tfevents_pattern="events*",
        reward_tag="Train/mean_reward",
        ep_length_tag="Train/mean_episode_length",
        total_fps_tag="Perf/total_fps",
        collection_fps_tag=None,
    ),
    "rl_games": BackendDescriptor(
        framework="rl_games",
        cfg_entry_point="rl_games_cfg_entry_point",
        tfevents_pattern="summaries/events*",
        reward_tag="rewards/iter",
        ep_length_tag="episode_lengths/iter",
        total_fps_tag="performance/step_inference_rl_update_fps",
        collection_fps_tag="performance/step_inference_fps",
    ),
    "skrl": BackendDescriptor(
        framework="skrl",
        cfg_entry_point="skrl_cfg_entry_point",
        tfevents_pattern="events*",
        reward_tag="Reward / Total reward (mean)",
        ep_length_tag="Episode / Total timesteps (mean)",
        total_fps_tag=None,
        collection_fps_tag=None,
    ),
    "sb3": BackendDescriptor(
        framework="sb3",
        cfg_entry_point="sb3_cfg_entry_point",
        tfevents_pattern="PPO_*/events*",
        reward_tag="rollout/ep_rew_mean",
        ep_length_tag="rollout/ep_len_mean",
        total_fps_tag="time/fps",
        collection_fps_tag=None,
    ),
}
