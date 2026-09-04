# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck lift task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckLiftPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck lift task.

    Differs from the velocity runner in the two fields every sibling differs in, so the network
    shapes, the optimizer, the asymmetric actor-critic wiring and the 24-step rollout stay the
    family's.
    """

    experiment_name = "microduck_lift"
    # **4000, not the family's 20000.** The pick-and-place task's own logs settle the question: it
    # had solved its objective by iteration 2000 and the remaining 18000 bought almost nothing, and
    # this task is strictly simpler -- no locomotion, no drop point, no curriculum, and a five-second
    # episode. Asking for a budget the evidence says will not be used costs cluster time that other
    # people are queueing for.
    max_iterations = 4000
