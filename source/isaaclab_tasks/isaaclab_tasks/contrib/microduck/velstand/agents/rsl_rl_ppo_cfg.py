# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck velocity-plus-recovery task.

Every hyper-parameter is upstream's; see ``artifacts/microduck/upstream_reference_tasks3.md``
section 3.7.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckVelStandPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck velocity-plus-fall-recovery task.

    Upstream's runner differs from its velocity one in exactly two fields, so the rest is inherited
    rather than restated: the network shapes, the optimizer, the asymmetric actor-critic wiring and
    the 24-step rollout are shared across the whole family.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_velstand"
    # Upstream's budget, and the second largest in the family after the walking task's. The recovery
    # curricula have all finished ramping by iteration 2500, but the task still has to learn to walk
    # *and* to get up on the same 61-wide vector.
    max_iterations = 20000
