# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck pick-and-place task.

There is no upstream runner to transcribe: this task has no upstream counterpart. The two fields
below are the designer's, and their justification is in
``artifacts/microduck/pickplace/DESIGN.md`` §9.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckPickPlacePPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck pick-and-place task.

    Differs from the velocity runner in exactly the two fields every sibling differs in, so the rest
    is inherited rather than restated: the network shapes, the optimizer, the asymmetric
    actor-critic wiring and the 24-step rollout are shared across the whole family. Symmetry
    augmentation is off, as it is on every task but the forward roll -- and here it would be wrong
    rather than merely unhelpful, since mirroring the robot without mirroring the object and the
    drop point would relabel a left-hand reach as a right-hand one.

    The observation groups are *not* the family's shape, so the inherited asymmetric wiring is doing
    real work here: the actor is 55 wide and the critic 71, against the walking family's 61 and 76.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_pickplace"
    # The family's ceiling, and this task has the strongest claim on it. It carries a locomotion
    # sub-problem *and* a manipulation one, and its object curriculum only reaches full width at
    # iteration 2000 -- so the first two thousand iterations are spent learning a task that is
    # strictly easier than the one being evaluated. At the family's sizing constant (98,304
    # environment steps per iteration at 4096 environments, ~1.2 s per iteration for
    # allcollisions-class plants) this is about 6.7 hours.
    max_iterations = 20000
