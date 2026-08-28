# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms MicroDuck needs that have no stock Isaac Lab counterpart.

Ported from ``pollen-robotics/microduck_rl``; see sections 3 and 9 of
``artifacts/microduck/upstream_reference.md`` for the upstream processing chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from .events import encoder_bias

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BiasedJointPositionAction(JointPositionAction):
    r"""Joint position action that compensates the joint-encoder calibration error.

    The servo closes its position loop on the encoder reading, so a command of :math:`q^*` parks
    the joint at :math:`q^* - b`, where :math:`b` is the encoder bias
    (:func:`~isaaclab_tasks.contrib.microduck.mdp.events.encoder_bias`). Upstream reproduces this by
    subtracting the bias from the target (reference section 3):

    .. math::

        \text{target} = \text{default} + \text{scale} \times \text{action} - \text{bias}

    This is the action-side half of the encoder-bias randomization, and it only makes sense
    together with the observation-side half,
    :func:`~isaaclab_tasks.contrib.microduck.mdp.observations.joint_pos_rel_biased`: the actor sees
    ``joint_pos + bias`` and the target subtracts ``bias``, so a policy commanding ``a`` reads back
    ``a`` while the true joint sits at ``default + a``. What the bias then perturbs is the *robot*,
    not the policy's frame of reference -- the two joints of a leg no longer agree on where zero
    is, which is exactly the calibration error the term models. Using only one half instead trains
    a policy against a permanent offset in its own commands.

    Both halves read the same tensor, so a task must give this term and the biased observation the
    same asset.
    """

    cfg: BiasedJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: BiasedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # resolved eagerly so a mismatched asset fails at construction rather than at the first step
        self._encoder_bias = encoder_bias(env, SceneEntityCfg(cfg.asset_name))

    def apply_actions(self):
        # written out rather than delegated: this runs once per physics step, so subtracting the
        # bias into ``processed_actions`` would compound it over the decimation window
        target = self.processed_actions - self._encoder_bias[:, self._joint_ids]
        self._asset.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)


@configclass
class BiasedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the encoder-bias-compensating joint position action term.

    Please refer to the :class:`BiasedJointPositionAction` class for more details.
    """

    class_type: type[BiasedJointPositionAction] = BiasedJointPositionAction
