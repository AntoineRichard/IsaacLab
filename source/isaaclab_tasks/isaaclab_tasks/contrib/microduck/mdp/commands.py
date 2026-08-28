# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms MicroDuck needs that have no stock Isaac Lab counterpart.

Both terms are ported from ``pollen-robotics/microduck_rl``; see section 6 of
``artifacts/microduck/upstream_reference.md`` for the verbatim upstream formulas.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class JointPoseCommand(CommandTerm):
    """Generic N-dimensional uniform pose-delta command.

    Each dimension is drawn independently and uniformly from its own range and then held until the
    next resample. The width of the command is the length of :attr:`JointPoseCommandCfg.ranges`,
    so one term serves both MicroDuck pose commands:

    * ``head_pose`` -- 4 joint-position deltas [rad] from the stand pose, in the upstream servo
      order ``(neck_pitch, head_pitch, head_yaw, head_roll)``.
    * ``body_pose`` -- a 6-dimensional trunk-pose delta from the nominal stand, ordered
      ``(x, y, z)`` [m] then ``(roll, pitch, yaw)`` [rad].

    Upstream calls this ``UniformPoseCommand`` (reference section 6). It carries no metrics and no
    debug visualization on purpose: an environment holds several of these and none of them tracks
    an error the command term itself could measure -- the reward terms do that.

    The ranges are read on every resample rather than cached, so a curriculum can widen them by
    reassigning :attr:`JointPoseCommandCfg.ranges` on the live term configuration. Widening only:
    the command width is fixed at construction.
    """

    cfg: JointPoseCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: JointPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.dim = len(cfg.ranges)
        """Number of commanded dimensions."""

        self._command = torch.zeros(self.num_envs, self.dim, device=self.device)

    def __str__(self) -> str:
        msg = "JointPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The commanded pose delta. Shape is (num_envs, dim)."""
        return self._command

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        num_envs = len(env_ids)
        if num_envs == 0:
            return
        r = torch.empty(num_envs, device=self.device)
        for dim, (low, high) in enumerate(self.cfg.ranges):
            self._command[env_ids, dim] = r.uniform_(low, high)


@configclass
class JointPoseCommandCfg(CommandTermCfg):
    """Configuration for the N-dimensional uniform pose-delta command term.

    Please refer to the :class:`JointPoseCommand` class for more details.
    """

    class_type: type[JointPoseCommand] = JointPoseCommand

    ranges: tuple[tuple[float, float], ...] = ()
    """Per-dimension ``(low, high)`` sampling range. Its length sets the command width.

    Left as a plain tuple rather than a structured range class because a curriculum reassigns it
    wholesale as the training progresses.
    """


class MicroDuckVelocityCommand(UniformVelocityCommand):
    """Velocity command with MicroDuck's forward-only and turn-in-place buckets.

    On top of the stock uniform sampling and the standing bucket, a resample assigns two further
    buckets (reference sections 2.7 and 6):

    * **forward-only** (:attr:`MicroDuckVelocityCommandCfg.rel_forward_envs`) -- the surge command
      is rectified and floored, ``vx = |vx|.clamp(min=0.3)`` [m/s], with sway and yaw rate zeroed.
      This is upstream's base-template behaviour, which MicroDuck inherits.
    * **turn-in-place** (:attr:`MicroDuckVelocityCommandCfg.rel_turn_in_place_envs`) -- the linear
      command is zeroed and the yaw rate is forced to ``±U(0.4 * max, max)`` [rad/s], where ``max``
      is the larger magnitude of :attr:`~UniformVelocityCommandCfg.Ranges.ang_vel_z`. Independent
      uniform sampling almost never produces "no translation, large rotation", so spinning on the
      spot would otherwise go untrained.

    The buckets are independent draws, not a partition, so an environment can fall in several of
    them. Upstream resolves the overlap by ordering, and this term reproduces it exactly:
    turn-in-place beats standing beats forward-only. Forward-only rewrites the command in place;
    standing is a flag the base class re-zeroes the command from on every step, which overrides a
    forward-only value; and turn-in-place is applied last and clears the standing flag for its
    environments, so it survives both.

    Both fractions and :attr:`~UniformVelocityCommandCfg.rel_standing_envs` are read on every
    resample, so a curriculum can ramp them on the live term configuration.
    """

    cfg: MicroDuckVelocityCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: MicroDuckVelocityCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.is_forward_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether each environment is currently in the forward-only bucket."""

    def __str__(self) -> str:
        msg = super().__str__().replace("UniformVelocityCommand:", "MicroDuckVelocityCommand:", 1)
        msg += f"\n\tForward-only probability: {self.cfg.rel_forward_envs}"
        msg += f"\n\tTurn-in-place probability: {self.cfg.rel_turn_in_place_envs}"
        return msg

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # the bucket masks index into ``env_ids``, which the command manager may hand over as a
        # plain sequence
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._resample_command(env_ids)

        r = torch.empty(len(env_ids), device=self.device)
        self.is_forward_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_forward_envs
        forward_ids = env_ids[self.is_forward_env[env_ids]]
        if len(forward_ids) > 0:
            self.vel_command_b[forward_ids, 0] = self.vel_command_b[forward_ids, 0].abs().clamp(min=0.3)
            self.vel_command_b[forward_ids, 1] = 0.0
            self.vel_command_b[forward_ids, 2] = 0.0

        if self.cfg.rel_turn_in_place_envs <= 0.0:
            return
        turn_ids = env_ids[r.uniform_(0.0, 1.0) < self.cfg.rel_turn_in_place_envs]
        if len(turn_ids) == 0:
            return
        self.vel_command_b[turn_ids, :2] = 0.0
        low, high = self.cfg.ranges.ang_vel_z
        max_rate = max(abs(low), abs(high))
        turn_r = torch.empty(len(turn_ids), device=self.device)
        sign = torch.where(turn_r.uniform_(0.0, 1.0) < 0.5, -1.0, 1.0)
        magnitude = turn_r.uniform_(0.4 * max_rate, max_rate)
        self.vel_command_b[turn_ids, 2] = sign * magnitude
        # these environments must actually turn, so un-mark them as standing -- otherwise
        # ``_update_command`` would zero the command they were just given
        self.is_standing_env[turn_ids] = False


@configclass
class MicroDuckVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the MicroDuck velocity command term.

    Please refer to the :class:`MicroDuckVelocityCommand` class for more details.
    """

    class_type: type[MicroDuckVelocityCommand] = MicroDuckVelocityCommand

    rel_forward_envs: float = 0.0
    """Probability that an environment is commanded to walk straight forward. Defaults to 0.0.

    Such an environment gets ``vx = |vx|.clamp(min=0.3)`` [m/s] and zero sway and yaw rate.
    """

    rel_turn_in_place_envs: float = 0.0
    """Probability that an environment is commanded to turn on the spot. Defaults to 0.0.

    Such an environment gets a zero linear command and a yaw rate of ``±U(0.4 * max, max)``
    [rad/s], where ``max`` is the larger magnitude of :attr:`Ranges.ang_vel_z`.
    """
