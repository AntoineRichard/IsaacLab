# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms modelling the sensor imperfections MicroDuck trains against.

The policy deployed on the robot reads servo encoders and an IMU over a bus, so upstream corrupts
the actor observations in three systematic ways that per-step noise cannot express: a constant
per-robot encoder bias, a constant per-robot IMU mounting misalignment, and a stochastic bus
latency. The critic keeps the true values -- it is privileged -- plus a few sensor-derived terms
that are guarded against non-finite reads.

Every term is ported from ``pollen-robotics/microduck_rl``; the upstream sources are quoted in
sections 2.3, 5, 6 and 8 of ``artifacts/microduck/upstream_reference.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply, quat_from_angle_axis

from .events import encoder_bias

if TYPE_CHECKING:
    from collections.abc import Callable

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor

_IMU_MISALIGNMENT_ATTR = "_microduck_imu_misalignment"
"""Attribute the per-environment IMU misalignment rotation is cached under on the environment."""


"""
Encoder bias.
"""


def joint_pos_rel_biased(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    biased: bool = True,
) -> torch.Tensor:
    """Joint positions w.r.t. the default joint positions, as the encoders report them.

    With ``biased=True`` this is the stock :func:`~isaaclab.envs.mdp.observations.joint_pos_rel`
    plus the per-environment encoder bias (see
    :func:`~isaaclab_tasks.contrib.microduck.mdp.events.encoder_bias`), which is what the servos on
    the real robot report. With ``biased=False`` it is the stock term exactly. Upstream feeds the
    biased value to the actor and the true one to the critic (reference sections 2.3 and 5).

    The bias is a *closed loop* with
    :class:`~isaaclab_tasks.contrib.microduck.mdp.actions.BiasedJointPositionAction`, which
    subtracts the same offset from its position target: a policy commanding ``a`` reads ``a`` back
    while the true joint settles at ``default + a - bias``. What the bias perturbs is therefore the
    robot's actual posture, not the policy's frame of reference -- exactly the calibration error it
    models. Wiring this term without the matching action term breaks that loop and instead trains a
    permanent offset into the policy's own commands.

    Args:
        env: The environment.
        asset_cfg: The articulation and the joints to report, in the requested order.
        biased: Whether to add the encoder bias. Defaults to True.

    Returns:
        Joint positions relative to their defaults [m or rad, depending on joint type].
        Shape is (num_envs, num_selected_joints).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    if biased:
        joint_pos = joint_pos + encoder_bias(env, asset_cfg)[:, asset_cfg.joint_ids]
    return joint_pos - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]


"""
IMU misalignment.
"""


def _imu_misalignment_quat(env: ManagerBasedEnv, max_angle_rad: float) -> torch.Tensor:
    """The per-environment IMU mounting-misalignment rotation, sampled once per run.

    Returns:
        The misalignment quaternion in (x, y, z, w). Shape is (num_envs, 4).
    """
    cached: tuple[float, torch.Tensor] | None = getattr(env, _IMU_MISALIGNMENT_ATTR, None)
    if cached is not None:
        angle, quat = cached
        if angle != max_angle_rad:
            raise ValueError(
                "The IMU misalignment is one rotation shared by every misaligned observation term, so"
                f" they must agree on 'max_angle_deg'. Sampled with {math.degrees(angle)} deg, now asked"
                f" for {math.degrees(max_angle_rad)} deg."
            )
        return quat
    # a uniformly random axis, so the misalignment has no preferred direction
    axis = torch.randn(env.num_envs, 3, device=env.device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    angles = torch.rand(env.num_envs, device=env.device) * max_angle_rad
    quat = quat_from_angle_axis(angles, axis)
    setattr(env, _IMU_MISALIGNMENT_ATTR, (max_angle_rad, quat))
    return quat


def projected_gravity_imu_misaligned(
    env: ManagerBasedEnv,
    max_angle_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gravity projected on the root frame, seen through a misaligned IMU.

    The IMU on the real robot is bolted on with a small, unknown, and *fixed* orientation error, so
    the misalignment is drawn once per environment and never resampled -- it is a property of the
    robot, like a mass offset, not a per-step disturbance (reference section 6).

    The rotation is ``U(0, max_angle_deg)`` about a uniformly random axis. It is deliberately
    zero-centred: it trains tolerance to the *magnitude* of a mounting error, not to a particular
    tilt. Upstream corrects the systematic pitch offset of its own board in the runtime instead of
    baking it into the policy.

    Upstream applies this to the actor only; the critic keeps the stock
    :func:`~isaaclab.envs.mdp.observations.projected_gravity`. Both misaligned terms share one
    rotation per environment, so they must be configured with the same ``max_angle_deg``.

    Args:
        env: The environment.
        max_angle_deg: Upper bound [deg] on the misalignment angle. Zero disables the term, which
            then reproduces the stock observation exactly.
        asset_cfg: The articulation carrying the IMU.

    Returns:
        The misaligned gravity direction in the root frame. Shape is (num_envs, 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(quat, asset.data.projected_gravity_b.torch)


def base_ang_vel_imu_misaligned(
    env: ManagerBasedEnv,
    max_angle_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Root angular velocity, seen through the same misaligned IMU as the gravity term.

    See :func:`projected_gravity_imu_misaligned` for the misalignment model.

    Args:
        env: The environment.
        max_angle_deg: Upper bound [deg] on the misalignment angle.
        asset_cfg: The articulation carrying the IMU.

    Returns:
        The misaligned angular velocity [rad/s] in the root frame. Shape is (num_envs, 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat = _imu_misalignment_quat(env, math.radians(max_angle_deg))
    return quat_apply(quat, asset.data.root_ang_vel_b.torch)


"""
Observation delay.
"""


class delayed_observation(ManagerTermBase):
    """Wraps another observation term in the stochastic bus latency upstream models.

    The value the policy reads on the robot is between zero and a few control periods old,
    depending on when the sample landed relative to the control tick. Upstream expresses this as a
    per-environment integer lag re-drawn on a slow cadence, so that a lag persists over many steps
    instead of flickering every step -- a policy can filter out flicker, but not a latency regime
    (reference section 8). MicroDuck uses two settings:

    * ``base_ang_vel`` and ``projected_gravity``: ``min_lag=0``, ``max_lag=1``,
      ``update_period=64`` -- a 0 or 20 ms latency that switches at most every 64 control steps;
    * ``joint_vel``: ``min_lag=1``, ``max_lag=1``, ``update_period=0`` -- a constant 20 ms lag,
      because the servo firmware derives velocity from the previous position-sample window.

    The lag update follows upstream exactly (reference section 8):

    #. the newest sample is pushed first, so lag 0 is the current value;
    #. with ``update_period = N > 0`` a lag is re-drawn only when ``(step + phase) % N == 0``,
       where ``phase`` is a per-environment offset drawn at construction and again on reset, which
       staggers the environments instead of switching them all on the same step;
    #. ``hold_prob`` further suppresses a re-draw, giving temporally correlated latency; upstream
       leaves it at 0.0;
    #. a lag is clamped to the frames actually buffered, so a freshly reset environment gets the
       newest frame rather than a stale one from before the reset;
    #. the buffer advances exactly once per control step, however often the observation manager
       recomputes the group within that step -- the term memoizes its output against the
       environment's ``common_step_counter``, and advances on every call for an environment that
       has none.

    Rule 5 is why a reset does not clear the history immediately. An environment that computes a
    final observation before resetting part of the batch calls the term twice within one control
    step with a reset in between; clearing the history there and refilling it on the second call
    would advance the buffer twice, and every environment that was *not* reset would silently skip
    a frame. The clear is therefore deferred to the next advance, and the second call serves the
    reset environments the fresh value -- which is what they would read anyway, having no history
    left to be delayed by -- while the rest keep the frame they were already served.

    One deviation from upstream's pipeline: mjlab orders it compute, noise, then delay, so the
    stale frame carries the noise sample it was drawn with; here the term wraps the *compute*
    stage, so Isaac Lab's noise is applied fresh to the stale frame. Both give a stale signal plus
    an independent uniform draw, so the observation distribution is the same; only the particular
    noise sample attached to a repeated frame differs.

    Isaac Lab's :class:`~isaaclab.utils.buffers.DelayBuffer` resamples its lag only on reset and
    has no ``update_period``, phase or hold concept, so this term keeps its own lag state over the
    same :class:`~isaaclab.utils.buffers.CircularBuffer` that ``DelayBuffer`` wraps. Doing the lag
    bookkeeping here also keeps it free of the host-device synchronization
    :meth:`~isaaclab.utils.buffers.DelayBuffer.set_time_lag` performs on every call.

    ``per_env`` and ``per_env_phase`` are not exposed: upstream leaves both at their per-environment
    defaults, and a shared lag would defeat the point of the term.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._min_lag = int(cfg.params.get("min_lag", 0))
        self._max_lag = int(cfg.params.get("max_lag", 0))
        self._update_period = int(cfg.params.get("update_period", 0))
        self._hold_prob = float(cfg.params.get("hold_prob", 0.0))
        if self._max_lag < 1:
            raise ValueError(f"A delayed observation needs 'max_lag' >= 1, got {self._max_lag}.")
        if not 0 <= self._min_lag <= self._max_lag:
            raise ValueError(f"Expected 0 <= min_lag <= max_lag, got ({self._min_lag}, {self._max_lag}).")
        if self._update_period < 0:
            raise ValueError(f"A delayed observation needs 'update_period' >= 0, got {self._update_period}.")
        if not 0.0 <= self._hold_prob <= 1.0:
            raise ValueError(f"Expected 'hold_prob' in [0, 1], got {self._hold_prob}.")

        self._buffer = CircularBuffer(max_len=self._max_lag + 1, batch_size=self.num_envs, device=self.device)
        self._lags = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase_offsets = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # environments reset since the last advance; their history is cleared on the next one
        self._pending_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._has_pending_reset = False
        # the step the buffer was last advanced on, so a recomputed group does not advance it twice
        self._last_step: int | None = None
        self._output: torch.Tensor | None = None
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        resolved = slice(None) if env_ids is None else env_ids
        self._lags[resolved] = 0
        self._step_count[resolved] = 0
        if self._update_period > 0:
            phases = torch.randint(0, self._update_period, (self.num_envs,), device=self.device)
            self._phase_offsets[resolved] = phases[resolved]
        # the history itself is cleared on the next advance, so that a reset landing between two
        # computes of one control step cannot cost the untouched environments a frame
        self._pending_reset[resolved] = True
        self._has_pending_reset = True

    def __call__(
        self,
        env: ManagerBasedEnv,
        term_func: Callable[..., torch.Tensor],
        term_params: dict | None = None,
        min_lag: int = 0,
        max_lag: int = 0,
        update_period: int = 0,
        hold_prob: float = 0.0,
    ) -> torch.Tensor:
        """Compute the wrapped term and return a stale copy of it.

        Args:
            env: The environment.
            term_func: The observation term to delay.
            term_params: The parameters to call ``term_func`` with. Defaults to None, which calls it
                with its own defaults.
            min_lag: Smallest lag [control steps] to draw. Defaults to 0.
            max_lag: Largest lag [control steps] to draw, inclusive. Defaults to 0, which is invalid
                and only present so the manager can validate the term signature.
            update_period: Control steps between two lag draws. Defaults to 0, which re-draws every
                step.
            hold_prob: Probability of keeping the previous lag on a draw step. Defaults to 0.0.

        Returns:
            The delayed observation, shaped like the wrapped term's output.
        """
        step = getattr(env, "common_step_counter", None)
        if step is not None and step == self._last_step and self._output is not None:
            if not self._has_pending_reset:
                return self._output
            # an environment reset since the buffer advanced has no history left to be delayed by,
            # so it reads the fresh value while the others keep the frame they were already served
            fresh = term_func(env, **(term_params or {}))
            mask = self._pending_reset.view(-1, *([1] * (fresh.dim() - 1)))
            return torch.where(mask, fresh, self._output)

        obs = term_func(env, **(term_params or {}))
        if self._has_pending_reset:
            self._buffer.reset(self._pending_reset)
            self._pending_reset.fill_(False)
            self._has_pending_reset = False
        self._last_step = step

        self._buffer.append(obs)
        self._update_lags()
        self._output = self._buffer[self._lags]
        return self._output

    def _update_lags(self) -> None:
        """Draw new lags for the environments whose update period has come round."""
        if self._update_period > 0:
            should_update = (self._step_count + self._phase_offsets) % self._update_period == 0
        else:
            should_update = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self._hold_prob > 0.0:
            should_update &= torch.rand(self.num_envs, device=self.device) >= self._hold_prob
        candidates = torch.randint(
            self._min_lag, self._max_lag + 1, (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._lags = torch.where(should_update, candidates, self._lags)
        self._step_count += 1


"""
NaN-safe critic terms.
"""


def _finite(value: torch.Tensor) -> torch.Tensor:
    """Replace non-finite entries with zero."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def foot_contact(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Whether each selected body is in contact with anything.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor and the bodies to report, in the requested order.

    Returns:
        One per body, 1.0 while in contact. Shape is (num_envs, num_selected_bodies).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return (forces.norm(dim=-1) > 0.0).float()


def foot_contact_forces_safe(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Net contact force on each selected body, log-compressed and guarded against NaNs.

    The compression is upstream's ``sign(F) * log1p(|F|)`` (reference section 5), which keeps the
    impulsive forces of a footstep in the same range as the quasi-static ones of standing.

    The critic's sensor-derived terms are the one observation path the NaN termination cannot
    protect: it checks joint and root state, while these read contact data, which the solver can
    return non-finite for while the state itself is still clean. A single NaN reaching the learner
    kills the whole run, so upstream zeroes them here and lets the termination catch the underlying
    state corruption.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor and the bodies to report, in the requested order.

    Returns:
        Compressed contact forces [log1p(N)], flattened per body.
        Shape is (num_envs, 3 * num_selected_bodies).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].flatten(start_dim=1)
    return _finite(torch.sign(forces) * torch.log1p(forces.abs()))


def foot_air_time_safe(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Time each selected body has spent in the air since it last left the ground, guarded.

    See :func:`foot_contact_forces_safe` for why the critic's sensor reads are guarded.

    Args:
        env: The environment.
        sensor_cfg: The contact sensor and the bodies to report, in the requested order. The sensor
            must be configured with ``track_air_time``.

    Returns:
        Current air time [s] per body. Shape is (num_envs, num_selected_bodies).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = sensor.data.current_air_time
    if air_time is None:
        raise RuntimeError(f"The contact sensor '{sensor_cfg.name}' does not track air time.")
    return _finite(air_time.torch[:, sensor_cfg.body_ids])


def foot_height_safe(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height of each selected body above the ground its environment sits on, guarded.

    See :func:`foot_contact_forces_safe` for why the critic's sensor reads are guarded.

    Upstream raycasts below each foot site to get a clearance that follows the terrain
    (reference section 2.2). MicroDuck trains on flat ground, where that clearance is the body
    height above the environment origin, which is what this term measures. A terrain-generator task
    needs the raycast instead.

    Args:
        env: The environment.
        asset_cfg: The articulation and the bodies to report, in the requested order.

    Returns:
        Body heights [m] above the environment origin. Shape is (num_envs, num_selected_bodies).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    heights = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    return _finite(heights - env.scene.env_origins[:, 2].unsqueeze(-1))
