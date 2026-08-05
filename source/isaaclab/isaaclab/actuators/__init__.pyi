# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

__all__ = [
    "ActuatorBase",
    "ActuatorBaseCfg",
    "ActuatorCollection",
    "ActuatorControl",
    "ActuatorJointProperties",
    "ActuatorNetLSTM",
    "ActuatorNetMLP",
    "ActuatorNetLSTMCfg",
    "ActuatorNetMLPCfg",
    "DCMotor",
    "DelayedPDActuator",
    "IdealPDActuator",
    "ImplicitActuator",
    "RemotizedPDActuator",
    "DCMotorCfg",
    "DelayedPDActuatorCfg",
    "IdealPDActuatorCfg",
    "ImplicitActuatorCfg",
    "RemotizedPDActuatorCfg",
]

from .actuator_base import ActuatorBase
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_collection import ActuatorCollection
from .actuator_collection import ActuatorCollection as _ActuatorCollection
from .actuator_control import ActuatorControl, ActuatorJointProperties
from .actuator_net import ActuatorNetLSTM, ActuatorNetMLP
from .actuator_net_cfg import ActuatorNetLSTMCfg, ActuatorNetMLPCfg
from .actuator_pd import (
    DCMotor,
    DelayedPDActuator,
    IdealPDActuator,
    ImplicitActuator,
    RemotizedPDActuator,
)
from .actuator_pd_cfg import (
    DCMotorCfg,
    DelayedPDActuatorCfg,
    IdealPDActuatorCfg,
    ImplicitActuatorCfg,
    RemotizedPDActuatorCfg,
)


class ActuatorCollection(_ActuatorCollection):
    """Simulation-scoped actuator registration manager."""

    def __init__(
        self,
        sim_context_or_actuator_cfgs: Any,
        control: ActuatorControl | None = None,
        *,
        debug_value_resolution: bool = False,
    ) -> None: ...

    class TypeView:
        """Compact exact-class actuator view for one articulation generation."""

        @property
        def actuator_type(self) -> type[ActuatorBase]: ...

        @property
        def num_instances(self) -> int: ...

        @property
        def num_joints(self) -> int: ...

        @property
        def joint_names(self) -> tuple[str, ...]: ...

        @property
        def joint_indices(self) -> torch.Tensor: ...

        @property
        def group_slices(self) -> dict[str, slice]: ...

        @property
        def parameter_names(self) -> tuple[str, ...]: ...

        @property
        def parameters(self) -> Mapping[str, ProxyArray]: ...

        def set_parameter_index(
            self,
            name: str,
            value: float | torch.Tensor | wp.array(dtype=wp.float32) | Sequence[float] | Sequence[Sequence[float]],
            *,
            env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | wp.array(dtype=wp.int64) | None = None,
            joint_ids: Sequence[int]
            | torch.Tensor
            | wp.array(dtype=wp.int32)
            | wp.array(dtype=wp.int64)
            | None = None,
        ) -> None: ...

        def set_parameter_mask(
            self,
            name: str,
            value: float | torch.Tensor | wp.array(dtype=wp.float32) | Sequence[float] | Sequence[Sequence[float]],
            *,
            env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
        ) -> None: ...

    class ArticulationView(dict[str, ActuatorBase]):
        """Guarded articulation-scoped actuator facade."""

        class Command:
            """Raw commands received by the actuator models."""

            @property
            def position(self) -> ProxyArray: ...

            @property
            def velocity(self) -> ProxyArray: ...

            @property
            def effort(self) -> ProxyArray: ...

            def set_position_index(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                env_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                full_data: bool = False,
            ) -> None: ...

            def set_velocity_index(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                env_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                full_data: bool = False,
            ) -> None: ...

            def set_effort_index(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                env_ids: Sequence[int]
                | torch.Tensor
                | wp.array(dtype=wp.int32)
                | wp.array(dtype=wp.int64)
                | None = None,
                full_data: bool = False,
            ) -> None: ...

            def set_position_mask(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
                env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            ) -> None: ...

            def set_velocity_mask(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
                env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            ) -> None: ...

            def set_effort_mask(
                self,
                *,
                value: torch.Tensor | wp.array(dtype=wp.float32),
                joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
                env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
            ) -> None: ...

        class JointCommand:
            """Processed commands produced for the simulated joints."""

            @property
            def position(self) -> ProxyArray: ...

            @property
            def velocity(self) -> ProxyArray: ...

            @property
            def effort(self) -> ProxyArray: ...

        @property
        def generation(self) -> int: ...

        @property
        def is_ready(self) -> bool: ...

        @property
        def by_type(self) -> Mapping[type[ActuatorBase], ActuatorCollection.TypeView]: ...

        @property
        def command(self) -> ActuatorCollection.ArticulationView.Command: ...

        @property
        def joint_command(self) -> ActuatorCollection.ArticulationView.JointCommand: ...

        @property
        def computed_effort(self) -> ProxyArray: ...

        @property
        def applied_effort(self) -> ProxyArray: ...

        @property
        def has_implicit_actuators(self) -> bool: ...

        def write_actuator_stiffness_to_sim(
            self, *, stiffness: torch.Tensor, env_ids: torch.Tensor, joint_ids: torch.Tensor
        ) -> None: ...

        def write_actuator_damping_to_sim(
            self, *, damping: torch.Tensor, env_ids: torch.Tensor, joint_ids: torch.Tensor
        ) -> None: ...

        def reset(
            self,
            env_ids: Sequence[int]
            | torch.Tensor
            | wp.array(dtype=wp.int32)
            | wp.array(dtype=wp.int64)
            | slice
            | None = None,
        ) -> None: ...

        def compute(self, dt: float = 0.0) -> None: ...

        def submit_commands(self) -> None: ...

    def register_articulation(
        self,
        *,
        key: object,
        cfgs: Mapping[str, ActuatorBaseCfg],
        control: ActuatorControl,
        replication_cfg_id: int,
        debug_validation: bool,
        debug_value_resolution: bool,
    ) -> ArticulationView: ...

    @property
    def registration_keys(self) -> tuple[object, ...]: ...

    @property
    def generation(self) -> int | None: ...

    @property
    def is_finalized(self) -> bool: ...

    def finalize(self) -> None: ...

    def clear_generation(self) -> None: ...

    def close(self) -> None: ...
