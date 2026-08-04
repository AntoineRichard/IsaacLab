# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private typed parameter storage used by managed actuator groups."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from isaaclab.utils.warp import ProxyArray


@dataclass(frozen=True)
class _FieldSpec:
    """Description of one typed actuator-storage field."""

    name: str
    dtype: type
    unit: str
    role: Literal["parameter", "output", "scratch", "state"]
    fill: float
    backend_side_effect: str | None


@dataclass(frozen=True)
class _ActuatorSchema:
    """Typed storage contract declared by one exact built-in actuator class."""

    fields: tuple[_FieldSpec, ...]
    graphable: bool
    stateful: bool

    @property
    def parameter_names(self) -> frozenset[str]:
        """Names of fields used as actuator-model parameters."""
        return frozenset(field.name for field in self.fields if field.role == "parameter")


@dataclass(frozen=True)
class _GroupBinding:
    """Canonical typed-array binding for one logical actuator group."""

    generation: int
    joint_indices: torch.Tensor
    joint_names: tuple[str, ...]
    type_slice: slice
    arrays: Mapping[str, ProxyArray]


# Ownership rules:
# typed actuator parameters: stiffness, damping, actuator effort/velocity limits, saturation_effort
# typed outputs: computed_effort, applied_effort
# solver compatibility only: effort_limit_sim, velocity_limit_sim, armature, all friction fields
# structural/state: delay/history/recurrent buffers, network metadata, lookup tables
# legacy fill only when no type declares it: gear_ratio = 1.0
_PD_PARAMETERS = (
    _FieldSpec("stiffness", torch.Tensor, "[N/m or N·m/rad, depending on joint type]", "parameter", 0.0, None),
    _FieldSpec("damping", torch.Tensor, "[N·s/m or N·m·s/rad, depending on joint type]", "parameter", 0.0, None),
    _FieldSpec("effort_limit", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", torch.inf, None),
    _FieldSpec("velocity_limit", torch.Tensor, "[m/s or rad/s, depending on joint type]", "parameter", torch.inf, None),
)
_MOTOR_PARAMETERS = _PD_PARAMETERS + (
    _FieldSpec("saturation_effort", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", 0.0, None),
)
_NEURAL_PARAMETERS = (
    _FieldSpec("effort_limit", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", torch.inf, None),
    _FieldSpec("velocity_limit", torch.Tensor, "[m/s or rad/s, depending on joint type]", "parameter", torch.inf, None),
    _FieldSpec("saturation_effort", torch.Tensor, "[N or N·m, depending on joint type]", "parameter", 0.0, None),
)
_OUTPUTS = (
    _FieldSpec("computed_effort", torch.Tensor, "[N or N·m, depending on joint type]", "output", 0.0, None),
    _FieldSpec("applied_effort", torch.Tensor, "[N or N·m, depending on joint type]", "output", 0.0, None),
)

_IMPLICIT_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=True, stateful=False)
_IDEAL_PD_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=True, stateful=False)
_DC_MOTOR_SCHEMA = _ActuatorSchema(_MOTOR_PARAMETERS + _OUTPUTS, graphable=True, stateful=False)
_DELAYED_PD_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
_REMOTIZED_PD_ACTUATOR_SCHEMA = _ActuatorSchema(_PD_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
_ACTUATOR_NET_LSTM_SCHEMA = _ActuatorSchema(_NEURAL_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
_ACTUATOR_NET_MLP_SCHEMA = _ActuatorSchema(_NEURAL_PARAMETERS + _OUTPUTS, graphable=False, stateful=True)
