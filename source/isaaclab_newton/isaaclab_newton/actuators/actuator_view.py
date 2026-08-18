# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone mapped view over Newton actuator parameters."""

from __future__ import annotations

from typing import Any

import warp as wp
from newton.actuators import Actuator


@wp.kernel
def _gather_parameter(src: Any, mapping: wp.array2d[wp.int32], dst: Any):
    row, col = wp.tid()
    index = mapping[row, col]
    if index >= 0:
        dst[row, col] = src[index]


@wp.kernel
def _scatter_parameter(
    values: Any,
    mapping: wp.array2d[wp.int32],
    mask: wp.array[wp.bool],
    dst: Any,
):
    row, col = wp.tid()
    if mask[row]:
        index = mapping[row, col]
        if index >= 0:
            dst[index] = values[row, col]


class ActuatorView:
    """Read and write actuator parameters through precomputed mappings.

    Args:
        mappings: Per-actuator ``wp.int32`` arrays shaped ``(world_count,
            dofs_per_world)``. ``-1`` marks a DOF the actuator does not drive.
            Mappings must be non-empty and share a world count and device.
    """

    def __init__(self, mappings: dict[Actuator, wp.array]):
        self._mappings = mappings
        mapping = next(iter(mappings.values()))
        self._full_mask = wp.full(mapping.shape[0], True, dtype=wp.bool, device=mapping.device)

    @classmethod
    def from_articulation_view(
        cls,
        articulation_view: Any,
        actuators: list[Actuator] | None = None,
    ) -> ActuatorView:
        """Extract mappings without retaining the articulation view or model.

        Args:
            articulation_view: Newton articulation view to extract from.
            actuators: Actuators to include, or all model actuators by default.

        Returns:
            Standalone actuator view.
        """
        if actuators is None:
            actuators = articulation_view.model.actuators
        world_count = articulation_view.world_count
        return cls(
            {
                actuator: articulation_view._get_actuator_dof_mapping(actuator).reshape((world_count, -1))
                for actuator in actuators
            }
        )

    def get_actuator_parameter(
        self,
        actuator: Actuator,
        component_name: str,
        parameter_name: str,
    ) -> wp.array:
        """Read an actuator parameter in the mapped layout.

        Args:
            actuator: Actuator to read.
            component_name: Component path, such as ``"clamping.0"``.
            parameter_name: Parameter attribute on the component.

        Returns:
            Parameter values with unmapped entries set to zero.
        """
        mapping = self._mappings[actuator]
        parameter = self._get_parameter(actuator, component_name, parameter_name)
        values = wp.zeros(mapping.shape, dtype=parameter.dtype, device=mapping.device)
        wp.launch(
            _gather_parameter,
            dim=mapping.shape,
            inputs=[parameter, mapping],
            outputs=[values],
            device=mapping.device,
        )
        return values

    def set_actuator_parameter(
        self,
        actuator: Actuator,
        component_name: str,
        parameter_name: str,
        values: wp.array,
        mask: wp.array | None = None,
    ) -> None:
        """Write an actuator parameter from the mapped layout.

        Args:
            actuator: Actuator to update.
            component_name: Component path, such as ``"clamping.0"``.
            parameter_name: Parameter attribute on the component.
            values: Values shaped like the actuator mapping.
            mask: Optional Boolean world mask.
        """
        mapping = self._mappings[actuator]
        parameter = self._get_parameter(actuator, component_name, parameter_name)
        wp.launch(
            _scatter_parameter,
            dim=mapping.shape,
            inputs=[values, mapping, self._full_mask if mask is None else mask],
            outputs=[parameter],
            device=mapping.device,
        )

    @staticmethod
    def _get_parameter(actuator: Actuator, component_name: str, parameter_name: str) -> wp.array:
        component_name, separator, index = component_name.partition(".")
        component = getattr(actuator, component_name)
        if separator:
            component = component[int(index)]
        return getattr(component, parameter_name)
