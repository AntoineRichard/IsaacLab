# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone mapped view over Newton actuator parameters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp
from newton.actuators import Actuator


@wp.kernel
def _gather_parameter(
    src: Any,
    mapping: wp.array2d[wp.int32],
    dst: Any,
):
    """Gather a flat actuator parameter into a two-dimensional view."""
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
    """Scatter a two-dimensional view into a flat actuator parameter."""
    row, col = wp.tid()
    if mask[row]:
        index = mapping[row, col]
        if index >= 0:
            dst[index] = values[row, col]


class ActuatorView:
    """Read and write Newton actuator parameters through external mappings.

    Args:
        bindings: Actuator/mapping pairs. Each mapping has shape
            ``(world_count, dofs_per_world)`` and contains flat indices into
            that actuator's parameter arrays. An index of ``-1`` marks a view
            DOF that the actuator does not drive.
    """

    def __init__(self, bindings: Sequence[tuple[Actuator, wp.array]]):
        if not bindings:
            raise ValueError("At least one actuator binding is required.")

        self._mappings: dict[Actuator, wp.array] = {}
        first_mapping = bindings[0][1]
        self._validate_mapping(first_mapping)
        self.shape = first_mapping.shape
        self.device = first_mapping.device

        for actuator, mapping in bindings:
            if actuator in self._mappings:
                raise ValueError("Each actuator may be bound only once.")
            self._validate_mapping(mapping)
            if mapping.shape != self.shape:
                raise ValueError(f"Expected mapping shape {self.shape}, got {mapping.shape}.")
            if mapping.device != self.device:
                raise ValueError(f"Expected mapping on device {self.device}, got {mapping.device}.")
            self._mappings[actuator] = mapping

        self.world_count = self.shape[0]
        self.dofs_per_world = self.shape[1]
        self._full_mask = wp.full(self.world_count, True, dtype=wp.bool, device=self.device)

    @classmethod
    def from_articulation_view(
        cls,
        articulation_view: Any,
        actuators: Sequence[Actuator] | None = None,
    ) -> ActuatorView:
        """Build from mappings already provided by a Newton articulation view.

        Args:
            articulation_view: Newton articulation view that provides
                ``_get_actuator_dof_mapping`` and ``world_count``.
            actuators: Actuators to bind. When ``None``, all actuators in the
                source view's model are bound.

        Returns:
            Standalone view containing reshaped source mappings.
        """
        world_count = articulation_view.world_count
        if world_count <= 0:
            raise ValueError(f"Expected a positive world count, got {world_count}.")
        if actuators is None:
            actuators = articulation_view.model.actuators

        bindings = []
        for actuator in actuators:
            flat_mapping = articulation_view._get_actuator_dof_mapping(actuator)
            if flat_mapping.ndim != 1:
                raise ValueError(f"Expected a one-dimensional source mapping, got {flat_mapping.ndim} dimensions.")
            if len(flat_mapping) % world_count != 0:
                raise ValueError(
                    f"Source mapping length {len(flat_mapping)} is not divisible by world count {world_count}."
                )
            dofs_per_world = len(flat_mapping) // world_count
            bindings.append((actuator, flat_mapping.reshape((world_count, dofs_per_world))))
        return cls(bindings)

    def get_actuator_parameter(
        self,
        actuator: Actuator,
        component_name: str,
        parameter_name: str,
    ) -> wp.array:
        """Read one actuator parameter in the view layout.

        Args:
            actuator: Bound actuator to read.
            component_name: Actuator component path.
            parameter_name: Parameter attribute on the component.

        Returns:
            Parameter values with shape ``(world_count, dofs_per_world)``.
            Unmapped entries are zero.
        """
        mapping = self._mapping_for(actuator)
        parameter = self._resolve_parameter(actuator, component_name, parameter_name)
        self._validate_parameter(parameter, component_name, parameter_name)
        result = wp.zeros(self.shape, dtype=parameter.dtype, device=self.device)
        wp.launch(
            _gather_parameter,
            dim=self.shape,
            inputs=[parameter, mapping],
            outputs=[result],
            device=self.device,
        )
        return result

    def set_actuator_parameter(
        self,
        actuator: Actuator,
        component_name: str,
        parameter_name: str,
        values: wp.array,
        mask: wp.array | None = None,
    ) -> None:
        """Write one actuator parameter from values in the view layout.

        Args:
            actuator: Bound actuator to update.
            component_name: Actuator component path.
            parameter_name: Parameter attribute on the component.
            values: Parameter values shaped ``(world_count, dofs_per_world)``.
            mask: Optional Boolean world mask shaped ``(world_count,)``.
        """
        mapping = self._mapping_for(actuator)
        parameter = self._resolve_parameter(actuator, component_name, parameter_name)
        self._validate_parameter(parameter, component_name, parameter_name)
        self._validate_values(values, parameter)
        resolved_mask = self._resolve_mask(mask)
        wp.launch(
            _scatter_parameter,
            dim=self.shape,
            inputs=[values, mapping, resolved_mask],
            outputs=[parameter],
            device=self.device,
        )

    def _mapping_for(self, actuator: Actuator) -> wp.array:
        try:
            return self._mappings[actuator]
        except KeyError as error:
            raise KeyError("The actuator is not bound to this view.") from error

    @staticmethod
    def _validate_mapping(mapping: Any) -> None:
        if not isinstance(mapping, wp.array) or mapping.ndim != 2 or mapping.dtype != wp.int32:
            raise ValueError("Actuator mappings must be two-dimensional wp.int32 arrays.")

    @staticmethod
    def _resolve_parameter(actuator: Actuator, component_name: str, parameter_name: str) -> Any:
        if component_name == "controller":
            component = actuator.controller
        elif component_name == "delay":
            component = actuator.delay
            if component is None:
                raise ValueError("The actuator does not have a delay component.")
        elif component_name.startswith("clamping.") and component_name.removeprefix("clamping.").isdigit():
            index = int(component_name.removeprefix("clamping."))
            try:
                component = actuator.clamping[index]
            except IndexError as error:
                raise IndexError(
                    f"Actuator clamping index {index} is out of range for {len(actuator.clamping)} components."
                ) from error
        else:
            raise ValueError(f"Unsupported actuator component path: {component_name!r}.")
        try:
            return getattr(component, parameter_name)
        except AttributeError as error:
            raise AttributeError(
                f"Actuator component {component_name!r} has no parameter {parameter_name!r}."
            ) from error

    def _validate_parameter(self, parameter: Any, component_name: str, parameter_name: str) -> None:
        if not isinstance(parameter, wp.array) or parameter.ndim != 1:
            raise ValueError(
                f"Actuator parameter {component_name!r}.{parameter_name!r} must be a one-dimensional Warp array."
            )
        if parameter.device != self.device:
            raise ValueError(
                f"Expected actuator parameter {component_name!r}.{parameter_name!r} on device {self.device}, "
                f"got {parameter.device}."
            )

    def _validate_values(self, values: wp.array, parameter: wp.array) -> None:
        if not isinstance(values, wp.array) or values.shape != self.shape:
            shape = values.shape if isinstance(values, wp.array) else None
            raise ValueError(f"Expected values shape {self.shape}, got {shape}.")
        if values.dtype != parameter.dtype:
            raise ValueError(f"Expected values dtype {parameter.dtype}, got {values.dtype}.")
        if values.device != self.device:
            raise ValueError(f"Expected values on device {self.device}, got {values.device}.")

    def _resolve_mask(self, mask: wp.array | None) -> wp.array:
        if mask is None:
            return self._full_mask
        if not isinstance(mask, wp.array) or mask.dtype != wp.bool:
            dtype = mask.dtype if isinstance(mask, wp.array) else None
            raise ValueError(f"Expected mask dtype {wp.bool}, got {dtype}.")
        if mask.shape != (self.world_count,):
            raise ValueError(f"Expected mask shape ({self.world_count},), got {mask.shape}.")
        if mask.device != self.device:
            raise ValueError(f"Expected mask on device {self.device}, got {mask.device}.")
        return mask
