# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral actuator control interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

from .actuator_base import ActuatorBase
from .actuator_base_cfg import ActuatorBaseCfg
from .actuator_pd import ImplicitActuator

if TYPE_CHECKING:
    from .actuator_collection import ActuatorCollection, _ArticulationBinding
    from .actuator_storage import _BackendParameterStaging, _GroupBinding


@dataclass(frozen=True)
class ActuatorJointProperties:
    """Default joint properties used to construct actuator models."""

    stiffness: torch.Tensor
    """Default joint stiffness values [N/m or N·m/rad, depending on joint type]."""

    damping: torch.Tensor
    """Default joint damping values [N·s/m or N·m·s/rad, depending on joint type]."""

    armature: torch.Tensor
    """Default joint armature values [kg or kg·m², depending on joint type]."""

    friction: torch.Tensor
    """Default backend-specific joint friction values.

    The physical meaning and units depend on the concrete backend and solver. See
    :attr:`isaaclab.assets.ArticulationData.joint_friction_coeff` for the active
    backend's convention.
    """

    dynamic_friction: torch.Tensor
    """Default backend-specific joint dynamic friction values.

    The physical meaning and units depend on the concrete backend and solver.
    """

    viscous_friction: torch.Tensor
    """Default backend-specific joint viscous friction values.

    The physical meaning and units depend on the concrete backend and solver.
    """

    effort_limit: torch.Tensor
    """Default joint effort limits [N or N·m, depending on joint type]."""

    velocity_limit: torch.Tensor
    """Default joint velocity limits [m/s or rad/s, depending on joint type]."""


@dataclass(frozen=True)
class _ActuatorParameterWrite:
    """Canonical selection and backend ownership for one actuator parameter write."""

    value: torch.Tensor | wp.array
    canonical: ProxyArray | None = None
    env_ids: torch.Tensor | wp.array | None = None
    joint_ids: torch.Tensor | wp.array | None = None
    env_mask: torch.Tensor | wp.array | None = None
    joint_mask: torch.Tensor | wp.array | None = None
    group_binding: _GroupBinding | None = None
    type_csr_offsets: ProxyArray | None = None
    type_csr_slots: ProxyArray | None = None
    backend_owner_slots: torch.Tensor | wp.array | None = None
    backend_parameter_staging: _BackendParameterStaging | None = None
    scope: str = ""


@dataclass(frozen=True)
class _ResolvedSolverProperty:
    """One final solver-property transport chosen by the owning backend.

    ``transport == \"device\"`` supplies the candidate-owned device source
    rows, world-to-source assignment, and dense canonical target used by
    device-native backends.  ``transport == \"cpu\"`` instead supplies
    backend-owned CPU source rows and a CPU local-source-slot mapping.  Core
    never copies device data to the host to satisfy the latter contract.
    """

    transport: Literal["cpu", "device"]
    source_rows: torch.Tensor
    source_slot_by_backend_row: torch.Tensor | None
    source_assignment: torch.Tensor | None
    canonical_target: ProxyArray | None


@dataclass(frozen=True)
class _ResolvedSolverProperties:
    """Candidate solver-property payload retained through transactional binding.

    ``properties`` contains one already config-order-merged transport per
    solver property.  It deliberately carries no logical-group overlays: a
    control either consumes compact source rows without a device-to-host copy,
    or consumes the pointer-stable canonical dense target on device.
    """

    properties: Mapping[str, _ResolvedSolverProperty]
    clone_plan_metadata: object | None = None


class ActuatorControl(ABC):
    """Backend-neutral bridge used by :class:`~isaaclab.actuators.ActuatorCollection`."""

    @property
    @abstractmethod
    def num_instances(self) -> int:
        """Number of articulation instances."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Number of articulation joints."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_fixed_tendons(self) -> int:
        """Number of fixed tendons."""
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> str:
        """Warp/Torch device string."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_pos(self) -> ProxyArray:
        """Current joint positions [m or rad, depending on joint type]."""
        raise NotImplementedError

    @property
    @abstractmethod
    def joint_vel(self) -> ProxyArray:
        """Current joint velocities [m/s or rad/s, depending on joint type]."""
        raise NotImplementedError

    @abstractmethod
    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[list[int] | ProxyArray, list[str]]:
        """Resolve joint name expressions to user-order joint indices and names."""
        raise NotImplementedError

    @abstractmethod
    def resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        """Resolve optional environment indices to a supported signed index array."""
        raise NotImplementedError

    @abstractmethod
    def resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        """Resolve optional joint indices to a supported signed index array."""
        raise NotImplementedError

    @abstractmethod
    def resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        """Resolve optional environment mask to a full Warp bool mask."""
        raise NotImplementedError

    @abstractmethod
    def resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        """Resolve optional joint mask to a full Warp bool mask."""
        raise NotImplementedError

    @abstractmethod
    def assert_shape_and_dtype(
        self,
        tensor: torch.Tensor | wp.array | float,
        shape: tuple[int, ...],
        dtype: type,
        name: str,
    ) -> None:
        """Validate tensor shape and dtype using the owning asset's policy."""
        raise NotImplementedError

    @abstractmethod
    def assert_shape_and_dtype_mask(
        self,
        tensor: torch.Tensor | wp.array | float,
        masks: tuple[wp.array, ...],
        dtype: type,
        name: str,
    ) -> None:
        """Validate full-sized mask write tensor shape and dtype."""
        raise NotImplementedError

    @abstractmethod
    def get_default_joint_properties(self, joint_ids: torch.Tensor | wp.array | slice) -> ActuatorJointProperties:
        """Return backend defaults used to construct one actuator group."""
        raise NotImplementedError

    def get_source_joint_properties(
        self,
        joint_ids: torch.Tensor | wp.array | slice,
        source_env_ids: torch.Tensor | wp.array,
    ) -> ActuatorJointProperties:
        """Return compact backend-default rows for the used clone sources.

        The fallback preserves third-party controls while keeping the selected
        source rows device-resident. Concrete controls should override this to
        gather directly from their backend-owned property arrays.

        Args:
            joint_ids: Articulation joints in logical-group order.
            source_env_ids: Source articulation rows used by the clone plan.

        Returns:
            Backend defaults with shape ``[num_sources, num_group_joints]``.
        """
        defaults = self.get_default_joint_properties(joint_ids)
        source_ids = source_env_ids.torch if isinstance(source_env_ids, ProxyArray) else source_env_ids
        if source_ids.dtype not in (torch.int32, torch.int64) or source_ids.ndim != 1:
            raise TypeError("Source environment IDs must be a one-dimensional signed Torch tensor.")
        source_device = defaults.stiffness.device
        if source_ids.device != source_device:
            if source_ids.device.type != "cpu" and source_device.type == "cpu":
                raise ValueError(
                    "CPU source resolution requires clone-plan source IDs on CPU; refusing a device-to-host copy."
                )
            source_ids = source_ids.to(device=source_device)
        if not isinstance(joint_ids, slice):
            joint_tensor = joint_ids.torch if isinstance(joint_ids, ProxyArray) else joint_ids
            if joint_tensor.dtype not in (torch.int32, torch.int64) or joint_tensor.ndim != 1:
                raise TypeError("Joint IDs must be a one-dimensional signed Torch tensor.")
        source_indices = source_ids.to(dtype=torch.long)

        def select(value: torch.Tensor) -> torch.Tensor:
            """Select compact source rows from one backend default field."""
            return value.index_select(0, source_indices)

        return ActuatorJointProperties(
            stiffness=select(defaults.stiffness),
            damping=select(defaults.damping),
            armature=select(defaults.armature),
            friction=select(defaults.friction),
            dynamic_friction=select(defaults.dynamic_friction),
            viscous_friction=select(defaults.viscous_friction),
            effort_limit=select(defaults.effort_limit),
            velocity_limit=select(defaults.velocity_limit),
        )

    @abstractmethod
    def write_resolved_joint_properties(self, actuator: ActuatorBase, *, native_managed: bool) -> None:
        """Write actuator-resolved limits and physical properties to the backend."""
        raise NotImplementedError

    def write_resolved_joint_properties_staged(self, properties: _ResolvedSolverProperties) -> None:
        """Write one resolved articulation-wide rowset before publication.

        Concrete controls consume each property exactly once. The supplied
        candidate-owned arrays are valid only until registration either
        publishes or rolls back.

        Args:
            properties: Candidate payload containing ``[world, joint]`` device
                targets and source-bounded default/overlay rows. CPU-only
                backends may resolve their own host write rows from the latter
                without synchronizing device staging targets.
        """
        del properties

    def validate_resolved_joint_properties(self) -> None:
        """Validate backend state after consolidated solver-property writes."""

    def restore_resolved_joint_properties(self) -> None:
        """Restore backend solver defaults after a candidate construction failure."""

    def commit_resolved_joint_properties(self) -> None:
        """Discard reversible solver-property write state after successful publication."""

    def resolved_solver_property_transport(self) -> Literal["cpu", "device"]:
        """Return the transport accepted by staged solver-property writes.

        Device-native controls consume compact source rows, clone assignments,
        and the candidate-owned dense canonical target. CPU controls opt into
        ``"cpu"`` only when their clone-plan adapter has supplied CPU source
        rows and source-slot metadata directly; core never performs a
        device-to-host conversion for this path.
        """
        return "device"

    def discover_native_actuators(self, cfgs: Mapping[str, ActuatorBaseCfg]) -> set[str]:
        """Discover native actuator groups during order-10 registration."""
        return set()

    def prepare_actuator_binding(self, binding: _ArticulationBinding) -> None:
        """Prepare private candidate binding state before generation publication."""

    def bind_actuator_view(self, view: ActuatorCollection.ArticulationView) -> None:
        """Bind the now-live public articulation facade after publication."""

    def invalidate_actuator_view(self) -> None:
        """Discard a published or pending articulation facade binding."""

    def complete_articulation_initialization(self) -> None:
        """Mark the owning articulation ready after every facade has bound."""

    def write_actuator_parameter(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Apply one canonical parameter write to backend-owned slots."""
        del name, write

    def preflight_actuator_parameter_write(self, name: str, write: _ActuatorParameterWrite) -> None:
        """Reject an unsafe backend parameter write before canonical mutation.

        Backends that must flush host state, update USD, or otherwise leave a
        CUDA capture-safe path inspect the resolved route here.  The collection
        invokes this hook after selector and ownership validation, but before
        launching the canonical compact-storage write or patching its dense
        backend staging target.

        Args:
            name: Backend parameter name.
            write: Fully normalized selection and route metadata.
        """
        del name, write

    def stage_user_command(
        self,
        command_name: str,
        collection: ActuatorCollection | ActuatorCollection.ArticulationView,
        env_ids: torch.Tensor | wp.array | None,
        joint_ids: torch.Tensor | wp.array | None,
        env_mask: wp.array | None,
        joint_mask: wp.array | None,
    ) -> None:
        """Optionally stage a raw user command into backend bindings when setters are called."""

    def prepare_native_actuators(
        self, collection: ActuatorCollection, actuator_cfgs: dict[str, ActuatorBaseCfg]
    ) -> set[str]:
        """Prepare backend-native actuators and return config group names managed natively."""
        return set()

    def finalize_native_actuators(self, collection: ActuatorCollection) -> None:
        """Finalize backend-native masks, callbacks, and telemetry views after group construction."""

    def compute_native_actuators(self, collection: ActuatorCollection, dt: float) -> bool:
        """Compute backend-native actuator outputs.

        Args:
            collection: Collection that owns actuator command and telemetry buffers.
            dt: Physics step size [s].

        Returns:
            True when native handling replaced the standard Python actuator loop.
        """
        return False

    @abstractmethod
    def submit_commands(self, collection: ActuatorCollection) -> None:
        """Submit processed collection command buffers to the backend simulation."""
        raise NotImplementedError

    def reset_native_actuators(self, env_ids: Sequence[int] | slice) -> None:
        """Reset backend-native actuator state for selected environments."""

    def write_native_actuator_gain(
        self,
        attr: str,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ) -> None:
        """Write backend-native actuator gain values."""


class ArticulationActuatorControl(ActuatorControl):
    """Shared control adapter for articulation-backed actuator collections.

    This class implements the backend-independent forwarding and joint-property
    plumbing used by articulation backends. Backend subclasses only need to
    provide command submission and override the small hooks where their write
    APIs differ.

    Args:
        articulation: Articulation object that owns backend simulation handles.
    """

    def __init__(self, articulation):
        self._articulation = articulation

    @property
    def native_active(self) -> bool:
        """Whether a backend-native actuator path is active."""
        return getattr(self, "_native_active", False)

    @property
    def num_instances(self) -> int:
        return self._articulation.num_instances

    @property
    def num_joints(self) -> int:
        return self._articulation.num_joints

    @property
    def num_fixed_tendons(self) -> int:
        return self._articulation.num_fixed_tendons

    @property
    def device(self) -> str:
        return self._articulation.device

    @property
    def joint_pos(self) -> ProxyArray:
        return self._articulation.data.joint_pos

    @property
    def joint_vel(self) -> ProxyArray:
        return self._articulation.data.joint_vel

    def find_joints(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        """Resolve host topology metadata without reading a device selector."""
        return self._articulation.find_joints(name_keys)

    def resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        return self._articulation._resolve_env_ids(env_ids)

    def resolve_joint_ids(self, joint_ids: Sequence[int] | torch.Tensor | wp.array | None) -> torch.Tensor | wp.array:
        return self._articulation._resolve_joint_ids(joint_ids)

    def resolve_env_mask(self, env_mask: wp.array | None) -> wp.array:
        if hasattr(self._articulation, "_resolve_env_mask"):
            return self._articulation._resolve_env_mask(env_mask)
        return self._articulation._resolve_mask(env_mask, self._articulation._ALL_ENV_MASK)

    def resolve_joint_mask(self, joint_mask: wp.array | None) -> wp.array:
        if hasattr(self._articulation, "_resolve_joint_mask"):
            return self._articulation._resolve_joint_mask(joint_mask)
        return self._articulation._resolve_mask(joint_mask, self._articulation._ALL_JOINT_MASK)

    def assert_shape_and_dtype(
        self,
        tensor: torch.Tensor | wp.array | float,
        shape: tuple[int, ...],
        dtype: type,
        name: str,
    ) -> None:
        self._articulation.assert_shape_and_dtype(tensor, shape, dtype, name)

    def assert_shape_and_dtype_mask(
        self,
        tensor: torch.Tensor | wp.array | float,
        masks: tuple[wp.array, ...],
        dtype: type,
        name: str,
    ) -> None:
        self._articulation.assert_shape_and_dtype_mask(tensor, masks, dtype, name)

    def get_default_joint_properties(self, joint_ids: torch.Tensor | wp.array | slice) -> ActuatorJointProperties:
        data = self._articulation.data
        stiffness = data.joint_stiffness.torch[:, joint_ids]
        return ActuatorJointProperties(
            stiffness=stiffness,
            damping=data.joint_damping.torch[:, joint_ids],
            armature=data.joint_armature.torch[:, joint_ids],
            friction=data.joint_friction_coeff.torch[:, joint_ids],
            dynamic_friction=self._joint_property_or_zeros("joint_dynamic_friction_coeff", joint_ids, stiffness),
            viscous_friction=self._joint_property_or_zeros("joint_viscous_friction_coeff", joint_ids, stiffness),
            effort_limit=data.joint_effort_limits.torch[:, joint_ids].clone(),
            velocity_limit=data.joint_vel_limits.torch[:, joint_ids],
        )

    def get_source_joint_properties(
        self,
        joint_ids: torch.Tensor | wp.array | slice,
        source_env_ids: torch.Tensor | wp.array,
    ) -> ActuatorJointProperties:
        """Gather only clone-plan source rows from articulation property arrays."""
        source_ids = source_env_ids.torch if isinstance(source_env_ids, ProxyArray) else source_env_ids
        joint_indices: torch.Tensor | slice
        if isinstance(joint_ids, slice):
            joint_indices = joint_ids
        else:
            joint_indices = joint_ids.torch if isinstance(joint_ids, ProxyArray) else joint_ids
            joint_indices = joint_indices.to(dtype=torch.long)
        source_indices = source_ids.to(dtype=torch.long)

        def _gather(values: ProxyArray) -> torch.Tensor:
            return values.torch.index_select(0, source_indices)[:, joint_indices]

        data = self._articulation.data
        stiffness = _gather(data.joint_stiffness)
        dynamic_friction = getattr(data, "joint_dynamic_friction_coeff", None)
        viscous_friction = getattr(data, "joint_viscous_friction_coeff", None)
        return ActuatorJointProperties(
            stiffness=stiffness,
            damping=_gather(data.joint_damping),
            armature=_gather(data.joint_armature),
            friction=_gather(data.joint_friction_coeff),
            dynamic_friction=(
                _gather(dynamic_friction) if dynamic_friction is not None else torch.zeros_like(stiffness)
            ),
            viscous_friction=(
                _gather(viscous_friction) if viscous_friction is not None else torch.zeros_like(stiffness)
            ),
            effort_limit=_gather(data.joint_effort_limits),
            velocity_limit=_gather(data.joint_vel_limits),
        )

    def write_resolved_joint_properties(self, actuator: ActuatorBase, *, native_managed: bool) -> None:
        articulation = self._articulation
        articulation.write_joint_effort_limit_to_sim_index(
            limits=actuator.effort_limit_sim,
            joint_ids=actuator.joint_indices,
        )
        articulation.write_joint_velocity_limit_to_sim_index(
            limits=actuator.velocity_limit_sim,
            joint_ids=actuator.joint_indices,
        )
        articulation.write_joint_armature_to_sim_index(armature=actuator.armature, joint_ids=actuator.joint_indices)
        self._write_joint_friction_properties(actuator)
        if isinstance(actuator, ImplicitActuator) and not native_managed:
            articulation.write_joint_stiffness_to_sim_index(
                stiffness=actuator.stiffness, joint_ids=actuator.joint_indices
            )
            articulation.write_joint_damping_to_sim_index(damping=actuator.damping, joint_ids=actuator.joint_indices)
        else:
            articulation.write_joint_stiffness_to_sim_index(stiffness=0.0, joint_ids=actuator.joint_indices)
            articulation.write_joint_damping_to_sim_index(damping=0.0, joint_ids=actuator.joint_indices)

    def _write_joint_friction_properties(self, actuator: ActuatorBase) -> None:
        self._articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=actuator.friction,
            joint_ids=actuator.joint_indices,
        )

    def _joint_property_or_zeros(
        self, attr_name: str, joint_ids: torch.Tensor | wp.array | slice, reference: torch.Tensor
    ) -> torch.Tensor:
        joint_property = getattr(self._articulation.data, attr_name, None)
        if joint_property is None:
            return torch.zeros_like(reference)
        return joint_property.torch[:, joint_ids]

    @staticmethod
    def _is_implicit_cfg(actuator_cfg: ActuatorBaseCfg) -> bool:
        class_type = actuator_cfg.class_type
        return (
            "ImplicitActuator" in class_type
            if isinstance(class_type, str)
            else issubclass(class_type, ImplicitActuator)
        )
