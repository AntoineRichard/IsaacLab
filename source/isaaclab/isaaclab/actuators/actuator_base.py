# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import warp as wp

import isaaclab.utils.string as string_utils
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils.warp import ProxyArray

from .actuator_storage import (
    _GroupBinding,
    _GuardedItemsView,
    _GuardedIterator,
    _GuardedKeysView,
    _GuardedValuesView,
)

if TYPE_CHECKING:
    from .actuator_base_cfg import ActuatorBaseCfg


class _ManagedParameter:
    """Descriptor that keeps bound actuator parameter identities stable."""

    def __init__(self, name: str, *, solver_compatibility: bool = False) -> None:
        self.name = name
        self._solver_compatibility = solver_compatibility

    def __get__(self, instance: ActuatorBase | None, owner: type[ActuatorBase]) -> torch.Tensor | _ManagedParameter:
        if instance is None:
            return self
        instance._require_current_facade()
        binding = instance.__dict__.get("_parameter_binding")
        if binding is None:
            return instance.__dict__[self.name]
        if self._solver_compatibility:
            if binding.solver_proxies is not None and self.name in binding.solver_proxies:
                return binding.solver_proxies[self.name].torch
            return instance._get_compatibility_sidecar(self.name)
        if self.name in binding.arrays:
            return binding.arrays[self.name].torch[:, binding.type_slice]
        return instance._get_deprecated_gain_sidecar(self.name)

    def __set__(self, instance: ActuatorBase, value: torch.Tensor) -> None:
        instance._require_facade_execution_ready()
        binding = instance.__dict__.get("_parameter_binding")
        if binding is None:
            instance.__dict__[self.name] = value
            return
        if self._solver_compatibility:
            if binding.solver_proxies is not None and self.name in binding.solver_proxies:
                binding.solver_proxies[self.name].torch.copy_(value)
                return
            instance._get_compatibility_sidecar(self.name).copy_(value)
            return
        if self.name in binding.arrays:
            if self.name in (binding.parameter_proxies or {}) and instance.__dict__.get("_facade_view") is not None:
                instance.set_parameter_index(self.name, value)
                return
            binding.arrays[self.name].torch[:, binding.type_slice].copy_(value)
            return
        instance._get_deprecated_gain_sidecar(self.name).copy_(value)


@dataclass(frozen=True)
class _ResolvedManagedRegistration:
    """Exact-class actuator parameters resolved over source-prototype rows.

    The source shell is private candidate state.  Its parameter tensors have one
    row per used clone source and are expanded into canonical storage before one
    runtime group shell is bound to the final generation.
    """

    cfg: Any
    actuator_type: type[ActuatorBase]
    joint_names: tuple[str, ...]
    joint_indices: slice | torch.Tensor
    source_shell: ActuatorBase
    source_values: Mapping[str, torch.Tensor]
    structural_signature: tuple[Any, ...]


@dataclass(frozen=True)
class _SolverCompatibilitySeed:
    """Compact source rows that lazily recreate one runtime solver field."""

    source_rows: torch.Tensor
    source_assignment: torch.Tensor
    source_joint_indices: torch.Tensor

    def materialize(self, *, num_envs: int, device: str) -> torch.Tensor:
        """Expand compact source rows to the requesting runtime group shape."""
        assignment = self.source_assignment.to(device=device, dtype=torch.long)
        if assignment.shape != (num_envs,):
            raise ValueError("Solver compatibility source assignment has an invalid runtime shape.")
        source_joint_indices = self.source_joint_indices.to(device=self.source_rows.device, dtype=torch.long)
        source_rows = self.source_rows.index_select(1, source_joint_indices).to(device=device)
        return source_rows.index_select(0, assignment)


class _GuardedParameterMapping(Mapping[str, ProxyArray]):
    """Read-only parameter map that validates its owning group generation."""

    def __init__(self, actuator: ActuatorBase, values: Mapping[str, ProxyArray]) -> None:
        self._actuator = actuator
        self._values = values

    def __getitem__(self, name: str) -> ProxyArray:
        self._actuator._require_current_facade()
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        self._actuator._require_current_facade()
        return _GuardedIterator(self._actuator._require_current_facade, iter(self._values))

    def __reversed__(self) -> Iterator[str]:
        self._actuator._require_current_facade()
        return _GuardedIterator(self._actuator._require_current_facade, reversed(self._values))

    def __len__(self) -> int:
        self._actuator._require_current_facade()
        return len(self._values)

    def __repr__(self) -> str:
        self._actuator._require_current_facade()
        return repr(self._values)

    def copy(self) -> dict[str, ProxyArray]:
        """Return an ordinary dictionary snapshot of the guarded mapping."""
        self._actuator._require_current_facade()
        return dict(self._values)

    def keys(self) -> _GuardedKeysView:
        self._actuator._require_current_facade()
        return _GuardedKeysView(self, self._actuator._require_current_facade)

    def items(self) -> _GuardedItemsView:
        self._actuator._require_current_facade()
        return _GuardedItemsView(self, self._actuator._require_current_facade)

    def values(self) -> _GuardedValuesView:
        self._actuator._require_current_facade()
        return _GuardedValuesView(self, self._actuator._require_current_facade)


class ActuatorBase(ABC):
    """Base class for actuator models over a collection of actuated joints in an articulation.

    Actuator models augment the simulated articulation joints with an external drive dynamics model.
    The model is used to convert the user-provided joint commands (positions, velocities and efforts)
    into the desired joint positions, velocities and efforts that are applied to the simulated articulation.

    The base class provides the interface for the actuator models. It is responsible for parsing the
    actuator parameters from the configuration and storing them as buffers. It also provides the
    interface for resetting the actuator state and computing the desired joint commands for the simulation.

    For each actuator model, a corresponding configuration class is provided. The configuration class
    is used to parse the actuator parameters from the configuration. It also specifies the joint names
    for which the actuator model is applied. These names can be specified as regular expressions, which
    are matched against the joint names in the articulation.

    To see how the class is used, check the :class:`isaaclab.assets.Articulation` class.
    """

    is_implicit_model: ClassVar[bool] = False
    """Flag indicating if the actuator is an implicit or explicit actuator model.

    If a class inherits from :class:`ImplicitActuator`, then this flag should be set to :obj:`True`.
    """

    _SOLVER_COMPATIBILITY_PARAMETER_NAMES: ClassVar[tuple[str, ...]] = (
        "effort_limit_sim",
        "velocity_limit_sim",
        "armature",
        "friction",
        "dynamic_friction",
        "viscous_friction",
    )
    _supports_execution_aggregation: ClassVar[bool] = False

    def __getattribute__(self, name: str) -> Any:
        """Guard public group access when the owning facade generation expires."""
        if not name.startswith("_"):
            state = object.__getattribute__(self, "__dict__")
            view = state.get("_facade_view")
            if view is not None:
                token = state.get("_facade_token")
                if name in {"compute", "reset"}:
                    view._require_execution_ready(token)
                else:
                    view._require_current_generation(token)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Guard public group writes when the owning facade generation expires."""
        if not name.startswith("_"):
            state = object.__getattribute__(self, "__dict__")
            view = state.get("_facade_view")
            if view is not None:
                view._require_execution_ready(state.get("_facade_token"))
        object.__setattr__(self, name, value)

    effort_limit = _ManagedParameter("effort_limit")
    velocity_limit = _ManagedParameter("velocity_limit")
    stiffness = _ManagedParameter("stiffness")
    damping = _ManagedParameter("damping")
    saturation_effort = _ManagedParameter("saturation_effort")
    computed_effort = _ManagedParameter("computed_effort")
    applied_effort = _ManagedParameter("applied_effort")
    effort_limit_sim = _ManagedParameter("effort_limit_sim", solver_compatibility=True)
    velocity_limit_sim = _ManagedParameter("velocity_limit_sim", solver_compatibility=True)
    armature = _ManagedParameter("armature", solver_compatibility=True)
    friction = _ManagedParameter("friction", solver_compatibility=True)
    dynamic_friction = _ManagedParameter("dynamic_friction", solver_compatibility=True)
    viscous_friction = _ManagedParameter("viscous_friction", solver_compatibility=True)

    computed_effort: torch.Tensor
    """The computed effort [N or N·m, depending on joint type]. Shape is (num_envs, num_joints)."""

    applied_effort: torch.Tensor
    """The applied effort [N or N·m, depending on joint type]. Shape is (num_envs, num_joints).

    This is the effort obtained after clipping the :attr:`computed_effort` based on the
    actuator characteristics.
    """

    effort_limit: torch.Tensor
    """The effort limit for the actuator group. Shape is (num_envs, num_joints).

    This limit is used differently depending on the actuator type:

    - **Explicit actuators**: Used for internal torque clipping within the actuator model
      (e.g., motor torque limits in DC motor models).
    - **Implicit actuators**: Same as :attr:`effort_limit_sim` (aliased for consistency).
    """

    effort_limit_sim: torch.Tensor
    """The effort limit for the actuator group in the simulation. Shape is (num_envs, num_joints).

    For implicit actuators, the :attr:`effort_limit` and :attr:`effort_limit_sim` are the same.

    - **Explicit actuators**: Typically set to a large value (1.0e9) to avoid double-clipping,
      since the actuator model already clips efforts using :attr:`effort_limit`.
    - **Implicit actuators**: Same as :attr:`effort_limit` (both values are synchronized).
    """

    velocity_limit: torch.Tensor
    """The joint velocity limit for the actuator group [rad/s or m/s]. Shape is (num_envs, num_joints).

    The peak velocity of the actuated joint (the actuator's rated speed reflected at the joint,
    after any gearbox). Feeds the articulation data buffers (e.g. soft joint velocity limits) and
    explicit-model effort clipping; it is not pushed to the physics solver. Defaults to
    :attr:`velocity_limit_sim` when only the solver clamp is configured.
    """

    velocity_limit_sim: torch.Tensor
    """The solver-level velocity clamp for the actuator group [rad/s or m/s]. Shape is (num_envs, num_joints).

    Written to the simulation (PhysX ``maxJointVelocity``); resolved independently of
    :attr:`velocity_limit`.
    """

    stiffness: torch.Tensor
    """The stiffness (P gain) of the PD controller. Shape is (num_envs, num_joints)."""

    damping: torch.Tensor
    """The damping (D gain) of the PD controller. Shape is (num_envs, num_joints)."""

    armature: torch.Tensor
    """The armature of the actuator joints. Shape is (num_envs, num_joints)."""

    friction: torch.Tensor
    """The joint static friction of the actuator joints. Shape is (num_envs, num_joints)."""

    dynamic_friction: torch.Tensor
    """The joint dynamic friction of the actuator joints. Shape is (num_envs, num_joints)."""

    viscous_friction: torch.Tensor
    """The joint viscous friction of the actuator joints. Shape is (num_envs, num_joints)."""

    saturation_effort: torch.Tensor
    """The peak actuator effort [N or N·m, depending on joint type]. Shape is (num_envs, num_joints)."""

    _DEFAULT_MAX_EFFORT_SIM: ClassVar[float] = 1.0e9
    """The default maximum effort for the actuator joints in the simulation. Defaults to 1.0e9.

    If the :attr:`ActuatorBaseCfg.effort_limit_sim` is not specified and the actuator is an explicit
    actuator, then this value is used.
    """

    def __init__(
        self,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_ids: slice | torch.Tensor,
        num_envs: int,
        device: str,
        stiffness: torch.Tensor | float = 0.0,
        damping: torch.Tensor | float = 0.0,
        armature: torch.Tensor | float = 0.0,
        friction: torch.Tensor | float = 0.0,
        dynamic_friction: torch.Tensor | float = 0.0,
        viscous_friction: torch.Tensor | float = 0.0,
        effort_limit: torch.Tensor | float = torch.inf,
        velocity_limit: torch.Tensor | float = torch.inf,
        *,
        debug_value_resolution: bool = True,
    ):
        """Initialize the actuator.

        The actuator parameters are parsed from the configuration and stored as buffers. If the parameters
        are not specified in the configuration, then their values provided in the constructor are used.

        .. note::
            The values in the constructor are typically obtained through the USD values passed from the PhysX API calls
            corresponding to the joints in the actuator model; these values serve as default values if the parameters
            are not specified in the cfg.



        Args:
            cfg: The configuration of the actuator model.
            joint_names: The joint names in the articulation.
            joint_ids: The joint indices in the articulation. If :obj:`slice(None)`, then all
                the joints in the articulation are part of the group.
            num_envs: Number of articulations in the view.
            device: Device used for processing.
            stiffness: The default joint stiffness (P gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            damping: The default joint damping (D gain). Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            armature: The default joint armature. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            friction: The default joint static friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            dynamic_friction: The default joint dynamic friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            viscous_friction: The default joint viscous friction. Defaults to 0.0.
                If a tensor, then the shape is (num_envs, num_joints).
            effort_limit: The default effort limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
            velocity_limit: The default velocity limit. Defaults to infinity.
                If a tensor, then the shape is (num_envs, num_joints).
            debug_value_resolution: Whether to materialize the diagnostic table
                describing configuration/default-value resolution. This private
                construction flag is disabled while resolving source prototypes
                so CUDA construction never reads a device scalar on the host.
        """
        # save parameters
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = device
        self._joint_names = joint_names
        self._joint_indices = joint_ids
        self.joint_property_resolution_table: dict[str, list] = {}
        # For explicit models, we do not want to enforce the effort limit through the solver
        # (unless it is explicitly set)
        if not self.is_implicit_model and self.cfg.effort_limit_sim is None:
            self.cfg.effort_limit_sim = self._DEFAULT_MAX_EFFORT_SIM

        # resolve usd, actuator configuration values
        # case 1: if usd_value == actuator_cfg_value: all good,
        # case 2: if usd_value != actuator_cfg_value: we use actuator_cfg_value
        # case 3: if actuator_cfg_value is None: we use usd_value

        to_check = [
            ("velocity_limit_sim", velocity_limit),
            ("effort_limit_sim", effort_limit),
            ("stiffness", stiffness),
            ("damping", damping),
            ("armature", armature),
            ("friction", friction),
            ("dynamic_friction", dynamic_friction),
            ("viscous_friction", viscous_friction),
        ]
        for param_name, usd_val in to_check:
            cfg_val = getattr(self.cfg, param_name)
            setattr(self, param_name, self._parse_joint_parameter(cfg_val, usd_val))
            new_val = getattr(self, param_name)

            if debug_value_resolution:
                allclose = (
                    torch.all(new_val == usd_val)
                    if isinstance(usd_val, (float, int))
                    else torch.allclose(new_val, usd_val)
                )
                if cfg_val is None or not allclose:
                    self._record_actuator_resolution(
                        cfg_val=getattr(self.cfg, param_name),
                        new_val=new_val[0],  # new val always has the shape of (num_envs, num_joints)
                        usd_val=usd_val,
                        joint_names=joint_names,
                        joint_ids=joint_ids,
                        actuator_param=param_name,
                    )

        self.velocity_limit = self._parse_joint_parameter(self.cfg.velocity_limit, self.velocity_limit_sim)
        # Parse effort_limit with special default handling:
        # - If cfg.effort_limit is None, use the original USD value (effort_limit parameter from constructor)
        # - Otherwise, use effort_limit_sim as the default
        # Please refer to the documentation of the effort_limit and effort_limit_sim parameters for more details.
        effort_default = effort_limit if self.cfg.effort_limit is None else self.effort_limit_sim
        self.effort_limit = self._parse_joint_parameter(self.cfg.effort_limit, effort_default)

        # create commands buffers for allocation
        self.computed_effort = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        self.applied_effort = torch.zeros_like(self.computed_effort)

    def __str__(self) -> str:
        """Returns: A string representation of the actuator group."""
        # resolve joint indices for printing
        joint_indices = self.joint_indices
        if isinstance(joint_indices, slice):
            joint_indices = list(range(self.num_joints))
        # resolve model type (implicit or explicit)
        model_type = "implicit" if self.is_implicit_model else "explicit"

        return (
            f"<class {self.__class__.__name__}> object:\n"
            f"\tModel type            : {model_type}\n"
            f"\tNumber of joints      : {self.num_joints}\n"
            f"\tJoint names expression: {self.cfg.joint_names_expr}\n"
            f"\tJoint names           : {self.joint_names}\n"
            f"\tJoint indices         : {joint_indices}\n"
        )

    """
    Properties.
    """

    @property
    def num_joints(self) -> int:
        """Number of actuators in the group."""
        self._require_current_facade()
        return len(self._joint_names)

    @property
    def num_envs(self) -> int:
        """Number of articulation instances represented by the group."""
        self._require_current_facade()
        return self._num_envs

    @property
    def joint_names(self) -> list[str]:
        """Articulation's joint names that are part of the group."""
        self._require_current_facade()
        return self._joint_names

    @property
    def joint_indices(self) -> slice | torch.Tensor:
        """Articulation's joint indices that are part of the group.

        Note:
            If :obj:`slice(None)` is returned, then the group contains all the joints in the articulation.
            We do this to avoid unnecessary indexing of the joints for performance reasons.
        """
        self._require_current_facade()
        return self._joint_indices

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Managed parameter names in exact-schema declaration order."""
        return tuple(self.parameters)

    @property
    def parameters(self) -> Mapping[str, ProxyArray]:
        """Stable zero-copy managed parameter arrays for inspection.

        The returned mapping is read-only and its :class:`~isaaclab.utils.warp.ProxyArray`
        values alias canonical actuator storage. Do not mutate their ``.torch`` or ``.warp``
        arrays directly: raw mutation cannot notify a backend about parameter side effects.
        Use :meth:`set_parameter_index` or :meth:`set_parameter_mask` for supported writes.
        """
        self._require_current_facade()
        binding = self.__dict__.get("_parameter_binding")
        if binding is None or binding.parameter_proxies is None:
            return _GuardedParameterMapping(self, {})
        mapping = self.__dict__.get("_parameter_mapping")
        if mapping is None:
            mapping = _GuardedParameterMapping(self, binding.parameter_proxies)
            self.__dict__["_parameter_mapping"] = mapping
        return mapping

    """
    Operations.
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int]):
        """Reset the internals within the group.

        Args:
            env_ids: List of environment IDs to reset.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Process the actuator group actions and compute the articulation actions.

        It computes the articulation actions based on the actuator model type

        Args:
            control_action: The joint action instance comprising of the desired joint positions, joint velocities
                and (feed-forward) joint efforts.
            joint_pos: The current joint positions of the joints in the group. Shape is (num_envs, num_joints).
            joint_vel: The current joint velocities of the joints in the group. Shape is (num_envs, num_joints).

        Returns:
            The computed desired joint positions, joint velocities and joint efforts.
        """
        raise NotImplementedError

    """
    Helper functions.
    """

    @classmethod
    def _resolve_managed_registration(
        cls,
        *,
        cfg: ActuatorBaseCfg,
        joint_names: list[str],
        joint_indices: slice | torch.Tensor,
        defaults_by_source: Sequence[Any],
        debug_value_resolution: bool = False,
    ) -> _ResolvedManagedRegistration:
        """Resolve one managed actuator class over source-prototype default rows.

        The existing concrete constructor remains the authority for configuration
        parsing.  It receives only the used source rows, never every cloned
        world, so its resolved numeric tensors can be expanded into canonical
        storage later without creating world-sized temporary parameter buffers.

        Args:
            cfg: Logical actuator-group configuration.
            joint_names: Resolved articulation joint names in group order.
            joint_indices: Articulation joint indices in group order.
            defaults_by_source: One backend-default row per used clone source,
                or one compact property record with a source-row leading axis.
            debug_value_resolution: Whether to retain a materialized
                configuration/default-resolution diagnostic table. Disabled by
                default so source resolution remains device-only on CUDA.

        Returns:
            Private source-bounded resolved registration metadata.

        Raises:
            TypeError: If this exact class does not opt into managed storage or
                a source-default row is malformed.
            ValueError: If no source rows are supplied or their shapes/devices
                do not agree with the logical group topology.
        """
        if cls.__dict__.get("_parameter_schema") is None:
            raise TypeError(f"{cls.__name__} does not opt into managed parameter storage.")
        num_joints = len(joint_names)
        field_names = (
            "stiffness",
            "damping",
            "armature",
            "friction",
            "dynamic_friction",
            "viscous_friction",
            "effort_limit",
            "velocity_limit",
        )
        source_defaults: dict[str, torch.Tensor] = {}
        source_device: torch.device | None = None
        compact_defaults = getattr(defaults_by_source, "stiffness", None) is not None
        if compact_defaults:
            num_sources: int | None = None
            for field_name in field_names:
                value = getattr(defaults_by_source, field_name, None)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{cls.__name__} source default {field_name!r} must be a Torch tensor.")
                if value.dtype is not torch.float32 or value.ndim != 2 or value.shape[1] != num_joints:
                    raise ValueError(
                        f"{cls.__name__} source default {field_name!r} must have shape "
                        f"(num_sources, {num_joints}) and float32 dtype."
                    )
                if num_sources is None:
                    num_sources = value.shape[0]
                elif value.shape[0] != num_sources:
                    raise ValueError(f"{cls.__name__} source defaults must share one source-row count.")
                if source_device is None:
                    source_device = value.device
                elif value.device != source_device:
                    raise ValueError(f"{cls.__name__} source defaults must share one device.")
                source_defaults[field_name] = value
            if not num_sources:
                raise ValueError(f"{cls.__name__} requires at least one source-prototype default row.")
        else:
            if not defaults_by_source:
                raise ValueError(f"{cls.__name__} requires at least one source-prototype default row.")
            for field_name in field_names:
                rows: list[torch.Tensor] = []
                for source_index, defaults in enumerate(defaults_by_source):
                    value = getattr(defaults, field_name, None)
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"{cls.__name__} source {source_index} default {field_name!r} must be a Torch tensor."
                        )
                    if value.dtype is not torch.float32 or value.ndim != 2 or value.shape != (1, num_joints):
                        raise ValueError(
                            f"{cls.__name__} source {source_index} default {field_name!r} must have shape "
                            f"(1, {num_joints}) and float32 dtype."
                        )
                    if source_device is None:
                        source_device = value.device
                    elif value.device != source_device:
                        raise ValueError(f"{cls.__name__} source defaults must share one device.")
                    rows.append(value)
                source_defaults[field_name] = torch.cat(rows, dim=0)

        copied_cfg = cfg.copy() if hasattr(cfg, "copy") else copy.deepcopy(cfg)
        num_sources = next(iter(source_defaults.values())).shape[0]
        source_shell = cls(
            cfg=copied_cfg,
            joint_names=list(joint_names),
            joint_ids=joint_indices,
            num_envs=num_sources,
            device=str(source_device),
            **source_defaults,
            debug_value_resolution=debug_value_resolution,
        )
        schema = cls._parameter_schema()
        source_values = {
            field.name: getattr(source_shell, field.name).detach()
            for field in schema.fields
            if field.role == "parameter"
        }
        structural_signature = getattr(cls, "_structural_signature", lambda _cfg: ())(copied_cfg)
        return _ResolvedManagedRegistration(
            cfg=copied_cfg,
            actuator_type=cls,
            joint_names=tuple(joint_names),
            joint_indices=joint_indices,
            source_shell=source_shell,
            source_values=source_values,
            structural_signature=(
                cls,
                tuple(field.name for field in schema.fields),
                structural_signature,
            ),
        )

    @classmethod
    def _build_managed_runtime_shell(
        cls,
        *,
        resolved: _ResolvedManagedRegistration,
        binding: _GroupBinding,
        num_envs: int,
        device: str,
        joint_indices: slice | torch.Tensor,
    ) -> ActuatorBase:
        """Bind one exact managed runtime shell to canonical parameter storage.

        Stateless exact classes inherit this source-shell rebinding path.  A
        class with world-sized structural state may override it privately to
        allocate only that state after canonical parameter binding.

        Args:
            resolved: Source-prototype metadata from
                :meth:`_resolve_managed_registration`.
            binding: Canonical exact-type parameter binding for this group.
            num_envs: Final articulation-world count.
            device: Device hosting canonical storage.
            joint_indices: Final articulation joint indices in group order.

        Returns:
            The exact configured actuator class bound to canonical storage.
        """
        if resolved.actuator_type is not cls:
            raise TypeError(f"Resolved registration belongs to {resolved.actuator_type.__name__}, not {cls.__name__}.")
        runtime = copy.copy(resolved.source_shell)
        runtime._num_envs = num_envs
        runtime._device = device
        runtime._joint_names = list(resolved.joint_names)
        runtime._joint_indices = joint_indices
        runtime._move_managed_runtime_structure(device)
        runtime._bind_parameter_storage(binding)
        runtime._rebuild_managed_runtime_state()
        return runtime

    def _move_managed_runtime_structure(self, device: str) -> None:
        """Move private non-parameter structure to the final runtime device.

        Managed source shells are resolved on the backend's source transport,
        which may be CPU even when the final runtime actuator executes on CUDA.
        Each source shell is consumed by exactly one runtime group and then
        released with candidate build state. Subclasses override this hook for
        structural tensors or modules that cannot be recreated during
        :meth:`_rebuild_managed_runtime_state`.

        Args:
            device: Device hosting the final runtime actuator.
        """
        del device

    def _rebuild_managed_runtime_state(self) -> None:
        """Rebuild private world-sized state after canonical parameter binding.

        Stateless actuator classes retain their source-shell structural objects.
        Stateful subclasses override this hook to allocate only state and scratch
        whose leading dimension depends on the final articulation-world count.
        """

    @classmethod
    def _build_execution_actuator(cls, actuators: Sequence[ActuatorBase]) -> ActuatorBase:
        """Build one private executor from resolved logical actuator groups."""
        parameter_values = {
            name: torch.cat([getattr(actuator, name) for actuator in actuators], dim=1)
            for name in cls._execution_parameter_names()
        }
        executor = copy.copy(actuators[0])
        executor.__dict__.pop("_facade_view", None)
        executor.__dict__.pop("_facade_token", None)
        executor.__dict__.pop("_parameter_binding", None)
        executor._joint_names = [name for actuator in actuators for name in actuator.joint_names]
        for name, value in parameter_values.items():
            setattr(executor, name, value)
        executor.computed_effort = torch.zeros(executor._num_envs, len(executor._joint_names), device=executor._device)
        executor.applied_effort = torch.zeros_like(executor.computed_effort)
        return executor

    @classmethod
    def _execution_parameter_names(cls) -> tuple[str, ...]:
        """Return execution parameters derived from an exact-class schema."""
        if cls.__dict__.get("_parameter_schema") is None:
            return cls._SOLVER_COMPATIBILITY_PARAMETER_NAMES
        schema = cls._parameter_schema()
        return tuple(field.name for field in schema.fields if field.role == "parameter") + (
            cls._SOLVER_COMPATIBILITY_PARAMETER_NAMES
        )

    def _bind_parameter_storage(self, binding: _GroupBinding) -> None:
        """Bind this exact built-in group to canonical typed parameter arrays."""
        if type(self).__dict__.get("_parameter_schema") is None:
            raise TypeError(f"{type(self).__name__} does not opt into managed parameter storage.")
        self.__dict__["_parameter_binding"] = binding
        self.__dict__["_deprecated_sidecars"] = {}
        self.__dict__["_deprecated_sidecar_warnings"] = set()
        self.__dict__["_solver_compatibility_sidecars"] = {}

    def _bind_solver_compatibility_seeds(self, seeds: Mapping[str, _SolverCompatibilitySeed]) -> None:
        """Attach final compact solver rows for lazy public compatibility fields."""
        self.__dict__["_solver_compatibility_seeds"] = dict(seeds)

    def _release_managed_source_parameters(self) -> None:
        """Drop source-shell tensors now shadowed by canonical arrays or compact seeds."""
        for field in type(self)._parameter_schema().fields:
            self.__dict__.pop(field.name, None)
        for name in self._SOLVER_COMPATIBILITY_PARAMETER_NAMES:
            self.__dict__.pop(name, None)

    def _bind_facade_view(self, view: Any, token: object) -> None:
        """Bind generation checks to the owning articulation facade."""
        self.__dict__["_facade_view"] = view
        self.__dict__["_facade_token"] = token

    def _require_current_facade(self) -> None:
        """Reject public group access after its articulation generation expires."""
        view = self.__dict__.get("_facade_view")
        if view is not None:
            view._require_current_generation(self.__dict__.get("_facade_token"))

    def _require_facade_execution_ready(self) -> None:
        """Reject public group writes unless its articulation can execute."""
        view = self.__dict__.get("_facade_view")
        if view is not None:
            view._require_execution_ready(self.__dict__.get("_facade_token"))

    def _release_facade_storage(self) -> None:
        """Detach canonical aliases while keeping stale-generation checks alive."""
        state = self.__dict__
        mapping = state.get("_parameter_mapping")
        if mapping is not None:
            mapping._values.clear()
        binding = state.get("_parameter_binding")
        if binding is not None and binding.parameter_proxies is not None:
            clear = getattr(binding.parameter_proxies, "clear", None)
            if clear is not None:
                clear()
        retained = {
            name: state[name]
            for name in ("cfg", "_num_envs", "_device", "_joint_names", "_facade_view", "_facade_token")
            if name in state
        }
        state.clear()
        state.update(retained)

    def set_parameter_index(
        self,
        name: str,
        value: float | torch.Tensor | wp.array(dtype=wp.float32) | Sequence[float] | Sequence[Sequence[float]],
        *,
        env_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | wp.array(dtype=wp.int64) | None = None,
        joint_ids: Sequence[int] | torch.Tensor | wp.array(dtype=wp.int32) | wp.array(dtype=wp.int64) | None = None,
    ) -> None:
        """Set one managed parameter using Cartesian articulation index selectors.

        Args:
            name: Managed parameter name.
            value: A scalar; compact values with shape
                ``[len(joint_ids)]``; or Cartesian values with shape
                ``[len(env_ids), len(joint_ids)]``. When :paramref:`joint_ids`
                is ``None``, its length is this group's compact DOF count; when
                :paramref:`env_ids` is ``None``, its length is ``num_worlds``.
                Units follow :paramref:`name`: stiffness [N/m or N·m/rad],
                damping [N·s/m or N·m·s/rad], effort and saturation limits
                [N or N·m], and velocity limits [m/s or rad/s], depending on
                joint type.
            env_ids: Signed articulation-world indices with shape
                ``[num_selected_worlds]``. ``None`` selects every world.
            joint_ids: Signed articulation-DOF indices with shape
                ``[num_selected_joints]``. ``None`` selects the group's compact
                slots in configuration order.

        The selected Cartesian pairs retain their supplied value rows and
        columns: filtering out-of-range worlds, out-of-range joints, or joints
        outside this group's scope does not shift source columns. In normal
        mode, duplicate environment or joint IDs use the last Cartesian
        occurrence. ``joint_ids=None`` addresses compact group slots
        individually in stable configuration order. With debug validation
        enabled, bounds, duplicates, and ownership violations synchronously
        raise instead.

        Raises:
            KeyError: If :paramref:`name` is not managed by this group.
            TypeError: If a value or selector has an unsupported dtype.
            ValueError: If selector/value metadata is malformed, values cannot
                broadcast, an overlapping value source exceeds the bounded
                staging capacity, or debug validation rejects selector contents.
            RuntimeError: If the group is stale or its facade is not execution-ready.
        """
        self._require_facade_execution_ready()
        view = self.__dict__.get("_facade_view")
        if view is None:
            raise RuntimeError("Actuator group is not bound to a scoped facade.")
        view._write_group_parameter_index(self, name, value, env_ids, joint_ids)

    def set_parameter_mask(
        self,
        name: str,
        value: float | torch.Tensor | wp.array(dtype=wp.float32) | Sequence[float] | Sequence[Sequence[float]],
        *,
        env_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
        joint_mask: torch.Tensor | wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Set one managed parameter using full-articulation masks.

        Args:
            name: Managed parameter name.
            value: A scalar; compact values with shape ``[num_scope_dofs]``;
                or world-by-compact values with shape
                ``[num_worlds, num_scope_dofs]``. Units follow
                :paramref:`name`: stiffness [N/m or N·m/rad], damping
                [N·s/m or N·m·s/rad], effort and saturation limits [N or N·m],
                and velocity limits [m/s or rad/s], depending on joint type.
            env_mask: Boolean full-articulation world mask with shape
                ``[num_worlds]``. ``None`` selects every world.
            joint_mask: Boolean full-articulation DOF mask with shape
                ``[num_joints]``. ``None`` selects every joint.

        Values are indexed by stable compact group slots, not by the count of
        ``True`` entries in :paramref:`joint_mask`. The masks select full
        articulation domains; entries outside this group's scope are ignored in
        every mode. Debug validation performs value-dependent bounds, ownership,
        and duplicate checks only for index selectors.

        Raises:
            KeyError: If :paramref:`name` is not managed by this group.
            TypeError: If a value or mask has an unsupported dtype.
            ValueError: If selector/value metadata is malformed or values cannot
                broadcast, or an overlapping value source exceeds the bounded
                staging capacity.
            RuntimeError: If the group is stale or its facade is not execution-ready.
        """
        self._require_facade_execution_ready()
        view = self.__dict__.get("_facade_view")
        if view is None:
            raise RuntimeError("Actuator group is not bound to a scoped facade.")
        view._write_group_parameter_mask(self, name, value, env_mask, joint_mask)

    def _get_compatibility_sidecar(self, name: str) -> torch.Tensor:
        """Return a lazy solver-only compatibility buffer for a bound group."""
        sidecars = self.__dict__.setdefault("_solver_compatibility_sidecars", {})
        if name not in sidecars:
            seed = self.__dict__.get("_solver_compatibility_seeds", {}).pop(name, None)
            sidecars[name] = (
                seed.materialize(num_envs=self._num_envs, device=self._device)
                if seed is not None
                else self._make_group_local_sidecar(name)
            )
        return sidecars[name]

    def _get_deprecated_gain_sidecar(self, name: str) -> torch.Tensor:
        """Return a lazy neural gain compatibility buffer for a bound group."""
        sidecars = self.__dict__.setdefault("_deprecated_sidecars", {})
        if name not in sidecars:
            if name not in {"stiffness", "damping"}:
                raise AttributeError(f"Managed actuator parameter '{name}' is not allocated.")
            sidecars[name] = self._make_group_local_sidecar(name)
            view = self.__dict__.get("_facade_view")
            if view is None:
                warned = self.__dict__.setdefault("_deprecated_sidecar_warnings", set())
                if name not in warned:
                    warnings.warn(
                        f"{type(self).__name__}.{name} is a deprecated neural-actuator compatibility sidecar.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    warned.add(name)
            else:
                view._warn_deprecated(
                    f"neural_gain_{name}",
                    f"{type(self).__name__}.{name} is a deprecated neural-actuator compatibility sidecar.",
                    stacklevel=4,
                )
        return sidecars[name]

    def _refresh_solver_compatibility_sidecar(self, name: str, values: torch.Tensor) -> None:
        """Refresh an already-materialized solver compatibility sidecar safely."""
        sidecar = self.__dict__.get("_solver_compatibility_sidecars", {}).get(name)
        binding = self.__dict__.get("_parameter_binding")
        if sidecar is not None and binding is not None:
            sidecar.copy_(values[:, binding.joint_indices])

    def _make_group_local_sidecar(self, name: str) -> torch.Tensor:
        """Allocate a group-local sidecar initialized from its resolved value."""
        owned = self.__dict__.get(name)
        if isinstance(owned, torch.Tensor):
            return owned.clone()
        return torch.zeros(self._num_envs, self.num_joints, device=self._device)

    def _record_actuator_resolution(self, cfg_val, new_val, usd_val, joint_names, joint_ids, actuator_param: str):
        if actuator_param not in self.joint_property_resolution_table:
            self.joint_property_resolution_table[actuator_param] = []
        table = self.joint_property_resolution_table[actuator_param]

        ids = joint_ids if isinstance(joint_ids, torch.Tensor) else list(range(len(joint_names)))
        for idx, name in enumerate(joint_names):
            cfg_val_log = "Not Specified" if cfg_val is None else float(new_val[idx])
            default_usd_val = usd_val if isinstance(usd_val, (float, int)) else float(usd_val[0][idx])
            applied_val_log = default_usd_val if cfg_val is None else float(new_val[idx])
            table.append([name, int(ids[idx]), default_usd_val, cfg_val_log, applied_val_log])

    def _parse_joint_parameter(
        self, cfg_value: float | dict[str, float] | None, default_value: float | torch.Tensor | None
    ) -> torch.Tensor:
        """Parse the joint parameter from the configuration.

        Args:
            cfg_value: The parameter value from the configuration. If None, then use the default value.
            default_value: The default value to use if the parameter is None. If it is also None,
                then an error is raised.

        Returns:
            The parsed parameter value.

        Raises:
            TypeError: If the parameter value is not of the expected type.
            TypeError: If the default value is not of the expected type.
            ValueError: If the parameter value is None and no default value is provided.
            ValueError: If the default value tensor is the wrong shape.
        """
        # create parameter buffer
        param = torch.zeros(self._num_envs, self.num_joints, device=self._device)
        # parse the parameter
        if cfg_value is not None:
            if isinstance(cfg_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(cfg_value)
            elif isinstance(cfg_value, dict):
                # if dict, then parse the regular expression
                indices, _, values = string_utils.resolve_matching_names_values(cfg_value, self.joint_names)
                # note: need to specify type to be safe (e.g. values are ints, but we want floats)
                param[:, indices] = torch.tensor(values, dtype=torch.float, device=self._device)
            else:
                raise TypeError(
                    f"Invalid type for parameter value: {type(cfg_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or dict."
                )
        elif default_value is not None:
            if isinstance(default_value, (float, int)):
                # if float, then use the same value for all joints
                param[:] = float(default_value)
            elif isinstance(default_value, torch.Tensor):
                # if tensor, then use the same tensor for all joints
                if default_value.shape == (self._num_envs, self.num_joints):
                    param = default_value.float()
                else:
                    raise ValueError(
                        "Invalid default value tensor shape.\n"
                        f"Got: {default_value.shape}\n"
                        f"Expected: {(self._num_envs, self.num_joints)}"
                    )
            else:
                raise TypeError(
                    f"Invalid type for default value: {type(default_value)} for "
                    + f"actuator on joints {self.joint_names}. Expected float or Tensor."
                )
        else:
            raise ValueError("The parameter value is None and no default value is provided.")

        return param

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """Clip the desired torques based on the motor limits.

        Args:
            desired_torques: The desired torques to clip.

        Returns:
            The clipped torques.
        """
        return torch.clip(effort, min=-self.effort_limit, max=self.effort_limit)
