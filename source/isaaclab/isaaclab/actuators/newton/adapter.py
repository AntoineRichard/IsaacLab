# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-actuator adapter shared by Newton, PhysX, and OVPhysX.

Owns the actuator-state lifecycle, the pre-clamp computed-effort buffer,
and the per-step ``step`` / ``reset`` / ``finalize`` calls. The
:meth:`~NewtonActuatorAdapter.from_usd` classmethod parses
``NewtonActuator`` USD prims for PhysX and OVPhysX. Newton populates
``model.actuators`` itself.

DR gain updates bypass the adapter — the articulation writes straight
to drive arrays.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
import warp as wp
from newton.actuators import Actuator, ClampingBase, Delay, parse_actuator_prim

from .kernels import (
    build_implicit_dof_mask,
    build_per_dof_env_mask_kernel,
    set_mask_kernel,
    zero_at_indices_kernel,
)

if TYPE_CHECKING:
    from .. import ActuatorCollection

# ---------------------------------------------------------------------------
# Abstract base — backend-independent logic
# ---------------------------------------------------------------------------


class NewtonActuatorAdapter:
    """Adapter that wraps a list of :class:`newton.actuators.Actuator`.

    Owns the actuator-state lifecycle, DOF-to-actuator bookkeeping,
    stepping, reset, and the pre-clamp computed-effort buffer the
    in-graph telemetry kernel reads on the post-actuator hook.
    """

    @dataclass(frozen=True)
    class ArticulationBinding:
        """Newton fast-path init state for one articulation.

        Returned by :meth:`bind_articulation`. Bundles the implicit-DOF mask
        and the per-articulation view of the adapter's computed-effort buffer.
        """

        implicit_dof_mask: wp.array
        """Per-DOF mask consumed by ``sync_torque_telemetry``; ``1`` on implicit-actuator DOFs, ``0`` otherwise."""

        implicit_dof_mask_owner: torch.Tensor
        """Torch tensor owning the memory :attr:`implicit_dof_mask` aliases; keep referenced for the mask's lifetime."""

        computed_effort_view: wp.array
        """This articulation's slice of the adapter's pre-clamp computed-effort buffer, ``(num_envs, num_joints)``."""

    def __init__(
        self,
        actuators: list[Actuator],
        num_envs: int,
        num_joints: int,
        dof_offset: int,
        device: str,
    ):
        self.actuators = actuators
        self.num_joints = num_joints

        self._num_envs = num_envs
        self._dof_offset = dof_offset
        self._device = device

        # Collect the set of local DOFs covered by some actuator. Only the
        # env-0 slice of each actuator's flat ``indices`` array is needed —
        # later envs are repeats with a constant ``num_joints`` stride.
        managed: set[int] = set()
        for act in actuators:
            all_indices = act.indices.numpy()
            num_per_act = len(all_indices) // num_envs
            for global_dof in all_indices[:num_per_act]:
                local_dof = global_dof - dof_offset
                if 0 <= local_dof < num_joints:
                    managed.add(local_dof)

        if len(managed) == num_joints:
            self.joint_indices: torch.Tensor | slice = slice(None)
        else:
            self.joint_indices = torch.tensor(sorted(managed), dtype=torch.int32, device=device)

        self._states_a = [act.state() for act in actuators]
        self._states_b = [act.state() for act in actuators]

        # Pre-clamp computed effort buffer. Each Newton actuator scatter-adds
        # its raw drive output to ``sim_control.joint_computed_f`` when
        # ``control_computed_output_attr`` is set; we route that to this
        # buffer so the post-actuator telemetry kernel can report the actual
        # computed (pre-clamp) effort instead of mirroring ``joint_f``. The
        # binding onto ``sim_control`` happens in :meth:`finalize`.
        self._computed_effort = wp.zeros(
            num_envs * num_joints,
            dtype=wp.float32,
            device=device,
        )
        self.computed_effort_2d = self._computed_effort.reshape((num_envs, num_joints))
        for act in actuators:
            act.control_computed_output_attr = "joint_computed_f"

    def finalize(self, sim_control: Any) -> None:
        """Bind the pre-clamp computed-effort buffer onto ``sim_control``.

        Args:
            sim_control: The ``sim_control`` object that will be passed
                to :meth:`step` for this adapter's lifetime. Newton's
                ``Control`` on the Newton backend, an
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
        """
        sim_control.joint_computed_f = self._computed_effort

    def step(self, sim_state: Any, sim_control: Any, dt: float, *, swap_state: bool = True) -> None:
        """Zero actuated DOFs, step all actuators, and advance state buffers.

        Args:
            sim_state: Object with ``joint_q``, ``joint_qd``, etc.
                Newton ``State`` on the Newton backend,
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            sim_control: Object with ``joint_f``, ``joint_target_q``, etc.
                Newton ``Control`` on the Newton backend,
                :class:`~isaaclab.actuators.newton.physx_wrapper.PhysxActuatorWrapper`
                on the PhysX backend.
            dt: Physics timestep [s].
            swap_state: Swap state buffers after stepping. Set to ``False`` on the last step
                of an odd-length graph to copy the output back into the input buffers instead,
                keeping their addresses stable across replays.
        """
        # Zero before scatter-add (actuators accumulate into this buffer).
        self._computed_effort.zero_()
        for act in self.actuators:
            wp.launch(
                zero_at_indices_kernel,
                dim=act.indices.shape[0],
                inputs=[sim_control.joint_f, act.indices],
            )
        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            act.step(sim_state, sim_control, sa, sb, dt=dt)
        if swap_state:
            self._swap_state_buffers()
        else:
            for sa, sb in zip(self._states_a, self._states_b):
                if sa is not None:
                    sa.assign(sb)

    def _swap_state_buffers(self) -> None:
        """Advance the actuator state ping-pong after an eager step or graph replay."""
        self._states_a, self._states_b = self._states_b, self._states_a

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Reset actuator states for the given environments.

        Args:
            env_ids: Environment indices to reset. ``None`` (or
                ``slice(None)``, which IsaacLab callers sometimes pass)
                resets all environments. Otherwise expects a torch tensor
                or sequence of int indices.

        Newton's :meth:`Actuator.State.reset` expects a per-DOF boolean
        mask of length ``num_actuators`` (= ``num_envs * dofs_per_actuator``),
        not a per-env mask — each entry gates the corresponding column of
        the actuator's state buffers (delay queue, drive integral,
        etc.). We therefore build a per-actuator per-DOF mask from the
        env mask before delegating to each state.
        """
        if env_ids is None or env_ids == slice(None):
            for sa, sb in zip(self._states_a, self._states_b):
                if sa is not None:
                    sa.reset(None)
                if sb is not None:
                    sb.reset(None)
            return

        num_envs = len(range(self._num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        if num_envs == 0:
            return
        env_mask = wp.zeros(self._num_envs, dtype=wp.bool, device=self._device)
        if isinstance(env_ids, slice):
            wp.to_torch(env_mask)[env_ids] = True
        else:
            # Torch indexed scalar assignment uploads True and synchronizes on CUDA.
            indices = (
                wp.from_torch(env_ids)
                if isinstance(env_ids, torch.Tensor)
                else wp.array(env_ids, dtype=wp.int32, device=self._device)
            )
            wp.launch(set_mask_kernel, dim=num_envs, inputs=[env_mask, indices], device=self._device)

        for act, sa, sb in zip(self.actuators, self._states_a, self._states_b):
            per_dof_mask = wp.zeros(act.indices.shape[0], dtype=wp.bool, device=self._device)
            wp.launch(
                build_per_dof_env_mask_kernel,
                dim=act.indices.shape[0],
                inputs=[act.indices, env_mask, self._dof_offset, self.num_joints, per_dof_mask],
                device=self._device,
            )
            if sa is not None:
                sa.reset(per_dof_mask)
            if sb is not None:
                sb.reset(per_dof_mask)

    def bind_articulation(
        self,
        *,
        implicit_joint_indices: Sequence[slice | torch.Tensor | None],
        dof_offset: int,
        num_joints: int,
    ) -> ArticulationBinding:
        """Assemble the Newton fast-path init state for one articulation.

        Builds the implicit-DOF mask and slices this adapter's
        computed-effort buffer to the articulation's columns.

        Args:
            implicit_joint_indices: Joint selectors of the articulation's implicit
                actuator groups in public joint order; they define
                :attr:`ArticulationBinding.implicit_dof_mask`.
            dof_offset: Offset of this articulation's DOFs in the adapter's
                env-major global index space (``0`` on PhysX, view-dependent
                on Newton).
            num_joints: Articulation-local joint count. Distinct from
                :attr:`num_joints`, which is the whole-model per-env DOF
                stride used to lay out the actuator index arrays.

        Returns:
            The bundled :class:`ArticulationBinding` for this articulation.
        """
        implicit_dof_mask, implicit_dof_mask_owner = build_implicit_dof_mask(
            implicit_joint_indices, num_joints, self._device
        )
        computed_effort_view = self.computed_effort_2d[:, dof_offset : dof_offset + num_joints]
        return self.ArticulationBinding(
            implicit_dof_mask=implicit_dof_mask,
            implicit_dof_mask_owner=implicit_dof_mask_owner,
            computed_effort_view=computed_effort_view,
        )

    @property
    def is_all_graphable(self) -> bool:
        """``True`` when all actuators are CUDA-graph-safe."""
        return len(self.actuators) > 0 and all(a.is_graphable() for a in self.actuators)

    @property
    def is_stateful(self) -> bool:
        """``True`` when any actuator maintains delay or drive state."""
        return any(a.is_stateful() for a in self.actuators)

    @classmethod
    def from_usd(
        cls,
        stage: Any,
        joint_names: list[str],
        num_envs: int,
        num_joints: int,
        device: str,
        articulation_prim_path: str | None = None,
    ) -> NewtonActuatorAdapter:
        """Build an adapter from ``NewtonActuator`` prims authored on *stage*.

        This is the host-adapter counterpart of Newton's
        ``ModelBuilder.add_usd``. It reads the same prims and constructs matching
        :class:`~newton.actuators.Actuator` objects. Structurally compatible
        joints are merged into one actuator with per-DOF parameter arrays and
        combined indices. Newton backends use ``model.actuators`` instead.

        On PhysX and OVPhysX, :paramref:`joint_names` is in this adapter's local
        public order and defines the local indices assigned to parsed actuator targets.

        Args:
            stage: USD stage containing ``NewtonActuator`` prims.
            joint_names: All articulation joint names in adapter-local public order.
            num_envs: Number of environments.
            num_joints: Number of joints per environment.
            device: Warp device string, for example ``"cuda:0"``.
            articulation_prim_path: Root prim path of environment zero's
                articulation. When set, only prims under this subtree are
                considered; otherwise the whole stage is scanned.

        Returns:
            Adapter whose actuator indices use :paramref:`joint_names` order.

        Raises:
            ValueError: If no authored actuator targets a name in
                :paramref:`joint_names`.
        """
        actuators = _create_actuators_from_usd(
            stage,
            joint_names,
            num_envs,
            num_joints,
            device,
            articulation_prim_path=articulation_prim_path,
        )
        return cls(actuators, num_envs, num_joints, dof_offset=0, device=device)


# ---------------------------------------------------------------------------
# Stand-in for the proposed ``Actuator.get_parameter`` / ``Actuator.set_parameter``.
# Both index through ``actuator.indices``, the array the actuator already uses to
# read its targets every step, so no Model or view is needed. Delete this block
# once Newton ships the methods and call them on the actuator instead.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _get_actuator_parameter_kernel(
    param: wp.array(dtype=Any),
    indices: wp.array(dtype=wp.uint32),
    out: wp.array(dtype=Any),
):
    """Scatter one actuator parameter into a flat DOF-layout array."""
    i = wp.tid()
    out[indices[i]] = param[i]


@wp.kernel(enable_backward=False)
def _set_actuator_parameter_kernel(
    values: wp.array(dtype=Any),
    indices: wp.array(dtype=wp.uint32),
    mask: wp.array(dtype=wp.bool),
    use_mask: bool,
    param: wp.array(dtype=Any),
):
    """Gather one actuator parameter from a flat DOF-layout array, gated by a per-DOF mask."""
    i = wp.tid()
    dof = int(indices[i])
    if use_mask:
        if not mask[dof]:
            return
    param[i] = values[dof]


def _actuator_get_parameter(actuator: Actuator, component: Any, name: str, out: wp.array) -> wp.array:
    """Stand-in for the proposed ``Actuator.get_parameter``.

    Writes ``out[actuator.indices[i]] = component.<name>[i]``. Entries of
    ``out`` that the actuator does not drive are left untouched.

    Args:
        actuator: Actuator whose ``indices`` place the parameter in ``out``.
        component: The actuator's component that owns the parameter.
        name: Parameter name on ``component``.
        out: Flat array in the actuator's DOF layout, in the parameter's dtype.

    Returns:
        ``out``.
    """
    wp.launch(
        _get_actuator_parameter_kernel,
        dim=actuator.indices.shape[0],
        inputs=[getattr(component, name), actuator.indices],
        outputs=[out],
        device=out.device,
    )
    return out


def _actuator_set_parameter(
    actuator: Actuator, component: Any, name: str, values: wp.array, mask: wp.array | None = None
) -> None:
    """Stand-in for the proposed ``Actuator.set_parameter``.

    Writes ``component.<name>[i] = values[actuator.indices[i]]`` where
    ``mask[actuator.indices[i]]`` is set, or everywhere when ``mask`` is ``None``.

    Args:
        actuator: Actuator whose ``indices`` locate its entries in ``values``.
        component: The actuator's component that owns the parameter.
        name: Parameter name on ``component``.
        values: Flat array in the actuator's DOF layout, in the parameter's dtype.
        mask: Optional per-DOF mask in the same layout as ``values``.
    """
    param = getattr(component, name)
    wp.launch(
        _set_actuator_parameter_kernel,
        dim=actuator.indices.shape[0],
        inputs=[values, actuator.indices, mask, mask is not None],
        outputs=[param],
        device=param.device,
    )


# ---------------------------------------------------------------------------
# Component-addressed, group-scoped parameter access.
# ---------------------------------------------------------------------------


def read_group_parameter(collection: ActuatorCollection, name: str, component: str, attr: str) -> torch.Tensor:
    """Read one live Newton actuator parameter for a native group.

    Group-scoped, user-ordered reads of the drive-owned storage. For raw
    component access, use the group's Newton actuator object (the collection
    mapping entry) directly.

    Args:
        collection: The articulation's actuator collection.
        name: Actuator group name.
        component: Component kind: ``"drive"``, ``"delay"``, or ``"clamping"``.
            ``"controller"`` is a deprecated alias for ``"drive"``.
        attr: Parameter name on that component (e.g. ``"kp"``, ``"max_effort"``).

    Returns:
        Live values in the group's joint order, shape
        ``(num_instances, group_num_joints)``, in the parameter's dtype.
        Units follow the addressed parameter.

    Raises:
        ValueError: If the group is not executed by Newton actuators, the
            component name is unknown, or no actuator exposes the parameter.
    """
    owners = _group_parameter_owners(collection, name, component, attr)
    # Each owner fills the DOFs it drives; DOFs no owner drives read as zeros.
    values = _dof_layout_zeros(collection, getattr(owners[0][1], attr).dtype)
    for actuator, owner in owners:
        _actuator_get_parameter(actuator, owner, attr, values)
    return wp.to_torch(values).view(collection.num_instances, -1)[:, collection._newton_group_columns(name)]


def write_group_parameter(
    collection: ActuatorCollection,
    name: str,
    component: str,
    attr: str,
    values: torch.Tensor,
    env_ids: torch.Tensor | slice | None = None,
    joint_ids: torch.Tensor | None = None,
) -> None:
    """Write one Newton actuator parameter for a native group.

    Group-scoped, user-ordered writes that reach the drive-owned storage
    through the actuator's own DOF indices. For raw component access, use the group's
    Newton actuator object (the collection mapping entry) directly.

    Args:
        collection: The articulation's actuator collection.
        name: Actuator group name.
        component: Component kind: ``"drive"``, ``"delay"``, or ``"clamping"``.
            ``"controller"`` is a deprecated alias for ``"drive"``.
        attr: Parameter name on that component (e.g. ``"kp"``, ``"max_effort"``).
        values: New values, shape ``(len(env_ids), len(joint_ids))``. Units
            follow the addressed parameter.
        env_ids: Environment indices to update. Defaults to all environments.
        joint_ids: Group-local joint indices to update. Defaults to all of
            the group's joints.

    Raises:
        ValueError: Same conditions as :func:`read_group_parameter`.
    """
    owners = _group_parameter_owners(collection, name, component, attr)
    columns = collection._newton_group_columns(name)
    if joint_ids is not None:
        columns = columns[joint_ids.to(collection.device, dtype=torch.long)]
    rows = slice(None) if env_ids is None else env_ids
    if isinstance(rows, torch.Tensor):
        rows = rows[:, None]
    # The per-DOF mask selects exactly the written cells, so no read-back of current values is needed.
    staged = _dof_layout_zeros(collection, getattr(owners[0][1], attr).dtype)
    mask = _dof_layout_zeros(collection, wp.bool)
    staged_torch = wp.to_torch(staged).view(collection.num_instances, -1)
    staged_torch[rows, columns] = values.to(collection.device, dtype=staged_torch.dtype)
    wp.to_torch(mask).view(collection.num_instances, -1)[rows, columns] = True
    for actuator, owner in owners:
        _actuator_set_parameter(actuator, owner, attr, staged, mask)


def _group_parameter_owners(
    collection: ActuatorCollection, name: str, component: str, attr: str
) -> list[tuple[Actuator, Any]]:
    """Resolve the component instances exposing ``attr`` for one native group."""
    if component == "controller":
        # stacklevel=3 points at the caller of read_group_parameter / write_group_parameter.
        warnings.warn(
            "The 'controller' actuator component is deprecated. Use 'drive' instead; "
            "'controller' will be removed in 3.2.",
            DeprecationWarning,
            stacklevel=3,
        )
        component = "drive"
    if name not in collection._groups:
        raise KeyError(name)
    if collection._newton_selection is None or name not in collection._native_group_names:
        raise ValueError(f"Actuator group '{name}' is not executed by Newton actuators.")
    group_actuators = collection._groups[name]
    if not isinstance(group_actuators, tuple):
        group_actuators = (group_actuators,)
    owners = [
        (actuator, owner)
        for actuator in group_actuators
        if (owner := resolve_actuator_component(actuator, component, attr)) is not None
    ]
    if not owners:
        raise ValueError(f"No Newton actuator exposes parameter ('{component}', '{attr}').")
    return owners


def _dof_layout_zeros(collection: ActuatorCollection, dtype: type) -> wp.array:
    """Zeroed flat array in the Newton actuators' DOF layout, ``num_instances * dof_stride`` long."""
    size = collection.num_instances * collection._newton_selection.dof_stride
    return wp.zeros(size, dtype=dtype, device=collection.device)


def resolve_actuator_component(actuator: Actuator, component: str, attr: str) -> Any | None:
    """Return the component instance that exposes ``attr`` on the addressed component kind.

    ``component`` selects the actuator's ``"drive"``, ``"delay"``, or
    ``"clamping"`` entry; the returned object is what the actuator's
    parameter get/set take as their ``component`` argument. Returns ``None`` when the component is absent
    on this actuator or does not expose ``attr``. Raises ``ValueError`` on
    unknown component names or ambiguous clamping matches.
    """
    if component == "drive":
        owner = actuator.drive
    elif component == "delay":
        owner = getattr(actuator, "delay", None)
    elif component == "clamping":
        matches = [entry for entry in (getattr(actuator, "clamping", None) or []) if hasattr(entry, attr)]
        if len(matches) > 1:
            names = ", ".join(type(entry).__name__ for entry in matches)
            raise ValueError(f"Ambiguous clamping parameter '{attr}': exposed by {names}.")
        owner = matches[0] if matches else None
    else:
        raise ValueError(f"Unknown actuator component '{component}'. Expected 'drive', 'delay', or 'clamping'.")
    if owner is None or not hasattr(owner, attr):
        return None
    return owner


@dataclass(frozen=True)
class NewtonActuatorSelection:
    """Execution-setup handoff for Newton actuator parameter access.

    Pure data returned by
    :meth:`~isaaclab.actuators.ActuatorControl.finalize_native_actuators` and
    consumed by the group-scoped parameter access functions
    (:func:`read_group_parameter` / :func:`write_group_parameter`) and by the
    collection when it maps native groups to their Newton actuator objects.

    The actuators' flat ``indices`` address DOF ``world * dof_stride + dof_offset + column``,
    where ``column`` is the articulation's backend-order DOF.
    """

    actuators: list[Actuator]
    """Newton actuators that may drive this articulation's DOFs."""

    dof_stride: int
    """Per-world stride of the actuators' flat DOF indices [DOFs]."""

    dof_offset: int = 0
    """Offset of this articulation's first DOF within one world's stride [DOFs]."""

    joint_user_to_backend_indices: tuple[int, ...] | None = None
    """Optional public-to-backend joint permutation for the articulation's DOF columns."""


# ---------------------------------------------------------------------------
# PhysX-only USD parsing
# ---------------------------------------------------------------------------

_ResolvedComponent: TypeAlias = tuple[type, dict[str, Any]]
_ResolvedActuatorSpec: TypeAlias = tuple[int, type, dict[str, Any], list[_ResolvedComponent]]


def _actuator_signature(
    drive_class: type,
    drive_arguments: dict[str, Any],
    component_arguments: list[_ResolvedComponent],
) -> tuple:
    """Build Newton's structural grouping key for a parsed actuator spec."""

    def make_hashable(value: Any) -> Any:
        if isinstance(value, list | tuple):
            return tuple(make_hashable(item) for item in value)
        return value

    def shared_key(component_class: type, resolved: dict[str, Any]) -> tuple:
        shared_names = getattr(component_class, "SHARED_PARAMS", set())
        return tuple(sorted((name, make_hashable(resolved[name])) for name in shared_names if name in resolved))

    clamping_key: list[tuple] = []
    has_delay = False
    for comp_cls, resolved in component_arguments:
        if issubclass(comp_cls, Delay):
            has_delay = True
        elif issubclass(comp_cls, ClampingBase):
            clamping_key.append((comp_cls, shared_key(comp_cls, resolved)))

    return (drive_class, has_delay, tuple(clamping_key), shared_key(drive_class, drive_arguments))


def _tile_per_dof_arguments(
    arguments: list[dict[str, Any]],
    num_envs: int,
    dtype: type,
    device: wp.Device,
) -> dict[str, wp.array]:
    """Pack per-joint scalar arguments in environment-major order."""
    if not arguments:
        return {}

    numpy_dtype = np.int32 if dtype == wp.int32 else np.float32
    return {
        name: wp.array(
            np.tile(np.asarray([per_joint[name] for per_joint in arguments], dtype=numpy_dtype), num_envs),
            dtype=dtype,
            device=device,
        )
        for name in arguments[0]
    }


def _create_actuators_from_usd(
    stage: Any,
    joint_names: list[str],
    num_envs: int,
    num_total_joints: int,
    device: str,
    articulation_prim_path: str | None = None,
) -> list[Actuator]:
    """Parse ``NewtonActuator`` prims and instantiate standalone actuators.

    This mirrors the actuator construction that Newton's
    ``ModelBuilder.add_usd`` performs, but operates independently of a
    Newton ``Model``.  It is used on the PhysX backend where there is no
    Newton simulation — actuators are stepped manually via the adapter.

    Because PhysX articulations have no free or ball joints, every
    joint's coordinate count equals its DOF count.  A single
    ``indices`` array is therefore sufficient for all index roles
    (``indices``, ``pos_indices``, ``target_pos_indices``).

    Joints with the same drive and clamping structure are merged into
    one :class:`Actuator`. Scalar parameters (``kp``, ``kd``,
    ``saturation_effort``, delay, etc.) are packed per DOF. Parameters marked
    as ``SHARED_PARAMS`` (e.g. ``model_path``, ``lookup_positions``) remain
    part of the grouping key and are passed through directly.
    """
    from pxr import Usd  # noqa: PLC0415

    wp_device = wp.get_device(device)

    joint_name_to_idx: dict[str, int] = {name: i for i, name in enumerate(joint_names)}

    if articulation_prim_path is not None:
        root_prim = stage.GetPrimAtPath(articulation_prim_path)
    else:
        root_prim = stage.GetPseudoRoot()

    parsed_per_joint: dict[int, Any] = {}
    for prim in Usd.PrimRange(root_prim):
        parsed = parse_actuator_prim(prim)
        if parsed is None:
            continue
        target_name = parsed.target_path.rsplit("/", 1)[-1]
        if target_name in joint_name_to_idx:
            parsed_per_joint[joint_name_to_idx[target_name]] = parsed

    if not parsed_per_joint:
        raise ValueError(f"No NewtonActuator prims found targeting any of: {joint_names}")

    groups: dict[tuple, list[_ResolvedActuatorSpec]] = defaultdict(list)
    for local_idx, parsed in sorted(parsed_per_joint.items()):
        drive_arguments = parsed.drive_class.resolve_arguments(dict(parsed.drive_kwargs))
        component_arguments = [
            (comp_cls, comp_cls.resolve_arguments(comp_kwargs)) for comp_cls, comp_kwargs in parsed.component_specs
        ]
        sig = _actuator_signature(parsed.drive_class, drive_arguments, component_arguments)
        groups[sig].append((local_idx, parsed.drive_class, drive_arguments, component_arguments))

    actuators = []
    for grouped_specs in groups.values():
        local_indices = [spec[0] for spec in grouped_specs]
        drive_class = grouped_specs[0][1]
        resolved_drives = [spec[2] for spec in grouped_specs]
        resolved_components = [spec[3] for spec in grouped_specs]

        flat_indices = np.array(
            [idx + e * num_total_joints for e in range(num_envs) for idx in local_indices],
            dtype=np.uint32,
        )
        indices = wp.array(flat_indices, device=wp_device)

        # Drive
        shared_drive = getattr(drive_class, "SHARED_PARAMS", set())
        drive_arguments = [
            {key: value for key, value in resolved.items() if key not in shared_drive} for resolved in resolved_drives
        ]
        drive_shared = {key: value for key, value in resolved_drives[0].items() if key in shared_drive}
        drive = drive_class(
            **_tile_per_dof_arguments(drive_arguments, num_envs, wp.float32, wp_device),
            **drive_shared,
        )

        # Components (delay + clampings)
        clamping_components = [
            [(comp_cls, resolved) for comp_cls, resolved in components if issubclass(comp_cls, ClampingBase)]
            for components in resolved_components
        ]
        delay_arguments = [
            resolved
            for components in resolved_components
            for comp_cls, resolved in components
            if issubclass(comp_cls, Delay)
        ]

        delay = None
        if delay_arguments:
            max_delay = max(int(arguments["delay_steps"]) for arguments in delay_arguments)
            if max_delay > 0:
                delay = Delay(
                    **_tile_per_dof_arguments(delay_arguments, num_envs, wp.int32, wp_device),
                    max_delay=max_delay,
                )

        clampings = []
        for component_index, (comp_cls, _) in enumerate(clamping_components[0]):
            resolved_clampings = [components[component_index][1] for components in clamping_components]
            shared_clamp = getattr(comp_cls, "SHARED_PARAMS", set())
            clamp_arguments = [
                {key: value for key, value in resolved.items() if key not in shared_clamp}
                for resolved in resolved_clampings
            ]
            clamp_shared = {key: value for key, value in resolved_clampings[0].items() if key in shared_clamp}
            clampings.append(
                comp_cls(
                    **_tile_per_dof_arguments(clamp_arguments, num_envs, wp.float32, wp_device),
                    **clamp_shared,
                )
            )

        actuator = Actuator(
            indices=indices,
            drive=drive,
            delay=delay,
            clamping=clampings if clampings else None,
            control_target_pos_attr="joint_target_pos",
            control_target_vel_attr="joint_target_vel",
        )
        actuators.append(actuator)

    return actuators
