# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit OVPhysX data and operations, usable without a SimulationContext.

The native runtime still permits one attached stage per process. Instance-owned
Python data does not provide independently stepped native scenes.
"""

from __future__ import annotations

import importlib.util
import math
import os
import stat
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import ovphysx
import ovstage
import warp as wp

from pxr import Plug

from isaaclab.scene_data import SceneDataBackend, SceneDataFormat
from isaaclab.scene_data.deformable_discovery import deformable_geometry_batches
from isaaclab.utils import instantiate
from isaaclab.utils.buffers import TimestampedBuffer

from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView
from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

from . import ovphysx_compat
from .ovphysx_compat import OVPHYSX_LIFECYCLE_ENTRY_POINTS
from .ovphysx_manager_cfg import DEFAULT_COOKED_COLLIDER_CACHE_DIR, OvPhysxBackendCfg

if TYPE_CHECKING:
    from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

    from .ovphysx_manager import OvPhysxManager

__all__ = [
    "OvPhysxBackend",
    "OvPhysxSceneDataBackend",
    "step",
    "forward",
    "update_kinematics",
    "set_gravity",
]


_physx_schemas_registered = False


def step(backend: OvPhysxBackend, dt: float) -> None:
    """Advance the attached stage by one timestep [s] and publish its state.

    Native completion is synchronous. This function does not capture a CUDA graph
    or advance a manager's simulation clock.

    Args:
        backend: Native resource whose attached stage is advanced.
        dt: Physics timestep [s].

    Raises:
        RuntimeError: If the resource has closed.
    """
    if backend.physx is None:
        raise RuntimeError("The OVPhysX backend has closed.")
    backend.physx.step_sync(dt=dt)
    backend.kinematics_dirty = True
    forward(backend)


def forward(backend: OvPhysxBackend) -> None:
    """Evaluate pending kinematics and invalidate the resource's scene-data publication.

    Args:
        backend: Native resource whose manually written state is published.
    """
    update_kinematics(backend)
    if backend.scene_data is not None:
        backend.scene_data.transforms_timestamp += 1
        backend.scene_data.geometry_timestamp += 1


def update_kinematics(backend: OvPhysxBackend) -> None:
    """Evaluate dirty articulation kinematics without publishing or rendering.

    Args:
        backend: Native resource whose articulation state may have changed.
    """
    if backend.kinematics_dirty and backend.physx is not None:
        backend.physx.update_articulations_kinematic()
        backend.kinematics_dirty = False


def set_gravity(
    backend: OvPhysxBackend,
    gravity: tuple[float, float, float],
    *,
    physics_prim_path: str = "/World/physicsScene",
) -> None:
    """Apply a world-frame gravity vector [m/s^2] through a sealed OVStage control edit.

    Args:
        backend: Native resource with an attached stage.
        gravity: Three finite world-frame acceleration components [m/s^2].
        physics_prim_path: Path of the physics scene to update.

    Raises:
        RuntimeError: If the backend has no attached stage.
        ValueError: If gravity does not contain three finite values.
    """
    if backend.physx is None or backend.stage is None:
        raise RuntimeError("The OVPhysX backend has no attached stage.")
    gravity_array = np.asarray(gravity, dtype=np.float32)
    if gravity_array.shape != (3,) or not np.all(np.isfinite(gravity_array)):
        raise ValueError("Gravity must contain three finite values.")
    magnitude = float(np.linalg.norm(gravity_array))
    if math.isclose(magnitude, 0.0):
        direction = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    else:
        direction = (gravity_array / magnitude).reshape(1, 3)
    scene = backend.scene
    # Consume only this edit, excluding pending rendering/output writes.
    scene.commit()
    ordinal = scene.ordinal
    with scene.query([physics_prim_path]) as query:
        scene.stage.write_attribute(query, "physics:gravityDirection", ordinal, direction, is_array=False).wait()
        scene.stage.write_attribute(
            query, "physics:gravityMagnitude", ordinal, np.array([magnitude], dtype=np.float32), is_array=False
        ).wait()
        scene.commit()
        operation = backend.physx.update_from_ovstage(ordinal, ordinal)
        ovphysx_compat.finish_stage_operation(backend.physx, operation)
    # A failed native update need not be atomic; publish only successful completion.
    backend.gravity = tuple(float(component) for component in gravity_array)


class OvPhysxSceneDataBackend(SceneDataBackend):
    """Scene-data backend for the OVPhysX physics manager.

    The runtime's rigid-body view fills one pose buffer directly. Deformable views
    fill their native slices without an intermediate packing pass.
    """

    def __init__(self, manager: OvPhysxManager | None = None):
        self.backend: OvPhysxBackend | None = None
        """Native resource borrowed from the context registry after warmup."""
        self._transforms = TimestampedBuffer(SceneDataFormat.Transform())
        self.transforms_timestamp = 0
        self.geometry_timestamp = 0
        self._geometry = TimestampedBuffer()
        self._deformable_bindings: list[tuple[OvPhysxView, Any, wp.array]] = []
        self._pending_setup: tuple[OvPhysxBackend, str, Sequence[DeformableStageEntry] | None] | None = None

    def _defer_setup(
        self, backend: OvPhysxBackend, device: str, entries: Sequence[DeformableStageEntry] | None = None
    ) -> None:
        """Defer renderer-only bindings until scene data is requested."""
        self._clear()
        self._pending_setup = (backend, device, entries)

    def _clear(self) -> None:
        """Discard publications and deferred reads before their native handles are released."""
        self.backend = None
        self._transforms.data.transforms = None
        self.transforms_timestamp += 1
        self._deformable_bindings = []
        self._geometry = TimestampedBuffer()
        self.geometry_timestamp += 1
        self._pending_setup = None

    def _ensure_setup(self) -> None:
        if self._pending_setup is not None:
            backend, device, entries = self._pending_setup
            # Native output metadata names the actual bodies, not articulation-root aliases.
            # Keep these renderer-only bindings out of headless simulation startup.
            from ovphysx.types import SimObjectType

            kinds = (SimObjectType.RIGID_BODY, SimObjectType.ARTICULATION_LINK)
            body_paths = []
            if ovphysx_compat.OVPHYSX_USES_TASKS:
                for kind in kinds:
                    with backend.physx.read_sync(kind, ["mass"]) as bodies:
                        for group in bodies.groups:
                            body_paths.extend(bodies.dictionary.get_path_strings(group.prim_list))
            else:
                from ovstage import PathDictionary

                with PathDictionary() as paths:
                    for kind in kinds:
                        with backend.physx.read(kind, ["mass"]) as bodies:
                            for group in bodies.groups:
                                body_paths.extend(paths.get_path_strings(group.prim_list))
            if body_paths:
                backend.rigid_body_view = backend.physx.create_tensor_binding(prim_paths=body_paths)
            self.setup(backend, device, entries)
            self._pending_setup = None

    @property
    def transform_count(self) -> int:
        """Number of poses in the native publication."""
        self._ensure_setup()
        poses = self._transforms.data.transforms
        return 0 if poses is None else len(poses)

    @property
    def transform_paths(self) -> list[str]:
        """Native body paths in the same order as the published poses."""
        self._ensure_setup()
        view = None if self.backend is None else self.backend.rigid_body_view
        return [] if view is None else view.prim_paths

    def setup(
        self, backend: OvPhysxBackend, device: str, entries: Sequence[DeformableStageEntry] | None = None
    ) -> None:
        """Bind the native body table and declared deformable geometry.

        Args:
            backend: Initialized native physics resource.
            device: Warp device string used to allocate the published buffers.
            entries: Declared deformables captured before native stage import. ``None`` leaves
                scene geometry uninitialized; an empty sequence declares no deformables.
        """
        self.backend = backend
        self._transforms.data.transforms = None
        self.transforms_timestamp += 1
        self._deformable_bindings = []
        self.geometry_timestamp += 1
        self._geometry = TimestampedBuffer()

        if backend.rigid_body_view is not None and backend.rigid_body_view.count:
            self._transforms.data.transforms = wp.empty(
                backend.rigid_body_view.count, dtype=wp.transformf, device=device
            )

        if entries is not None:
            self._setup_deformable_bindings(backend.physx, entries, device)

    def _setup_deformable_bindings(self, physx, entries: Sequence[DeformableStageEntry], device: str) -> None:
        """Bind exact planned deformables directly into one flat publication buffer."""
        from isaaclab_ov import tensor_types as TT

        groups = {}
        for entry in entries:
            groups.setdefault((entry.deformable_type, entry.vertex_count), []).append(entry)

        views, native_entries = [], []
        for (deformable_type, count), entries in groups.items():
            tensor_type = (
                TT.DEFORMABLE_SIM_NODAL_POSITION if deformable_type == "volume" else TT.SURFACE_DEFORMABLE_SIM_POSITION
            )
            view = OvPhysxView(
                physx,
                prim_paths=[entry.root_path for entry in entries],
                device=device,
                tensor_types=[tensor_type],
                eager=True,
            )
            by_path = {path: entry for entry in entries for path in (entry.root_path, entry.sim_mesh_path)}
            ordered = [by_path[path] for path in view.prim_paths]
            if sorted(entry.root_path for entry in ordered) != sorted(entry.root_path for entry in entries):
                raise RuntimeError("OVPhysX deformable binding does not cover the declared clone-plan entries.")
            if tuple(view.binding_for(tensor_type).shape) != (len(entries), count, 3):
                raise RuntimeError("OVPhysX deformable node counts disagree with the clone plan.")
            native_entries.extend(ordered)
            views.append((view, tensor_type, count))

        if not views:
            self._geometry.data = []
            return
        counts = [entry.vertex_count for entry in native_entries]
        points = wp.empty(sum(counts), dtype=wp.vec3f, device=device)
        offset = 0
        for view, tensor_type, count in views:
            buffer = wp.array(
                ptr=points.ptr + offset * wp.types.type_size_in_bytes(wp.vec3f),
                shape=(view.count, count),
                dtype=wp.vec3f,
                device=device,
                copy=False,
            )
            self._deformable_bindings.append((view, tensor_type, buffer))
            offset += view.count * count
        offsets = np.cumsum(np.r_[0, counts[:-1]])
        self._geometry.data = deformable_geometry_batches(native_entries, offsets, device=device)
        for publication, _ in self._geometry.data:
            publication.points = points

    def get_geometry_batches(self, output_format: Any = SceneDataFormat.Points) -> list:
        """Publish native positions with exact visual paths and interpolation metadata.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry.data is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        if self._geometry.timestamp != self.geometry_timestamp:
            for view, tensor_type, buffer in self._deformable_bindings:
                view.read_into(tensor_type, buffer)
            self._geometry.timestamp = self.geometry_timestamp
        return self._geometry.data

    @property
    def native_geometry_formats(self) -> tuple[Any, ...]:
        """Return the native geometry formats compiled from the declared prototypes.

        Raises:
            RuntimeError: If scene geometry was not initialized from a clone plan.
        """
        self._ensure_setup()
        if self._geometry.data is None:
            raise RuntimeError("Declare and replicate a ClonePlan before requesting scene geometry.")
        return tuple(dict.fromkeys(publication._cls for publication, _ in self._geometry.data))

    @property
    def transforms(self) -> SceneDataFormat.Transform:
        """Publish native rigid-body poses [m, xyzw]."""
        self._ensure_setup()
        if self._transforms.timestamp != self.transforms_timestamp:
            if self.backend is not None:
                update_kinematics(self.backend)
            if self._transforms.data.transforms is not None:
                self.backend.rigid_body_view.read(self._transforms.data.transforms)
            self._transforms.timestamp = self.transforms_timestamp
        return self._transforms.data


class OvPhysxBackend:
    """Own one native handle and its stage-bound data, independently of a simulation context.

    Multiple handles may coexist, but OVPhysX 0.6.4 permits only one attached stage
    per process. A second :meth:`load` fails until the owner detaches or closes.
    Use separate processes for independently stepped simulations.
    """

    def __init__(
        self,
        cfg: OvPhysxBackendCfg,
        *,
        scene: OvstageBackend | None = None,
        scene_data: OvPhysxSceneDataBackend | None = None,
    ):
        """Construct the native resource without attaching a USD stage.

        Args:
            cfg: Native runtime configuration.
            scene: Shared OVStage resource to borrow, or ``None`` to own a private stage.
            scene_data: Optional publication updated by :func:`step` and :func:`forward`.
        """
        ovphysx.bootstrap()
        self.device = cfg.device
        is_gpu = cfg.device.startswith("cuda:")
        cache_dir = cfg.cooked_collider_cache_dir
        if cache_dir == DEFAULT_COOKED_COLLIDER_CACHE_DIR:
            cache_dir = _prepare_default_cache_dir(cache_dir)
        carbonite_overrides = {
            "/physics/physxDispatcher": True,
            "/physics/updateToUsd": False,
            "/physics/updateVelocitiesToUsd": False,
            "/physics/updateParticlesToUsd": False,
            "/ovphysx/clone/useEnvIds": is_gpu,
        }
        if is_gpu:
            carbonite_overrides.update({"/physics/suppressReadback": True, "/physics/suppressFabricUpdate": True})
        # CPU scenes select CPU dynamics through their scene attributes (see _set_scene_dynamics_device).
        # OVPhysX's process-wide CPU-only mode is irreversible and would block later CUDA scenes.
        self.physx = ovphysx.PhysX(
            config=ovphysx.PhysXConfig(
                num_threads=8, cooked_collider_cache_dir=cache_dir, carbonite_overrides=carbonite_overrides
            ),
            active_cuda_gpus=cfg.device.removeprefix("cuda:") if is_gpu else None,
        )
        self.scene = scene
        self.owns_scene = scene is None
        self.stage: Any = None
        self.rigid_body_view: Any = None
        self.scene_data = scene_data
        self.kinematics_dirty = False
        self.gravity: tuple[float, float, float] | None = None
        """Last successfully applied gravity [m/s^2], or ``None`` before a control update."""

    def load(self, stage_usda: str) -> None:
        """Replace the attached scene after draining its native bindings.

        Args:
            stage_usda: Exported USD scene to populate before physics attaches.
        """
        _ensure_physx_schemas_registered()
        if self.scene_data is not None:
            self.scene_data._clear()
        if self.stage is not None:
            if self.rigid_body_view is not None:
                self.rigid_body_view.destroy()
                self.rigid_body_view = None
            OvPhysxView._close_all_for(self.physx)
            ovphysx_compat.detach_stage(self.physx)
            self.scene.reset()
            self.stage = None
        if self.scene is None:
            # PHYSICS population currently omits native-instance collider dependencies.
            self.scene = instantiate(OvstageBackendCfg(population_domains=ovstage.PopulationDomain.ALL))
        try:
            self.scene.populate(stage_usda)
            operation = self.physx.attach_ovstage(self.scene.stage, read_ordinal=self.scene.ordinal - 1)
        except Exception:
            if self.owns_scene:
                self.scene.close()
                self.scene = None
            raise
        self.stage = self.scene.stage
        # Accepted attach failures can retain native ownership. Keep the stage for cleanup.
        ovphysx_compat.finish_stage_operation(self.physx, operation)
        self.kinematics_dirty = True
        self.gravity = None
        if self.scene_data is not None:
            self.scene_data._defer_setup(self, self.device)

    def close(self) -> None:
        """Release bindings, the runtime, and its stage in native teardown order."""
        if self.scene_data is not None:
            self.scene_data._clear()
        physx = self.physx
        if physx is None:
            if self.owns_scene and self.scene is not None:
                self.scene.close()
            self.stage = None
            return

        # Legacy release errors are terminal; current destroy errors may be retryable.
        destroy_entry_point = OVPHYSX_LIFECYCLE_ENTRY_POINTS["destroy"]
        release_owners = destroy_entry_point == "release"
        try:
            try:
                if self.rigid_body_view is not None:
                    self.rigid_body_view.destroy()
                    self.rigid_body_view = None
                OvPhysxView._close_all_for(physx)
            finally:
                try:
                    ovphysx_compat.detach_stage(physx)
                finally:
                    try:
                        getattr(physx, destroy_entry_point)()
                    except Exception:
                        if destroy_entry_point == "destroy":
                            # Keep both owners if native teardown did not reach its terminal state.
                            try:
                                physx.handle
                            except RuntimeError:
                                release_owners = True
                            except Exception:
                                release_owners = False
                        raise
                    else:
                        release_owners = True
        finally:
            if release_owners:
                self.physx = None
                self.kinematics_dirty = False
                if self.owns_scene and self.scene is not None:
                    self.scene.close()
                self.stage = None


def _prepare_default_cache_dir(cache_dir: str) -> str:
    """Create the default cooked-collider cache directory and refuse one this user does not own.

    The default sits in the shared temporary directory, so any local user can pre-create the path;
    OVPhysX follows a symlink there and writes through it. A directory the caller configured is
    their own choice and is passed through untouched.

    Args:
        cache_dir: Default cache directory to create or validate.

    Returns:
        The validated directory.

    Raises:
        RuntimeError: If the path exists as a symlink, a non-directory, or another user's directory.
    """
    try:
        os.makedirs(cache_dir, mode=0o700)
        return cache_dir
    except FileExistsError:
        pass
    entry = os.lstat(cache_dir)
    if stat.S_ISLNK(entry.st_mode):
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' is a symlink; refusing to write through it.")
    if not stat.S_ISDIR(entry.st_mode):
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' exists and is not a directory.")
    if hasattr(os, "getuid") and entry.st_uid != os.getuid():
        raise RuntimeError(f"OVPhysX cache directory '{cache_dir}' is owned by another user; refusing to use it.")
    return cache_dir


def _ensure_physx_schemas_registered() -> None:
    """Register process-wide USD schemas before either host or native population."""
    global _physx_schemas_registered
    if _physx_schemas_registered:
        return
    schema_paths = ovphysx_compat.physx_schema_paths(ovphysx)
    ovstage.population.register_usd_schemas([str(path) for path in schema_paths])
    # Locate the required schema package without importing it into the host USD registry.
    ovstage.population.register_usd_schemas(os.path.dirname(importlib.util.find_spec("newton_usd_schemas").origin))
    registry = Plug.Registry()
    registered_names = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
    # The wheel documents ``<module>/resources`` as its stable layout and its
    # bundled plugin names match those module directory names case-insensitively.
    schema_paths = [str(path) for path in schema_paths if path.parent.name.casefold() not in registered_names]
    if schema_paths:
        registry.RegisterPlugins(schema_paths)
    _physx_schemas_registered = True
