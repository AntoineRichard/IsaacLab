Changelog
---------

0.2.18 (2026-05-05)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.assets.Articulation` and
  :class:`~isaaclab_ovphysx.assets.ArticulationData` mirroring the PhysX/Newton
  articulation API: index/mask split for every state writer, simulation-parameter
  writer, setter, and tendon setter; pull-on-demand timestamped buffers; first-class
  CPU-only binding routing through pinned-host staging.
* Added dedicated :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_dynamic_friction_coefficient_to_sim_index`,
  :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_dynamic_friction_coefficient_to_sim_mask`,
  :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_viscous_friction_coefficient_to_sim_index`,
  and :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_viscous_friction_coefficient_to_sim_mask`
  to mirror PhysX's per-component friction surface; each touches only its slot in the
  combined ``(N, J, 3)`` ``DOF_FRICTION_PROPERTIES`` buffer.
* Added deprecated non-indexed shorthand shims
  :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_friction_coefficient_to_sim`,
  :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_dynamic_friction_coefficient_to_sim`,
  and :meth:`~isaaclab_ovphysx.assets.Articulation.write_joint_viscous_friction_coefficient_to_sim`
  for migration parity with PhysX.

Changed
^^^^^^^

* Reorganized :class:`~isaaclab_ovphysx.assets.Articulation` and
  :class:`~isaaclab_ovphysx.assets.ArticulationData` to mirror PhysX's section
  layout exactly: bare ``"""Section."""`` docstring blocks (replacing
  ``# --- Section ---`` comment banners), with internal helpers and deprecated
  surface placed at the bottom of each class.
* Renamed the three friction-coefficient properties on
  :class:`~isaaclab_ovphysx.assets.ArticulationData` to match PhysX:
  ``joint_friction_static`` → :attr:`~isaaclab_ovphysx.assets.ArticulationData.joint_friction_coeff`,
  ``joint_friction_dynamic`` → :attr:`~isaaclab_ovphysx.assets.ArticulationData.joint_dynamic_friction_coeff`,
  ``joint_friction_viscous`` → :attr:`~isaaclab_ovphysx.assets.ArticulationData.joint_viscous_friction_coeff`.
* Tightened :attr:`~isaaclab_ovphysx.assets.Articulation.instantaneous_wrench_composer`
  and :attr:`~isaaclab_ovphysx.assets.Articulation.permanent_wrench_composer`
  return types from ``WrenchComposer | None`` to ``WrenchComposer``,
  matching PhysX/Newton.
* Type-annotated ``binding_getter`` on
  :meth:`~isaaclab_ovphysx.assets.ArticulationData.__init__` as
  ``Callable[[int], Any] | None``.
* Expanded docstrings on the OVPhysX articulation Warp kernels
  (:func:`~isaaclab_ovphysx.assets.articulation.kernels.compute_soft_joint_pos_limits_func`,
  :func:`~isaaclab_ovphysx.assets.articulation.kernels.update_soft_joint_pos_limits`,
  :func:`~isaaclab_ovphysx.assets.articulation.kernels._fd_joint_acc`) to match
  PhysX's documentation depth (purpose, ``Args:`` table with shape/dtype/SI units,
  divergence notes where the OVPhysX implementation differs).

Removed
^^^^^^^

* Removed OVPhysX-only :meth:`~isaaclab_ovphysx.assets.Articulation.set_external_force_and_torque_index`
  and :meth:`~isaaclab_ovphysx.assets.Articulation.set_external_force_and_torque_mask`.
  Use :attr:`~isaaclab_ovphysx.assets.Articulation.instantaneous_wrench_composer` /
  :attr:`~isaaclab_ovphysx.assets.Articulation.permanent_wrench_composer` and call
  ``add_forces_and_torques_index`` / ``add_forces_and_torques_mask`` on the composer
  directly, matching PhysX/Newton.
* Removed the OVPhysX-only ``set_spatial_tendon_limit_{index,mask}`` and
  ``set_spatial_tendon_rest_length_{index,mask}`` ``NotImplementedError`` stubs;
  the equivalents are not exposed by PhysX.
* Removed redundant or behavior-divergent overrides of base-class shorthand
  properties on :class:`~isaaclab_ovphysx.assets.ArticulationData`:
  ``body_pose_w``, ``body_lin_vel_w``, ``body_ang_vel_w``, ``body_acc_w``,
  ``body_link_acc_w``.  The defaults from
  :class:`~isaaclab.assets.BaseArticulationData` apply and match PhysX.
* Removed forward-compat ``body_inv_mass``, ``body_inv_inertia``,
  ``fixed_tendon_limit``, ``spatial_tendon_limit``, and
  ``spatial_tendon_rest_length`` properties on
  :class:`~isaaclab_ovphysx.assets.ArticulationData` along with their backing
  buffers; these had no PhysX equivalent.

0.2.17 (2026-05-05)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Made :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._release_physx` a
  soft reset that calls ``physx.reset()`` and keeps the cached
  :class:`ovphysx.PhysX` reference alive, instead of dropping it to ``None``
  (which triggered a dual-Carbonite destructor race on refcount drop).
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._warmup_and_load` now
  reuses the cached instance on subsequent calls, re-running ``add_usd``,
  pending clones, and (on GPU) ``warmup_gpu`` per stage swap.  This makes
  back-to-back :class:`SimulationContext` lifetimes work natively without
  the test-side monkey patches the previous iteration of the rigid-object
  tests required.

Added
^^^^^

* Added :attr:`~isaaclab_ovphysx.physics.OvPhysxManager._locked_device` so
  the manager raises a clear :exc:`RuntimeError` when a later
  :class:`SimulationContext` requests a different device, surfacing the
  wheel's process-global device-mode lock as a Python error before
  :exc:`ovphysx.types.PhysXDeviceError` would fire.

0.2.16 (2026-04-30)
~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Aligned :class:`~isaaclab_ovphysx.assets.RigidObject` with the Newton-style
  index/mask kernel split: ``set_root_link_pose_to_sim``,
  ``set_root_com_pose_to_sim``, ``set_root_com_velocity_to_sim``, and
  ``set_root_link_velocity_to_sim`` are renamed to ``*_to_sim_index`` and no
  longer take a ``from_mask`` flag or unused ``root_link_state_w`` /
  ``root_state_w`` outputs.  Same simplification for
  ``write_2d_data_to_buffer_with_indices``,
  ``write_body_inertia_to_buffer_index``, and
  ``write_body_com_pose_to_buffer_index``.
* Dropped the ``full_data`` parameter from every
  :class:`~isaaclab_ovphysx.assets.RigidObject` ``*_index`` writer / setter
  (``write_root_*_to_sim_index``, ``set_masses_index``, ``set_coms_index``,
  ``set_inertias_index``).  Index methods now strictly accept partial data
  shaped ``(len(env_ids), ...)``; full-data callers should use the matching
  ``*_mask`` overload instead.  This matches the Newton/PhysX convention.
* Reworded every public docstring on
  :class:`~isaaclab_ovphysx.assets.RigidObject` to follow the Newton/PhysX
  template (one-line summary, ``.. note::`` and ``.. tip::`` blocks,
  ``Args:`` block with shape/dtype on the parameter line).

Removed
^^^^^^^

* Removed the GPU-side write plumbing
  (``RigidObject._write_body_state``, ``_com_pose_to_link_pose``,
  ``_to_flat_f32``, ``_as_gpu_f32_2d``, ``_get_write_scratch``,
  ``_stage_to_pinned_cpu``, ``_binding_write``, ``_binding_read``,
  ``_to_cpu_numpy``, ``_to_cpu_indices``, ``_env_ids_to_gpu_warp``,
  ``_n_envs_index``).  The deprecated ``write_root_state_to_sim`` /
  ``write_root_com_state_to_sim`` / ``write_root_link_state_to_sim`` shims now
  call the public ``write_root_*_to_sim_index`` methods directly, mirroring
  PhysX/Newton.
* Removed the now-unused ``_compose_root_link_pose_from_com`` kernel from
  :mod:`isaaclab_ovphysx.assets.kernels`; the ``set_root_com_pose_to_sim_*``
  kernels recover the link pose inline via ``get_com_pose_in_link_frame_func``.
* Removed the ``masses`` 1-D-to-``(K, 1)`` auto-reshape from
  :meth:`~isaaclab_ovphysx.assets.RigidObject.set_masses_index`; callers must
  pass shape ``(len(env_ids), len(body_ids))`` explicitly (matches PhysX).

0.2.15 (2026-04-30)
~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`~isaaclab_ovphysx.assets.RigidObject` and
  :class:`~isaaclab_ovphysx.assets.RigidObjectData` for single-actor rigid-body
  simulation against the OVPhysX backend, satisfying the
  :class:`~isaaclab.assets.BaseRigidObject` and
  :class:`~isaaclab.assets.BaseRigidObjectData` contracts.
* Added the ``RIGID_BODY_*`` :class:`TensorType` aliases in
  :mod:`isaaclab_ovphysx.tensor_types` (``POSE``, ``VELOCITY``, ``WRENCH``,
  ``MASS``, ``COM_POSE``, ``INERTIA``; plus ``ACCELERATION``, ``INV_MASS``,
  ``INV_INERTIA`` declared for forward-compat once the wheel ships them).
* Added the shared kernel module :mod:`isaaclab_ovphysx.assets.kernels`
  vendored from PhysX/Newton: frame-conversion (``get_root_link_vel_from_root_com_vel``,
  ``get_root_com_pose_from_root_link_pose``, ``_compose_root_link_pose_from_com``),
  state concatenation (``concat_root_pose_and_vel_to_state`` + ``vec13f``),
  derivative kernels (``derive_body_acceleration_from_body_com_velocities``),
  index-style writers (``write_2d_data_to_buffer_with_indices``,
  ``write_body_inertia_to_buffer``, ``write_body_com_pose_to_buffer``,
  ``set_root_*_to_sim``), and mask-style writers
  (``write_2d_data_to_buffer_with_mask``, ``write_body_inertia_to_buffer_mask``,
  ``write_body_com_pose_to_buffer_mask``, ``set_root_*_to_sim_mask``).
* Added USD prim-scan validation to
  :meth:`~isaaclab_ovphysx.assets.RigidObject._initialize_impl` (mirrors PhysX):
  ``RuntimeError`` is raised when ``cfg.prim_path`` resolves to no
  ``UsdPhysics.RigidBodyAPI`` prim, multiple rigid-body prims, or a prim with an
  enabled ``UsdPhysics.ArticulationRootAPI``.
* Documented the wheel's process-global device-mode lock (gap G5) in the
  ``test/assets/test_rigid_object.py`` module docstring and the
  ``scripts/run_ovphysx.sh`` header: full coverage requires two separate
  pytest invocations (``-k 'cpu'`` and ``-k 'cuda:0'``).

Changed
^^^^^^^

* Aligned :class:`~isaaclab_ovphysx.assets.RigidObject` and
  :class:`~isaaclab_ovphysx.assets.RigidObjectData` with the PhysX (PR #5329)
  and Newton conventions:

  * Eager :class:`~isaaclab.utils.buffers.TimestampedBufferWarp` allocation in
    :meth:`_create_buffers` (called from ``__init__``); ``num_instances`` /
    ``num_bodies`` / ``body_names`` are now constructor args.
  * Setters / writers (``set_masses_index`` / ``set_coms_index`` /
    ``set_inertias_index`` / ``write_root_*_to_sim_index``) scatter user data
    into the cached buffer via the matching ``write_*`` / ``set_root_*_to_sim``
    kernel, then push the cache via ``binding.write(cache, indices=...)`` with
    pre-allocated pinned-host CPU staging buffers.  The cache is the single
    source of truth post-write -- no ``_invalidate_caches`` machinery.
  * ``*_mask`` setters / writers use the wheel's native
    ``binding.write(cache, mask=...)`` after running the matching mask kernel,
    avoiding the ``torch.nonzero`` round-trip.
  * Pinned-host staging on the read side too: CPU-only bindings (``MASS`` /
    ``COM_POSE`` / ``INERTIA``) are read into a lazily-allocated pinned-host
    :class:`wp.array` and ``wp.copy``-ed into the device cache, satisfying the
    wheel's device-match contract.
  * :attr:`~isaaclab_ovphysx.assets.RigidObjectData.root_link_vel_w` derives
    from the COM velocity via the lever-arm transform
    ``get_root_link_vel_from_root_com_vel``; ``root_com_vel_w`` reads the
    binding directly (standard PhysX convention).
  * :meth:`~isaaclab_ovphysx.assets.RigidObject.reset` matches PhysX/Newton:
    only resets the wrench composers; callers must explicitly call
    ``write_root_pose_to_sim_*`` / ``write_root_velocity_to_sim_*`` to restore
    the initial state.
  * :class:`~isaaclab_ovphysx.assets.RigidObjectData` section layout, public
    docstrings, ``Args:`` blocks, shape/dtype/SI-unit annotations, and naming
    (e.g. ``_write_root_state`` -> ``_write_body_state``) match PhysX.
  * Demoted ``device`` / ``num_instances`` / ``num_bodies`` from ``@property``
    accessors to plain instance attributes.
* Implemented seven deprecated state-concat properties
  (``default_root_state``, ``root_state_w``, ``root_link_state_w``,
  ``root_com_state_w``, ``body_state_w``, ``body_link_state_w``,
  ``body_com_state_w``) that were ``NotImplementedError`` stubs.  Each emits a
  ``DeprecationWarning`` and lazily populates a ``vec13f`` buffer via
  ``concat_root_pose_and_vel_to_state``.
* Implemented ``default_root_pose`` / ``default_root_vel`` properties that were
  ``NotImplementedError`` stubs; they now wrap the buffer populated from
  ``RigidObjectCfg.init_state``.
* Unified the CPU and GPU paths in
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager._configure_physx_scene_prim`:
  ``PhysxSceneAPI`` schema and ``enableSceneQuerySupport`` are applied on both
  CPU and GPU; the GPU-only attrs (``enableGPUDynamics``, ``broadphaseType``,
  ``gpu*`` capacity attrs from
  :class:`~isaaclab_ovphysx.physics.OvPhysxCfg`) remain gated on
  ``device == "gpu"``.
* Aligned ``test/assets/test_rigid_object.py`` 1-to-1 with
  :mod:`isaaclab_physx.test.assets.test_rigid_object` (same set of test
  functions, names, parametrizations, and assertions; same CPU + GPU coverage)
  and parameterised every test on ``device``.  Two session-scoped autouse
  fixtures (``_ovphysx_session_patches`` + ``_ovphysx_skip_other_device``)
  encapsulate the kitless invocation: a single :class:`ovphysx.PhysX` is shared
  across the pytest session and reused via ``physx.reset()`` /
  ``physx.add_usd()`` for subsequent tests, working around
  ``ovphysx<=0.3.7``'s dual-Carbonite static-destructor race and process-global
  device-mode lock.
* Added Google-style docstrings to every kernel and helper in
  :mod:`isaaclab_ovphysx.assets.kernels`.

Removed
^^^^^^^

* Removed :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_link_acc_w`.
  This OVPhysX-only convenience alias for
  :attr:`~isaaclab_ovphysx.assets.RigidObjectData.body_com_acc_w` was not
  present on the base contract or the PhysX/Newton backends.

Fixed
^^^^^

* Fixed :attr:`~isaaclab_ovphysx.assets.RigidObjectData.GRAVITY_VEC_W` returning
  ``(0, 0, -1)`` instead of ``(0, 0, 0)`` when ``cfg.gravity`` is the zero vector.
* Fixed a stale-buffer bug in
  :meth:`~isaaclab_ovphysx.assets.RigidObject._com_pose_to_link_pose` where the
  cached ``RIGID_BODY_COM_POSE`` body-frame offset was used after a write to the
  same binding, producing an incorrect frame conversion.  The buffer is now
  unconditionally refreshed at write time.
* Fixed :meth:`~isaaclab_ovphysx.assets.RigidObject._initialize_impl`:
  the ``hasattr(root_pose, "body_names")`` guard suppressed ``AttributeError``
  only but the real wheel raises ``TypeError`` on non-articulation tensor types,
  propagating instead of falling back to ``["base_link"]``.  Also fixed the
  silent ``"cuda:0"`` fallback when ``self._ovphysx.device`` did not exist; the
  device is now read from
  :meth:`~isaaclab_ovphysx.physics.OvPhysxManager.get_device`.
* Fixed shape-validation gaps in the index/mask write paths: full-write calls
  with too many rows now raise a clear ``RuntimeError`` instead of bubbling a
  binding ``ValueError`` or silently truncating; 1-D binding writes
  (e.g. ``RIGID_BODY_MASS``) normalise the source array to 1-D so the
  boolean-mask scatter receives a flat buffer.

0.1.2 (2026-04-23)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Wrapped :attr:`~isaaclab_ovphysx.assets.ArticulationData.GRAVITY_VEC_W` and
  :attr:`~isaaclab_ovphysx.assets.ArticulationData.FORWARD_VEC_B` in
  :class:`~isaaclab.utils.warp.ProxyArray` to match the PhysX and Newton
  backends. Public observations such as
  :func:`~isaaclab.envs.mdp.observations.projected_gravity` access
  ``asset.data.GRAVITY_VEC_W.torch``; the previous raw ``wp.array`` lacked
  ``.torch`` and raised ``AttributeError`` on the ovphysx backend.


0.1.1 (2026-04-21)
~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced private ``_find_names`` (fnmatch + regex) with the standard
  :func:`~isaaclab.utils.string.resolve_matching_names` for all finder
  methods, unifying name-resolution behavior across backends. Fnmatch-style
  glob patterns (e.g. ``joint_*``) are no longer supported; use regex
  equivalents (e.g. ``joint_.*``). ``find_fixed_tendons`` and
  ``find_spatial_tendons`` now raise ``ValueError`` on empty tendon lists,
  matching the PhysX backend.
* Changed ``find_joints`` ``joint_subset`` parameter from ``list[int]``
  (indices) to ``list[str]`` (names) to match the ``BaseArticulation``
  interface. Callers passing indices should convert to names first.


0.1.0 (2026-04-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Initial release of the ``isaaclab_ovphysx`` extension.
