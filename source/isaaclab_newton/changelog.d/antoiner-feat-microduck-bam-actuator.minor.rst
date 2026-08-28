Added
^^^^^

* Added :class:`~isaaclab_newton.physics.MjWarpActuatorBridge`, the single place Isaac Lab
  touches MuJoCo Warp's device model on behalf of a Newton actuator component. It publishes a
  component's per-step dry-friction budget and viscous damping into ``dof_frictionloss`` /
  ``dof_damping``, reads back the true external load on the driven DOFs
  (``-qfrc_bias + qfrc_constraint`` with the component's own friction rows removed), and can
  stiffen the friction constraint's solver reference. The Newton articulation binds it to every
  :class:`~isaaclab.actuators.BamActuatorCfg` group on the Newton-native actuator path. Writes
  go straight into the MuJoCo Warp model and are therefore not visible through Isaac Lab's
  joint-friction property; the module documents the resulting ordering contract.
* Added :meth:`~isaaclab_newton.physics.NewtonManager.register_pre_actuator_callback`, an
  in-graph hook that runs immediately before the actuator step so a component can consume
  solver quantities on the same decimation iteration, and
  :meth:`~isaaclab_newton.physics.NewtonManager.register_solver_init_callback`, a one-shot hook
  that runs once the solver exists and before any CUDA graph capture. Assets initialize while
  the model is still being built, so anything that needs the concrete solver has to defer to
  the latter.

Changed
^^^^^^^

* **Breaking:** Joint DOFs that do not author their own limit gains now use MuJoCo's default
  joint-limit ``solref`` (``(0.02, 1.0)``, critically damped) on the MuJoCo Warp backend instead
  of the force-space conversion of Newton's generic ``limit_ke=1e4`` / ``limit_kd=1e1`` defaults.
  That conversion scaled the gains by ``dof_invweight0`` and produced a stiffer, roughly four
  times underdamped pair (``~(0.008, 0.25)`` for a light servo robot), so joint limits rebounded
  instead of absorbing and light limit-bounded robots could diverge to non-finite state at
  realistic joint damping. Joints that author limit gains (USD limit stiffness/damping, MJCF
  ``solreflimit``, or explicit force-space gains) are unaffected. Limit dynamics change for
  every other Newton asset, so re-validate policies trained before this change; set
  :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_default_joint_limit_solref` to
  ``False`` to restore the previous conversion.

Fixed
^^^^^

* CUDA graph capture with stateful Newton actuators now raises at a decimation of one and warns
  at any other odd decimation. Actuator state is double buffered and the buffers are swapped
  host-side, so a replayed graph always restarts from the buffer that was current at capture
  time: with an odd number of actuator steps per graph the last state update is discarded on
  every replay, and with a decimation of one delay buffers, integral terms and other actuator
  state never advance at all -- silently wrong physics rather than a small staleness. Use an
  even decimation or disable ``use_cuda_graph`` until the capture strategy alternates graphs the
  way the PhysX host runtime already does.
