Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.KaminoSolverCfg.dynamics_solver` to select
  compatible Newton constrained dynamics solvers while preserving the default.
* Added constrained-dynamics linear-solver and DVI tuning options to
  :class:`~isaaclab_newton.physics.KaminoSolverCfg`.


Changed
^^^^^^^

* Changed Kamino DVI tuning manifests to schema 1.2 and adaptive decisions to
  schema 1.1 to record the imported ``isaaclab_newton`` package location and
  strictly recompute decisions before adaptive launches. Existing schema 1.1
  tuning manifests cannot be resumed; start a new tuning artifact root.

Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.physics.KaminoSolverCfg.dvi_warmstart_mode`
  to interpret Python ``None`` as the solver mode ``"none"``.
* Fixed Kamino benchmark execution to require observed package provenance and
  validate it against the launched checkout, and fixed report funnels so
  qualified tie-break finalists are not counted as learning rejections.
* Fixed generic benchmark resume to preserve unreadable or
  provenance-incompatible manifests and logs, requiring a new artifact root
  instead of overwriting existing evidence.
