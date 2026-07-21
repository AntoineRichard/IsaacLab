Added
^^^^^

* Added :attr:`~isaaclab_newton.physics.KaminoSolverCfg.dynamics_solver` to select
  compatible Newton constrained dynamics solvers while preserving the default.
* Added constrained-dynamics linear-solver and DVI tuning options to
  :class:`~isaaclab_newton.physics.KaminoSolverCfg`.

Fixed
^^^^^

* Fixed :attr:`~isaaclab_newton.physics.KaminoSolverCfg.dvi_warmstart_mode`
  to interpret Python ``None`` as the solver mode ``"none"``.
