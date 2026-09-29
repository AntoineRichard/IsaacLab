Changed
^^^^^^^

* Configured :data:`~isaaclab_assets.MICRODUCK_CFG` and its all-collisions and roller variants
  without a fixed solver joint ``friction`` on the BAM servo group. On MJWarp the Newton controller
  published its own live friction budget on every physics step, replacing the solver's initial
  seed. Joint viscous damping remained configured through
  :data:`~isaaclab_assets.MICRODUCK_JOINT_DAMPING`. These assets required Newton with
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` enabled.
