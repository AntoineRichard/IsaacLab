Changed
^^^^^^^

* Configured the MicroDuck tasks to execute their BAM servos on Newton / MJWarp with
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` enabled. The controller published its
  live friction budget and viscous coefficient into the solver on every physics step. The tasks
  required this native controller; disabling native actuators or selecting PhysX / OVPhysX was
  unsupported. Policies trained with a different actuator model required retraining.
