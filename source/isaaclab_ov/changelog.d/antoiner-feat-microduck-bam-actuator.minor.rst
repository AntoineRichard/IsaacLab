Added
^^^^^

* Native actuator execution now rejects a :class:`~isaaclab.actuators.BamActuatorCfg` group with a
  message naming the supported configuration. The BAM servo model's native form is written in terms
  of solver quantities -- it publishes its gearbox friction budget into the solver's joint dry
  friction and reads the external load back out of the solver's generalized forces -- and OVPhysX
  steps native actuators through the shared host adapter, which provides neither. Running BAM on
  this backend is supported through the Isaac Lab-executed model: leave
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` unset (or ``False``). Every other
  supported actuator configuration is unaffected.
