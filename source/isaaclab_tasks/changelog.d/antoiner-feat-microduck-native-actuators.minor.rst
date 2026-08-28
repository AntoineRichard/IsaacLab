Changed
^^^^^^^

* Changed the MicroDuck tasks -- ``IsaacContrib-Velocity-Flat-MicroDuck``,
  ``IsaacContrib-Velocity-Rough-MicroDuck`` and ``IsaacContrib-StandUp-Flat-MicroDuck`` -- to execute
  their BAM servos on the backend-native path by default, by setting
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` to ``True``. That is what the upstream
  reference does: the controller runs inside the MuJoCo Warp step and republishes its live friction
  budget and viscous coefficient into the solver's ``dof_frictionloss`` and ``dof_damping`` every
  physics tick, instead of clipping the torque once per control step from Isaac Lab. The two are
  different plants, so **policies trained on these tasks before this change should be retrained**;
  ``env.sim.use_newton_actuators = False`` restores the previous, Isaac Lab-executed behaviour.
  Their ``decimation`` of 4 is a precondition of the switch rather than only a control rate: the BAM
  command delay is actuator state, and the Newton backend refuses to CUDA-graph-capture stateful
  native actuators at a decimation of one and warns at an odd one.
* Changed the MicroDuck physics presets to document that MuJoCo Warp is now the only backend that
  runs these tasks as configured. The BAM model is solver-hosted -- it publishes its friction budget
  into the solver's joint dry friction and reads the external load back out -- so a backend that
  steps native actuators through the shared host adapter (PhysX, OVPhysX) rejects the configuration.
  A PhysX preset would have to force ``use_newton_actuators = False`` and accept the Isaac
  Lab-executed model, which is the documented fallback but a different plant; none is offered.
