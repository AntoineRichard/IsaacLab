Added
^^^^^

* Added :meth:`~isaaclab.actuators.newton.ControllerBam.bind_backlash_indices`, which lets the
  Newton-native BAM servo close its firmware position loop on the joint it drives *plus* a second,
  masked joint. This models a gearbox whose backlash is a hinge in series with the servo: the real
  servo's magnetic encoder sits on the output side of the play, so while the rotor winds through the
  dead zone the measured position, and hence the proportional error, does not move. The velocities
  stay motor-side, since in this model they drive only the back-EMF, the Stribeck blend and the
  stiction clip, which are rotor physics rather than an encoder-derived firmware signal. The
  binding is per DOF, so a DOF with no play hinge takes a zero mask and reproduces the plain servo
  bit for bit -- one configuration therefore covers plants with and without modelled play. The mask
  gates the read itself rather than weighting it by zero, so a masked-off DOF is unaffected by
  whatever the joint it nominally points at holds, non-finite values included. The
  indices are into the whole position array and so are a finalize-time property of the articulation
  rather than something an actuator prim can author; Isaac Lab's Newton backend resolves them from
  the joint names at articulation initialization. Controllers that are never bound are unaffected.
* Added :class:`~isaaclab.actuators.BamBacklashActuatorCfg`, the configuration that asks for that
  binding. It is :class:`~isaaclab.actuators.BamActuatorCfg` in every parameter and adds a naming
  contract: each driven joint's play hinge is ``passive_<joint>_backlash``, which the Newton backend
  pairs up and binds at articulation initialization. A driven joint whose plant carries no such
  hinge keeps the plain servo's behaviour bit for bit, so one configuration covers a robot whose
  joints are only partly modelled with play. The configuration runs on the Newton-native actuator
  path only and refuses to run anywhere else rather than degrading: its firmware loop reads a joint
  outside the actuator group, which the Isaac Lab-executed actuator loop is never handed, so a
  backend that steps native actuators through the host adapter (PhysX, OVPhysX) and a simulation
  configured with ``use_newton_actuators=False`` both raise with the one-line fix. Authoring marks
  the group on its ``NewtonActuator`` prims, which also keeps a plant with modelled gear play and
  one without in separate Newton actuators.
