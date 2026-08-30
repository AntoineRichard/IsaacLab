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
  bit for bit -- one configuration therefore covers plants with and without modelled play. The
  indices are into the whole position array and so are a finalize-time property of the articulation
  rather than something an actuator prim can author; Isaac Lab's Newton backend resolves them from
  the joint names at articulation initialization. Controllers that are never bound are unaffected.
