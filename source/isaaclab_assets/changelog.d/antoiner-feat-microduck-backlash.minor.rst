Added
^^^^^

* Added the self-contained gear-backlash USD export, with 28 joints and 2 degrees of total
  play per servo. Each servo was connected in series with a passive play hinge through an
  intermediate body. The export retained the play hinges' limit solver parameters and damping;
  no conversion step was required when loading the asset.
* Added :data:`~isaaclab_assets.MICRODUCK_BACKLASH_CFG`, which spawns that asset as MicroDuck's
  fourth robot model. It is :data:`~isaaclab_assets.MICRODUCK_CFG` in every parameter, on a plant
  with 28 joints instead of 14, with two additions: the initial state gives the play hinges their
  centred rest angle, and the servo group is a
  :class:`~isaaclab.actuators.BamBacklashActuatorCfg`, so each servo's firmware closes its position
  loop on the encoder reading it would have on the real robot -- the motor angle plus the play. The
  action space is unchanged at 14. It is a separate configuration rather than an option on the
  shared one because Isaac Lab resolves an initial state strictly: a configuration that names the
  play hinges cannot spawn the three models that do not have them.
* Restricted that configuration to the Newton-native actuator path
  (``use_newton_actuators=True`` on MuJoCo Warp): it raises anywhere else rather than silently
  falling back to a servo model that cannot see the play.
* Added backlash cases to ``test/test_microduck_variant_assets.py``, comparing the new asset against
  its source MJCF: the 28-joint inventory that an unrepaired conversion fails at 14, the play range
  in both the degrees the USD authors it in and the radians the articulation reports, the play
  armature, one intermediate body per hinge and their mass and inertia, the limit solver properties
  and damping as authored and as they reach the built model, and that the servo joints still match
  the plain walking asset's limits, armature and effort. The configuration is covered on the
  Newton-native path at four environments: all 14 servos resolve the play hinge in series with them
  and none reads another environment's, the play hinges rest centred inside their range, and a hold
  at the home pose stays finite and comes to rest with 14 permanently active limit rows.
