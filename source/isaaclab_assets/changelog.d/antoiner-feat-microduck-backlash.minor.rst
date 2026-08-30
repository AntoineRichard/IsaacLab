Added
^^^^^

* Added upstream's gear-backlash MicroDuck model to the conversion tooling, selected with
  ``scripts/tools/convert_microduck.py --model walk_backlash``. It is the walking model with 2
  degrees of total play per servo, and converting it needs one repair the other models do not: each
  ``passive_<servo>_backlash`` hinge shares a MJCF body with its servo joint, which UsdPhysics
  cannot express, so the importer collapses the pair into a single D6 and discards the play axis --
  producing an asset that silently *is* the plain walking model. The conversion rebuilds each pair
  as ``parent -> servo hinge -> dummy body -> play hinge -> child``, which restores all 14 play DOFs
  at upstream's plus or minus 1 degree of range. The play hinges are interleaved with the servos in
  the built articulation's joint order, so a consumer that indexes joints positionally rather than
  by name has to account for them.
* Added the play hinges' MuJoCo solver properties to that conversion: each one now carries
  upstream's limit ``solreflimit`` (``0.01 1``) and ``solimplimit`` (``0.95 0.999 0.0001 0.5 2``)
  and its ``damping`` (``0.01``), authored per joint prim through
  :class:`~isaaclab_newton.sim.schemas.MujocoJointCfg`. The play range *is* the gear teeth, so the
  constraint that models it is part of the plant: at MuJoCo's default limit reference the hinges
  ride roughly twice their declared play under load. Nothing else could supply the damping, since
  no actuator group owns these joints. Regenerate the asset to pick this up.
* Added backlash cases to ``test/test_microduck_variant_assets.py``, comparing the new asset against
  its source MJCF: the 28-joint inventory that an unrepaired conversion fails at 14, the play range
  in both the degrees the USD authors it in and the radians the articulation reports, the play
  armature, one intermediate body per hinge and their mass and inertia, the limit solver properties
  and damping as authored and as they reach the built model, and that the servo joints still match
  the plain walking asset's limits, armature and effort.
