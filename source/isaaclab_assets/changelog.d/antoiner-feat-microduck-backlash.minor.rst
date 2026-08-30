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
* Added backlash cases to ``test/test_microduck_variant_assets.py``, comparing the new asset against
  its source MJCF: the 28-joint inventory that an unrepaired conversion fails at 14, the play range
  in both the degrees the USD authors it in and the radians the articulation reports, the play
  armature, one intermediate body per hinge and their mass and inertia, and that the servo joints
  still match the plain walking asset's limits, armature and effort.
