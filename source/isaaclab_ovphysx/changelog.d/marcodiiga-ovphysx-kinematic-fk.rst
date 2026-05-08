Fixed
^^^^^

* Fixed same-frame OVPhysX articulation link-pose reads after joint-position
  writes by requiring ``ovphysx>=0.4.2`` and calling
  ``PhysX.update_articulations_kinematic()`` before reading link poses.
