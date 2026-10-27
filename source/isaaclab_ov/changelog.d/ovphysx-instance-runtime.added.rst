* Added explicit OvPhysX backend operations for stepping, forward updates, kinematics and
  gravity, with backend state owned by each instance and direct use without a simulation context.
  Preserved the manager API and documented the native runtime's single attached stage per process.
