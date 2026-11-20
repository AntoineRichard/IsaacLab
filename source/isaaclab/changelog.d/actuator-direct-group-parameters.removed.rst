* **Breaking:** Removed ``isaaclab.actuators.newton.adapter.LightArticulationView``, which borrowed private
  Newton selection internals to serve actuator parameter access on PhysX and OVPhysX. Return
  ``NewtonActuatorSelection(actuators=...)`` without a view instead.
