# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native contracts for explicitly owned OVPhysX backends without a simulation context."""

import numpy as np
import pytest

from isaaclab.test.utils import DeviceScope, test_devices

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_explicit_backends_preserve_native_stage_ownership_and_live_gravity(device):
    """Explicit owners preserve motion, live gravity, and publications through rejection, reload, and close."""
    from isaaclab_ov.physics import OvPhysxBackendCfg
    from isaaclab_ov.physics.ovphysx_backend import OvPhysxBackend, OvPhysxSceneDataBackend, set_gravity, step
    from ovphysx import TensorType

    stage_usda = """#usda 1.0
(defaultPrim = "World"
 metersPerUnit = 1
 upAxis = "Z")
def Xform "World" {
    def PhysicsScene "physicsScene" (prepend apiSchemas = ["PhysxSceneAPI"]) {
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 10
        bool physxScene:enableGPUDynamics = false
        token physxScene:broadphaseType = "MBP"
    }
    def Cube "Body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]) {
        double size = 0.1
        double3 xformOp:translate = (0, 0, 10)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        float physics:mass = 1
    }
}
"""
    publication = OvPhysxSceneDataBackend()
    first = OvPhysxBackend(OvPhysxBackendCfg(device=device), scene_data=publication)
    second = OvPhysxBackend(OvPhysxBackendCfg(device=device))
    first_pose = second_pose = None
    try:
        first.load(stage_usda)
        assert publication.transform_count == 1
        assert publication.transform_paths == ["/World/Body"]
        first_pose = first.physx.create_tensor_binding(pattern="/World/Body", tensor_type=TensorType.RIGID_BODY_POSE)
        first_state = np.empty(first_pose.shape, dtype=np.float32)
        first_pose.read(first_state)
        initial_height = float(first_state.reshape(-1, 7)[0, 2])
        for _ in range(10):
            step(first, 0.01)
        first_pose.read(first_state)
        falling_height = float(first_state.reshape(-1, 7)[0, 2])
        assert falling_height < initial_height - 0.04
        assert publication.transforms.transforms.numpy()[0, 2] == pytest.approx(falling_height)

        with pytest.raises(RuntimeError, match="attached stage|stage attached"):
            step(second, 0.01)
        with pytest.raises(RuntimeError, match="another instance already owns the live PhysX attach"):
            second.load(stage_usda)
        with pytest.raises(RuntimeError, match="attached stage|stage attached"):
            step(second, 0.01)
        first_pose.read(first_state)
        assert float(first_state.reshape(-1, 7)[0, 2]) == falling_height
        second.close()
        for _ in range(10):
            step(first, 0.01)
        first_pose.read(first_state)
        before_gravity_change = float(first_state.reshape(-1, 7)[0, 2])
        assert before_gravity_change < falling_height - 0.1

        set_gravity(first, (0.0, 0.0, 30.0))
        for _ in range(30):
            step(first, 0.01)
        first_pose.read(first_state)
        assert float(first_state.reshape(-1, 7)[0, 2]) > before_gravity_change + 0.5

        first_pose.destroy()
        first_pose = None
        first.load(stage_usda.replace("(0, 0, 10)", "(0, 0, 30)"))
        assert publication.transform_count == 1
        assert publication.transform_paths == ["/World/Body"]
        assert publication.transforms.transforms.numpy()[0, 2] == pytest.approx(30.0, abs=1.0e-4)
        first.close()
        assert publication.transform_count == 0

        second = OvPhysxBackend(OvPhysxBackendCfg(device=device))
        second.load(stage_usda)
        set_gravity(second, (0.0, 0.0, 0.0))
        second_pose = second.physx.create_tensor_binding(pattern="/World/Body", tensor_type=TensorType.RIGID_BODY_POSE)
        second_state = np.empty(second_pose.shape, dtype=np.float32)
        for _ in range(10):
            step(second, 0.01)
        second_pose.read(second_state)
        assert float(second_state.reshape(-1, 7)[0, 2]) == pytest.approx(initial_height, abs=1.0e-4)
    finally:
        if first_pose is not None:
            first_pose.destroy()
        if second_pose is not None:
            second_pose.destroy()
        second.close()
        first.close()
