# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native contracts for explicitly owned OVPhysX backends without a simulation context."""

import sys

import pytest

from pxr import Usd

from isaaclab.test.utils import DeviceScope, test_devices

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
def test_explicit_backends_preserve_native_stage_ownership_and_live_gravity(device):
    """Explicit owners preserve motion, live gravity, and publications through rejection, reload, and close."""
    from isaaclab_ov.physics import OvPhysxBackendCfg
    from isaaclab_ov.physics.ovphysx_backend import OvPhysxBackend, OvPhysxSceneDataBackend, forward, set_gravity, step

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

    def height(publication):
        return float(publication.transforms.transforms.numpy()[0, 2])

    host_pxr = sys.modules["pxr"]
    publication = OvPhysxSceneDataBackend()
    first = OvPhysxBackend(OvPhysxBackendCfg(device=device), scene_data=publication)
    second = OvPhysxBackend(OvPhysxBackendCfg(device=device))
    try:
        assert sys.modules["pxr"] is host_pxr
        assert sys.modules["pxr.Usd"] is Usd
        first.load(stage_usda)
        assert publication.transform_count == 1
        assert publication.transform_paths == ["/World/Body"]
        for _ in range(10):
            step(first, 0.01)
        falling_height = height(publication)
        assert falling_height < 9.96

        with pytest.raises(RuntimeError, match="attached stage|stage attached"):
            step(second, 0.01)
        with pytest.raises(RuntimeError, match="another instance already owns the live PhysX attach"):
            second.load(stage_usda)
        with pytest.raises(RuntimeError, match="attached stage|stage attached"):
            step(second, 0.01)
        forward(first)
        assert height(publication) == falling_height
        second.close()
        for _ in range(10):
            step(first, 0.01)
        before_gravity_change = height(publication)
        assert before_gravity_change < falling_height - 0.1

        set_gravity(first, (0.0, 0.0, 30.0))
        assert first.gravity == (0.0, 0.0, 30.0)
        for _ in range(30):
            step(first, 0.01)
        assert height(publication) > before_gravity_change + 0.5

        first.load(stage_usda.replace("(0, 0, 10)", "(0, 0, 30)"))
        assert publication.transform_count == 1
        assert publication.transform_paths == ["/World/Body"]
        assert height(publication) == pytest.approx(30.0, abs=1.0e-4)
        first.close()
        assert publication.transform_count == 0

        second = OvPhysxBackend(OvPhysxBackendCfg(device=device), scene_data=publication)
        second.load(stage_usda)
        set_gravity(second, (0.0, 0.0, 0.0))
        assert second.gravity == (0.0, 0.0, 0.0)
        for _ in range(10):
            step(second, 0.01)
        assert height(publication) == pytest.approx(10.0, abs=1.0e-4)
    finally:
        second.close()
        first.close()
