# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Geometry contracts needed for a stocking that can actually contain rigid candies."""

import numpy as np
import pytest
import trimesh

from pxr import Usd, UsdGeom

from examples.christmas_atelier.assets import candy_mesh, export_assets, panel_stocking_mesh, stocking_mesh

pytestmark = [pytest.mark.kitless, pytest.mark.unit]


@pytest.mark.parametrize("make_stocking", [stocking_mesh, panel_stocking_mesh])
def test_stocking_has_only_one_open_mouth(make_stocking):
    """A missing toe cap or disconnected seam would let candy escape."""
    vertices, faces, colors, cuff = make_stocking()
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert len(mesh.split(only_watertight=False)) == 1
    assert mesh.is_winding_consistent
    assert np.min(mesh.area_faces) > 1.0e-7
    edges, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    assert np.all(counts <= 2)
    boundary = edges[counts == 1]
    assert set(boundary.ravel()) == set(cuff)
    assert len(boundary) == len(cuff)
    assert np.allclose(vertices[cuff, 2], 0.0)
    assert np.ptp(vertices[:, 2]) > 0.3
    assert colors.shape == (len(faces), 3)


def test_candy_and_usd_export(tmp_path):
    """Exported assets must preserve geometry, colors, scale and an uncapped mouth."""
    vertices, faces, _ = candy_mesh()
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0
    assert np.min(mesh.area_faces) > 1.0e-9
    export_assets(tmp_path)
    stage = Usd.Stage.Open(str(tmp_path / "stocking.usda"))
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    exported = UsdGeom.Mesh(stage.GetPrimAtPath("/Stocking"))
    expected_vertices, expected_faces, _, _ = stocking_mesh()
    np.testing.assert_allclose(exported.GetPointsAttr().Get(), expected_vertices)
    np.testing.assert_array_equal(exported.GetFaceVertexIndicesAttr().Get(), expected_faces.ravel())
    assert len(exported.GetDisplayColorPrimvar().Get()) == len(expected_faces)
