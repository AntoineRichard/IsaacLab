# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for renderer gallery capture configuration."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import torch

from pxr import Gf, Usd, UsdGeom

MEDIA_TOOLS_DIR = Path(__file__).resolve().parents[1] / "docs" / "media"
sys.path.insert(0, str(MEDIA_TOOLS_DIR))

import capture_renderer_gallery  # noqa: E402
from capture_renderer_gallery import (  # noqa: E402
    add_gallery_arguments,
    capture_data_types,
    gallery_asset_name,
    gallery_modes,
    snapshot_camera_tensor,
)


def test_gallery_modes_cover_distinct_outputs_supported_by_every_renderer():
    common_modes = {
        "rgb",
        "albedo",
        "depth",
        "normals",
        "semantic_segmentation",
        "instance_segmentation",
    }

    for renderer in ("newton", "ovrtx", "isaac_rtx"):
        assert common_modes <= {mode.output_name for mode in gallery_modes(renderer)}


def test_gallery_modes_include_rtx_only_outputs_for_both_rtx_renderers():
    rtx_only_modes = {
        "motion_vectors",
        "simple_shading_constant_diffuse",
        "simple_shading_diffuse_mdl",
        "simple_shading_full_mdl",
    }

    assert rtx_only_modes.isdisjoint({mode.output_name for mode in gallery_modes("newton")})
    for renderer in ("ovrtx", "isaac_rtx"):
        assert rtx_only_modes <= {mode.output_name for mode in gallery_modes(renderer)}


def test_gallery_modes_omit_visually_redundant_output_aliases():
    redundant_modes = {
        "rgba",
        "rgb_hdr",
        "distance_to_camera",
        "distance_to_image_plane",
        "instance_id_segmentation_fast",
    }

    for renderer in ("newton", "ovrtx", "isaac_rtx"):
        assert redundant_modes.isdisjoint({mode.output_name for mode in gallery_modes(renderer)})


def test_only_rgb_is_animated():
    for renderer in ("newton", "ovrtx", "isaac_rtx"):
        animated_modes = [mode.output_name for mode in gallery_modes(renderer) if mode.animated]
        assert animated_modes == ["rgb"]

def test_standard_capture_groups_share_one_render_product():
    for renderer in ("newton", "ovrtx", "isaac_rtx"):
        expected = tuple(
            mode.output_name for mode in gallery_modes(renderer) if not mode.output_name.startswith("simple_")
        )
        assert capture_data_types(renderer, "standard") == expected


def test_simple_shading_capture_groups_are_isolated():
    simple_modes = (
        "simple_shading_constant_diffuse",
        "simple_shading_diffuse_mdl",
        "simple_shading_full_mdl",
    )

    for renderer in ("ovrtx", "isaac_rtx"):
        for mode in simple_modes:
            assert capture_data_types(renderer, mode) == (mode,)


def test_renderer_gallery_generator_keeps_kitless_and_kit_extras_separate():
    generator = (MEDIA_TOOLS_DIR / "generate_renderer_gallery.sh").read_text()

    assert "--extra ovrtx" in generator
    assert "--extra isaacsim" in generator
    assert "--extra isaacsim --extra ovrtx" not in generator
    assert "--extra ovrtx --extra isaacsim" not in generator


def test_gallery_asset_names_are_stable_and_renderer_specific():
    assert gallery_asset_name("isaac_rtx", "rgb") == "camera-renderer-isaac-rtx.webp"
    assert gallery_asset_name("ovrtx", "depth") == "camera-renderer-ovrtx-depth.png"


def test_gallery_renderer_argument_does_not_conflict_with_app_launcher_renderer():
    parser = argparse.ArgumentParser()
    add_gallery_arguments(parser)

    args = parser.parse_args(["--renderer-backend", "newton"])

    assert args.renderer_backend == "newton"
    assert args.physics_steps_per_frame == 3
    assert "--renderer" not in parser._option_string_actions


def test_gallery_scene_uses_rigid_spheres_for_cross_renderer_motion():
    stage = Usd.Stage.Open(str(MEDIA_TOOLS_DIR / "renderer_gallery_scene.usda"))
    spheres = [prim for prim in stage.Traverse() if prim.GetTypeName() == "Sphere"]

    assert len(spheres) == 6
    assert stage.GetPrimAtPath("/RendererGallery/PhysicsScene")
    for sphere in spheres:
        assert "PhysicsRigidBodyAPI" in sphere.GetAppliedSchemas()
        assert "PhysicsCollisionAPI" in sphere.GetAppliedSchemas()
        assert sphere.GetAttribute("xformOp:translate").GetNumTimeSamples() == 0


def test_gallery_spheres_author_mjwarp_bounce_contact_response():
    stage = Usd.Stage.Open(str(MEDIA_TOOLS_DIR / "renderer_gallery_scene.usda"))
    spheres = [prim for prim in stage.Traverse() if prim.GetTypeName() == "Sphere"]

    for sphere in spheres:
        assert sphere.GetAttribute("mjc:priority").Get() == 1
        assert tuple(sphere.GetAttribute("mjc:solref").Get()) == (0.02, 0.14)


def test_gallery_spheres_share_one_semantic_class_but_remain_distinct_from_the_scene():
    stage = Usd.Stage.Open(str(MEDIA_TOOLS_DIR / "renderer_gallery_scene.usda"))
    spheres = [prim for prim in stage.Traverse() if prim.GetTypeName() == "Sphere"]

    assert {tuple(sphere.GetAttribute("semantics:labels:class").Get()) for sphere in spheres} == {("sphere",)}
    assert tuple(stage.GetPrimAtPath("/RendererGallery/Table").GetAttribute("semantics:labels:class").Get()) == (
        "table",
    )
    assert tuple(stage.GetPrimAtPath("/RendererGallery/Backdrop").GetAttribute("semantics:labels:class").Get()) == (
        "backdrop",
    )



def test_gallery_table_and_backdrop_are_authored_size_texturable_meshes():
    stage = Usd.Stage.Open(str(MEDIA_TOOLS_DIR / "renderer_gallery_scene.usda"))
    expected_bounds = {
        "/RendererGallery/Table": (Gf.Vec3f(-20.0, -20.0, -0.25), Gf.Vec3f(20.0, 20.0, 0.25)),
        "/RendererGallery/Backdrop": (Gf.Vec3f(-20.0, -0.09, -6.0), Gf.Vec3f(20.0, 0.09, 6.0)),
    }

    for prim_path, (expected_min, expected_max) in expected_bounds.items():
        mesh = UsdGeom.Mesh.Get(stage, prim_path)
        assert mesh
        assert not mesh.GetPrim().HasAttribute("xformOp:scale")
        bounds = Gf.Range3f()
        for point in mesh.GetPointsAttr().Get():
            bounds.UnionWith(point)
        assert bounds.GetMin() == expected_min
        assert bounds.GetMax() == expected_max
        uv_primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar("st")
        assert uv_primvar
        assert uv_primvar.GetInterpolation() == UsdGeom.Tokens.faceVarying
        assert len(uv_primvar.ComputeFlattened()) == len(mesh.GetFaceVertexIndicesAttr().Get())

    table = stage.GetPrimAtPath("/RendererGallery/Table")
    assert "PhysicsCollisionAPI" in table.GetAppliedSchemas()


def test_gallery_camera_points_at_authored_target():
    stage = Usd.Stage.Open(str(MEDIA_TOOLS_DIR / "renderer_gallery_scene.usda"))
    camera = stage.GetPrimAtPath("/RendererGallery/Camera")
    target = stage.GetPrimAtPath("/RendererGallery/CameraTarget")

    assert target
    camera_transform = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    target_transform = UsdGeom.Xformable(target).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    camera_position = camera_transform.ExtractTranslation()
    target_position = target_transform.ExtractTranslation()
    camera_forward = camera_transform.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
    expected_forward = (target_position - camera_position).GetNormalized()

    assert Gf.Dot(camera_forward, expected_forward) > 0.999


def test_camera_tensor_snapshot_does_not_alias_mutable_renderer_buffer():
    renderer_buffer = torch.zeros((2, 2, 3), dtype=torch.uint8)

    snapshot = snapshot_camera_tensor(renderer_buffer)
    renderer_buffer.fill_(255)

    assert snapshot.data_ptr() != renderer_buffer.data_ptr()
    assert torch.count_nonzero(snapshot) == 0


def test_poster_frame_is_selected_after_two_thirds_of_the_fall():
    assert capture_renderer_gallery.poster_frame_index(37) == 24
    assert capture_renderer_gallery.poster_frame_index(2) == 1


def test_only_isaac_rtx_capture_launches_kit():
    assert capture_renderer_gallery.renderer_requires_kit("isaac_rtx")
    assert not capture_renderer_gallery.renderer_requires_kit("newton")
    assert not capture_renderer_gallery.renderer_requires_kit("ovrtx")


def test_gallery_uses_single_environment_paths_required_by_ovrtx():
    assert capture_renderer_gallery.gallery_stage_paths() == (
        "/World/envs/env_0/Scene",
        "/World/envs/env_.*/Scene/Camera",
    )

def test_ovrtx_gallery_disables_renderer_ambient_fill_without_changing_other_settings():
    render_product_usd = """\
float omni:rtx:rt:ambientLight:intensity = 1.0
float unrelated = 1.0
"""

    adjusted_usd = capture_renderer_gallery.override_ovrtx_ambient_light(render_product_usd)

    assert "float omni:rtx:rt:ambientLight:intensity = 0.0" in adjusted_usd
    assert "float unrelated = 1.0" in adjusted_usd

def test_renderer_lighting_override_covers_deferred_camera_initialization():
    events = []

    @contextlib.contextmanager
    def record_override(renderer):
        events.append(("enter", renderer))
        yield
        events.append(("exit", renderer))

    original_override = capture_renderer_gallery.gallery_lighting_override
    capture_renderer_gallery.gallery_lighting_override = record_override
    try:
        camera = capture_renderer_gallery._create_camera_and_reset(
            "ovrtx",
            lambda: events.append(("camera", "ovrtx")) or object(),
            lambda: events.append(("reset", "ovrtx")),
        )
    finally:
        capture_renderer_gallery.gallery_lighting_override = original_override

    assert camera is not None
    assert events == [
        ("enter", "ovrtx"),
        ("camera", "ovrtx"),
        ("reset", "ovrtx"),
        ("exit", "ovrtx"),
    ]
