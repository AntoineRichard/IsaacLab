# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ASCII terminal visualizer."""

from __future__ import annotations

import io
import re
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from isaaclab_visualizers.ascii import AsciiVisualizer, AsciiVisualizerCfg
from isaaclab_visualizers.ascii import ascii_visualizer as ascii_visualizer_module
from isaaclab_visualizers.ascii.ascii_renderer import AsciiMesh, AsciiRenderer, AsciiRenderInstance
from isaaclab_visualizers.ascii.usd_geometry import _build_edge_adjacency, extract_scene_geometry

from pxr import Usd, UsdGeom


class _TerminalBuffer(io.StringIO):
    """In-memory stream that behaves like an interactive terminal."""

    def isatty(self) -> bool:
        return True


class _FakeSceneDataProvider:
    """Minimal scene data provider for the ASCII visualizer."""

    def __init__(self, scene: Any, num_envs: int = 1) -> None:
        self._scene = scene
        self.num_envs = num_envs

    def get_interactive_scene(self) -> Any:
        return self._scene


def _make_scene() -> SimpleNamespace:
    articulation = SimpleNamespace(
        body_names=["base", "pole"],
        data=SimpleNamespace(
            body_pos_w=torch.tensor([[[10.0, 0.0, 0.0], [10.0, 0.0, 1.0]]], dtype=torch.float32),
            body_quat_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.3827, 0.9239]]]),
        ),
    )
    rigid_object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[11.0, 0.0, 0.0]], dtype=torch.float32),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        )
    )
    return SimpleNamespace(
        env_origins=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
        articulations={"robot": articulation},
        rigid_objects={"ball": rigid_object},
    )


def _make_box_mesh(size: tuple[float, float, float]) -> AsciiMesh:
    half_x, half_y, half_z = (dimension / 2.0 for dimension in size)
    vertices = (
        (-half_x, -half_y, -half_z),
        (half_x, -half_y, -half_z),
        (half_x, half_y, -half_z),
        (-half_x, half_y, -half_z),
        (-half_x, -half_y, half_z),
        (half_x, -half_y, half_z),
        (half_x, half_y, half_z),
        (-half_x, half_y, half_z),
    )
    faces = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    return AsciiMesh(vertices=vertices, faces=faces, edges=_build_edge_adjacency(faces))


def test_ascii_renderer_rasterizes_cartpole_geometry() -> None:
    """Filled meshes and silhouettes produce a recognizable cart and upright pole."""
    renderer = AsciiRenderer()
    cart = AsciiRenderInstance(_make_box_mesh((0.8, 0.4, 0.3)), (0.0, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0))
    pole = AsciiRenderInstance(_make_box_mesh((0.08, 0.08, 1.8)), (0.0, 0.0, 1.1), (0.0, 0.0, 0.0, 1.0))

    canvas = renderer.render(
        [cart, pole],
        width=60,
        height=24,
        eye=(4.0, -4.0, 3.0),
        lookat=(0.0, 0.0, 0.5),
        view_span=6.0,
        auto_fit=True,
        auto_fit_margin=1.2,
    )

    occupied_rows = [row for row, values in enumerate(canvas) if any(character != " " for character in values)]
    occupied_cells = sum(character != " " for row in canvas for character in row)
    assert len(occupied_rows) >= 12
    assert occupied_cells >= 80
    assert any(character not in {" ", "#"} for row in canvas for character in row)


def test_ascii_renderer_ignores_fixed_rail_when_auto_framing() -> None:
    """A long fixed rail remains visible without shrinking the dynamic bodies."""
    renderer = AsciiRenderer()
    rail = AsciiRenderInstance(
        _make_box_mesh((0.08, 8.0, 0.08)),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        include_in_auto_fit=False,
    )
    pole = AsciiRenderInstance(_make_box_mesh((0.08, 0.08, 1.0)), (0.0, 0.0, 0.6), (0.0, 0.0, 0.0, 1.0))

    canvas = renderer.render(
        [rail, pole],
        width=44,
        height=22,
        eye=(4.0, -4.0, 3.0),
        lookat=(0.0, 0.0, 0.0),
        view_span=6.0,
        auto_fit=True,
        auto_fit_margin=1.2,
    )

    occupied_rows = [row for row, values in enumerate(canvas) if any(character != " " for character in values)]
    assert len(occupied_rows) >= 12


ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CELL = re.compile(r"((?:\x1b\[[0-9;]*m)*)(.)")


def _covered_cells(row: str) -> int:
    """Cells of *row* carrying any color.

    Measured by color rather than by glyph on purpose. A fully covered cell encodes as a
    *space with a background color*, because the encoder prefers the fewest foreground
    subpixels when two splits tie, so counting non-space characters reads solid geometry as
    empty.
    """
    covered = lit = 0
    for style, _ in _CELL.findall(row):
        if style:
            lit = "38;2;" in style or "48;2;" in style
        covered += bool(lit)
    return covered


def _color_scene() -> list[AsciiRenderInstance]:
    """Two bodies of different colors, far enough apart to occupy different cells."""
    return [
        AsciiRenderInstance(
            _make_box_mesh((0.8, 0.4, 0.3)), (0.0, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0), True, (118, 185, 0)
        ),
        AsciiRenderInstance(
            _make_box_mesh((0.3, 0.3, 0.9)), (0.9, 0.0, 0.6), (0.0, 0.0, 0.0, 1.0), True, (77, 217, 232)
        ),
    ]


def test_color_rows_are_exactly_the_requested_size() -> None:
    """Escapes occupy no columns, so a row must measure its cell width once they are stripped.

    The panel draws a border either side of every row. A row that measures wrong because its
    color was counted as printable would push that border out of alignment.
    """
    rows = AsciiRenderer().render_color(
        _color_scene(),
        width=60,
        height=24,
        eye=(4.0, -4.0, 3.0),
        lookat=(0.0, 0.0, 0.5),
        view_span=6.0,
        auto_fit=True,
        auto_fit_margin=1.2,
    )

    assert len(rows) == 24
    assert {len(ANSI.sub("", row)) for row in rows} == {60}


def test_color_carries_each_body_separately() -> None:
    """Both body colors reach the output, rather than one overdrawing the other."""
    rows = AsciiRenderer().render_color(
        _color_scene(),
        width=60,
        height=24,
        eye=(4.0, -4.0, 3.0),
        lookat=(0.0, 0.0, 0.5),
        view_span=6.0,
        auto_fit=True,
        auto_fit_margin=1.2,
    )
    frame = "".join(rows)

    # shaded, so the exact triple varies by face; the hue each body is drawn in does not
    greens = sum(1 for match in re.finditer(r"2;(\d+);(\d+);(\d+)m", frame) if _is_green(match))
    cyans = sum(1 for match in re.finditer(r"2;(\d+);(\d+);(\d+)m", frame) if _is_cyan(match))
    assert greens > 0, "the green body did not reach the output"
    assert cyans > 0, "the cyan body did not reach the output"


def _is_green(match: re.Match[str]) -> bool:
    red, green, blue = (int(value) for value in match.groups())
    return green > red and green > blue


def _is_cyan(match: re.Match[str]) -> bool:
    red, green, blue = (int(value) for value in match.groups())
    return blue > red and green > red


def test_color_resolves_finer_detail_than_characters() -> None:
    """Quadrant subpixels resolve geometry that one character per cell cannot.

    The color path samples four times per cell, so a thin body that falls between character
    centres still lands on a subpixel. This is the whole reason for the second path.
    """
    thin = [
        AsciiRenderInstance(
            _make_box_mesh((0.02, 0.6, 0.6)), (0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0), True, (118, 185, 0)
        )
    ]
    arguments = dict(
        width=40,
        height=16,
        eye=(3.0, -3.0, 2.0),
        lookat=(0.0, 0.0, 0.3),
        view_span=6.0,
        auto_fit=True,
        auto_fit_margin=1.2,
    )

    characters = AsciiRenderer().render(thin, **arguments)
    rows = AsciiRenderer().render_color(thin, **arguments)

    character_cells = sum(character != " " for row in characters for character in row)
    color_cells = sum(_covered_cells(row) for row in rows)
    assert color_cells >= character_cells


def test_color_renders_nothing_for_an_empty_scene() -> None:
    """No geometry yields blank rows rather than a short frame the panel cannot border."""
    rows = AsciiRenderer().render_color(
        [],
        width=30,
        height=10,
        eye=(1.0, 0.0, 0.0),
        lookat=(0.0, 0.0, 0.0),
        view_span=2.0,
        auto_fit=True,
        auto_fit_margin=1.2,
    )
    assert rows == [""] * 10


def test_body_colors_are_stable_as_the_scene_grows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A body keeps its color when another body appears, so nothing changes hue mid-run."""
    visualizer = AsciiVisualizer(AsciiVisualizerCfg(color=True))
    first = visualizer._body_color(("robot", "base"))
    visualizer._body_color(("robot", "thigh"))
    assert visualizer._body_color(("robot", "base")) == first


def test_ascii_geometry_extractor_preserves_body_dimensions() -> None:
    """USD primitive transforms are cached in their owning body frames."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    cart = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/cart")
    pole = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/pole")
    cart_geometry = UsdGeom.Cube.Define(stage, f"{cart.GetPath()}/geometry")
    cart_geometry.GetSizeAttr().Set(1.0)
    cart_geometry.AddScaleOp().Set((0.8, 0.4, 0.3))
    pole_geometry = UsdGeom.Cube.Define(stage, f"{pole.GetPath()}/geometry")
    pole_geometry.GetSizeAttr().Set(1.0)
    pole_geometry.AddScaleOp().Set((0.08, 0.08, 1.8))
    articulation = SimpleNamespace(
        body_names=["cart", "pole"],
        backend_body_names=["cart", "pole"],
        root_view=SimpleNamespace(link_paths=[["/World/envs/env_0/Robot/cart", "/World/envs/env_0/Robot/pole"]]),
    )
    scene = SimpleNamespace(articulations={"robot": articulation}, rigid_objects={})

    geometry = extract_scene_geometry(scene, stage, env_index=0, max_faces_per_body=96)

    cart_vertices = torch.tensor(geometry[("robot", "cart")].vertices)
    pole_vertices = torch.tensor(geometry[("robot", "pole")].vertices)
    torch.testing.assert_close(
        cart_vertices.max(dim=0).values - cart_vertices.min(dim=0).values, torch.tensor([0.8, 0.4, 0.3])
    )
    torch.testing.assert_close(
        pole_vertices.max(dim=0).values - pole_vertices.min(dim=0).values, torch.tensor([0.08, 0.08, 1.8])
    )


def test_ascii_visualizer_renders_dynamic_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frame contains articulation links, rigid objects, and run status."""
    monkeypatch.setattr(
        ascii_visualizer_module.shutil,
        "get_terminal_size",
        lambda fallback: SimpleNamespace(columns=60, lines=18),
    )
    visualizer = AsciiVisualizer(AsciiVisualizerCfg(width=60, height=18))
    visualizer._stream = _TerminalBuffer()
    visualizer.initialize(_FakeSceneDataProvider(_make_scene()))

    visualizer.refresh_scene_state(0.25)

    output = visualizer._stream.getvalue()
    assert "ISAAC LAB ASCII VISUALIZER" in output
    assert "bodies=3" in output
    assert "robot, ball" in output
    assert "@" in output
    assert "P" in output
    assert "#" in output
    assert any(character in output for character in "-:|")
    visualizer.close()


def test_ascii_visualizer_renders_orientation_for_coincident_body_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Body axes remain visible when multiple link origins project to one cell."""
    scene = _make_scene()
    scene.articulations["robot"].data.body_pos_w[:] = torch.tensor([10.0, 0.0, 0.0])
    monkeypatch.setattr(
        ascii_visualizer_module.shutil,
        "get_terminal_size",
        lambda fallback: SimpleNamespace(columns=80, lines=24),
    )
    visualizer = AsciiVisualizer(AsciiVisualizerCfg(width=80, height=24, body_axis_length=2.0))
    visualizer._stream = _TerminalBuffer()
    visualizer.initialize(_FakeSceneDataProvider(scene))

    visualizer.refresh_scene_state(0.25)
    frame = visualizer._render_frame()

    assert "P" in frame
    assert frame.count("|") > 2
    assert "-" in frame
    visualizer.close()


def test_ascii_visualizer_captures_training_output_in_side_pane(monkeypatch: pytest.MonkeyPatch) -> None:
    """Captured stdout and stderr are rendered without escaping the dashboard."""
    monkeypatch.setattr(
        ascii_visualizer_module.shutil,
        "get_terminal_size",
        lambda fallback: SimpleNamespace(columns=80, lines=20),
    )
    visualizer = AsciiVisualizer(AsciiVisualizerCfg(width=80, height=20))
    visualizer._stream = _TerminalBuffer()
    visualizer.initialize(_FakeSceneDataProvider(_make_scene()))
    previous_stdout = sys.stdout
    previous_stderr = sys.stderr

    with visualizer.capture_console_output():
        print("\x1b[32mLearning iteration 4/10\x1b[0m")
        sys.stderr.write("                                        mean reward: 12.5\n")

    output = visualizer._stream.getvalue()
    assert sys.stdout is previous_stdout
    assert sys.stderr is previous_stderr
    assert "RSL-RL" in output
    assert "Learning iteration 4/10" in output
    assert "mean reward: 12.5" in output
    assert "\x1b[32m" not in output
    visualizer.close()


def test_ascii_visualizer_restores_terminal_on_close() -> None:
    """Closing the visualizer restores the cursor and stops it."""
    visualizer = AsciiVisualizer(AsciiVisualizerCfg())
    visualizer._stream = _TerminalBuffer()
    visualizer.initialize(_FakeSceneDataProvider(_make_scene()))

    visualizer.close()

    assert visualizer._stream.getvalue().endswith("\x1b[?25h\n")
    assert not visualizer.is_running()


def test_ascii_visualizer_rejects_out_of_range_environment() -> None:
    """The configured environment must exist when the scene reports its size."""
    visualizer = AsciiVisualizer(AsciiVisualizerCfg(env_index=1))
    with pytest.raises(ValueError, match="must be smaller than the number of environments"):
        visualizer.initialize(_FakeSceneDataProvider(_make_scene(), num_envs=1))


def test_ascii_visualizer_filters_non_finite_positions() -> None:
    """Invalid simulation positions do not terminate terminal rendering."""
    values = torch.tensor([[[1.0, 2.0, 3.0], [torch.nan, 0.0, 0.0], [torch.inf, 0.0, 0.0]]])

    assert AsciiVisualizer._read_positions(values, 0, (0.0, 0.0, 0.0)) == [(1.0, 2.0, 3.0)]


def test_ascii_visualizer_bounds_far_offscreen_points() -> None:
    """Projection bounds the amount of line work for distant bodies."""
    visualizer = AsciiVisualizer(AsciiVisualizerCfg())

    points = visualizer._project_points([(1.0e9, -1.0e9, 1.0e9)], width=40, height=20)

    assert -40 <= points[0][0] < 80
    assert -20 <= points[0][1] < 40


def test_ascii_visualizer_stays_silent_for_redirected_output() -> None:
    """Non-interactive output does not receive ANSI frames or cursor controls."""
    visualizer = AsciiVisualizer(AsciiVisualizerCfg())
    visualizer._stream = io.StringIO()
    visualizer.initialize(_FakeSceneDataProvider(_make_scene()))

    visualizer.step(0.25)
    visualizer.close()

    assert visualizer._stream.getvalue() == ""


def test_ascii_visualizer_cfg_creates_registered_backend() -> None:
    """The shared visualizer factory resolves the ASCII backend."""
    cfg = AsciiVisualizerCfg()
    visualizer = cfg.create_visualizer()

    assert isinstance(visualizer, AsciiVisualizer)
    assert not visualizer.requires_simulation_rendering()
    assert cfg.auto_fit_margin == 1.4


def test_ascii_visualizer_uses_original_stdout_during_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rich-style stdout redirection does not hide the attached terminal."""
    terminal = _TerminalBuffer()
    monkeypatch.setattr(sys, "__stdout__", terminal)
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    assert AsciiVisualizer(AsciiVisualizerCfg())._stream is terminal


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"width": 19}, "width must be at least 20"),
        ({"height": 7}, "height must be at least 8"),
        ({"max_fps": 0.0}, "max_fps must be positive"),
        ({"env_index": -1}, "env_index must be non-negative"),
        ({"view_span": 0.0}, "view_span must be positive"),
        ({"auto_fit_margin": 1.0}, "auto_fit_margin must be greater than 1"),
        ({"max_faces_per_body": 11}, "max_faces_per_body must be at least 12"),
        ({"body_axis_length": 0.0}, "body_axis_length must be positive"),
    ],
)
def test_ascii_visualizer_cfg_rejects_invalid_values(kwargs: dict[str, Any], match: str) -> None:
    """Invalid terminal and view settings fail during config construction."""
    with pytest.raises(ValueError, match=match):
        AsciiVisualizerCfg(**kwargs)
