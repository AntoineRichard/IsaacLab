# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render Isaac Lab geometry in an ASCII terminal dashboard."""

from __future__ import annotations

import atexit
import contextlib
import math
import os
import re
import shutil
import sys
import time
from collections import deque
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TextIO

from isaaclab.visualizers import BaseVisualizer

from .ascii_renderer import AsciiMesh, AsciiRenderer, AsciiRenderInstance
from .ascii_visualizer_cfg import AsciiVisualizerCfg
from .usd_geometry import extract_scene_geometry

if TYPE_CHECKING:
    from isaaclab.scene_data import SceneDataProvider


_Vector3 = tuple[float, float, float]
_Quaternion = tuple[float, float, float, float]
_BodyPose = tuple[_Vector3, _Quaternion, str]
_BodyGroup = tuple[str, list[_BodyPose], bool]
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")


class AsciiVisualizer(BaseVisualizer):
    """Terminal visualizer that projects one environment's dynamic bodies to ASCII."""

    def __init__(self, cfg: AsciiVisualizerCfg):
        """Initialize the ASCII visualizer.

        Args:
            cfg: ASCII visualizer configuration.
        """
        super().__init__(cfg)
        self.cfg: AsciiVisualizerCfg = cfg
        self._stream: TextIO = sys.__stdout__ or sys.stdout
        self._owns_stream = False
        self._terminal_active = False
        self._sim_time = 0.0
        self._last_render_time = -math.inf
        self._eye = tuple(float(value) for value in cfg.eye)
        self._lookat = tuple(float(value) for value in cfg.lookat)
        self._console_lines: deque[str] = deque(maxlen=200)
        self._console_buffer = ""
        self._console_panel_enabled = False
        self._console_title = "OUTPUT"
        self._renderer = AsciiRenderer()
        self._geometry: dict[tuple[str, str], AsciiMesh] = {}
        self._auto_fit_exclusions: set[tuple[str, str]] = set()
        self._body_groups: list[_BodyGroup] = []

    def initialize(self, scene_data_provider: SceneDataProvider) -> None:
        """Bind the scene and prepare the terminal.

        Args:
            scene_data_provider: Scene data provider containing the interactive scene.
        """
        if self._is_initialized:
            return
        self._set_scene_data_provider(scene_data_provider)
        scene = scene_data_provider.get_interactive_scene()
        num_envs = getattr(scene, "num_envs", scene_data_provider.num_envs)
        if num_envs > 0 and self.cfg.env_index >= num_envs:
            raise ValueError(
                f"AsciiVisualizerCfg.env_index ({self.cfg.env_index}) must be smaller than the number of "
                f"environments ({num_envs})."
            )
        self._env_ids = [self.cfg.env_index]
        get_usd_stage = getattr(scene_data_provider, "get_usd_stage", None)
        stage = get_usd_stage() if callable(get_usd_stage) else None
        if scene is not None and stage is not None:
            try:
                self._geometry = extract_scene_geometry(
                    scene,
                    stage,
                    self.cfg.env_index,
                    self.cfg.max_faces_per_body,
                )
            except (RuntimeError, TypeError, ValueError):
                self._geometry = {}
        for asset_name, articulation in getattr(scene, "articulations", {}).items():
            body_names = list(getattr(articulation, "body_names", []))
            if body_names and bool(getattr(articulation, "is_fixed_base", False)):
                self._auto_fit_exclusions.add((asset_name, body_names[0]))
        self._stream, self._owns_stream = self._resolve_terminal_stream(self._stream)
        self._terminal_active = bool(getattr(self._stream, "isatty", lambda: False)())
        if self._terminal_active:
            self._stream.write("\x1b[2J\x1b[H\x1b[?25l")
            self._stream.flush()
        self._is_initialized = True
        atexit.register(self.close)

    def step(self, dt: float) -> None:
        """Accept a simulation render-loop update without sampling physics state.

        RSL-RL drives rendering through :meth:`refresh_scene_state` after complete
        environment steps. This callback intentionally does no work when another
        visualizer keeps the simulation render loop active.

        Args:
            dt: Simulation time step in seconds.
        """
        pass

    def requires_simulation_rendering(self) -> bool:
        """Return whether the simulation render loop must drive this visualizer."""
        return False

    def refresh_scene_state(self, dt: float = 0.0) -> None:
        """Cache and render the latest completed environment state.

        Args:
            dt: Elapsed simulation time in seconds.
        """
        if self._is_initialized and not self._is_closed and self._terminal_active:
            self._sim_time += dt
            self._body_groups = self._collect_body_poses()
            self._render_to_terminal()

    @contextlib.contextmanager
    def capture_console_output(self, title: str = "RSL-RL") -> Iterator[None]:
        """Display stdout and stderr in a log pane beside the simulation.

        Output is redirected only when the visualizer owns an interactive terminal.
        Redirected output is restored when the context exits.

        Args:
            title: Title displayed above the captured log pane.

        Yields:
            Control to the caller while console output is captured.
        """
        if not self._terminal_active or self._is_closed:
            yield
            return

        capture = _ConsoleCapture(self)
        previous_stdout = sys.stdout
        previous_stderr = sys.stderr
        self._console_panel_enabled = True
        self._console_title = title
        sys.stdout = capture
        sys.stderr = capture
        try:
            yield
        finally:
            try:
                capture.flush()
            finally:
                sys.stdout = previous_stdout
                sys.stderr = previous_stderr
            self._render_to_terminal(force=True)

    def close(self) -> None:
        """Restore terminal state and stop rendering."""
        if self._is_closed:
            return
        if self._terminal_active:
            self._stream.write("\x1b[?25h\n")
            self._stream.flush()
        self._terminal_active = False
        if self._owns_stream:
            self._stream.close()
            self._owns_stream = False
        self._is_closed = True

    def is_running(self) -> bool:
        """Return whether the visualizer remains active.

        Returns:
            True after initialization until close is called.
        """
        return self._is_initialized and not self._is_closed

    def set_camera_view(self, eye: tuple, target: tuple) -> None:
        """Set the orthographic projection orientation.

        Args:
            eye: Camera eye position.
            target: Camera look-at position.
        """
        self._eye = tuple(float(value) for value in eye)
        self._lookat = tuple(float(value) for value in target)

    def _render_to_terminal(self, force: bool = False) -> None:
        """Render a frame when the terminal refresh interval has elapsed."""
        if not self._terminal_active:
            return
        now = time.monotonic()
        if not force and now - self._last_render_time < 1.0 / self.cfg.max_fps:
            return
        self._last_render_time = now
        frame = self._render_frame()
        self._stream.write(f"\x1b[2J\x1b[H{frame}")
        self._stream.flush()

    def _render_frame(self) -> str:
        """Build one complete terminal frame."""
        try:
            terminal_size = os.get_terminal_size(self._stream.fileno())
        except (AttributeError, OSError, ValueError):
            terminal_size = shutil.get_terminal_size(fallback=(self.cfg.width, self.cfg.height))
        width = max(20, min(self.cfg.width, terminal_size.columns))
        height = max(8, min(self.cfg.height, terminal_size.lines))

        if not self._console_panel_enabled:
            return "\n".join(self._render_scene_panel(width, height))
        if width >= 80:
            scene_width = max(44, round(width * 0.55))
            console_width = width - scene_width - 1
            scene_lines = self._render_scene_panel(scene_width, height)
            console_lines = self._render_console_panel(console_width, height)
            return "\n".join(
                f"{scene_line} {console_line}" for scene_line, console_line in zip(scene_lines, console_lines)
            )

        scene_height = max(5, round(height * 0.62))
        console_height = height - scene_height
        return "\n".join(
            [*self._render_scene_panel(width, scene_height), *self._render_console_panel(width, console_height)]
        )

    def _render_scene_panel(self, width: int, height: int) -> list[str]:
        """Build the simulation pane at an exact character size."""
        plot_width = width - 2
        plot_height = max(1, height - 2)
        groups = self._body_groups
        body_count = sum(len(poses) for _, poses, _ in groups)
        instances = self._create_render_instances(groups)
        if instances:
            canvas = self._renderer.render(
                instances,
                plot_width,
                plot_height,
                self._eye,
                self._lookat,
                self.cfg.view_span,
                self.cfg.auto_fit,
                self.cfg.auto_fit_margin,
            )
        else:
            canvas = self._render_pose_fallback(groups, plot_width, plot_height)

        title = self._fit_text(" ISAAC LAB ASCII VISUALIZER ", width - 2, fill="-")
        names = ", ".join(name for name, _, _ in groups) or "waiting for dynamic scene data"
        triangle_count = sum(len(instance.mesh.faces) for instance in instances)
        geometry_status = (
            f"geometry={len(instances)}/{body_count} | tris={triangle_count}"
            if instances
            else f"pose fallback | bodies={body_count}"
        )
        status = f" env {self.cfg.env_index} | t={self._sim_time:.2f}s | {geometry_status} | {names} "
        status = self._fit_text(status, width - 2, fill="-")
        return [f"+{title}+", *(f"|{''.join(row)}|" for row in canvas), f"+{status}+"]

    def _create_render_instances(self, groups: list[_BodyGroup]) -> list[AsciiRenderInstance]:
        """Pair cached geometry with live body poses."""
        instances = []
        for asset_name, poses, is_rigid in groups:
            for position, orientation, body_name in poses:
                geometry_name = asset_name if is_rigid else body_name
                geometry_key = (asset_name, geometry_name)
                mesh = self._geometry.get(geometry_key)
                if mesh is not None:
                    instances.append(
                        AsciiRenderInstance(
                            mesh,
                            position,
                            orientation,
                            include_in_auto_fit=geometry_key not in self._auto_fit_exclusions,
                        )
                    )
        return instances

    def _render_pose_fallback(self, groups: list[_BodyGroup], width: int, height: int) -> list[list[str]]:
        """Render orientation glyphs when no USD geometry is available."""
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        projected_groups = [
            (self._project_points([pose[0] for pose in poses], width, height), poses, is_rigid)
            for _, poses, is_rigid in groups
        ]
        origin = self._project_points([(0.0, 0.0, 0.0)], width, height)
        if origin:
            self._set_cell(canvas, origin[0], "+")
        for points, poses, is_rigid in projected_groups:
            for pose in poses:
                self._draw_body_axes(canvas, pose, width, height)
            if not is_rigid:
                for index in range(1, len(points)):
                    parent = min(points[:index], key=lambda point: self._distance_squared(point, points[index]))
                    self._draw_line(canvas, parent, points[index], ".")
        for points, poses, is_rigid in projected_groups:
            for index, (point, pose) in enumerate(zip(points, poses)):
                if is_rigid:
                    marker = "#"
                elif index == 0:
                    marker = "@"
                else:
                    marker = next((character.upper() for character in pose[2] if character.isalnum()), "o")
                self._set_cell(canvas, point, marker)
        if canvas:
            canvas[-1][:] = list(self._fit_text(" @ root | letters links | # rigid ", width))
        return canvas

    def _render_console_panel(self, width: int, height: int) -> list[str]:
        """Build a scrolling console pane at an exact character size."""
        inner_width = max(1, width - 2)
        inner_height = max(1, height - 2)
        title = self._fit_text(f" {self._console_title} ", inner_width, fill="-")
        visible_lines = list(self._console_lines)[-inner_height:]
        padding = [""] * (inner_height - len(visible_lines))
        lines = [self._fit_text(line, inner_width) for line in [*padding, *visible_lines]]
        return [f"+{title}+", *(f"|{line}|" for line in lines), f"+{'-' * inner_width}+"]

    def _collect_body_poses(self) -> list[_BodyGroup]:
        """Collect articulation-link and rigid-object poses for the selected environment."""
        if self._scene_data_provider is None:
            return []
        scene = self._scene_data_provider.get_interactive_scene()
        if scene is None:
            return []

        env_index = self.cfg.env_index
        origin = self._read_env_origin(scene, env_index)
        groups: list[_BodyGroup] = []
        for name, articulation in getattr(scene, "articulations", {}).items():
            body_names = list(getattr(articulation, "body_names", []))
            poses = self._read_articulation_poses(articulation, body_names, env_index, origin)
            if poses:
                groups.append((name, poses, False))
        for name, rigid_object in getattr(scene, "rigid_objects", {}).items():
            poses = self._read_body_poses(
                rigid_object.data.root_pos_w,
                getattr(rigid_object.data, "root_quat_w", None),
                [name],
                env_index,
                origin,
            )
            if poses:
                groups.append((name, poses, True))
        return groups

    @staticmethod
    def _read_articulation_poses(
        articulation: Any,
        body_names: list[str],
        env_index: int,
        origin: _Vector3,
    ) -> list[_BodyPose]:
        """Read link poses from the articulation's coherent scene-data buffers."""
        return AsciiVisualizer._read_body_poses(
            articulation.data.body_pos_w,
            getattr(articulation.data, "body_quat_w", None),
            body_names,
            env_index,
            origin,
        )

    def _draw_body_axes(
        self,
        canvas: list[list[str]],
        pose: _BodyPose,
        plot_width: int,
        plot_height: int,
    ) -> None:
        """Draw a body-local orientation glyph centered on its origin."""
        position, orientation, _ = pose
        half_length = self.cfg.body_axis_length / 2.0
        axes = (((1.0, 0.0, 0.0), "-"), ((0.0, 1.0, 0.0), ":"), ((0.0, 0.0, 1.0), "|"))
        for local_axis, character in axes:
            direction = self._quat_rotate(orientation, local_axis)
            start = tuple(position[index] - half_length * direction[index] for index in range(3))
            end = tuple(position[index] + half_length * direction[index] for index in range(3))
            projected = self._project_points([start, end], plot_width, plot_height)
            self._draw_line(canvas, projected[0], projected[1], character)

    def _write_console(self, text: str) -> None:
        """Collect complete redirected console lines and request a dashboard refresh."""
        self._console_buffer += text
        complete_lines = re.split(r"\r\n|\r|\n", self._console_buffer)
        self._console_buffer = complete_lines.pop()
        for line in complete_lines:
            self._console_lines.append(self._sanitize_console_line(line))
        if complete_lines:
            self._render_to_terminal()

    def _flush_console(self) -> None:
        """Commit a partial redirected line."""
        if self._console_buffer:
            self._console_lines.append(self._sanitize_console_line(self._console_buffer))
            self._console_buffer = ""
            self._render_to_terminal()

    @staticmethod
    def _sanitize_console_line(line: str) -> str:
        """Remove terminal controls from a captured console line."""
        line = _ANSI_ESCAPE_PATTERN.sub("", line).expandtabs(4).strip()
        return "".join(character for character in line if character.isprintable())

    @staticmethod
    def _read_env_origin(scene: Any, env_index: int) -> _Vector3:
        """Read one environment origin as host floats."""
        origins = getattr(scene, "env_origins", None)
        if origins is None:
            return (0.0, 0.0, 0.0)
        tensor = getattr(origins, "torch", origins)
        values = tensor[env_index].detach().cpu().tolist()
        return (float(values[0]), float(values[1]), float(values[2]))

    @staticmethod
    def _read_body_poses(
        position_values: Any,
        orientation_values: Any,
        body_names: list[str],
        env_index: int,
        origin: _Vector3,
    ) -> list[_BodyPose]:
        """Read one environment's body poses from tensors or ProxyArrays."""
        position_tensor = getattr(position_values, "torch", position_values)
        selected_positions = position_tensor[env_index]
        if selected_positions.ndim == 1:
            selected_positions = selected_positions.unsqueeze(0)
        host_positions = selected_positions[..., :3].detach().cpu().tolist()

        if orientation_values is None:
            host_orientations = [[0.0, 0.0, 0.0, 1.0] for _ in host_positions]
        else:
            orientation_tensor = getattr(orientation_values, "torch", orientation_values)
            selected_orientations = orientation_tensor[env_index]
            if selected_orientations.ndim == 1:
                selected_orientations = selected_orientations.unsqueeze(0)
            host_orientations = selected_orientations[..., :4].detach().cpu().tolist()

        poses = []
        for index, (position, orientation) in enumerate(zip(host_positions, host_orientations)):
            point = (
                float(position[0]) - origin[0],
                float(position[1]) - origin[1],
                float(position[2]) - origin[2],
            )
            quaternion = tuple(float(value) for value in orientation)
            if all(math.isfinite(value) for value in (*point, *quaternion)):
                body_name = body_names[index] if index < len(body_names) else ""
                poses.append((point, quaternion, body_name))
        return poses

    @staticmethod
    def _read_positions(values: Any, env_index: int, origin: _Vector3) -> list[_Vector3]:
        """Read one environment's positions from a tensor or ProxyArray."""
        return [pose[0] for pose in AsciiVisualizer._read_body_poses(values, None, [], env_index, origin)]

    def _project_points(self, points: list[_Vector3], width: int, height: int) -> list[tuple[int, int]]:
        """Orthographically project world points into character-cell coordinates."""
        forward = self._normalize(self._subtract(self._lookat, self._eye))
        up_reference = (0.0, 0.0, 1.0)
        if abs(self._dot(forward, up_reference)) > 0.99:
            up_reference = (0.0, 1.0, 0.0)
        right = self._normalize(self._cross(forward, up_reference))
        up = self._cross(right, forward)
        horizontal_span = self.cfg.view_span
        vertical_span = horizontal_span * max(height, 1) * 2.0 / max(width, 1)

        projected = []
        for point in points:
            relative = self._subtract(point, self._lookat)
            x = self._dot(relative, right)
            y = self._dot(relative, up)
            column = round((x / horizontal_span + 0.5) * (width - 1))
            row = round((0.5 - y / vertical_span) * (height - 1))
            column = max(-width, min(2 * width - 1, column))
            row = max(-height, min(2 * height - 1, row))
            projected.append((column, row))
        return projected

    @staticmethod
    def _draw_line(canvas: list[list[str]], start: tuple[int, int], end: tuple[int, int], character: str) -> None:
        """Draw a clipped Bresenham line into a character canvas."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            if 0 <= y0 < len(canvas) and 0 <= x0 < len(canvas[0]):
                canvas[y0][x0] = character
            if x0 == x1 and y0 == y1:
                break
            doubled_error = 2 * error
            if doubled_error >= dy:
                error += dy
                x0 += sx
            if doubled_error <= dx:
                error += dx
                y0 += sy

    @staticmethod
    def _set_cell(canvas: list[list[str]], point: tuple[int, int], character: str) -> None:
        """Set one character when the point is inside the canvas."""
        x, y = point
        if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
            canvas[y][x] = character

    @staticmethod
    def _fit_text(text: str, width: int, fill: str = " ") -> str:
        """Truncate or pad text to an exact width."""
        if width <= 0:
            return ""
        if len(text) > width:
            return "." * width if width <= 3 else text[: width - 3] + "..."
        return text.ljust(width, fill)

    @staticmethod
    def _resolve_terminal_stream(stream: TextIO) -> tuple[TextIO, bool]:
        """Return a stable stream connected to the controlling terminal."""
        if bool(getattr(stream, "isatty", lambda: False)()):
            return stream, False
        terminal_path = "CONOUT$" if os.name == "nt" else os.ctermid()
        try:
            return open(terminal_path, "w", buffering=1), True
        except OSError:
            return stream, False

    @staticmethod
    def _quat_rotate(quaternion: _Quaternion, vector: _Vector3) -> _Vector3:
        """Rotate a vector by an xyzw quaternion."""
        x, y, z, w = quaternion
        magnitude = math.sqrt(x * x + y * y + z * z + w * w)
        if magnitude == 0.0:
            return vector
        x, y, z, w = x / magnitude, y / magnitude, z / magnitude, w / magnitude
        unit = (x, y, z)
        unit_dot_vector = AsciiVisualizer._dot(unit, vector)
        unit_dot_unit = AsciiVisualizer._dot(unit, unit)
        cross = AsciiVisualizer._cross(unit, vector)
        return tuple(
            2.0 * unit_dot_vector * unit[index]
            + (w * w - unit_dot_unit) * vector[index]
            + 2.0 * w * cross[index]
            for index in range(3)
        )

    @staticmethod
    def _distance_squared(first: tuple[int, int], second: tuple[int, int]) -> int:
        """Return squared screen-space distance between two cells."""
        return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2

    @staticmethod
    def _subtract(first: _Vector3, second: _Vector3) -> _Vector3:
        """Subtract two three-dimensional vectors."""
        return (first[0] - second[0], first[1] - second[1], first[2] - second[2])

    @staticmethod
    def _dot(first: _Vector3, second: _Vector3) -> float:
        """Return the dot product of two three-dimensional vectors."""
        return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]

    @staticmethod
    def _cross(first: _Vector3, second: _Vector3) -> _Vector3:
        """Return the cross product of two three-dimensional vectors."""
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    @staticmethod
    def _normalize(vector: _Vector3) -> _Vector3:
        """Normalize a three-dimensional vector."""
        magnitude = math.sqrt(AsciiVisualizer._dot(vector, vector))
        if magnitude == 0.0:
            return (1.0, 0.0, 0.0)
        return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)


class _ConsoleCapture:
    """Line-buffered stream adapter for an ASCII visualizer."""

    encoding = "utf-8"

    def __init__(self, visualizer: AsciiVisualizer) -> None:
        self._visualizer = visualizer

    def write(self, text: str) -> int:
        """Capture text and return the number of consumed characters."""
        self._visualizer._write_console(text)
        return len(text)

    def flush(self) -> None:
        """Flush a partial line to the console pane."""
        self._visualizer._flush_console()

    def isatty(self) -> bool:
        """Report that captured output is not a standalone terminal."""
        return False
