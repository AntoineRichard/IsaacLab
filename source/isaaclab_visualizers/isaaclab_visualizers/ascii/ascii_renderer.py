# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Triangle rasterizer for the ASCII terminal visualizer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .quadrant import render as encode_quadrants

_Vector3 = tuple[float, float, float]
_Quaternion = tuple[float, float, float, float]
_ScreenVertex = tuple[float, float, float]
_SHADING_CHARACTERS = ".:-=+*#%@"


@dataclass(frozen=True)
class AsciiMesh:
    """Body-local triangle mesh prepared for terminal rendering."""

    vertices: tuple[_Vector3, ...]
    faces: tuple[tuple[int, int, int], ...]
    edges: tuple[tuple[int, int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class AsciiRenderInstance:
    """A cached mesh and its live world pose."""

    mesh: AsciiMesh
    position: _Vector3
    orientation: _Quaternion
    include_in_auto_fit: bool = True
    color: tuple[int, int, int] | None = None
    """Body color for :meth:`AsciiRenderer.render_color`. Ignored by :meth:`render`."""


class AsciiRenderer:
    """Render posed triangle meshes into a depth-buffered character canvas."""

    def __init__(self) -> None:
        """Initialize auto-framing state."""
        self._frame_center: tuple[float, float] | None = None
        self._frame_span: float | None = None

    def render(
        self,
        instances: list[AsciiRenderInstance],
        width: int,
        height: int,
        eye: _Vector3,
        lookat: _Vector3,
        view_span: float,
        auto_fit: bool,
        auto_fit_margin: float,
    ) -> list[list[str]]:
        """Rasterize posed meshes.

        Args:
            instances: Meshes and their world poses.
            width: Canvas width in character cells.
            height: Canvas height in character cells.
            eye: Camera eye position.
            lookat: Camera look-at position.
            view_span: Manual horizontal span in meters.
            auto_fit: Whether to derive framing from the current geometry.
            auto_fit_margin: Multiplicative margin around auto-fitted geometry.

        Returns:
            Two-dimensional character canvas.
        """
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        if not instances or width <= 0 or height <= 0:
            return canvas

        forward, world_meshes, projected_meshes = self._project_scene(
            instances, width, height, eye, lookat, view_span, auto_fit, auto_fit_margin
        )
        horizontal_span = self._frame_span if self._frame_span is not None else view_span

        depth_buffer = [[math.inf for _ in range(width)] for _ in range(height)]
        face_orientations: list[list[float]] = []
        light_direction = self._normalize((0.35, -0.45, 0.82))
        for (world_vertices, mesh), projected_vertices in zip(world_meshes, projected_meshes):
            orientations = []
            for face in mesh.faces:
                first, second, third = (world_vertices[index] for index in face)
                normal = self._normalize(self._cross(self._subtract(second, first), self._subtract(third, first)))
                orientations.append(self._dot(normal, forward))
                intensity = 0.18 + 0.82 * abs(self._dot(normal, light_direction))
                character_index = min(
                    len(_SHADING_CHARACTERS) - 1,
                    round(intensity * (len(_SHADING_CHARACTERS) - 1)),
                )
                self._rasterize_triangle(
                    canvas,
                    depth_buffer,
                    *(projected_vertices[index] for index in face),
                    _SHADING_CHARACTERS[character_index],
                )
            face_orientations.append(orientations)

        depth_tolerance = horizontal_span * 0.02
        for (_, mesh), projected_vertices, orientations in zip(
            world_meshes,
            projected_meshes,
            face_orientations,
        ):
            for first_index, second_index, adjacent_faces in mesh.edges:
                if len(adjacent_faces) > 1 and all(
                    orientations[adjacent_faces[0]] * orientations[index] > 0.0 for index in adjacent_faces[1:]
                ):
                    continue
                self._rasterize_line(
                    canvas,
                    depth_buffer,
                    projected_vertices[first_index],
                    projected_vertices[second_index],
                    "#",
                    depth_tolerance,
                )
        return canvas

    def render_color(
        self,
        instances: list[AsciiRenderInstance],
        width: int,
        height: int,
        eye: _Vector3,
        lookat: _Vector3,
        view_span: float,
        auto_fit: bool,
        auto_fit_margin: float,
        fallback_color: tuple[int, int, int] = (150, 154, 160),
    ) -> list[str]:
        """Rasterize posed meshes into colored terminal rows, one color per body.

        Draws onto a grid of quadrant subpixels, two across and two down per character cell,
        and lets :mod:`.quadrant` choose the glyph and the two colors that best reproduce each
        cell. That is four times the spatial resolution of :meth:`render`, and it carries each
        body's own color rather than a shading character.

        A subpixel keeps the cell's proportions, so the framing maths is the one :meth:`render`
        uses, applied at twice the resolution in each axis.

        Args:
            instances: Meshes and their world poses. Each may carry a color.
            width: Canvas width in character cells.
            height: Canvas height in character cells.
            eye: Camera eye position.
            lookat: Camera look-at position.
            view_span: Manual horizontal span in meters.
            auto_fit: Whether to derive framing from the current geometry.
            auto_fit_margin: Multiplicative margin around auto-fitted geometry.
            fallback_color: Color for an instance that carries none.

        Returns:
            One string per row, carrying its own color escapes. Rows are exactly *width*
            cells wide; the escapes occupy no columns, so anything measuring these strings
            must strip them first.
        """
        if not instances or width <= 0 or height <= 0:
            return [""] * max(height, 0)

        pixel_width, pixel_height = width * 2, height * 2
        canvas: list[list[tuple[int, int, int] | None]] = [
            [None for _ in range(pixel_width)] for _ in range(pixel_height)
        ]
        _, world_meshes, projected_meshes = self._project_scene(
            instances, pixel_width, pixel_height, eye, lookat, view_span, auto_fit, auto_fit_margin
        )

        depth_buffer = [[math.inf for _ in range(pixel_width)] for _ in range(pixel_height)]
        light_direction = self._normalize((0.35, -0.45, 0.82))
        for (world_vertices, mesh), projected_vertices, instance in zip(world_meshes, projected_meshes, instances):
            color = instance.color or fallback_color
            for face in mesh.faces:
                first, second, third = (world_vertices[index] for index in face)
                normal = self._normalize(self._cross(self._subtract(second, first), self._subtract(third, first)))
                # the same Lambert term the character path uses, applied to the body's color
                # rather than used to pick a glyph from a ramp
                intensity = 0.30 + 0.70 * abs(self._dot(normal, light_direction))
                shaded = tuple(min(255, round(channel * intensity)) for channel in color)
                self._rasterize_triangle(canvas, depth_buffer, *(projected_vertices[index] for index in face), shaded)

        return encode_quadrants(canvas, width, height).split("\n")

    def _project_scene(
        self,
        instances: list[AsciiRenderInstance],
        width: int,
        height: int,
        eye: _Vector3,
        lookat: _Vector3,
        view_span: float,
        auto_fit: bool,
        auto_fit_margin: float,
    ):
        """Frame the scene and project every vertex, shared by both rasterizing paths.

        Returns:
            The camera forward vector, the world-space meshes, and the projected vertices.
        """
        forward, right, up = self._camera_basis(eye, lookat)
        world_meshes = [
            ([self._transform_vertex(vertex, instance) for vertex in instance.mesh.vertices], instance.mesh)
            for instance in instances
        ]
        camera_vertices = [
            [self._to_camera(vertex, eye, forward, right, up) for vertex in vertices] for vertices, _ in world_meshes
        ]
        framing_vertices = [
            vertices for vertices, instance in zip(camera_vertices, instances) if instance.include_in_auto_fit
        ]
        center, horizontal_span = self._resolve_frame(
            framing_vertices or camera_vertices, width, height, view_span, auto_fit, auto_fit_margin
        )
        vertical_span = horizontal_span * max(height, 1) * 2.0 / max(width, 1)
        projected_meshes = [
            [self._project_vertex(vertex, center, horizontal_span, vertical_span, width, height) for vertex in vertices]
            for vertices in camera_vertices
        ]
        return forward, world_meshes, projected_meshes

    def _resolve_frame(
        self,
        camera_vertices: list[list[_Vector3]],
        width: int,
        height: int,
        view_span: float,
        auto_fit: bool,
        margin: float,
    ) -> tuple[tuple[float, float], float]:
        """Resolve stable camera-space framing."""
        if not auto_fit:
            return (0.0, 0.0), view_span
        all_vertices = [vertex for vertices in camera_vertices for vertex in vertices]
        if not all_vertices:
            return (0.0, 0.0), view_span

        min_x = min(vertex[0] for vertex in all_vertices)
        max_x = max(vertex[0] for vertex in all_vertices)
        min_y = min(vertex[1] for vertex in all_vertices)
        max_y = max(vertex[1] for vertex in all_vertices)
        target_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        horizontal_extent = max_x - min_x
        vertical_extent = (max_y - min_y) * max(width, 1) / (max(height, 1) * 2.0)
        target_span = max(0.25, horizontal_extent, vertical_extent) * margin

        if self._frame_center is None or self._frame_span is None:
            self._frame_center = target_center
            self._frame_span = target_span
        else:
            smoothing = 0.2
            self._frame_center = (
                self._frame_center[0] + smoothing * (target_center[0] - self._frame_center[0]),
                self._frame_center[1] + smoothing * (target_center[1] - self._frame_center[1]),
            )
            self._frame_span += smoothing * (target_span - self._frame_span)
        return self._frame_center, self._frame_span

    @staticmethod
    def _camera_basis(eye: _Vector3, lookat: _Vector3) -> tuple[_Vector3, _Vector3, _Vector3]:
        """Return forward, right, and up camera vectors."""
        forward = AsciiRenderer._normalize(AsciiRenderer._subtract(lookat, eye))
        up_reference = (0.0, 0.0, 1.0)
        if abs(AsciiRenderer._dot(forward, up_reference)) > 0.99:
            up_reference = (0.0, 1.0, 0.0)
        right = AsciiRenderer._normalize(AsciiRenderer._cross(forward, up_reference))
        return forward, right, AsciiRenderer._cross(right, forward)

    @staticmethod
    def _transform_vertex(vertex: _Vector3, instance: AsciiRenderInstance) -> _Vector3:
        """Transform one body-local vertex to world space."""
        rotated = AsciiRenderer._quat_rotate(instance.orientation, vertex)
        return tuple(rotated[index] + instance.position[index] for index in range(3))

    @staticmethod
    def _to_camera(
        vertex: _Vector3,
        eye: _Vector3,
        forward: _Vector3,
        right: _Vector3,
        up: _Vector3,
    ) -> _Vector3:
        """Transform a world vertex to orthographic camera coordinates."""
        relative = AsciiRenderer._subtract(vertex, eye)
        return (
            AsciiRenderer._dot(relative, right),
            AsciiRenderer._dot(relative, up),
            AsciiRenderer._dot(relative, forward),
        )

    @staticmethod
    def _project_vertex(
        vertex: _Vector3,
        center: tuple[float, float],
        horizontal_span: float,
        vertical_span: float,
        width: int,
        height: int,
    ) -> _ScreenVertex:
        """Project a camera-space vertex to floating-point cell coordinates."""
        column = ((vertex[0] - center[0]) / horizontal_span + 0.5) * (width - 1)
        row = (0.5 - (vertex[1] - center[1]) / vertical_span) * (height - 1)
        return (column, row, vertex[2])

    @staticmethod
    def _rasterize_triangle(
        canvas: list[list[str]],
        depth_buffer: list[list[float]],
        first: _ScreenVertex,
        second: _ScreenVertex,
        third: _ScreenVertex,
        character: str,
    ) -> None:
        """Fill one projected triangle using barycentric depth interpolation."""
        width = len(canvas[0])
        height = len(canvas)
        area = AsciiRenderer._edge(first, second, third[0], third[1])
        if abs(area) < 1.0e-8:
            return
        min_x = max(0, math.floor(min(first[0], second[0], third[0])))
        max_x = min(width - 1, math.ceil(max(first[0], second[0], third[0])))
        min_y = max(0, math.floor(min(first[1], second[1], third[1])))
        max_y = min(height - 1, math.ceil(max(first[1], second[1], third[1])))
        for row in range(min_y, max_y + 1):
            for column in range(min_x, max_x + 1):
                sample_x = column + 0.5
                sample_y = row + 0.5
                first_weight = AsciiRenderer._edge(second, third, sample_x, sample_y) / area
                second_weight = AsciiRenderer._edge(third, first, sample_x, sample_y) / area
                third_weight = 1.0 - first_weight - second_weight
                if min(first_weight, second_weight, third_weight) < -1.0e-6:
                    continue
                depth = first_weight * first[2] + second_weight * second[2] + third_weight * third[2]
                if depth < depth_buffer[row][column]:
                    depth_buffer[row][column] = depth
                    canvas[row][column] = character

    @staticmethod
    def _rasterize_line(
        canvas: list[list[str]],
        depth_buffer: list[list[float]],
        first: _ScreenVertex,
        second: _ScreenVertex,
        character: str,
        depth_tolerance: float,
    ) -> None:
        """Draw a depth-tested projected line."""
        delta_x = second[0] - first[0]
        delta_y = second[1] - first[1]
        steps = max(1, math.ceil(max(abs(delta_x), abs(delta_y))))
        for step in range(steps + 1):
            fraction = step / steps
            column = round(first[0] + fraction * delta_x)
            row = round(first[1] + fraction * delta_y)
            if not (0 <= row < len(canvas) and 0 <= column < len(canvas[0])):
                continue
            depth = first[2] + fraction * (second[2] - first[2])
            if depth <= depth_buffer[row][column] + depth_tolerance:
                depth_buffer[row][column] = min(depth, depth_buffer[row][column])
                canvas[row][column] = character

    @staticmethod
    def _edge(first: _ScreenVertex, second: _ScreenVertex, x: float, y: float) -> float:
        """Return the signed edge function for a two-dimensional point."""
        return (x - first[0]) * (second[1] - first[1]) - (y - first[1]) * (second[0] - first[0])

    @staticmethod
    def _quat_rotate(quaternion: _Quaternion, vector: _Vector3) -> _Vector3:
        """Rotate a vector by an xyzw quaternion."""
        x, y, z, w = quaternion
        magnitude = math.sqrt(x * x + y * y + z * z + w * w)
        if magnitude == 0.0:
            return vector
        x, y, z, w = x / magnitude, y / magnitude, z / magnitude, w / magnitude
        unit = (x, y, z)
        unit_dot_vector = AsciiRenderer._dot(unit, vector)
        unit_dot_unit = AsciiRenderer._dot(unit, unit)
        cross = AsciiRenderer._cross(unit, vector)
        return tuple(
            2.0 * unit_dot_vector * unit[index] + (w * w - unit_dot_unit) * vector[index] + 2.0 * w * cross[index]
            for index in range(3)
        )

    @staticmethod
    def _subtract(first: _Vector3, second: _Vector3) -> _Vector3:
        """Subtract two vectors."""
        return (first[0] - second[0], first[1] - second[1], first[2] - second[2])

    @staticmethod
    def _dot(first: _Vector3, second: _Vector3) -> float:
        """Return the dot product of two vectors."""
        return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]

    @staticmethod
    def _cross(first: _Vector3, second: _Vector3) -> _Vector3:
        """Return the cross product of two vectors."""
        return (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )

    @staticmethod
    def _normalize(vector: _Vector3) -> _Vector3:
        """Normalize a vector."""
        magnitude = math.sqrt(AsciiRenderer._dot(vector, vector))
        if magnitude == 0.0:
            return (1.0, 0.0, 0.0)
        return (vector[0] / magnitude, vector[1] / magnitude, vector[2] / magnitude)
