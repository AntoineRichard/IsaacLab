# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batch skinned pompon ribbons by color and material for OvRTX."""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from pompon import PALETTE, RIBBON_SEGMENTS, RIBBONS_PER_GUIDE


@dataclass
class RibbonBatch:
    """Render mesh indices and material for one group of ribbons."""

    name: str
    vertices: np.ndarray
    triangles: wp.array
    color: tuple[float, float, float]
    roughness: float
    metallic: float


def make_batches(model: newton.Model) -> list[RibbonBatch]:
    """Group the skinned triangles by color and finish with fixed topology."""
    groups: dict[tuple[int, bool], list[np.ndarray]] = {}
    vertex_offset = 0
    indices_per_ribbon = 6 * RIBBON_SEGMENTS
    vertices_per_ribbon = 2 * (RIBBON_SEGMENTS + 1)
    for guide, mesh in enumerate(model.deformable_visual_meshes):
        indices = mesh.indices.numpy().reshape(RIBBONS_PER_GUIDE, indices_per_ribbon)
        palette_positions = mesh.uvs.numpy()[::vertices_per_ribbon, 0]
        for ribbon, triangles in enumerate(indices):
            shade = min(int(palette_positions[ribbon] * len(PALETTE)), len(PALETTE) - 1)
            is_foil = (ribbon + guide) % 4 == 0
            groups.setdefault((shade, is_foil), []).append(triangles + vertex_offset)
        vertex_offset += mesh.vertex_count

    batches = []
    for (shade, is_foil), indices in sorted(groups.items()):
        vertices, compact_indices = np.unique(np.concatenate(indices), return_inverse=True)
        srgb = PALETTE[shade] / 255.0
        color = tuple(np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4).tolist())
        batches.append(
            RibbonBatch(
                name=f"/ribbons/{'foil' if is_foil else 'plastic'}_{shade}",
                vertices=vertices,
                triangles=wp.array(compact_indices.astype(np.int32), dtype=wp.int32),
                color=color,
                roughness=0.16 if is_foil else 0.53,
                metallic=0.88 if is_foil else 0.0,
            )
        )
    return batches
