# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private revocable ownership for actuator CUDA graphs."""

from __future__ import annotations

from typing import Any

import warp as wp
from warp._src.context import runtime


class _CapturedGraphLease:
    """Own one captured graph and make retained references safely revocable."""

    def __init__(self, graph: wp.Graph, *, generation: object, label: str):
        self._graph: wp.Graph | None = graph
        # Keep the raw graph privately until every native handle is released.
        # ``_graph`` is cleared as soon as revocation starts so no retained
        # lease can replay it while a failed cleanup is retried.
        self._cleanup_graph: wp.Graph | None = graph
        self._generation: object | None = generation
        self._label = label
        self._revoked = False

    @property
    def is_live(self) -> bool:
        """Whether this lease still accepts launches."""
        return not self._revoked and self._graph is not None

    def launch(self, stream: wp.Stream | None = None) -> None:
        """Replay the graph while this lease is live."""
        graph = self._graph
        if self._revoked or graph is None:
            raise RuntimeError(f"{self._label} lease was revoked and cannot be launched")
        if stream is None:
            wp.capture_launch(graph)
        else:
            wp.capture_launch(graph, stream=stream)

    def revoke(self) -> None:
        """Synchronize, detach, and release every native graph handle.

        A failed synchronization or native destroy leaves the raw graph and
        generation pin attached so a later call can finish cleanup without
        releasing captured buffers. The lease itself becomes non-launchable
        before the first cleanup attempt.
        """
        if self._revoked and self._cleanup_graph is None:
            return
        self._revoked = True
        graph = self._cleanup_graph
        self._graph = None
        if graph is None:
            return

        # Native graph handles and the captured generation must remain alive
        # when synchronization fails. Destroying either before the device is
        # known idle can leave in-flight kernels referring to released storage.
        wp.synchronize_device(graph.device)

        failures: list[BaseException] = []
        self._destroy_handle(graph, "graph", runtime.core.wp_cuda_graph_destroy, failures)
        self._destroy_handle(graph, "graph_exec", runtime.core.wp_cuda_graph_exec_destroy, failures)

        if getattr(graph, "graph", None) is None and getattr(graph, "graph_exec", None) is None:
            self._cleanup_graph = None
            self._generation = None

        if failures:
            primary, *remaining = failures
            for error in remaining:
                primary.add_note(f"Additional {self._label} revocation failure: {error!r}")
            raise primary

    @staticmethod
    def _destroy_handle(graph: Any, name: str, destroy: Any, failures: list[BaseException]) -> None:
        """Destroy and null one Warp graph handle, retaining cleanup errors."""
        handle = getattr(graph, name, None)
        if handle is None:
            return
        try:
            with graph.device.context_guard:
                destroyed = destroy(graph.device.context, handle)
            if not destroyed:
                raise RuntimeError(f"Failed to destroy native CUDA {name!r} handle")
            setattr(graph, name, None)
        except BaseException as error:
            failures.append(error)
