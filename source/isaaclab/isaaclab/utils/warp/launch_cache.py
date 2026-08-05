# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any

import warp as wp
from warp._src.context import runtime


class _CapturedGraphLease:
    """Own a CUDA graph until its native handles are deterministically revoked."""

    def __init__(self, graph: wp.Graph, *, retained: object) -> None:
        self._graph: wp.Graph | None = graph
        self._cleanup_graph: wp.Graph | None = graph
        self._retained: object | None = retained
        self._revoked = False

    @property
    def is_live(self) -> bool:
        """Whether the graph can still be launched."""
        return not self._revoked and self._graph is not None

    def launch(self, stream: wp.Stream) -> None:
        """Replay the graph on its captured Torch-interoperable stream."""
        if not self.is_live:
            raise RuntimeError("actuator graph lease was revoked and cannot be launched")
        assert self._graph is not None
        wp.capture_launch(self._graph, stream=stream)

    def revoke(self) -> None:
        """Synchronize and destroy graph handles, retaining failures for retry."""
        self._revoked = True
        self._graph = None
        graph = self._cleanup_graph
        if graph is None:
            return
        wp.synchronize_device(graph.device)
        failures: list[BaseException] = []
        self._destroy_handle(graph, "graph", runtime.core.wp_cuda_graph_destroy, failures)
        self._destroy_handle(graph, "graph_exec", runtime.core.wp_cuda_graph_exec_destroy, failures)
        if getattr(graph, "graph", None) is None and getattr(graph, "graph_exec", None) is None:
            self._cleanup_graph = None
            self._retained = None
        if failures:
            raise failures[0]

    @staticmethod
    def _destroy_handle(graph: Any, name: str, destroy: Any, failures: list[BaseException]) -> None:
        """Destroy one native graph handle while leaving failures retryable."""
        handle = getattr(graph, name, None)
        if handle is None:
            return
        try:
            with graph.device.context_guard:
                if not destroy(graph.device.context, handle):
                    raise RuntimeError(f"failed to destroy CUDA graph {name}")
            setattr(graph, name, None)
        except BaseException as error:
            failures.append(error)


class _WarpLaunchCache:
    """Cache stable eager-CUDA launches."""

    def __init__(self, device: str):
        self._device = wp.get_device(device)
        self._commands: dict[object, wp.Launch] = {}

    @property
    def is_cuda(self) -> bool:
        """Whether this cache records fixed launches on a CUDA device."""
        return self._device.is_cuda

    def launch(self, key: object, kernel: wp.Kernel, *, dim, inputs, outputs) -> None:
        if not self._device.is_cuda or self._device.is_capturing:
            wp.launch(kernel, dim=dim, inputs=inputs, outputs=outputs, device=self._device)
            return
        command = self._commands.get(key)
        if command is None:
            command = wp.launch(
                kernel,
                dim=dim,
                inputs=inputs,
                outputs=outputs,
                device=self._device,
                record_cmd=True,
            )
            if command is None:
                return
            self._commands[key] = command
        command.launch()

    def clear(self) -> None:
        self._commands.clear()
