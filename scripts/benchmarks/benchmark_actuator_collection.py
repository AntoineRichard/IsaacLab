# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private, revision-adaptive actuator collection benchmark driver.

This module intentionally has no Isaac Lab, Torch, or Warp import at module
scope.  A coordinator can consequently launch clean historical children
without importing their target actuator implementation before process isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "actuator_collection_attempt/v1"
_REVISIONS = ("develop", "current", "global")
_EXECUTIONS = ("cached_eager", "graph")
_OPAQUE_ACTUATOR_TYPE: type | None = None


@dataclass(frozen=True)
class BuildCase:
    """One frozen build workload definition."""

    name: str
    worlds: tuple[int, ...]
    num_sources: int
    num_articulations: int
    actuator_types: tuple[str, ...]
    groups: tuple[int, ...]
    global_only: bool = False


@dataclass(frozen=True)
class BuildRow:
    """One resolved build workload."""

    case: str
    num_worlds: int
    num_sources: int
    num_articulations: int
    groups: int
    actuator_types: tuple[str, ...]
    global_only: bool = False


@dataclass(frozen=True)
class RuntimeRow:
    """One requested runtime workload."""

    actuator_type: str
    groups: int
    requested_execution: str
    effective_execution: str | None
    num_worlds: int = 4096


@dataclass(frozen=True)
class RevisionCapability:
    """Feature decision for a revision adapter."""

    supported: bool
    reason: str | None = None


@dataclass(frozen=True)
class AttemptIdentity:
    """Immutable identity shared by one final child document."""

    batch_id: str
    observation_key: str
    attempt_id: str
    candidate_sha: str
    revision_shas: dict[str, str]
    harness_sha256: str


@dataclass(frozen=True)
class TelemetrySample:
    """One GPU telemetry observation."""

    timestamp_s: float
    temperature_c: float | None
    utilization_pct: float | None
    sm_clock_mhz: float | None
    memory_clock_mhz: float | None
    throttle_reasons: str | None
    compute_pids: tuple[int, ...]


@dataclass(frozen=True)
class _Workload:
    row: BuildRow
    device: str
    joint_names: tuple[str, ...]
    group_values: tuple[tuple[float, ...], ...]
    first_command: tuple[float, ...]
    network_file: str | None = None


def build_matrix() -> tuple[BuildCase, ...]:
    """Return B0 through B8 in the frozen contract order."""
    return (
        BuildCase("B0", (1,), 0, 0, (), (0,), True),
        BuildCase("B1", (1, 64, 4096), 1, 1, ("implicit",), (3,)),
        BuildCase("B2", (4096,), 1, 2, ("implicit",), (3,), True),
        BuildCase("B3", (4096,), 4, 1, ("implicit",), (3,)),
        BuildCase("B4", (4096,), 1, 1, ("implicit", "ideal_pd", "dc_motor"), (3,)),
        BuildCase("B5", (4096,), 1, 1, ("implicit", "ideal_pd", "dc_motor"), (1, 3, 12)),
        BuildCase("B6", (4096,), 1, 1, ("implicit",), (3,), True),
        BuildCase("B7", (64,), 1, 1, ("neural", "delayed", "remotized", "opaque"), (4,)),
        BuildCase("B8", (4096,), 1, 1, ("implicit",), (3,), True),
    )


def expand_build_matrix(selector: str = "all") -> tuple[BuildRow, ...]:
    """Expand frozen cases deterministically, without dictionary ordering."""
    rows: list[BuildRow] = []
    for case in build_matrix():
        if selector != "all" and case.name != selector:
            continue
        if case.name == "B5":
            for actuator_type in case.actuator_types:
                for groups in case.groups:
                    rows.append(BuildRow(case.name, 4096, 1, 1, groups, (actuator_type,)))
            continue
        for worlds in case.worlds:
            rows.append(
                BuildRow(
                    case.name,
                    worlds,
                    case.num_sources,
                    case.num_articulations,
                    case.groups[0],
                    case.actuator_types,
                    case.global_only,
                )
            )
    return tuple(rows)


def runtime_matrix(revision: str) -> tuple[RuntimeRow, ...]:
    """Return all eighteen requested runtime rows in stable order."""
    _require_revision(revision)
    return tuple(
        RuntimeRow(
            actuator_type, groups, execution, execution if execution == "cached_eager" or revision == "global" else None
        )
        for actuator_type in ("implicit", "ideal_pd", "dc_motor")
        for groups in (1, 3, 12)
        for execution in _EXECUTIONS
    )


def expand_runtime_matrix(revision: str, selectors: tuple[str, ...] | None = None) -> tuple[RuntimeRow, ...]:
    """Filter requested rows while preserving the frozen matrix order."""
    rows = runtime_matrix(revision)
    return rows if not selectors else tuple(row for row in rows if row_key(row) in selectors)


def row_key(row: BuildRow | RuntimeRow) -> str:
    """Return a stable, human-readable row key."""
    if isinstance(row, BuildRow):
        dimensions = f"{row.case}:w{row.num_worlds}:s{row.num_sources}:a{row.num_articulations}:g{row.groups}"
        return f"{dimensions}:{','.join(row.actuator_types)}"
    return f"{row.actuator_type}:g{row.groups}:{row.requested_execution}"


def _require_revision(revision: str) -> None:
    if revision not in _REVISIONS:
        raise ValueError(f"unknown revision: {revision}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate driver arguments before heavyweight imports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("build", "runtime", "coordinate"), required=True)
    parser.add_argument("--case", choices=("all", *(case.name for case in build_matrix())), default="all")
    parser.add_argument("--revision", choices=_REVISIONS)
    parser.add_argument("--revision_sha")
    parser.add_argument("--candidate_sha")
    parser.add_argument("--observation_key")
    parser.add_argument("--attempt_id")
    parser.add_argument("--phase")
    parser.add_argument("--child_row")
    parser.add_argument("--harness_sha256")
    parser.add_argument("--batch_id", default="cpu-smoke")
    parser.add_argument("--final_run", action="store_true")
    parser.add_argument("--num_worlds", type=int)
    parser.add_argument("--num_sources", type=int)
    parser.add_argument("--num_articulations", type=int)
    parser.add_argument("--groups", type=int)
    parser.add_argument("--actuator_types")
    parser.add_argument("--warmup_iterations", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--benchmark_formatter", default="schema")
    parser.add_argument("--output_path", type=Path, required=False, default=Path("actuator-benchmarks"))
    parser.add_argument("--matrix", choices=("build", "runtime"), default="build")
    parser.add_argument("--run_root", type=Path)
    parser.add_argument("--repetitions", type=int, default=6)
    args = parser.parse_args(argv)
    for name in (
        "num_worlds",
        "num_sources",
        "num_articulations",
        "groups",
        "warmup_iterations",
        "num_iterations",
        "repetitions",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"argument --{name}: must be greater than zero")
    if (
        args.mode == "build"
        and args.case == "all"
        and any(
            getattr(args, name) is not None
            for name in ("num_worlds", "num_sources", "num_articulations", "groups", "actuator_types")
        )
    ):
        parser.error("--case all does not accept scalar workload overrides")
    if args.final_run:
        missing = [
            name
            for name in (
                "revision",
                "revision_sha",
                "candidate_sha",
                "observation_key",
                "attempt_id",
                "phase",
                "child_row",
                "harness_sha256",
            )
            if not getattr(args, name)
        ]
        if missing:
            parser.error("--final_run requires " + ", ".join("--" + name for name in missing))
    if args.benchmark_formatter != "schema":
        parser.error("--benchmark_formatter must be schema for actuator_collection_attempt/v1")
    return args


def validate_attempt(record: dict[str, Any]) -> None:
    """Validate the narrow private attempt schema without repairing records."""
    if record.get("schema") != SCHEMA:
        raise ValueError("schema")
    identity = record.get("identity") or {}
    for key in ("batch_id", "observation_key", "attempt_id", "candidate_sha", "revision_shas", "harness_sha256"):
        if not identity.get(key):
            raise ValueError(key)
    if record.get("kind") not in {"pair", "singleton"} or record.get("status") not in {
        "accepted",
        "rejected",
        "unsupported",
    }:
        raise ValueError("kind or status")
    if record.get("boundary") not in {
        "resolved_construction_to_first_application",
        "runtime_application",
        "empty_finalize_clear",
    }:
        raise ValueError("boundary")
    for member in record.get("members", []):
        capability = member.get("capability")
        if not capability or "supported" not in capability:
            raise ValueError("capability")
        if not capability["supported"]:
            if member.get("effective_execution") is not None or member.get("timing"):
                raise ValueError("unsupported member has timing or effective execution")
            if not capability.get("reason"):
                raise ValueError("unsupported reason")
    if not record.get("members"):
        raise ValueError("members")


def allocate_attempt_dir(observation_path: Path) -> Path:
    """Create the next attempt directory using exclusive mkdir semantics."""
    observation_path.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10_000):
        path = observation_path / f"attempt-{number:02}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise RuntimeError("attempt space exhausted")


def write_attempt_atomically(attempt_dir: Path, record: dict[str, Any]) -> Path:
    """Write a validated immutable attempt document by atomic rename."""
    validate_attempt(record)
    target = attempt_dir / "attempt.json"
    if target.exists():
        raise FileExistsError(target)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=attempt_dir, delete=False) as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp = Path(handle.name)
    try:
        os.link(temp, target)
    except FileExistsError:
        raise
    finally:
        temp.unlink(missing_ok=True)
    return target


def make_workload(row: BuildRow, device: str) -> _Workload:
    """Create all driver-owned control data before an adapter boundary."""
    values = tuple(
        tuple(float(source + group + 1) for _ in range(3))
        for source in range(max(row.num_sources, 1))
        for group in range(max(row.groups, 1))
    )
    joint_names = tuple(f"joint_{index}" for index in range(max(row.groups, 1)))
    network_file = _tiny_mlp_checkpoint(device) if row.case == "B7" else None
    return _Workload(row, device, joint_names, values, (0.1, 0.2, 0.3), network_file)


class _Adapter(Protocol):
    def build_workload(self, workload: _Workload) -> None: ...
    def first_application(self, workload: _Workload) -> None: ...
    def warmup_execution(self, row: RuntimeRow) -> bool: ...
    def run_execution(self, count: int) -> None: ...
    def close(self) -> None: ...
    def introspect(self) -> dict[str, Any] | None: ...


class _MemoryAdapter:
    """Driver-owned CPU fallback used only for smoke and unavailable feature probes."""

    def __init__(self, revision: str, device: str) -> None:
        self.revision = revision
        self.device = device
        self.workload: _Workload | None = None
        self.applications = 0

    def build_workload(self, workload: _Workload) -> None:
        self.workload = workload

    def first_application(self, workload: _Workload) -> None:
        if self.workload is not workload:
            raise RuntimeError("adapter did not receive driver workload")
        self.applications += 1

    def warmup_execution(self, row: RuntimeRow) -> bool:
        return row.requested_execution != "graph" or self.revision == "global"

    def run_execution(self, count: int) -> None:
        self.applications += count

    def close(self) -> None:
        if self.workload is not None and self.workload.network_file is not None:
            Path(self.workload.network_file).unlink(missing_ok=True)
        self.workload = None

    def introspect(self) -> dict[str, Any] | None:
        return {} if self.revision == "global" else None


class _DriverControl:
    """Benchmark-owned backend-neutral control fixture for real collection paths."""

    def __init__(self, device: str, num_worlds: int, joint_names: tuple[str, ...], num_sources: int = 1) -> None:
        import torch
        import warp as wp

        from isaaclab.utils.warp import ProxyArray

        self._torch, self._wp = torch, wp
        self._device, self._joint_names, self._num_instances, self._num_sources = (
            device,
            joint_names,
            num_worlds,
            num_sources,
        )
        self._joint_pos = ProxyArray(wp.zeros((num_worlds, len(joint_names)), dtype=wp.float32, device=device))
        self._joint_vel = ProxyArray(wp.zeros((num_worlds, len(joint_names)), dtype=wp.float32, device=device))
        self.submissions = 0
        self.command_stages: list[str] = []

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_joints(self) -> int:
        return len(self._joint_names)

    @property
    def num_fixed_tendons(self) -> int:
        return 0

    @property
    def device(self) -> str:
        return self._device

    @property
    def joint_pos(self) -> Any:
        return self._joint_pos

    @property
    def joint_vel(self) -> Any:
        return self._joint_vel

    def find_joints(self, names: str | list[str] | tuple[str, ...]) -> tuple[list[int], list[str]]:
        import re

        expressions = [names] if isinstance(names, str) else list(names)
        found = [
            (index, name)
            for index, name in enumerate(self._joint_names)
            if any(re.fullmatch(expression, name) for expression in expressions)
        ]
        return [index for index, _ in found], [name for _, name in found]

    def resolve_env_ids(self, ids: Any) -> Any:
        return self._resolve_ids(ids, self.num_instances)

    def resolve_joint_ids(self, ids: Any) -> Any:
        return self._resolve_ids(ids, self.num_joints)

    def _resolve_ids(self, ids: Any, count: int) -> Any:
        if ids is None:
            return self._wp.array(list(range(count)), dtype=self._wp.int32, device=self.device)
        if isinstance(ids, (self._torch.Tensor, self._wp.array)):
            return ids
        return self._wp.array(list(ids), dtype=self._wp.int32, device=self.device)

    def resolve_env_mask(self, mask: Any) -> Any:
        return (
            mask
            if mask is not None
            else self._wp.array([True] * self.num_instances, dtype=self._wp.bool, device=self.device)
        )

    def resolve_joint_mask(self, mask: Any) -> Any:
        return (
            mask
            if mask is not None
            else self._wp.array([True] * self.num_joints, dtype=self._wp.bool, device=self.device)
        )

    def assert_shape_and_dtype(self, value: Any, shape: tuple[int, ...], dtype: Any, name: str) -> None:
        del dtype, name
        if isinstance(value, (float, int)):
            return
        if tuple(value.shape) != shape:
            raise ValueError("benchmark command shape mismatch")

    def assert_shape_and_dtype_mask(self, value: Any, masks: tuple[Any, ...], dtype: Any, name: str) -> None:
        self.assert_shape_and_dtype(value, tuple(mask.shape[0] for mask in masks), dtype, name)

    def _properties(self, count: int, source_rows: int | None = None) -> Any:
        from isaaclab.actuators.actuator_control import ActuatorJointProperties

        rows = source_rows or self.num_instances
        zeros = self._torch.zeros((rows, count), dtype=self._torch.float32, device=self.device)
        if self._num_sources > 1:
            source_values = self._torch.arange(2, 2 + self._num_sources, dtype=self._torch.float32, device=self.device)
            zeros[:, :] = source_values[self._torch.arange(rows, device=self.device) % self._num_sources].unsqueeze(1)
        return ActuatorJointProperties(
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            self._torch.full_like(zeros, 100.0),
            self._torch.full_like(zeros, 30.0),
        )

    def get_default_joint_properties(self, joint_ids: Any) -> Any:
        return self._properties(self.num_joints if isinstance(joint_ids, slice) else joint_ids.shape[0])

    def get_source_joint_properties(self, joint_ids: Any, source_env_ids: Any) -> Any:
        return self._properties(joint_ids.shape[0], source_env_ids.shape[0])

    def prepare_native_actuators(self, collection: Any, cfgs: Any = None) -> set[str]:
        del collection, cfgs
        return set()

    def finalize_native_actuators(self, collection: Any) -> None:
        del collection

    def write_resolved_joint_properties(self, actuator: Any, *, native_managed: bool) -> None:
        del actuator, native_managed

    def compute_native_actuators(self, collection: Any, dt: float) -> bool:
        del collection, dt
        return False

    def reset_native_actuators(self, env_ids: Any) -> None:
        del env_ids

    def stage_user_command(self, command_name: str, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.command_stages.append(command_name)

    def submit_commands(self, owner: Any) -> None:
        del owner
        self.submissions += 1

    def discover_native_actuators(self, cfgs: Any) -> set[str]:
        del cfgs
        return set()

    def write_resolved_joint_properties_staged(self, properties: Any) -> None:
        del properties

    def validate_resolved_joint_properties(self) -> None:
        pass

    def restore_resolved_joint_properties(self) -> None:
        pass

    def commit_resolved_joint_properties(self) -> None:
        pass

    def prepare_actuator_binding(self, binding: Any) -> None:
        del binding

    def bind_actuator_view(self, view: Any) -> None:
        del view

    def complete_articulation_initialization(self) -> None:
        pass

    def invalidate_actuator_view(self) -> None:
        pass

    def invalidate_actuator_graphs(self) -> None:
        pass


def _group_cfgs(workload: _Workload) -> dict[str, Any]:
    """Build ordered real config objects with non-overlapping joint ownership."""
    from isaaclab.actuators.actuator_net_cfg import ActuatorNetMLPCfg
    from isaaclab.actuators.actuator_pd_cfg import (
        DCMotorCfg,
        DelayedPDActuatorCfg,
        IdealPDActuatorCfg,
        ImplicitActuatorCfg,
        RemotizedPDActuatorCfg,
    )

    types = workload.row.actuator_types
    configs: dict[str, Any] = {}
    for index in range(workload.row.groups):
        actuator_type = types[index % len(types)]
        cfg_type = {
            "implicit": ImplicitActuatorCfg,
            "ideal_pd": IdealPDActuatorCfg,
            "dc_motor": DCMotorCfg,
            "neural": ActuatorNetMLPCfg,
            "delayed": DelayedPDActuatorCfg,
            "remotized": RemotizedPDActuatorCfg,
            "opaque": IdealPDActuatorCfg,
        }.get(actuator_type)
        if cfg_type is None:
            raise RuntimeError(f"unsupported real fixture actuator type: {actuator_type}")
        values: dict[str, Any] = {
            "joint_names_expr": [workload.joint_names[index]],
            "stiffness": 2.0 + index,
            "damping": 0.5,
            "effort_limit": 100.0,
            "velocity_limit": 30.0,
        }
        if workload.row.case == "B3":
            values.update(stiffness=None, damping=None, effort_limit=None, velocity_limit=None)
        if actuator_type == "dc_motor":
            values["saturation_effort"] = 100.0
        if actuator_type == "neural":
            values.update(
                saturation_effort=100.0,
                network_file=workload.network_file,
                pos_scale=1.0,
                vel_scale=1.0,
                torque_scale=1.0,
                input_order="pos_vel",
                input_idx=(0,),
            )
        if actuator_type == "delayed":
            values.update(min_delay=1, max_delay=1)
        if actuator_type == "remotized":
            values.update(
                min_delay=1,
                max_delay=1,
                joint_parameter_lookup=[[-1.0, 1.0, 20.0], [0.0, 1.0, 20.0], [1.0, 1.0, 20.0]],
            )
        if actuator_type == "opaque":
            values["class_type"] = _opaque_actuator_type()
        configs[f"group_{index}"] = cfg_type(**values)
    return configs


def _opaque_actuator_type() -> type:
    """Return the one driver-owned exact subclass used for opaque fallback."""
    global _OPAQUE_ACTUATOR_TYPE
    if _OPAQUE_ACTUATOR_TYPE is None:
        from isaaclab.actuators.actuator_pd import IdealPDActuator

        class _OpaqueIdealPD(IdealPDActuator):
            """Driver-owned exact subclass used to retain the eager fallback boundary."""

        _OPAQUE_ACTUATOR_TYPE = _OpaqueIdealPD
    return _OPAQUE_ACTUATOR_TYPE


def _tiny_mlp_checkpoint(device: str) -> str:
    """Create one deterministic local TorchScript checkpoint without network access."""
    import torch

    class _TinyMLP(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return 2.0 * values[:, :1] + 3.0 * values[:, 1:2]

    file_descriptor, path = tempfile.mkstemp(prefix="isaaclab-actuator-benchmark-", suffix=".pt")
    os.close(file_descriptor)
    try:
        module = torch.jit.trace(_TinyMLP().eval(), torch.zeros((1, 2), device=device))
        module.save(path)
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    return path


class _DevelopAdapter(_MemoryAdapter):
    """Direct historical execution adapter using real ActuatorBase.compute calls."""

    def build_workload(self, workload: _Workload) -> None:
        import torch

        from isaaclab.actuators.actuator_net import ActuatorNetMLP
        from isaaclab.actuators.actuator_pd import (
            DCMotor,
            DelayedPDActuator,
            IdealPDActuator,
            ImplicitActuator,
            RemotizedPDActuator,
        )

        self.workload = workload
        self.groups = []
        self._direct_zero = torch.zeros((workload.row.num_worlds, 1), device=workload.device)
        self._direct_position = torch.full(
            (workload.row.num_worlds, 1), workload.first_command[0], device=workload.device
        )
        self._reset_env_ids = torch.arange(workload.row.num_worlds, device=workload.device)
        from isaaclab.utils.types import ArticulationActions

        self._direct_actions = [
            ArticulationActions(
                joint_positions=self._direct_position,
                joint_velocities=self._direct_zero,
                joint_efforts=self._direct_zero,
            )
            for _ in range(workload.row.groups)
        ]
        concrete = {
            "implicit": ImplicitActuator,
            "ideal_pd": IdealPDActuator,
            "dc_motor": DCMotor,
            "neural": ActuatorNetMLP,
            "delayed": DelayedPDActuator,
            "remotized": RemotizedPDActuator,
            "opaque": None,
        }
        for name, cfg in _group_cfgs(workload).items():
            index = int(name.rsplit("_", 1)[1])
            group_type = concrete[workload.row.actuator_types[index % len(workload.row.actuator_types)]]
            if group_type is None:
                group_type = cfg.class_type
            defaults = torch.zeros((workload.row.num_worlds, 1), device=workload.device)
            if workload.row.num_sources > 1:
                defaults[:, 0] = torch.arange(2, 2 + workload.row.num_sources, device=workload.device)[
                    torch.arange(workload.row.num_worlds, device=workload.device) % workload.row.num_sources
                ]
            self.groups.append(
                group_type(
                    cfg,
                    [workload.joint_names[index]],
                    slice(None),
                    workload.row.num_worlds,
                    workload.device,
                    stiffness=defaults,
                    damping=defaults,
                    armature=defaults,
                    friction=defaults,
                    dynamic_friction=defaults,
                    viscous_friction=defaults,
                    effort_limit=torch.full_like(defaults, 100.0),
                    velocity_limit=torch.full_like(defaults, 30.0),
                )
            )
        for group in self.groups:
            group.reset(self._reset_env_ids)

    def _apply(self) -> None:
        workload = self.workload
        if workload is None:
            raise RuntimeError("direct adapter workload is not built")
        for group, action in zip(self.groups, self._direct_actions):
            action.joint_positions = self._direct_position
            action.joint_velocities = self._direct_zero
            action.joint_efforts = self._direct_zero
            group.compute(
                action,
                self._direct_zero,
                self._direct_zero,
            )

    def first_application(self, workload: _Workload) -> None:
        self._apply()
        self.applications += 1

    def run_execution(self, count: int) -> None:
        for _ in range(count):
            self._apply()
        self.applications += count


class _CurrentPrAdapter(_MemoryAdapter):
    """Articulation-local collection adapter with a driver-owned control double."""

    def build_workload(self, workload: _Workload) -> None:
        from isaaclab.actuators.actuator_collection import ActuatorCollection

        self.workload = workload
        self.control = _DriverControl(
            workload.device, workload.row.num_worlds, workload.joint_names, workload.row.num_sources
        )
        import torch

        self._reset_env_ids = torch.arange(workload.row.num_worlds, device=workload.device)
        self.collection = ActuatorCollection(_group_cfgs(workload), self.control)
        self.collection.reset(self._reset_env_ids)

    def first_application(self, workload: _Workload) -> None:
        import torch

        value = torch.full(
            (workload.row.num_worlds, workload.row.groups), workload.first_command[0], device=workload.device
        )
        self.collection.command.set_position_index(value=value)
        self.collection.command.set_velocity_index(value=torch.zeros_like(value))
        self.collection.command.set_effort_index(value=torch.zeros_like(value))
        self.collection.compute()
        self.collection.submit_commands()
        self.applications += 1

    def run_execution(self, count: int) -> None:
        for _ in range(count):
            self.collection.compute()
            self.collection.submit_commands()
        self.applications += count


class _GlobalCollectionAdapter(_MemoryAdapter):
    """Simulation-scoped manager adapter; it never uses the legacy local constructor."""

    def build_workload(self, workload: _Workload) -> None:
        import torch

        from isaaclab.actuators.actuator_collection import ActuatorCollection
        from isaaclab.cloner import ClonePlan

        class _Simulation:
            def __init__(self) -> None:
                sources = tuple(f"/World/envs/env_{index}" for index in range(workload.row.num_sources))
                clone_mask = torch.zeros(
                    (workload.row.num_sources, workload.row.num_worlds), dtype=torch.bool, device=workload.device
                )
                columns = torch.arange(workload.row.num_worlds, device=workload.device)
                clone_mask[columns % workload.row.num_sources, columns] = True
                self.plan = ClonePlan(
                    sources=sources,
                    destinations=("/World/envs/env_{}",),
                    clone_mask=clone_mask,
                    cfg_rows={
                        index + 1: tuple(range(workload.row.num_sources))
                        for index in range(workload.row.num_articulations)
                    },
                )

            def get_clone_plan(self) -> Any:
                return self.plan

        self.workload = workload
        self._cfgs = _group_cfgs(workload)
        self._introspector = _GlobalIntrospector()
        self._projection_launches = 0
        manager_started = time.perf_counter_ns()
        self.manager = ActuatorCollection(_Simulation())
        manager_finished = time.perf_counter_ns()
        self.build_decomposition_ms = {
            "manager_construction": (manager_finished - manager_started) / 1_000_000,
        }
        self._register_generation(workload)

    def _register_generation(self, workload: _Workload) -> None:
        """Register and publish one generation on the existing manager."""
        import torch

        control_started = time.perf_counter_ns()
        self.controls = [
            _DriverControl(workload.device, workload.row.num_worlds, workload.joint_names, workload.row.num_sources)
            for _ in range(workload.row.num_articulations)
        ]
        control_finished = time.perf_counter_ns()
        registration_started = time.perf_counter_ns()
        self.views = [
            self.manager.register_articulation(
                key=f"benchmark-{index}",
                cfgs=self._cfgs,
                control=control,
                replication_cfg_id=index + 1,
                debug_validation=False,
                debug_value_resolution=False,
            )
            for index, control in enumerate(self.controls)
        ]
        registration_finished = time.perf_counter_ns()
        self.control = self.controls[0]
        self.view = self.views[0]
        finalization_started = time.perf_counter_ns()
        self.manager.finalize()
        finalization_finished = time.perf_counter_ns()
        if not self.manager.is_finalized or not self.view.is_ready or self.view._execution_plan is None:
            raise RuntimeError("global collection lifecycle probe failed")
        reset_started = time.perf_counter_ns()
        self._reset_env_ids = torch.arange(workload.row.num_worlds, device=workload.device)
        for view in self.views:
            view.reset(self._reset_env_ids)
        reset_finished = time.perf_counter_ns()
        self.build_decomposition_ms.update(
            {
                "control_construction": (control_finished - control_started) / 1_000_000,
                "registration": (registration_finished - registration_started) / 1_000_000,
                "finalization": (finalization_finished - finalization_started) / 1_000_000,
                "state_reset": (reset_finished - reset_started) / 1_000_000,
            }
        )

    def first_application(self, workload: _Workload) -> None:
        import torch

        value = torch.full(
            (workload.row.num_worlds, workload.row.groups), workload.first_command[0], device=workload.device
        )
        zeros = torch.zeros_like(value)
        for view in self.views:
            view.command.set_position_index(value=value)
            view.command.set_velocity_index(value=zeros)
            view.command.set_effort_index(value=zeros)
            view.compute()
            view.submit_commands()
        self.applications += 1

    def run_execution(self, count: int) -> None:
        for _ in range(count):
            for view in self.views:
                view.compute()
                view.submit_commands()
        self.applications += count

    def warmup_execution(self, row: RuntimeRow) -> bool:
        """Capture the scoped plan and accept only a live graph."""
        if row.requested_execution != "graph":
            return True
        plan = self.view._execution_plan
        if plan is None:
            return False
        plan.warmup_and_capture()
        return plan._full_graph is not None or plan._prefix_graph is not None

    def introspect(self) -> dict[str, Any] | None:
        """Return allocation ownership from the manager's live generation only."""
        generation = getattr(self.manager, "_active_generation", None)
        return self._introspector.inspect(
            generation,
            manager=self.manager,
            projection_launches=self._projection_launches,
        )

    def close(self) -> None:
        if hasattr(self, "manager"):
            self.manager.close()
        for cfg in getattr(self, "_cfgs", {}).values():
            if (path := getattr(cfg, "network_file", None)) is not None:
                Path(path).unlink(missing_ok=True)
        super().close()


class _CompatibilityLaunchCounter:
    """Count actual compatibility kernel dispatches through one plan's launch cache."""

    def __init__(self, plan: Any) -> None:
        self.plan = plan
        self.count = 0
        self._original: Any = None

    def __enter__(self) -> _CompatibilityLaunchCounter:
        launch_cache = self.plan._launch_cache
        self._original = launch_cache.launch

        def wrapped(key: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(key, tuple) and key and str(key[0]).startswith("compatibility_"):
                self.count += 1
            return self._original(key, *args, **kwargs)

        launch_cache.launch = wrapped
        return self

    def __exit__(self, *_: Any) -> None:
        self.plan._launch_cache.launch = self._original


def run_global_structural_case(workload: _Workload) -> dict[str, Any]:
    """Run one global-only lifecycle/projection observation without timing it."""
    if workload.row.case == "B0":
        from isaaclab.actuators.actuator_collection import ActuatorCollection

        class _EmptySimulation:
            def get_clone_plan(self) -> None:
                return None

        manager = ActuatorCollection(_EmptySimulation())
        manager.finalize()
        manager.clear_generation()
        clear_report = _GlobalIntrospector().inspect(None)
        manager.close()
        manager.close()
        try:
            manager.register_articulation(
                key="late",
                cfgs={},
                control=None,
                replication_cfg_id=1,
                debug_validation=False,
                debug_value_resolution=False,
            )
        except RuntimeError:
            registration_rejected = True
        else:
            registration_rejected = False
        return {
            "cleared": True,
            "manager_closed": manager._closed,
            "registration_rejected": registration_rejected,
            "clear_state_ownership": clear_report["clear_state_ownership"],
        }

    adapter = _GlobalCollectionAdapter("global", workload.device)
    try:
        adapter.build_workload(workload)
        if workload.row.case == "B2":
            adapter.first_application(workload)
            applied_facades = tuple(
                key
                for key, control in zip(adapter.manager.registration_keys, adapter.controls, strict=True)
                if control.submissions == 1
            )
            return {
                "articulation_count": len(adapter.manager.registration_keys),
                "applied_facades": applied_facades,
                "submission_counts": tuple(control.submissions for control in adapter.controls),
            }
        if workload.row.case == "B6":
            states = ["untouched"]
            bytes_by_state = {"untouched": adapter.introspect()["projection_bytes"]}
            plan = adapter.view._execution_plan
            if plan is None:
                raise RuntimeError("B6 requires a live execution plan")
            with _CompatibilityLaunchCounter(plan) as launches:
                first = adapter.view._get_compatibility_projection("soft_joint_vel_limits")
                states.append("first")
                bytes_by_state["first"] = adapter.introspect()["projection_bytes"]
                first_pointer = (str(first.warp.device), int(first.warp.ptr))
                allocation_probe = _begin_repeat_allocation_probe(workload.device)
                repeated = adapter.view._get_compatibility_projection("soft_joint_vel_limits")
                allocator = _end_repeat_allocation_probe(allocation_probe)
                states.append("repeated")
                bytes_by_state["repeated"] = adapter.introspect()["projection_bytes"]
                repeated_pointer = (str(repeated.warp.device), int(repeated.warp.ptr))
                adapter.view._get_compatibility_projection("gear_ratio")
                states.append("both")
                bytes_by_state["both"] = adapter.introspect()["projection_bytes"]
                adapter.view.compute()
            adapter._projection_launches = launches.count
            report = adapter.introspect()
            retained_owner_delta = bytes_by_state["repeated"] - bytes_by_state["first"]
            allocation_free = (
                None
                if allocator["delta_bytes"] is None
                else retained_owner_delta == 0
                and first_pointer == repeated_pointer
                and allocator["delta_bytes"] == 0
                and allocator["peak_delta_bytes"] == 0
            )
            return {
                "projection_states": tuple(states),
                "projection_count": len(getattr(plan, "_compatibility_projection_refreshes", {})),
                "projection_launches": launches.count,
                "projection_bytes": report["projection_bytes"],
                "projection_bytes_by_state": bytes_by_state,
                "repeated_pointer_stable": first_pointer == repeated_pointer,
                "repeat_retained_owner_delta_bytes": retained_owner_delta,
                "repeat_allocator_observation": allocator["observation"],
                "repeat_allocator_delta_bytes": allocator["delta_bytes"],
                "repeat_allocator_peak_delta_bytes": allocator["peak_delta_bytes"],
                "repeat_allocation_free": allocation_free,
                "pointer_replacements": report["pointer_replacements"],
            }
        if workload.row.case == "B8":
            old_view = adapter.view
            manager = adapter.manager
            adapter.introspect()
            adapter.manager.clear_generation()
            clear_report = adapter.introspect()
            try:
                old_view.compute()
            except RuntimeError:
                stale = True
            else:
                stale = False
            adapter._register_generation(workload)
            return {
                "re_registered": stale and adapter.manager.is_finalized,
                "same_manager": adapter.manager is manager,
                "old_view_stale": stale,
                "clear_state_ownership": clear_report["clear_state_ownership"],
            }
        raise ValueError(f"not a global structural case: {workload.row.case}")
    finally:
        adapter.close()


def _import_actuator_collection() -> Any:
    from isaaclab.actuators.actuator_collection import ActuatorCollection

    return ActuatorCollection


def select_adapter(revision: str, device: str) -> _Adapter | RevisionCapability:
    """Select only a feature-compatible adapter; never silently fall back."""
    _require_revision(revision)
    collection_spec = importlib.util.find_spec("isaaclab.actuators.actuator_collection")
    if collection_spec is None:
        if revision != "develop":
            return RevisionCapability(False, "actuator_collection unavailable for requested revision")
        return _DevelopAdapter(revision, device)
    try:
        collection = _import_actuator_collection()
    except Exception as error:
        return RevisionCapability(False, f"actuator collection import unavailable: {type(error).__name__}")
    global_required = ("register_articulation", "finalize", "clear_generation", "close")
    is_global = all(callable(getattr(collection, name, None)) for name in global_required)
    is_current = not callable(getattr(collection, "register_articulation", None)) and callable(
        getattr(collection, "_build_execution_batches", None)
    )
    if revision == "develop":
        return RevisionCapability(False, "requested develop has an actuator collection surface")
    if revision == "current" and is_current:
        return _CurrentPrAdapter(revision, device)
    if revision == "global" and is_global:
        return _GlobalCollectionAdapter(revision, device)
    return RevisionCapability(False, "unrecognized required actuator collection feature set")


def _pointer_stability_record(report: dict[str, Any] | None) -> dict[str, Any]:
    """Extract pointer-stability evidence from one structural report."""
    if report is None or "pointer_snapshot_count" not in report:
        return {
            "observation": "unavailable",
            "pointer_replacements": None,
            "pointer_snapshot_count": None,
        }
    return {
        "observation": "global_introspection",
        "pointer_replacements": report["pointer_replacements"],
        "pointer_snapshot_count": report["pointer_snapshot_count"],
    }


def _pointer_stability_snapshot(adapter: _Adapter) -> dict[str, Any]:
    """Take one untimed global ownership snapshot, or report that it is unavailable."""
    return _pointer_stability_record(adapter.introspect())


def _synchronize_boundary(device: str) -> None:
    """Synchronize a completed CUDA construction boundary."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def measure_build(
    revision: str,
    row: BuildRow,
    device: str,
    phase: str,
    *,
    adapter_factory: Any = None,
    workload_factory: Any = make_workload,
    warmup_constructions: int = 10,
    measured_constructions: int = 100,
) -> dict[str, Any]:
    """Measure fresh resolved-construction-to-first-application boundaries."""
    if phase not in {"cold", "warm"}:
        raise ValueError(f"unknown build phase: {phase}")
    adapter_factory = select_adapter if adapter_factory is None else adapter_factory
    warmup_count = 0 if phase == "cold" else warmup_constructions
    measured_count = 1 if phase == "cold" else measured_constructions
    samples_ms: list[float] = []
    decomposition_samples: dict[str, list[float]] = {}
    last_structural: dict[str, Any] | None = None
    last_pointer_stability = {
        "observation": "unavailable",
        "pointer_replacements": None,
        "pointer_snapshot_count": None,
    }
    adapter_name: str | None = None

    def construct_once(*, measured: bool) -> None:
        nonlocal adapter_name, last_pointer_stability, last_structural
        workload: _Workload | None = workload_factory(row, device)
        adapter: _Adapter | None = None
        try:
            selected = adapter_factory(revision, device)
            if isinstance(selected, RevisionCapability):
                raise RuntimeError(selected.reason)
            adapter = selected
            adapter_name = type(adapter).__name__
            started = time.perf_counter_ns() if measured else None
            adapter.build_workload(workload)
            post_finalize = _pointer_stability_snapshot(adapter)
            adapter.first_application(workload)
            _synchronize_boundary(device)
            if measured:
                assert started is not None
                samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            last_structural = adapter.introspect()
            last_pointer_stability = _pointer_stability_record(last_structural)
            if measured:
                for name, elapsed_ms in getattr(adapter, "build_decomposition_ms", {}).items():
                    decomposition_samples.setdefault(name, []).append(float(elapsed_ms))
            if last_pointer_stability["observation"] == "unavailable":
                last_pointer_stability = post_finalize
        finally:
            try:
                if adapter is not None:
                    adapter.close()
            finally:
                _cleanup_workload(workload)

    for _ in range(warmup_count):
        construct_once(measured=False)
    for _ in range(measured_count):
        construct_once(measured=True)

    total_ms = sum(samples_ms)
    return {
        "status": "accepted",
        "adapter_name": adapter_name,
        "timing": {
            "samples_ms": samples_ms,
            "total_ms": total_ms,
            "per_construction_ms": total_ms / measured_count,
            "construction_count": measured_count,
            "warmup_construction_count": warmup_count,
            "first_application_count": measured_count,
        },
        "counters": {
            "global_decomposition_samples_ms": decomposition_samples or None,
            "pointer_stability": last_pointer_stability,
        },
        "structural": last_structural,
    }


def measure_runtime(adapter: _Adapter, row: RuntimeRow, warmups: int, iterations: int) -> dict[str, Any]:
    """Measure one runtime mode without relabelling failed graph capture."""
    import warp as wp

    pointer_stability = _pointer_stability_snapshot(adapter)
    capture = _ScopedInstrumentation(wp)
    if row.requested_execution == "graph":
        with capture:
            captured = adapter.warmup_execution(row)
        if not captured:
            return {
                "status": "rejected",
                "requested_execution": "graph",
                "effective_execution": None,
                "reason": "graph capture failed",
                "counters": {
                    "capture": capture.as_record(),
                    "warmup": None,
                    "transfer_probe": None,
                    "replay": None,
                    "pointer_stability": _pointer_stability_snapshot(adapter),
                },
            }
    warmup = _ScopedInstrumentation(wp)
    with warmup:
        adapter.run_execution(warmups)
    pointer_stability = _pointer_stability_snapshot(adapter)
    transfer_probe = _observe_transfer_replay(adapter, wp)
    pointer_stability = _pointer_stability_snapshot(adapter)
    allocation_before = _steady_allocation_bytes(getattr(adapter, "device", "cpu"))
    replay = _ScopedInstrumentation(wp)
    with replay:
        elapsed_ms = _time_runtime_execution(adapter, iterations)
    allocation_after = _steady_allocation_bytes(getattr(adapter, "device", "cpu"))
    pointer_stability = _pointer_stability_snapshot(adapter)
    per_application_ms = elapsed_ms / iterations
    return {
        "status": "accepted",
        "requested_execution": row.requested_execution,
        "effective_execution": row.effective_execution,
        "timing": {
            "samples_ms": [per_application_ms],
            "total_ms": elapsed_ms,
            "per_application_ms": per_application_ms,
            "application_count": iterations,
        },
        "counters": {
            "capture": capture.as_record(),
            "warmup": warmup.as_record(),
            "transfer_probe": transfer_probe,
            "replay": replay.as_record(),
            "steady_allocation_delta_bytes": allocation_after - allocation_before,
            "pointer_stability": pointer_stability,
        },
    }


def _time_runtime_execution(adapter: _Adapter, iterations: int) -> float:
    """Time actual replay with CUDA events, or use a CPU-only smoke fallback."""
    if getattr(adapter, "device", "cpu").startswith("cuda"):
        import torch

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        adapter.run_execution(iterations)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)
    started = time.perf_counter_ns()
    adapter.run_execution(iterations)
    return (time.perf_counter_ns() - started) / 1_000_000


def _steady_allocation_bytes(device: str) -> int:
    """Read the existing Torch allocation counter without forcing synchronization."""
    if not device.startswith("cuda"):
        return 0
    import torch

    return torch.cuda.memory_allocated(device)


def _begin_repeat_allocation_probe(device: str) -> dict[str, Any]:
    """Start a CUDA allocator/peak probe, or state why it is unavailable."""
    if not device.startswith("cuda"):
        return {"observation": "unavailable_cpu", "allocated_before": None}
    import torch

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "observation": "torch_cuda",
        "allocated_before": torch.cuda.memory_allocated(device),
        "device": device,
    }


def _end_repeat_allocation_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Finish a CUDA allocator/peak probe without fabricating CPU values."""
    if probe["observation"] != "torch_cuda":
        return {"observation": probe["observation"], "delta_bytes": None, "peak_delta_bytes": None}
    import torch

    device = probe["device"]
    torch.cuda.synchronize(device)
    allocated_before = probe["allocated_before"]
    return {
        "observation": "torch_cuda",
        "delta_bytes": torch.cuda.memory_allocated(device) - allocated_before,
        "peak_delta_bytes": torch.cuda.max_memory_allocated(device) - allocated_before,
    }


class _TorchTransferLedger:
    """Record observed Torch transfers and synchronizing CUDA readbacks."""

    def __init__(self) -> None:
        self.h2d_bytes = 0
        self.d2h_sync_count = 0

    def record_transfer(self, dest: Any, src: Any, *, synchronizing: bool) -> None:
        """Record one cross-device tensor transfer."""
        dest_is_cuda = self._is_cuda(dest)
        src_is_cuda = self._is_cuda(src)
        if dest_is_cuda == src_is_cuda:
            return
        if dest_is_cuda:
            self.h2d_bytes += self._nbytes(src)
        elif synchronizing:
            self.d2h_sync_count += 1

    def record_readback(self, value: Any, *, final_timing_sync: bool = False) -> None:
        """Record one scalar CUDA readback unless it belongs to the timing harness."""
        if not final_timing_sync and self._is_cuda(value):
            self.d2h_sync_count += 1

    def as_record(self) -> dict[str, Any]:
        return {
            "observation": "torch_dispatch",
            "h2d_bytes": self.h2d_bytes,
            "d2h_sync_count": self.d2h_sync_count,
        }

    @staticmethod
    def _is_cuda(value: Any) -> bool:
        return str(getattr(value, "device", "cpu")).startswith("cuda")

    @staticmethod
    def _nbytes(value: Any) -> int:
        nbytes = getattr(value, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
        shape = getattr(value, "shape", ())
        element_size = getattr(value, "element_size", None)
        return int(math.prod(shape)) * (int(element_size()) if callable(element_size) else 0)


class _ScopedInstrumentation:
    """Temporarily observe explicit Warp launch/copy sites in a measured scope."""

    def __init__(self, warp: Any) -> None:
        self.warp = warp
        self._originals: dict[str, Any] = {}
        self.launches: dict[str, int] = {}
        self.h2d_bytes = 0
        self.d2h_copies = 0
        self.d2h_sync_count = 0

    def __enter__(self) -> _ScopedInstrumentation:
        if self.warp is None:
            return self
        for name in ("launch", "launch_tiled", "copy", "capture_launch"):
            original = getattr(self.warp, name, None)
            if original is None:
                continue
            self._originals[name] = original

            def wrapped(*args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
                self.launches[_name] = self.launches.get(_name, 0) + 1
                if _name == "copy":
                    self._record_copy(args, kwargs)
                return _original(*args, **kwargs)

            setattr(self.warp, name, wrapped)
        return self

    def __exit__(self, *_: Any) -> None:
        for name, original in self._originals.items():
            setattr(self.warp, name, original)

    def record_readback(self, final_timing_sync: bool = False) -> None:
        if not final_timing_sync:
            self.d2h_sync_count += 1

    def _record_copy(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Record direction and size for one explicit Warp copy."""
        dest = args[0] if args else kwargs.get("dest")
        src = args[1] if len(args) > 1 else kwargs.get("src")
        if dest is None or src is None:
            return
        dest_is_cuda = str(getattr(dest, "device", "cpu")).startswith("cuda")
        src_is_cuda = str(getattr(src, "device", "cpu")).startswith("cuda")
        if dest_is_cuda == src_is_cuda:
            return
        if not src_is_cuda and dest_is_cuda:
            self.h2d_bytes += self._copy_nbytes(src, kwargs.get("count", args[2] if len(args) > 2 else None))
        elif src_is_cuda and not dest_is_cuda:
            self.d2h_copies += 1

    @staticmethod
    def _copy_nbytes(value: Any, count: int | None) -> int:
        if count is None and (nbytes := getattr(value, "nbytes", None)) is not None:
            return int(nbytes)
        if count is None:
            count = math.prod(getattr(value, "shape", (0,)))
        try:
            import warp as wp

            item_size = wp.types.type_size_in_bytes(value.dtype)
        except Exception:
            item_size = 0
        return int(count) * int(item_size)

    def as_record(self, torch_transfer: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return only benchmark-side launch/readback observations for this scope."""
        torch_available = torch_transfer is not None and torch_transfer.get("observation") == "torch_dispatch"
        return {
            "launches": dict(self.launches),
            "h2d_bytes": self.h2d_bytes + int(torch_transfer["h2d_bytes"]) if torch_available else None,
            "d2h_sync_count": (
                self.d2h_sync_count + int(torch_transfer["d2h_sync_count"]) if torch_available else None
            ),
            "warp_h2d_bytes": self.h2d_bytes,
            "warp_d2h_copies": self.d2h_copies,
            "warp_d2h_sync_count": self.d2h_sync_count,
            "torch_transfer_observation": "torch_dispatch" if torch_available else "unavailable",
        }


def _observe_transfer_replay(adapter: _Adapter, warp: Any) -> dict[str, Any]:
    """Observe one representative replay outside graph capture and timing scopes."""
    scoped = _ScopedInstrumentation(warp)
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
    except (AttributeError, ImportError):
        with scoped:
            adapter.run_execution(1)
        return scoped.as_record()

    ledger = _TorchTransferLedger()

    class _TransferMode(TorchDispatchMode):
        def __torch_dispatch__(self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: Any = None) -> Any:
            del types
            kwargs = {} if kwargs is None else kwargs
            result = func(*args, **kwargs)
            operator = str(func)
            if "_to_copy" in operator and args:
                ledger.record_transfer(
                    result,
                    args[0],
                    synchronizing=not bool(kwargs.get("non_blocking", False)),
                )
            elif "copy_" in operator and len(args) >= 2:
                non_blocking = kwargs.get("non_blocking", args[2] if len(args) > 2 else False)
                ledger.record_transfer(args[0], args[1], synchronizing=not bool(non_blocking))
            elif "_local_scalar_dense" in operator and args:
                ledger.record_readback(args[0])
            return result

    with scoped, _TransferMode():
        adapter.run_execution(1)
    return scoped.as_record(ledger.as_record())


def observe_runtime_scopes(adapter: _Adapter, iterations: int) -> list[str]:
    """Keep capture and replay observation boundaries distinct."""
    scopes = ["capture"]
    adapter.warmup_execution(RuntimeRow("implicit", 1, "graph", "graph"))
    scopes.append("replay")
    adapter.run_execution(iterations)
    return scopes


class _GlobalIntrospector:
    """Read actual finalized generation owners without analytical layout guesses."""

    _DESCRIPTOR_CATEGORIES = (
        "registration",
        "resolved_group",
        "binding",
        "store",
        "articulation_binding",
        "view",
        "execution_plan",
        "execution_range",
        "eager_segment",
    )

    def __init__(self) -> None:
        self._previous_pointers: dict[str, tuple[str, int]] = {}
        self._pointer_replacements = 0
        self._pointer_snapshot_count = 0

    def inspect(
        self,
        generation: Any,
        *,
        manager: Any = None,
        projection_launches: int = 0,
    ) -> dict[str, Any]:
        if generation is None:
            self._previous_pointers.clear()
            self._pointer_replacements = 0
            self._pointer_snapshot_count = 0
            return {
                "canonical_allocation_count": 0,
                "canonical_allocation_bytes": 0,
                "storage_wrapper_count": 0,
                "python_descriptor_count": 0,
                "python_descriptor_counts": dict.fromkeys(self._DESCRIPTOR_CATEGORIES, 0),
                "plan_staging_owner_count": 0,
                "plan_staging_owner_bytes": 0,
                "projection_bytes": 0,
                "projection_launches": projection_launches,
                "pointer_replacements": 0,
                "pointer_snapshot_count": 0,
                "clear_state_ownership": 0,
            }

        stores = getattr(generation, "stores", {})
        store_items = tuple(stores.items()) if hasattr(stores, "items") else tuple(enumerate(stores))
        joint_store = getattr(generation, "joint_store", None)
        canonical_named = self._named_owners_from(
            [
                *[(f"stores[{getattr(key, '__qualname__', key)!s}]", store) for key, store in store_items],
                ("joint_store.fields", getattr(joint_store, "_fields", {})),
            ]
        )
        bindings = tuple(getattr(generation, "bindings", ()))
        staging_named = self._named_owners_from(
            [
                *[
                    (f"bindings[{index}].execution_plan", getattr(binding, "execution_plan", None))
                    for index, binding in enumerate(bindings)
                ],
                *[
                    (
                        f"bindings[{index}].backend_parameter_staging",
                        getattr(binding, "backend_parameter_staging", None),
                    )
                    for index, binding in enumerate(bindings)
                ],
                *[
                    (f"generation.backend_parameter_staging[{key!r}]", staging)
                    for key, staging in getattr(generation, "backend_parameter_staging", {}).items()
                ],
            ]
        )
        projection_named = self._named_owners_from(
            [("joint_store.compatibility_projections", getattr(joint_store, "_compatibility_projections", {}))]
        )

        canonical = self._deduplicate(canonical_named)
        staging = {key: owner for key, owner in self._deduplicate(staging_named).items() if key not in canonical}
        projection = {
            key: owner
            for key, owner in self._deduplicate(projection_named).items()
            if key not in canonical and key not in staging
        }
        current_pointers = {
            **{f"canonical:{path}": owner[0] for path, owner in canonical_named.items()},
            **{f"staging:{path}": owner[0] for path, owner in staging_named.items()},
            **{f"projection:{path}": owner[0] for path, owner in projection_named.items()},
        }
        pointer_replacements = sum(
            previous != pointer
            for path, pointer in current_pointers.items()
            if (previous := self._previous_pointers.get(path)) is not None
        )
        self._pointer_replacements += pointer_replacements
        self._pointer_snapshot_count += 1
        self._previous_pointers = current_pointers
        all_owners = {*canonical, *staging, *projection}
        storage_wrapper_ids = {
            owner[3] for named in (canonical_named, staging_named, projection_named) for owner in named.values()
        }
        python_descriptor_count, python_descriptor_counts = self._python_descriptor_inventory(generation, manager)
        return {
            "canonical_allocation_count": len(canonical),
            "canonical_allocation_bytes": sum(size for _, size in canonical.values()),
            "storage_wrapper_count": len(storage_wrapper_ids),
            "python_descriptor_count": python_descriptor_count,
            "python_descriptor_counts": python_descriptor_counts,
            "plan_staging_owner_count": len(staging),
            "plan_staging_owner_bytes": sum(size for _, size in staging.values()),
            "projection_bytes": sum(size for _, size in projection.values()),
            "projection_launches": projection_launches,
            "pointer_replacements": self._pointer_replacements,
            "pointer_snapshot_count": self._pointer_snapshot_count,
            "clear_state_ownership": len(all_owners),
        }

    @classmethod
    def _python_descriptor_inventory(cls, generation: Any, manager: Any) -> tuple[int, dict[str, int]]:
        """Count exact manager-owned Python descriptors by semantic category."""

        def values(container: Any) -> tuple[Any, ...]:
            if container is None:
                return ()
            if isinstance(container, Mapping):
                return tuple(container.values())
            return tuple(container)

        articulation_bindings = tuple(getattr(generation, "bindings", ()))
        registrations = values(getattr(manager, "_registrations", None)) if manager is not None else ()
        if not registrations:
            registrations = tuple(getattr(binding, "registration", None) for binding in articulation_bindings)
        nested_groups = values(getattr(generation, "groups", {}))
        resolved_groups = tuple(group for groups in nested_groups for group in values(groups))
        stores = values(getattr(generation, "stores", {}))
        views = values(getattr(manager, "_views", None)) if manager is not None else ()
        execution_plans = tuple(getattr(binding, "execution_plan", None) for binding in articulation_bindings)
        execution_ranges = tuple(
            execution_range
            for plan in execution_plans
            if plan is not None
            for execution_range in getattr(plan, "stateless_ranges", ())
        )
        eager_segments = tuple(
            segment for plan in execution_plans if plan is not None for segment in getattr(plan, "eager_segments", ())
        )
        binding_owners = (
            *resolved_groups,
            *(getattr(execution_range, "executor", None) for execution_range in execution_ranges),
            *(getattr(segment, "actuator", None) for segment in eager_segments),
        )
        group_bindings = tuple(
            getattr(owner, "__dict__", {}).get("_parameter_binding") for owner in binding_owners if owner is not None
        )
        category_values = {
            "registration": registrations,
            "resolved_group": resolved_groups,
            "binding": group_bindings,
            "store": stores,
            "articulation_binding": articulation_bindings,
            "view": views,
            "execution_plan": execution_plans,
            "execution_range": execution_ranges,
            "eager_segment": eager_segments,
        }
        ids_by_category = {
            category: {id(value) for value in category_values[category] if value is not None}
            for category in cls._DESCRIPTOR_CATEGORIES
        }
        all_ids = set().union(*ids_by_category.values())
        return len(all_ids), {category: len(ids_by_category[category]) for category in cls._DESCRIPTOR_CATEGORIES}

    @staticmethod
    def _deduplicate(
        named: dict[str, tuple[tuple[str, int], Any, int, int]],
    ) -> dict[tuple[str, int], tuple[Any, int]]:
        return {key: (owner, size) for key, owner, size, _ in named.values()}

    @staticmethod
    def _named_owners_from(roots: list[tuple[str, Any]]) -> dict[str, tuple[tuple[str, int], Any, int, int]]:
        """Collect named allocation leaves from real current-generation owners only."""
        found: dict[str, tuple[tuple[str, int], Any, int, int]] = {}
        visited: set[int] = set()

        def visit(value: Any, path: str) -> None:
            if value is None or id(value) in visited:
                return
            visited.add(id(value))
            raw = getattr(value, "warp", value)
            if (ptr := getattr(raw, "ptr", None)) is not None:
                nbytes = int(getattr(raw, "nbytes", 0))
                if nbytes == 0 and hasattr(raw, "shape") and hasattr(raw, "dtype"):
                    try:
                        import warp as wp

                        nbytes = math.prod(raw.shape) * wp.types.type_size_in_bytes(raw.dtype)
                    except Exception:
                        nbytes = 0
                key = (str(getattr(raw, "device", "unknown")), int(ptr))
                found[path] = (key, raw, nbytes, id(value))
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    visit(item, f"{path}[{key!r}]")
                return
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")
                return
            for name in (
                "_fields",
                "_targets",
                "_owner_slots",
                "staging",
                "stateless_ranges",
                "eager_segments",
                "static_scatter_epochs",
                "joint_indices",
                "owner_slots_by_field",
                "gather_inputs",
                "gather_outputs",
                "implicit_inputs",
                "implicit_outputs",
                "scatter_inputs",
                "scatter_outputs",
                "action_scatter_outputs",
                "telemetry_scatter_outputs",
            ):
                visit(getattr(value, name, None), f"{path}.{name}")

        for path, root in roots:
            visit(root, path)
        return found

    @classmethod
    def _owners_from(cls, roots: list[Any]) -> dict[tuple[str, int], tuple[Any, int]]:
        """Compatibility helper for literal owner probes."""
        return cls._deduplicate(cls._named_owners_from([(f"roots[{index}]", root) for index, root in enumerate(roots)]))


def prepare_harness(run_root: Path, source: Path) -> tuple[Path, str]:
    """Copy, digest, and make an immutable candidate driver for a final batch."""
    harness = run_root / "harness"
    harness.mkdir(parents=True, exist_ok=True)
    target = harness / "benchmark_actuator_collection.py"
    digest_path = harness / "benchmark_actuator_collection.sha256"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists() or digest_path.exists():
        if (
            not target.exists()
            or not digest_path.exists()
            or digest_path.read_text().strip() != digest
            or hashlib.sha256(target.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("immutable harness digest differs")
        return target, digest
    shutil.copyfile(source, target)
    digest_path.write_text(digest + "\n", encoding="utf-8")
    target.chmod(target.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    return target, digest


def balanced_pair_schedule(baseline: str, global_revision: str = "global") -> tuple[tuple[str, str], ...]:
    """Return three baseline-first then three global-first pair orders."""
    code = {"develop": "D", "current": "C", "global": "G"}
    return tuple(
        (
            f"{number:02}",
            f"{code[baseline]}-{code[global_revision]}" if number <= 3 else f"{code[global_revision]}-{code[baseline]}",
        )
        for number in range(1, 7)
    )


def validate_pair_telemetry(pre: list[TelemetrySample], post: list[TelemetrySample], device: str) -> list[str]:
    """Return GPU acceptance gate failures; CPU intentionally bypasses them."""
    if not device.startswith("cuda"):
        return []
    if len(pre) != 20 or len(post) != 20:
        return ["required telemetry unavailable"]
    reasons: list[str] = []
    if any(sample.utilization_pct is None or sample.temperature_c is None for sample in [*pre, *post]):
        reasons.append("required telemetry unavailable")
    if any((sample.utilization_pct or 0) >= 5 for sample in pre):
        reasons.append("pre-run utilization >= 5%")
    if any(sample.throttle_reasons for sample in [*pre, *post]):
        reasons.append("throttle reason")
    if any(sample.compute_pids for sample in [*pre, *post]):
        reasons.append("competing compute PID")
    if pre and post and abs((pre[0].temperature_c or 0) - (post[-1].temperature_c or 0)) > 5:
        reasons.append("temperature envelope")
    return reasons


class Coordinator:
    """Parent-only process coordinator; it does not import target actuator packages."""

    def __init__(self, run_root: Path, runner: Any = None) -> None:
        self.run_root = run_root
        self.runner = runner or self._run

    @staticmethod
    def _run(command: list[str]) -> dict[str, Any]:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

    def schedule_cold_children(
        self, rows: list[BuildRow] | tuple[BuildRow, ...], repetitions: int
    ) -> list[dict[str, Any]]:
        results = []
        for row in rows:
            for repetition in range(repetitions):
                results.append(
                    self.runner(
                        [
                            sys.executable,
                            "benchmark_actuator_collection.py",
                            "--child_row",
                            row_key(row),
                            "--phase",
                            "cold",
                            "--attempt_id",
                            f"attempt-{repetition + 1:02}",
                        ]
                    )
                )
        return results


def _smoke_record(
    args: argparse.Namespace,
    row: BuildRow,
    adapter: _Adapter | None,
    *,
    structural: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    counters: dict[str, Any] | None = None,
    adapter_name: str | None = None,
) -> dict[str, Any]:
    revision = args.revision or "global"
    identity = AttemptIdentity(
        args.batch_id,
        row_key(row),
        "attempt-01",
        args.candidate_sha or "cpu-smoke",
        {revision: args.revision_sha or "cpu-smoke"},
        args.harness_sha256 or "cpu-smoke",
    )
    if structural is None and adapter is not None:
        structural = adapter.introspect()
    first_application_count = getattr(adapter, "applications", 0)
    if adapter is None and row.case == "B2":
        first_application_count = 1
    timing = {"samples_ms": [], "first_application_count": first_application_count} if timing is None else timing
    return {
        "schema": SCHEMA,
        "identity": asdict(identity),
        "kind": "singleton",
        "status": "accepted",
        "boundary": "empty_finalize_clear" if row.case == "B0" else "resolved_construction_to_first_application",
        "telemetry": {"required": False, "available": True, "samples": [], "rejection_reasons": []},
        "members": [
            {
                "revision": revision,
                "requested_execution": "cached_eager",
                "effective_execution": "cached_eager",
                "revision_sha": args.revision_sha or "cpu-smoke",
                "adapter": adapter_name
                or (type(adapter).__name__ if adapter is not None else "_GlobalCollectionAdapter"),
                "resolved_row": asdict(row),
                "source_emulation": row.case == "B3" and revision != "global",
                "capability": {"supported": True, "reason": None},
                "timing": timing,
                "counters": counters or {},
                "structural": structural,
            }
        ],
        "paths": {"harness": None, "worktrees": {}, "cache": {}},
        "command": sys.argv,
        "device": args.device,
        "cache": {"policy": "private"},
        "process": {"returncode": 0},
        "metadata": {},
    }


def _runtime_smoke_record(
    args: argparse.Namespace,
    row: RuntimeRow,
    adapter: _Adapter | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Build one schema-valid singleton around an actual runtime observation."""
    revision = args.revision or "global"
    identity = AttemptIdentity(
        args.batch_id,
        row_key(row),
        "attempt-01",
        args.candidate_sha or "cpu-smoke",
        {revision: args.revision_sha or "cpu-smoke"},
        args.harness_sha256 or "cpu-smoke",
    )
    supported = result["status"] != "unsupported"
    return {
        "schema": SCHEMA,
        "identity": asdict(identity),
        "kind": "singleton",
        "status": result["status"],
        "boundary": "runtime_application",
        "telemetry": {"required": False, "available": True, "samples": [], "rejection_reasons": []},
        "members": [
            {
                "revision": revision,
                "requested_execution": row.requested_execution,
                "effective_execution": result.get("effective_execution"),
                "revision_sha": args.revision_sha or "cpu-smoke",
                "adapter": type(adapter).__name__ if adapter is not None else None,
                "resolved_row": asdict(row),
                "source_emulation": False,
                "capability": {"supported": supported, "reason": None if supported else result.get("reason")},
                "timing": result.get("timing", {}),
                "counters": result.get("counters", {}),
                "structural": adapter.introspect() if adapter is not None else None,
            }
        ],
        "paths": {"harness": None, "worktrees": {}, "cache": {}},
        "command": sys.argv,
        "device": args.device,
        "cache": {"policy": "private"},
        "process": {"returncode": 0},
        "metadata": {"reason": result.get("reason")},
    }


def _cleanup_workload(workload: _Workload | None) -> None:
    if workload is not None and workload.network_file is not None:
        Path(workload.network_file).unlink(missing_ok=True)


def _write_smoke_attempt(args: argparse.Namespace, key: str, record: dict[str, Any]) -> None:
    output = args.output_path / key.replace(":", "_")
    attempt = allocate_attempt_dir(output)
    write_attempt_atomically(attempt, record)


def _run_runtime_child(args: argparse.Namespace) -> int:
    revision = args.revision or "global"
    if not args.child_row:
        raise RuntimeError("runtime mode requires --child_row")
    try:
        selected_row = next(row for row in runtime_matrix(revision) if row_key(row) == args.child_row)
    except StopIteration as error:
        raise RuntimeError(f"unknown runtime child row: {args.child_row}") from error
    row = RuntimeRow(
        selected_row.actuator_type,
        selected_row.groups,
        selected_row.requested_execution,
        selected_row.effective_execution,
        args.num_worlds or selected_row.num_worlds,
    )
    if row.effective_execution is None:
        result = {
            "status": "unsupported",
            "requested_execution": row.requested_execution,
            "effective_execution": None,
            "reason": f"{revision} does not support actuator graph execution",
        }
        _write_smoke_attempt(args, row_key(row), _runtime_smoke_record(args, row, None, result))
        return 0

    selected = select_adapter(revision, args.device)
    if isinstance(selected, RevisionCapability):
        raise RuntimeError(selected.reason)
    adapter = selected
    build_row = BuildRow("B5", row.num_worlds, 1, 1, row.groups, (row.actuator_type,))
    workload: _Workload | None = make_workload(build_row, args.device)
    try:
        adapter.build_workload(workload)
        adapter.introspect()
        adapter.first_application(workload)
        adapter.introspect()
        result = measure_runtime(adapter, row, args.warmup_iterations, args.num_iterations)
        _write_smoke_attempt(args, row_key(row), _runtime_smoke_record(args, row, adapter, result))
    finally:
        adapter.close()
        _cleanup_workload(workload)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a local smoke child or validate coordinate arguments."""
    args = parse_args(argv)
    if args.mode == "coordinate":
        return 0
    if args.mode == "runtime":
        return _run_runtime_child(args)
    rows = list(expand_build_matrix(args.case))
    if args.case != "all":
        row = rows[0]
        row = BuildRow(
            row.case,
            args.num_worlds or row.num_worlds,
            args.num_sources or row.num_sources,
            args.num_articulations or row.num_articulations,
            args.groups or row.groups,
            tuple(args.actuator_types.split(",")) if args.actuator_types else row.actuator_types,
            row.global_only,
        )
    else:
        row = rows[0]
    revision = args.revision or "global"
    if row.global_only:
        if revision != "global":
            raise RuntimeError(f"{row.case} is a global-only structural case")
        workload = make_workload(row, args.device)
        try:
            structural = run_global_structural_case(workload)
            _write_smoke_attempt(args, row_key(row), _smoke_record(args, row, None, structural=structural))
        finally:
            _cleanup_workload(workload)
        return 0

    if args.phase in {"cold", "warm"}:
        result = measure_build(
            revision,
            row,
            args.device,
            args.phase,
            warmup_constructions=args.warmup_iterations,
            measured_constructions=args.num_iterations,
        )
        _write_smoke_attempt(
            args,
            row_key(row),
            _smoke_record(
                args,
                row,
                None,
                structural=result["structural"],
                timing=result["timing"],
                counters=result["counters"],
                adapter_name=result["adapter_name"],
            ),
        )
        return 0
    selected = select_adapter(revision, args.device)
    if isinstance(selected, RevisionCapability):
        raise RuntimeError(selected.reason)
    adapter = selected
    workload = make_workload(row, args.device)
    try:
        adapter.build_workload(workload)
        adapter.first_application(workload)
        _write_smoke_attempt(args, row_key(row), _smoke_record(args, row, adapter))
    finally:
        adapter.close()
        _cleanup_workload(workload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
