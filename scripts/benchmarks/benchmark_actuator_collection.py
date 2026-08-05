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
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "actuator_collection_attempt/v1"
_REVISIONS = ("develop", "current", "global")
_EXECUTIONS = ("cached_eager", "graph")


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
    if args.case == "all" and any(
        getattr(args, name) is not None
        for name in ("num_worlds", "num_sources", "num_articulations", "groups", "actuator_types")
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
    return _Workload(row, device, joint_names, values, (0.1, 0.2, 0.3))


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
    from isaaclab.actuators.actuator_pd_cfg import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg

    types = workload.row.actuator_types
    configs: dict[str, Any] = {}
    for index in range(workload.row.groups):
        actuator_type = types[index % len(types)]
        cfg_type = {"implicit": ImplicitActuatorCfg, "ideal_pd": IdealPDActuatorCfg, "dc_motor": DCMotorCfg}.get(
            actuator_type
        )
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
        configs[f"group_{index}"] = cfg_type(**values)
    return configs


class _DevelopAdapter(_MemoryAdapter):
    """Direct historical execution adapter using real ActuatorBase.compute calls."""

    def build_workload(self, workload: _Workload) -> None:
        import torch

        from isaaclab.actuators.actuator_pd import DCMotor, IdealPDActuator, ImplicitActuator

        self.workload = workload
        self.groups = []
        concrete = {"implicit": ImplicitActuator, "ideal_pd": IdealPDActuator, "dc_motor": DCMotor}
        for name, cfg in _group_cfgs(workload).items():
            index = int(name.rsplit("_", 1)[1])
            group_type = concrete[workload.row.actuator_types[index % len(workload.row.actuator_types)]]
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

    def first_application(self, workload: _Workload) -> None:
        import torch

        from isaaclab.utils.types import ArticulationActions

        for group in self.groups:
            shape = (workload.row.num_worlds, 1)
            zeros = torch.zeros(shape, device=workload.device)
            group.compute(
                ArticulationActions(
                    joint_positions=torch.full(shape, workload.first_command[0], device=workload.device),
                    joint_velocities=zeros,
                    joint_efforts=zeros,
                ),
                zeros,
                zeros,
            )
        self.applications += 1


class _CurrentPrAdapter(_MemoryAdapter):
    """Articulation-local collection adapter with a driver-owned control double."""

    def build_workload(self, workload: _Workload) -> None:
        from isaaclab.actuators.actuator_collection import ActuatorCollection

        self.workload = workload
        self.control = _DriverControl(
            workload.device, workload.row.num_worlds, workload.joint_names, workload.row.num_sources
        )
        self.collection = ActuatorCollection(_group_cfgs(workload), self.control)

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
                    cfg_rows={1: tuple(range(workload.row.num_sources))},
                )

            def get_clone_plan(self) -> Any:
                return self.plan

        self.workload = workload
        self.control = _DriverControl(
            workload.device, workload.row.num_worlds, workload.joint_names, workload.row.num_sources
        )
        self.manager = ActuatorCollection(_Simulation())
        self.view = self.manager.register_articulation(
            key="benchmark",
            cfgs=_group_cfgs(workload),
            control=self.control,
            replication_cfg_id=1,
            debug_validation=False,
            debug_value_resolution=False,
        )
        self.manager.finalize()
        if not self.manager.is_finalized or not self.view.is_ready or self.view._execution_plan is None:
            raise RuntimeError("global collection lifecycle probe failed")

    def first_application(self, workload: _Workload) -> None:
        import torch

        value = torch.full(
            (workload.row.num_worlds, workload.row.groups), workload.first_command[0], device=workload.device
        )
        self.view.command.set_position_index(value=value)
        self.view.command.set_velocity_index(value=torch.zeros_like(value))
        self.view.command.set_effort_index(value=torch.zeros_like(value))
        self.view.compute()
        self.view.submit_commands()
        self.applications += 1

    def run_execution(self, count: int) -> None:
        for _ in range(count):
            self.view.compute()
            self.view.submit_commands()
        self.applications += count

    def close(self) -> None:
        if hasattr(self, "manager"):
            self.manager.clear_generation()
        super().close()


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


def measure_runtime(adapter: _Adapter, row: RuntimeRow, warmups: int, iterations: int) -> dict[str, Any]:
    """Measure one runtime mode without relabelling failed graph capture."""
    if row.requested_execution == "graph" and not adapter.warmup_execution(row):
        return {
            "status": "rejected",
            "requested_execution": "graph",
            "effective_execution": None,
            "reason": "graph capture failed",
        }
    adapter.run_execution(warmups)
    started = time.perf_counter_ns()
    adapter.run_execution(iterations)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return {
        "status": "accepted",
        "requested_execution": row.requested_execution,
        "effective_execution": row.effective_execution,
        "timing": {"samples_ms": [elapsed_ms]},
    }


class _ScopedInstrumentation:
    """Temporarily observe explicit Warp launch/copy sites in a measured scope."""

    def __init__(self, warp: Any) -> None:
        self.warp = warp
        self._originals: dict[str, Any] = {}
        self.launches: dict[str, int] = {}
        self.h2d_bytes = 0
        self.d2h_readbacks = 0

    def __enter__(self) -> _ScopedInstrumentation:
        if self.warp is None:
            return self
        for name in ("launch", "launch_tiled", "copy"):
            original = getattr(self.warp, name, None)
            if original is None:
                continue
            self._originals[name] = original

            def wrapped(*args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
                self.launches[_name] = self.launches.get(_name, 0) + 1
                return _original(*args, **kwargs)

            setattr(self.warp, name, wrapped)
        return self

    def __exit__(self, *_: Any) -> None:
        for name, original in self._originals.items():
            setattr(self.warp, name, original)

    def record_readback(self, final_timing_sync: bool = False) -> None:
        if not final_timing_sync:
            self.d2h_readbacks += 1


def observe_runtime_scopes(adapter: _Adapter, iterations: int) -> list[str]:
    """Keep capture and replay observation boundaries distinct."""
    scopes = ["capture"]
    adapter.warmup_execution(RuntimeRow("implicit", 1, "graph", "graph"))
    scopes.append("replay")
    adapter.run_execution(iterations)
    return scopes


class _GlobalIntrospector:
    """Read actual finalized generation owners without analytical layout guesses."""

    def inspect(self, generation: Any) -> dict[str, Any]:
        owners = [*getattr(generation, "stores", ()), *getattr(generation, "plans", ())]
        unique: dict[tuple[Any, Any], Any] = {}
        for owner in owners:
            key = (getattr(owner, "device", None), getattr(owner, "ptr", None))
            if key[1] is not None:
                unique[key] = owner
        return {
            "canonical_allocation_count": len(unique),
            "canonical_allocation_bytes": sum(getattr(owner, "nbytes", 0) for owner in unique.values()),
            "descriptor_count": len(getattr(generation, "stores", ())),
            "plan_staging_owner_count": len(getattr(generation, "plans", ())),
        }


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


def _smoke_record(args: argparse.Namespace, row: BuildRow, adapter: _Adapter) -> dict[str, Any]:
    identity = AttemptIdentity(
        args.batch_id,
        row_key(row),
        "attempt-01",
        args.candidate_sha or "cpu-smoke",
        {"global": args.revision_sha or "cpu-smoke"},
        args.harness_sha256 or "cpu-smoke",
    )
    return {
        "schema": SCHEMA,
        "identity": asdict(identity),
        "kind": "singleton",
        "status": "accepted",
        "boundary": "resolved_construction_to_first_application",
        "telemetry": {"required": False, "available": True, "samples": [], "rejection_reasons": []},
        "members": [
            {
                "revision": args.revision or "global",
                "requested_execution": "cached_eager",
                "effective_execution": "cached_eager",
                "revision_sha": args.revision_sha or "cpu-smoke",
                "adapter": type(adapter).__name__,
                "resolved_row": asdict(row),
                "source_emulation": row.case == "B3" and (args.revision or "global") != "global",
                "capability": {"supported": True, "reason": None},
                "timing": {"samples_ms": [], "first_application_count": getattr(adapter, "applications", 1)},
                "counters": {},
                "structural": adapter.introspect(),
            }
        ],
        "paths": {"harness": None, "worktrees": {}, "cache": {}},
        "command": sys.argv,
        "device": args.device,
        "cache": {"policy": "private"},
        "process": {"returncode": 0},
        "metadata": {},
    }


def main(argv: list[str] | None = None) -> int:
    """Run a local smoke child or validate coordinate arguments."""
    args = parse_args(argv)
    if args.mode == "coordinate":
        return 0
    if args.mode != "build":
        return 0
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
    selected = select_adapter(args.revision or "global", args.device)
    if isinstance(selected, RevisionCapability):
        raise RuntimeError(selected.reason)
    adapter = selected
    workload = make_workload(row, args.device)
    adapter.build_workload(workload)
    adapter.first_application(workload)
    output = args.output_path / row_key(row).replace(":", "_")
    attempt = allocate_attempt_dir(output)
    write_attempt_atomically(attempt, _smoke_record(args, row, adapter))
    adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
