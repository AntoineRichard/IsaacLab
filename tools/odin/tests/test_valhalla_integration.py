# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end test: T3.1 run_dispatch + synthetic Hugin output -> aggregate.json."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.runner import DispatchOptions, run_dispatch
from tools.odin.asgard.transport import RsyncResult, SSHResult

# --- Fakes ---------------------------------------------------------------------------------------


@dataclass
class _FakeSSH:
    """SSH stub.

    Handles three classes of remote commands:

    - Preflight probes (``echo ...``, ``docker ps ...``, ``docker inspect ...``,
      ``test -d ...``) return zero-exit with the stdout the real callers parse.
    - Provisioner probes (``docker inspect`` status, ``./docker/container.py start``)
      return "running" / zero-exit so :func:`provision_valkyrie` succeeds.
    - The real job command (containing ``run.py``) materialises a synthetic
      Hugin bundle into ``dispatch_dir / <bundle_dir_name>/`` so that when the
      fake rsync pull runs (a no-op) the file is already in the expected spot.
    """

    dispatch_dir: Path
    dispatch_id: str
    calls: list[tuple[str, str]] = field(default_factory=list)

    def run(
        self,
        host,
        cmd: str,
        *,
        timeout_s: float | None = None,
        stdout_tee: Path | None = None,
    ) -> SSHResult:
        del timeout_s, stdout_tee
        self.calls.append((host.host, cmd))

        # Dispatch on command shape.
        if "run.py" in cmd:
            # Extract --task / --backend / --seed from the cmd string.
            task = _extract_flag(cmd, "--task")
            backend = _extract_flag(cmd, "--backend")
            seed_str = _extract_flag(cmd, "--seed")
            if task is None or backend is None or seed_str is None:
                return SSHResult(exit_code=1, stdout="", stderr="missing flag in cmd", duration_s=0.1)
            seed = int(seed_str)
            framework_slug = "rsl-rl" if "hugin/run.py" in cmd else "skrl"
            run_id = f"{framework_slug}_{backend}_{task}_{self.dispatch_id}_seed{seed}"
            bundle = self.dispatch_dir / run_id
            bundle.mkdir(parents=True, exist_ok=True)
            _write_synthetic_bundle(bundle, run_id, seed=seed)
            return SSHResult(exit_code=0, stdout="", stderr="", duration_s=0.1)

        # docker inspect State.Status -> "running" (preflight + provisioner).
        if "docker inspect" in cmd and "State.Status" in cmd:
            return SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.1)

        # Everything else (echo, docker ps, test -d, container.py start/stop, ...) -> ok.
        return SSHResult(exit_code=0, stdout="ok\n", stderr="", duration_s=0.1)


@dataclass
class _FakeRsync:
    """No-op rsync — the fake SSH already wrote the bundle directly into ``dispatch_dir``."""

    def push(self, host, local_path, remote_path) -> RsyncResult:
        del host, local_path, remote_path
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.1)

    def pull(self, host, remote_path, local_path) -> RsyncResult:
        del host, remote_path, local_path
        return RsyncResult(exit_code=0, stdout="", stderr="", duration_s=0.1)


# --- Helpers -------------------------------------------------------------------------------------


_FLAG_RE = re.compile(r"(--[A-Za-z_][A-Za-z0-9_-]*)\s+(\S+)")


def _extract_flag(cmd: str, flag: str) -> str | None:
    """Return the value for ``flag`` in a ``--flag value`` shell command, else ``None``."""
    for m in _FLAG_RE.finditer(cmd):
        if m.group(1) == flag:
            return m.group(2)
    return None


def _write_synthetic_bundle(bundle: Path, run_id: str, *, seed: int) -> None:
    """Write a manifest.json + training.json pair that the aggregator accepts."""
    # Reward varies by seed so aggregate stats are non-trivial.
    reward = 100.0 + seed * 0.5
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "machine": {
            "hostname": "valkyrie-01.internal",
            "git_commit": "abc123",
            "git_branch": "main",
        },
        "phases": {
            "startup": {
                "file": "startup.json",
                "status": "completed",
                "duration_s": 30.0,
                "exit_code": 0,
            },
            "training": {
                "file": "training.json",
                "status": "completed",
                "duration_s": 150.0,
                "exit_code": 0,
            },
        },
        "config": {
            "framework": "rsl_rl",
            "backend": "physx",
            "task": "Isaac-Ant-Direct-v0",
            "seed": seed,
            "num_envs": 4096,
            "max_iterations": 300,
        },
        "run_start_time_utc": "2026-04-23T10:00:00Z",
        "run_end_time_utc": "2026-04-23T10:03:00Z",
        "run_duration_s": 180.0,
        "artifacts": ["logs", "startup.json", "training.json", "training_data"],
    }
    training = {
        "schema_version": "1.0",
        "runtime": {
            "iterations_completed": 300,
            "total_wall_time_s": 150.0,
            "iteration_time_s": {"mean": 0.5, "std": 0.05},
            "env_steps_per_s": {"mean": 250000.0, "std": 2500.0},
            "iterations_per_s": {"mean": 2.0, "std": 0.01},
            "startup_phase_times_s": {"app_launch": 4.5, "env_creation": 12.4, "first_step": 0.006},
        },
        "resources": {
            "ram_gb": {"mean": 7.2, "peak": 8.0},
            "gpu_mem_gb": {"mean": 3.6, "peak": 4.0},
        },
        "learning": {
            "reward": {"final_raw": reward + 1, "final_ema": reward, "series_per_iter": [0.0] * 300},
            "ep_length": {"final_raw": 950, "final_ema": 940, "series_per_iter": [0.0] * 300},
        },
    }
    with (bundle / "manifest.json").open("w") as fh:
        json.dump(manifest, fh)
    with (bundle / "training.json").open("w") as fh:
        json.dump(training, fh)


def _make_env_yaml(tmp_path: Path) -> Path:
    from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list

    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=300,
            keep=True,
            status="current",
        )
    ]
    path = tmp_path / "physx.yaml"
    write_env_list(path, el, generator="test")
    return path


# --- The test ------------------------------------------------------------------------------------


def test_run_dispatch_produces_aggregate_with_correct_stats(tmp_path: Path):
    dispatch_id = "20260423-110000"
    dispatch_dir = tmp_path / dispatch_id
    dispatch_dir.mkdir()

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(
                host="valkyrie-01.internal",
                ssh_user="odin",
                ssh_key=None,
                isaaclab_path="/opt/IsaacLab",
            )
        ],
    )
    physx_yaml = _make_env_yaml(tmp_path)

    ssh = _FakeSSH(dispatch_dir=dispatch_dir, dispatch_id=dispatch_id)
    rsync = _FakeRsync()

    run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43, 44]),
        ssh=ssh,
        rsync=rsync,
    )

    agg_path = dispatch_dir / "aggregate.json"
    assert agg_path.exists()
    with agg_path.open("r") as fh:
        agg = json.load(fh)
    assert agg["schema_version"] == "1.0"
    assert agg["totals"]["tasks"] == 1
    assert agg["totals"]["completed"] == 3
    assert agg["totals"]["failed"] == 0
    row = agg["rows"][0]
    assert row["task"] == "Isaac-Ant-Direct-v0"
    assert row["aggregate"]["n_seeds_completed"] == 3
    # Seeds 42, 43, 44 -> rewards 121.0, 121.5, 122.0 -> mean 121.5
    assert row["aggregate"]["reward_final_ema"]["mean"] == 121.5
