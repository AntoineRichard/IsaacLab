# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slow-marked integration test: Asgard dispatch against ssh localhost.

Replaces the 'docker exec' command with a shell stub so the test runs
without docker or Isaac Sim. Still exercises the real ShellSSHRunner and
ShellRsyncRunner subprocess paths, covering the transport + dispatch wiring
end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.runner import DispatchOptions, run_dispatch
from tools.odin.asgard.transport import ShellRsyncRunner, ShellSSHRunner
from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list


def _ssh_localhost_works() -> bool:
    """Probe: can we `ssh localhost "echo ok"` without a password?"""
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "echo ok"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


pytestmark = pytest.mark.slow


@pytest.fixture
def stub_ssh_runner(monkeypatch, tmp_path: Path):
    """Replace ShellSSHRunner.run's docker-exec command with a local stub that
    materialises a valid bundle and exits 0."""
    from tools.odin.asgard import worker as worker_mod

    real_build = worker_mod._build_docker_exec_cmd

    def _fake_build(host: ValkyrieConfig, job) -> str:
        # Write a minimal valid bundle into the host's odin_runs/ directory
        # (same path the real runner + rsync pull target expects).
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = {
            "schema_version": "1.0",
            "phases": {"training": {"status": "completed"}, "startup": {"status": "completed"}},
        }
        training = {"schema_version": "1.0"}
        startup = {"schema_version": "1.0"}
        manifest_s = json.dumps(manifest).replace("'", r"\'")
        training_s = json.dumps(training).replace("'", r"\'")
        startup_s = json.dumps(startup).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir} && "
            f"printf '%s' '{manifest_s}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{training_s}' > {bundle_dir}/training.json && "
            f"printf '%s' '{startup_s}' > {bundle_dir}/startup.json"
        )

    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", _fake_build)
    yield
    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", real_build)


@pytest.fixture
def stub_provisioner(monkeypatch):
    """Preflight (docker ps / docker inspect) would fail on vanilla localhost.
    Short-circuit preflight and provisioner to always pass."""
    from tools.odin.asgard import preflight as pf
    from tools.odin.asgard import provisioner as pv

    def _fake_pf(host, *, ssh):
        return pf.PreflightResult(
            host=host.host,
            ok=True,
            checks={"ssh_reach": True, "docker_running": True, "container_up": True, "isaaclab_present": True},
            message="",
        )

    def _fake_pv(host, working_tree, *, fresh, ssh, rsync):
        return pv.ProvisionResult(host=host.host, ok=True, commit_sha="integration-stub")

    monkeypatch.setattr("tools.odin.asgard.runner.preflight_valkyrie", _fake_pf)
    monkeypatch.setattr("tools.odin.asgard.runner.provision_valkyrie", _fake_pv)


def test_loopback_dispatch_against_localhost(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    # Build a one-row env list.
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
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    # One-host fleet targeting localhost; use the current repo as the remote
    # "isaaclab_path" so the fake bundle lands in a path we can rsync back.
    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-test", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260422-loopback"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], per_job_timeout_s=60),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # The one job must complete.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"

    # Bundle must have been pulled back to the dispatch directory.
    bundle = dispatch_dir / state.jobs[0].bundle_dir_name
    assert (bundle / "manifest.json").exists()
    assert (bundle / "logs" / "ssh-tail.log").exists()

    # dispatch.json must record success.
    dj = json.loads((dispatch_dir / "dispatch.json").read_text())
    assert dj["jobs"][0]["status"] == "completed"
