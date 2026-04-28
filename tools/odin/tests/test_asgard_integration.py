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

    # Spec 0 / Task 7: aggregator now also writes hardware.json.
    # The loopback _build_docker_exec stub doesn't populate
    # training.json.hardware on every kernel, so accept absence —
    # but if present, validate the schema shape.
    hw_path = dispatch_dir / "hardware.json"
    if hw_path.exists():
        hw = json.loads(hw_path.read_text())
        assert hw["schema_version"] == "1.0"
        assert hw["dispatch_id"] == dispatch_dir.name
        assert hw["fingerprint"].startswith("gpu:")
        assert isinstance(hw["hosts"], dict)


def test_unsupported_pair_lands_in_skipped_array(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    """End-to-end: a (task, backend) the task doesn't support → skipped[]."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    # Build a physx yaml with one supported and one unsupported task.
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=["physx", "newton"],
        ),
        EnvEntry(
            task_id="Isaac-NewtonOnly-v0",
            entry_point="ep:N",
            env_cfg_entry_point="ec:N",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=["newton"],
        ),
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    # One-host fleet targeting localhost; same loopback shape as the
    # success-path test above.
    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-test", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260427-140000"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43], per_job_timeout_s=60, skip_aggregate=True),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # Ant ran (2 seeds); NewtonOnly skipped (2 seeds).
    assert {j.task_id for j in state.jobs} == {"Isaac-Ant-Direct-v0"}
    assert len(state.jobs) == 2
    assert {sk.task_id for sk in state.skipped} == {"Isaac-NewtonOnly-v0"}
    assert len(state.skipped) == 2
    assert all(sk.reason == "preset_unsupported" for sk in state.skipped)
    assert all(sk.backend == "physx" for sk in state.skipped)
    assert {sk.seed for sk in state.skipped} == {42, 43}

    # And dispatch.json on disk reflects the same.
    from tools.odin.asgard.state import read_dispatch_state

    reloaded = read_dispatch_state(dispatch_dir)
    assert reloaded is not None
    assert len(reloaded.skipped) == 2
    assert reloaded.skipped[0].presets_available == ["newton"]


def test_native_match_runs_unsupported_pair_routes_to_skipped(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    """End-to-end: presets_available=[] AND native_backend != requested → skipped[]."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost not configured")

    el = EnvList()
    el.groups["direct/quadcopter"] = [
        EnvEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/quadcopter",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=[],
            native_backend="physx",
        ),
    ]
    yaml_path = tmp_path / "envs.yaml"
    write_env_list(yaml_path, el, generator="test")

    dispatch_dir = tmp_path / "20260427-160000"
    dispatch_dir.mkdir()
    fleet = Fleet(
        fleet_name="loopback-test",
        hosts=[
            ValkyrieConfig(
                host="localhost",
                ssh_user=os.environ.get("USER") or "root",
                ssh_key=None,
                isaaclab_path=str(tmp_path / "remote_isaaclab"),
                container_name="loopback-container",
            ),
        ],
    )
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=None,
        newton_yaml=yaml_path,  # request newton on a physx-native task
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43], skip_aggregate=True, per_job_timeout_s=60),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # Quadcopter never queued; landed in skipped[] with the new reason.
    assert state.jobs == []
    assert len(state.skipped) == 2
    assert {sk.task_id for sk in state.skipped} == {"Isaac-Quadcopter-Direct-v0"}
    assert all(sk.reason == "native_backend_mismatch" for sk in state.skipped)
    assert all(sk.backend == "newton" for sk in state.skipped)
    assert all(sk.native_backend == "physx" for sk in state.skipped)
    assert {sk.seed for sk in state.skipped} == {42, 43}

    # And dispatch.json on disk reflects the same.
    from tools.odin.asgard.state import read_dispatch_state

    reloaded = read_dispatch_state(dispatch_dir)
    assert reloaded is not None
    assert reloaded.schema_version == "1.3"
    assert len(reloaded.skipped) == 2
    assert reloaded.skipped[0].native_backend == "physx"


@pytest.fixture
def stub_ssh_runner_first_job_nvml(monkeypatch, tmp_path: Path):
    """Like ``stub_ssh_runner``, but the FIRST docker-exec call per host fails
    with NVML stderr; subsequent calls succeed and materialise a valid bundle.

    Yields the ``seen_per_host`` counter so the test can assert that the second
    attempt landed on the same host after a successful recovery.
    """
    from tools.odin.asgard import worker as worker_mod

    real_build = worker_mod._build_docker_exec_cmd
    seen_per_host: dict[str, int] = {}

    def _fake_build(host: ValkyrieConfig, job) -> str:
        seen = seen_per_host.get(host.host, 0)
        seen_per_host[host.host] = seen + 1
        if seen == 0:
            # First call on this host: emit a recognised NVML signature on stderr
            # and exit non-zero. Worker._classify maps this to gpu_lost.
            return "echo 'Failed to initialize NVML: Unknown Error' 1>&2 && exit 1"
        # Subsequent calls: materialise a valid bundle and exit 0.
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
    yield seen_per_host
    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", real_build)


def test_loopback_dispatch_recovers_from_gpu_lost(
    tmp_path: Path,
    stub_ssh_runner_first_job_nvml,
    stub_provisioner,
    monkeypatch,
):
    """End-to-end: host's first job emits NVML stderr → ``_classify`` flags
    ``gpu_lost`` → fake ``recover_valkyrie_gpu`` returns ``recovered=True`` →
    second attempt completes the job. Final ``dispatch.json`` shows job
    completed, attempts==2, and ``fleet[host].last_error == 'gpu_lost: recovered'``.
    """
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult

    recover_calls: list[str] = []

    def _fake_recover(host, *, ssh):
        recover_calls.append(host.host)
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=10.0,
            message="recovered_via_container_restart",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    # Build a one-row env list (mirror of the success-path test above).
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

    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-recovery", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260427-recover"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], per_job_timeout_s=60, skip_aggregate=True),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # The job recovered and completed on its second attempt.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"
    assert state.jobs[0].attempts == 2
    fleet_entry = next(f for f in state.fleet if f.host == "localhost")
    assert fleet_entry.last_error == "gpu_lost: recovered"
    assert recover_calls == ["localhost"]

    # On-disk dispatch.json mirrors the in-memory state.
    dj = json.loads((dispatch_dir / "dispatch.json").read_text())
    assert dj["jobs"][0]["status"] == "completed"
    assert dj["jobs"][0]["attempts"] == 2
    on_disk_fleet = next(f for f in dj["fleet"] if f["host"] == "localhost")
    assert on_disk_fleet["last_error"] == "gpu_lost: recovered"

    # Bundle was pulled back on the recovered attempt.
    bundle = dispatch_dir / state.jobs[0].bundle_dir_name
    assert (bundle / "manifest.json").exists()
