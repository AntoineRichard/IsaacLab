# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import tarfile
from pathlib import Path

from tools.odin.bifrost.workflow import osmo_safe_task_name, stage_source_tarball

DNS_1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


def test_simple_run_id_lowercased_and_dashed():
    out = osmo_safe_task_name("rsl-rl_physx_Isaac-Ant-Direct-v0_20260505-150000_seed42")
    assert DNS_1123_LABEL.match(out), f"not DNS-1123-safe: {out!r}"
    assert out.startswith("rsl-rl-physx-isaac-ant-direct-v0")


def test_no_underscores_in_output():
    out = osmo_safe_task_name("a_b_c")
    assert "_" not in out


def test_no_dots_in_output():
    out = osmo_safe_task_name("a.b.c")
    assert "." not in out


def test_truncation_appends_hash():
    long = "x" * 80
    out = osmo_safe_task_name(long)
    assert len(out) <= 63
    assert DNS_1123_LABEL.match(out)
    # Two different long inputs should produce different outputs
    other = osmo_safe_task_name("y" * 80)
    assert out != other


def test_no_leading_or_trailing_dash():
    out = osmo_safe_task_name("_foo_")
    assert not out.startswith("-")
    assert not out.endswith("-")


def test_idempotent_on_safe_name():
    safe = "rsl-rl-physx-x-seed42"
    assert osmo_safe_task_name(safe) == safe


import yaml

from tools.odin.bifrost.config import (
    BifrostConfig,
    CodeDeliverySpec,
    DefaultsSpec,
    ImageSpec,
    ResourcesSpec,
    RetrySpec,
)
from tools.odin.bifrost.workflow import RenderRow, render_workflow_yaml


def _cfg(mode: str = "files_upload") -> BifrostConfig:
    return BifrostConfig(
        osmo_profile="prod",
        pool="rtx-pro-6000-eval",
        priority="NORMAL",
        image=ImageSpec(reference="nvcr.io/nvidia/isaac-lab:2.2.0", pull_credential="ngc-readonly"),
        defaults=DefaultsSpec(
            resources=ResourcesSpec(cpu=16, gpu=1, memory="64Gi", storage="64Gi", platform="rtx-pro-6000"),
            exec_timeout=14400,
            queue_timeout=7200,
        ),
        retry=RetrySpec(reschedule_codes="3001-3006", restart_codes=""),
        bundle_dataset_prefix="odin",
        code_delivery=CodeDeliverySpec(mode=mode, source_root="tools/odin"),
    )


def _row(seed: int = 42, framework: str = "rsl-rl") -> RenderRow:
    return RenderRow(
        run_id=f"{framework}_physx_Isaac-Ant-Direct-v0_20260505-150000_seed{seed}",
        osmo_task_name=f"{framework}-physx-isaac-ant-seed{seed}".replace("_", "-"),
        framework=framework,
        framework_runner="hugin" if framework == "rsl-rl" else "munin",
        task_id="Isaac-Ant-Direct-v0",
        backend="physx",
        seed=seed,
        num_envs=4096,
        max_iterations=500,
    )


def _first_task(parsed: dict) -> dict:
    """Extract the first task from the first group of a rendered workflow."""
    return parsed["workflow"]["groups"][0]["tasks"][0]


def test_render_workflow_yaml_is_valid_yaml_with_one_task():
    cfg = _cfg()
    out = render_workflow_yaml(
        dispatch_id="20260505-150000",
        rows=[_row()],
        cfg=cfg,
        tarball_path="/tmp/odin-source.tar.gz",
        exec_timeout="2h",
    )
    parsed = yaml.safe_load(out)
    wf = parsed["workflow"]
    assert wf["name"] == "odin-disp-20260505-150000"
    assert len(wf["groups"]) == 1
    assert len(wf["groups"][0]["tasks"]) == 1
    task = _first_task(parsed)
    assert task["image"] == "nvcr.io/nvidia/isaac-lab:2.2.0"
    assert (
        task["outputs"][0]["dataset"]["name"]
        == "odin-20260505-150000-rsl-rl_physx_Isaac-Ant-Direct-v0_20260505-150000_seed42"
    )
    assert task["exitActions"]["RESCHEDULE"] == "3001-3006"


def test_render_workflow_yaml_n_parallel_tasks():
    cfg = _cfg()
    rows = [_row(seed=42), _row(seed=43, framework="skrl")]
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=rows, cfg=cfg, tarball_path="/tmp/odin-source.tar.gz", exec_timeout="2h"
    )
    parsed = yaml.safe_load(out)
    # Each task becomes its own one-task group.
    assert len(parsed["workflow"]["groups"]) == 2
    assert all(len(g["tasks"]) == 1 for g in parsed["workflow"]["groups"])


def test_render_workflow_group_name_differs_from_task_name():
    """Per OSMO schema, group and task names must be unique within a workflow."""
    cfg = _cfg()
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path="/tmp/x.tar.gz", exec_timeout="2h"
    )
    parsed = yaml.safe_load(out)
    grp = parsed["workflow"]["groups"][0]
    assert grp["name"] != grp["tasks"][0]["name"]


def test_render_workflow_special_token_output_survives_render():
    """OSMO's `{{output}}` must appear literally in the rendered YAML."""
    cfg = _cfg()
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path="/tmp/odin-source.tar.gz", exec_timeout="2h"
    )
    assert "{{output}}" in out


def test_render_workflow_files_upload_mode_includes_tarball():
    cfg = _cfg(mode="files_upload")
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path="/abs/odin-source.tar.gz", exec_timeout="2h"
    )
    parsed = yaml.safe_load(out)
    files = _first_task(parsed)["files"]
    paths = [f.get("path") for f in files]
    assert "/workspace/odin-source.tar.gz" in paths
    tarball_entry = [f for f in files if f.get("path") == "/workspace/odin-source.tar.gz"][0]
    assert tarball_entry["localpath"] == "odin-source.tar.gz"


def test_render_workflow_rsync_mode_omits_tarball():
    cfg = _cfg(mode="rsync")
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path=None, exec_timeout="2h"
    )
    parsed = yaml.safe_load(out)
    paths = [f.get("path") for f in _first_task(parsed)["files"]]
    assert "/workspace/odin-source.tar.gz" not in paths


def test_render_workflow_exec_timeout_threaded_into_yaml():
    """Different ``exec_timeout`` values land in the rendered workflow.timeout.

    The timeout is per-chunk now (spec §5.3): the same rows can be
    rendered into two workflows with different exec_timeouts so each
    timeout_class's chunk gets the right value.
    """
    cfg = _cfg()
    rows = [_row()]
    out_short = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=rows, cfg=cfg, tarball_path="/tmp/x.tar.gz", exec_timeout="30m"
    )
    out_long = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=rows, cfg=cfg, tarball_path="/tmp/x.tar.gz", exec_timeout="8h"
    )
    assert yaml.safe_load(out_short)["workflow"]["timeout"]["exec_timeout"] == "30m"
    assert yaml.safe_load(out_long)["workflow"]["timeout"]["exec_timeout"] == "8h"


def test_stage_source_tarball_produces_readable_archive(tmp_path: Path):
    src = tmp_path / "src"
    (src / "tools" / "odin").mkdir(parents=True)
    (src / "tools" / "odin" / "hello.py").write_text("print('hi')\n")
    out = tmp_path / "src.tar.gz"
    stage_source_tarball(src / "tools" / "odin", out, repo_root=src)
    assert out.exists()
    with tarfile.open(out, "r:gz") as t:
        names = t.getnames()
    assert "tools/odin/hello.py" in names
