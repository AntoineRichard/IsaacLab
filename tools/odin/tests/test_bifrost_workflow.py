# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import re

from tools.odin.bifrost.workflow import osmo_safe_task_name

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


def test_render_workflow_yaml_is_valid_yaml_with_one_task():
    cfg = _cfg()
    out = render_workflow_yaml(
        dispatch_id="20260505-150000",
        rows=[_row()],
        cfg=cfg,
        tarball_path="/tmp/odin-source.tar.gz",
    )
    parsed = yaml.safe_load(out)
    wf = parsed["workflow"]
    assert wf["name"] == "odin-disp-20260505-150000"
    assert wf["pool"] == "rtx-pro-6000-eval"
    assert len(wf["tasks"]) == 1
    task = wf["tasks"][0]
    assert task["image"] == "nvcr.io/nvidia/isaac-lab:2.2.0"
    assert task["credentials"]["registry"] == "ngc-readonly"
    assert (
        task["outputs"][0]["dataset"]["name"]
        == "odin-20260505-150000-rsl-rl_physx_Isaac-Ant-Direct-v0_20260505-150000_seed42"
    )
    assert task["exitActions"]["RESCHEDULE"] == "3001-3006"


def test_render_workflow_yaml_n_parallel_tasks():
    cfg = _cfg()
    rows = [_row(seed=42), _row(seed=43, framework="skrl")]
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=rows, cfg=cfg, tarball_path="/tmp/odin-source.tar.gz"
    )
    parsed = yaml.safe_load(out)
    assert len(parsed["workflow"]["tasks"]) == 2


def test_render_workflow_special_token_output_survives_render():
    """OSMO's `{{output}}` must appear literally in the rendered YAML."""
    cfg = _cfg()
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path="/tmp/odin-source.tar.gz"
    )
    assert "{{output}}" in out


def test_render_workflow_files_upload_mode_includes_tarball():
    cfg = _cfg(mode="files_upload")
    out = render_workflow_yaml(
        dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path="/abs/odin-source.tar.gz"
    )
    parsed = yaml.safe_load(out)
    files = parsed["workflow"]["tasks"][0]["files"]
    paths = [f.get("path") for f in files]
    assert "/workspace/odin-source.tar.gz" in paths
    tarball_entry = [f for f in files if f.get("path") == "/workspace/odin-source.tar.gz"][0]
    assert tarball_entry["localpath"] == "/abs/odin-source.tar.gz"


def test_render_workflow_rsync_mode_omits_tarball():
    cfg = _cfg(mode="rsync")
    out = render_workflow_yaml(dispatch_id="20260505-150000", rows=[_row()], cfg=cfg, tarball_path=None)
    parsed = yaml.safe_load(out)
    paths = [f.get("path") for f in parsed["workflow"]["tasks"][0]["files"]]
    assert "/workspace/odin-source.tar.gz" not in paths
