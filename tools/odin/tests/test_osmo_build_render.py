# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OSMO-native build workflow rendering."""

from __future__ import annotations

import pytest
import yaml

from tools.odin.osmo_build.render import render_build_workflow

_COMMIT = "a" * 40
_KWARGS = {
    "commit_sha": _COMMIT,
    "git_remote": "https://github.com/example/IsaacLab.git",
    "destination": "nvcr.io/nvidian/antoiner-isaac-lab:aaaaaaa",
    "auth_b64": "ZHVtbXk6ZHVtbXk=",
    "git_token": None,
    "cpu": 16,
    "memory": "64Gi",
    "storage": "256Gi",
}


@pytest.mark.parametrize("builder", ["kaniko", "buildkit"])
def test_renders_valid_yaml_with_one_task(builder):
    doc = yaml.safe_load(render_build_workflow(builder=builder, **_KWARGS))
    groups = doc["workflow"]["groups"]
    assert len(groups) == 1
    assert len(groups[0]["tasks"]) == 1


@pytest.mark.parametrize("builder", ["kaniko", "buildkit"])
def test_pins_the_exact_commit(builder):
    rendered = render_build_workflow(builder=builder, **_KWARGS)
    assert _COMMIT in rendered


@pytest.mark.parametrize("builder", ["kaniko", "buildkit"])
def test_requests_no_gpu(builder):
    doc = yaml.safe_load(render_build_workflow(builder=builder, **_KWARGS))
    assert doc["workflow"]["resources"]["default"]["gpu"] == 0


@pytest.mark.parametrize("builder", ["kaniko", "buildkit"])
def test_sizes_are_threaded_through(builder):
    doc = yaml.safe_load(render_build_workflow(builder=builder, **_KWARGS))
    default = doc["workflow"]["resources"]["default"]
    assert default["cpu"] == 16
    assert default["memory"] == "64Gi"
    assert default["storage"] == "256Gi"


@pytest.mark.parametrize("builder", ["kaniko", "buildkit"])
def test_credential_travels_in_files_not_environment(builder):
    """OSMO does not expose GENERIC credentials, so the auth blob rides in
    `files:`. It must not leak into `environment:`, which is echoed in logs."""
    doc = yaml.safe_load(render_build_workflow(builder=builder, **_KWARGS))
    task = doc["workflow"]["groups"][0]["tasks"][0]
    files = {f["path"]: f["contents"] for f in task["files"]}
    assert any("config.json" in path for path in files)
    assert any(_KWARGS["auth_b64"] in contents for contents in files.values())
    assert _KWARGS["auth_b64"] not in yaml.safe_dump(task.get("environment", {}))


def test_git_token_is_not_embedded_in_the_context_url():
    """A token in the context URL lands in build logs. It must use the
    builder's credential channel instead."""
    rendered = render_build_workflow(builder="kaniko", **{**_KWARGS, "git_token": "s3cret"})
    doc = yaml.safe_load(rendered)
    task = doc["workflow"]["groups"][0]["tasks"][0]
    context_args = [a for a in task["args"] if "context" in a]
    assert context_args, "expected a context argument"
    assert all("s3cret" not in a for a in context_args)


def test_rejects_an_unknown_builder():
    with pytest.raises(KeyError):
        render_build_workflow(builder="podman", **_KWARGS)


@pytest.mark.parametrize(
    ("builder", "build_arg_flag"),
    [("kaniko", "--build-arg=ODIN_COMMIT_SHA="), ("buildkit", "--opt=build-arg:ODIN_COMMIT_SHA=")],
)
def test_commit_sha_is_threaded_through_as_a_build_arg(builder, build_arg_flag):
    """`Dockerfile.odin` is committed once and reused for every future build, so
    the commit assertion inside it must come from a build ARG rather than a
    hardcoded SHA. The renderer is responsible for supplying that ARG."""
    doc = yaml.safe_load(render_build_workflow(builder=builder, **_KWARGS))
    task = doc["workflow"]["groups"][0]["tasks"][0]
    assert f"{build_arg_flag}{_COMMIT}" in task["args"]
