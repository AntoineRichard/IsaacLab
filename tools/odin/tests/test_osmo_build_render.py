# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OSMO-native build workflow rendering."""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from tools.odin.image import DEFAULT_CUDA_IMAGE
from tools.odin.osmo_build.render import NVDATASET_CARRIER_IMAGE, read_push_auth, render_build_workflow
from tools.odin.plan import UV_EXTRAS

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


def test_read_push_auth_returns_the_nvcr_credential(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"auths": {"nvcr.io": {"auth": "ZHVtbXk6ZHVtbXk="}}}))
    assert read_push_auth(config) == "ZHVtbXk6ZHVtbXk="


def test_read_push_auth_raises_when_the_config_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_push_auth(tmp_path / "does-not-exist.json")


def test_read_push_auth_raises_when_nvcr_is_not_configured(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"auths": {"docker.io": {"auth": "ZHVtbXk6ZHVtbXk="}}}))
    with pytest.raises(FileNotFoundError):
        read_push_auth(config)


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


def test_dockerfile_odin_matches_the_live_extras_and_cuda_base():
    """`Dockerfile.odin` hardcodes what `templates/Dockerfile.j2` renders from
    `UV_EXTRAS` (plan.py) and `DEFAULT_CUDA_IMAGE` (image.py) because the OSMO
    builder has no templating step. If either constant changes and this file
    is not, the OSMO image silently ships without the new extra (or the wrong
    CUDA base) while `dispatch.yaml.j2` still expects it -- a gap that only
    surfaces mid-dispatch, inside a GPU pod, as a runtime install or a
    mismatched base."""
    dockerfile = pathlib.Path(__file__).parents[3] / "Dockerfile.odin"
    text = dockerfile.read_text()
    assert " --extra ".join(UV_EXTRAS) in text
    assert DEFAULT_CUDA_IMAGE in text


def test_dockerfile_odin_copies_the_pinned_nvdataset_carrier():
    """`nvdataset` reaches the OSMO image through a hand-built carrier, because
    OSMO cannot route to artifactory. Bumping `NVDATASET_VERSION` without
    rebuilding and pushing the carrier would otherwise fail 20-odd minutes into
    an OSMO build, on a `COPY --from` of a tag that was never pushed. Failing
    here instead costs nothing."""
    dockerfile = pathlib.Path(__file__).parents[3] / "Dockerfile.odin"
    assert NVDATASET_CARRIER_IMAGE in dockerfile.read_text()


def test_nvdataset_carrier_shares_dockerfile_odin_base_image():
    """The carrier hands over `/root/.local`, whose `console_scripts` launcher
    has a shebang baked to the interpreter path `uv tool install` used. That
    path only resolves in the consuming image if both were built from the same
    base, so the two Dockerfiles must not drift apart."""
    root = pathlib.Path(__file__).parents[3]
    carrier = (root / "tools/odin/osmo_build/Dockerfile.nvdataset").read_text()
    assert f"FROM {DEFAULT_CUDA_IMAGE}" in carrier
    assert DEFAULT_CUDA_IMAGE in (root / "Dockerfile.odin").read_text()
