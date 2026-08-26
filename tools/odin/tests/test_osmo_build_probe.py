# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OSMO builder cache-mount probe."""

from __future__ import annotations

import pytest
import yaml

from tools.odin.osmo_build.probe_cache_mount import render_probe

_AUTH_B64 = "ZHVtbXk6ZHVtbXk="
_DESTINATION = "nvcr.io/nvidian/antoiner-isaac-lab:probe-test"


@pytest.mark.parametrize("builder", ["kaniko", "buildkit"])
def test_credential_is_placed_in_files_not_environment(builder):
    """OSMO does not expose GENERIC credentials, so the auth blob rides in
    `files:` and must not leak into `environment:`, which is echoed in logs."""
    doc = yaml.safe_load(render_probe(builder, _AUTH_B64, _DESTINATION))
    task = doc["workflow"]["groups"][0]["tasks"][0]
    files = {f["path"]: f["contents"] for f in task["files"]}
    assert any("config.json" in path for path in files)
    assert any(_AUTH_B64 in contents for contents in files.values())
    assert _AUTH_B64 not in yaml.safe_dump(task.get("environment", {}))


def test_rejects_an_unknown_builder():
    with pytest.raises(KeyError):
        render_probe("podman", _AUTH_B64, _DESTINATION)
