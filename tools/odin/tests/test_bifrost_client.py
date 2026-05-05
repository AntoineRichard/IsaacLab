# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.odin.bifrost.client import (
    OsmoAuthError,
    OsmoCliError,
    OsmoClient,
    OsmoTransientError,
)


SUBMIT_STDOUT_OK = """\
Workflow submit successful.
Workflow ID        - my-wf-1
Workflow Overview  - https://osmo.example.com/workflows/my-wf-1
"""


def _completed(returncode=0, stdout="", stderr=""):
    cp = MagicMock(spec=subprocess.CompletedProcess)
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def test_submit_parses_workflow_id_from_stdout(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout=SUBMIT_STDOUT_OK)) as run:
        wf_id = client.submit(yaml)
    assert wf_id == "my-wf-1"
    args, kwargs = run.call_args
    assert args[0][:3] == ["osmo", "workflow", "submit"]
    assert str(yaml) in args[0]
    assert kwargs.get("env", {}).get("OSMO_PROFILE") == "prod"


def test_submit_with_rsync_appends_flags(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout=SUBMIT_STDOUT_OK)) as run:
        client.submit(yaml, rsync_pairs=[("./tools/odin", "/workspace/odin-source")])
    cmd = run.call_args[0][0]
    assert "--rsync" in cmd
    rsync_idx = cmd.index("--rsync")
    assert cmd[rsync_idx + 1] == "./tools/odin:/workspace/odin-source"


def test_submit_raises_auth_error_on_401(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="HTTP 401 Unauthorized")):
        with pytest.raises(OsmoAuthError):
            client.submit(yaml)


def test_submit_raises_transient_on_503(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="HTTP 503 Service Unavailable")):
        with pytest.raises(OsmoTransientError):
            client.submit(yaml)


def test_submit_raises_generic_cli_error_on_other_failure(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="bad spec")):
        with pytest.raises(OsmoCliError, match="bad spec"):
            client.submit(yaml)


def test_submit_raises_when_id_unparseable(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout="weird output")):
        with pytest.raises(OsmoCliError, match="Workflow ID"):
            client.submit(yaml)
