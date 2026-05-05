# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.odin.bifrost.client import (
    OsmoAuthError,
    OsmoClient,
    OsmoCliError,
    OsmoTransientError,
    WorkflowSnapshot,
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


def test_submit_raises_transient_on_connection_timeout_single_word(tmp_path: Path):
    yaml = tmp_path / "wf.yaml"
    yaml.write_text("workflow: {}\n")
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="connection timeout")):
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


STATUS_JSON_OK = json.dumps(
    {
        "id": "my-wf-1",
        "status": "RUNNING",
        "tasks": [
            {"name": "rsl-rl-physx-x-seed42", "status": "COMPLETED", "exit_code": 0},
            {"name": "rsl-rl-physx-x-seed43", "status": "FAILED", "exit_code": 137},
            {"name": "rsl-rl-physx-x-seed44", "status": "RUNNING", "exit_code": None},
        ],
    }
)

STATUS_TABLE_OK = """\
Workflow ID: my-wf-1
Status: RUNNING

Tasks:
NAME                       STATUS      EXIT
rsl-rl-physx-x-seed42      COMPLETED   0
rsl-rl-physx-x-seed43      FAILED      137
rsl-rl-physx-x-seed44      RUNNING     -
"""


def test_status_parses_json_output_when_available():
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(stdout=STATUS_JSON_OK)) as run:
        snap = client.status("my-wf-1")
    cmd = run.call_args[0][0]
    assert "--output" in cmd and "json" in cmd
    assert isinstance(snap, WorkflowSnapshot)
    assert snap.workflow_id == "my-wf-1"
    assert snap.status == "RUNNING"
    assert len(snap.tasks) == 3
    completed = [t for t in snap.tasks if t.name.endswith("seed42")][0]
    assert completed.status == "COMPLETED"
    assert completed.exit_code == 0


def test_status_falls_back_to_table_parser_when_json_unsupported():
    """When `--output json` is unrecognized, retry without it and parse the table."""
    client = OsmoClient(profile="prod")
    json_attempt = _completed(returncode=2, stderr="unknown flag --output")
    table_attempt = _completed(stdout=STATUS_TABLE_OK)
    with patch("subprocess.run", side_effect=[json_attempt, table_attempt]) as run:
        snap = client.status("my-wf-1")
    assert run.call_count == 2
    assert snap.workflow_id == "my-wf-1"
    assert len(snap.tasks) == 3
    seed44 = [t for t in snap.tasks if t.name.endswith("seed44")][0]
    assert seed44.status == "RUNNING"
    assert seed44.exit_code is None


def test_status_raises_on_real_failure():
    client = OsmoClient(profile="prod")
    with patch("subprocess.run", return_value=_completed(returncode=1, stderr="HTTP 401 Unauthorized")):
        with pytest.raises(OsmoAuthError):
            client.status("my-wf-1")
