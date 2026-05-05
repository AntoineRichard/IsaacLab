# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest

from tools.odin.bifrost.config import BifrostConfig, BifrostConfigError, load_bifrost_config

VALID_YAML = """\
osmo_profile: prod
pool: rtx-pro-6000-eval
priority: NORMAL
image:
  reference: nvcr.io/nvidia/isaac-lab:2.2.0
  pull_credential: ngc-readonly
defaults:
  resources:
    cpu: 16
    gpu: 1
    memory: 64Gi
    storage: 64Gi
    platform: rtx-pro-6000
  exec_timeout: 14400
  queue_timeout: 7200
retry:
  reschedule_codes: "3001-3006"
  restart_codes: ""
bundle_dataset_prefix: odin
code_delivery:
  mode: files_upload
  source_root: tools/odin
"""


def test_load_valid_config(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML)
    cfg = load_bifrost_config(p)
    assert isinstance(cfg, BifrostConfig)
    assert cfg.pool == "rtx-pro-6000-eval"
    assert cfg.image.reference == "nvcr.io/nvidia/isaac-lab:2.2.0"
    assert cfg.image.pull_credential == "ngc-readonly"
    assert cfg.defaults.resources.gpu == 1
    assert cfg.code_delivery.mode == "files_upload"
    assert cfg.priority == "NORMAL"


def test_priority_must_be_in_enum(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("priority: NORMAL", "priority: URGENT"))
    with pytest.raises(BifrostConfigError, match="priority"):
        load_bifrost_config(p)


def test_code_delivery_mode_must_be_in_enum(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("mode: files_upload", "mode: ftp_upload"))
    with pytest.raises(BifrostConfigError, match="code_delivery.mode"):
        load_bifrost_config(p)


def test_missing_required_field_raises(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("pool: rtx-pro-6000-eval\n", ""))
    with pytest.raises(BifrostConfigError, match="pool"):
        load_bifrost_config(p)


def test_pull_credential_optional(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("  pull_credential: ngc-readonly\n", ""))
    cfg = load_bifrost_config(p)
    assert cfg.image.pull_credential is None


def test_resources_must_have_all_keys(tmp_path: Path):
    p = tmp_path / "bifrost-osmo.yaml"
    p.write_text(VALID_YAML.replace("    platform: rtx-pro-6000\n", ""))
    with pytest.raises(BifrostConfigError, match="resources.platform"):
        load_bifrost_config(p)
