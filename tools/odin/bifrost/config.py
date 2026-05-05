# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load and validate ``bifrost-osmo.yaml``.

See spec §5.1 for the schema. All errors raise :class:`BifrostConfigError`
with a key path so the operator knows which field to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "BifrostConfig",
    "BifrostConfigError",
    "ImageSpec",
    "ResourcesSpec",
    "DefaultsSpec",
    "RetrySpec",
    "CodeDeliverySpec",
    "load_bifrost_config",
]


_PRIORITIES = {"HIGH", "NORMAL", "LOW"}
_CODE_DELIVERY_MODES = {"files_upload", "rsync", "image_baked"}


class BifrostConfigError(ValueError):
    """Raised when ``bifrost-osmo.yaml`` fails validation."""


@dataclass(frozen=True)
class ImageSpec:
    reference: str
    pull_credential: str | None


@dataclass(frozen=True)
class ResourcesSpec:
    cpu: int
    gpu: int
    memory: str
    storage: str
    platform: str


@dataclass(frozen=True)
class DefaultsSpec:
    resources: ResourcesSpec
    exec_timeout: int
    queue_timeout: int


@dataclass(frozen=True)
class RetrySpec:
    reschedule_codes: str
    restart_codes: str


@dataclass(frozen=True)
class CodeDeliverySpec:
    mode: str  # files_upload | rsync | image_baked
    source_root: str


@dataclass(frozen=True)
class BifrostConfig:
    osmo_profile: str
    pool: str
    priority: str  # HIGH | NORMAL | LOW
    image: ImageSpec
    defaults: DefaultsSpec
    retry: RetrySpec
    bundle_dataset_prefix: str
    code_delivery: CodeDeliverySpec


def _require(d: dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise BifrostConfigError(f"missing required field: {ctx}.{key}" if ctx else f"missing required field: {key}")
    return d[key]


def _require_str(d: dict[str, Any], key: str, ctx: str) -> str:
    v = _require(d, key, ctx)
    if not isinstance(v, str) or not v:
        raise BifrostConfigError(f"{ctx}.{key} must be a non-empty string")
    return v


def _require_int(d: dict[str, Any], key: str, ctx: str) -> int:
    v = _require(d, key, ctx)
    if not isinstance(v, int) or isinstance(v, bool):
        raise BifrostConfigError(f"{ctx}.{key} must be an integer")
    return v


def load_bifrost_config(path: Path) -> BifrostConfig:
    """Load and validate ``bifrost-osmo.yaml``.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        A fully populated :class:`BifrostConfig`.

    Raises:
        BifrostConfigError: If any required field is missing or invalid.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise BifrostConfigError("top-level YAML must be a mapping")

    osmo_profile = _require_str(raw, "osmo_profile", "")
    pool = _require_str(raw, "pool", "")
    priority = _require_str(raw, "priority", "")
    if priority not in _PRIORITIES:
        raise BifrostConfigError(f"priority must be one of {sorted(_PRIORITIES)}; got {priority!r}")

    image_d = _require(raw, "image", "")
    if not isinstance(image_d, dict):
        raise BifrostConfigError("image must be a mapping")
    image = ImageSpec(
        reference=_require_str(image_d, "reference", "image"),
        pull_credential=image_d.get("pull_credential"),
    )

    defaults_d = _require(raw, "defaults", "")
    if not isinstance(defaults_d, dict):
        raise BifrostConfigError("defaults must be a mapping")
    res_d = _require(defaults_d, "resources", "defaults")
    if not isinstance(res_d, dict):
        raise BifrostConfigError("defaults.resources must be a mapping")
    resources = ResourcesSpec(
        cpu=_require_int(res_d, "cpu", "defaults.resources"),
        gpu=_require_int(res_d, "gpu", "defaults.resources"),
        memory=_require_str(res_d, "memory", "defaults.resources"),
        storage=_require_str(res_d, "storage", "defaults.resources"),
        platform=_require_str(res_d, "platform", "defaults.resources"),
    )
    defaults = DefaultsSpec(
        resources=resources,
        exec_timeout=_require_int(defaults_d, "exec_timeout", "defaults"),
        queue_timeout=_require_int(defaults_d, "queue_timeout", "defaults"),
    )

    retry_d = raw.get("retry") or {}
    retry = RetrySpec(
        reschedule_codes=str(retry_d.get("reschedule_codes") or ""),
        restart_codes=str(retry_d.get("restart_codes") or ""),
    )

    bundle_dataset_prefix = _require_str(raw, "bundle_dataset_prefix", "")

    cd_d = _require(raw, "code_delivery", "")
    if not isinstance(cd_d, dict):
        raise BifrostConfigError("code_delivery must be a mapping")
    cd_mode = _require_str(cd_d, "mode", "code_delivery")
    if cd_mode not in _CODE_DELIVERY_MODES:
        raise BifrostConfigError(
            f"code_delivery.mode must be one of {sorted(_CODE_DELIVERY_MODES)}; got {cd_mode!r}"
        )
    code_delivery = CodeDeliverySpec(
        mode=cd_mode,
        source_root=_require_str(cd_d, "source_root", "code_delivery"),
    )

    return BifrostConfig(
        osmo_profile=osmo_profile,
        pool=pool,
        priority=priority,
        image=image,
        defaults=defaults,
        retry=retry,
        bundle_dataset_prefix=bundle_dataset_prefix,
        code_delivery=code_delivery,
    )
