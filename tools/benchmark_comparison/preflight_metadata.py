# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structured metadata probes and parsers used by benchmark preflight."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .manifest import SoftwareIdentity
from .models import Version

_METADATA_PREFIX = "__ISAACLAB_BENCHMARK_METADATA__"


def host_metadata_probe() -> str:
    """Return a dependency-free host identity probe."""
    script = (
        "import json, os, platform\n"
        "marker = 'ISAACLAB_BENCHMARK_HOST_METADATA'\n"
        "cpu = platform.processor()\n"
        "if not cpu:\n"
        "    try:\n"
        "        lines = open('/proc/cpuinfo', encoding='utf-8').read().splitlines()\n"
        "        cpu = next(line.split(':', 1)[1].strip() for line in lines if line.startswith('model name'))\n"
        "    except (OSError, StopIteration):\n"
        "        cpu = 'unknown'\n"
        "try:\n"
        "    power_profile = open("
        "'/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', encoding='utf-8'"
        ").read().strip() or None\n"
        "except OSError:\n"
        "    power_profile = None\n"
        "print('__ISAACLAB_BENCHMARK_METADATA__' + json.dumps({'hostname': platform.node(), 'os': platform.platform(), "
        "'cpu_model': cpu, 'logical_cpu_count': os.cpu_count() or 1, "
        "'cpu_power_profile': power_profile}, sort_keys=True))\n"
    )
    return f"exec({script!r})"


def software_metadata_probe() -> str:
    """Return an import-metadata probe for one pinned execution environment."""
    script = (
        "import importlib.metadata as metadata, importlib.util, json, os, pathlib, platform\n"
        "marker = 'ISAACLAB_BENCHMARK_SOFTWARE_METADATA'\n"
        "def dist(*names):\n"
        "    for name in names:\n"
        "        try:\n"
        "            return metadata.version(name)\n"
        "        except metadata.PackageNotFoundError:\n"
        "            pass\n"
        "    raise RuntimeError('missing distribution: ' + ', '.join(names))\n"
        "def isaacsim_version():\n"
        "    candidates = [\n"
        "        pathlib.Path(os.environ[name]) / 'VERSION'\n"
        "        for name in ('ISAAC_PATH', 'ISAACSIM_PATH')\n"
        "        if os.environ.get(name)\n"
        "    ]\n"
        "    spec = importlib.util.find_spec('isaacsim')\n"
        "    if spec is not None and spec.submodule_search_locations:\n"
        "        candidates.extend(pathlib.Path(root) / 'VERSION' for root in spec.submodule_search_locations)\n"
        "    for candidate in candidates:\n"
        "        try:\n"
        "            value = candidate.read_text(encoding='utf-8').strip()\n"
        "        except OSError:\n"
        "            continue\n"
        "        if value:\n"
        "            return value\n"
        "    raise RuntimeError('cannot resolve exact Isaac Sim VERSION')\n"
        "import torch\n"
        "print('__ISAACLAB_BENCHMARK_METADATA__' + json.dumps({\n"
        "    'isaac_lab': open('VERSION', encoding='utf-8').read().strip(),\n"
        "    'isaac_sim': isaacsim_version(),\n"
        "    'python': platform.python_version(),\n"
        "    'pytorch': torch.__version__,\n"
        "    'rsl_rl': dist('rsl-rl-lib', 'rsl-rl'),\n"
        "    'cuda': torch.version.cuda,\n"
        "}, sort_keys=True))\n"
    )
    return f"exec({script!r})"


def metadata_object(stdout: str, name: str) -> Mapping[str, object]:
    """Parse one strict JSON object emitted by a metadata probe."""
    marked = [line.removeprefix(_METADATA_PREFIX) for line in stdout.splitlines() if line.startswith(_METADATA_PREFIX)]
    if len(marked) != 1:
        raise ValueError(f"{name} metadata must contain exactly one marked JSON object")
    try:
        value = json.loads(marked[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} marked metadata is not a JSON object") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} metadata must be an object")
    return value


def metadata_text(value: Mapping[str, object], field: str, name: str) -> str:
    """Read one required nonempty text metadata field."""
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{name} metadata {field} must be a nonempty string")
    return result


def optional_metadata_text(value: Mapping[str, object], field: str, name: str) -> str | None:
    """Read one optional nonempty text metadata field."""
    result = value.get(field)
    if result is not None and (not isinstance(result, str) or not result):
        raise ValueError(f"{name} metadata {field} must be null or a nonempty string")
    return result


def metadata_int(value: Mapping[str, object], field: str, name: str) -> int:
    """Read one required positive integer metadata field."""
    result = value.get(field)
    if not isinstance(result, int) or isinstance(result, bool) or result < 1:
        raise ValueError(f"{name} metadata {field} must be a positive integer")
    return result


def parse_nvidia_identity(stdout: str) -> tuple[str, str, int, str]:
    """Parse physical GPU 0 model, driver, idle memory [MiB], and UUID."""
    rows = [tuple(item.strip() for item in line.split(",")) for line in stdout.splitlines() if line.strip()]
    try:
        if len(rows) != 1 or len(rows[0]) != 5:
            raise ValueError
        model, driver, memory, utilization, gpu_uuid = rows[0]
        if not model or not driver or re.fullmatch(r"GPU-[0-9A-Za-z-]+", gpu_uuid) is None:
            raise ValueError
        idle_memory = int(memory)
        int(utilization)
    except (ValueError, IndexError) as error:
        raise ValueError(f"NVIDIA SMI output is malformed: {stdout!r}") from error
    return model, driver, idle_memory, gpu_uuid


def software_metadata(stdout: str, version: Version) -> tuple[SoftwareIdentity, str | None]:
    """Parse the exact software stack returned by one version probe."""
    value = metadata_object(stdout, version.value)
    expected = {"isaac_lab", "isaac_sim", "python", "pytorch", "rsl_rl", "cuda"}
    if set(value) != expected:
        raise ValueError(f"{version.value} software metadata must contain exactly {sorted(expected)}")
    cuda = value["cuda"]
    if cuda is not None and (not isinstance(cuda, str) or not cuda):
        raise ValueError(f"{version.value} software metadata cuda must be null or nonempty text")
    return (
        SoftwareIdentity(
            isaac_lab=metadata_text(value, "isaac_lab", version.value),
            isaac_sim=metadata_text(value, "isaac_sim", version.value),
            python=metadata_text(value, "python", version.value),
            pytorch=metadata_text(value, "pytorch", version.value),
            rsl_rl=metadata_text(value, "rsl_rl", version.value),
        ),
        cuda,
    )
