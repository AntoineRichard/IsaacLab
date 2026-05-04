# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-job timeout budgets keyed by (task, framework, gpu_class).

A single global ``--per-job-timeout`` is hopeless across IsaacLab's task
suite — Cartpole finishes in 2 minutes while Repose-Cube-Allegro wants 6
hours. The :class:`Budgets` table reads a yaml file and looks up the
right wall-clock budget for each (task, framework) pair, scaled by the
target host's GPU class.

Yaml shape::

    defaults:
      rsl_rl: 3600         # 1h fallback when a task is missing
      skrl: 7200           # 2h fallback (skrl tends heavier)
    budgets:               # per-task overrides
      Isaac-Cartpole-Direct-v0:
        rsl_rl: 600        # 10 minutes
      Isaac-Deploy-Reach-Rizon4s-v0:
        rsl_rl: 36000      # 10h
    gpu_multipliers:
      default: 1.5         # generous fallback for an unrecognised GPU
      blackwell-pro-5000: 1.0
      l40: 1.4

Lookup order:

1. ``budgets[task][framework]`` if present, else ``defaults[framework]``,
   else a hard 12h floor (so a misconfigured table can't kill jobs at
   start).
2. Multiply by ``gpu_multipliers[gpu_class]`` if present, else by
   ``gpu_multipliers["default"]`` (or 1.0 if neither is set).

The resulting timeout is always returned as an integer number of
seconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

__all__ = ["Budgets", "load_budgets", "parse_gpu_class"]


# Hard ceiling when a (task, fw) combo has no entry AND ``defaults`` is
# missing the framework — pick something obviously generous so a typo in
# the yaml never kills a job in the first second.
_HARD_FALLBACK_S = 12 * 3600


@dataclass(frozen=True)
class Budgets:
    """Resolved budget table loaded from yaml.

    Attributes:
        defaults: Per-framework fallback budgets in seconds.
        budgets: ``task_id → framework → seconds`` overrides.
        gpu_multipliers: ``gpu_class → float`` scale factors. Must
            include a ``"default"`` key for unknown classes (the loader
            adds ``{"default": 1.0}`` if absent).
    """

    defaults: dict[str, int] = field(default_factory=dict)
    budgets: dict[str, dict[str, int]] = field(default_factory=dict)
    gpu_multipliers: dict[str, float] = field(default_factory=dict)

    def lookup(self, task_id: str, framework: str, gpu_class: str | None) -> int:
        """Resolve the per-job timeout in seconds.

        Args:
            task_id: Gym task id (e.g. ``Isaac-Cartpole-Direct-v0``).
            framework: ``"rsl_rl"`` or ``"skrl"`` (or any key listed
                under ``defaults`` / ``budgets``).
            gpu_class: Output of :func:`parse_gpu_class` for the host the
                job is going to run on. ``None`` is allowed and routes
                through ``gpu_multipliers["default"]`` so jobs still get
                a sane budget when GPU detection fails.

        Returns:
            Timeout in whole seconds.
        """
        per_task = self.budgets.get(task_id, {})
        base = per_task.get(framework)
        if base is None:
            base = self.defaults.get(framework, _HARD_FALLBACK_S)
        mult = None
        if gpu_class is not None:
            mult = self.gpu_multipliers.get(gpu_class)
        if mult is None:
            mult = self.gpu_multipliers.get("default", 1.0)
        return int(base * mult)


def load_budgets(path: Path) -> Budgets:
    """Read a budget yaml and resolve it into a :class:`Budgets`.

    The loader is forgiving on missing top-level keys (each defaults to
    an empty dict) but raises on unreadable files so the operator gets a
    fast failure instead of silent fall-through to global defaults.

    Args:
        path: Yaml file path. Schema documented at module level.

    Returns:
        Loaded :class:`Budgets`.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        yaml.YAMLError: When the file is not valid yaml.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"budgets yaml not found: {path}")
    raw = yaml.safe_load(Path(path).read_text()) or {}
    defaults = dict(raw.get("defaults") or {})
    budgets = {task: dict(per_fw) for task, per_fw in (raw.get("budgets") or {}).items()}
    gpu_multipliers = dict(raw.get("gpu_multipliers") or {})
    if "default" not in gpu_multipliers:
        # Pass-through default keeps the math correct when the table
        # only declares one GPU class.
        gpu_multipliers["default"] = 1.0
    return Budgets(defaults=defaults, budgets=budgets, gpu_multipliers=gpu_multipliers)


# Match the model name out of ``nvidia-smi -L`` output.
_NVIDIA_SMI_LINE = re.compile(r"^GPU \d+:\s*NVIDIA\s+(?P<model>.+?)\s+\(UUID:")


def parse_gpu_class(stdout: str) -> str | None:
    """Extract a normalised GPU class string from ``nvidia-smi -L`` output.

    Examples:
        ``"GPU 0: NVIDIA RTX PRO 5000 Blackwell (UUID: ...)"``
            → ``"blackwell-pro-5000"``
        ``"GPU 0: NVIDIA L40 (UUID: ...)"``
            → ``"l40"``
        ``"GPU 0: NVIDIA A100-SXM4-80GB (UUID: ...)"``
            → ``"a100-sxm4-80gb"``

    Multi-GPU hosts get classified by the first device. Returns ``None``
    when the output contains no parseable GPU line (NVML wedge, empty
    output, etc.).

    Args:
        stdout: Output of ``nvidia-smi -L``.

    Returns:
        Lowercase, hyphen-joined GPU class string, or ``None``.
    """
    for line in stdout.splitlines():
        m = _NVIDIA_SMI_LINE.match(line)
        if not m:
            continue
        model = m.group("model")
        # Drop "RTX" / "GeForce" / "Tesla" prefixes that don't carry useful
        # signal for budget lookup; collapse spaces into hyphens; lowercase.
        tokens = [t for t in re.split(r"\s+", model) if t and t.upper() not in {"RTX", "GEFORCE", "TESLA"}]
        if not tokens:
            return None
        # If the last token is a known marketing tier word ("Blackwell",
        # "Ada", "Ampere"), prefer it as the leading discriminator so
        # families with the same numeric tier don't collide.
        family_words = {"BLACKWELL", "ADA", "AMPERE", "HOPPER", "TURING", "VOLTA"}
        if tokens[-1].upper() in family_words:
            tokens = [tokens[-1]] + tokens[:-1]
        return "-".join(t.lower() for t in tokens)
    return None
