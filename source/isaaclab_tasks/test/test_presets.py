# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`isaaclab_tasks.utils.presets.has_physics_preset`."""

from dataclasses import dataclass, field

from isaaclab_tasks.utils.presets import has_physics_preset

# --- Synthetic fixtures -----------------------------------------------------
# We build minimal classes that imitate the shape `has_physics_preset` walks:
# - a "PhysicsCfg" object carrying named preset attributes,
# - a "SimCfg" wrapping a physics object,
# - an "EnvCfg" wrapping a SimCfg,
# - a "PresetCfg" wrapper (has both __dataclass_fields__ and a `default` attr
#   and NO `class_type` on its type — the exact gate the helper checks).


@dataclass
class _PhysicsWithNewton:
    newton: object = None


@dataclass
class _PhysicsWithOther:
    mjwarp: object = None


@dataclass
class _Sim:
    physics: object = None


@dataclass
class _EnvCfg:
    sim: _Sim = field(default_factory=_Sim)


@dataclass
class _PresetWrapper:
    """Imitates a top-level PresetCfg wrapper: has `default` attr, is a
    dataclass, but its type has no `class_type` attribute."""

    default: _EnvCfg = field(default_factory=_EnvCfg)


def test_has_physics_preset_newton_present():
    cfg = _EnvCfg(sim=_Sim(physics=_PhysicsWithNewton()))
    assert has_physics_preset(cfg, "newton") is True


def test_has_physics_preset_newton_absent():
    cfg = _EnvCfg(sim=_Sim(physics=_PhysicsWithOther()))
    assert has_physics_preset(cfg, "newton") is False


def test_has_physics_preset_no_physics():
    cfg = _EnvCfg(sim=_Sim(physics=None))
    assert has_physics_preset(cfg, "newton") is False


def test_has_physics_preset_dict_short_circuits():
    # Dicts are not unwrapped — caller must pass a raw config object.
    assert has_physics_preset({"sim": {"physics": {"newton": {}}}}, "newton") is False


def test_has_physics_preset_unwraps_top_level_preset():
    inner = _EnvCfg(sim=_Sim(physics=_PhysicsWithNewton()))
    wrapper = _PresetWrapper(default=inner)
    assert has_physics_preset(wrapper, "newton") is True


def test_has_physics_preset_other_preset_name():
    cfg = _EnvCfg(sim=_Sim(physics=_PhysicsWithOther()))
    assert has_physics_preset(cfg, "mjwarp") is True
    assert has_physics_preset(cfg, "newton") is False
