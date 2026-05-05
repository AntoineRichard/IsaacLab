# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin Bifrost — OSMO-backed dispatch path.

Peer to :mod:`tools.odin.asgard`. Bifrost submits eval jobs as a
single OSMO workflow with N parallel tasks; bundles return via
``osmo dataset download`` into the canonical
``odin_runs/<dispatch_id>/<run_id>/`` layout.

See ``docs/superpowers/specs/2026-05-05-odin-bifrost-osmo-backend-design.md``.
"""
