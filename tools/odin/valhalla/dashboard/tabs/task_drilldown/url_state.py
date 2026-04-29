# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""URL query-string parsing for Tab B's deep-linked picker state.

The query string carries ``?task=<task_id>&framework=<rsl_rl|skrl>
&backend=<physx|newton>``. Empty fields are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode

__all__ = ["TaskSelection", "parse_query_string", "serialize"]


@dataclass(frozen=True)
class TaskSelection:
    """One picker selection — pinned by URL state."""

    task: str | None
    framework: str | None
    backend: str | None


def parse_query_string(search: str) -> TaskSelection:
    """Parse Dash's ``dcc.Location.search`` into a :class:`TaskSelection`.

    ``search`` may start with ``'?'`` (Dash convention); both forms accepted.
    Duplicate keys take the last value (standard query-string semantics).
    """
    raw = search.lstrip("?")
    pairs = parse_qs(raw, keep_blank_values=False)
    return TaskSelection(
        task=_last(pairs, "task"),
        framework=_last(pairs, "framework"),
        backend=_last(pairs, "backend"),
    )


def serialize(selection: TaskSelection) -> str:
    """Return a query string starting with '?'.

    Empty / None fields are omitted; if all three are None, returns ''.
    """
    fields = [
        ("task", selection.task),
        ("framework", selection.framework),
        ("backend", selection.backend),
    ]
    populated = [(k, v) for k, v in fields if v is not None and v != ""]
    if not populated:
        return ""
    return "?" + urlencode(populated)


def _last(pairs: dict[str, list[str]], key: str) -> str | None:
    values = pairs.get(key)
    if not values:
        return None
    return values[-1]
