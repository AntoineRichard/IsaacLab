# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Odin log_tail helper."""

from tools.odin.common.log_tail import DEFAULT_TAIL_BYTES, tail_bytes


def test_tail_bytes_shorter_than_limit():
    assert tail_bytes(b"hello") == b"hello"


def test_tail_bytes_longer_than_limit():
    data = b"x" * (DEFAULT_TAIL_BYTES + 100)
    out = tail_bytes(data)
    assert len(out) == DEFAULT_TAIL_BYTES
    assert out == b"x" * DEFAULT_TAIL_BYTES


def test_tail_bytes_custom_limit():
    assert tail_bytes(b"abcdef", max_bytes=3) == b"def"
