# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heartbeat-thread unit tests for :func:`~tools.odin.asgard.worker._heartbeat_loop`."""

from __future__ import annotations

import queue
import threading
import time

from tools.odin.asgard.worker import StateEvent, _heartbeat_loop


def _drain(q: queue.Queue) -> list[StateEvent]:
    out: list[StateEvent] = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def test_heartbeat_loop_emits_one_event_per_inflight_run_id_per_tick():
    state_chan: queue.Queue = queue.Queue()
    inflight = {"run-a": object(), "run-b": object()}
    stop = threading.Event()

    t = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "host_name": "host-x",
            "state_chan": state_chan,
            "inflight_view": lambda: list(inflight.keys()),
            "interval_s": 0.05,
            "stop_event": stop,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.16)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()

    events = _drain(state_chan)
    runs = {e.run_id for e in events}
    assert runs == {"run-a", "run-b"}
    for e in events:
        assert e.host == "host-x"
        assert e.transition == "heartbeat"
        assert e.at is not None


def test_heartbeat_loop_stops_promptly_on_signal():
    state_chan: queue.Queue = queue.Queue()
    stop = threading.Event()

    t = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "host_name": "host-x",
            "state_chan": state_chan,
            "inflight_view": lambda: [],
            "interval_s": 5.0,
            "stop_event": stop,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_heartbeat_loop_pre_set_stop_is_noop():
    """A loop entered with stop already set returns immediately, no events."""
    stop = threading.Event()
    stop.set()
    state_chan: queue.Queue = queue.Queue()

    _heartbeat_loop(
        host_name="host-x",
        state_chan=state_chan,
        inflight_view=lambda: ["run-a"],
        interval_s=1.0,
        stop_event=stop,
    )
    assert state_chan.qsize() == 0


def test_heartbeat_loop_no_inflight_jobs_emits_nothing():
    """An empty inflight view ticks without producing events."""
    state_chan: queue.Queue = queue.Queue()
    stop = threading.Event()

    t = threading.Thread(
        target=_heartbeat_loop,
        kwargs={
            "host_name": "host-x",
            "state_chan": state_chan,
            "inflight_view": lambda: [],
            "interval_s": 0.05,
            "stop_event": stop,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.12)
    stop.set()
    t.join(timeout=1.0)

    assert state_chan.qsize() == 0
