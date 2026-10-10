"""Broker outage behaviour of the retail publisher (vision_base client).

Contract (see retail_core/publisher.py):
* after every reconnect the first PUBLISH is the retained ``online`` status;
* results produced while the session is down are not queued: only the latest
  payload per topic is held, and it is replayed after ``online`` only if it is
  at most ``replay_max_age_s`` old at reconnect time, otherwise dropped;
* ``publish()`` never blocks during an outage; it raises ``OSError``.
"""
from __future__ import annotations

import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from retail_recording_broker import RecordingBroker, packet_type  # noqa: E402
from test_wire_golden import parse_publish  # noqa: E402

from retail_core.publisher import REPLAY_MAX_AGE_S, MqttPublisher, status_topic  # noqa: E402

STATUS = status_topic("site")
CAM_A = "site/retail-vision/results/cam-a"
CAM_B = "site/retail-vision/results/cam-b"


def make(broker, **kw):
    pub = MqttPublisher({"host": "127.0.0.1", "port": broker.port, "client_id": "outage"},
                        status_topic=STATUS, reconnect_min_s=0.05, reconnect_max_s=0.2, **kw)
    pub.connect(timeout_s=5.0)
    return pub


def session(broker, conn):
    """[(kind, topic, payload, retain)] of one connection, PINGREQ excluded."""
    out = []
    for idx, raw in broker.snapshot():
        if idx != conn or packet_type(raw) == 12:
            continue
        if packet_type(raw) == 3:
            topic, payload, qos, retain = parse_publish(raw)
            assert qos == 0
            out.append(("PUBLISH", topic, payload, retain))
        else:
            out.append((packet_type(raw), None, raw, None))
    return out


def wait_disconnected(pub, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while pub.client.connected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not pub.client.connected


def outage(broker, pub):
    broker.stop()
    wait_disconnected(pub)


def wait_session(broker, conn, n_publishes):
    assert broker.wait_for(lambda pkts: sum(
        1 for i, r in pkts if i == conn and packet_type(r) == 3) >= n_publishes, 5.0)


def test_default_bound():
    assert REPLAY_MAX_AGE_S == 2.0


def test_reconnect_replays_status_then_fresh_latest_result():
    broker = RecordingBroker().start()
    pub = make(broker)
    try:
        pub.publish(CAM_A, {"n": 1})
        wait_session(broker, 0, 2)
        outage(broker, pub)
        t0 = time.monotonic()
        for n in (2, 3, 4):
            with pytest.raises(OSError):
                pub.publish(CAM_A, {"n": n})
        pub.publish_offline()           # not sent: no session
        assert time.monotonic() - t0 < 0.5, "publish blocked during outage"
        broker.start()                  # same port
        wait_session(broker, 1, 2)
        first, second = session(broker, 0), session(broker, 1)
        assert first[0][1] is None and first[0][2] == second[0][2], \
            "reconnect CONNECT differs from the first one"
        assert second[1:] == [("PUBLISH", STATUS, b"online", True),
                              ("PUBLISH", CAM_A, b'{"n":4}', False)]
        stats = pub.stats()
        assert stats["replayed"] == 1 and stats["dropped_stale"] == 0
        assert stats["held_now"] == 0
        pub.publish(CAM_A, {"n": 5})
        wait_session(broker, 1, 3)
        assert session(broker, 1)[-1] == ("PUBLISH", CAM_A, b'{"n":5}', False)
    finally:
        pub.close()
        broker.stop()


def test_stale_results_dropped_status_still_replayed():
    broker = RecordingBroker().start()
    pub = make(broker, replay_max_age_s=0.3)
    try:
        wait_session(broker, 0, 1)
        outage(broker, pub)
        with pytest.raises(OSError):
            pub.publish(CAM_A, {"old": True})
        time.sleep(0.5)                 # older than the 0.3 s bound
        with pytest.raises(OSError):
            pub.publish(CAM_B, {"fresh": True})
        broker.start()
        wait_session(broker, 1, 2)
        time.sleep(0.2)                 # nothing else may follow
        assert session(broker, 1)[1:] == [("PUBLISH", STATUS, b"online", True),
                                          ("PUBLISH", CAM_B, b'{"fresh":true}', False)]
        stats = pub.stats()
        assert stats["replayed"] == 1 and stats["dropped_stale"] == 1
    finally:
        pub.close()
        broker.stop()


def test_initial_connect_failure_then_background_connect():
    broker = RecordingBroker().start()
    port = broker.port
    broker.stop()
    pub = MqttPublisher({"host": "127.0.0.1", "port": port, "client_id": "late"},
                        status_topic=STATUS, reconnect_min_s=0.05, reconnect_max_s=0.2)
    try:
        with pytest.raises(OSError):
            pub.connect(timeout_s=1.0)
        with pytest.raises(OSError):
            pub.publish(CAM_A, {"n": 1})
        broker.start()
        wait_session(broker, 0, 2)
        assert session(broker, 0)[1:] == [("PUBLISH", STATUS, b"online", True),
                                          ("PUBLISH", CAM_A, b'{"n":1}', False)]
    finally:
        pub.close()
        broker.stop()


def test_graceful_close_sends_offline_then_disconnect():
    broker = RecordingBroker().start()
    pub = make(broker)
    try:
        pub.publish_offline()
        pub.close()
        assert broker.wait_for(lambda p: p and packet_type(p[-1][1]) == 14)
        seq = session(broker, 0)
        assert seq[1:] == [("PUBLISH", STATUS, b"online", True),
                           ("PUBLISH", STATUS, b"offline", True),
                           (14, None, b"\xe0\x00", None)]
    finally:
        broker.stop()
