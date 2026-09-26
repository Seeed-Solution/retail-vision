"""MQTT client + PublishWorker tests (M1.10) against the fake broker, plus
optional real-broker tests selected with ``-m broker`` and ``VB_TEST_BROKER``
(host:port)."""
from __future__ import annotations

import json
import os
import pathlib
import socket
import ssl
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fake_broker import FakeBroker  # noqa: E402
from vision_base.mqtt import MqttClient, Will, topic_matches  # noqa: E402
from vision_base.publish import PublishWorker  # noqa: E402

WILL = Will("dev/status", b'{"online":false}', qos=1, retain=True)


def make_client(broker, **kw):
    return MqttClient(broker.host, broker.port, "test-client", keepalive_s=kw.pop("keepalive_s", 60),
                      reconnect_min_s=0.05, reconnect_max_s=0.2, will=WILL, **kw)


@pytest.fixture
def broker():
    b = FakeBroker().start()
    yield b
    b.stop()


@pytest.fixture
def client(broker):
    c = make_client(broker)
    yield c
    c.close()


def wait_until(cond, timeout_s=5.0, msg="condition"):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"timeout waiting for {msg}")


# ---------------------------------------------------------------- fake broker

def test_connect_lwt_bytes_and_nodelay(broker, client):
    assert client.connect(timeout_s=5.0) is True
    wait_until(lambda: broker.connects, msg="CONNECT")
    info = broker.connects[0]
    assert info["protocol"] == "MQTT" and info["level"] == 4
    assert info["clean_session"] is True
    assert info["keepalive"] == 60
    assert info["will"] == {"topic": "dev/status", "payload": '{"online":false}'}
    assert info["will_qos"] == 1
    assert info["will_retain"] is True
    # TCP_NODELAY is set on the client socket
    assert client._sock is not None
    assert client._sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0


def test_qos1_publish_gets_puback(broker, client):
    assert client.connect(timeout_s=5.0)
    assert client.publish("t/1", "hello", qos=1) is True
    m = broker.wait_publish("t/1")
    assert m is not None
    assert m["payload"] == b"hello" and m["qos"] == 1
    assert client.stats()["published"] >= 1


def test_subscribe_downlink_and_puback(broker, client):
    assert client.connect(timeout_s=5.0)
    got = []
    client.subscribe("cmd/#", 1, lambda t, p, r: got.append((t, p, r)))
    wait_until(lambda: "cmd/#" in broker.subscribes, msg="SUBSCRIBE")
    broker.publish_down("cmd/control", b'{"x":1}', qos=1)
    wait_until(lambda: got, msg="downlink")
    assert got[0] == ("cmd/control", b'{"x":1}', False)
    wait_until(lambda: broker.pubacks == 1, msg="client PUBACK")


def test_pingreq_on_keepalive(broker):
    c = make_client(broker, keepalive_s=1)
    try:
        assert c.connect(timeout_s=5.0)
        time.sleep(1.6)
        assert broker.pingreqs >= 1
    finally:
        c.close()


def test_auto_reconnect_after_drop(broker, client):
    assert client.connect(timeout_s=5.0)
    reasons = []
    client.on_disconnect = reasons.append
    n0 = len(broker.connects)
    broker.kill_all()
    wait_until(lambda: not client.connected, msg="disconnect")
    wait_until(lambda: client.connected and len(broker.connects) > n0,
               timeout_s=10.0, msg="reconnect")
    assert reasons


def test_puback_timeout(broker):
    b = FakeBroker(no_puback=True).start()
    try:
        c = make_client(b)
        try:
            assert c.connect(timeout_s=5.0)
            t0 = time.monotonic()
            ok = c.publish("t/1", b"x", qos=1, timeout_s=0.5)
            dt = time.monotonic() - t0
            assert ok is False
            assert 0.3 <= dt <= 0.7, dt
            assert c.stats()["puback_timeouts"] == 1
        finally:
            c.close()
    finally:
        b.stop()


def test_publish_not_connected_immediate_false(broker):
    c = make_client(broker)
    try:
        t0 = time.monotonic()
        assert c.publish("t/1", "x", qos=0) is False
        assert c.publish("t/1", "x", qos=1) is False
        assert time.monotonic() - t0 < 0.2
    finally:
        c.close()


def test_disconnect_during_puback_wait(broker):
    b = FakeBroker(no_puback=True, close_on_publish=True).start()
    try:
        c = make_client(b)
        try:
            assert c.connect(timeout_s=5.0)
            t0 = time.monotonic()
            ok = c.publish("t/1", b"x", qos=1, timeout_s=5.0)
            dt = time.monotonic() - t0
            assert ok is False
            assert dt < 2.0, dt
        finally:
            c.close()
    finally:
        b.stop()


def test_wildcard_matching(broker, client):
    assert client.connect(timeout_s=5.0)
    hits = []
    client.subscribe("+/x", 1, lambda t, p, r: hits.append(t))
    client.subscribe("a/#", 1, lambda t, p, r: hits.append(t))
    client.subscribe("#", 1, lambda t, p, r: hits.append(t))
    wait_until(lambda: len(broker.subscribes) >= 3, msg="subscribes")
    for t in ("b/x", "c/x", "d/x",        # '+' x3
              "a", "a/b", "a/b/c",        # '#' in a/# x3
              "$SYS/x"):                  # leading $ not matched by wildcard
        broker.publish_down(t, b"p", qos=0)
    wait_until(lambda: len(hits) >= 8, msg="wildcard deliveries")
    time.sleep(0.2)
    assert "$SYS/x" not in hits
    assert {"b/x", "c/x", "d/x", "a", "a/b", "a/b/c"} <= set(hits)
    # a/b/c matched by both a/# and # (two callbacks)
    assert hits.count("a/b/c") == 2


def test_resubscribe_after_reconnect(broker, client):
    assert client.connect(timeout_s=5.0)
    got = []
    client.subscribe("cmd/#", 1, lambda t, p, r: got.append(t))
    wait_until(lambda: "cmd/#" in broker.subscribes, msg="subscribe")
    broker.kill_all()
    wait_until(lambda: client.connected, timeout_s=10.0, msg="reconnect")
    time.sleep(0.3)  # let resubscribe finish
    assert len([s for s in broker.subscribes if s == "cmd/#"]) == 2
    broker.publish_down("cmd/re", b"hi", qos=1)
    wait_until(lambda: got == ["cmd/re"], msg="delivery after resubscribe")


def test_qos1_publish_in_callback_raises(broker, client):
    assert client.connect(timeout_s=5.0)
    errs = []

    def cb(topic, payload, retain):
        if payload == b"deny":
            try:
                client.publish("t/x", "m", qos=1, timeout_s=1.0)
                errs.append("no-error")
            except RuntimeError:
                errs.append("runtime-error")

    client.subscribe("in/cb", 1, cb)
    wait_until(lambda: "in/cb" in broker.subscribes, msg="subscribe")
    broker.publish_down("in/cb", b"deny", qos=1)
    wait_until(lambda: errs, msg="callback ran")
    assert errs == ["runtime-error"]


def test_qos0_publish_in_callback_allowed(broker, client):
    assert client.connect(timeout_s=5.0)
    res = []
    client.subscribe("in/cb0", 1, lambda t, p, r: res.append(
        client.publish("t/from-cb", "m", qos=0)))
    wait_until(lambda: "in/cb0" in broker.subscribes, msg="subscribe")
    broker.publish_down("in/cb0", b"p", qos=1)
    wait_until(lambda: res, msg="callback ran")
    assert res == [True]
    assert broker.wait_publish("t/from-cb") is not None


def test_inflight_full_timeout(broker):
    b = FakeBroker(no_puback=True).start()
    try:
        c = make_client(b, max_inflight=1)
        try:
            assert c.connect(timeout_s=5.0)
            first = threading.Thread(
                target=lambda: c.publish("t/a", b"1", qos=1, timeout_s=3.0))
            first.start()
            wait_until(lambda: c.stats()["inflight"] == 1, msg="inflight=1")
            t0 = time.monotonic()
            assert c.publish("t/b", b"2", qos=1, timeout_s=0.5) is False
            dt = time.monotonic() - t0
            assert 0.3 <= dt <= 0.9, dt
            assert len([m for m in b.published if m["topic"] == "t/b"]) == 0
            first.join(timeout=5.0)
        finally:
            c.close()
    finally:
        b.stop()


def test_publish_validation_errors(client):
    with pytest.raises(ValueError):
        client.publish("t", "x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError):
        client.publish("t", "x", qos=2)
    with pytest.raises(TypeError):
        client.publish("t", 123)


def test_publish_worker_drop_oldest(broker, client):
    w = PublishWorker(client, queue_size=3)
    for i in range(5):
        w.submit("w/t", {"i": i})
    assert w.stats()["queued"] == 3
    assert w.stats()["dropped"] == 2
    with w._cv:
        items = list(w._q)  # (topic, payload, qos, retain)
    assert [x[1]["i"] for x in items] == [2, 3, 4]  # oldest dropped


def test_publish_worker_sends_json(broker, client):
    assert client.connect(timeout_s=5.0)
    w = PublishWorker(client, queue_size=16)
    w.start()
    w.submit("w/j", {"schema": "vb.ack/1", "ok": True})
    m = broker.wait_publish("w/j")
    assert m is not None
    assert json.loads(m["payload"]) == {"schema": "vb.ack/1", "ok": True}
    w.close()


def test_topic_matches_unit():
    assert topic_matches("+/x", "b/x")
    assert topic_matches("+/x", "c/x")
    assert not topic_matches("+/x", "a/b/x")
    assert topic_matches("a/#", "a")
    assert topic_matches("a/#", "a/b")
    assert topic_matches("a/#", "a/b/c")
    assert not topic_matches("a/#", "b/a")
    assert not topic_matches("#", "$SYS/x")
    assert not topic_matches("+/x", "$SYS/x")
    assert topic_matches("$SYS/x", "$SYS/x")


# ------------------------------------------------------------------- TLS

def _gen_certs(tmpdir: pathlib.Path):
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(tmpdir / "ca.key"), "-out", str(tmpdir / "ca.crt"),
         "-days", "2", "-subj", "/CN=vb-test-ca",
         "-addext", "basicConstraints=critical,CA:TRUE",
         "-addext", "keyUsage=critical,keyCertSign,cRLSign"], check=True, capture_output=True)
    subprocess.run(
        ["openssl", "req", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(tmpdir / "srv.key"), "-out", str(tmpdir / "srv.csr"),
         "-subj", "/CN=127.0.0.1"], check=True, capture_output=True)
    (tmpdir / "srv.ext").write_text(
        "subjectAltName=IP:127.0.0.1\n"
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n")
    subprocess.run(
        ["openssl", "x509", "-req", "-in", str(tmpdir / "srv.csr"),
         "-CA", str(tmpdir / "ca.crt"), "-CAkey", str(tmpdir / "ca.key"),
         "-CAcreateserial", "-out", str(tmpdir / "srv.crt"), "-days", "2",
         "-extfile", str(tmpdir / "srv.ext")], check=True, capture_output=True)
    return tmpdir / "ca.crt", tmpdir / "srv.crt", tmpdir / "srv.key"


def _tls_broker(ca_crt, srv_crt, srv_key) -> FakeBroker:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(srv_crt), str(srv_key))
    return FakeBroker(tls_ctx=ctx).start()


def test_tls_with_self_signed_ca(tmp_path):
    ca, crt, key = _gen_certs(tmp_path)
    b = _tls_broker(ca, crt, key)
    try:
        c = MqttClient("127.0.0.1", b.port, "tls-client", tls=True, ca_file=str(ca),
                       will=None, reconnect_min_s=0.05, reconnect_max_s=0.1)
        try:
            assert c.connect(timeout_s=5.0)
            assert c.publish("tls/t", "hello", qos=1) is True
            m = b.wait_publish("tls/t")
            assert m is not None and m["payload"] == b"hello"
        finally:
            c.close()
    finally:
        b.stop()


def test_tls_wrong_ca_fails(tmp_path):
    ca, crt, key = _gen_certs(tmp_path)
    # second, different CA
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
         "-keyout", str(tmp_path / "ca2.key"), "-out", str(tmp_path / "ca2.crt"),
         "-days", "2", "-subj", "/CN=other-ca"], check=True, capture_output=True)
    b = _tls_broker(ca, crt, key)
    try:
        c = MqttClient("127.0.0.1", b.port, "tls-client2", tls=True,
                       ca_file=str(tmp_path / "ca2.crt"), will=None,
                       reconnect_min_s=0.05, reconnect_max_s=0.1)
        try:
            assert c.connect(timeout_s=2.0) is False
            assert c.publish("tls/t", "x") is False
        finally:
            c.close()
    finally:
        b.stop()


# ------------------------------------------------------------- real broker

VB_TEST_BROKER = os.environ.get("VB_TEST_BROKER", "")
needs_broker = pytest.mark.skipif(not VB_TEST_BROKER, reason="set VB_TEST_BROKER=host:port")


@needs_broker
@pytest.mark.broker
def test_real_broker_basic():
    host, port = VB_TEST_BROKER.rsplit(":", 1)
    c = MqttClient(host, int(port), "vb-test-real", will=Will("R/status", b'{"online":false}'))
    try:
        assert c.connect(timeout_s=5.0)
        assert c.publish("R/events/x", b"hello", qos=1) is True
        got = []
        c.subscribe("R/cmd/#", 1, lambda t, p, r: got.append(t))
        time.sleep(0.5)
        assert c.publish("R/cmd/control", b"loopback", qos=1)
        wait_until(lambda: got, timeout_s=5.0, msg="loopback delivery")
    finally:
        c.close()


def test_publish_worker_retry_once_backoff_and_head_check():
    """Review 2026-09-26: failed head is retried once after a growing backoff,
    then dropped; a head already dropped by submit() is not popped twice."""
    import threading as _th
    from vision_base.publish import PublishWorker

    class FlakyClient:
        def __init__(self):
            self.calls = []
            self.lock = _th.Lock()
        def publish(self, topic, data, qos=0, retain=False):
            with self.lock:
                self.calls.append(topic)
            return False

    waits = []
    w = PublishWorker(FlakyClient(), queue_size=4)
    w._stop.wait = lambda t: (waits.append(t), False)[1]
    w.submit("a", {"i": 1})
    w.submit("b", {"i": 2})
    w.start()
    deadline = time.time() + 3
    while w.stats()["queued"] and time.time() < deadline:
        time.sleep(0.01)
    w.close()
    assert w.client.calls[:4] == ["a", "a", "b", "b"]   # each tried twice
    assert waits[:2] == [1.0, 2.0]                     # backoff grows across failures
    assert w.stats()["failed"] == 2 and w.stats()["queued"] == 0

    # head replaced while waiting: the new head must survive
    w2 = PublishWorker(FlakyClient(), queue_size=1)
    replaced = []
    def wait_and_replace(t):
        if not replaced:              # only while "a" is being retried
            replaced.append(True)
            w2.submit("c", {"i": 3})  # queue full -> drops current head "a"
        return False
    w2._stop.wait = wait_and_replace
    w2.submit("a", {"i": 1})
    w2.start()
    deadline = time.time() + 1
    while "c" not in w2.client.calls and time.time() < deadline:
        time.sleep(0.01)
    w2.close()
    assert w2.dropped == 1
    assert "c" in w2.client.calls
