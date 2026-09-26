"""vb-runtime --standalone --output mqtt tests (spec BASE-1 §8 M1.19).

Runs the real binary via VB_RUNTIME_BIN against the M1.10 fake broker; the
``broker``-marked test additionally needs a real mosquitto
(VB_TEST_BROKER=host:port). TLS cases generate a self-signed CA + server
certificate and serve the fake broker over TLS.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import ssl
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fake_broker import FakeBroker  # noqa: E402
from vision_base.mqtt import MqttClient  # noqa: E402

BIN = os.environ.get("VB_RUNTIME_BIN")
NOTLS_BIN = os.environ.get("VB_RUNTIME_BIN_NOTLS", "")
ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "contracts" / "fixtures" / "vb" / "standalone_synthetic.json"

pytestmark = pytest.mark.native


def wait_until(cond, timeout_s=10.0, msg="condition"):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {msg}")


def write_config(tmp_path, host, port, root="R", **mqtt):
    cfg = json.loads(FIXTURE.read_text())
    m = {"host": host, "port": port, "topic_root": root,
         "keepalive_s": mqtt.pop("keepalive_s", 30), **mqtt}
    cfg["mqtt"] = m
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def start(tmp_path, host, port, *extra, config=None, root="R"):
    cfg = config or write_config(tmp_path, host, port, root)
    proc = subprocess.Popen(
        [BIN, "--standalone", "--config", str(cfg), "--output", "mqtt",
         "--status-every", "2", *extra],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def stop(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=10.0)
    return out.decode(), err.decode()


# ------------------------------------------------------------ CONNECT / LWT

def test_connect_lwt_and_qos(tmp_path):
    broker = FakeBroker().start()
    proc = start(tmp_path, broker.host, broker.port)
    try:
        wait_until(lambda: broker.connects, msg="CONNECT")
        info = broker.connects[0]
        assert info["protocol"] == "MQTT" and info["level"] == 4
        assert info["clean_session"] is True
        will_payload = json.loads(info["will"]["payload"])
        assert info["will"]["topic"] == "R/status"
        assert will_payload["schema"] == "vb.status/1" and will_payload["online"] is False
        assert info["will_qos"] == 1
        assert info["will_retain"] is True

        # events: QoS1, PUBACK -> exactly one copy of each (seq, track)
        m = broker.wait_publish("R/events/cam-a")
        assert m is not None and m["qos"] == 1
        ev = json.loads(m["payload"])
        assert ev["schema"] == "vb.event/1"
        time.sleep(2.0)
        keyed = {}
        for p in broker.published:
            if p["topic"].startswith("R/events/"):
                o = json.loads(p["payload"])
                keyed[(p["topic"], o["seq"], o["track_id"], o["type"])] = \
                    keyed.get((p["topic"], o["seq"], o["track_id"], o["type"]), 0) + 1
        assert keyed and all(v == 1 for v in keyed.values())

        # retained online:true status arrives right after connect
        st = [p for p in broker.published if p["topic"] == "R/status" and p["retain"]]
        assert st and json.loads(st[0]["payload"])["online"] is True
    finally:
        stop(proc)
        broker.stop()


def test_frames_qos0(tmp_path):
    broker = FakeBroker().start()
    proc = start(tmp_path, broker.host, broker.port, "--frame-every", "5")
    try:
        m = broker.wait_publish("R/frames/cam-a")
        assert m is not None and m["qos"] == 0
        o = json.loads(m["payload"])
        assert o["schema"] == "vb.frame/1"
    finally:
        stop(proc)
        broker.stop()


def test_reconnect_sends_retained_online(tmp_path):
    broker = FakeBroker().start()
    proc = start(tmp_path, broker.host, broker.port)
    try:
        wait_until(lambda: broker.connects, msg="first CONNECT")
        n = len(broker.connects)
        broker.kill_all()
        # Reconnect within ~3 s (backoff 1 s +-20%), online:true first.
        wait_until(lambda: len(broker.connects) > n, timeout_s=6.0,
                   msg="reconnect")
        wait_until(lambda: [p for p in broker.published
                            if p["topic"] == "R/status" and p["retain"]
                            and json.loads(p["payload"])["online"]],
                   timeout_s=4.0, msg="retained online:true after reconnect")
    finally:
        stop(proc)
        broker.stop()


def test_puback_timeout_resend_once(tmp_path):
    broker = FakeBroker(no_puback=True).start()
    proc = start(tmp_path, broker.host, broker.port)
    try:
        # First event is published, never acked; 5 s later the client must
        # disconnect, reconnect and re-send it exactly once.
        wait_until(lambda: [p for p in broker.published
                            if p["topic"].startswith("R/events/")],
                   timeout_s=6.0, msg="first event")
        wait_until(lambda: len(broker.connects) >= 2, timeout_s=15.0,
                   msg="reconnect after PUBACK timeout")
        wait_until(lambda: _resend_count(broker) >= 2, timeout_s=10.0,
                   msg="resend")
        time.sleep(7.0)  # no further resends: exactly one duplicate
        assert _resend_count(broker) == 2, broker.published
    finally:
        stop(proc)
        broker.stop()


def _resend_count(broker):
    keyed = {}
    for p in broker.published:
        if not p["topic"].startswith("R/events/"):
            continue
        o = json.loads(p["payload"])
        k = (p["topic"], o["seq"], o["track_id"], o["type"])
        keyed[k] = keyed.get(k, 0) + 1
    return max(keyed.values()) if keyed else 0


def test_queue_overflow_drops_oldest(tmp_path):
    # 8 streams x 30 fps of QoS0 frames; stall the client by killing the
    # broker (listener closed) so the 1024-slot outbound queue overflows and
    # drops the oldest entries. The retained offline status published at
    # shutdown (one final connect against a fresh listener on the same port)
    # then reports mqtt_dropped > 0.
    import socket as sk
    import struct
    from fake_broker import _read_packet

    broker = FakeBroker().start()
    cfg = json.loads(FIXTURE.read_text())
    cfg["streams"] = [
        {"stream_id": f"s{i}", "url": "synthetic://?fps=30&boxes=1&w=160&h=120",
         "options": {"analyzers": [{"name": "zone", "config": {
             "zones": [{"id": "a", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}]}}]}}
        for i in range(8)]
    cfg["mqtt"] = {"host": broker.host, "port": broker.port,
                   "topic_root": "R", "keepalive_s": 30}
    p = tmp_path / "cfg8.json"
    p.write_text(json.dumps(cfg))
    proc = subprocess.Popen(
        [BIN, "--standalone", "--config", str(p), "--output", "mqtt",
         "--frame-every", "1", "--status-every", "2"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        wait_until(lambda: broker.connects, msg="initial connect")
        port = broker.port
        broker.kill_all()
        broker.stop()  # listener closed: reconnects fail, queue backs up

        got = {}

        def serve_once():
            srv = sk.socket()
            srv.setsockopt(sk.SOL_SOCKET, sk.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", port))
            srv.listen(1)
            srv.settimeout(30.0)
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            buf = bytearray()
            conn.settimeout(1.0)
            conn.sendall(bytes([0x20, 0x02, 0, 0]))  # CONNACK
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                try:
                    pkt = _read_packet(conn, buf)
                except OSError:
                    break
                if pkt is None:
                    continue
                ptype, flags, body = pkt
                if ptype == 3:
                    tlen = struct.unpack_from(">H", body, 0)[0]
                    topic = body[2:2 + tlen].decode()
                    pos = 2 + tlen
                    pid = None
                    if (flags >> 1) & 3:
                        pid = struct.unpack_from(">H", body, pos)[0]
                        pos += 2
                    got.setdefault(topic, []).append(body[pos:])
                    if pid is not None:
                        conn.sendall(struct.pack(">BBH", 0x40, 0x02, pid))
                elif ptype == 14:  # DISCONNECT
                    break
            conn.close()
            srv.close()

        import threading
        t = threading.Thread(target=serve_once)
        t.start()
        time.sleep(9.0)   # ~240 msg/s -> past the 1024-slot queue
        proc.send_signal(signal.SIGTERM)  # shutdown: final connect + offline
        t.join(timeout=40.0)
        out, err = proc.communicate(timeout=15.0)
        assert proc.returncode == 0, err
        statuses = [json.loads(x) for x in got.get("R/status", [])]
        assert any(s.get("online") is False and s.get("mqtt_dropped", 0) > 0
                   for s in statuses), statuses
    finally:
        stop(proc)


# -------------------------------------------------------------------- TLS

def make_certs(tmp_path, name, san="IP:127.0.0.1"):
    d = tmp_path / name
    d.mkdir()
    run = lambda *a: subprocess.run(a, cwd=d, check=True, capture_output=True)
    run("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", "ca.key", "-out", "ca.crt", "-days", "2",
        "-subj", "/CN=vb-test-ca")
    run("openssl", "req", "-newkey", "rsa:2048", "-nodes",
        "-keyout", "server.key", "-out", "server.csr", "-subj", "/CN=127.0.0.1")
    (d / "ext.cnf").write_text(f"subjectAltName={san}\n")
    run("openssl", "x509", "-req", "-in", "server.csr", "-CA", "ca.crt",
        "-CAkey", "ca.key", "-CAcreateserial", "-out", "server.crt",
        "-days", "2", "-extfile", "ext.cnf")
    return d


def serve_tls_broker(d):
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(d / "server.crt", d / "server.key")
    return FakeBroker(tls_ctx=ctx).start()


def test_tls_right_ca(tmp_path):
    d = make_certs(tmp_path, "good")
    broker = serve_tls_broker(d)
    cfg = write_config(tmp_path, "127.0.0.1", broker.port, "R",
                       tls=True, ca_file=str(d / "ca.crt"))
    proc = start(tmp_path, None, None, config=cfg)
    try:
        wait_until(lambda: broker.connects, timeout_s=8.0, msg="TLS CONNECT")
        m = broker.wait_publish("R/events/cam-a", timeout_s=5.0)
        assert m is not None and m["qos"] == 1
    finally:
        stop(proc)
        broker.stop()


def test_tls_wrong_ca_fails(tmp_path):
    good = make_certs(tmp_path, "good")
    bad = make_certs(tmp_path, "bad")   # different CA
    broker = serve_tls_broker(good)
    cfg = write_config(tmp_path, "127.0.0.1", broker.port, "R",
                       tls=True, ca_file=str(bad / "ca.crt"))
    proc = start(tmp_path, None, None, config=cfg)
    try:
        time.sleep(3.0)
        assert not broker.connects
        assert not broker.published
        out, err = stop(proc)
        assert "certificate" in err.lower(), err
    finally:
        broker.stop()


def test_tls_hostname_mismatch_fails(tmp_path):
    """Trusted CA, but the certificate names another host: must be rejected."""
    d = make_certs(tmp_path, "other", san="DNS:other.example")
    broker = serve_tls_broker(d)
    cfg = write_config(tmp_path, "127.0.0.1", broker.port, "R",
                       tls=True, ca_file=str(d / "ca.crt"))
    proc = start(tmp_path, None, None, config=cfg)
    try:
        time.sleep(3.0)
        assert not broker.connects
        assert not broker.published
        out, err = stop(proc)
        assert "certificate" in err.lower(), err
    finally:
        broker.stop()


def test_tls_not_compiled_in(tmp_path):
    if not NOTLS_BIN:
        pytest.skip("set VB_RUNTIME_BIN_NOTLS to a -DVB_WITH_TLS=OFF build")
    broker = FakeBroker().start()
    cfg = write_config(tmp_path, broker.host, broker.port, "R", tls=True,
                       ca_file="/dev/null")
    proc = subprocess.Popen(
        [NOTLS_BIN, "--standalone", "--config", str(cfg), "--output", "mqtt"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate(timeout=10.0)
    assert proc.returncode == 1
    assert "tls not compiled in" in err.decode()
    v = subprocess.run([NOTLS_BIN, "--version"], capture_output=True).stdout.decode()
    assert "tls:off" in v
    broker.stop()


# ------------------------------------------------------------- real broker

VB_TEST_BROKER = os.environ.get("VB_TEST_BROKER", "")
needs_broker = pytest.mark.skipif(not VB_TEST_BROKER, reason="set VB_TEST_BROKER=host:port")


@needs_broker
@pytest.mark.broker
def test_real_broker_events(tmp_path):
    host, port = VB_TEST_BROKER.rsplit(":", 1)
    got = []
    sub = MqttClient(host, int(port), "vb-test-sub")
    sub.connect(timeout_s=5.0)
    sub.subscribe("R/events/#", 1, lambda t, p, r: got.append((t, p)))
    try:
        proc = start(tmp_path, host, int(port))
        wait_until(lambda: [g for g in got if g[0].startswith("R/events/")],
                   timeout_s=8.0, msg="events via real broker")
        payload = json.loads(got[0][1])
        assert payload["schema"] == "vb.event/1"
        stop(proc)
    finally:
        sub.close()
