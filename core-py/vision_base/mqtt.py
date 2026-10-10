"""MQTT 3.1.1 client (subset) used by the base control session, PublishWorker
and application-built sessions alike (spec BASE-1 §6.7 / §6.7.1, B7).

Standard library only. Implemented subset:
CONNECT (clean session, optional LWT), CONNACK, PUBLISH QoS0/QoS1, PUBACK,
SUBSCRIBE/SUBACK, PINGREQ/PINGRESP, DISCONNECT, TLS (stdlib ``ssl``).
"""
from __future__ import annotations

import logging
import random
import select
import socket
import ssl
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("vision_base.mqtt")

MAX_PAYLOAD = 1024 * 1024  # 1 MiB, outgoing
# Inbound limits (BASE-1 review item 18): the broker is not trusted to declare
# a packet size we then accumulate verbatim — without a cap a single 4-byte
# remaining-length header can make us buffer close to 256 MiB. Our own
# publishes are <= 1 MiB, so 4 MiB leaves headroom for a peer with larger
# payloads while bounding both one packet and the receive buffer.
MAX_INBOUND_PACKET = 4 * 1024 * 1024
MAX_RX_BUFFER = MAX_INBOUND_PACKET + 5
# Consecutive reconnect failures before the backoff is pinned at
# reconnect_max_s; also bounds the `2 ** k` exponent so it can never overflow.
_MAX_BACKOFF_EXPONENT = 32
SUBACK_TIMEOUT_S = 5.0

__all__ = ["Will", "MqttClient"]


class _ProtocolError(Exception):
    """Peer sent a malformed packet; the session must be torn down.

    Never fatal to the IO thread: ``_run_session`` converts it into a session
    failure so the normal backoff reconnect path takes over (§6.7.1).
    """


@dataclass(frozen=True)
class Will:
    topic: str
    payload: bytes
    qos: int = 1
    retain: bool = True


def _enc_len(n: int) -> bytes:
    out = bytearray()
    while True:
        d = n % 128
        n //= 128
        if n:
            out.append(d | 0x80)
        else:
            out.append(d)
            return bytes(out)


def _enc_str(s: str | bytes) -> bytes:
    if isinstance(s, str):
        s = s.encode("utf-8")
    return struct.pack(">H", len(s)) + s


def topic_matches(topic_filter: str, topic: str) -> bool:
    """MQTT 3.1.1 §4.7 topic filter matching.

    ``+`` matches exactly one level; ``#`` (only valid as the last level)
    matches any number of trailing levels including the parent level itself.
    Topics beginning with ``$`` are not matched by filters whose first level
    is a wildcard (§4.7.2).
    """
    if topic.startswith("$") and (topic_filter.startswith("+") or topic_filter.startswith("#")):
        return False
    f_levels = topic_filter.split("/")
    t_levels = topic.split("/")
    for i, fl in enumerate(f_levels):
        if fl == "#":
            # must be last; matches remaining zero or more levels
            return i == len(f_levels) - 1
        if i >= len(t_levels):
            return False
        if fl != "+" and fl != t_levels[i]:
            return False
    return len(f_levels) == len(t_levels)


class _Waiter:
    __slots__ = ("event", "result")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: bool | None = None  # None = still waiting


class _SubEntry:
    __slots__ = ("topic_filter", "qos", "callback", "pending_ack", "suback_rc")

    def __init__(self, topic_filter: str, qos: int, callback):
        self.topic_filter = topic_filter
        self.qos = qos
        self.callback = callback
        self.pending_ack = threading.Event()
        self.suback_rc = 0


class MqttClient:
    def __init__(self, host: str, port: int = 1883, client_id: str = "", *, username: str = "",
                 password: str = "", tls: bool = False, ca_file: str = "", cert_file: str = "",
                 key_file: str = "", keepalive_s: int = 30, will: Will | None = None,
                 reconnect_min_s: float = 1.0, reconnect_max_s: float = 60.0, max_inflight: int = 32,
                 on_connect: Callable[[], None] | None = None,
                 on_disconnect: Callable[[str], None] | None = None):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.username = username
        self.password = password
        self.tls = tls
        self.ca_file = ca_file
        self.cert_file = cert_file
        self.key_file = key_file
        self.keepalive_s = keepalive_s
        self.will = will
        self.reconnect_min_s = reconnect_min_s
        self.reconnect_max_s = reconnect_max_s
        self.max_inflight = max_inflight
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect

        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._io_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = threading.Event()  # set == connected

        self._subs: list[_SubEntry] = []
        self._pending_subs: dict[int, _SubEntry] = {}
        self._subs_lock = threading.Lock()

        self._inflight: dict[int, _Waiter] = {}
        self._inflight_cv = threading.Condition()
        self._next_pid = 1

        self._stats = {"connects": 0, "disconnects": 0, "published": 0,
                       "puback_timeouts": 0, "inflight": 0, "rx": 0}
        self._first_connect_done = threading.Event()
        self._first_connect_ok = False

        self._rx_buf = b""

    # ------------------------------------------------------------------ public

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def connect(self, timeout_s: float = 10.0) -> bool:
        """Start the IO thread; return True if CONNACK(0) arrives in time.

        On failure the background thread keeps retrying with backoff.
        """
        if self._io_thread is None:
            self._io_thread = threading.Thread(
                target=self._io_loop, name=f"mqtt-io-{self.client_id}", daemon=True)
            self._io_thread.start()
        self._first_connect_done.wait(timeout_s)
        return self._first_connect_ok and self._connected.is_set()

    def publish(self, topic: str, payload: bytes | str, qos: int = 0, retain: bool = False,
                timeout_s: float = 5.0) -> bool:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload must be bytes or str")
        payload = bytes(payload)
        if len(payload) > MAX_PAYLOAD:
            raise ValueError("payload exceeds 1 MiB")
        if qos not in (0, 1):
            raise ValueError("qos must be 0 or 1")
        if not self.connected:
            return False

        if qos == 0:
            pkt = self._encode_publish(topic, payload, 0, 0, retain)
            try:
                self._send(pkt)
            except OSError:
                self._handle_drop("qos0 publish write failed")
                return False
            self._stats["published"] += 1
            return True

        # QoS 1 ------------------------------------------------------------
        if threading.current_thread() is self._io_thread:
            raise RuntimeError("qos1 publish is not allowed on the IO thread "
                               "(subscribe callback / on_connect) - it would deadlock")

        deadline = time.monotonic() + timeout_s
        with self._inflight_cv:
            while len(self._inflight) >= self.max_inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.connected:
                    return False
                self._inflight_cv.wait(remaining)
            pid = self._next_pid
            while True:
                self._next_pid = pid % 65535 + 1
                if pid not in self._inflight:
                    break
                pid = self._next_pid
            waiter = _Waiter()
            self._inflight[pid] = waiter
            self._stats["inflight"] = len(self._inflight)

        pkt = self._encode_publish(topic, payload, 1, pid, retain)
        try:
            self._send(pkt)
        except OSError:
            with self._inflight_cv:
                self._inflight.pop(pid, None)
                self._stats["inflight"] = len(self._inflight)
                self._inflight_cv.notify_all()
            self._handle_drop("qos1 publish write failed")
            return False

        remaining = deadline - time.monotonic()
        if waiter.event.wait(max(0.0, remaining)) and waiter.result is True:
            return True
        with self._inflight_cv:
            gone = self._inflight.pop(pid, None) is not None
            self._stats["inflight"] = len(self._inflight)
            self._inflight_cv.notify_all()
        if gone and waiter.result is not True:
            # timed out, or disconnected while waiting (disconnect already
            # resolved waiters); a late PUBACK for this id is dropped in
            # _on_puback because the id is no longer registered.
            if waiter.result is None and not waiter.event.is_set():
                self._stats["puback_timeouts"] += 1
            return False
        return waiter.result is True

    def subscribe(self, topic_filter: str, qos: int,
                  callback: Callable[[str, bytes, bool], None]) -> None:
        entry = _SubEntry(topic_filter, qos, callback)
        with self._subs_lock:
            self._subs = [s for s in self._subs if s.topic_filter != topic_filter]
            self._subs.append(entry)
        if self.connected:
            self._send_subscribe(entry)

    def unsubscribe(self, topic_filter: str) -> None:
        with self._subs_lock:
            self._subs = [s for s in self._subs if s.topic_filter != topic_filter]

    def close(self, timeout_s: float = 2.0) -> None:
        """Send DISCONNECT (broker must not fire the LWT) and stop the IO thread."""
        if self.connected:
            try:
                self._send(bytes([0xE0, 0x00]))
            except OSError:
                pass
        self._stop.set()
        self._connected.clear()
        with self._inflight_cv:
            self._inflight_cv.notify_all()
        if self._io_thread is not None:
            self._io_thread.join(timeout_s)

    def stats(self) -> dict:
        with self._inflight_cv:
            inflight = len(self._inflight)
        return dict(self._stats, inflight=inflight)

    # ------------------------------------------------------------- packet code

    def _encode_connect(self) -> bytes:
        vh = bytearray()
        vh += _enc_str("MQTT") + bytes([4])              # protocol name, level 4
        flags = 0x02                                     # clean session
        if self.will is not None:
            flags |= 0x04                               # will flag
            flags |= (min(1, max(0, self.will.qos)) << 3)
            if self.will.retain:
                flags |= 0x20
        if self.username:
            flags |= 0x80
            if self.password:
                flags |= 0x40
        vh += bytes([flags])
        vh += struct.pack(">H", max(1, int(self.keepalive_s)))
        payload = _enc_str(self.client_id)
        if self.will is not None:
            w = self.will.payload
            if isinstance(w, str):
                w = w.encode("utf-8")
            payload += _enc_str(self.will.topic) + _enc_str(w)
        if self.username:
            payload += _enc_str(self.username)
            if self.password:
                payload += _enc_str(self.password)
        body = bytes(vh) + payload
        return bytes([0x10]) + _enc_len(len(body)) + body

    def _encode_publish(self, topic: str, payload: bytes, qos: int, pid: int,
                        retain: bool) -> bytes:
        flags = 0
        if qos == 1:
            flags |= 0x02
        if retain:
            flags |= 0x01
        body = _enc_str(topic)
        if qos > 0:
            body += struct.pack(">H", pid)
        body += payload
        return bytes([0x30 | flags]) + _enc_len(len(body)) + body

    def _send(self, pkt: bytes) -> None:
        sock = self._sock
        if sock is None:
            raise OSError("not connected")
        with self._send_lock:
            sock.sendall(pkt)

    # ---------------------------------------------------------------- IO loop

    def _io_loop(self) -> None:
        k = 0
        while not self._stop.is_set():
            connects_before = self._stats["connects"]
            reason = self._run_session()
            if self._stats["connects"] > connects_before:
                # §6.7.1: k is reset after CONNACK(0), so the backoff only
                # grows while *consecutive* attempts fail to establish a
                # session (report: a session that connects then drops must
                # not inherit the exponent of the outage before it).
                k = 0
            if self._stop.is_set():
                break
            self._stats["disconnects"] += 1
            self._connected.clear()
            self._fail_inflight("disconnected")
            cb = self.on_disconnect
            if cb is not None:
                try:
                    cb(reason)
                except Exception:
                    log.exception("on_disconnect callback failed")
            # backoff reconnect_min_s * 2^k, capped, ±20% jitter. The exponent
            # is clamped *before* the power: an unbounded k makes `2 ** k`
            # raise OverflowError on the float multiply (report item 4).
            base = min(self.reconnect_max_s,
                       self.reconnect_min_s * (2 ** min(k, _MAX_BACKOFF_EXPONENT)))
            delay = base * random.uniform(0.8, 1.2)
            k = min(k + 1, _MAX_BACKOFF_EXPONENT)
            if self._stop.wait(delay):
                break

    def _run_session(self) -> str:
        try:
            raw = socket.create_connection((self.host, self.port), timeout=10.0)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            raw.settimeout(None)
            if self.tls:
                ctx = ssl.create_default_context(cafile=self.ca_file or None)
                if self.cert_file:
                    ctx.load_cert_chain(self.cert_file, self.key_file or None)
                raw = ctx.wrap_socket(raw, server_hostname=self.host)
        except (OSError, ssl.SSLError, ValueError) as exc:
            if not self._first_connect_done.is_set():
                self._first_connect_done.set()
                self._first_connect_ok = False
            return f"connect failed: {exc}"

        self._sock = raw
        self._rx_buf = b""
        last_rx = time.monotonic()
        last_tx = time.monotonic()
        try:
            with self._send_lock:
                raw.sendall(self._encode_connect())
                last_tx = time.monotonic()
            # wait for CONNACK
            connack_rc = None
            while connack_rc is None:
                pkt = self._read_packet(raw, deadline=time.monotonic() + 10.0)
                if pkt is None:
                    return "connack timeout"
                ptype, flags, body = pkt
                if ptype == 2:  # CONNACK
                    connack_rc = body[1] if len(body) > 1 else 1
                # ignore anything else before CONNACK
            if connack_rc != 0:
                return f"connack rc={connack_rc}"
            if not self._first_connect_done.is_set():
                self._first_connect_done.set()
                self._first_connect_ok = True
            self._stats["connects"] += 1
            self._connected.set()

            # resubscribe all filters in registration order
            with self._subs_lock:
                subs = list(self._subs)
            for entry in subs:
                rc = self._subscribe_inline(raw, entry)
                if rc is None or rc == 0x80:
                    return f"resubscribe failed: {entry.topic_filter}"

            cb = self.on_connect
            if cb is not None:
                try:
                    cb()
                except Exception:
                    log.exception("on_connect callback failed")

            # main traffic loop
            while not self._stop.is_set():
                now = time.monotonic()
                if now - last_rx > 1.5 * max(1, self.keepalive_s):
                    return "keepalive timeout"
                if now - last_tx >= max(1, self.keepalive_s):
                    with self._send_lock:
                        raw.sendall(bytes([0xC0, 0x00]))  # PINGREQ
                    last_tx = time.monotonic()
                timeout = max(0.0, min(1.0, max(1, self.keepalive_s) - (time.monotonic() - last_tx)))
                pkt = self._read_packet(raw, deadline=time.monotonic() + timeout)
                if pkt is None:
                    continue  # select timeout, loop re-checks keepalive
                ptype, flags, body = pkt
                last_rx = time.monotonic()
                if ptype == 3:  # PUBLISH
                    self._on_publish(raw, flags, body)
                elif ptype == 4:  # PUBACK
                    self._on_puback(body)
                elif ptype == 9:  # SUBACK (subscribes issued from other threads)
                    if len(body) >= 3:
                        spid = struct.unpack_from(">H", body, 0)[0]
                        with self._subs_lock:
                            entry = self._pending_subs.pop(spid, None)
                        if entry is not None:
                            entry.suback_rc = body[2]
                            entry.pending_ack.set()
                elif ptype == 13:  # PINGRESP
                    pass
            return "closed"
        except _ProtocolError as exc:
            # Malformed peer data is a session failure, not a client death:
            # return normally so _io_loop reconnects with backoff (§6.7.1).
            return f"protocol error: {exc}"
        except (OSError, ssl.SSLError) as exc:
            return f"socket error: {exc}"
        finally:
            try:
                raw.close()
            except OSError:
                pass
            self._sock = None

    # subscribe helper used from other threads; returns None on timeout,
    # raises ValueError if the broker rejected the filter
    def _send_subscribe(self, entry: _SubEntry) -> None:
        pid = self._alloc_pid()
        with self._subs_lock:
            self._pending_subs[pid] = entry
        body = struct.pack(">H", pid) + _enc_str(entry.topic_filter) + bytes([min(1, max(0, entry.qos))])
        pkt = bytes([0x82]) + _enc_len(len(body)) + body
        try:
            self._send(pkt)
        except OSError:
            with self._subs_lock:
                self._pending_subs.pop(pid, None)
            return  # will be sent on reconnect
        if not entry.pending_ack.wait(SUBACK_TIMEOUT_S):
            with self._subs_lock:
                self._pending_subs.pop(pid, None)
            return
        if entry.suback_rc == 0x80:
            with self._subs_lock:
                self._subs = [s for s in self._subs if s is not entry]
            raise ValueError("subscribe rejected")

    def _subscribe_inline(self, raw: socket.socket, entry: _SubEntry) -> int | None:
        """Send SUBSCRIBE from the IO thread and wait for its SUBACK inline."""
        pid = self._alloc_pid()
        body = struct.pack(">H", pid) + _enc_str(entry.topic_filter) + bytes([min(1, max(0, entry.qos))])
        pkt = bytes([0x82]) + _enc_len(len(body)) + body
        with self._send_lock:
            raw.sendall(pkt)
        deadline = time.monotonic() + SUBACK_TIMEOUT_S
        while time.monotonic() < deadline and not self._stop.is_set():
            pkt = self._read_packet(raw, deadline=min(deadline, time.monotonic() + 1.0))
            if pkt is None:
                continue
            ptype, flags, body = pkt
            if ptype == 9:  # SUBACK
                if len(body) >= 3 and struct.unpack_from(">H", body, 0)[0] == pid:
                    return body[2]
            elif ptype == 3:
                self._on_publish(raw, flags, body)
            elif ptype == 4:
                self._on_puback(body)
        return None  # timeout -> treat as session failure

    def _alloc_pid(self) -> int:
        pid = self._next_pid
        while True:
            self._next_pid = pid % 65535 + 1
            if pid not in self._inflight or pid == self._next_pid:
                return pid
            pid = self._next_pid

    def _read_packet(self, sock, deadline: float):
        """Read one full MQTT packet (blocking select until deadline).

        Returns (packet_type, flags, body) or None on timeout.
        """
        while True:
            pkt = self._try_parse()
            if pkt is not None:
                return pkt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            r, _, _ = select.select([sock], [], [], min(remaining, 1.0))
            if not r:
                if self._stop.is_set():
                    return None
                continue
            chunk = sock.recv(65536)
            if not chunk:
                raise OSError("connection closed by peer")
            self._rx_buf += chunk
            self._stats["rx"] += 1

    def _try_parse(self):
        buf = self._rx_buf
        if len(buf) < 2:
            return None
        # decode remaining length (max 4 bytes)
        mult = 1
        val = 0
        i = 1
        while True:
            if i >= len(buf):
                return None
            d = buf[i]
            val += (d & 0x7F) * mult
            mult *= 128
            i += 1
            if not (d & 0x80):
                break
            if i > 4:
                raise OSError("malformed remaining length")
        total = i + val
        if total > MAX_INBOUND_PACKET:
            raise _ProtocolError(f"inbound packet too large: {total} bytes")
        if len(buf) < total:
            if len(buf) > MAX_RX_BUFFER:
                # Backstop for the invariant the packet cap establishes: with
                # `total` bounded, the buffer can only reach one packet plus a
                # recv chunk, so this means the parse loop stopped consuming.
                raise _ProtocolError("receive buffer overflow")
            return None
        ptype = buf[0] >> 4
        flags = buf[0] & 0x0F
        body = buf[i:total]
        self._rx_buf = buf[total:]
        return ptype, flags, body

    # ------------------------------------------------------------ dispatchers

    def _on_publish(self, sock, flags: int, body: bytes) -> None:
        """Dispatch one inbound PUBLISH.

        Every length field is validated before use (report item 17): an
        empty or truncated body used to raise ``struct.error``, which no
        caller caught, so the IO thread died and the client never
        reconnected. Protocol damage now raises ``_ProtocolError``, which
        ``_run_session`` turns into an ordinary session failure.
        """
        qos = (flags >> 1) & 0x03
        retain = bool(flags & 0x01)
        if qos == 3:
            raise _ProtocolError("PUBLISH with reserved qos 3")
        if len(body) < 2:
            raise _ProtocolError(f"truncated PUBLISH: {len(body)} byte body")
        tlen = struct.unpack_from(">H", body, 0)[0]
        pos = 2 + tlen
        if pos > len(body):
            raise _ProtocolError(
                f"truncated PUBLISH topic: declared {tlen} bytes, "
                f"{len(body) - 2} available")
        topic = body[2:pos].decode("utf-8", "replace")
        pid = None
        if qos > 0:
            if pos + 2 > len(body):
                raise _ProtocolError("truncated PUBLISH packet id")
            pid = struct.unpack_from(">H", body, pos)[0]
            pos += 2
        payload = body[pos:]
        with self._subs_lock:
            subs = list(self._subs)
        for entry in subs:
            if topic_matches(entry.topic_filter, topic):
                try:
                    entry.callback(topic, payload, retain)
                except Exception:
                    log.exception("subscribe callback failed for %r", entry.topic_filter)
        if qos == 1 and pid is not None:
            try:
                self._send(struct.pack(">BBH", 0x40, 0x02, pid))
            except OSError:
                pass

    def _on_puback(self, body: bytes) -> None:
        if len(body) < 2:
            return
        pid = struct.unpack_from(">H", body, 0)[0]
        with self._inflight_cv:
            waiter = self._inflight.get(pid)
            if waiter is not None:
                del self._inflight[pid]
                self._stats["inflight"] = len(self._inflight)
                self._stats["published"] += 1
                self._inflight_cv.notify_all()
            else:
                return  # late PUBACK for an already-timed-out id: drop
        waiter.result = True
        waiter.event.set()

    def _fail_inflight(self, reason: str) -> None:
        with self._inflight_cv:
            waiters = list(self._inflight.values())
            self._inflight.clear()
            self._stats["inflight"] = 0
            self._inflight_cv.notify_all()
        for w in waiters:
            w.result = False
            w.event.set()

    def _handle_drop(self, reason: str) -> None:
        log.debug("dropping connection: %s", reason)
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
