"""A minimal in-process MQTT 3.1.1 broker for tests (M1.10).

Plain TCP by default; pass ``tls_ctx`` (a server-side ``ssl.SSLContext``) to
serve TLS. Records every received packet type and every published message so
tests can assert on the exact wire behaviour (CONNECT/LWT bytes, PUBACK,
PINGREQ, subscriptions). Configurable failure modes: ``no_puback`` (never
acknowledge QoS1), ``reject_subscribe``, ``close_on_publish`` (kill the
connection as soon as a QoS1 PUBLISH arrives).
"""
from __future__ import annotations

import select
import socket
import struct
import threading
import time

__all__ = ["FakeBroker"]


def _read_exact(sock: socket.socket, n: int, buf: bytearray) -> bytes:
    while len(buf) < n:
        chunk = sock.recv(65536)
        if not chunk:
            raise OSError("closed")
        buf.extend(chunk)
    out = bytes(buf[:n])
    del buf[:n]
    return out


def _read_packet(sock: socket.socket, buf: bytearray):
    """Return (type, flags, body) or None when no full packet is buffered
    (non-blocking sockets only; blocking sockets raise/wait in recv)."""
    # read fixed header byte + remaining length
    header = _peek_header(sock, buf)
    if header is None:
        return None
    ptype, flags, hlen, rlen = header
    if len(buf) < hlen + rlen:
        chunk = sock.recv(65536)
        if not chunk:
            raise OSError("closed")
        buf.extend(chunk)
        if len(buf) < hlen + rlen:
            return None
    body = bytes(buf[hlen:hlen + rlen])
    del buf[:hlen + rlen]
    return ptype, flags, body


def _peek_header(sock: socket.socket, buf: bytearray):
    while True:
        if len(buf) >= 2:
            mult, val, i = 1, 0, 1
            ok = True
            while True:
                if i >= len(buf):
                    ok = False
                    break
                d = buf[i]
                val += (d & 0x7F) * mult
                mult *= 128
                i += 1
                if not (d & 0x80):
                    break
            if ok:
                return buf[0] >> 4, buf[0] & 0x0F, i, val
        chunk = sock.recv(65536)
        if not chunk:
            raise OSError("closed")
        buf.extend(chunk)


class _Conn:
    def __init__(self, sock, addr):
        self.sock = sock
        self.addr = addr
        self.buf = bytearray()
        self.subscriptions: list[tuple[str, int]] = []  # (filter, qos)
        self.alive = True


class FakeBroker:
    def __init__(self, *, no_puback: bool = False, reject_subscribe: bool = False,
                 close_on_publish: bool = False, tls_ctx=None, host: str = "127.0.0.1"):
        self.no_puback = no_puback
        self.reject_subscribe = reject_subscribe
        self.close_on_publish = close_on_publish
        self.tls_ctx = tls_ctx
        self.host = host
        self.port = 0
        self.connects: list[dict] = []       # parsed CONNECT headers
        self.published: list[dict] = []      # all PUBLISHes from clients
        self.pingreqs = 0
        self.disconnects = 0
        self.subscribes: list[str] = []
        self.pubacks = 0
        self._lock = threading.Lock()
        self._new_msg = threading.Condition(self._lock)
        self._srv: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._conns: list[_Conn] = []

    # ------------------------------------------------------------- lifecycle

    def start(self) -> "FakeBroker":
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, 0))
        self._srv.listen(16)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            self._drop(c)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                r, _, _ = select.select([self._srv], [], [], 0.2)
            except OSError:
                break
            if not r:
                continue
            try:
                sock, addr = self._srv.accept()
            except OSError:
                break
            sock.setblocking(True)
            if self.tls_ctx is not None:
                try:
                    sock = self.tls_ctx.wrap_socket(sock, server_side=True)
                except OSError:
                    sock.close()
                    continue
            conn = _Conn(sock, addr)
            with self._lock:
                self._conns.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _drop(self, conn: _Conn) -> None:
        conn.alive = False
        try:
            conn.sock.close()
        except OSError:
            pass

    def kill_all(self) -> None:
        """Close all client connections (simulates broker/network failure)."""
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            self._drop(c)

    # ------------------------------------------------------------- per-conn

    def _handle(self, conn: _Conn) -> None:
        try:
            self._handle_conn(conn)
        except OSError:
            pass
        finally:
            self._drop(conn)

    def _handle_conn(self, conn: _Conn) -> None:
        while not self._stop.is_set():
            pkt = _read_packet(conn.sock, conn.buf)
            if pkt is None:
                continue
            ptype, flags, body = pkt
            if ptype == 1:  # CONNECT
                info = self._parse_connect(body)
                with self._lock:
                    self.connects.append(info)
                conn.sock.sendall(bytes([0x20, 0x02, 0x00, 0x00]))
            elif ptype == 3:  # PUBLISH
                qos = (flags >> 1) & 3
                retain = bool(flags & 1)
                tlen = struct.unpack_from(">H", body, 0)[0]
                topic = body[2:2 + tlen].decode()
                pos = 2 + tlen
                pid = None
                if qos > 0:
                    pid = struct.unpack_from(">H", body, pos)[0]
                    pos += 2
                payload = body[pos:]
                with self._new_msg:
                    self.published.append({"topic": topic, "payload": payload,
                                           "qos": qos, "retain": retain})
                    self._new_msg.notify_all()
                if self.close_on_publish:
                    self._drop(conn)
                    return
                if qos == 1 and not self.no_puback:
                    conn.sock.sendall(struct.pack(">BBH", 0x40, 0x02, pid))
            elif ptype == 4:  # PUBACK from client (for our QoS1 downlink)
                with self._lock:
                    self.pubacks += 1
            elif ptype == 8:  # SUBSCRIBE
                pid = struct.unpack_from(">H", body, 0)[0]
                rest = body[2:]
                filters = []
                qos_list = []
                while rest:
                    flen = struct.unpack_from(">H", rest, 0)[0]
                    filters.append(rest[2:2 + flen].decode())
                    rest = rest[2 + flen:]
                    if not rest:
                        break
                    qos_list.append(rest[0])
                    rest = rest[1:]
                rc = 0x80 if self.reject_subscribe else 0
                resp = struct.pack(">H", pid) + bytes([rc] * len(filters))
                conn.sock.sendall(bytes([0x90]) + bytes([len(resp)]) + resp)
                if not self.reject_subscribe:
                    conn.subscriptions.extend(zip(filters, qos_list))
                    with self._lock:
                        self.subscribes.extend(filters)
            elif ptype == 10:  # UNSUBSCRIBE
                pass
            elif ptype == 12:  # PINGREQ
                with self._lock:
                    self.pingreqs += 1
                conn.sock.sendall(bytes([0xD0, 0x00]))
            elif ptype == 14:  # DISCONNECT
                with self._lock:
                    self.disconnects += 1
                return
        return

    @staticmethod
    def _parse_connect(body: bytes) -> dict:
        pos = 0

        def rstr():
            nonlocal pos
            n = struct.unpack_from(">H", body, pos)[0]
            pos += 2
            s = body[pos:pos + n].decode()
            pos += n
            return s

        proto = rstr()
        level = body[pos]
        pos += 1
        flags = body[pos]
        pos += 1
        keepalive = struct.unpack_from(">H", body, pos)[0]
        pos += 2
        client_id = rstr()
        info = {"protocol": proto, "level": level, "keepalive": keepalive,
                "client_id": client_id,
                "clean_session": bool(flags & 0x02),
                "username": "", "will": None, "will_qos": 0, "will_retain": False}
        if flags & 0x04:
            wt = rstr()
            wp = rstr()
            info["will"] = {"topic": wt, "payload": wp}
            info["will_qos"] = (flags >> 3) & 3
            info["will_retain"] = bool(flags & 0x20)
        if flags & 0x80:
            info["username"] = rstr()
        return info

    # ------------------------------------------------------------- test API

    def wait_publish(self, topic: str, timeout_s: float = 3.0) -> dict | None:
        with self._new_msg:
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                for m in self.published:
                    if m["topic"] == topic:
                        return m
                self._new_msg.wait(timeout=deadline - time.monotonic())
        return None

    def publish_down(self, topic: str, payload: bytes, qos: int = 1,
                     retain: bool = False) -> None:
        """Send a downlink PUBLISH to every connection whose filter matches."""
        body = struct.pack(">H", len(topic.encode())) + topic.encode()
        with self._lock:
            conns = list(self._conns)
        for c in conns:
            if not c.alive:
                continue
            matched = any(self._match(f, topic) for f, _ in c.subscriptions)
            if not matched:
                continue
            flags = (0x02 if qos == 1 else 0) | (0x01 if retain else 0)
            b = body
            pid = 0
            if qos == 1:
                pid = 1
                b += struct.pack(">H", pid)
            b += payload
            rlen = len(b)
            rl = bytearray()
            while True:
                d = rlen % 128
                rlen //= 128
                if rlen:
                    rl.append(d | 0x80)
                else:
                    rl.append(d)
                    break
            try:
                c.sock.sendall(bytes([0x30 | flags]) + bytes(rl) + b)
            except OSError:
                self._drop(c)

    @staticmethod
    def _match(filt: str, topic: str) -> bool:
        f = filt.split("/")
        t = topic.split("/")
        if topic.startswith("$") and filt[0] in "+#":
            return False
        for i, fl in enumerate(f):
            if fl == "#":
                return True
            if i >= len(t) or (fl != "+" and fl != t[i]):
                return False
        return len(f) == len(t)
