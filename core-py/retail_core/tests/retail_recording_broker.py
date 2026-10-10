"""Byte-recording MQTT 3.1.1 test broker for the retail wire goldens.

vision_base's FakeBroker stores parsed fields; the retail goldens need the
exact bytes of every packet a client sends (CONNECT flags and will, PUBLISH
header byte, topic, payload). This broker records each complete packet as
raw bytes, answers CONNECT with CONNACK(0) and PINGREQ with PINGRESP, and can
be stopped and restarted on the same port to simulate a broker outage.
"""
from __future__ import annotations

import socket
import threading
import time

__all__ = ["RecordingBroker", "packet_type"]


def packet_type(raw: bytes) -> int:
    return raw[0] >> 4


def _split(buf: bytearray):
    """Pop one complete packet off ``buf`` or return None."""
    if len(buf) < 2:
        return None
    mult, length, i = 1, 0, 1
    while True:
        if i >= len(buf):
            return None
        d = buf[i]
        length += (d & 0x7F) * mult
        mult *= 128
        i += 1
        if not d & 0x80:
            break
    if len(buf) < i + length:
        return None
    raw = bytes(buf[:i + length])
    del buf[:i + length]
    return raw


class RecordingBroker:
    def __init__(self, host: str = "127.0.0.1"):
        self.host = host
        self.port = 0
        self.packets: list[tuple[int, bytes]] = []   # (connection index, raw)
        self._cv = threading.Condition()
        self._srv = None
        self._conns: list[socket.socket] = []
        self._next_conn = 0
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "RecordingBroker":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        srv.settimeout(0.1)
        self.port = srv.getsockname()[1]
        self._srv = srv
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept, args=(srv, self._stop),
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Close the listener and every client connection (broker outage)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)
        if self._srv is not None:
            self._srv.close()
            self._srv = None
        with self._cv:
            conns, self._conns = self._conns, []
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.close()

    # ----------------------------------------------------------------- serve
    def _accept(self, srv, stop):
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._cv:
                idx = self._next_conn
                self._next_conn += 1
                self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn, idx), daemon=True).start()

    def _serve(self, conn, idx):
        buf = bytearray()
        try:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf.extend(chunk)
                while (raw := _split(buf)) is not None:
                    with self._cv:
                        self.packets.append((idx, raw))
                        self._cv.notify_all()
                    kind = packet_type(raw)
                    if kind == 1:
                        conn.sendall(b"\x20\x02\x00\x00")
                    elif kind == 12:
                        conn.sendall(b"\xd0\x00")
                    elif kind == 14:
                        return
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    # -------------------------------------------------------------- test API
    def wait_for(self, predicate, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while not predicate(list(self.packets)):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
        return True

    def snapshot(self) -> list[tuple[int, bytes]]:
        with self._cv:
            return list(self.packets)
