"""MQTT publishing layer: minimal MQTT 3.1.1 QoS0 client with Last Will.

Pure I/O over a socket, no board or SDK dependency, so every backend shares it.

paho is deliberately not a dependency of the deployment image. The publisher
registers `<installation>/retail-vision/status` as its will (payload "offline",
retained) at CONNECT time and publishes "online" retained immediately after
CONNACK, so a broker-side disconnect flips the topic without any app action.
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import struct
import threading


def _remaining(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n % 128
        n //= 128
        if n:
            byte |= 128
        out.append(byte)
        if not n:
            return bytes(out)


def _mqtt_string(value: str) -> bytes:
    data = value.encode()
    return struct.pack("!H", len(data)) + data


class MqttPublisher:
    def __init__(self, cfg, status_topic=None, online_payload=b"online",
                 offline_payload=b"offline"):
        self.cfg = cfg
        self.status_topic = status_topic
        self.online_payload = online_payload
        self.offline_payload = offline_payload
        self.sock = None
        self.lock = threading.Lock()

    # -- wire helpers -----------------------------------------------------
    @staticmethod
    def _publish_packet(topic: str, data: bytes, retain: bool = False) -> bytes:
        body = _mqtt_string(topic) + data
        header = 0x30 | (0x01 if retain else 0x00)
        return bytes((header,)) + _remaining(len(body)) + body

    def _connect(self):
        raw = socket.create_connection((self.cfg["host"], int(self.cfg.get("port", 1883))), 5)
        if self.cfg.get("tls"):
            ctx = ssl.create_default_context(cafile=self.cfg.get("ca_file") or None)
            raw = ctx.wrap_socket(raw, server_hostname=self.cfg["host"])
        client = self.cfg.get("client_id") or f"retail-vision-{os.getpid()}"
        flags = 0x02  # clean session
        payload = _mqtt_string(client)
        if self.status_topic:
            # will flag + will retain, QoS 0
            flags |= 0x04 | 0x20
            payload += _mqtt_string(self.status_topic)
            payload += struct.pack("!H", len(self.offline_payload)) + self.offline_payload
        if self.cfg.get("username"):
            flags |= 0x80
            payload += _mqtt_string(self.cfg["username"])
        if self.cfg.get("password"):
            flags |= 0x40
            payload += _mqtt_string(self.cfg["password"])
        variable = (_mqtt_string("MQTT") + bytes((4, flags))
                    + struct.pack("!H", int(self.cfg.get("keepalive_sec", 30))))
        packet = variable + payload
        raw.sendall(b"\x10" + _remaining(len(packet)) + packet)
        if raw.recv(4)[-1:] != b"\x00":
            raise ConnectionError("MQTT CONNACK rejected")
        self.sock = raw
        if self.status_topic:
            raw.sendall(self._publish_packet(self.status_topic, self.online_payload, retain=True))

    def _send(self, packet: bytes):
        with self.lock:
            for attempt in range(2):
                try:
                    if self.sock is None:
                        self._connect()
                    self.sock.sendall(packet)
                    return
                except OSError:
                    if self.sock:
                        try:
                            self.sock.close()
                        except OSError:
                            pass
                    self.sock = None
                    if attempt:
                        raise

    # -- public API -------------------------------------------------------
    def connect(self):
        with self.lock:
            if self.sock is None:
                self._connect()

    def publish(self, topic: str, payload):
        data = (payload if isinstance(payload, (bytes, bytearray))
                else json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
        self._send(self._publish_packet(topic, data))

    def publish_offline(self):
        """Graceful shutdown: flip the retained status before closing."""
        if not self.status_topic:
            return
        try:
            self._send(self._publish_packet(self.status_topic, self.offline_payload, retain=True))
        except OSError:
            pass

    def close(self):
        with self.lock:
            if self.sock:
                try:
                    self.sock.sendall(b"\xe0\x00")  # DISCONNECT
                    self.sock.close()
                except OSError:
                    pass
                self.sock = None


class PublishCycle:
    """Fixed-rate gate for batched publishing.

    One batched message per cycle instead of one per person per frame: with
    per-person messages a broker shared by several cameras carries
    `people x cameras x fps` messages per second, which is the first thing to
    fall over. This makes it `cameras x publish_hz`.
    """

    def __init__(self, publish_hz=1.0):
        self.interval = 1.0 / publish_hz if publish_hz > 0 else 1.0
        self._next = None

    def due(self, now):
        if self._next is None:
            self._next = now
        if now < self._next:
            return False
        self._next += self.interval
        # Skip whole missed cycles rather than bursting to catch up.
        if self._next < now:
            self._next = now + self.interval
        return True


def results_topic(installation, camera_id):
    return f"{installation}/retail-vision/results/{camera_id}"


def status_topic(installation):
    return f"{installation}/retail-vision/status"
