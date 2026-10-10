"""MQTT publishing layer: legacy retail wire format over vision_base's client.

Transport (CONNECT/LWT, keepalive PINGREQ, TLS, background reconnect with
backoff) is ``vision_base.mqtt.MqttClient``. This module only keeps what the
retail wire contract needs on top of it, byte-for-byte as before:

* ``<installation>/retail-vision/status`` is registered as the will (payload
  ``offline``, QoS0, retained) at CONNECT time, and ``online`` (QoS0,
  retained) is published right after every CONNACK, so a broker-side
  disconnect flips the topic without any app action;
* results are compact JSON (``separators=(",", ":")``, ``ensure_ascii=False``),
  QoS0, not retained, on ``<installation>/retail-vision/results/<camera>``;
* a graceful shutdown publishes ``offline`` retained, then DISCONNECT.

Outage behaviour (changed from the synchronous publisher): ``publish()``
never blocks on a reconnect. While the session is down, each call keeps only
the latest payload per topic and raises ``OSError``. When the client
reconnects it publishes retained ``online`` first, then each topic's held
payload only if it was produced at most ``replay_max_age_s`` (default 2.0 s)
before the reconnect; older ones are dropped and counted in ``stats()``.
The previous implementation made two synchronous connect attempts (up to 5 s
each) inside every publish during an outage and then dropped the message.

paho is deliberately not a dependency of the deployment image; the
vision_base client is standard library only.
"""
from __future__ import annotations

import json
import os
import threading
import time

from vision_base.mqtt import MqttClient, Will

REPLAY_MAX_AGE_S = 2.0


class MqttPublisher:
    def __init__(self, cfg, status_topic=None, online_payload=b"online",
                 offline_payload=b"offline", replay_max_age_s=REPLAY_MAX_AGE_S,
                 reconnect_min_s=1.0, reconnect_max_s=30.0):
        self.cfg = cfg
        self.status_topic = status_topic
        self.online_payload = bytes(online_payload)
        self.offline_payload = bytes(offline_payload)
        self.replay_max_age_s = float(replay_max_age_s)
        will = (Will(status_topic, self.offline_payload, qos=0, retain=True)
                if status_topic else None)
        self.client = MqttClient(
            cfg["host"], int(cfg.get("port", 1883)),
            cfg.get("client_id") or f"retail-vision-{os.getpid()}",
            username=cfg.get("username") or "", password=cfg.get("password") or "",
            tls=bool(cfg.get("tls")), ca_file=cfg.get("ca_file") or "",
            keepalive_s=int(cfg.get("keepalive_sec", 30)), will=will,
            reconnect_min_s=reconnect_min_s, reconnect_max_s=reconnect_max_s,
            on_connect=self._on_connect, on_disconnect=self._on_disconnect)
        # Serializes results against the status publish of a new session, so
        # no result can precede that session's retained "online".
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._held = {}  # topic -> (monotonic time produced, payload bytes)
        self._stats = {"published": 0, "held": 0, "replayed": 0, "dropped_stale": 0}

    # -- session callbacks (vision_base IO thread) ---------------------------
    def _on_connect(self):
        with self._lock:
            if self.status_topic:
                self.client.publish(self.status_topic, self.online_payload,
                                    qos=0, retain=True)
            now = time.monotonic()
            held, self._held = self._held, {}
            for topic, (produced, data) in held.items():
                if now - produced <= self.replay_max_age_s and \
                        self.client.publish(topic, data, qos=0):
                    self._stats["replayed"] += 1
                else:
                    self._stats["dropped_stale"] += 1
            self._ready.set()

    def _on_disconnect(self, _reason):
        self._ready.clear()

    # -- public API -------------------------------------------------------
    def connect(self, timeout_s=10.0):
        """Wait for the first session (CONNACK + retained "online").

        Raises ``ConnectionError`` (an ``OSError``) when it is not up within
        ``timeout_s``; the client keeps reconnecting in the background.
        """
        deadline = time.monotonic() + timeout_s
        if not self.client.connect(timeout_s) or \
                not self._ready.wait(max(0.0, deadline - time.monotonic())):
            raise ConnectionError(f"MQTT broker {self.cfg['host']}:"
                                  f"{int(self.cfg.get('port', 1883))} not reachable")

    def publish(self, topic: str, payload):
        data = (bytes(payload) if isinstance(payload, (bytes, bytearray))
                else json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
        with self._lock:
            if self._ready.is_set() and self.client.publish(topic, data, qos=0):
                self._stats["published"] += 1
                return
            self._held[topic] = (time.monotonic(), data)
            self._stats["held"] += 1
        raise ConnectionError("MQTT session down; latest result held for replay")

    def publish_offline(self):
        """Graceful shutdown: flip the retained status before closing.

        Only sent on a live session; without one the broker has already
        published the will (``offline``, retained) when the session dropped.
        """
        if not self.status_topic:
            return
        with self._lock:
            if self._ready.is_set():
                self.client.publish(self.status_topic, self.offline_payload,
                                    qos=0, retain=True)

    def close(self):
        """DISCONNECT (the broker then does not publish the will) and stop."""
        self._ready.clear()
        self.client.close()

    def stats(self):
        with self._lock:
            return dict(self._stats, held_now=len(self._held),
                        connected=self.client.connected)


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
