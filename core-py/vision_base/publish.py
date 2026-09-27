"""PublishWorker: one bounded in-memory publish queue per process (BASE-1 §6.7.1
offline-queue table and M1 pseudocode `publish.py`).

Holds messages while disconnected and re-sends in order after reconnect;
when the queue is full the oldest message is dropped and `dropped` counted.
Process restart loses the queue (applications needing persistence keep their
own spool and advance its cursor based on ``publish(qos=1)`` return values).

The head of the queue is removed only by a successful send or by ``submit``
evicting it because the queue is full (§6.7.1: offline messages are kept and
re-sent in order). A payload that cannot be JSON-encoded is dropped
individually instead of killing the worker thread.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque

__all__ = ["PublishWorker"]

log = logging.getLogger("vision_base.publish")


class PublishWorker(threading.Thread):
    def __init__(self, client, *, queue_size: int = 256, name: str = "vb-publish"):
        super().__init__(name=name, daemon=True)
        self.client = client
        self.queue_size = queue_size
        self._q: deque = deque()
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self.dropped = 0
        self.sent = 0
        self.failed = 0

    def submit(self, topic: str, payload, qos: int = 0, retain: bool = False) -> None:
        """Queue one message. dict/list payloads are JSON-encoded compactly."""
        with self._cv:
            if len(self._q) >= self.queue_size:
                self._q.popleft()
                self.dropped += 1
            self._q.append((topic, payload, qos, retain))
            self._cv.notify()

    def stats(self) -> dict:
        with self._cv:
            return {"queued": len(self._q), "queue_depth": len(self._q),
                    "dropped": self.dropped,
                    "sent": self.sent, "failed": self.failed}

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        self.join(timeout_s)

    def _pop_if_head(self, item) -> None:
        with self._cv:
            if self._q and self._q[0] == item:
                self._q.popleft()

    def _send(self, topic, data, qos, retain) -> bool:
        try:
            return bool(self.client.publish(topic, data, qos=qos, retain=retain))
        except Exception:
            return False

    def _encode(self, topic: str, payload):
        """Encode one payload; ``None`` means "drop this message".

        A payload that is not JSON-serializable is isolated here (report item
        3): raising out of the worker loop used to kill the thread for good,
        after which every later message was silently lost.
        """
        if isinstance(payload, (bytes, str)):
            return payload
        try:
            return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        except Exception:               # noqa: BLE001 - one message, not the loop
            self.failed += 1
            log.warning("dropping unserializable publish payload for %r",
                        topic, exc_info=True)
            return None

    def run(self) -> None:
        # Consecutive-failure backoff (1, 2, 4 ... <= 30 s), reset on any success.
        backoff_s = 0.0
        while not self._stop.is_set():
            with self._cv:
                while not self._q and not self._stop.is_set():
                    self._cv.wait(1.0)
                if self._stop.is_set():
                    break
                item = self._q[0]
            topic, payload, qos, retain = item
            data = self._encode(topic, payload)
            if data is None:
                self._pop_if_head(item)     # unencodable: drop just this one
                continue
            if self._send(topic, data, qos, retain):
                self.sent += 1
                backoff_s = 0.0
                self._pop_if_head(item)
                continue
            # Offline (or a failed QoS1 send): the head stays queued. The
            # client reconnects on its own schedule and we retry it, in order,
            # until it goes out — `submit` is the only thing that may evict a
            # message while offline (§6.7.1, report item 9: the previous
            # "retry once then drop" silently lost messages whenever the
            # outage outlived one backoff interval).
            self.failed += 1
            backoff_s = min(30.0, backoff_s * 2) if backoff_s else 1.0
            if self._stop.wait(backoff_s):
                break
            # `submit` may have evicted this head while we waited (queue full);
            # the loop re-reads `self._q[0]` so the next attempt targets
            # whatever is at the head now.
