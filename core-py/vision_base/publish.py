"""PublishWorker: one bounded in-memory publish queue per process (BASE-1 §6.7.1
offline-queue table and M1 pseudocode `publish.py`).

Holds messages while disconnected and re-sends in order after reconnect;
when the queue is full the oldest message is dropped and `dropped` counted.
Process restart loses the queue (applications needing persistence keep their
own spool and advance its cursor based on ``publish(qos=1)`` return values).
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque

__all__ = ["PublishWorker"]


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
            return {"queued": len(self._q), "dropped": self.dropped,
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
            if isinstance(payload, (bytes, str)):
                data = payload
            else:
                data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
            if self._send(topic, data, qos, retain):
                self.sent += 1
                backoff_s = 0.0
                self._pop_if_head(item)
                continue
            # First failure: the client reconnects on its own schedule; wait our
            # backoff, retry this head message once, then drop it either way.
            backoff_s = min(30.0, backoff_s * 2) if backoff_s else 1.0
            if self._stop.wait(backoff_s):
                break
            if self._send(topic, data, qos, retain):
                self.sent += 1
                backoff_s = 0.0
            else:
                self.failed += 1
            # submit() may have dropped this item while we waited (queue full);
            # only remove it if it is still the head.
            self._pop_if_head(item)
