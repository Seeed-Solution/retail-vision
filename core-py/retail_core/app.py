#!/usr/bin/env python3
"""Retail people-flow analytics service.

Shared orchestration: the backend is chosen by name from the config and is the
only board-specific piece. Everything below the `detect()` call -- tracking,
dwell state, rolling zone metrics, payload, batched MQTT publishing -- is the
same code on every platform.
"""
from __future__ import annotations

import argparse
import importlib
import json
import signal
import threading
import time
from collections import deque
from pathlib import Path

from .payload import build_vision_payload
from .publisher import MqttPublisher, PublishCycle, results_topic, status_topic
from .tracker import PersonTracker, TrackerConfig
from .video import create_video_source, validate_backend_config
from .zone_metrics import ZoneMetrics

RUNNING = True


def load_backend(name):
    """Import `backends.<name>`; it registers its video source on import."""
    return importlib.import_module(f"backends.{name}")


class StreamWorker(threading.Thread):
    def __init__(self, cfg, stream, publisher, backend):
        super().__init__(name=f"stream-{stream['id']}", daemon=True)
        self.cfg = cfg
        self.stream = stream
        self.publisher = publisher
        self.backend = backend
        self.error = None

    def run(self):
        try:
            self._run()
        except Exception as exc:  # surfaced by main() so the container exits non-zero
            self.error = exc
            print(f"[{self.stream['id']}] fatal: {exc!r}", flush=True)

    def _run(self):
        cfg = self.cfg
        size = int(cfg.get("input_size", 640))
        frame_w = int(self.stream.get("frame_width", cfg.get("frame_width", 1280)))
        frame_h = int(self.stream.get("frame_height", cfg.get("frame_height", 720)))
        publish_hz = float(self.stream.get("publish_hz", cfg.get("publish_hz", 1.0)))

        tracker_cfg = TrackerConfig(frame_width=size, frame_height=size,
                                    **cfg.get("tracker", {}))
        metrics = ZoneMetrics(float(cfg.get("window_duration", 60.0)))
        tracker = PersonTracker(tracker_cfg, on_track_removed=metrics.on_track_removed)
        if cfg.get("count_zone"):
            tracker.set_count_zone([(float(p[0]), float(p[1])) for p in cfg["count_zone"]])
        if cfg.get("entry_line"):
            line = cfg["entry_line"]
            tracker.set_entry_line(tuple(line["a"]), tuple(line["b"]),
                                   bool(line.get("ab_in", True)))

        detector = self.backend.create_detector(cfg, self.stream)
        source = create_video_source(self.stream, cfg, size)
        cycle = PublishCycle(publish_hz)
        topic = results_topic(cfg["installation"], self.stream["id"])
        print(f"[{self.stream['id']}] video={source.active_backend} topic={topic} "
              f"publish_hz={publish_hz}", flush=True)

        score_threshold = float(cfg.get("score_threshold", 0.35))
        nms_threshold = float(cfg.get("nms_threshold", 0.45))
        reconnect = float(self.stream.get("reconnect_delay_ms", 1000)) / 1000.0

        frame_id = published = 0
        start_wall = time.monotonic()
        frame_times = deque(maxlen=90)
        inference_times = deque(maxlen=90)
        last_backend = source.active_backend

        try:
            while RUNNING:
                frame = source.read()
                if frame is None:
                    time.sleep(reconnect)
                    continue
                now_mono = time.monotonic()
                detections, inference_ms = detector.detect(frame, score_threshold,
                                                           nms_threshold)
                persons = tracker.update(detections, now_mono)
                metrics.update(tracker.state_counts(), tracker.entry_count,
                               tracker.exit_count, now_mono)
                frame_id += 1
                frame_times.append(now_mono)
                inference_times.append(inference_ms)

                if source.active_backend != last_backend:
                    print(f"[{self.stream['id']}] video backend now "
                          f"{source.active_backend}", flush=True)
                    last_backend = source.active_backend

                if not cycle.due(now_mono):
                    continue

                fps = 0.0
                if len(frame_times) > 1:
                    span = frame_times[-1] - frame_times[0]
                    if span > 0:
                        fps = (len(frame_times) - 1) / span
                mean_inference = sum(inference_times) / len(inference_times)
                payload = build_vision_payload(
                    timestamp_ms=time.time() * 1000.0, frame_id=frame_id, fps=fps,
                    inference_time_ms=mean_inference, zone=metrics.snapshot(),
                    persons=persons, frame_width=frame_w, frame_height=frame_h,
                    model_width=size, model_height=size)
                payload["camera_id"] = self.stream["id"]
                payload["source_backend"] = source.active_backend
                try:
                    self.publisher.publish(topic, payload)
                    published += 1
                except OSError as exc:
                    print(f"[{self.stream['id']}] MQTT: {exc}", flush=True)
                if published % 30 == 0:
                    elapsed = now_mono - start_wall
                    print(f"[{self.stream['id']}] frames={frame_id} fps={fps:.1f} "
                          f"published={published} mqtt_hz={published / elapsed:.2f} "
                          f"backend={source.active_backend} "
                          f"inference_ms={mean_inference:.1f}", flush=True)
        finally:
            source.close()
            detector.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/config/config.json")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    required = ("installation", "backend", "model_path", "mqtt", "streams")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"missing required config keys: {missing}")
    backend = load_backend(cfg["backend"])
    validate_backend_config(cfg)
    if args.validate:
        print(json.dumps({"valid": True, "backend": cfg["backend"],
                          "streams": len(cfg["streams"]),
                          "installation": cfg["installation"],
                          "status_topic": status_topic(cfg["installation"]),
                          "video": cfg.get("video", {})}))
        return

    global RUNNING

    def stop(*_):
        global RUNNING
        RUNNING = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    publisher = MqttPublisher(cfg["mqtt"], status_topic=status_topic(cfg["installation"]))
    try:
        publisher.connect()
        print(f"[mqtt] connected, {status_topic(cfg['installation'])} = online (retained)",
              flush=True)
    except OSError as exc:
        print(f"[mqtt] initial connect failed, will retry on publish: {exc}", flush=True)

    workers = [StreamWorker(cfg, s, publisher, backend)
               for s in cfg["streams"] if s.get("enabled", True)]
    for w in workers:
        w.start()
    while RUNNING and any(w.is_alive() for w in workers):
        time.sleep(0.5)
    RUNNING = False
    for w in workers:
        w.join(timeout=5)
    publisher.publish_offline()
    publisher.close()
    if any(w.error for w in workers):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
