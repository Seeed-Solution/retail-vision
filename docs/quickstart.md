# Vision base quickstart

This guide takes you from zero to MQTT events with a single JSON file, then
adds a small Python hook. Everything not mentioned here has a default that
just works — start with the five concepts below and look something up only
when you need it.

## The five things to know

| # | Concept | In one sentence |
|---|---|---|
| 1 | Config file `vb.config/1` | One JSON: which model (`backend`), which cameras (`streams`), where results go (`mqtt`) |
| 2 | Analyzer | A built-in per-camera decision (`streams[].options.analyzers`) that produces events |
| 3 | Event `vb.event/1` | `{stream_id, type, track_id, fields}`; coordinates are normalised [0,1] over the source frame |
| 4 | Hook | An optional Python class whose `on_event(ctx, ev)` returns the messages to publish (`Outgoing`) |
| 5 | Output | MQTT `<topic_root>/events/<stream_id>`, or stdout JSON Lines from the standalone runtime |

## Prerequisites

- Docker, and an RTSP camera reachable from the container (or use a
  `synthetic://` stream URL to try things out without one).
- A model file, downloaded into `./models/` next to your config:

```bash
mkdir -p models
curl -L -o models/yolox_tiny.onnx \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx
```

## Zero code: one config file

Save this as `config.json` (the same file ships as
`examples/quickstart/config.json`):

```json
{
  "schema": "vb.config/1",
  "device_id": "door-01",
  "backend": {"name": "cpu", "model_path": "/models/yolox_tiny.onnx",
              "input_size": [416, 416], "decoder": {"type": "yolox"}},
  "mqtt": {"host": "127.0.0.1", "topic_root": "demo/vision/door-01"},
  "app": {"module": "vision_base.apps:ConfigApp"},
  "streams": [{
    "stream_id": "door",
    "url": "rtsp://camera.local:554/stream1",
    "options": {"analyzers": [
      {"name": "line_cross", "config": {"lines": [{"id": "door", "a": [0.1, 0.6], "b": [0.9, 0.6]}], "classes": [0]}},
      {"name": "count_threshold", "config": {"max_count": 5, "hold_s": 2.0, "classes": [0]}}
    ]}
  }]
}
```

Run it with one command:

```bash
docker run --rm --network host -v "$PWD":/app -v "$PWD/models":/models:ro sensecraft-missionpack.seeed.cn/solution/vision-base-cpu:<semver> --config /app/config.json
```

Watch the results:

```bash
mosquitto_sub -h 127.0.0.1 -t 'demo/vision/door-01/events/#'
```

Every message is a `vb.event/1`:

```json
{"schema":"vb.event/1","device_id":"door-01","stream_id":"door","seq":120,"ts_ms":1790000000000,
 "analyzer":"line_cross","type":"line_cross","track_id":7,"fields":{"line_id":"door","direction":"forward","anchor":[0.52,0.61],"class_id":0,"score":0.81}}
```

## No Python, no MQTT broker

Without installing Python and without a broker, the same config file runs
directly in the standalone runtime, printing one JSON record per line to
stdout (logs go to stderr):

```bash
docker run --rm --network host -v "$PWD":/app -v "$PWD/models":/models:ro --entrypoint /opt/vb/bin/vb-runtime sensecraft-missionpack.seeed.cn/solution/vision-base-cpu:<semver> --standalone --config /app/config.json --output jsonl
```

`--output mqtt` publishes straight to the broker instead, with no Python
anywhere in the process.

## Add an event hook

When the events need business logic — a counter in your own payload, a
whitelist, a database write — keep the analyzers in the config and add a
hook. Save this next to `config.json` as `hooks.py` (the same file ships as
`examples/quickstart/hooks.py`):

```python
from vision_base.apps import ConfigApp
from vision_base.hooks import Outgoing

class DoorCounter(ConfigApp):              # analyzers still from config
    name = "door-counter"
    def on_event(self, ctx, ev):
        out = super().on_event(ctx, ev)    # keep default events/<stream_id>
        if ev.type == "line_cross":
            n = ctx.state.get("inside", 0) + (1 if ev.fields["direction"] == "forward" else -1)
            ctx.state["inside"] = n
            out.append(Outgoing(f"count/{ctx.stream_id}", {"inside": n}))
        return out
```

Then change the `"app"` line in `config.json` to
`{"module": "hooks:DoorCounter"}` — the config file's directory is on the
import path, so hooks load without an install step. Run the same
`docker run` command as above: besides the default `events/door` topic you
now get a running count on `count/door`.

For everything beyond this — per-frame hooks, C/C++ plugins, new models and
accelerators, or embedding the pieces individually — see
[extending.md](extending.md).
