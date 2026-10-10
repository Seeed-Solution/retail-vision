# Extending the vision base

Application developers start at the smallest layer that solves the problem
and move down one layer only when they need to. Everything above layer 5 is
insulated from the native runtime: no wire formats, no threading, no
per-accelerator code.

## The five layers at a glance

| Layer | What you write | Where it runs | You can | You cannot / boundary |
|---|---|---|---|---|
| 1 Configuration | Built-in analyzers with parameters in `streams[].options.analyzers` of a `vb.config/1` file | native runtime (C++) | Line crossing, zone enter/exit and dwell, parking-slot coverage, recognition voting, plus `dwell` (loitering), `speed`, `direction` (wrong-way), `count_threshold` (zone head-count limit), `pose_angle` (joint angles) | Business semantics — mapping enters/exits to your vocabulary, whitelists, persisted counters |
| 2 Event hooks | Python `AppHooks.on_event` (usually by subclassing `ConfigApp`) | Python hook thread, once per event | Assemble payloads, counters and persistence, whitelists, request a snapshot on an event, retune analyzers via `configure_analyzer` | No per-frame data |
| 3 Frame hooks | Python `AppHooks.on_frame` (`wants_frames=True`, sampled by `frame_stride`) | Python hook thread, once per frame | Pure-Python handling of the small lists in `FrameResult`: boxes, keypoints, `track_id`, analyzer attributes; helpers in `vision_base.geom` | No pixels or tensors; no numpy (hook-mode lint); hook CPU counts against the Python budget and is reported by the guardrails |
| 4 C ABI plugin | A C/C++ `.so` exporting `vb_analyzer_entry` | native runtime worker thread, once per frame | Everything a built-in analyzer can do: per-frame decisions, attributes, events | No pixel access (like built-ins: tracks and keypoints only) |
| 5 New model / new backend | Pick a built-in decoder via `backend.decoder`; a new accelerator goes in `core-cpp/vb/backends/<platform>/` | native runtime | Swap the model without writing code; debug a new decoder with the low-rate dev mode that forwards raw tensors to Python (spec §6.12, development builds only — production configs reject it) | Dev mode is debug-only (≤ 2 FPS, ≤ 1 stream) |

## Layer 1 — Configuration

Enable built-in analyzers per stream, no code at all:

```json
"options": {"analyzers": [
  {"name": "line_cross", "config": {"lines": [{"id": "door", "a": [0.1, 0.6], "b": [0.9, 0.6]}], "classes": [0]}},
  {"name": "count_threshold", "config": {"max_count": 5, "hold_s": 2.0, "classes": [0]}}
]}
```

Analyzer names, event types and per-analyzer fields are a stable surface
(spec §6.2); the full zero-code example is the
[quickstart](quickstart.md). Move down when events must become business
semantics or a custom payload.

## Layer 2 — Event hooks

Subclass `ConfigApp` so the analyzers keep coming from the config and you
only add the business view (the 12-line `DoorCounter` in the
[quickstart](quickstart.md) is the canonical example). Inside `on_event` you
get a `StreamContext` per camera with durable per-stream state
(`ctx.state`), a snapshot request (`ctx.request_snapshot`) and live analyzer
retuning (`ctx.configure_analyzer`). You may not see per-frame data here.

## Layer 3 — Frame hooks

Set `wants_frames = True` (and `frame_stride`) to receive a `FrameResult`
per sampled frame: detection boxes, keypoints, track ids and analyzer
attributes as plain Python lists — never pixels or tensors. Geometric
helpers (e.g. joint angles, box conversion) live in `vision_base.geom`.
Hook CPU time is measured per method and surfaced in the health report
(`hook_budget.exceeded`); do heavy lifting in layer 4 instead.

## Layer 4 — C ABI plugin

When Python is too slow or the target platform has no Python budget left,
the same per-frame decision logic compiles into a plugin `.so` exporting
`vb_analyzer_entry` (spec §6.2). Plugins are peers of the built-in
analyzers: same inputs (tracks, keypoints), same outputs (attributes,
events), loaded per config via `analyzers.plugins`. A plugin whose ABI
number does not match fails to load at startup.

## Layer 5 — New model / new backend

Select a decoder per model in the config — `backend.decoder.type` is one of
`yolox`, `yolov8` (pre-decoded), `yolov8_dfl` (raw head, incl. split
box/cls exports), `yolo_pose`, `classify` (plus `raw`, which decodes nothing
and is accepted only with `dev.raw_tensors=true`) — so switching models is a
config edit, not code. `backend.model_sha256` takes the sha256 of the model
file (64 hex characters, upper or lower case); when it is set, a model whose
digest differs is refused at startup instead of at the first inference, which
turns a swapped or truncated file into a clear error. A new accelerator is a
new `core-cpp/vb/backends/<platform>/` adapter implementing the inference and
frame-source contracts; everything above layer 5 is untouched.

## When to move down a layer

Move layer-3 hook logic to a layer-4 plugin when, with the target stream
count wired up, `/healthz` reports `hook_budget.exceeded` continuously for
60 s, or the Python CPU share crosses the budget lines (`py_share` ≤ 0.15, or
`py_core` ≤ 0.35 of one core over the run). Write the logic in Python first
with a fixture; when sinking it, run the same fixture through
`vb_selftest analyzer <name> <fixture> --plugin <so>` to prove the behavior
did not change.

A plugin is not a route to pixel access. Layer 4 receives what the built-in
analyzers receive — tracks and keypoints — and no frame buffer, so a
requirement that needs pixels, tensors or per-frame image math does not
belong there: it goes to layer 5 (a new decoder or accelerator adapter) or
stays in the native layer.

## Using only what you need

The modules can be used on their own; each drops what it does not need.

### a) Standalone `vb-runtime` only

One binary plus a model plus a `vb.config/1` file; no Python, no broker, no
service management — restart policy is docker's or systemd's job:

```bash
vb-runtime --standalone --config cfg.json --output jsonl [--frame-every N] [--status-every S]
```

stdout carries one `vb.event/1`, `vb.frame/1` or `vb.status/1` record per
line (logs on stderr). `--output mqtt` publishes directly:
`<topic_root>/events/<stream_id>` (QoS 1), `frames/<stream_id>` (QoS 0) and
a retained `status` with LWT. Boundaries: single process — no stream
add/remove over a control channel, no `/healthz`, no hooks; on broker
outage an in-memory queue of 1024 records is kept (oldest dropped, counted),
nothing is persisted to disk. Good fit for direct SI integration and
constrained devices.

### b) Python client library only

Standard-library `vision_base` plus a local `vb-runtime` binary; no broker,
no health endpoint. The runtime runs as one child process; when it exits,
`results()` raises `RuntimeGone` and restarting it is your call:

```python
from vision_base.embed import Runtime
with Runtime.from_config("cfg.json") as rt:
    rt.add_stream("cam", "rtsp://camera.local:554/s", analyzers=[{"name": "zone", "config": {"zones": [{"id": "a", "polygon": [[0,0],[1,0],[1,1]]}]}}])
    for item in rt.results(timeout_s=30):
        print(item)
```

### c) C++ library (`vb_algo`) only

The `vb_algo` static library and headers (from `cmake --install`) carry the
tracker, analyzers, geometry, wire codec and decoders — C++17, no ONNX
Runtime, no GStreamer, no thread pool, no vendor SDKs:

```cmake
find_package(vb CONFIG REQUIRED)
target_link_libraries(app PRIVATE vb::algo)
```

Use `vb::Tracker`, `vb::create_analyzer`, `vb::LetterboxGeom`,
`vb::wire_encode/decode` and `vb::make_decoder` for pure decision logic
embedded in your own loop (no capture, no inference). A complete example
ships in `examples/embed_cpp/`. The C++ API is an internal, changeable
surface — embed against a pinned `vision-base-v<semver>` tag.

### d) Configuration-only mode

The base image plus one `vb.config/1` file, no code of any kind — the
zero-code example in the [quickstart](quickstart.md). Limits: built-in
analyzers and decoders only, fixed `vb.event/1`/`vb.frame/1` output.

## The health endpoint

The supervisor serves `GET /healthz` on `health.host:health.port`, which
defaults to `127.0.0.1:8099`. The body carries device, process, per-stream and
application health, and the endpoint has no authentication, so reaching it
from another host means setting `health.host` explicitly — the bind address
is the only control, and who may connect is the network's business.

## The stable surface

What you may build against across updates (spec §6.14): the `vb.config/1`
schema — additive-only within the id, every nested key and its default value
included; breaking changes become `vb.config/2`, accepted in parallel for at
least one minor — the required fields of `vb.event/1`, `vb.frame/1`,
`vb.status/1`, `vb.command/1`, `vb.ack/1` plus each analyzer's event types and
field names, the Python hook signatures (`AppHooks`, `StreamContext`,
`Outgoing`, `ConfigApp`, the user-visible `vision_base.types` dataclass
fields, `vision_base.geom`, `vision_base.embed.Runtime`, `vision_base.mqtt`),
and the plugin C ABI (`VB_ANALYZER_ABI`). CI enforces it with
`tools/vb_stable_surface.py --check` against `contracts/stable-surface.json`:
a removed key or field, a changed default, a new *required* field, a renamed
hook parameter, a reordered C ABI struct member or a changed signature fails;
only additions pass. Everything else — the VBR1/VBE1/VBC1/VBS1 wire formats,
inter-process plumbing, C++ headers including `vb_algo`, and every CLI flag
except `--standalone/--config/--output/--frame-every/--status-every/--version`
— is internal and may change without notice; `vb-runtime` and `vision_base`
must always come from the same release.
