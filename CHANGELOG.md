# Changelog

## 0.1.0

First tracked release of the unified vision base (BASE-1, M1.1–M1.26).

### Review fixes (2026-09-27)

- **Stable-surface check tightened**, `contracts/stable-surface.json`
  regenerated (217 → 650 recorded items):
  - nested schema objects are recorded recursively, so a new **required**
    field at any depth fails instead of passing as an addition; declared
    types, defaults, bounds, patterns and enums are part of the comparison,
    so a changed default now fails;
  - `vision_base.types` dataclasses (field names, annotations, defaults) and
    `classmethod` / `property` / public state fields are collected, so
    `Runtime.from_config`, `ConfigApp.wants_frames` or `StreamContext.state`
    can no longer be renamed or removed unnoticed;
  - C ABI struct member lists are compared **in order** — reordering members
    keeps the name set but moves every binary offset, so the old set
    comparison missed it.
- **`health.host` default is `127.0.0.1`** in `vb-config.schema.json`,
  matching the implementation (the endpoint is unauthenticated). Exposing it
  beyond the host now requires setting `health.host` explicitly. Recorded in
  the snapshot; the value was `0.0.0.0` in the schema text only.
- **`backend.model_sha256`** added to the config schema: optional 64
  hex-digit sha256 (either case) of the model file; a mismatch refuses to
  load.
- **`mqtt.keepalive_s` below 1 is rejected at startup** instead of being read
  as MQTT's "keepalive 0 = no keepalive". The schema says minimum 1 and the
  schema is the contract; a disabled keepalive would also drop the
  1.5×-no-inbound dead-link detection.
- **Hot-path lint** resolves import aliases and follows simple dataflow
  (`mod = "numpy"; __import__(mod)`, `import ctypes as C`, byte taint through
  assignment / slicing / `memoryview()` / `bytes`-annotated parameters) and
  flags comprehensions and index loops over a byte buffer. Its `--help` and
  docstring now state that it is a first-pass screen, not a proof; §10.5 ②③
  bound what it cannot see.
- **CI matches the production image**: the native job builds with
  `-DVB_WITH_GST=ON` (it was `OFF`, so the RTSP pipeline path the image ships
  was never compiled in CI) and sets `VB_RUNTIME_BIN`, so the native
  `test_standalone*` tests execute against the built binary instead of
  failing on a missing executable path. `tests/test_control.cpp` asserts the
  snapshot branch the build actually compiles (VBS1 record with JPEG, refusal
  without) instead of only the no-JPEG branch.
- **`vb_algo` install boundary**: only the algorithm headers are installed —
  `runtime.h` used to ship with an include of the uninstalled
  `src/dev_tensor.h`, i.e. a header that could not be compiled on its own.
  The package now also installs `vbConfigVersion.cmake`
  (`SameMinorVersion`, version read from `include/vb/runtime.h`), so an
  embedder can pin the version it was built against.

### Stable external surface (§6.14.1)

Within the same schema id / same base major, these surfaces only receive
additions (new optional fields, new keys with defaults, new optional hook
methods or parameters with defaults). Removals, renames or semantic
changes require a new schema id (`vb.*/2`) or a new base major, with one
minor of dual acceptance. CI enforces this via
`tools/vb_stable_surface.py --check` against
`contracts/stable-surface.json`.

Stable as of 0.1.0:

- **Config schema** `vb.config/1` (`contracts/vb-config.schema.json`):
  all keys and their default values.
- **Event schemas** — required fields of `vb.event/1`, `vb.frame/1`,
  `vb.status/1`, `vb.command/1`, `vb.ack/1`, plus each analyzer's event
  `type` strings and field names (`line_cross`, `count_threshold`,
  `direction`, `dwell`, `zone`, `speed`, `pose_angle`, `count`).
- **Python hook surface**: `AppHooks` method names and parameters,
  `StreamContext` public methods, `Outgoing`, the user-visible dataclasses
  in `vision_base.types`, `ConfigApp`, `vision_base.geom`,
  `vision_base.embed.Runtime`, `vision_base.mqtt` public API.
- **C ABI**: `core-cpp/vb/include/vb/vb_analyzer_abi.h`
  (`VB_ANALYZER_ABI`).

Everything else — VBR1/VBE1/VBC1/VBS1/VBT1 wire formats, control lines,
supervisor↔shard IPC, C++ headers and the `vb_algo` API, `/healthz`
fields not required by the status schema, and CLI flags beyond
`--standalone/--config/--output/--frame-every/--status-every/--version` —
is internal and may change without notice.

`vb-runtime` and `vision_base` must agree on major.minor
(`hello.runtime_version` vs `vision_base.__version__`; patch level may
differ, mismatch fails fast in `RuntimeClient.start`).
