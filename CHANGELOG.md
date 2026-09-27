# Changelog

## 0.1.0

First tracked release of the unified vision base (BASE-1, M1.1–M1.26).

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
