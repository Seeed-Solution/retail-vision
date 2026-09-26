# vb wire format — VBR1 / VBE1 / VBC1 / VBS1 (spec BASE-1 §6.3)

Transport: `socketpair(AF_UNIX, SOCK_STREAM)`; native side fd given via `--ipc-fd`
(debug: `--listen <path>` Unix socket). All records are little-endian; the first
8 bytes of every record are `char[4] magic` + `u32 body_len`, where `body_len`
excludes the 8 header bytes.

Reference implementations: C++ `core-cpp/vb/src/wire.cpp`
(`vb::wire_encode_*`), Python `core-py/vision_base/wire.py` (`decode_result`,
`decode_event`, `decode_snapshot`, `decode_header`).

Consistency fixtures: `contracts/fixtures/vb/wire_case{1,2,3}.{json,bin}`
(case1: 0 detections; case2: 3 detections, no keypoints, 2 attrs per detection;
case3: 2 detections × 17 keypoints). The C++ encoder must reproduce the `.bin`
byte-for-byte from the `.json`; the Python decoder must produce objects equal
to the `.json` (float tolerance ≤ 1e-6). Truncated records raise `WireError`.

## VBR1 — frame result (native → Python)

| offset | type | field |
|---|---|---|
| 0 | char[4] | `VBR1` |
| 4 | u32 | body_len |
| 8 | u32 | stream_index (assigned by Python in the `add` control line, never reused) |
| 12 | u64 | seq (per-stream monotonic) |
| 20 | f64 | wall_ms (capture time, epoch ms) |
| 28 | i32×4 | src_w, src_h, model_w, model_h |
| 44 | f32×3 | scale, pad_x, pad_y |
| 56 | u8 | align (0 = center, 1 = top-left) |
| 57 | u8 | kpt_per_det (0 or 17) |
| 58 | u16 | n_det |
| 60 | f32 | inference_ms (preprocess + inference + postprocess) |
| 64 | f32 | queue_delay_ms (inference start − capture) |
| 68 | u8 | attr_per_det (total analyzer attrs) |
| 69 | u8 | reserved = 0 |
| 70 | u16 | reserved = 0 |
| 72 | n_det × {f32 cx, cy, w, h, score; i32 class_id; u32 track_id} | 28 bytes each, model-canvas normalized |
| … | n_det × kpt_per_det × {f32 x, y, conf} | 12 bytes per keypoint |
| … | n_det × attr_per_det × f32 | analyzer attrs, order matches hello `attr_names` |

The detection array is stored in full first, then all keypoints, then all attrs
(absolute offset 72 is the start of the detection array).

## VBE1 — event (native → Python)

Body is a UTF-8 JSON object:
`{"stream_index":3,"seq":120,"wall_ms":1790000000000.0,"analyzer":"line_cross","type":"line_cross","track_id":7,"fields":{...}}`.
Event records are never dropped.

## VBC1 — control/status (native → Python)

Body is UTF-8 JSON with an `op` field: `hello`, `reply`, `stats`, `stream_state`
(full schemas and state machine in spec BASE-1 §6.3).

## VBS1 — snapshot (native → Python)

Body = `u32 json_len` + JSON meta
(`{"req","stream_index","seq","track_id","w","h","mime"}`) + payload bytes
(e.g. JPEG).

## Python → native control lines

One UTF-8 JSON object per line, each with a `req` id: `add`, `remove`,
`set_threshold`, `configure_analyzer`, `snapshot`, `stop`, `infer_image`.
Single-line limit 2 MiB (enforced by the runtime, not the codec).
