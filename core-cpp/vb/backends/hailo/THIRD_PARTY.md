# Third-party provenance — core-cpp/vb/backends/hailo

## fall-detection `platforms/rpi-hailo/src/` batch runtime — Apache-2.0

Copied into this backend (spec BASE-1 §M2.3) from `fall-detection`
`origin/main@eb72e1e`, `platforms/rpi-hailo/src/`:

| file here | fall source |
|---|---|
| `batch_policy.{h,cpp}` | same name |
| `frame_batcher.{h,cpp}` | same name |
| `runner_lifecycle.h` | same name |
| `runtime_config.h` | same name |
| `batched_hailo_runner.{h,cpp}` | same name (+ `hailo_pose_decoder.h`'s RawTensor shape) |
| `batched_hailo_runner_stub.cpp` | new (VB_HAILO_STUB fake runner, no upstream code) |
| `hailo_backend.{h,cpp}`, `register.cpp`, `model_sha256.{h,cpp}` | new (BASE-1 §6.1 adapter; sha256 copied from `backends/rknn/model_sha256.cpp` under a distinct symbol) |
| `hailo_source.{h,cpp}` | new; pipeline shape from fall `main.cpp` `pipelinePrefix`/`sharedPipeline` + codec chain of fall `a2d0e58` |

- Upstream license: fall-detection is Apache-2.0; the copied files carry it.
- Namespace changed `rpi_hailo` / `rpi_hailo_config` → `vb` / `vb::hailo_config`;
  `BatchedHailoRunner` gained the synchronous `infer()` entry for
  `InferenceContext::infer` (§6.1) beside the copied batcher/worker path; logic
  otherwise unchanged. `hailo_pose_decoder.cpp` itself is NOT copied: its
  layout is decoded by the shared `vb::` yolo_pose decoder
  (`src/post/yolov8_decode.cpp`), and its keypoint formula question is
  covered by the M2.3 fixed-input measurement.

## HailoRT — not redistributed

Headers and `libhailort.so` come from the host (`/usr/include/hailo`,
`/usr/lib/libhailort.so`, found via `find_package(HailoRT)`); nothing
HailoRT-related is committed here or shipped in an image (spec §M2.3
guardrail). Runtime builds record `libhailort` as DT_NEEDED.

## Pose HEF used by the on-device acceptance (not in this repository)

`yolov8s_pose.hef`, Hailo Model Zoo v2.15.0 hailo8 build, sha256
`e19856699ed47cf866d23265827f960b263f287dab5e54e82c7ce37e12525a2d` — the same
model fall deploys. Upstream URL:
`https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8s_pose.hef`.
The Model Zoo records this network as trained on COCO keypoints with
`source: https://github.com/ultralytics/ultralytics` and
`license_name: AGPL-3.0`; the HEF is used for parity testing only, is never
committed or baked into an image, and the same caveat applies to the float
ONNX reference (`yolov8s_pose.onnx`, sha256
`a717661f682d4c4746fe5baff9ef73d651472e62ae13ed825bb1194d7db9965f`,
`PoseEstimation/yolov8/yolov8s/pretrained/2023-06-11`). License review of this
lineage is tracked separately (URD-5); it is not resolved here.
