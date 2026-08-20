# HEF model

`yolov8n.hef` is **reused**, not compiled here. It is the official Hailo Model
Zoo COCO detection HEF for Hailo-8:

    https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef
    sha256 4f250daa6ac16e030cfaaf164e1cbdfe83e5852c81ae6c339c94857dd2b7b8d2

Compiling a HEF needs the Hailo Dataflow Compiler, which is x86-only; the Pi
carries the HailoRT runtime alone. Reusing the published artifact avoids
standing up that toolchain for a model that already exists.

NMS runs **on-chip**: the HEF's single output vstream is
`HAILO NMS BY CLASS`, FLOAT32, 80 classes, 100 boxes per class. The process
therefore reads decoded boxes, not raw quantized feature maps, and
`src/hailo_det_decoder.cpp` only walks the class-0 (person) block.

`deploy.sh --accept-upstream-license` downloads it and refuses any file whose
digest differs. The binary is not stored in git and is not baked into the
runtime image; Compose bind-mounts this directory read-only.

The repository's own license does not cover this file — Hailo's upstream terms
apply.
