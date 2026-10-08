# syntax=docker/dockerfile:1
# Native Jetson vision-base runtime (TensorRT/NVDEC).
#
# The TensorRT/CUDA build is deliberately supplied as a local artifact image.
# Jetson SDK images are tied to the target JetPack/L4T and are too large and
# host-specific to redistribute from this repository. Build the artifact on a
# Jetson with the matching SDK, then pass its local image name as
# ARTIFACT_IMAGE. The artifact image must contain only:
#   /opt/vb/bin/vb-runtime
# and the caller must verify VB_RUNTIME_SHA256 before building this runtime.
#
# Example (after an SDK-side build, with the binary hash checked):
#   docker build --build-arg ARTIFACT_IMAGE=vb-jetson-artifact:local \
#     --build-arg VB_RUNTIME_SHA256=<sha256> \
#     -f docker/base-jetson.Dockerfile -t vb-base-jetson:local .
#
# No TensorRT/CUDA/host vendor library is copied into the runtime stage. The
# target Jetson supplies those ABI-locked libraries through the deployment
# compose file and NVIDIA Container Runtime.

# Ubuntu 22.04 matches the target JetPack 6.2 userspace (L4T R36.4.x).
# Digest resolved from the local arm64 mirror metadata on 2026-10-05.
ARG RUNTIME_BASE=docker.m.daocloud.io/library/ubuntu@sha256:b8b6ee6aa931ecd9d0d952abc34dc0e5f7c6a30c6bb71b079fe399fde0329c02
ARG ARTIFACT_IMAGE=localhost/vb-jetson-artifact:local

FROM ${ARTIFACT_IMAGE} AS vb-artifact

FROM ${RUNTIME_BASE} AS runtime
ARG VB_RUNTIME_SHA256
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i \
      -e 's|http://ports.ubuntu.com/ubuntu-ports|http://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports|g' \
      -e 's|http://archive.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
      -e 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' \
      /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates python3 libssl3 libstdc++6 libturbojpeg \
       libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
       gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
       gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/* \
    && if [ -d /opt/vb/py/vision_base ]; then \
         find /opt/vb/py/vision_base -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete; \
         find /opt/vb/py/vision_base -depth -type d -name __pycache__ -empty -delete; \
       fi

COPY --from=vb-artifact /opt/vb/bin/vb-runtime /opt/vb/bin/vb-runtime
COPY core-py/vision_base /opt/vb/py/vision_base

# Fail at image construction if the artifact identity was not supplied. The
# digest is evidence in the build log/review report, not an OCI registry digest.
RUN test -n "${VB_RUNTIME_SHA256}" \
    && echo "${VB_RUNTIME_SHA256}  /opt/vb/bin/vb-runtime" | sha256sum -c - \
    && chmod 0755 /opt/vb/bin/vb-runtime

ENV PYTHONPATH=/opt/vb/py \
    LD_LIBRARY_PATH=/opt/vb/bin:/usr/local/cuda/lib64:/host-nvidia:/host-trt:/usr/lib/aarch64-linux-gnu \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    VB_PRODUCTION=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python3", "-m", "vision_base.main"]
