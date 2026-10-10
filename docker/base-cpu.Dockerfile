# syntax=docker/dockerfile:1
# BASE-1 §6.10 / M1.14: CPU base image (dev tag). Build stage compiles
# vb-runtime with GStreamer/JPEG/TLS ON; the run stage carries only the
# runtime binary, ORT, GStreamer runtime libs, libssl, libturbojpeg and the
# vision_base package. CPU base is a development/CI platform, not a delivery
# target (§10.2 cpu row).
#
# Base image pin (debian:bookworm-slim manifest-list digest, resolved
# 2026-09-26 with `docker buildx imagetools inspect debian:bookworm-slim`):
#   sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251
#
# ONNX Runtime 1.20.1 linux tarball sha256 (recorded in M1.14):
#   aarch64  ae4fedbdc8c18d688c01306b4b50c63de3445cdf2dbd720e01a2fa3810b8106a
#   x64      67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105

ARG DEBIAN_DIGEST=sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251

FROM debian:bookworm-slim@${DEBIAN_DIGEST} AS build
ARG TARGETARCH
ARG ORT_VERSION=1.20.1
ARG ORT_SHA256_AARCH64=ae4fedbdc8c18d688c01306b4b50c63de3445cdf2dbd720e01a2fa3810b8106a
ARG ORT_SHA256_X64=67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config ca-certificates curl \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libturbojpeg0-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime C API, linux tarball for the target arch (§7).
RUN case "${TARGETARCH}" in \
      arm64) ort_sha="${ORT_SHA256_AARCH64}"; ort_arch=aarch64 ;; \
      amd64) ort_sha="${ORT_SHA256_X64}";  ort_arch=x64 ;; \
      *) echo "unsupported TARGETARCH ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL -o /tmp/ort.tgz \
       "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz" \
    && echo "${ort_sha}  /tmp/ort.tgz" | sha256sum -c - \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

COPY core-cpp/vb /src/core-cpp/vb
RUN cmake -S /src/core-cpp/vb -B /tmp/vb-build -DCMAKE_BUILD_TYPE=Release \
        -DVB_TESTS=OFF -DVB_BACKENDS="cpu" \
        -DVB_WITH_GST=ON -DVB_WITH_JPEG=ON -DVB_WITH_TLS=ON \
        -DORT_ROOT=/opt/onnxruntime \
    && cmake --build /tmp/vb-build -j"$(nproc)" \
    && mkdir -p /out/vb/bin /out/vb/py \
    && cp /tmp/vb-build/vb-runtime /out/vb/bin/ \
    && cp -r /src/core-cpp/vb /out/vb/src

FROM debian:bookworm-slim@${DEBIAN_DIGEST} AS runtime
ARG TARGETARCH
ARG ORT_VERSION=1.20.1

# Runtime libs only: GStreamer (uridecodebin soft-decode chain), OpenSSL,
# libturbojpeg, and a stdlib python3 for the vision_base orchestration layer.
# Note (spec R13): gstreamer1.0-libav licensing needs verification; the CPU
# base is a dev/CI image only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 \
        libssl3 libturbojpeg0 \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/vb/bin/vb-runtime /opt/vb/bin/vb-runtime
COPY --from=build /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY core-py/vision_base /opt/vb/py/vision_base

ENV PYTHONPATH=/opt/vb/py \
    LD_LIBRARY_PATH=/opt/vb/bin:/opt/onnxruntime/lib \
    VB_PRODUCTION=1

# Default entrypoint is the Python orchestrator, so `docker run <img> --config x`
# runs a config-only app (docs/quickstart.md). Standalone mode without Python:
# `--entrypoint /opt/vb/bin/vb-runtime <img> --standalone --config x`.
ENTRYPOINT ["python3", "-m", "vision_base.main"]
