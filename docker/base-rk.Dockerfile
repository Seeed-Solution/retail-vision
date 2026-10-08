# syntax=docker/dockerfile:1
# BASE-1 §M2.1 / "镜像与锁定": RK3576 + RK3588 base image.
#
# Build (the M2.1 offline acceptance):
#   docker build --target build -f docker/base-rk.Dockerfile .
#   -> arm64-native compile with -DVB_BACKENDS="cpu;rknn" -DVB_WITH_GST=ON
#
# Run (on a board; the boards supply the Rockchip stack as host mounts, spec
# §M2 "镜像与锁定" rule 3 / fall platforms/rk3576/docker-compose.yml):
#   docker run --rm --network host --privileged \
#     -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro \
#     -v /usr/lib/aarch64-linux-gnu/librga.so.2:/usr/lib/aarch64-linux-gnu/librga.so:ro \
#     -v /usr/lib/aarch64-linux-gnu/librockchip_mpp.so.1:... \
#     -v <host gstreamer libs and rockchip plugins> \
#     -v /dev/dri:/dev/dri  <image> ...
#
# Nothing Rockchip-specific is redistributed: the RKNN header and both vendor
# libraries are fetched at build time under a pinned sha256 and are absent from
# the runtime stage. Only the Apache-2.0 librga headers are committed
# (core-cpp/vb/backends/rknn/third_party/THIRD_PARTY.md records every URL, hash
# and the license finding for each).
#
# Base image pin: same debian:bookworm-slim manifest-list digest base-cpu uses
# (resolved 2026-09-26 with `docker buildx imagetools inspect debian:bookworm-slim`).
ARG DEBIAN_DIGEST=sha256:3783cc01769c7b2b1b83a5c5ad96c815348e28ed7da68e2e3687004faa906251

FROM debian:bookworm-slim@${DEBIAN_DIGEST} AS build
ARG TARGETARCH
ARG APT_MIRROR=http://deb.debian.org
ARG ORT_VERSION=1.20.1
ARG ORT_SHA256_AARCH64=ae4fedbdc8c18d688c01306b4b50c63de3445cdf2dbd720e01a2fa3810b8106a
ARG ORT_SHA256_X64=67db4dc1561f1e3fd42e619575c82c601ef89849afc7ea85a003abbac1a1a105
ARG ORT_URL=
ARG BUILD_JOBS=8

# Rockchip SDK inputs, all fetched here and all sha256-pinned:
#   rknn_api.h   airockchip/rknn-toolkit2 v2.3.2 (proprietary; see THIRD_PARTY.md)
#   librknnrt.so airockchip/rknn-toolkit2 v2.3.2, aarch64
#   librga.so    airockchip/librga v1.10.0 (fb3357d), gcc-aarch64
ARG RKNN_API_URL=https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h
ARG RKNN_API_SHA256=c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a
ARG RKNN_LIB_URL=https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
ARG RKNN_LIB_SHA256=d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8
ARG RGA_LIB_URL=https://raw.githubusercontent.com/airockchip/librga/fb3357d09008222bc5e27bdaadf74a0c5ea4c86e/libs/Linux/gcc-aarch64/librga.so
ARG RGA_LIB_SHA256=3da1413445885420abf00821640ec8a37289ec176fe4ffeee0de5f68418ed50e

RUN apt_bootstrap_mirror="$(printf '%s' "${APT_MIRROR}" | sed 's|^https://|http://|')" \
    && sed -i \
        -e "s|http://deb.debian.org/debian|${apt_bootstrap_mirror}/debian|g" \
        -e "s|http://deb.debian.org/debian-security|${apt_bootstrap_mirror}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i \
        -e "s|${apt_bootstrap_mirror}/debian|${APT_MIRROR}/debian|g" \
        -e "s|${apt_bootstrap_mirror}/debian-security|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config curl \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libturbojpeg0-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime C API for the cpu backend (the acceptance builds cpu;rknn, and
# the cpu backend is the parity reference implementation).
RUN case "${TARGETARCH}" in \
      arm64) ort_sha="${ORT_SHA256_AARCH64}"; ort_arch=aarch64 ;; \
      amd64) ort_sha="${ORT_SHA256_X64}";  ort_arch=x64 ;; \
      *) echo "unsupported TARGETARCH ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && ort_url="${ORT_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz}" \
    && curl -fsSL -o /tmp/ort.tgz \
       "${ort_url}" \
    && echo "${ort_sha}  /tmp/ort.tgz" | sha256sum -c - \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz

# Rockchip libraries. These are link-time inputs only: they are never copied
# into the runtime stage, so the shipped image contains no vendor library.
RUN case "${TARGETARCH}" in \
      arm64) ;; \
      *) echo "the RK base is arm64 only (TARGETARCH=${TARGETARCH})" >&2; exit 1 ;; \
    esac \
    && mkdir -p /opt/rk \
    && curl -fsSL -o /opt/rk/librknnrt.so "${RKNN_LIB_URL}" \
    && echo "${RKNN_LIB_SHA256}  /opt/rk/librknnrt.so" | sha256sum -c - \
    && curl -fsSL -o /opt/rk/librga.so "${RGA_LIB_URL}" \
    && echo "${RGA_LIB_SHA256}  /opt/rk/librga.so" | sha256sum -c -

COPY core-cpp/vb /src/core-cpp/vb

# The RKNN header is proprietary and is not tracked in git; it is placed at the
# path the backend expects at build time.
RUN curl -fsSL -o /src/core-cpp/vb/backends/rknn/third_party/rknn/rknn_api.h \
        "${RKNN_API_URL}" \
    && echo "${RKNN_API_SHA256}  /src/core-cpp/vb/backends/rknn/third_party/rknn/rknn_api.h" \
       | sha256sum -c -

RUN cmake -S /src/core-cpp/vb -B /tmp/vb-build -DCMAKE_BUILD_TYPE=Release \
        -DVB_TESTS=OFF -DVB_BACKENDS="cpu;rknn" \
        -DVB_WITH_GST=ON -DVB_WITH_JPEG=ON -DVB_WITH_TLS=ON \
        -DORT_ROOT=/opt/onnxruntime \
        -DRKNN_LIB_DIR=/opt/rk -DRGA_LIB_DIR=/opt/rk \
    && cmake --build /tmp/vb-build -j"${BUILD_JOBS}" \
    && mkdir -p /out/vb/bin /out/vb/py \
    && cp /tmp/vb-build/vb-runtime /out/vb/bin/ \
    && cp -r /src/core-cpp/vb /out/vb/src

FROM debian:bookworm-slim@${DEBIAN_DIGEST} AS runtime
ARG TARGETARCH
ARG APT_MIRROR=http://deb.debian.org

# No Rockchip library, no RKNN header and no Python compute library: the boards
# supply librknnrt.so / librga.so / librockchip_mpp.so.1 and the Rockchip
# GStreamer plugin as read-only host mounts. What is installed here is the
# GStreamer core plus plugins-good (rtspsrc and RTP depayloaders) and
# plugins-bad (h264parse/h265parse), OpenSSL, libturbojpeg and a stdlib python3
# for vision_base. gstreamer1.0-tools provides gst-inspect-1.0 for diagnostics.
RUN apt_bootstrap_mirror="$(printf '%s' "${APT_MIRROR}" | sed 's|^https://|http://|')" \
    && sed -i \
        -e "s|http://deb.debian.org/debian|${apt_bootstrap_mirror}/debian|g" \
        -e "s|http://deb.debian.org/debian-security|${apt_bootstrap_mirror}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && sed -i \
        -e "s|${apt_bootstrap_mirror}/debian|${APT_MIRROR}/debian|g" \
        -e "s|${apt_bootstrap_mirror}/debian-security|${APT_MIRROR}/debian-security|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        libssl3 libturbojpeg0 \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
        gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
        gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/vb/bin/vb-runtime /opt/vb/bin/vb-runtime
COPY --from=build /opt/onnxruntime/lib /opt/onnxruntime/lib
COPY core-py/vision_base /opt/vb/py/vision_base

ENV PYTHONPATH=/opt/vb/py \
    LD_LIBRARY_PATH=/opt/vb/bin:/opt/onnxruntime/lib \
    VB_PRODUCTION=1

ENTRYPOINT ["python3", "-m", "vision_base.main"]
