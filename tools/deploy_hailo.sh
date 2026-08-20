#!/bin/sh
# Board-local build + deploy. Nothing is pushed to a registry: the image is
# built on the Pi and stays there.
set -eu

BASE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL_DIR=${RETAIL_VISION_MODEL_DIR:-"$BASE/models"}
MODEL_PATH=$MODEL_DIR/yolov8n.hef
MODEL_URL=https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8/yolov8n.hef
EXPECTED_SHA256=4f250daa6ac16e030cfaaf164e1cbdfe83e5852c81ae6c339c94857dd2b7b8d2

accepted=0
skip_build=0
for arg in "$@"; do
    case "$arg" in
        --accept-upstream-license) accepted=1 ;;
        --skip-build) skip_build=1 ;;
        -h|--help)
            echo "Usage: ./deploy.sh --accept-upstream-license [--skip-build]"
            exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$accepted" -ne 1 ]; then
    echo "refusing: pass --accept-upstream-license after reviewing Hailo's terms for the HEF" >&2
    exit 2
fi

# 1. Model. Downloaded from Hailo's official fixed URL, never re-hosted here,
#    never baked into the image, and rejected unless the digest matches.
mkdir -p "$MODEL_DIR"
if [ ! -f "$MODEL_PATH" ]; then
    echo "downloading $MODEL_URL"
    curl -fsSL -o "$MODEL_PATH.tmp" "$MODEL_URL"
    mv "$MODEL_PATH.tmp" "$MODEL_PATH"
fi
actual=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "HEF digest mismatch: $actual != $EXPECTED_SHA256" >&2
    exit 3
fi
echo "HEF ok: $MODEL_PATH ($actual)"

# 2. ABI check before anything is built. The hailonet plugin, libhailort and the
#    kernel driver must be one version; a mismatch fails at the first frame with
#    an error that points at none of them.
drv=$(cat /sys/module/hailo_pci/version 2>/dev/null || echo unknown)
lib=$(ls /usr/lib/libhailort.so.* 2>/dev/null | head -1 | sed 's/.*libhailort\.so\.//')
echo "HailoRT driver=$drv userspace=$lib"
if [ "$drv" != "$lib" ]; then
    echo "HailoRT driver/library mismatch; refusing to deploy" >&2
    exit 4
fi

# 3. Build the native binary inside the builder container, then the runtime
#    image around it.
if [ "$skip_build" -eq 0 ]; then
    docker compose -f "$BASE/docker-compose.yml" --profile build build builder
    docker compose -f "$BASE/docker-compose.yml" --profile build run --rm builder
    docker compose -f "$BASE/docker-compose.yml" build retail-vision
fi

# 4. Start only this service. No project-wide down: other containers share this
#    host.
docker compose -f "$BASE/docker-compose.yml" up -d retail-vision
docker compose -f "$BASE/docker-compose.yml" ps retail-vision
