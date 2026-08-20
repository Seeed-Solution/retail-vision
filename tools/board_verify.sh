#!/bin/bash
# Build/run/verify retail-vision on one Rockchip board.
#   ./tools/board_verify.sh <rk3588|rk3576> <video-path> [seconds]
# Env: SKIP_BUILD=1     reuse an already-loaded image
#      START_RTSP=1     also start a mediamtx RTSP server (default port 8554);
#                       omit when the board already runs one and we only need
#                       to publish an extra path onto it.
# Creates only its own containers (rv-mqtt, rv-pub, optional rv-rtsp, and the
# compose service). Never touches anything else on the board.
set -u
T="${1:?target}"; VIDEO="${2:?video}"; SECS="${3:-70}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
MTX="${MTX_IMAGE:-docker.m.daocloud.io/bluenviron/mediamtx:latest-ffmpeg}"
MOSQ=docker.m.daocloud.io/library/eclipse-mosquitto:2
CNAME="retail-vision-$T"
RTSP_URL="rtsp://127.0.0.1:8554/retail"

echo "=== INFRA ==="
docker rm -f rv-mqtt rv-pub "$CNAME" >/dev/null 2>&1
[ "${START_RTSP:-0}" = "1" ] && docker rm -f rv-rtsp >/dev/null 2>&1
[ -f /tmp/rv-pub.pid ] && kill "$(cat /tmp/rv-pub.pid)" 2>/dev/null; rm -f /tmp/rv-pub.pid
docker run -d --name rv-mqtt --network host "$MOSQ" \
  sh -c 'printf "listener 1883 0.0.0.0\nallow_anonymous true\n" > /m.conf; exec mosquitto -c /m.conf' >/dev/null
if [ "${START_RTSP:-0}" = "1" ]; then
  # Default config: RTSP on 8554. Do not override the listen address by env --
  # getting that wrong is silent, the server simply never binds.
  docker run -d --name rv-rtsp --network host "$MTX" >/dev/null
  sleep 6
fi
echo "--- 8554 listener before publishing ---"; ss -ltn 2>/dev/null | grep 8554 || echo "(none)"
# Prefer a host ffmpeg: pulling the mediamtx ffmpeg-flavoured image costs
# 266 MB, and cat-remote has under 2 GB free.
if command -v ffmpeg >/dev/null 2>&1; then
  echo "(publishing with host ffmpeg)"
  nohup ffmpeg -re -stream_loop -1 -i "$VIDEO" -an -c copy -f rtsp \
    -rtsp_transport tcp "$RTSP_URL" > /tmp/rv-pub.log 2>&1 &
  echo $! > /tmp/rv-pub.pid
else
  docker run -d --name rv-pub --network host -v "$VIDEO":/v.mp4:ro --entrypoint ffmpeg "$MTX" \
    -re -stream_loop -1 -i /v.mp4 -an -c copy -f rtsp -rtsp_transport tcp "$RTSP_URL" >/dev/null
fi
sleep 8
docker ps -a --filter name=rv- --format "{{.Names}}\t{{.Status}}"
echo "--- publisher log ---"; { docker logs rv-pub 2>&1 || cat /tmp/rv-pub.log; } | tail -6
echo "--- probe the published path ---"
if command -v ffprobe >/dev/null 2>&1; then
  ffprobe -v error -rtsp_transport tcp -select_streams v:0 \
    -show_entries stream=codec_name,width,height -of default=nw=1 "$RTSP_URL" 2>&1 | tail -6
else
  docker run --rm --network host --entrypoint ffprobe "$MTX" -v error -rtsp_transport tcp \
    -select_streams v:0 -show_entries stream=codec_name,width,height -of default=nw=1 "$RTSP_URL" 2>&1 | tail -6
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  echo "=== BUILD ==="
  DOCKER_BUILDKIT=1 docker build --network host --target rknn -f docker/Dockerfile -t retail-vision-rknn:0.1.0 . 2>&1 | tail -12
fi
echo "=== IMAGE ==="
docker images retail-vision-rknn --format "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}"
docker image inspect retail-vision-rknn:0.1.0 --format "virtual_size_bytes={{.Size}}"

echo "=== RUN ==="
( cd "boards/$T" && docker compose up -d retail-vision 2>&1 | tail -4 )
sleep 30
echo "--- startup log ---"; docker logs "$CNAME" 2>&1 | tail -8

echo "=== HARDWARE DECODE EVIDENCE (while running) ==="
echo "--- plugin lookup inside the container ---"
docker exec "$CNAME" python3 -c "
import gi; gi.require_version('Gst','1.0')
from gi.repository import Gst; Gst.init(None)
r=Gst.Registry.get()
for n in ('mppvideodec','avdec_h264'):
    f=r.lookup_feature(n); print(n,'->','PRESENT' if f else 'ABSENT')
" 2>&1 | tail -4
echo "--- host MPP/RGA/RKNN libraries mapped into the live process ---"
# Plugin presence only proves it could be used. These are the libraries the
# Rockchip decoder and the NPU actually link against; mapped into the process
# means the hardware paths are live, not a silent CPU fallback.
docker exec "$CNAME" sh -c 'grep -oE "/[^ ]*(rockchip_mpp|librga|librknnrt)[^ ]*" /proc/1/maps | sort -u' 2>&1
echo "--- avdec_h264 (CPU decoder) NOT mapped: ---"
docker exec "$CNAME" sh -c 'grep -c libgstlibav /proc/1/maps || true' 2>&1
echo "--- /dev/dri in container ---"; docker exec "$CNAME" ls /dev/dri 2>&1 | head
echo "--- CPU usage (a 1280x720 CPU decode would pin a core) ---"
docker stats --no-stream --format "{{.Name}}\tCPU={{.CPUPerc}}\tMEM={{.MemUsage}}" "$CNAME" 2>&1
echo "--- host NPU load ---"; cat /sys/kernel/debug/rknpu/load 2>/dev/null || echo "(rknpu debugfs not readable)"

echo "=== MQTT CAPTURE (${SECS}s) ==="
docker run --rm --network host "$MOSQ" mosquitto_sub -h 127.0.0.1 \
  -t "store-demo/retail-vision/results/#" -v -W "$SECS" > /tmp/rv-results.txt 2>/tmp/rv-sub.err
docker run --rm --network host "$MOSQ" mosquitto_sub -h 127.0.0.1 \
  -t "store-demo/retail-vision/status" -v -C 1 -W 5 > /tmp/rv-status.txt 2>&1
echo "--- retained status topic ---"; cat /tmp/rv-status.txt
sed 's|^[^ ]* ||' /tmp/rv-results.txt > /tmp/rv-results.jsonl
N=$(grep -c . /tmp/rv-results.jsonl || true)
echo "--- MQTT rate ---"
echo "messages=$N window=${SECS}s rate=$(python3 -c "print(f'{$N/$SECS:.2f}')") msg/s"
echo "--- RAW payload with the most persons ---"
python3 - <<'PY'
import json
best=None
for line in open('/tmp/rv-results.jsonl'):
    line=line.strip()
    if not line: continue
    try: m=json.loads(line)
    except Exception: continue
    if best is None or len(m.get('persons',[]))>len(best.get('persons',[])): best=m
print(json.dumps(best) if best else 'NO MESSAGES')
print('---- pretty ----')
print(json.dumps(best, indent=1) if best else '')
PY
echo "--- measured FPS / backend across the capture ---"
python3 - <<'PY'
import json, collections
fps=[]; back=collections.Counter(); inf=[]; npeople=collections.Counter()
for l in open('/tmp/rv-results.jsonl'):
    l=l.strip()
    if not l: continue
    try: m=json.loads(l)
    except Exception: continue
    fps.append(m.get('fps',0)); inf.append(m.get('inference_time_ms',0))
    back[m.get('source_backend')]+=1; npeople[len(m.get('persons',[]))]+=1
if fps:
    print(f"fps: min={min(fps):.1f} max={max(fps):.1f} mean={sum(fps)/len(fps):.1f}")
    print(f"inference_time_ms: mean={sum(inf)/len(inf):.1f}")
print("source_backend counts:", dict(back))
print("persons-per-message histogram:", dict(sorted(npeople.items())))
PY
echo "--- contract validation ---"
python3 contracts/validate_payload.py /tmp/rv-results.jsonl
echo "=== LOGS (tail) ==="; docker logs "$CNAME" 2>&1 | tail -12
echo "=== LOG ERROR GREP ==="
docker logs "$CNAME" 2>&1 | grep -iE 'error|traceback|crash|fail' || echo "(no matches)"
