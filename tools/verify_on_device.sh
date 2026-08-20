#!/bin/bash
# On-device verification run. Opens the shortest possible window in which the
# NPU is free: stops the unrelated container that holds /dev/hailo0, runs the
# pipeline for a fixed benchmark period while sampling proof that inference is
# on the NPU, then restarts it and checks it came back healthy.
#
# Usage: sudo ./tools/verify_on_device.sh [seconds]
set -u

D=/home/harvest/project/retail-vision-hailo
L=/tmp/rv-verify2
SECS=${1:-60}
# Second arg overrides the source, so the same window/restore procedure covers
# both the local file replay and a real RTSP camera.
STREAM=${2:-"cam-01|file:///data/two_people_long.mp4"}
HOLDER=mcp_face_rec

rm -rf "$L"; mkdir -p "$L" /tmp/hmon_files; chmod 777 /tmp/hmon_files

echo "[1] stopping $HOLDER (it holds /dev/hailo0)"
date -Is > "$L/stop_ts"
docker stop "$HOLDER" >/dev/null 2>&1

docker rm -f rv-sub >/dev/null 2>&1
docker run -d --rm --network host --name rv-sub eclipse-mosquitto:2 \
  mosquitto_sub -h 127.0.0.1 -t 'store-01/#' -v >/dev/null 2>&1
sleep 2

echo "[2] evidence sampler armed"
(
  sleep 12
  # Who holds the NPU *while the run is in flight*.
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    if ls -l "/proc/$p/fd" 2>/dev/null | grep -q hailo0; then
      {
        echo "PID $p cmd=$(tr '\0' ' ' < "/proc/$p/cmdline")"
        head -1 "/proc/$p/cgroup"
      } >> "$L/holder.txt"
    fi
  done
  docker stats --no-stream --format '{{.Name}} CPU={{.CPUPerc}} MEM={{.MemUsage}}' \
    rv-run > "$L/stats.txt" 2>&1
  HAILO_MONITOR=1 timeout 12 hailortcli monitor > "$L/monitor.txt" 2>&1
) >/dev/null 2>&1 &

echo "[3] running pipeline for ${SECS}s on: $STREAM"
docker run --rm --network host --name rv-run \
  --device /dev/hailo0:/dev/hailo0 \
  -v "$D/models:/models:ro" -v "$D/testdata:/data:ro" \
  -v /tmp/hmon_files:/tmp/hmon_files \
  -v /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo.so:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo.so:ro \
  -v /usr/lib/libhailort.so.4.21.0:/usr/lib/libhailort.so.4.21.0:ro \
  -e HAILO_MONITOR=1 -e HEF_PATH=/models/yolov8n.hef \
  -e STREAMS="$STREAM" \
  -e MQTT_HOST=127.0.0.1 -e PUBLISH_HZ=1.0 -e SCORE_THRESHOLD=0.4 \
  -e INSTALLATION_ID=store-01 -e BENCHMARK_SECONDS="$SECS" \
  retail-vision-hailo:0.1.0 > "$L/app.log" 2>&1
echo "app exit=$?"

docker logs rv-sub > "$L/mqtt.log" 2>&1
docker rm -f rv-sub >/dev/null 2>&1
date -Is > "$L/start_ts"

echo "[4] restarting $HOLDER"
docker start "$HOLDER" >/dev/null 2>&1
sleep 15
docker ps --filter "name=$HOLDER" --format '{{.Names}} {{.Status}}'
curl -s -o /dev/null -w 'face_rec_health_http=%{http_code}\n' --max-time 10 \
  http://127.0.0.1:8001/health

echo "window: $(cat "$L/stop_ts") -> $(cat "$L/start_ts")"
wc -l "$L/mqtt.log" "$L/app.log"
