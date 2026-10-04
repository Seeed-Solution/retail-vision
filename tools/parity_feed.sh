#!/bin/bash
# parity_feed.sh — BASE-1 parity frame-alignment test fixture (publisher side).
#
# Design: <parity-run-dir>/design-parity-frame-alignment-fixture.md
#
# Why this exists: with the old `-stream_loop -1` source, each `vb-runtime
# --parity` consumer joins the looping clip at a different phase, so line i of
# one parity.jsonl is NOT the same frame as line i of the other. This fixture
# publishes the clip **once** (single pass, no loop) as all-intra (keyint=1)
# frames at a reduced fps, and holds the payload back for a short join window
# so every consumer is attached before the first payload frame flies:
#
#   1. Pre-render: clip -> all-intra frames at --fps, ALL timestamps shifted
#      by --window seconds (the published file simply starts at t=window).
#      `ffmpeg -re` paces by timestamps, so the publisher opens the RTSP
#      session (the mediamtx path becomes ready), then sends nothing for
#      --window seconds, then plays the payload once.
#   2. Consumers must be started BEFORE this script: with the path absent
#      they keep retrying the RTSP open (runtime.open_timeout_s must be
#      >= 120), and they attach within ~0.5 s of the path appearing.
#   3. The join window must stay BELOW the readers' RTP starvation timeout
#      (rtspsrc tears a reader down after ~4 s without data — measured on
#      the vb base images, GStreamer rtsp src). Default 3 s.
#   4. This script polls the mediamtx API until all --readers are attached,
#      then releases nothing (the payload timing is fixed); if the readers
#      have not all attached within 120 s it exits 4 (fixture failure).
#   5. mediamtx readTimeout is raised (default 10 s would kill the silent
#      publisher mid-window).
#
# Every consumer attached before the payload starts captures exactly N
# frames (drop=true appsink never overflows at --fps << consumer capacity),
# so line i of every parity.jsonl is the same frame. A consumer that attaches
# late misses a prefix (fewer lines) — parity_fixture_check.py flags it as a
# fixture failure (exit 3); re-run the experiment, do not touch thresholds.
#
# Usage:
#   parity_feed.sh <clip.(mp4|mkv|...)> [--fps N] [--readers K] [--window S]
#                  [--rtsp-port P] [--api-port Q] [--path NAME]
#   MEDIAMTX_BIN=/path/to/mediamtx  (default: $WORK_DIR/bin/mediamtx)
#
# Prints PARITY_FRAMES=<N> (total frames minus the doorbell frame F0).
#
# Exit codes: 0 publish completed; 1 bad usage; 2 ffmpeg/ffprobe failure;
# 3 environment failure (mediamtx did not come up); 4 readers did not attach
# in time.
set -euo pipefail

usage() { sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; }

CLIP=""
FPS=5
READERS=2
WINDOW=8
WARMUP=5
RTSP_PORT=18664
API_PORT=19970
PATH_NAME=vb-parity-fixture
MEDIAMTX_BIN="${MEDIAMTX_BIN:-${WORK_DIR:-$HOME/vb-work}/bin/mediamtx-1.12.3}"
READERS_TIMEOUT=120

while [ $# -gt 0 ]; do
  case "$1" in
    --fps)       FPS="$2"; shift 2 ;;
    --readers)   READERS="$2"; shift 2 ;;
    --window)    WINDOW="$2"; shift 2 ;;
    --warmup)    WARMUP="$2"; shift 2 ;;
    --rtsp-port) RTSP_PORT="$2"; shift 2 ;;
    --api-port)  API_PORT="$2"; shift 2 ;;
    --path)      PATH_NAME="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    -*)          echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    *)           if [ -n "$CLIP" ]; then echo "unexpected arg: $1" >&2; exit 1; fi
                 CLIP="$1"; shift ;;
  esac
done

[ -n "$CLIP" ] || { usage >&2; exit 1; }
[ -f "$CLIP" ] || { echo "clip not found: $CLIP" >&2; exit 1; }
[ -x "$MEDIAMTX_BIN" ] || { echo "mediamtx binary not executable: $MEDIAMTX_BIN (set MEDIAMTX_BIN)" >&2; exit 3; }

WORK="$(mktemp -d /tmp/parity-feed.XXXXXX)"
cleanup() { kill "${PUB_PID:-0}" "${MM_PID:-0}" 2>/dev/null || true; [ -z "${PARITY_FEED_KEEPLOG:-}" ] || cp "$MM_LOG" /tmp/parity-feed-mediamtx.log 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

CLIP_ABS="$(readlink -f "$CLIP")"
GAPPED="$WORK/gapped.mp4"
MM_YML="$WORK/mediamtx.yml"
MM_LOG="$WORK/mediamtx.log"

# 1. Pre-render: reduced fps, all-intra (every frame independently decodable,
#    uniform decode cost), timestamps staggered as described in the header:
#      frame F0 (T<0.1):   t=0                    doorbell, not counted
#      frame F1 (T<0.3):   t=WINDOW               payload start
#      frames F2.. (else): t=WINDOW+WARMUP+rest   warm-up gap after F1
#    (after fps=, the k-th frame has T=(k-1)/FPS; the >=0.3 shift is
#    WINDOW + WARMUP + 1.6 so that F2's delta after F1 is exactly WARMUP.)
#    tpad then clones the last frame 6 times: the final RTP frames of a
#    single-pass stream are routinely lost at publisher teardown, and a
#    consumer that captured N-1 rows would otherwise silently wait forever
#    (run_parity has no flush before its 30 s deadline). Consumers stop
#    collecting at --frames N, so the clones never enter parity.jsonl of a
#    well-aligned run. -fps_mode passthrough keeps the gaps (otherwise ffmpeg
#    duplicates frames to fill them).
echo "[parity_feed] pre-rendering $CLIP_ABS -> all-intra ${FPS}fps, doorbell + ${WINDOW}s join + ${WARMUP}s warm-up gap" >&2
# Cache the render (keyed by source mtime+size+fps+schedule) so that repeated
# runs start the publisher with no jitter: consumer attach phase is keyed to
# the publisher start, so a variable-length render would shift it.
CACHE_DIR=/tmp/parity-feed-cache
mkdir -p "$CACHE_DIR"
CACHE_KEY="$(stat -c '%s-%Y' "$CLIP_ABS")-fps${FPS}-w${WINDOW}-u${WARMUP}"
CACHE_FILE="$CACHE_DIR/$CACHE_KEY.mp4"
if [ -s "$CACHE_FILE" ]; then
  cp "$CACHE_FILE" "$GAPPED"
  echo "[parity_feed] using cached render $CACHE_FILE" >&2
else
  ffmpeg -hide_banner -loglevel error -y -i "$CLIP_ABS" \
    -vf "fps=${FPS},tpad=stop_mode=clone:stop=6,setpts=PTS-STARTPTS+${WINDOW}/TB*gte(T\,0.1)+$(awk -v w="$WARMUP" 'BEGIN{printf "%.3f", w+1.6}')/TB*gte(T\,0.3)" \
    -an -fps_mode passthrough \
    -c:v libx264 -preset veryfast -tune zerolatency \
    -x264-params keyint=1:min-keyint=1 -pix_fmt yuv420p \
    "$GAPPED" || exit 2
  cp "$GAPPED" "$CACHE_FILE" || true
fi

N="$(ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of csv=p=0 "$GAPPED")" || exit 2
TOTAL="$N"
DOORBELL=1; PAD=6
N=$((TOTAL - DOORBELL - PAD))   # doorbell frame F0 + trailing clones are not payload
[ "$N" -gt 0 ] || { echo "clip produced no payload frames" >&2; exit 2; }
echo "PARITY_FRAMES=$N"
echo "[parity_feed] run consumers with --frames $N" >&2

# 2. Own mediamtx instance: fixture RTSP port, API enabled, TCP only (the
#    pre-existing instance on :8664 owns UDP 8000/8001 and RTMP 1935). The
#    publisher sits silent for the whole join window; the default 10 s read
#    timeout would tear it down mid-window.
cat > "$MM_YML" <<EOF
logLevel: info
rtspAddress: :${RTSP_PORT}
rtspTransports: [tcp]
readTimeout: 150s
writeTimeout: 150s
rtmp: no
hls: no
webrtc: no
srt: no
api: yes
apiAddress: 127.0.0.1:${API_PORT}
paths:
  ${PATH_NAME}:
    source: publisher
EOF

( cd "$WORK" && exec "$MEDIAMTX_BIN" "$MM_YML" ) >"$MM_LOG" 2>&1 &
MM_PID=$!

api_ready=0
for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${API_PORT}/v3/paths/list" >/dev/null 2>&1; then
    api_ready=1; break
  fi
  kill -0 "$MM_PID" 2>/dev/null || { echo "mediamtx exited early:" >&2; cat "$MM_LOG" >&2; exit 3; }
  sleep 0.2
done
[ "$api_ready" = 1 ] || { echo "mediamtx API did not come up:" >&2; cat "$MM_LOG" >&2; exit 3; }

# 3. Single-pass publish. -re paces by timestamps: --window seconds of
#    silence (reader join window), then the payload once.
ffmpeg -hide_banner -loglevel error -re -i "$GAPPED" -c copy \
  -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:${RTSP_PORT}/${PATH_NAME}" &
PUB_PID=$!
echo "CONSUMERS_NOW"   # start the consumers right now (join window is ticking)

# 4. Verify that all readers attached during the join window. The payload
#    timing is fixed (t = window), so this is a check, not a trigger: if the
#    consumers are not attached by the time the payload starts, exit 4 —
#    the run would produce prefix-truncated parity.jsonl files.
payload_start=$(( SECONDS + WINDOW ))
payload_deadline=$(( payload_start + 2 ))
while :; do
  readers="$(curl -sf "http://127.0.0.1:${API_PORT}/v3/paths/list" \
    | jq -r --arg p "$PATH_NAME" '[.items[] | select(.name==$p) | .readers | length] | .[0] // 0')" || readers=0
  [ "${readers:-0}" -ge "$READERS" ] && break
  if [ "$SECONDS" -ge "$payload_deadline" ]; then
    echo "TIMEOUT: only ${readers:-0}/$READERS reader(s) attached during the ${WINDOW}s join window" >&2
    echo "(consumers must be started BEFORE parity_feed.sh; see tests/fixtures/parity/README.md)" >&2
    exit 4
  fi
  sleep 0.2
done
echo "[parity_feed] $readers reader(s) attached; payload starts in $(( payload_deadline - SECONDS ))s" >&2

# 5. Payload plays once; when ffmpeg finishes, the run is complete.
wait "$PUB_PID" || { echo "publisher failed" >&2; exit 2; }
echo "[parity_feed] publish complete ($N payload frames)" >&2
exit 0
