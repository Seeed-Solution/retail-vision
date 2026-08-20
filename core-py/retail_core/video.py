"""Shared GStreamer RTSP video I/O.

The pipeline shape (rtspsrc -> depay -> parse -> decoder -> RGB appsink) and
the software-decode fallback are identical on every platform; only the
hardware decoder element differs, so backends register theirs by name with
`register_source`. Rockchip's mppvideodec lives in backends/rknn/video_mpp.py.

Adapted from the fall-detection RKNN runtime. The OpenCV/FFmpeg fallback of
that implementation was replaced by a GStreamer software path so the image
does not have to carry opencv-python-headless (~350 MB installed); the boards
used here have under 2 GB of free disk. The active backend is reported in
every published payload as `source_backend`, so a silent downgrade from
`gstreamer_mpp` to `gstreamer_soft` is visible in the MQTT stream itself.

A third backend, `cv2_ffmpeg`, exists for images whose OpenCV is built with
`GStreamer: NO` (the NVIDIA Jetson ultralytics images are), where neither
GStreamer path can be constructed at all. It is opt-in per config and imports
cv2 lazily, so nothing changes for the RK boards.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np


def aspect_fit_geometry(source_w: int, source_h: int, size: int = 640):
    if source_w <= 0 or source_h <= 0 or size <= 0:
        raise ValueError("source dimensions and size must be positive")
    scale = min(size / source_w, size / source_h)
    scaled_w = int(round(source_w * scale))
    scaled_h = int(round(source_h * scale))
    return scaled_w, scaled_h, (size - scaled_w) // 2, (size - scaled_h) // 2


def pad_scaled_rgb(scaled: np.ndarray, size: int = 640, color: int = 114):
    if scaled.dtype != np.uint8 or scaled.ndim != 3 or scaled.shape[2] != 3:
        raise ValueError("scaled image must be HWC uint8 RGB")
    height, width = scaled.shape[:2]
    if width > size or height > size:
        raise ValueError("scaled image does not fit the letterbox canvas")
    left, top = (size - width) // 2, (size - height) // 2
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    canvas[top:top + height, left:left + width] = scaled
    return canvas


def copy_strided_rgb_to_letterbox(data, width: int, height: int, stride: int,
                                  size: int = 640, color: int = 114):
    """Copy mapped, potentially row-aligned RGB into an owned letterbox canvas."""
    if width <= 0 or height <= 0 or stride < width * 3:
        raise ValueError("invalid mapped RGB dimensions or stride")
    if len(data) < stride * height:
        raise ValueError("mapped RGB buffer is shorter than its negotiated layout")
    borrowed = np.ndarray((height, width, 3), dtype=np.uint8, buffer=data,
                          strides=(stride, 3, 1))
    return pad_scaled_rgb(borrowed, size, color)


def resize_nearest(image: np.ndarray, out_w: int, out_h: int):
    """NumPy nearest-neighbour resize; only used on the rare oversize path."""
    h, w = image.shape[:2]
    ys = (np.arange(out_h) * (h / out_h)).astype(np.int32).clip(0, h - 1)
    xs = (np.arange(out_w) * (w / out_w)).astype(np.int32).clip(0, w - 1)
    return image[ys][:, xs]


class _GstBase:
    def __init__(self, url, size=640, transport="tcp", codec="h264",
                 latency_ms=100, appsink_timeout_ms=2000, appsink_queue=3, **_):
        self.url = url
        self.size = int(size)
        self.transport = transport
        self.codec = str(codec).lower()
        self.latency_ms = int(latency_ms)
        self.timeout_ns = int(appsink_timeout_ms) * 1_000_000
        self.appsink_queue = max(1, int(appsink_queue))
        self.pipeline = self.sink = self.bus = None
        self.Gst = None
        self.GstVideo = None

    def _import_gi(self):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstRtsp", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstRtsp, GstVideo
        except Exception as exc:
            raise RuntimeError("PyGObject GStreamer bindings are unavailable") from exc
        if self.codec not in ("h264", "h265"):
            raise ValueError(f"unsupported RTSP codec: {self.codec}")
        Gst.init(None)
        self.Gst, self.GstVideo = Gst, GstVideo
        return Gst, GstRtsp

    def _make(self, factory, name):
        element = self.Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"missing GStreamer element: {factory}")
        return element

    def _configure_source(self, source, GstRtsp):
        source.set_property("location", self.url)
        source.set_property("latency", self.latency_ms)
        source.set_property("drop-on-latency", True)
        source.set_property("protocols",
                            GstRtsp.RTSPLowerTrans.TCP if self.transport == "tcp"
                            else GstRtsp.RTSPLowerTrans.UDP)

    def _configure_sink(self, sink):
        sink.set_property("sync", False)
        # Depth 1 means read() can only hand back the frame arriving next, so
        # any jitter costs a whole frame period. A small queue absorbs that for
        # at most a couple of frames of latency; drop=True keeps the reader on
        # recent frames.
        sink.set_property("max-buffers", self.appsink_queue)
        sink.set_property("drop", True)
        sink.set_property("emit-signals", False)

    def _bus_error(self):
        if self.bus is None:
            return None
        message = self.bus.pop_filtered(self.Gst.MessageType.ERROR | self.Gst.MessageType.EOS)
        if message is None:
            return None
        if message.type == self.Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            return RuntimeError(f"{self.backend_name} error: {error}; {debug or ''}")
        return RuntimeError(f"{self.backend_name} stream reached EOS")

    def _pull(self):
        if self.pipeline is None:
            self.start()
        error = self._bus_error()
        if error:
            self.close()
            raise error
        sample = self.sink.emit("try-pull-sample", self.timeout_ns)
        if sample is None:
            error = self._bus_error()
            if error:
                self.close()
                raise error
            # A pull that simply timed out is not fatal. Tearing the pipeline
            # down here means the next read() rebuilds the RTSP connection from
            # scratch, which cannot finish inside one appsink timeout either --
            # that is how hardware decode never survives its own startup and
            # every deployment silently falls back to software.
            return None
        return sample

    def close(self):
        if self.pipeline is not None:
            self.pipeline.set_state(self.Gst.State.NULL)
        self.pipeline = self.sink = self.bus = None


class GStreamerSoftware(_GstBase):
    """CPU decode fallback: avdec_* + videoscale add-borders letterbox."""

    backend_name = "gstreamer_soft"

    def start(self):
        Gst, GstRtsp = self._import_gi()
        pipeline = Gst.Pipeline.new("retail-rtsp-soft")
        source = self._make("rtspsrc", "source")
        depay = self._make(f"rtp{self.codec}depay", "depay")
        parser = self._make(f"{self.codec}parse", "parser")
        decoder = self._make(f"avdec_{self.codec}", "decoder")
        convert = self._make("videoconvert", "convert")
        scale = self._make("videoscale", "scale")
        capsfilter = self._make("capsfilter", "rgb-caps")
        sink = self._make("appsink", "sink")
        self._configure_source(source, GstRtsp)
        scale.set_property("add-borders", True)
        capsfilter.set_property("caps", Gst.Caps.from_string(
            f"video/x-raw,format=RGB,width={self.size},height={self.size},"
            "pixel-aspect-ratio=1/1"))
        self._configure_sink(sink)
        chain = (source, depay, parser, decoder, convert, scale, capsfilter, sink)
        for element in chain:
            pipeline.add(element)
        previous = depay
        for element in (parser, decoder, convert, scale, capsfilter, sink):
            if not previous.link(element):
                raise RuntimeError(f"failed to link {previous.get_name()} -> {element.get_name()}")
            previous = element
        expected = self.codec.upper()

        def on_pad_added(_source, pad):
            caps = pad.get_current_caps() or pad.query_caps(None)
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            if (structure and structure.get_string("media") == "video"
                    and structure.get_string("encoding-name") == expected):
                pad.link(depay.get_static_pad("sink"))

        source.connect("pad-added", on_pad_added)
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer software pipeline refused PLAYING state")
        self.pipeline, self.sink, self.bus = pipeline, sink, pipeline.get_bus()

    def read(self):
        sample = self._pull()
        if sample is None:
            return None
        sample_caps = sample.get_caps()
        caps = sample_caps.get_structure(0)
        width, height = int(caps.get_value("width")), int(caps.get_value("height"))
        buffer = sample.get_buffer()
        ok, mapped = buffer.map(self.Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("failed to map GStreamer appsink buffer")
        try:
            video_info = self.GstVideo.VideoInfo.new_from_caps(sample_caps)
            stride = int(video_info.stride[0])
            borrowed = np.ndarray((height, width, 3), dtype=np.uint8,
                                  buffer=mapped.data, strides=(stride, 3, 1))
            frame = np.ascontiguousarray(borrowed)
        finally:
            buffer.unmap(mapped)
        return frame


class OpenCVFFmpeg:
    """cv2.VideoCapture source for images without GStreamer in OpenCV.

    Two properties of `cv2.VideoCapture` shape this class:

    1. `read()` is not thread-safe. On the FFmpeg backend two concurrent reads
       trip `Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:175`
       and abort the whole process -- not an exception, an abort. A single
       background thread therefore owns the capture object; `read()` only ever
       reads a frame the thread has already published.
    2. FFmpeg keeps an internal packet queue, so a consumer slower than the
       stream falls further behind forever. The background thread drains at
       source rate and keeps only the newest frame, so `read()` always returns
       current video and the queue never grows.

    Reconnects are handled inside the thread: a dead RTSP source makes `read()`
    return None (which the caller already treats as "retry later"), never an
    exception, so a network blip cannot turn into a container restart loop.
    """

    backend_name = "cv2_ffmpeg"

    def __init__(self, url, size=640, transport="tcp", read_timeout_ms=5000,
                 reopen_delay_ms=1000, read_failure_limit=10, **_):
        self.url = url
        self.size = int(size)
        self.transport = str(transport).lower()
        self.read_timeout = max(0.05, int(read_timeout_ms) / 1000.0)
        self.reopen_delay = max(0.05, int(reopen_delay_ms) / 1000.0)
        self.read_failure_limit = max(1, int(read_failure_limit))
        self._cv2 = None
        self._cap = None
        self._thread = None
        self._running = False
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._consumed = 0

    # -- lifecycle --------------------------------------------------------
    def start(self):
        import cv2  # lazy: the RK image does not carry OpenCV at all
        self._cv2 = cv2
        if self.url.startswith("rtsp://") and self.transport in ("tcp", "udp"):
            # Read by the FFmpeg backend when the capture is opened, so it has
            # to be in the environment before VideoCapture() is constructed.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.transport}")
        self._running = True
        self._thread = threading.Thread(target=self._reader,
                                        name=f"cv2-reader-{id(self):x}",
                                        daemon=True)
        self._thread.start()

    def close(self):
        self._running = False
        with self._cond:
            self._cond.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=3.0)
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    # -- background reader ------------------------------------------------
    def _open(self):
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _reader(self):
        failures = 0
        while self._running:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    time.sleep(self.reopen_delay)
                    continue
                failures = 0
                print(f"[video] cv2_ffmpeg opened {self.url}", flush=True)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= self.read_failure_limit:
                    print(f"[video] cv2_ffmpeg reopening after {failures} "
                          f"failed reads", flush=True)
                    self._cap.release()
                    self._cap = None
                    time.sleep(self.reopen_delay)
                continue
            failures = 0
            letterboxed = self._letterbox(frame)
            with self._cond:
                self._frame = letterboxed
                self._seq += 1
                self._cond.notify_all()
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()

    def _letterbox(self, bgr):
        """BGR frame of any size -> owned RGB letterbox canvas of self.size."""
        cv2 = self._cv2
        height, width = bgr.shape[:2]
        scaled_w, scaled_h, _, _ = aspect_fit_geometry(width, height, self.size)
        interpolation = cv2.INTER_AREA if scaled_w < width else cv2.INTER_LINEAR
        scaled = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=interpolation)
        return pad_scaled_rgb(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB), self.size)

    # -- consumer ---------------------------------------------------------
    def read(self):
        if self._thread is None:
            self.start()
        deadline = time.monotonic() + self.read_timeout
        with self._cond:
            while self._seq == self._consumed:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._running:
                    return None
                self._cond.wait(remaining)
            self._consumed = self._seq
            return self._frame


class FallbackVideoSource:
    def __init__(self, primary, fallback=None, strict=False, failure_limit=3):
        self.primary = primary
        self.fallback = fallback
        self.strict = strict
        self.failure_limit = max(1, int(failure_limit))
        self.active = primary
        self.failures = 0

    @property
    def active_backend(self):
        return self.active.backend_name

    def _switch(self, exc):
        if self.strict or self.fallback is None or self.active is self.fallback:
            if exc:
                raise exc
            return
        self.active.close()
        self.active = self.fallback
        self.failures = 0
        print(f"[video] primary failed, switching to {self.active_backend}: {exc}", flush=True)

    def read(self):
        try:
            frame = self.active.read()
        except Exception as exc:
            self._switch(exc)
            return None
        if frame is not None:
            self.failures = 0
            return frame
        self.failures += 1
        if self.failures >= self.failure_limit:
            self._switch(RuntimeError(f"{self.active_backend} returned no frame {self.failures} times"))
        return None

    def close(self):
        self.active.close()
        if self.fallback is not None and self.fallback is not self.active:
            self.fallback.close()




_SOURCES = {}


def register_source(name, cls):
    """Backends register their hardware decoder implementation by name."""
    _SOURCES[name] = cls


def available_sources():
    return sorted(_SOURCES)


def create_video_source(stream, config, size):
    video = dict(config.get("video", {}))
    video.update(stream.get("video", {}))
    backend = str(video.get("backend", "gstreamer_mpp"))
    strict = bool(video.get("strict", False))
    fallback_name = str(video.get("fallback", "gstreamer_soft"))
    if backend not in _SOURCES:
        raise ValueError(f"unsupported video backend: {backend} "
                         f"(registered: {available_sources()})")
    if fallback_name not in ("none", "gstreamer_soft"):
        raise ValueError(f"unsupported video fallback: {fallback_name}")
    kwargs = dict(url=stream["rtsp_url"], size=size,
                  transport=stream.get("transport", "tcp"), **video)
    for key in ("backend", "strict", "fallback", "failure_limit"):
        kwargs.pop(key, None)
    primary = _SOURCES[backend](**kwargs)
    fallback = None
    if backend != "gstreamer_soft" and fallback_name == "gstreamer_soft":
        fallback = GStreamerSoftware(**kwargs)
    return FallbackVideoSource(primary, fallback, strict,
                               int(video.get("failure_limit", 3)))


def validate_backend_config(config):
    """Constructors are intentionally not started during config validation."""
    dummy = {"id": "validate", "rtsp_url": "rtsp://127.0.0.1/validate"}
    create_video_source(dummy, config, int(config.get("input_size", 640))).close()


register_source("gstreamer_soft", GStreamerSoftware)
register_source("cv2_ffmpeg", OpenCVFFmpeg)
