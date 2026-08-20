"""Rockchip MPP hardware decode source.

Backend-specific: mppvideodec plus librga do the decode and the aspect-fit
scale on the VPU/2D engine. librockchip_mpp / librga are bind-mounted from the
host, never baked into the image.
"""
from __future__ import annotations

import numpy as np

from retail_core.video import (_GstBase, aspect_fit_geometry,
                               copy_strided_rgb_to_letterbox, pad_scaled_rgb,
                               register_source, resize_nearest)


class GStreamerMPP(_GstBase):
    """RTSP -> Rockchip MPP decoder -> RGB appsink (hardware decode + RGA scale)."""

    backend_name = "gstreamer_mpp"

    def start(self):
        Gst, GstRtsp = self._import_gi()
        pipeline = Gst.Pipeline.new("retail-rtsp-mpp")
        source = self._make("rtspsrc", "source")
        depay = self._make(f"rtp{self.codec}depay", "depay")
        parser = self._make(f"{self.codec}parse", "parser")
        decoder = self._make("mppvideodec", "decoder")
        capsfilter = self._make("capsfilter", "rgb-caps")
        sink = self._make("appsink", "sink")
        self._configure_source(source, GstRtsp)
        decoder.set_property("fast-mode", True)
        # The parser's first CAPS event supplies coded source dimensions; the
        # probe below sets an aspect-fitted MPP/RGA output size before decoder
        # negotiation. Zero here preserves the source when CAPS lacks
        # dimensions, which the read-side fallback handles.
        decoder.set_property("width", 0)
        decoder.set_property("height", 0)
        decoder.set_property("format", 15)  # GstMppVideoDecFormat RGB
        capsfilter.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB"))
        self._configure_sink(sink)
        for element in (source, depay, parser, decoder, capsfilter, sink):
            pipeline.add(element)
        if not (depay.link(parser) and parser.link(decoder)
                and decoder.link(capsfilter) and capsfilter.link(sink)):
            raise RuntimeError("failed to link MPP GStreamer pipeline")

        def on_parser_event(_pad, probe_info):
            event = probe_info.get_event()
            if event is None or event.type != Gst.EventType.CAPS:
                return Gst.PadProbeReturn.OK
            caps = event.parse_caps()
            structure = caps.get_structure(0) if caps and caps.get_size() else None
            if structure is None:
                return Gst.PadProbeReturn.OK
            ok_w, source_w = structure.get_int("width")
            ok_h, source_h = structure.get_int("height")
            if ok_w and ok_h and source_w > 0 and source_h > 0:
                scaled_w, scaled_h, _, _ = aspect_fit_geometry(source_w, source_h, self.size)
                decoder.set_property("width", scaled_w)
                decoder.set_property("height", scaled_h)
            return Gst.PadProbeReturn.OK

        parser.get_static_pad("src").add_probe(Gst.PadProbeType.EVENT_DOWNSTREAM,
                                               on_parser_event)
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
            raise RuntimeError("GStreamer MPP pipeline refused PLAYING state")
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
            # Respect the negotiated row stride: MPP/RGA often aligns RGB rows,
            # so treating the buffer as tightly packed shears pixels.
            video_info = self.GstVideo.VideoInfo.new_from_caps(sample_caps)
            stride = int(video_info.stride[0])
            if stride < width * 3 or len(mapped.data) < stride * height:
                raise RuntimeError("invalid RGB stride from mppvideodec")
            if width <= self.size and height <= self.size:
                # One required copy, straight from borrowed Gst memory into the
                # owned canvas. No mapped view escapes buffer.unmap().
                frame = copy_strided_rgb_to_letterbox(mapped.data, width, height,
                                                      stride, self.size)
            else:
                borrowed = np.ndarray((height, width, 3), dtype=np.uint8,
                                      buffer=mapped.data, strides=(stride, 3, 1))
                sw, sh, _, _ = aspect_fit_geometry(width, height, self.size)
                frame = pad_scaled_rgb(resize_nearest(borrowed, sw, sh), self.size)
        finally:
            buffer.unmap(mapped)
        return frame


register_source("gstreamer_mpp", GStreamerMPP)
