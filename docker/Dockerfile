# Shared base + per-backend stage. The `core` stage installs retail_core and is
# identical for every platform; only the backend stage adds runtime bindings.
# ABI-locked host libraries (librknnrt.so, librockchip_mpp, librga, rknnlite)
# are bind-mounted at runtime, never baked in: the user-space library has to
# match the kernel driver on the board it runs on.
ARG RUNTIME_BASE=docker.m.daocloud.io/library/python:3.11-slim-bookworm
ARG MIRROR_SED='s|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g; s|http://security.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g'

# Supplies the GstVideo/GstRtsp typelibs and a handful of base plugins.
# Installing gir1.2-gst-plugins-base-1.0 in the runtime stage would pull
# libgstreamer-gl1.0-0 -> libgl1 -> mesa -> libllvm15: ~150 MB of OpenGL stack
# for an app that only decodes video into an RGB array.
FROM ${RUNTIME_BASE} AS gstgir
ARG MIRROR_SED
RUN sed -i -e "${MIRROR_SED}" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
       gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-base \
    && rm -rf /var/lib/apt/lists/*
# The plugin split differs between GStreamer releases, so take whichever of
# these names exists.
RUN mkdir -p /stage/plugins && cd /usr/lib/aarch64-linux-gnu/gstreamer-1.0 \
    && for so in libgstapp.so libgsttypefindfunctions.so libgstvideoconvertscale.so \
                 libgstvideoconvert.so libgstvideoscale.so libgstvideorate.so; do \
         if [ -f "$so" ]; then cp "$so" /stage/plugins/; fi; \
       done && ls -1 /stage/plugins

# ---------------------------------------------------------------- core stage
# Shared by every backend: GStreamer RTSP I/O, NumPy, and the retail_core
# package. Nothing here knows what an NPU is.
FROM ${RUNTIME_BASE} AS core
ARG MIRROR_SED
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
RUN sed -i -e "${MIRROR_SED}" /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
       python3-gi gir1.2-gstreamer-1.0 \
       libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 \
       gstreamer1.0-plugins-good gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*
COPY --from=gstgir /usr/lib/aarch64-linux-gnu/girepository-1.0/GstVideo-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstRtsp-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstAudio-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstPbutils-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstTag-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/GstSdp-1.0.typelib \
                   /usr/lib/aarch64-linux-gnu/girepository-1.0/
COPY --from=gstgir /stage/plugins/ /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
# NumPy is the only Python dependency. The fall-detection image carried
# opencv-python-headless for its RTSP fallback and letterbox resize; both are
# done with GStreamer and NumPy here, keeping ~350 MB out of the image. The RK
# boards this targets have under 2 GB of free disk.
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" numpy==1.26.4
WORKDIR /opt/retail-vision
COPY core-py/retail_core ./retail_core
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/retail-vision:/usr/lib/python3/dist-packages \
    GST_REGISTRY=/tmp/gst-registry.bin

# -------------------------------------------------------------- rknn backend
FROM core AS rknn
COPY backends/__init__.py ./backends/__init__.py
COPY backends/rknn ./backends/rknn
ENTRYPOINT ["python3", "-m", "retail_core.app"]
CMD ["--config", "/config/config.json"]
