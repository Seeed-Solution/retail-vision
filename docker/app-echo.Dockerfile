# syntax=docker/dockerfile:1
# BASE-1 M1.14: minimal echo app image on top of a vision-base image.
# Build:
#   docker build -f docker/app-echo.Dockerfile --build-arg BASE=vb-base-cpu:dev -t vb-echo:dev .
#
# The app is the built-in zero-code EchoApp (vision_base.hooks:EchoApp), so
# the image only adds the example configs; the entrypoint is the Python
# orchestrator, vb-runtime stays available at /opt/vb/bin/vb-runtime.

ARG BASE=vb-base-cpu:dev
FROM ${BASE}
COPY examples/ /app/examples/

WORKDIR /app
ENV PYTHONPATH=/opt/vb/py

ENTRYPOINT ["python3", "-m", "vision_base.main"]
