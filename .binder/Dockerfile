# Binder Dockerfile - uses pre-built base image from GHCR
# The base image contains pixi + full runtime environment + underworld3

FROM ghcr.io/underworldcode/uw3-base:2025.01

USER root

# Copy start script (path relative to repo root, not .binder/)
COPY .binder/start /usr/local/bin/start
RUN chmod +x /usr/local/bin/start

USER jovyan
WORKDIR /home/jovyan

# Create workspace with tutorials
RUN mkdir -p Tutorials && \
    cp -r underworld3/docs/beginner/tutorials/* ./Tutorials/ 2>/dev/null || true

# Binder environment variables
ENV NB_USER=jovyan
ENV NB_UID=1000
ENV HOME=/home/jovyan

# Binder runs this before jupyter
ENTRYPOINT ["/usr/local/bin/start"]
