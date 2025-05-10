# This image is published to ghcr.io/katherlab/toolarena-runtime

# The base image is either ubuntu:24.04 (for CPU) or nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 (for GPU)
ARG BASE=ubuntu:24.04
FROM ${BASE}

# Install common commands (loosely according to buildpack-deps:24.04)
RUN apt-get update && \
    apt-get install -y \
      ca-certificates \
      curl \
      gnupg \
      netbase \
      sq \
      wget \
      tzdata \
      git \
      make \
      unzip && \
    rm -rf /var/lib/apt/lists/*

# Install python
ARG PYTHON_VERSION=3.12
RUN apt-get update && \
    apt-get install -y python${PYTHON_VERSION} python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip config --global set global.break-system-packages true

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6 /uv /uvx /bin/

WORKDIR /toolarena

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-install-project

COPY toolarena ./toolarena

WORKDIR /workspace

COPY scripts/subprocess_utils.py /toolarena/subprocess_utils.py
COPY scripts/function_runner.py /toolarena/function_runner.py

# For backwards compatibility with the old toolmaker runtime
RUN mkdir -p /toolmaker && ln -s /toolarena/subprocess_utils.py /toolmaker/subprocess_utils.py

VOLUME /mount/input
VOLUME /mount/output

ENV WORKSPACE_DIR=/workspace
ENV HOST=0.0.0.0
ENV PORT=8000

CMD /toolarena/.venv/bin/uvicorn --app-dir /toolarena --host ${HOST} --port ${PORT} toolarena.server:app