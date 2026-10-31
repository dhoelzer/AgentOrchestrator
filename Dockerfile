# Orchestrator harness: web UI + forever-running orchestrator loop.
#
#   docker build -t orchestrator-harness .
#
# Runtime state (config, board state, persisted conversations, project
# workspaces) is expected to live in the mounted volume at /data, not in the
# image. See compose.yaml.

# ---------------------------------------------------------------------------
# Stage 1: modern Docker CLI + compose plugin, copied into the final image so
# agents can drive the Docker-in-Docker sidecar (or a mounted daemon socket).
# ---------------------------------------------------------------------------
FROM docker:cli AS docker-cli

# ---------------------------------------------------------------------------
# Stage 2: application image
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENHANDS_SUPPRESS_BANNER=1

# tmux: the SDK terminal tool prefers a tmux backend and silently degrades to
# subprocess without it. git/curl/jq/ripgrep: agents routinely need them.
# build-essential: lets agents compile native deps or C deliverables.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux \
        git \
        curl \
        wget \
        ca-certificates \
        jq \
        ripgrep \
        procps \
        iproute2 \
        dnsutils \
        openssh-client \
        build-essential \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI + buildx/compose plugins (client only; the daemon is separate).
# Plugin dir is /usr/local/libexec/docker/cli-plugins in the docker:cli image.
COPY --from=docker-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker-cli /usr/local/libexec/docker/cli-plugins /usr/local/libexec/docker/cli-plugins

WORKDIR /app

# Dependency layer cached independently of source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_harness ./agent_harness
COPY tests ./tests

# Baked-in defaults; the mounted volume at /data is the source of truth at
# runtime, and the entrypoint seeds it on first run.
COPY config.json ./config.json
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Everything the orchestrator and its sub-agents read or write lives here.
# No VOLUME declared: an anonymous volume would silently hide state when the
# bind mount is forgotten. compose.yaml always binds ./harness to /data.
ENV HARNESS_CONFIG=/data/config.json \
    HARNESS_DATA_DIR=/data \
    HARNESS_HOST=0.0.0.0 \
    HARNESS_PORT=8090

EXPOSE 8090

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${HARNESS_PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "-m", "agent_harness.main"]
