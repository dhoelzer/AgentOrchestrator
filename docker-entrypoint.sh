#!/bin/sh
# Seed the mounted data dir on first run, wait for the Docker sidecar, exec CMD.
set -eu

CONFIG="${HARNESS_CONFIG:-/data/config.json}"
DATA="${HARNESS_DATA_DIR:-/data}"

mkdir -p "$DATA/projects"

# First run: drop the image's default config into the mount so the stack works
# out of the box and the host copy becomes the editable source of truth.
# chmod 666 so the (root-owned) file stays editable by a non-root host user.
if [ ! -f "$CONFIG" ] && [ -f /app/config.json ]; then
    cp /app/config.json "$CONFIG"
    chmod 666 "$CONFIG" 2>/dev/null || true
    echo "seeded $CONFIG from image default"
fi

# Best-effort wait so an agent's first `docker` call does not race the daemon.
# Never fatal: the harness runs fine without Docker for non-container tasks.
if [ "${WAIT_FOR_DOCKER:-1}" = "1" ]; then
    i=0
    until docker info >/dev/null 2>&1; do
        i=$((i + 1))
        [ "$i" -ge "${DOCKER_WAIT_SECONDS:-60}" ] && break
        sleep 1
    done
    if docker info >/dev/null 2>&1; then
        echo "docker reachable at ${DOCKER_HOST:-default socket}"
    else
        echo "warning: docker unreachable at ${DOCKER_HOST:-default socket}; \
container-related sub-agent tasks will fail"
    fi
fi

exec "$@"
