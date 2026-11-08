# Deployment and Operations

## Running locally

```bash
pip install -r requirements.txt
python -m agent_harness.main
```

```
Orchestrator harness v1.10.0+6cf4127
LLM router: http://127.0.0.1:41207 (endpoints: primary@http://192.168.2.107:8080/v1/)
LLM: model=openai/qwen base_url=http://192.168.2.107:8080/v1/
Sub-agent turn limit: 100 | tick: 60s
Data dir: /app/data
Web UI: http://0.0.0.0:8090
```

The router URL is printed at startup — with `router_port: 0` the OS picks the port, so read it from the log rather than assuming.

### CLI flags

```
--config PATH      config.json to load (also $HARNESS_CONFIG)
--host / --port    web bind
--tick SECONDS     cycle interval
--model NAME       OpenAI-style model name
--base-url URL     OpenAI-compatible base URL
--api-key KEY      credential for that endpoint
--once             run a single cycle and exit
```

`--once` is the fastest way to check a config change: it runs one cycle against the current board and exits.

```bash
python -m agent_harness.main --once --base-url http://ollama:11434/v1 --model qwen3:30b
```

### Startup order

`main.py` starts things in a specific order, and it matters:

1. `register_harness_tools()` — tools must exist before any agent is built.
2. `Store(data_dir)` — load board state.
3. **`router.start()`** — before any agent exists, because every LLM request from every agent and every condenser is addressed to it.
4. `AgentRuntime(config, store, router=router)`
5. `Orchestrator(...)`
6. Orchestrator loop on a daemon thread.
7. Uvicorn serves the web app.

On `SIGINT`/`SIGTERM`: `orchestrator.stop()` → uvicorn exits → loop joined (10s) → `router.stop()` → `store.flush()`.

## Dependencies

```
openhands-sdk>=1.58.0
openhands-tools>=1.58.0
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.7.0
httpx>=0.27.0
python-multipart>=0.0.9
```

## Docker

```bash
docker compose up -d --build
open http://localhost:8090
```

Two services:

| Service | Purpose |
|---|---|
| `harness` | Orchestrator + web UI. Runs forever. `restart: unless-stopped`. |
| `docker` | Docker-in-Docker daemon, so sub-agents get a real daemon: they can build images, run containers, use compose. |

> **Why `privileged: true`.** Agents are general-purpose software workers: they install packages, compile, bind ports, use containers, and touch arbitrary paths. There is no meaningful sandbox to keep them out of. `privileged` implies all capabilities, all host devices, unconfined seccomp/apparmor, and writable `/proc` + `/sys`. `shm_size: 1gb` covers browsers/test runners/build tooling; `nofile` is raised well past the default.

### The bind mount

Everything is in `./harness` on the host, mounted at `/data`:

```
./harness/config.json        LLM + runtime config (the editable source of truth)
./harness/state.json         board: projects, tasks, agent hierarchy
./harness/events.jsonl       activity feed
./harness/conversations/     SDK-persisted conversation state per agent
./harness/projects/<name>/   each project workspace = agent output
```

The entrypoint seeds `config.json` into the mount on first run from the image default, and `chmod 666`s it so the root-owned file stays editable by a non-root host user.

> **Run exactly one stack per `./harness` directory.** The store is in-memory with a JSON flush; two containers sharing the mount clobber each other.

> **No `VOLUME` is declared in the Dockerfile on purpose.** An anonymous volume would silently hide state when the bind mount is forgotten.

The DinD service mirrors `./harness:/data` at the **same path**, because the daemon resolves bind sources against its own filesystem — an agent that runs `docker run -v /data/projects/foo:/app` gets the same host directory. Keep that path in sync if you change the app mount.

### Container env

```yaml
HARNESS_CONFIG: /data/config.json
HARNESS_DATA_DIR: /data
HARNESS_HOST: 0.0.0.0
HARNESS_PORT: 8090
DOCKER_HOST: tcp://docker:2375      # agents' docker commands go to the sidecar
DOCKER_DRIVER: overlay2
DOCKER_TLS_VERIFY: "0"              # plaintext on the private compose network
HARNESS_LLM_BASE_URL: ${HARNESS_LLM_BASE_URL:-}
HARNESS_LLM_MODEL: ${HARNESS_LLM_MODEL:-}
HARNESS_LLM_API_KEY: ${HARNESS_LLM_API_KEY:-}
HARNESS_LLM_SERVERS: ${HARNESS_LLM_SERVERS:-}
HARNESS_LLM_QUEUE_DEPTH: ${HARNESS_LLM_QUEUE_DEPTH:-}
```

Empty means "use whatever `/data/config.json` says". To reach an endpoint on the host, use `http://host.docker.internal:PORT` (the `host-gateway` extra host is configured).

```bash
HARNESS_LLM_SERVERS='[{"name":"ollama-a","provider":"ollama_chat","base_url":"http://host.docker.internal:11434/v1","model":"qwen2.5:7b"}]' \
  docker compose up -d --build
```

`stop_grace_period: 60s` lets in-flight agent runs finish and the store flush on `SIGTERM`.

The image `HEALTHCHECK` curls `/health`.

## Configuration and restarts

Config is re-read at the start of **every cycle**, so most changes need no restart:

```bash
curl -X POST localhost:8090/api/config \
  -H 'Content-Type: application/json' \
  -d '{"max_parallel_workers": 6, "idle_tick_seconds": 120, "persist": true}'
```

Editing `./harness/config.json` by hand is also picked up on the next tick — the loop fingerprints the file to notice external edits.

`docker compose restart` is only needed for things the entrypoint reads, or after changing `compose.yaml`.

### Config file ownership

A real failure mode: if `./harness/config.json` is owned by a uid the container cannot write, `POST /api/config` returns `500 Could not write config: …` while the **in-memory change still applies** — and is lost on restart. Fix ownership on the host, or run the container as a uid that owns the mount.

## Observability

### `/health`

```json
{ "ok": true, "harness_version": "1.10.0+6cf4127",
  "model": "openai/qwen", "base_url": "http://192.168.2.107:8080/v1/",
  "tick_seconds": 60, "sub_agent_turn_limit": 100,
  "last_tick": 1762500000, "inflight": ["agent_abc123"] }
```

### `/api/state`

The whole board plus loop metadata: `cycle_status`, `cycle_duration`, `effective_tick`, `cycle_skips`, `cycle_stuck_seconds`, `cycle_skip_reason`, `cycle_error`, `cycle_count`.

`last_tick` is written **only on a successful cycle**; `last_cycle_attempt_at` on every attempt. A frozen `last_tick` with a moving `last_cycle_attempt_at` means cycles are running and failing — check `cycle_error`.

### `/api/llm/router`

Per-endpoint health, in-flight, cooldown, measured `tokens_per_second`, and live request cards. See [LLM Router](LLM-Router).

### `events.jsonl`

Append-only activity feed: `loop_started`, `loop_error`, `task_unblocked`, `delivery_candidate`, `agent_dispatched`, `config_updated`, and more.

## Tests

Standalone scripts, no test framework required:

```bash
python tests/test_board_mechanics.py
python tests/test_runtime_and_config.py
python tests/test_dashboard_features.py
python tests/test_blocks_cost_version.py
python tests/test_queue_input_recovery.py
python tests/test_llm_failures_and_reentrancy.py
python tests/test_scheduler_and_stuck_cycles.py
python tests/test_multi_llm_config_reload.py
python tests/test_llm_router.py
python tests/test_uploads_and_pause.py
python tests/test_concurrency_and_tick.py
```

LLM calls are stubbed, so the suite runs without a live endpoint.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `model 'ollama_chat/qwen3:30b' not found` | A provider prefix reached the wire. `model` must be the literal name Ollama knows. |
| `400 think value "high" is not supported` | `reasoning_effort` not dropped. Check `drop_body_keys` / set `reasoning_effort` deliberately. |
| `LLM Provider NOT provided` | An unprefixed model on the **direct-SDK** path. The wire value stays verbatim; `route()` adds the prefix. |
| Every run reports `$0.00` | litellm cannot price a custom model. Set `input_cost_per_token` / `output_cost_per_token`. |
| Requests queue forever, one endpoint stuck | A half-open socket holding a slot. Check `router_stall_timeout` and `router_max_call_timeout`. |
| Board frozen in `backlog` | Cycles are being skipped. Check `cycle_skip_reason`; the deterministic scheduler still starts explicitly-staffed backlog cards. |
| A card stuck in `review` forever | The reviewer had no `record_review` in its tool set. |
| Orchestrator complains it is "running out of turns" | A display regression — the orchestrator budget is per-cycle, not lifetime. See [Orchestrator](Orchestrator). |
| UI shows an old build | The bind mount means a container that was not rebuilt serves an old build against new data. Compare `harness_version` in `/health` against the expected version. |
| Config change lost on restart | The file is not writable by the container. See ownership above. |

## Security notes

- The router binds to loopback (`router_host: 127.0.0.1`) and is not exposed outside the container.
- `GET /api/config` redacts every API key to `***`.
- `resolved_api_key(use_env=False)` prevents the harness-wide key being sent to an unsaved endpoint during a UI test prompt.
- `confirm_tools: false` because agents run unattended; `sandbox_workspaces: true` confines worker file access to the project workspace.
- The container is `privileged` by design — treat it as a root-equivalent host, not a sandbox.
