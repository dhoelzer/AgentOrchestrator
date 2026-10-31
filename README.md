# Orchestrator Harness

A long-running agent harness built on the [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk).

An **orchestrator** agent owns a portfolio of projects. Every minute it looks at the
board, creates **specialist sub-agents** for the expertise each task needs, dispatches
them, coordinates reviews, decides what to do when an agent runs out of its turn
budget, and signals a project for delivery. It runs forever. A web UI shows project
creation plus a kanban view of every project, including the hierarchy of agents and
sub-agents and each one's status.

```
python -m agent_harness.main            # web UI + forever loop
open http://localhost:8090              # create projects, watch the board

docker compose up -d --build            # containerized, with Docker-in-Docker
```

---

## Requirements → how they are met

| Requirement | Where |
|---|---|
| LLM swappable via base URL + model name, OpenAI-style API, preconfigured for Qwen | `agent_harness/config.py`, `config.json` |
| Automatic creation of specialist sub-agents | `create_sub_agent` tool → `agent_harness/tools.py`, prompt template in `agent_harness/prompts.py` |
| Orchestrator system prompt for long-running projects; decomposes work and names required expertise | `ORCHESTRATOR_SYSTEM_PROMPT` in `agent_harness/prompts.py` |
| Web interface to create projects | `agent_harness/web/templates/index.html` |
| Kanban view incl. agent/sub-agent hierarchy and per-agent status | `agent_harness/web/api.py` + `agent_harness/web/static/app.js` |
| Every-minute check: staff waiting work, coordinate reviews, detect deliverables, update the board; runs forever | `Orchestrator.run_forever()` / `run_cycle()` in `agent_harness/orchestrator.py` |
| Sub-agents capped at 100 turns; on exhaustion they report back and the orchestrator decides whether to re-run, reconfigure, or split | `agent_harness/runtime.py` (enforcement) + `extend_agent` / `reconfigure_agent` / `retire_agent` in `agent_harness/tools.py` |
| Deployable container with full agent privileges, Docker-in-Docker, and a `./harness` mount for config + workspace output | `Dockerfile`, `docker-entrypoint.sh`, `compose.yaml` |

---

## Configuration

The LLM is changed **without touching code** — only `base_url` and `model`:

```json
{
  "llm": {
    "base_url": "http://192.168.2.107:8080/v1/",
    "model": "openai/qwen",
    "api_key": "local-no-key"
  },
  "tick_seconds": 60,
  "sub_agent_turn_limit": 100
}
```

Environment overrides (take precedence over `config.json`):

```bash
HARNESS_LLM_BASE_URL=https://api.openai.com/v1 \
HARNESS_LLM_MODEL=openai/gpt-4o \
HARNESS_LLM_API_KEY=sk-... \
python -m agent_harness.main
```

CLI overrides: `--base-url`, `--model`, `--api-key`, `--tick`, `--port`, `--config`.

| Setting | Default | Meaning |
|---|---|---|
| `tick_seconds` | 60 | How often the orchestrator wakes up |
| `sub_agent_turn_limit` | 100 | Hard turn budget per sub-agent |
| `orchestrator_turn_limit` | 60 | Turn budget granted per orchestrator cycle |
| `max_parallel_workers` | 3 | Sub-agents that may run concurrently |
| `max_sub_agent_depth` | 3 | How deep sub-agents may nest |
| `max_children_per_agent` | 6 | Fan-out limit per agent |
| `review_attempts_before_escalation` | 3 | Failed reviews before escalation |
| `sandbox_workspaces` | true | Confine sub-agent file access to its project workspace |
| `llm.request_timeout` | 300 | Starting per-request timeout (seconds) |
| `llm.adaptive_timeout` | true | Grow the timeout after slow/empty replies |
| `llm.min_request_timeout` | 60 | Floor for the adaptive timeout |
| `llm.max_request_timeout` | 3600 | Ceiling for the adaptive timeout |
| `llm.timeout_growth` | 2.0 | Geometric growth factor per widening |
| `llm.empty_response_retries` | 2 | Retries when the endpoint returns nothing |
| `llm.input_cost_per_token` | 0.000002 | USD/token billed for prompts |
| `llm.output_cost_per_token` | 0.000015 | USD/token billed for completions |

### Cost accounting

litellm cannot price an unknown custom model, so without explicit rates every
agent reports `$0.00`. The defaults price the preconfigured Qwen endpoint at
$2/M input and $15/M output and are passed straight to `LLM(input_cost_per_token=…,
output_cost_per_token=…)`. Override per endpoint with
`HARNESS_LLM_INPUT_COST_PER_TOKEN` / `HARNESS_LLM_OUTPUT_COST_PER_TOKEN`, or set
them in the Settings panel.

### Changing config while the harness runs

`GET /api/config` returns the live config (API key redacted) plus runtime LLM
health. `POST /api/config` patches an allowlisted subset and applies it
immediately — turn limits and LLM settings for new runs, timeout and
parallelism also for conversations already open:

```bash
curl -X POST localhost:8090/api/config \
  -H 'content-type: application/json' \
  -d '{"llm.base_url":"http://other:8080/v1","llm.request_timeout":900,"max_parallel_workers":5}'
```

Unknown keys and wrong types are rejected with `400`. Changes are written back
to the config file, so they survive a restart. The same controls are in the
web UI under **Settings**.

### Slow or flaky local endpoints

Local/self-hosted servers are slow and sometimes answer with an empty
completion. Rather than reporting that as an agent failure, the runtime:

1. retries the run (a retry nudges the agent to continue; it does not resend
   the assignment),
2. geometrically widens the shared request timeout, up to
   `llm.max_request_timeout`, and
3. if the endpoint still produces nothing, marks the agent `llm_unavailable`
   and blocks its task with a reason, so the next cycle re-dispatches instead
   of treating the work as failed.

Only genuinely transient faults (timeouts, rate limits, connection errors, 5xx)
are retried — an auth or bad-request error fails fast rather than burning the
agent's turn budget. The widened timeout is persisted, so a restart does not go
back to a value the endpoint already proved too small. `GET /api/state` exposes
the current timeout and retry counters under `meta.llm`.

---

## The loop

Each tick, per project:

1. **`reconcile()`** — deterministic transitions: a backlog task whose dependencies are
   all `done` moves to `ready`; a project whose tasks are all `done` becomes
   `ready_for_delivery`.
2. **`detect()`** — flags everything needing judgement: tasks waiting for review,
   budget-exhausted agents, rework, unstaffed ready tasks, in-progress tasks with no
   agent running, projects ready to deliver.
3. **`_run_project_cycle()`** — hands the orchestrator a JSON briefing of the board,
   the agent tree, and the attention list, then lets it act through its tools.

The loop never exits; a cycle that throws is logged and the next tick still runs.

## Kanban columns

`backlog → ready → in_progress → review → done`, with `rework` (review failed) and
`blocked` (needs a decision, or a budget ran out).

## Turn budget

A budget is enforced by capping `max_iteration_per_run` for a single run and counting
cumulative turns across runs. When the SDK stops a run with `MaxIterationsReached`,
the agent is marked `budget_exhausted`, its task moves to `blocked`, and the
orchestrator receives a briefing containing the agent's last report. It then chooses:

- `extend_agent` — same agent and config, another 100 turns (real progress, ran out of time)
- `reconfigure_agent` — new prompt/specialty/tools and a **fresh conversation** (the SDK
  keeps the persisted system prompt and tool schema when resuming, so config changes
  only take effect in a new conversation; prior progress carries over via `last_report`)
- `retire_agent` + split the task into smaller tasks

Every decision carries a `decision` string so it is auditable in the activity feed.

## Tools

**Orchestrator:** `create_project`, `list_projects`, `create_task`, `update_task`,
`list_tasks`, `create_sub_agent`, `assign_task`, `run_agent`, `resume_agent`,
`extend_agent`, `reconfigure_agent`, `retire_agent`, `list_agents`, `request_review`,
`record_review`, `signal_delivery`, `read_file`, `report_cycle`

**Sub-agents:** `terminal`, `file_editor`, `task_tracker`, `report_to_parent`,
`create_sub_agent`, `request_review`, `read_file` — plus `finish`/`think` built in by
the SDK.

`run_agent` dispatches in the background by default, so one cycle can staff several
specialists at once.

## Web UI

- `GET /` — project creation form, alerts, kanban board, agent hierarchy tree, activity feed
- `GET /api/state` — full board + agent tree + events + LLM health (polled every 5 s)
- `POST /api/projects` — create a project
- `POST /api/tick` — run a cycle immediately
- `GET /api/config`, `POST /api/config` — read / live-patch runtime settings
- `POST /api/projects/{id}/tasks/{id}/directives` — send the worker direction; a
  blocked task moves back to ready and the directive is injected into its next
  assignment prompt
- `GET /api/projects/{id}/tasks/{id}`, `GET /api/projects/{id}/agents/{id}` — detail
- `GET /health` — model, base URL, tick, in-flight agents

Task cards show live `turns_used/turn_limit` (updated during a run, not only at
the end) and the reason a task is blocked. The header shows the current LLM
request timeout and how many replies had to be retried.

## Deploying with Docker

```bash
docker compose up -d --build      # builds the image, starts the stack
open http://localhost:8090
docker compose logs -f harness    # orchestrator + agent activity
```

Two services:

| Service | Role |
|---|---|
| `harness` | The image built from `Dockerfile`: web UI + forever orchestrator loop. |
| `docker` | `docker:29-dind` — a real daemon the agents drive, so they can build images, run containers, and use compose. |

### Permissions

The container runs `privileged: true` with `seccomp=unconfined`, all host
devices, generous `ulimits`, and 1 GiB `/dev/shm`. That is deliberate: the
orchestrator and its sub-agents are general-purpose software workers that
install packages, compile code, bind ports, and run containers, so there is no
sandbox worth keeping them out of. `privileged` is also what lets the DinD side
mount overlayfs for the inner daemon.

**Treat this as root-equivalent on the host.** A privileged container can write
host block devices and load kernel modules, so anyone who can reach port 8090
can run arbitrary code as root on the host. Do not expose 8090 publicly; bind
it to a private network or put auth in front of it. Agents also see
`HARNESS_LLM_API_KEY` in their own shell environment.

### Docker-in-Docker

Agents get a real daemon, not a mounted socket: `DOCKER_HOST=tcp://docker:2375`
is set in the container environment, and the SDK terminal tool passes it
through to agent commands (verified against `build_terminal_env`). The daemon's
state lives in the `dind-data` volume, so image and build cache survive
restarts. Plaintext TCP is safe here only because the compose network is
private to this stack.

One DinD subtlety: the daemon runs in a different container, so it resolves
bind-mount sources against *its own* filesystem. The `docker` service therefore
mirrors the `./harness:/data` mount at the same path, so an agent that runs
`docker run -v /data/projects/foo:/app` gets the same host directory its
workspace lives in. If you change the app's mount path, change both.

To instead share the host daemon, replace the env var with a socket mount:

```yaml
    environment:
      DOCKER_HOST: unix:///var/run/docker.sock
    volumes:
      - ./harness:/data
      - /var/run/docker.sock:/var/run/docker.sock
```

That gives agents access to the *host* daemon and every container on it.

### The `./harness` mount

Everything mutable lives in `./harness`, which the container sees as `/data`:

```
harness/config.json       LLM + runtime config (the source of truth at runtime)
harness/state.json        board: projects, tasks, agent hierarchy
harness/events.jsonl      activity feed
harness/conversations/    SDK-persisted conversation state per agent
harness/projects/<name>/  each project workspace = agent output
```

On first start the entrypoint seeds `harness/config.json` from the image
default, so the stack runs with no setup. Edit that file and
`docker compose restart` to change the model, base URL, tick interval, or turn
limits. `HARNESS_LLM_BASE_URL` / `HARNESS_LLM_MODEL` / `HARNESS_LLM_API_KEY`
override it without an edit; use `http://host.docker.internal:<port>` to reach
an LLM endpoint running on the host.

Files the container writes into the mount are owned by root. If that matters,
`chown -R` the directory afterwards or set a matching `user:` on the service.

Run exactly one stack per `./harness` directory — the store is in-memory with a
JSON flush, so two containers sharing a mount clobber each other.

### Building without compose

```bash
docker build -t orchestrator-harness .
docker run --privileged --shm-size=1g -v "$PWD/harness:/data" \
  -v /var/run/docker.sock:/var/run/docker.sock -p 8090:8090 orchestrator-harness
```

Standalone runs have no DinD sidecar, so give agents the host socket as above
(or drop the socket and they simply cannot use Docker). Without a reachable
daemon the entrypoint logs a warning and continues; only container-related
sub-agent tasks fail. Set `WAIT_FOR_DOCKER=0` to skip the wait.

## Layout

```
agent_harness/
  config.py       LLM + runtime config (OpenAI-compatible endpoint)
  store.py        Durable projects / tasks / agents / event log
  prompts.py      Orchestrator, specialist, reviewer prompts + briefings
  runtime.py      Conversation lifecycle, turn-budget enforcement, resume
  tools.py        Custom tools: board control, delegation, review, delivery
  orchestrator.py Orchestrator agent + forever scheduler
  main.py         Entry point (web UI + loop)
  web/            FastAPI app, templates, static assets
Dockerfile        Image build (multi-stage: Docker CLI + compose plugin)
docker-entrypoint.sh  Seeds /data config, waits for the daemon, execs CMD
compose.yaml      harness + docker:29-dind, ./harness bind mount
tests/
  test_board_mechanics.py  Board logic with no LLM
  test_e2e_llm.py          Full loop against the configured LLM
  test_turn_budget.py      Budget exhaustion, extension, reconfiguration
```

The Python package is `agent_harness`, not `harness` — the top-level `harness/`
name is reserved for the bind-mounted data directory.

## Running the tests

```bash
pip install -r requirements.txt

# No LLM needed — board mechanics, delegation, review, delivery, persistence:
python tests/test_board_mechanics.py

# No LLM needed — turn-count sync, retry/adaptive-timeout, blocked-task resume,
# live config patching, token costs:
python tests/test_runtime_and_config.py

# Needs the configured LLM endpoint:
python tests/test_e2e_llm.py 3
python tests/test_turn_budget.py
```

## Notes

- **Single process.** The store is an in-memory model flushed to `data/state.json`;
  run one instance per `data_dir`.
- Sub-agents run unattended (`confirm_tools: false`). Give each a project workspace and
  keep `sandbox_workspaces: true`.
- `data/` is gitignored; it holds board state and the SDK's persisted conversations.
