# Configuration Reference

Configuration comes from three layers, applied in this order — each overrides the previous:

1. **Built-in defaults** in `agent_harness/config.py`
2. **`config.json`** — path from `--config`, else `$HARNESS_CONFIG`, else `<repo>/config.json`
3. **Environment variables** — `HARNESS_*`
4. **CLI flags** — `--model`, `--base-url`, `--api-key`, `--tick`, `--host`, `--port`

The harness re-reads `config.json` at the start of **every cycle**, so edits made through the Settings UI or by hand take effect without a restart.

```bash
# where the file lives
python -m agent_harness.main --config /data/config.json
HARNESS_CONFIG=/data/config.json python -m agent_harness.main
```

> **One stack per config/state directory.** The store is in-memory with a JSON flush; two containers sharing the same mount clobber each other.

---

## Swapping the LLM

The whole point: change a base URL and a model name.

```json
{
  "llm": {
    "base_url": "http://192.168.2.107:8080/v1/",
    "model": "openai/qwen",
    "api_key": "local-no-key"
  }
}
```

Or without touching the file:

```bash
export HARNESS_LLM_BASE_URL="http://192.168.2.107:8080/v1/"
export HARNESS_LLM_MODEL="openai/qwen"
export HARNESS_LLM_API_KEY="local-no-key"
python -m agent_harness.main
```

Every agent in the harness is built from an OpenAI-compatible endpoint. The preconfigured default is the local Qwen endpoint above.

> **`base_url` is used as-is.** The router appends `/chat/completions` to it exactly as the OpenAI client does, so it must already carry the `/v1` prefix. For Ollama, use `http://host:11434/v1` — the OpenAI-compatible surface, not `/api/chat`.

---

## `llm` — endpoint settings

`LLMConfig` extends `LLMServerConfig`, so it has every per-server field plus the global tuning knobs.

### Per-endpoint fields (also valid on each `llm_servers` entry)

| Field | Default | Meaning |
|---|---|---|
| `name` | `"primary"` | Identity used for routing keys and metrics labels. Must be unique across the fleet. |
| `provider` | `"openai"` | `openai` \| `ollama` \| `ollama_chat`. Only decides the litellm route prefix for the direct-SDK fallback. **Does not change the wire model.** |
| `base_url` | Qwen endpoint | Used verbatim; `/chat/completions` is appended to it. |
| `model` | `"openai/qwen"` | The **literal** value put in the request body's `model` key. Sent verbatim — never prefix it for the wire. |
| `api_key` | `"local-no-key"` | Credential for this endpoint. |
| `api_key_env` | `null` | Env var to read the key from; overrides `api_key`. Keeps secrets out of the file. |
| `max_input_tokens` | `131072` | Context window advertised to the SDK. |
| `max_output_tokens` | `8192` | Response cap. |
| `reasoning_effort` | `null` | Endpoint-specific value. Setting it also stops `reasoning_effort` being auto-dropped. |
| `extra_body` | `{}` | Keys merged into the body, e.g. Ollama's `{"num_ctx": 32768, "keep_alive": "10m"}`. |
| `drop_body_keys` | `null` | Body keys this endpoint must not receive. `null` = decide from provider. |
| `headers` | `{}` | Extra HTTP headers required by this endpoint, e.g. a gateway's `X-Project-Slug`. The router attaches them; agents never do. |
| `input_cost_per_token` | `2e-06` | USD per token. litellm cannot price an unknown custom model, so without this every run reports `$0.00`. |
| `output_cost_per_token` | `1.5e-05` | USD per token. |

### Global tuning knobs (on `llm` only)

| Field | Default | Meaning |
|---|---|---|
| `request_timeout` | `300` | Starting per-request deadline, seconds. |
| `num_retries` | `3` | SDK-level retries (the agent-side LLM uses `0` when a router exists — the router owns retries). |
| `drop_params` | `true` | Let litellm drop unsupported parameters instead of erroring. |
| `native_tool_calling` | `true` | Use the endpoint's tool-calling rather than a text protocol. |
| `adaptive_timeout` | `true` | Grow the timeout after slow/empty replies rather than failing the task. |
| `min_request_timeout` | `60` | Floor for the adaptive timeout. |
| `max_request_timeout` | `900` | Ceiling for the adaptive **widening**. Deliberately modest: widening is for "slow but alive"; a huge value turns a dead endpoint into an hour of dead time per request. |
| `timeout_growth` | `2.0` | Geometric growth factor per widening. |
| `empty_response_retries` | `2` | Retries when the endpoint returns a clean but empty completion. |
| `timeout_decay_after_runs` | `5` | Consecutive clean runs after which the widened timeout shrinks back toward `request_timeout`. Without this the widening is one-way and permanent, and it survives a restart via `state.json`. |

### `drop_body_keys` in detail

The SDK puts `reasoning_effort: "high"` in every request whether or not the endpoint implements it. Ollama's OpenAI layer maps it onto its own `think` option, which only understands `high`/`low` for models it marks as reasoning-capable, and answers:

```
400 think value "high" is not supported for this model
```

Default rule:

```python
if provider in ("ollama", "ollama_chat") and not reasoning_effort:
    return ["reasoning_effort"]
return []
```

Set `drop_body_keys: []` to force the key through, or list other keys to strip.

---

## `llm_servers` — a fleet of endpoints

```json
{
  "llm": { "base_url": "http://192.168.2.107:8080/v1/", "model": "openai/qwen" },
  "llm_servers": [
    { "name": "ollama-a", "provider": "ollama_chat",
      "base_url": "http://ollama-a:11434/v1", "model": "qwen2.5:7b",
      "extra_body": { "num_ctx": 32768, "keep_alive": "10m" } },
    { "name": "vllm-b", "provider": "openai",
      "base_url": "http://vllm-b:8000/v1", "model": "Qwen2.5-14B-Instruct" }
  ]
}
```

- `llm` is always first in rotation; `llm_servers` adds to it.
- Entries are de-duplicated by `(base_url, model)`.
- Names must be unique or config validation fails.

Full behaviour, including selection: [Multiple LLMs and Selection](Multiple-LLMs-and-Selection).

---

## Scheduling and the forever loop

| Field | Default | Meaning |
|---|---|---|
| `tick_seconds` | `60` | Nominal seconds between cycles. |
| `dynamic_tick` | `true` | Re-derive the interval from measured cycle duration. |
| `min_tick_seconds` | `10` | Floor for the auto-tuner. Never overrides an explicitly configured faster tick — the floor is `min(min_tick_seconds, tick_seconds)`. |
| `max_tick_seconds` | `600` | Ceiling. |
| `tick_duty_cycle` | `0.5` | Fraction of the interval a cycle should occupy. `0.5` = "let a cycle take about half the tick", so a 40s cycle settles at an 80s interval. |
| `idle_tick_seconds` | `300` | When a cycle finds nothing to act on **and** nothing is in flight, back off toward this instead of toward the floor. |
| `idle_backoff_factor` | `2.0` | How fast an idle loop widens: each idle cycle multiplies the current interval by this. |

**Why dynamic ticking exists.** A fixed interval is wrong in both directions: too short and cycles pile up behind each other; too long and idle capacity goes unused. The loop re-derives the interval from measured duration. Tuning is asymmetric — grow quickly when falling behind, shrink slowly (×0.35) so a burst of cheap cycles does not make the loop spin. A deque keeps the slowest recent cycle, so one expensive cycle holds the interval wide for several ticks.

**Why idle backoff exists.** An empty cycle costs ~0s. Feeding that to the duty-cycle maths collapsed the interval to the floor, so the harness retried (and skipped) every floor-length tick for nothing. Idle cycles now push toward `idle_tick_seconds`.

The effective interval is written to `state.json` as `effective_tick`, and shown in the UI.

---

## Agents, delegation, and budgets

| Field | Default | Meaning |
|---|---|---|
| `sub_agent_turn_limit` | `100` | Turn budget per sub-agent **assignment**. On exhaustion the agent reports back and the orchestrator decides: extend, reconfigure, or split. |
| `orchestrator_turn_limit` | `60` | Turns per **cycle**, not a lifetime cap. Re-granted every cycle; the loop runs forever. |
| `max_sub_agent_depth` | `3` | How deep sub-agents may nest. |
| `max_children_per_agent` | `0` | **`0` = unlimited.** Ceiling on how many sub-agents one parent may have *created* over a project. |
| `max_concurrent_sub_agents_per_project` | `3` | How many sub-agents may **run at once** inside one project. This is the real ceiling on fan-out. |
| `max_parallel_workers` | `3` | How many **worker** agents may hold the LLM at once. The orchestrator is deliberately not counted. |
| `llm_queue_depth` | `8` | How many runs may be waiting on a concurrency slot at once. |
| `review_attempts_before_escalation` | `3` | Failed review rounds before the orchestrator is told to escalate. |
| `auto_dispatch` | `true` | Start already-staffed cards without asking the LLM. |

### Creation cap vs concurrency cap

Since **1.10.0** the limit is about concurrency, not creation. `max_children_per_agent` defaults to `0` (unlimited) because a project that needs more than N experts previously got a refusal the orchestrator could only dodge by reusing a badly-scoped agent. Creating an agent is cheap; what costs is holding the LLM.

```json
{
  "max_children_per_agent": 0,
  "max_concurrent_sub_agents_per_project": 3,
  "max_parallel_workers": 3
}
```

### Why the orchestrator is not counted in `max_parallel_workers`

It holds a gate slot for the whole cycle while it decides what to dispatch. Counting it silently costs one worker slot and caps real concurrency at `max_parallel_workers - 1`. The gate is therefore `max_parallel_workers + 1`, and one slot is reserved.

### Why `llm_queue_depth` must exceed `max_parallel_workers`

With a pool the same size as the gate, a dispatched run always finds a slot free, so nothing is ever *visibly* queued — the backlog hides in the thread pool instead of showing on the board.

---

## Router

| Field | Default | Meaning |
|---|---|---|
| `router_host` | `127.0.0.1` | Bind address. Loopback by default; not exposed outside the container. |
| `router_port` | `0` | `0` = OS picks a free port, so a second stack never collides. |
| `llm_server_concurrency` | `2` | Requests one endpoint may serve at once. The real bound on upstream concurrency. |
| `router_selection` | `"throughput"` | `"throughput"` ranks by measured speed; `"round_robin"` spreads evenly. |
| `router_queue_timeout` | `600` | How long a request may sit unassigned before a clean error. |
| `router_max_call_timeout` | `900` | Hard ceiling on one upstream call, independent of adaptive widening. |
| `router_stall_timeout` | `300` | How long one call may go with **no bytes at all**. This is what detects a dead endpoint. |
| `llm_probe_seconds` | `20` | Reachability probe interval (floor 5s). Probes are advisory and never remove an endpoint from rotation. |

See [LLM Router](LLM-Router) for why each of these exists.

---

## Web, storage, and safety

| Field | Default | Meaning |
|---|---|---|
| `host` | `0.0.0.0` | Web UI bind address. |
| `port` | `8090` | Web UI port. |
| `data_dir` | `<repo>/data` | `state.json`, `events.jsonl`, `conversations/`. |
| `projects_dir` | `<data_dir>/projects` | One workspace directory per project = agent output. |
| `max_upload_bytes` | `26214400` (25 MiB) | Per-file upload cap. |
| `max_upload_files` | `12` | Per-request upload cap. |
| `confirm_tools` | `false` | Gate tool calls on human confirmation. `false` because agents run unattended. |
| `sandbox_workspaces` | `true` | Restrict worker/executor file access to each project's workspace. |

Uploads land in `<project workspace>/inputs` so every agent in the project can read them with its own workspace-relative `read_file`.

---

## Environment variables

| Variable | Maps to |
|---|---|
| `HARNESS_CONFIG` | path to `config.json` |
| `HARNESS_LLM_BASE_URL` | `llm.base_url` |
| `HARNESS_LLM_MODEL` | `llm.model` |
| `HARNESS_LLM_API_KEY` | `llm.api_key` (also the fallback key for every server) |
| `HARNESS_LLM_SERVERS` | JSON array; **replaces** the file's `llm_servers` so an env-based deployment is authoritative rather than merged |
| `HARNESS_LLM_QUEUE_DEPTH` | `llm_queue_depth` |
| `HARNESS_LLM_REQUEST_TIMEOUT` / `_MAX_REQUEST_TIMEOUT` / `_MIN_REQUEST_TIMEOUT` / `_NUM_RETRIES` / `_EMPTY_RESPONSE_RETRIES` | matching `llm.*` fields |
| `HARNESS_LLM_TIMEOUT_GROWTH` | `llm.timeout_growth` |
| `HARNESS_LLM_ADAPTIVE_TIMEOUT` | `llm.adaptive_timeout` (falsy: `0 false no off`) |
| `HARNESS_LLM_INPUT_COST_PER_TOKEN` / `_OUTPUT_COST_PER_TOKEN` | matching `llm.*` fields |
| `HARNESS_DATA_DIR` | `data_dir` (and re-derives `projects_dir`) |
| `HARNESS_TICK_SECONDS` | `tick_seconds` |
| `HARNESS_SUB_AGENT_TURN_LIMIT` | `sub_agent_turn_limit` |
| `HARNESS_ORCHESTRATOR_TURN_LIMIT` | `orchestrator_turn_limit` |
| `HARNESS_MAX_CONCURRENT_SUB_AGENTS_PER_PROJECT` | `max_concurrent_sub_agents_per_project` |
| `HARNESS_IDLE_TICK_SECONDS` | `idle_tick_seconds` |
| `HARNESS_MAX_PARALLEL_WORKERS` | `max_parallel_workers` |
| `HARNESS_MAX_SUB_AGENT_DEPTH` | `max_sub_agent_depth` |
| `HARNESS_ROUTER_MAX_CALL_TIMEOUT` | `router_max_call_timeout` |
| `HARNESS_ROUTER_STALL_TIMEOUT` | `router_stall_timeout` |
| `HARNESS_HOST` / `HARNESS_PORT` | `host` / `port` |
| `HARNESS_GIT_SHA` | overrides the git SHA in the version string |

---

## Worked examples

### Point everything at OpenAI

```json
{
  "llm": {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-5.5",
    "api_key_env": "OPENAI_API_KEY",
    "input_cost_per_token": 0.0000025,
    "output_cost_per_token": 0.0000125
  }
}
```

### Local Ollama only

```json
{
  "llm": {
    "provider": "ollama_chat",
    "base_url": "http://host.docker.internal:11434/v1",
    "model": "qwen3:30b",
    "api_key": "ollama",
    "extra_body": { "num_ctx": 32768, "keep_alive": "10m" },
    "input_cost_per_token": 0,
    "output_cost_per_token": 0
  }
}
```

`reasoning_effort` is dropped automatically — Ollama would 400 on it.

### Mixed fleet, speed-aware

```json
{
  "llm": { "base_url": "http://gateway:8080/v1/", "model": "openai/qwen" },
  "llm_servers": [
    { "name": "local-7b", "provider": "ollama_chat",
      "base_url": "http://ollama:11434/v1", "model": "qwen2.5:7b" }
  ],
  "router_selection": "throughput",
  "llm_server_concurrency": 2,
  "max_parallel_workers": 4
}
```

The gateway and the 7B model are not interchangeable; `throughput` keeps requests off the slow one.

### More parallelism

```json
{
  "max_parallel_workers": 6,
  "max_concurrent_sub_agents_per_project": 4,
  "llm_server_concurrency": 3,
  "llm_queue_depth": 12
}
```

Total upstream capacity is `endpoints × llm_server_concurrency`; the worker gate must not exceed it or requests just queue.

### A quieter, cheaper loop

```json
{
  "tick_seconds": 120,
  "idle_tick_seconds": 900,
  "idle_backoff_factor": 3.0,
  "min_tick_seconds": 30
}
```

### Turn-budget experiments

```json
{
  "sub_agent_turn_limit": 40,
  "orchestrator_turn_limit": 30,
  "review_attempts_before_escalation": 2
}
```

Smaller sub-agent budgets force the orchestrator to split coarse tasks; the prompt already treats a task needing two budgets as too coarse.

---

## Changing config at runtime

`POST /api/config` writes the file and the change applies on the next cycle. The orchestrator is told:

```
CONFIG CHANGED (applied this tick): LLM servers now: primary, ollama-a
CAPACITY NOW: 3 of 4 sub-agent slots free (1 running), 0 action(s) already queued, …
```

Capacity numbers appear in every briefing but are deliberately **not** an attention item — attention decides whether a project gets an LLM cycle at all, so always-present capacity would run a paid cycle per project per tick even with nothing to decide.

---

## Version

`agent_harness/version.py` holds `__version__`; `version_string()` returns `1.10.0+6cf4127` from a git checkout, `1.10.0` in a built image. It is stamped into `state.json` meta at startup, and a mismatch against the previous writer is logged as a warning — because the stack runs from a bind-mounted volume, and a container that was not rebuilt otherwise keeps serving an old build against new data invisibly.
