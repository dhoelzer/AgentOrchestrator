# Web Interface and API

A FastAPI app serving a single-page UI plus a JSON API. Source: `agent_harness/web/api.py`, `web/templates/index.html`, `web/static/app.js`.

```bash
open http://localhost:8090
```

## The interface

### Creating projects

A form for **name**, **goal**, and **constraints**. Creating a project calls `orchestrator.wake("project '…' created")`, so the loop does not have to wait out its current interval — including a 300s idle interval — before anyone looks at it.

### Kanban board

One board per project, columns in `KANBAN_COLUMNS` order:

```
backlog │ ready │ in_progress │ review │ rework │ done │ blocked
```

Each card shows its title, expertise, assigned agent, turn usage, dependency state, and — when blocked — the reason and cause. Cards carry a live status derived from the runtime: a card in `in_progress` whose agent is not in `runtime.inflight` is visibly inconsistent, which is how a lying board gets noticed.

### Agent hierarchy

A tree of agents rooted at the orchestrator, showing parent/child relationships, role, specialty, status, and turns used. This is the view that answers "who is working on what, and how deep did it go."

- Finished specialists are **archived** out of the live tree by `sweep_finished_agents()` so they do not hide the agents actually working. Their cost, tokens, and transcripts remain queryable.
- Orphans (agents whose project is gone, agents detached from a deleted parent) are pruned by `sweep_orphans()`.

### LLM router panel

Live view of the request funnel: per-endpoint health, in-flight count, cooldown, measured `tokens_per_second`, and the request cards currently `queued` / `awaiting` / `generating`. Completed cards stay visible ~12s so a card does not vanish the instant it lands.

### Settings

Edits `config.json` through `POST /api/config` and hot-reloads. Fields exposed include the LLM endpoint and fleet, `max_parallel_workers`, `max_concurrent_sub_agents_per_project`, `max_children_per_agent` (0 = unlimited), `idle_tick_seconds`, turn limits, and router timeouts.

### Alerts

Raised for conditions the user should see: repeated skipped cycles, LLM outages, projects ready for delivery, review escalations. Acknowledge via `POST /api/alerts/ack`.

---

## API

### Health and state

| Method | Path | Notes |
|---|---|---|
| `GET` | `/health` | `ok`, `harness_version` (e.g. `1.10.0+6cf4127`), `model`, `base_url`, `tick_seconds`, `sub_agent_turn_limit`, `last_tick`, `inflight` (agent ids). |
| `GET` | `/api/state` | The whole board: projects, boards by column, revisions, agents, alerts, loop metadata. `?show_archived=true` includes archived agents. |
| `POST` | `/api/tick` | Force a cycle now. Skips if one is already running. |
| `GET` | `/api/queue/{entry_id}/prompt` | The full prompt a queued action is waiting to send. `404 Queue entry already dispatched or gone` once it has left the queue. |

### Projects

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/projects` | Create a project. Body: `name`, `goal`, `constraints`. Wakes the loop. |
| `POST` | `/projects` | Form-encoded variant used by the UI. |
| `POST` | `/api/projects/{id}/remove` | Delete a project and its agents. |
| `POST` | `/api/projects/{id}/pause` | Freeze/unfreeze. In-flight runs finish; nothing new starts. |
| `POST` | `/api/projects/{id}/revise` | Change the goal; appends to the revision log (last 50 kept). |
| `POST` | `/api/projects/{id}/messages` | Send user input into the project. **Wakes the loop.** Highest-priority attention item. |
| `POST` | `/api/projects/{id}/uploads` | Files into `<workspace>/inputs`. Bounded by `max_upload_bytes` (25 MiB) and `max_upload_files` (12). Wakes the loop. |
| `POST` | `/api/projects/{id}/mark-delivered` | Mark delivered without an orchestrator cycle. |
| `GET` | `/api/projects/{id}/queue` | This project's queued LLM requests. |
| `GET` | `/api/projects/{id}/agents/{agent_id}` | One agent: status, turns, cost, tokens, transcript. |
| `GET` | `/api/projects/{id}/tasks/{task_id}` | One card. |
| `POST` | `/api/projects/{id}/tasks/{task_id}/directives` | Attach direction to a blocked card. The orchestrator is then told to apply it and re-dispatch. |

### Config

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/config` | Current config via `model_dump()`, so new fields appear automatically. **API keys are redacted to `***`** for the primary and every server, so the response is safe to render in the UI. |
| `POST` | `/api/config` | Patch + hot-reload + persist. |

```json
POST /api/config
{ "max_parallel_workers": 6, "idle_tick_seconds": 120, "persist": true }
```

```json
→ { "changed": { "max_parallel_workers": 6, "idle_tick_seconds": 120 },
    "runtime": { "llm_timeout": 300, "max_parallel_workers": 6, "llm_slots": 7 } }
```

Behaviour:
- Applies immediately for new agents; for LLM timeouts and parallelism, also for conversations already open.
- Writes back to `config_path` (`--config`, `$HARNESS_CONFIG`, or `<repo>/config.json`). A patch with no actual change is rejected with `400 Patch contained no changes.`
- A write failure is a `500 Could not write config: …` — **the live config is still updated**, so the change takes effect even when persistence fails.
- After a successful write the loop is told not to read its own write back as an external edit.

> **Ownership matters.** In Docker the config lives at `/data/config.json` (the bind-mounted `./harness/config.json`). If that file is owned by a uid the container cannot write, the API returns 500 while the in-memory change still applies — and is lost on restart. Run the container as a uid that owns the mount, or fix ownership on the host.

### LLM router

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/llm/router` | Per-endpoint snapshots, live request cards, `served` / `retries` counters. |
| `GET` | `/api/llm/router/requests/{request_id}` | One request, including the prompt it carried. |
| `POST` | `/api/llm/router/probe` | Re-probe every endpoint immediately. |
| `POST` | `/api/llm/router/test` | Run a test prompt against a named server, or against an **unsaved** definition you pass in. |

`/api/llm/router/test` against an unsaved definition uses `resolved_api_key(use_env=False)`, so the harness-wide key is not silently sent to an endpoint nobody has vetted.

### Alerts

| Method | Path | Notes |
|---|---|---|
| `POST` | `/api/alerts/ack` | Acknowledge alerts. |
| `DELETE` | `/api/alerts/ack` | Clear acknowledgements. |

---

## Loop metadata in `/api/state`

Useful for operations:

| Key | Meaning |
|---|---|
| `last_tick`, `last_cycle_completed_at` | Only written on a **successful** cycle. |
| `last_cycle_attempt_at` | Written every attempt, successful or not. |
| `cycle_status` | `running` / `ok` / `failed`. |
| `cycle_duration`, `last_cycle_seconds` | Measured cost of the last cycle. |
| `effective_tick` | The interval the tuner settled on. |
| `cycle_skips`, `cycle_stuck_seconds`, `cycle_skip_reason` | Why cycles are being refused. Cleared once a cycle runs. |
| `cycle_error` | Last failure detail. |
| `cycle_count` | Total cycles. |
| `loop_started_at` | When the forever loop started. |
| `harness_version` | Build actually running. |

`last_tick` alone would make a harness whose cycles keep failing look like "the last cycle never updates". The status and failure fields make the stall visible instead.

## Next

[Orchestrator](Orchestrator) · [Deployment and Operations](Deployment-and-Operations)
