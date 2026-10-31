# Orchestrator Harness

A long-running agent harness built on the [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk).

An **orchestrator** agent owns a portfolio of projects. Every minute it looks at the
board, creates **specialist sub-agents** for the expertise each task needs, dispatches
them, coordinates reviews, decides what to do when an agent runs out of its turn
budget, and signals a project for delivery. It runs forever. A web UI shows project
creation plus a kanban view of every project, including the hierarchy of agents and
sub-agents and each one's status.

```
python -m harness.main            # web UI + forever loop
open http://localhost:8090        # create projects, watch the board
```

---

## Requirements → how they are met

| Requirement | Where |
|---|---|
| LLM swappable via base URL + model name, OpenAI-style API, preconfigured for Qwen | `harness/config.py`, `config.json` |
| Automatic creation of specialist sub-agents | `create_sub_agent` tool → `harness/tools.py`, prompt template in `harness/prompts.py` |
| Orchestrator system prompt for long-running projects; decomposes work and names required expertise | `ORCHESTRATOR_SYSTEM_PROMPT` in `harness/prompts.py` |
| Web interface to create projects | `harness/web/templates/index.html` |
| Kanban view incl. agent/sub-agent hierarchy and per-agent status | `harness/web/api.py` + `harness/web/static/app.js` |
| Every-minute check: staff waiting work, coordinate reviews, detect deliverables, update the board; runs forever | `Orchestrator.run_forever()` / `run_cycle()` in `harness/orchestrator.py` |
| Sub-agents capped at 100 turns; on exhaustion they report back and the orchestrator decides whether to re-run, reconfigure, or split | `harness/runtime.py` (enforcement) + `extend_agent` / `reconfigure_agent` / `retire_agent` in `harness/tools.py` |

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
python -m harness.main
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
- `GET /api/state` — full board + agent tree + events (polled every 5 s)
- `POST /api/projects` — create a project
- `POST /api/tick` — run a cycle immediately
- `GET /api/projects/{id}/tasks/{id}`, `GET /api/projects/{id}/agents/{id}` — detail
- `GET /health` — model, base URL, tick, in-flight agents

## Layout

```
harness/
  config.py       LLM + runtime config (OpenAI-compatible endpoint)
  store.py        Durable projects / tasks / agents / event log
  prompts.py      Orchestrator, specialist, reviewer prompts + briefings
  runtime.py      Conversation lifecycle, turn-budget enforcement, resume
  tools.py        Custom tools: board control, delegation, review, delivery
  orchestrator.py Orchestrator agent + forever scheduler
  main.py         Entry point (web UI + loop)
  web/            FastAPI app, templates, static assets
tests/
  test_board_mechanics.py  Board logic with no LLM
  test_e2e_llm.py          Full loop against the configured LLM
  test_turn_budget.py      Budget exhaustion, extension, reconfiguration
```

## Running the tests

```bash
pip install -r requirements.txt

# No LLM needed — board mechanics, delegation, review, delivery, persistence:
python tests/test_board_mechanics.py

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
