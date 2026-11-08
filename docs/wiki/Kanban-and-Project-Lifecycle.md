# Kanban and Project Lifecycle

The board is the shared state between the user, the orchestrator, and the deterministic scheduler. Everything on it is persisted to `state.json`.

Source: `agent_harness/store.py` (`KANBAN_COLUMNS`, `TaskSpec`, `ProjectSpec`), transitions in `orchestrator.py` (`reconcile`) and `tools.py`.

## Columns

```python
KANBAN_COLUMNS = [
    "backlog", "ready", "in_progress", "review", "rework", "done", "blocked",
]
```

| Column | Meaning | Who moves a card here |
|---|---|---|
| `backlog` | Created, dependencies may be unmet. | `create_task` default. |
| `ready` | Unblocked, needs staffing (or is staffed and waiting on capacity). | `reconcile()` when deps land; scheduler starts staffed cards from here. |
| `in_progress` | A worker stretch is running or was expected to run. | `assign_task` / `run_agent`. |
| `review` | Worker stretch finished; awaiting a verdict. | A clean worker run; `request_review`. |
| `rework` | Review said `changes_requested`. | `record_review`. |
| `done` | Approved. | `record_review` with `approved`. |
| `blocked` | Needs a decision the harness cannot make alone. | Budget exhaustion, LLM outage, agent failure, or the orchestrator. |

An invalid column name is rejected when the record is loaded.

## Transitions that need no judgement

`reconcile()` runs every cycle, before any LLM call:

| Condition | Transition |
|---|---|
| `backlog` and all `depends_on` are `done` | → `ready`, event `task_unblocked`. |
| `blocked`, deps met, has dependencies, and `blocked_cause` **not** in `budget` / `llm` / `manual` | → `ready`, reason and cause cleared. Another agent resolved the upstream work, so re-queue rather than wait on judgement. |
| Any column other than `blocked` | Strike counters reset, so a later unrelated block starts from one. |
| Every task `done` and project `active` | Project → `ready_for_delivery`, `ready_at` stamped, `ready_generation` incremented, revision logged, event `delivery_candidate`. |

Blocks the harness wrote for a reason dependencies **cannot** fix — a budget exhaustion or an LLM outage — deliberately stay put for the cycle briefing.

## Transitions a run produces

When a sub-agent run finishes, `_dispatch` reconciles the card:

| Run outcome | Card |
|---|---|
| `budget_exhausted` | → `blocked`, `blocked_cause = "budget"`. |
| `llm_unavailable` | → `blocked`, `blocked_cause = "llm"`, reason names the current widened timeout. |
| `failed` / error | → `blocked`, `blocked_cause = "agent_failed"`, reason carries the error. |
| Clean run, card was `blocked` / `rework` / `in_progress`, **not** a reviewer | → `review`, stale `blocked_reason` cleared, event `task_unblocked`. |
| Clean **reviewer** run | Card does **not** move — `record_review` moves it. |

> A run that did not fail or run out of budget means whatever was holding the card is no longer holding it. Leaving a card in `blocked` after a clean run is what makes the board lie.

Reviewer dispatches are excluded because their verdict, not their completion, decides the card's fate.

## Review flow

```
worker run clean ──► review ──► orchestrator creates reviewer-<expertise>
                              │  (or scheduler starts an already-staffed reviewer)
                              ▼
                     reviewer inspects files with read_file
                              ▼
                     record_review(verdict=…)
                        ├── approved          ──► done
                        └── changes_requested ──► rework ──► re-dispatch with feedback
```

`review_rounds` increments per verdict. At `review_attempts_before_escalation` (default 3) the briefing escalates:

```
Task task_x has failed review 3 times - escalate: reconfigure the specialist,
split the task, or mark it blocked for the user.
```

The orchestrator prompt forbids approving on a self-report:

> Approve only when acceptance criteria are demonstrably met. Never approve on the worker's self-report alone.

**Reviewer gotcha:** `record_review` is an orchestrator tool and is not in the default sub-agent tool set. Grant it when creating a reviewer, or the reviewer can inspect and report but cannot move the card. See [Sub-Agents and Delegation](Sub-Agents-and-Delegation).

## Blocked cards and strikes

A blocked card is evaluated every tick, so the harness counts how long it has sat blocked on the **same** reason:

```python
strikes = self._count_strike(task)   # a changed blocker is a new problem: count restarts
```

The strike count selects a recovery menu, so advice escalates instead of repeating. A card leaving `blocked` resets its strikes.

`blocked_cause` vocabulary:

| Cause | Meaning | Scheduler behaviour |
|---|---|---|
| `llm` | Endpoint outage. | **Hard block** — never auto-started. The orchestrator is told to hold, not re-dispatch. |
| `manual` | Waiting on a human. | Hard block. |
| `budget` | Turn budget exhausted. | Not auto-started; needs `extend_agent` / `reconfigure_agent`. |
| `agent_failed` | Run raised. | Eligible once the agent is reconfigured or replaced. |

## Delivery

A project becomes `ready_for_delivery` deterministically when every task is `done`. The orchestrator is then told:

```
ALL tasks for '<project>' are done - verify the deliverables with read_file, then
call signal_delivery so the user is notified.
```

`signal_delivery(project_id, summary, evidence)` sets the delivery signal, which surfaces in the UI as a notification to the user. `evidence` is the list of paths actually verified.

`ready_generation` increments each time a project re-enters the ready state, so a project that gains work after being ready signals again rather than looking already-handled.

The user can also mark delivery directly with `POST /api/projects/{id}/mark-delivered`.

## Pausing

`POST /api/projects/{id}/pause` freezes a project. While frozen:

- No attention is computed for it, and no orchestrator cycle runs for it.
- The scheduler will not start new work in it.
- `reconcile()` skips it, so no card silently advances and nothing is declared delivery-ready while the halt is on.
- **Runs already in flight still finish** and still write their own results through the tools — that is the queue draining.

Frozen projects are hidden from the default board view; `?show_archived=true` reveals archived agents.

## User input

Two ways to talk to a running project:

| Mechanism | Scope | Effect |
|---|---|---|
| Project message (`POST /api/projects/{id}/messages`) | Whole project | Highest-priority attention item. Wakes the loop. The orchestrator decides what cards it implies. |
| Card directive (`POST …/tasks/{id}/directives`) | One blocked card | The briefing says: `BLOCKED and the user has supplied direction - apply it and re-dispatch`. |

A user directive **overrides** the "hold during an LLM outage" instruction — it may be telling the orchestrator the endpoint is fixed.

## Uploads

`POST /api/projects/{id}/uploads` places files in `<workspace>/inputs`, where every agent in the project can read them with its own workspace-relative `read_file`. Assignment prompts include an inputs block, and reviewers are told:

> Judge the deliverables against this uploaded material: work that contradicts it or ignores it is not acceptable.

Bounded by `max_upload_bytes` (25 MiB) and `max_upload_files` (12). Uploads wake the loop.

## Persistence

`state.json` holds projects, tasks, agents, and meta; `events.jsonl` is the append-only activity feed; `conversations/` holds SDK-persisted conversation state per agent, which is what makes `resume_agent` continue with context intact.

The store is **in-memory with a JSON flush**. Run exactly one stack per data directory — two containers sharing a mount clobber each other.

## Next

[Orchestrator](Orchestrator) · [Tools Reference](Tools-Reference)
