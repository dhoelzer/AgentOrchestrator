# Orchestrator

The orchestrator is the root agent and the owner of the forever-running loop. It does not write production code. It decomposes goals, creates specialists, delegates, reviews, and drives projects to delivery.

Source: `agent_harness/orchestrator.py`, prompts in `agent_harness/prompts.py`.

## The loop

`run_forever()` starts on a daemon thread in `main.py` and never returns until the process is signalled.

```python
while not self._stop.is_set():
    self._wake.clear()
    self._scheduler_notes = self._run_scheduler()   # deterministic, no LLM
    reports = self.run_cycle()                      # judgement, may be skipped
    wait = self._next_tick(elapsed, self._cycle_had_work)
    self._wake.wait(timeout=max(1.0, wait))         # interruptible sleep
```

Two properties are load-bearing:

- **The scheduler sweep runs before the cycle and is not gated on an orchestrator slot.** Starting already-staffed work needs no model, so a cycle that gets skipped still starts work. Without this, one stuck orchestrator prompt freezes the whole board.
- **The sleep is interruptible.** `wake()` cuts an idle wait short, so a project the user just created is not left sitting for the whole idle interval.

`SIGINT`/`SIGTERM` call `orchestrator.stop()`, which sets both `_stop` and `_wake` so shutdown is prompt regardless of the current interval.

## Cycle phases

`_run_cycle()` runs in this order:

| # | Phase | LLM? | What it does |
|---|---|---|---|
| 1 | `reload_config_if_changed()` | no | Re-reads `config.json`; a hand edit or API patch takes effect without a restart. Returns human-readable notes about what changed. |
| 2 | `sweep_orphans()` | no | Forgets agents whose project is gone, detaches them from deleted parents, and prunes dead child ids (which otherwise count toward fan-out accounting). |
| 3 | `sweep_finished_agents()` | no | Archives specialists whose work is finished, out of the live hierarchy. Cost, tokens, and transcripts stay queryable. |
| 4 | `reconcile()` | no | Deterministic board transitions: promote `backlog` → `ready` once dependencies land, mark delivery readiness, keep columns honest. |
| 5 | `detect()` | no | Builds the per-project **attention map** — the only thing that decides whether a project gets an LLM cycle. |
| 6 | config + scheduler notes merged in | no | `CONFIG CHANGED` and `SCHEDULER:` lines are appended to attention. |
| 7 | `_run_project_cycle()` | **yes** | For each visible, unfrozen project **with attention**, brief the orchestrator and let it act through its tools. |

A project with no attention gets **no LLM call**. That is the main cost control in the harness.

## Skipping instead of queueing

`run_cycle()` refuses rather than queues:

```python
if not self._cycle_lock.acquire(blocking=False):
    self._note_cycle_skip("another cycle was still running")
    return None
busy = self.runtime.llm_busy("orchestrator")
if busy is not None:
    self._note_cycle_skip(f"an orchestrator prompt is already {busy['status']} ({busy['id']})")
    return None
```

There is deliberately no way to queue a cycle behind a running one. A queued cycle would run a briefing built from the board as it looked *before* the running cycle changed it, so it would act on stale state and then write over the fresher decisions. Refusing is correct; waiting is not.

Skips are recorded in `state.json` meta (`cycle_skips`, `cycle_stuck_seconds`, `cycle_skip_reason`) and surfaced as an alert, so a stalled loop is visible rather than silent.

## Attention detection

`detect()` returns `{project_id: [notes]}`. These are the triggers, in the order they are emitted:

| Trigger | Instruction given to the orchestrator |
|---|---|
| New user message on a project | Highest priority. Decide what cards this implies, create them, staff them. |
| Card in `review` | Coordinate a review now: create a `reviewer-<expertise>` specialist and run it with `review=true`. |
| Card in `blocked` with user directives | Apply the direction and re-dispatch. |
| Card blocked on `llm` | **Hold.** An LLM outage is not something the orchestrator can fix; re-dispatching into a dead endpoint every tick is how a project looks "stuck". A user directive still wins. |
| Card blocked on `budget` | The budget-exhausted briefing: extend, reconfigure, or split. |
| Card blocked on `agent_failed` | Reconfigure or replace the agent, then re-dispatch. |
| Card in `rework` | Re-dispatch with the review feedback. At `review_attempts_before_escalation` rounds, escalate. |
| Card in `ready`, unstaffed | Create a `<expertise>` specialist and dispatch it. |
| Card in `in_progress` with no running agent | Re-dispatch it or move it to `rework` — the board is lying. |
| Card in `backlog` with unmet deps | Names the blocking card ids. |
| All tasks done, not yet signaled | Verify deliverables with `read_file`, then `signal_delivery`. |
| Project has no tasks | Decompose the goal now. |
| Agent `budget_exhausted` / `failed` / `llm_unavailable` | Per-agent recovery instructions (skipped when already surfaced via its blocked card). |
| Endpoint health degraded | Appended to **every** project: empty-response count, transient errors, current widened timeout. |

Frozen (paused) projects are skipped entirely — computing attention for them would put a "waiting for review" line in the UI that promises a review never scheduled.

### Strikes and recovery menus

A blocked card is evaluated every tick, so the harness counts how long it has sat blocked on the *same* reason (`_count_strike`). A changed blocker is a new problem and restarts the count. The strike count selects a recovery menu, so the advice escalates rather than repeating itself.

## Briefings

Each cycle the orchestrator receives a fresh briefing: the board state, the attention items, and a capacity section.

```
CAPACITY NOW: 2 of 3 sub-agent slots free (1 running), 0 action(s) already
queued, max 3 running per project, 2 LLM server(s) in rotation (primary,
ollama-2). Creating sub-agents is unlimited; running them at once is not.
```

**Capacity is in every briefing but is deliberately never an attention item.** Attention decides whether a project gets a cycle at all, so making capacity always-present would run a paid LLM cycle per project per tick even when there is nothing to decide.

## Turn budgets

Two different budgets, and conflating them has caused real failures:

| Budget | Config key | Semantics |
|---|---|---|
| Orchestrator | `orchestrator_turn_limit` (60) | **Per-cycle slice, not a lifetime cap.** Re-granted every cycle: `turn_limit = turns_used + orchestrator_turn_limit`. The loop runs forever. |
| Sub-agent | `sub_agent_turn_limit` (100) | **Real and finite per assignment.** Exhaustion is a decision point for the orchestrator. |

The briefing renders these differently on purpose. Sub-agents show `used/limit`. The orchestrator shows `N lifetime (no limit; 60 per cycle)`.

Why that matters: an earlier build rendered the orchestrator's row as `3272/3320`. The model read that as a countdown, filed `needs_user` reports about being down to its last cycle, and **stopped planning and started triaging**. Enforcement was always correct; only the display was wrong. The orchestrator system prompt now spends a section on this:

> The turn count in your briefing is a record of work already done, not a countdown of work remaining. … never report that you are "running out of turns". That framing is wrong and it is harmful.

## The orchestrator system prompt

Key sections, abridged:

- **How you operate** — invoked once per cycle with a fresh briefing; each cycle must make concrete progress; never "done"; never wait for the user unless a task is genuinely blocked on a human decision.
- **Your budget** — see above.
- **Planning discipline** — each task needs a single verifiable objective, explicit acceptance criteria, and named deliverables (file paths a reviewer can inspect). Name the expertise specifically: `python-engineer`, not `coder`. Order with `depends_on` so parallel tasks never touch the same files. Split any task a specialist could not finish in one 100-turn budget — a task needing two budgets is usually too coarse.
- **Capacity discipline** — the harness re-reads its config every cycle; act on `CONFIG CHANGED` / `CAPACITY NOW`. Never try to reconfigure the harness yourself.
- **Delegation discipline** — `create_sub_agent` then `assign_task`; reuse with `resume_agent` rather than proliferating clones. Specialists may create their own sub-specialists; you stay accountable for the outcome, not the internals. Fan-out is purposeful: creating an agent is cheap, so scope each narrowly.
- **Review discipline** — anything in `review` must be reviewed the same cycle; read the actual deliverables with `read_file`; **never approve on the worker's self-report alone**.
- **Budget discipline** — on `budget_exhausted`, explicitly choose `extend_agent`, `reconfigure_agent`, or `retire_agent` + split, and state the reasoning in the `decision` argument so it is auditable.

The expertise menu is a suggestion, not a whitelist — the prompt says to invent new ones freely.

## Next

[LLM Router](LLM-Router) · [Sub-Agents and Delegation](Sub-Agents-and-Delegation) · [Kanban and Project Lifecycle](Kanban-and-Project-Lifecycle)
