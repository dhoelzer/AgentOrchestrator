# Sub-Agents and Delegation

The orchestrator creates specialists on demand. Specialists may create their own sub-specialists. Every sub-agent has a finite turn budget per assignment, and the orchestrator decides what happens when it runs out.

Source: `agent_harness/tools.py`, `agent_harness/runtime.py`, `agent_harness/scheduler.py`.

## Creating a specialist

```
create_sub_agent(
  role="python-engineer",
  specialty="async Python performance under high concurrency",
  mission="Implement the token-bucket rate limiter in limiter.py and its tests.",
  project_id="proj_…",
  system_prompt=null,          # omit to use the harness specialist template
  tools=null,                  # default: terminal, file_editor, task_tracker
                               #        + report_to_parent, create_sub_agent,
                               #          request_review, read_file
  turn_limit=null              # default: sub_agent_turn_limit (100)
)
```

Omit `system_prompt` and the harness fills its specialist template with the mission, specialty, project name, goal, and workspace. Supplying one replaces the template entirely — use that for a genuinely different mode of working, not for emphasis.

`tools` is validated against the union of worker, executor, and orchestrator tool names; an unknown name is refused with `Unknown tools: [...]`.

## Hierarchy and depth

Each agent records its parent and its `depth`. The orchestrator is depth 0.

```python
depth = (caller.depth + 1) if caller else 0
if caller is not None and depth > config.max_sub_agent_depth:
    return _err(
        f"Sub-agent depth limit ({config.max_sub_agent_depth}) reached. "
        "Report back instead of delegating deeper."
    )
```

Default `max_sub_agent_depth = 3`. The instruction that comes back is a redirect, not just a refusal: report upward rather than nesting further.

The orchestrator stays accountable for the **outcome**, not a child's internals. A parent that needs to know what a child did should read its `report_to_parent` result, not try to supervise it turn by turn.

## Creation is unlimited; concurrency is not

Since **1.10.0** the constraint is about running, not creating.

| Knob | Default | Meaning |
|---|---|---|
| `max_children_per_agent` | `0` = **unlimited** | Ceiling on how many sub-agents one parent may have *created* over a project. |
| `max_concurrent_sub_agents_per_project` | `3` | How many sub-agents may **hold the LLM at once** inside one project. |
| `max_parallel_workers` | `3` | How many worker agents may hold the LLM at once, harness-wide. |

Previously `max_children_per_agent` defaulted to 6 and a project needing more than six experts got a refusal the orchestrator could only dodge by reusing a badly-scoped agent. Creating an agent is cheap; what costs is holding the LLM.

The refusal you get when a project is saturated:

```
Project already has 3 sub-agents running (its concurrency limit). There is no
limit on how many agents you may create - they just cannot all run at once.
Dispatch in the background and the scheduler starts it as soon as a slot frees,
or work a card in another project.
```

If you set a non-zero `max_children_per_agent`, the refusal is:

```
You already have 6 sub-agents (limit 6). Reuse one with resume_agent.
```

## Three different limits — do not conflate them

```
max_parallel_workers  ──► gate slots (harness-wide LLM concurrency)
                        + 1 slot reserved for the orchestrator

max_concurrent_sub_agents_per_project ──► project seats (per project)

llm_server_concurrency × len(servers) ──► actual upstream capacity
```

- **Gate slots** = `max_parallel_workers + 1`. The orchestrator holds one for the whole cycle while it decides what to dispatch; counting it as a worker silently caps real concurrency at `max_parallel_workers - 1`.
- **Project seats** are claimed inside `AgentRuntime.run()`, so the limit holds on *every* dispatch path — tool call, scheduler sweep, or nested dispatch — rather than only where a tool politely checks. Claims are re-entrant per agent id, so a nested dispatch on the parent's own thread cannot deadlock.
- **The orchestrator never consumes a project seat.** It is not a sub-agent.

## Turn budgets

`sub_agent_turn_limit` (default 100) is a budget **per assignment**, and it is real.

```python
remaining = agent.turn_limit - agent.turns_used
if remaining <= 0:
    return _err(
        f"Agent {agent.id} has no turns left ({agent.turns_used}/{agent.turn_limit}). "
        "Use extend_agent or reconfigure_agent first."
    )
```

When a budget runs out mid-task, the agent's status becomes `budget_exhausted`, its card moves to `blocked` with `blocked_cause = "budget"`, and the orchestrator receives:

```
BUDGET EXHAUSTED: agent agent_x (python-engineer) used 100/100 turns on task task_y ('Implement rate limiter').

Last report from that agent:
<the agent's last report_to_parent summary>

Decide: extend the budget, reconfigure the agent, or split the work.
```

### The three options

The orchestrator system prompt requires an explicit choice, with the reasoning in the `decision` argument so it is auditable:

| Action | When |
|---|---|
| `extend_agent(extra_turns=100, decision=…)` | Same agent, same config, another slice. Use when it was making real progress and simply ran out of time. |
| `reconfigure_agent(decision=…, system_prompt/specialty/mission/tools/turn_limit, fresh_conversation=…)` | Use when it was thrashing, misinformed, or over-scoped. |
| `retire_agent` + split the task | Break the work into smaller tasks with new specialists. |

The planning rule that makes budgets meaningful: **split any task a specialist could not finish in one 100-turn budget — a task that needs more than one budget is usually too coarse.**

## The scheduler: dispatch without an LLM

`agent_harness/scheduler.py` handles the part of orchestration that needs no model.

> The orchestrator decides *which* agent owns a card. Launching that agent is not a decision — it is the mechanical consequence of one. Doing it through the LLM anyway costs a full round trip through a saturated endpoint before anything starts, which is how an idle endpoint ends up with nothing running: the card is staffed, the agent is waiting, and the only thing standing between them is an orchestrator turn that has to queue for a slot itself.

The split:

| | Owner |
|---|---|
| Judgement: decompose, staff, re-specify, review, deliver | Orchestrator (LLM) |
| Consequence: start what is already staffed and unblocked | Scheduler (no LLM) |

The scheduler deliberately makes no decisions. `assigned_agent_id` is the gate: an unstaffed card is left alone for the briefing, and a card the orchestrator chose not to staff stays unstaffed.

### What it will start

| Column | Condition |
|---|---|
| `ready`, `rework` | Always eligible. |
| `backlog` | Eligible **only** if it already has an agent attached — an explicit assignment is itself the staffing decision. This matters when cycles keep getting refused: `reconcile()` runs inside a cycle, so a harness with a stuck orchestrator prompt would never promote anything and the board would freeze in `backlog` forever. |
| `review` | Only if `review_verdict is None and review_rounds == 0` — a reviewer run does not move its own card, `record_review` does. Without this a reviewer that already reported would be dispatched again forever. |

Hard blocks it will not work through: `blocked_cause` in `("llm", "manual")`, or an agent with `status == "llm_unavailable"` or `llm_failures >= 1`.

It also skips archived agents, agents already in flight, agents with no turns left, and cards whose dependencies are not `done`. Selection is oldest-first, so a backlog drains in creation order rather than dict order.

When it cannot start something, it tells the orchestrator why, so the orchestrator does not waste a tool call:

```
SCHEDULER: proj_x is at its sub-agent concurrency limit (3 running). task_y '…' is
staffed and waiting on a slot, not on a decision. It starts on its own when one
frees; dispatch work in another project meanwhile.
```

```
SCHEDULER: no free LLM slot; task_y '…' is staffed and waiting on capacity, not on a
decision. Do not run_agent it; it starts on its own once a slot frees.
```

Set `auto_dispatch: false` to disable this and make every dispatch go through the LLM.

## Background vs synchronous dispatch

`run_agent(background=true)` is the default and the right choice almost always: the orchestrator can dispatch several cards in one cycle instead of serializing them through one saturated endpoint.

```
Dispatched python-engineer (agent_x) in the BACKGROUND on task task_y with 100
turns. It moves to 'review' when done; dispatch other ready work now.
```

Use `background: false` only for short, must-finish-now work.

## Reuse rather than clone

The prompt is explicit: reuse an existing specialist with `resume_agent` when the same expertise is needed again — do not proliferate clones. A resumed agent keeps its conversation, so it retains what it already learned about the codebase.

## Archiving

`sweep_finished_agents()` archives specialists whose work is finished, out of the live hierarchy. Otherwise a long-running project ends up with a wall of finished agents hiding the ones actually doing work. Archiving, not deleting, keeps cost, tokens, and the transcript queryable and leaves the agent available to resume.

`sweep_orphans()` handles agents that can never run again — an agent whose project was deleted in an earlier process, an agent detached from a deleted parent, and parents still holding dead child ids. Those dead ids would otherwise count toward fan-out accounting.

## Gotchas

- **A reviewer needs `record_review`.** It is an orchestrator tool and is **not** in the default sub-agent tool set, yet the review prompt tells the reviewer to call it. When creating a reviewer, pass it explicitly:
  ```json
  "tools": ["read_file", "report_to_parent", "record_review"]
  ```
  Otherwise the reviewer can inspect and report but cannot move the card, and the card sits in `review` until the orchestrator records the verdict itself.
- **`report_to_parent` is how a sub-agent finishes.** A sub-agent that stops without filing a report leaves its parent guessing.
- **Sub-agents cannot create projects or move the board.** Those are orchestrator tools.

## Next

[Tools Reference](Tools-Reference) · [Kanban and Project Lifecycle](Kanban-and-Project-Lifecycle)
