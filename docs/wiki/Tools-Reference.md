# Tools Reference

Tools define what each role can do. Names are exact — they are the strings passed to `Tool(name=…)`.

Source: `agent_harness/tools.py`. Registered by `register_harness_tools()`, called once at startup.

## Tool sets by role

```python
EXEC_TOOLS       = ["terminal", "file_editor", "task_tracker"]      # from the SDK

ORCHESTRATOR_TOOLS = [
    "create_project", "list_projects", "create_task", "update_task", "list_tasks",
    "create_sub_agent", "assign_task", "run_agent", "resume_agent",
    "extend_agent", "reconfigure_agent", "retire_agent", "list_agents",
    "request_review", "record_review", "signal_delivery",
    "read_file", "report_cycle",
]

WORKER_TOOLS = ["report_to_parent", "create_sub_agent", "request_review", "read_file"]
```

| Tool | Orchestrator | Sub-agent (default) |
|---|:--:|:--:|
| `terminal`, `file_editor`, `task_tracker` | — | ✅ |
| `create_project`, `list_projects` | ✅ | — |
| `create_task`, `update_task`, `list_tasks` | ✅ | — |
| `assign_task`, `run_agent`, `resume_agent` | ✅ | — |
| `extend_agent`, `reconfigure_agent`, `retire_agent`, `list_agents` | ✅ | — |
| `request_review` | ✅ | ✅ |
| `record_review` | ✅ | — *(grant explicitly for reviewers)* |
| `signal_delivery` | ✅ | — |
| `report_cycle` | ✅ | — |
| `report_to_parent` | — | ✅ |
| `create_sub_agent` | ✅ | ✅ |
| `read_file` | ✅ | ✅ |

Every agent also gets the SDK's `FinishTool` and `ThinkTool`.

A sub-agent may be granted orchestrator tools by passing `tools=[…]` to `create_sub_agent` — the union of all three lists is accepted. This is how a reviewer gets `record_review`.

---

## Project and task management

### `create_project`
| Param | Type | Notes |
|---|---|---|
| `name` | str | Short human-readable name. |
| `goal` | str | What "done" means. |
| `constraints` | str | Stack, do-not-touch areas. Default `""`. |

Creates the project and its workspace directory under `projects_dir`.

### `list_projects`
No parameters. Returns projects with status and board counts.

### `create_task`
| Param | Type | Notes |
|---|---|---|
| `project_id` | str | |
| `title` | str | Imperative, e.g. `Implement rate limiter`. |
| `objective` | str | **One** verifiable objective. |
| `expertise` | str | Be specific: `python-engineer`, not `coder`. |
| `acceptance_criteria` | list[str] | What a reviewer checks. |
| `deliverables` | list[str] | File paths a reviewer can inspect. |
| `depends_on` | list[str] | Task ids. Order work so parallel tasks never touch the same files. |
| `parent_task_id` | str \| null | For sub-tasks. |
| `column` | str | One of `KANBAN_COLUMNS`. Default `backlog`. |

### `update_task`
`task_id` plus any of `column`, `title`, `objective`, `expertise`, `acceptance_criteria`, `deliverables`, `depends_on`, `blocked_reason`. `note` records your reasoning for the change.

### `list_tasks`
`project_id`, optional `column` to restrict to one column.

---

## Agents and delegation

### `create_sub_agent`
| Param | Type | Notes |
|---|---|---|
| `role` | str | Short label, e.g. `python-engineer`. |
| `specialty` | str | The expertise embodied, e.g. `async Python performance`. |
| `mission` | str | What it is accountable for, one or two sentences. |
| `project_id` | str | |
| `system_prompt` | str \| null | Full replacement prompt. **Omit to use the harness specialist template** with your mission filled in. |
| `tools` | list[str] \| null | Default `EXEC_TOOLS + WORKER_TOOLS`. |
| `turn_limit` | int \| null | Default `sub_agent_turn_limit` (100). |

Guardrails:
- Unknown project → `Unknown project_id …`
- Frozen project → refusal
- Depth: `Sub-agent depth limit (3) reached. Report back instead of delegating deeper.`
- Creation cap (only if `max_children_per_agent` is non-zero): `You already have N sub-agents (limit N). Reuse one with resume_agent.`
- Bad tool name → `Unknown tools: [...]`

### `assign_task`
`task_id`, `agent_id`. Attaches the agent to the card. This is the gate the deterministic scheduler uses: an unstaffed card is never auto-started.

### `run_agent`
| Param | Type | Notes |
|---|---|---|
| `agent_id` | str | |
| `task_id` | str \| null | Defaults to the agent's assigned task. |
| `instructions` | str \| null | Appended to the assignment as `## Additional instructions`. |
| `review` | bool | `true` runs the agent as an independent reviewer. |
| `background` | bool | **Default `true`.** Lets you dispatch other work this cycle. Set `false` only for short, must-finish-now work. |

Guardrails, with the exact refusal text:

| Condition | Refusal |
|---|---|
| No turns left | `Agent X has no turns left (100/100). Use extend_agent or reconfigure_agent first.` |
| Already running | `Agent X is already running; wait for it next cycle.` |
| Project at concurrency limit | `Project already has 3 sub-agents running (its concurrency limit). There is no limit on how many agents you may create - they just cannot all run at once. Dispatch in the background and the scheduler starts it as soon as a slot frees, or work a card in another project.` |
| Project frozen | refusal |

### `resume_agent`
`agent_id`, `instructions` (required), optional `task_id`. Continues an existing agent with its conversation intact — prefer this over creating a clone for the same expertise.

### `extend_agent`
`agent_id`, `extra_turns` (default 100), `decision` (required — why extending is the right call). The `decision` string is stored so the choice is auditable.

### `reconfigure_agent`
`agent_id`, `decision` (required), plus any of `system_prompt`, `specialty`, `mission`, `tools`, `turn_limit` (new **total** budget), `fresh_conversation` (bool). Use when the agent was thrashing, misinformed, or over-scoped.

### `retire_agent`
`agent_id`, `reason` (required).

### `list_agents`
Optional `project_id`. Returns agents with status, turns used, and parent — the data behind the hierarchy view.

---

## Review

### `request_review`
`task_id`, optional `reviewer_expertise`. Moves the card to `review` and names the expertise the reviewer should have. Callable by sub-agents, so a worker can put its own card up for review.

### `record_review`
| Param | Type | Notes |
|---|---|---|
| `task_id` | str | |
| `verdict` | str | `approved` or `changes_requested`. Anything else → `verdict must be 'approved' or 'changes_requested'.` |
| `feedback` | str | Specific, actionable findings **with file paths**. |
| `criteria_met` | list[str] | |
| `criteria_failed` | list[str] | |

Effects: sets `review_verdict`, `review_feedback`, increments `review_rounds`, and records the reviewer as `reviewer_agent_id`.

- `approved` → card moves to `done`, `finished_at` stamped, and any stale `blocked_reason` / `blocked_cause` cleared. A card that just landed cannot also be blocked; leaving the old reason behind makes a done card advertise a blocker it resolved.
- `changes_requested` → card moves to `rework`.

> **`record_review` is not in the default sub-agent tool set.** The review prompt tells a reviewer to call it, so when creating a reviewer grant it explicitly: `"tools": ["read_file", "report_to_parent", "record_review"]`.

---

## Delivery

### `signal_delivery`
`project_id`, `summary` (what was delivered and where to find it), `evidence` (list of paths you verified with `read_file`).

Sets the project's delivery signal so the UI notifies the user. The orchestrator prompt requires verifying deliverables with `read_file` **before** calling it.

---

## Reporting

### `report_cycle` — orchestrator only
`summary` (2–4 sentences: what you did and what happens next), optional `needs_user`. Ends the cycle's turn slice; the loop continues forever regardless.

### `report_to_parent` — sub-agents
| Param | Type | Notes |
|---|---|---|
| `status` | str | `complete`, `partial`, or `blocked`. |
| `summary` | str | What you did, with file paths. |
| `deliverables` | list[str] | Paths produced. |
| `concerns` | str \| null | Risks, unknowns, next steps. |
| `blocker` | str \| null | What you need to proceed. |

This is how a sub-agent finishes. A sub-agent that stops without filing a report leaves its parent guessing.

### `read_file`
`path` (absolute, or relative to the project workspace), `max_lines` (default 200). Available to every role — reviewers and the orchestrator are required to inspect real files rather than trust a self-report.

---

## Guardrails that are not in the tools

Enforced in `AgentRuntime`, so they hold on every dispatch path regardless of which tool asked:

- **Project concurrency seat** claimed inside `run()`, not just checked by `run_agent`.
- **Gate slots** = `max_parallel_workers + 1`, one reserved for the orchestrator.
- **`sandbox_workspaces`** confines worker file access to the project workspace.
- **`confirm_tools = false`** — tool calls are never gated on a human, because agents run unattended.

## Next

[Sub-Agents and Delegation](Sub-Agents-and-Delegation) · [Web Interface and API](Web-Interface-and-API)
