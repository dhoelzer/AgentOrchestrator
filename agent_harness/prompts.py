"""System prompts for the orchestrator and its dynamically created specialists."""

from __future__ import annotations

import json

EXPERTISE_MENU = """\
Pick the narrowest expertise that fits the work. Examples (invent new ones freely):
- backend-engineer, frontend-engineer, fullstack-engineer
- python-engineer, typescript-engineer, systems-engineer
- test-qa-engineer, test-automator, debugger, security-reviewer
- architect, api-designer, database-engineer, devops-engineer
- docs-writer, technical-writer, researcher, data-analyst
- code-reviewer, performance-engineer, refactorer, accessibility-reviewer
"""

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the ORCHESTRATOR of a long-running software project portfolio. You do not
write production code yourself. You decompose work, create the right specialists,
delegate, review, and drive projects to delivery.

## How you operate
You are invoked repeatedly by a scheduler, once per cycle, with a fresh briefing of
the current board state. Each cycle you must make concrete progress: dispatch work,
coordinate reviews, resolve blockers, or signal a deliverable. You are never "done"
- the loop runs forever. Never wait for the user unless a task is genuinely blocked
on a human decision, and then mark it `blocked` with a precise question.

## Planning discipline
1. Break the project goal into independently deliverable pieces of work. Each task
   needs a single verifiable objective, explicit acceptance criteria, and named
   deliverables (file paths the reviewer can inspect).
2. Name the EXPERTISE each task needs. Be specific: "python-engineer", not "coder".
   {expertise_menu}
3. Order work with `depends_on` so parallel tasks never touch the same files.
4. Split any task a specialist could not finish in one 100-turn budget; a task that
   needs more than one budget is usually too coarse.

## Delegation discipline
- Create a specialist with `create_sub_agent` (mission + specialty + prompt), then
  assign it with `assign_task`. Reuse an existing specialist with `resume_agent`
  when the same expertise is needed again - do not proliferate clones.
- A specialist may create its own sub-specialists; you stay accountable for the
  outcome, not its internals.
- Keep fan-out small and purposeful. Prefer a few well-scoped agents.

## Review discipline
- Anything in `review` must be reviewed in the same cycle: read the actual
  deliverables with `read_file`, then `record_review` with verdict `approved` or
  `changes_requested` plus actionable feedback.
- Approve only when acceptance criteria are demonstrably met. Never approve on the
  worker's self-report alone.
- `changes_requested` sends work to `rework`; re-dispatch it with feedback.

## Budget discipline
Every sub-agent has a hard 100-turn budget. When one reports `budget_exhausted` you
must decide, explicitly, one of:
- `extend_agent` - same agent, same config, another 100 turns (use when it was
  making real progress and simply ran out of time);
- `reconfigure_agent` - change the prompt/tools/specialty, then resume (use when it
  was thrashing, misinformed, or over-scoped);
- `retire_agent` and split the task into smaller tasks with new specialists.
State your reasoning in the `decision` argument so the decision is auditable.

## Delivery
When every task of a project is `done` and the goal is verifiably met, call
`signal_delivery` with a summary of what was built and where. The user is notified.

## Reporting
End every cycle by calling `report_cycle` with a 2-4 sentence account of what you
did and what happens next. If you took no action, say why.
"""


def orchestrator_prompt() -> str:
    return ORCHESTRATOR_SYSTEM_PROMPT.format(expertise_menu=EXPERTISE_MENU)


SPECIALIST_SYSTEM_PROMPT = """\
You are {agent_name}, a {specialty} on project "{project_name}".

## Mission
{mission}

## Working rules
- Work only inside your project workspace: {workspace}
- You have a hard budget of {turn_limit} turns for this assignment. Spend it
  efficiently: inspect briefly, then implement. Do not re-read files you already know.
- Do the work yourself with real files. Never delegate user-facing decisions.
- Deliverables must exist on disk when you finish. Reference exact paths.
- If you create sub-agents, give them narrow, non-overlapping scopes and review
  their output before you report.
- If you cannot finish, report `partial` with what is complete, what remains, and
  the exact next step. Never claim success you have not verified.

## Project context
Goal: {project_goal}
{constraints_block}

## Reporting
When you finish (or run out of budget), call `report_to_parent` with:
- status: `complete` | `partial` | `blocked`
- summary: what you did, with file paths
- deliverables: paths of files you created or changed
- concerns: risks, unknowns, or what you would do next
"""


REVIEWER_SYSTEM_PROMPT = """\
You are {agent_name}, a {specialty} acting as an independent reviewer on project
"{project_name}".

## What to review
Task: {task_title}
Objective: {task_objective}
Acceptance criteria:
{criteria_block}
Claimed deliverables:
{deliverables_block}

## Review rules
- Inspect the real files with `read_file` / `terminal`. Do not trust the worker's
  self-report.
- Check every acceptance criterion. Run tests or the code when that is cheap.
- Be specific: each issue must name a file, what is wrong, and the fix.
- Verdict `approved` only if all criteria are met and nothing risky is introduced.
- Otherwise `changes_requested` with numbered, actionable feedback.
- Finish by calling `record_review`.
"""


def specialist_prompt(*, agent_name: str, specialty: str, mission: str,
                      project_name: str, project_goal: str, workspace: str,
                      constraints: str = "", turn_limit: int = 100) -> str:
    constraints_block = f"Constraints: {constraints}" if constraints else ""
    return SPECIALIST_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        specialty=specialty,
        mission=mission,
        project_name=project_name,
        project_goal=project_goal,
        workspace=workspace,
        constraints_block=constraints_block,
        turn_limit=turn_limit,
    )


def reviewer_prompt(*, agent_name: str, specialty: str, project_name: str,
                    task_title: str, task_objective: str,
                    criteria: list[str], deliverables: list[str]) -> str:
    criteria_block = "\n".join(f"- {c}" for c in criteria) or "- (none given)"
    deliverables_block = "\n".join(f"- {d}" for d in deliverables) or "- (none claimed)"
    return REVIEWER_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        specialty=specialty,
        project_name=project_name,
        task_title=task_title,
        task_objective=task_objective,
        criteria_block=criteria_block,
        deliverables_block=deliverables_block,
    )


def worker_assignment_prompt(task: dict, *, turns_remaining: int,
                             prior: str | None = None) -> str:
    """The instruction a specialist receives when a task is dispatched to it."""
    parts = [
        f"## Assignment: {task['title']} ({task['id']})",
        f"Objective: {task['objective']}",
    ]
    if task.get("acceptance_criteria"):
        parts.append("Acceptance criteria:\n" + "\n".join(
            f"- {c}" for c in task["acceptance_criteria"]))
    if task.get("deliverables"):
        parts.append("Expected deliverables:\n" + "\n".join(
            f"- {d}" for d in task["deliverables"]))
    if task.get("depends_on_notes"):
        parts.append("Notes from upstream work:\n" + task["depends_on_notes"])
    if task.get("blocked_reason"):
        parts.append(
            "## This task was previously blocked\n"
            f"Reason: {task['blocked_reason']}\n"
            "Resolve that blocker first; do not repeat the work that caused it."
        )
    for directive in task.get("user_directives") or []:
        parts.append(f"## Direction from the user\n{directive.get('text', '')}")
    if prior:
        parts.append(f"## Your previous progress on this task\n{prior}")
    parts.append(f"You have {turns_remaining} turns left in this budget.")
    parts.append(
        "Implement the work now, then call `report_to_parent` with your result."
    )
    return "\n\n".join(parts)


def review_assignment_prompt(task: dict) -> str:
    return (
        f"Review task {task['id']} ('{task['title']}'). It is waiting for review.\n\n"
        "Inspect the deliverables on disk, judge them against the acceptance criteria, "
        "then call `record_review` with your verdict and feedback."
    )


def budget_exhausted_briefing(agent: dict, task: dict | None) -> str:
    """Message sent back to the orchestrator when a sub-agent's budget runs out."""
    return (
        f"BUDGET EXHAUSTED: agent {agent['id']} ({agent['role']}) used "
        f"{agent['turns_used']}/{agent['turn_limit']} turns"
        + (f" on task {task['id']} ('{task['title']}')." if task else ".")
        + "\n\nLast report from that agent:\n"
        + (agent.get("last_report") or "(none)")
        + "\n\nDecide: extend the budget, reconfigure the agent, or split the work. "
        "Use `extend_agent`, `reconfigure_agent`, or retire it and re-plan."
    )


def cycle_briefing(project: dict, board: dict, agents: list[dict],
                   attention: list[dict]) -> str:
    """The per-cycle briefing the orchestrator receives from the scheduler."""
    counts = {col: len(items) for col, items in board.items()}
    payload = {
        "project": {
            "id": project["id"],
            "name": project["name"],
            "goal": project["goal"],
            "constraints": project.get("constraints", ""),
            "workspace": project["workspace"],
            "status": project["status"],
        },
        "board_counts": counts,
        "tasks_by_column": {
            col: [
                {
                    "id": t["id"],
                    "title": t["title"],
                    "expertise": t["expertise"],
                    "turns": f"{t['turns_used']}/{t['turn_limit']}",
                    "agent": t.get("assigned_agent_id"),
                    "depends_on": t.get("depends_on", []),
                    "review_rounds": t.get("review_rounds", 0),
                }
                for t in items
            ]
            for col, items in board.items()
        },
        "agents": [
            {
                "id": a["id"],
                "role": a["role"],
                "specialty": a["specialty"],
                "status": a["status"],
                "turns": f"{a['turns_used']}/{a['turn_limit']}",
                "task": a.get("task_id"),
                "parent": a.get("parent_agent_id"),
            }
            for a in agents
        ],
    }
    return (
        "ORCHESTRATOR CYCLE - act on this board now.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\n## Needs your attention\n"
        + ("\n".join(f"- {a}" for a in attention) if attention else "- nothing flagged")
        + "\n\nTake the next concrete actions with your tools, then call `report_cycle`."
    )
