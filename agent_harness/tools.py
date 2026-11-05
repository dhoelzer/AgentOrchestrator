"""Custom tools that give the orchestrator and its specialists board control.

Every tool is registered once at import time and resolves the calling agent from
the conversation ID the SDK passes to the executor, so a single set of executors
serves every agent in the hierarchy.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field

from openhands.sdk import Action, Observation, Tool, ToolDefinition
from openhands.sdk.tool import ToolExecutor, register_tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.tools.task_tracker import TaskTrackerTool

from . import prompts
from .config import HarnessConfig
from .runtime import AgentRuntime
from .store import AgentSpec, KANBAN_COLUMNS, Store, TaskSpec, new_id, now_iso

# Canonical SDK tool names (registered by importing openhands.tools.*).
EXEC_TOOLS = ["terminal", "file_editor", "task_tracker"]


class HarnessObservation(Observation):
    """Concrete Observation subclass (the SDK base is a discriminated union)."""

ORCHESTRATOR_TOOLS = [
    "create_project",
    "list_projects",
    "create_task",
    "update_task",
    "list_tasks",
    "create_sub_agent",
    "assign_task",
    "run_agent",
    "resume_agent",
    "extend_agent",
    "reconfigure_agent",
    "retire_agent",
    "list_agents",
    "request_review",
    "record_review",
    "signal_delivery",
    "read_file",
    "report_cycle",
]

WORKER_TOOLS = [
    "report_to_parent",
    "create_sub_agent",
    "request_review",
    "read_file",
]


@dataclass
class ToolContext:
    """Shared services handed to every tool executor."""

    config: HarnessConfig
    store: Store
    runtime: AgentRuntime
    # Non-conversation agents (the orchestrator) map a stable key -> agent_id.
    keyed_agents: dict[str, str] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def agent_for_conversation(self, conversation) -> AgentSpec | None:
        """Resolve which AgentSpec is calling a tool."""
        conv_id = getattr(conversation, "id", None)
        if conv_id is None:
            return None
        agent_id = self.runtime.conversation_index.get(str(conv_id))
        return self.store.agents.get(agent_id) if agent_id else None

    def root_agent(self) -> AgentSpec | None:
        return next((a for a in self.store.agents.values()
                     if a.role == "orchestrator"), None)

    def current_agent(self, conversation) -> AgentSpec | None:
        """Caller of a tool; unrecognised conversations act as the orchestrator."""
        return self.agent_for_conversation(conversation) or self.root_agent()


CTX: ToolContext | None = None


def init_context(config: HarnessConfig, store: Store, runtime: AgentRuntime) -> ToolContext:
    global CTX
    CTX = ToolContext(config=config, store=store, runtime=runtime)
    return CTX


def ctx() -> ToolContext:
    assert CTX is not None, "Tool context not initialised; call init_context() first."
    return CTX


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _ok(text: str, **kwargs) -> Observation:
    return HarnessObservation.from_text(text, is_error=False, **kwargs)


def _err(text: str) -> Observation:
    return HarnessObservation.from_text(text, is_error=True)


def _frozen_reason(store, project_id: str | None) -> str | None:
    """Message when a project is frozen (paused/delivered), else None.

    A pause means "finish what is running, then stop". Creating or dispatching
    new work from inside a finishing agent's own run is exactly what would keep
    the project alive forever, so the tool layer refuses it rather than trusting
    the model to behave.
    """
    project = store.projects.get(project_id or "")
    if project is None or not project.is_frozen:
        return None
    return (
        f"Project '{project.name}' is {project.status}: no new work can start. "
        "Finish the task you are on and call report_to_parent; the user can "
        "resume the project to continue."
    )


def _project_inputs(store, task: TaskSpec) -> list[dict]:
    """Uploaded input files for the task's project, workspace-relative."""
    project = store.projects.get(task.project_id)
    if project is None:
        return []
    return [{"path": f"inputs/{e['name']}", "size": e.get("size", 0)}
            for e in (project.input_files or [])]


def _task_dict(task: TaskSpec) -> dict:
    return {
        "id": task.id,
        "title": task.title,
        "objective": task.objective,
        "expertise": task.expertise,
        "column": task.column,
        "depends_on": task.depends_on,
        "acceptance_criteria": task.acceptance_criteria,
        "deliverables": task.deliverables,
        "assigned_agent_id": task.assigned_agent_id,
        "turns_used": task.turns_used,
        "turn_limit": task.turn_limit,
        "review_rounds": task.review_rounds,
        "result_summary": task.result_summary,
        "review_verdict": task.review_verdict,
        "review_feedback": task.review_feedback,
        "blocked_reason": task.blocked_reason,
        "blocked_cause": task.blocked_cause,
        "block_strikes": task.block_strikes,
        "parent_task_id": task.parent_task_id,
        "orchestrator_note": task.orchestrator_note,
        "user_directives": task.user_directives,
    }


def _agent_dict(agent: AgentSpec) -> dict:
    return {
        "id": agent.id,
        "role": agent.role,
        "specialty": agent.specialty,
        "mission": agent.mission,
        "status": agent.status,
        "depth": agent.depth,
        "parent_agent_id": agent.parent_agent_id,
        "task_id": agent.task_id,
        "turns_used": agent.turns_used,
        "turn_limit": agent.turn_limit,
        "children": agent.child_agent_ids,
        "last_report": agent.last_report,
    }


def _project_workspace(project_id: str) -> str:
    proj = ctx().store.projects.get(project_id)
    return proj.workspace if proj else str(ctx().config.root_dir)


def _register(name: str, description: str, action_type, executor: ToolExecutor) -> None:
    """Register a tool under an exact name.

    ``ToolDefinition.name`` is a ClassVar derived from the class name, so the
    name must be set on the generated subclass rather than passed to ``__init__``.
    """
    namespace = {
        "name": name,
        "create": classmethod(
            lambda cls, conv_state=None, **params: [
                cls(description=description, action_type=action_type,
                    observation_type=HarnessObservation, executor=executor)
            ]
        ),
    }
    cls_name = "Harness" + "".join(p.title() for p in name.split("_")) + "Tool"
    tool_cls = type(cls_name, (ToolDefinition,), namespace)
    register_tool(name, tool_cls)


# --------------------------------------------------------------------------
# orchestrator: project + task management
# --------------------------------------------------------------------------

class CreateProjectAction(Action):
    name: str = Field(description="Short human-readable project name.")
    goal: str = Field(description="What 'done' means for this project.")
    constraints: str = Field(default="", description="Constraints, stack, do-not-touch areas.")


class CreateProjectExecutor(ToolExecutor):
    def __call__(self, action: CreateProjectAction, conversation=None) -> Observation:
        c = ctx()
        workspace = str(Path(c.config.projects_dir) / action.name.strip().replace(" ", "-")[:40])
        project = c.store.create_project(
            name=action.name, goal=action.goal,
            constraints=action.constraints, workspace=workspace,
        )
        c.store.flush()
        return _ok(
            f"Project {project.id} '{project.name}' created at {workspace} "
            f"(version {project.version}). "
            "Now decompose it into tasks with create_task."
        )


class ListProjectsAction(Action):
    pass


class ListProjectsExecutor(ToolExecutor):
    def __call__(self, action: ListProjectsAction, conversation=None) -> Observation:
        c = ctx()
        rows = [
            {
                "id": p.id, "name": p.name, "goal": p.goal, "status": p.status,
                "version": p.version, "workspace": p.workspace,
                "columns": c.store.column_counts(p.id),
            }
            for p in c.store.projects.values()
        ]
        return _ok(json.dumps(rows, indent=2) if rows else "No projects yet.")


class CreateTaskAction(Action):
    project_id: str = Field(description="Project this task belongs to.")
    title: str = Field(description="Imperative title, e.g. 'Implement rate limiter'.")
    objective: str = Field(description="One verifiable objective for this task.")
    expertise: str = Field(
        default="generalist",
        description="The specialist expertise required, e.g. 'python-engineer'.",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="Concrete, checkable criteria.")
    deliverables: list[str] = Field(
        default_factory=list, description="File paths expected to exist when done.")
    depends_on: list[str] = Field(
        default_factory=list, description="Task IDs that must be done first.")
    parent_task_id: str | None = Field(
        default=None,
        description="Set when this card is a piece of a bigger task you split. "
                    "Give it the blocked/oversized task's id so the split is "
                    "traceable on the board.")
    column: str = Field(default="backlog", description=f"One of {KANBAN_COLUMNS}.")


class CreateTaskExecutor(ToolExecutor):
    def __call__(self, action: CreateTaskAction, conversation=None) -> Observation:
        c = ctx()
        if action.project_id not in c.store.projects:
            return _err(f"Unknown project_id {action.project_id}.")
        if frozen := _frozen_reason(c.store, action.project_id):
            return _err(frozen)
        if action.column not in KANBAN_COLUMNS:
            return _err(f"Invalid column {action.column}. Use one of {KANBAN_COLUMNS}.")
        if action.parent_task_id and action.parent_task_id not in c.store.tasks:
            return _err(f"Unknown parent_task_id {action.parent_task_id}.")
        task = TaskSpec(
            project_id=action.project_id, title=action.title, objective=action.objective,
            expertise=action.expertise, acceptance_criteria=action.acceptance_criteria,
            deliverables=action.deliverables, depends_on=action.depends_on,
            parent_task_id=action.parent_task_id,
            column=action.column, turn_limit=c.config.sub_agent_turn_limit,
        )
        c.store.upsert_task(task)
        c.store.add_event(
            "task_created",
            f"Task '{task.title}' created in {task.column}."
            + (f" (split from {action.parent_task_id})" if action.parent_task_id else ""),
            project_id=task.project_id, task_id=task.id)
        c.store.revise_project(task.project_id, f"plan: task '{task.title}' added")
        return _ok(f"Task {task.id} created in column '{task.column}'.")


class UpdateTaskAction(Action):
    task_id: str
    column: str | None = Field(default=None, description=f"Move to column, one of {KANBAN_COLUMNS}.")
    title: str | None = None
    objective: str | None = None
    expertise: str | None = None
    acceptance_criteria: list[str] | None = None
    deliverables: list[str] | None = None
    depends_on: list[str] | None = None
    note: str | None = Field(default=None, description="Your reasoning for this change.")
    blocked_reason: str | None = Field(
        default=None,
        description="When moving to 'blocked', what is stopping the task. Required "
                    "for a manual block so the next cycle knows what to resolve.",
    )


class UpdateTaskExecutor(ToolExecutor):
    def __call__(self, action: UpdateTaskAction, conversation=None) -> Observation:
        c = ctx()
        task = c.store.tasks.get(action.task_id)
        if task is None:
            return _err(f"Unknown task_id {action.task_id}.")
        if action.column is not None:
            if action.column not in KANBAN_COLUMNS:
                return _err(f"Invalid column {action.column}.")
            if action.column == "blocked" and not (
                action.blocked_reason or task.blocked_reason
            ):
                return _err(
                    "Blocking a task requires blocked_reason so the next cycle can "
                    "act on it. Say what is stopping the task."
                )
            task.column = action.column
            if action.column != "blocked":
                task.blocked_reason = None
                task.blocked_cause = None
        if action.blocked_reason:
            task.blocked_reason = action.blocked_reason
            # A manual block from the orchestrator always needs judgement.
            task.blocked_cause = "manual"
        for attr in ("title", "objective", "expertise"):
            value = getattr(action, attr)
            if value is not None:
                setattr(task, attr, value)
        for attr in ("acceptance_criteria", "deliverables", "depends_on"):
            value = getattr(action, attr)
            if value is not None:
                setattr(task, attr, value)
        if action.note:
            task.orchestrator_note = action.note
        content_changed = any(
            getattr(action, attr) is not None
            for attr in ("title", "objective", "expertise",
                         "acceptance_criteria", "deliverables", "depends_on")
        )
        task.touch()
        c.store.add_event("task_updated", f"Task '{task.title}' -> {task.column}.",
                          project_id=task.project_id, task_id=task.id)
        if content_changed:
            c.store.revise_project(task.project_id, f"plan: task '{task.title}' redefined")
        else:
            c.store.flush()
        return _ok(f"Task {task.id} updated; now in '{task.column}'.")


class ListTasksAction(Action):
    project_id: str
    column: str | None = Field(default=None, description="Restrict to one column.")


class ListTasksExecutor(ToolExecutor):
    def __call__(self, action: ListTasksAction, conversation=None) -> Observation:
        c = ctx()
        tasks = c.store.project_tasks(action.project_id)
        if action.column:
            tasks = [t for t in tasks if t.column == action.column]
        return _ok(json.dumps([_task_dict(t) for t in tasks], indent=2))


# --------------------------------------------------------------------------
# dynamic specialist creation
# --------------------------------------------------------------------------

class CreateSubAgentAction(Action):
    role: str = Field(description="Short label, e.g. 'python-engineer'.")
    specialty: str = Field(
        description="The expertise this agent embodies, e.g. 'async Python performance'.")
    mission: str = Field(
        description="What this agent is accountable for, in one or two sentences.")
    project_id: str = Field(description="Project the agent works on.")
    system_prompt: str | None = Field(
        default=None,
        description="Optional full replacement prompt. Omit to use the harness "
                    "specialist template with your mission filled in.",
    )
    tools: list[str] | None = Field(
        default=None,
        description=f"Tool names. Default: {EXEC_TOOLS + WORKER_TOOLS}.",
    )
    turn_limit: int | None = Field(
        default=None, description="Turn budget for this agent (default 100).")


class CreateSubAgentExecutor(ToolExecutor):
    def __call__(self, action: CreateSubAgentAction, conversation=None) -> Observation:
        c = ctx()
        caller = c.current_agent(conversation)
        project = c.store.projects.get(action.project_id)
        if project is None:
            return _err(f"Unknown project_id {action.project_id}.")
        if project.is_frozen:
            return _err(_frozen_reason(c.store, project.id))

        depth = (caller.depth + 1) if caller else 0
        if caller is not None and depth > c.config.max_sub_agent_depth:
            return _err(
                f"Sub-agent depth limit ({c.config.max_sub_agent_depth}) reached. "
                "Report back instead of delegating deeper."
            )
        # 0 means unlimited, which is the default: there is no cap on how many
        # specialists an agent may create. The real ceiling is how many may run
        # at once, enforced by the project concurrency limit at dispatch time.
        cap = c.config.max_children_per_agent
        if caller is not None and cap and len(caller.child_agent_ids) >= cap:
            return _err(
                f"You already have {len(caller.child_agent_ids)} sub-agents "
                f"(limit {cap}). Reuse one with resume_agent."
            )

        tools = action.tools or (EXEC_TOOLS + WORKER_TOOLS)
        invalid = [t for t in tools if t not in EXEC_TOOLS + WORKER_TOOLS + ORCHESTRATOR_TOOLS]
        if invalid:
            return _err(f"Unknown tools: {invalid}.")

        agent_id = new_id("agent")
        prompt = action.system_prompt or prompts.specialist_prompt(
            agent_name=f"{action.role}-{agent_id}",
            specialty=action.specialty,
            mission=action.mission,
            project_name=project.name,
            project_goal=project.goal,
            workspace=project.workspace,
            constraints=project.constraints,
            turn_limit=action.turn_limit or c.config.sub_agent_turn_limit,
        )
        spec = AgentSpec(
            id=agent_id, project_id=project.id, role=action.role,
            specialty=action.specialty, mission=action.mission, system_prompt=prompt,
            parent_agent_id=caller.id if caller else None,
            depth=depth, workspace=project.workspace,
            turn_limit=action.turn_limit or c.config.sub_agent_turn_limit,
            created_by=caller.id if caller else "orchestrator",
        )
        with c.lock:
            c.store.upsert_agent(spec)
            if caller is not None:
                caller.child_agent_ids.append(agent_id)
                caller.touch()
                c.store.upsert_agent(caller)
        c.store.add_event(
            "agent_created",
            f"Created specialist '{spec.role}' ({spec.id}) for project {project.name}.",
            project_id=project.id, agent_id=agent_id,
        )
        c.store.flush()
        return _ok(
            f"Sub-agent {agent_id} created (depth {depth}, budget {spec.turn_limit} turns). "
            f"Assign work with assign_task(task_id=..., agent_id='{agent_id}')."
        )


class AssignTaskAction(Action):
    task_id: str
    agent_id: str


class AssignTaskExecutor(ToolExecutor):
    def __call__(self, action: AssignTaskAction, conversation=None) -> Observation:
        c = ctx()
        task = c.store.tasks.get(action.task_id)
        agent = c.store.agents.get(action.agent_id)
        if task is None:
            return _err(f"Unknown task_id {action.task_id}.")
        if agent is None:
            return _err(f"Unknown agent_id {action.agent_id}.")
        if frozen := _frozen_reason(c.store, task.project_id):
            return _err(frozen)
        # A task already in 'review' has finished its worker stretch, so an
        # agent attached to it now is a reviewer; it must not take the worker
        # link, which is what drives the card's turn counters.
        if agent.is_reviewer or task.column == "review":
            _reactivate(c, agent)
            agent.is_reviewer = True
            task.reviewer_agent_id = agent.id
            agent.task_id = task.id
            agent.touch()
            task.touch()
            c.store.flush()
            return _ok(
                f"{agent.role} ({agent.id}) attached as reviewer of {task.id}. "
                "Run it with run_agent(agent_id='" + agent.id + "', task_id='"
                + task.id + "', review=true)."
            )
        _reactivate(c, agent)
        task.assigned_agent_id = agent.id
        task.touch()
        agent.task_id = task.id
        agent.touch()
        c.store.flush()
        return _ok(
            f"Task {task.id} assigned to {agent.role} ({agent.id}). "
            f"Run it now with run_agent(agent_id='{agent.id}', task_id='{task.id}')."
        )


class RunAgentAction(Action):
    agent_id: str
    task_id: str | None = Field(
        default=None, description="Task to dispatch. Defaults to the agent's assigned task.")
    instructions: str | None = Field(
        default=None, description="Extra instructions appended to the assignment.")
    review: bool = Field(
        default=False, description="Set true to run this agent as an independent reviewer.")
    background: bool = Field(
        default=True,
        description="Run the agent in the background so you can dispatch other work "
                    "this cycle. Set false only for short, must-finish-now work.",
    )


class RunAgentExecutor(ToolExecutor):
    def __call__(self, action: RunAgentAction, conversation=None) -> Observation:
        c = ctx()
        agent = c.store.agents.get(action.agent_id)
        if agent is None:
            return _err(f"Unknown agent_id {action.agent_id}.")
        task = c.store.tasks.get(action.task_id or agent.task_id or "")
        if task is None:
            return _err("No task to run: pass task_id or assign one first.")
        if frozen := _frozen_reason(c.store, task.project_id):
            return _err(frozen)

        remaining = agent.turn_limit - agent.turns_used
        if remaining <= 0:
            return _err(
                f"Agent {agent.id} has no turns left ({agent.turns_used}/{agent.turn_limit}). "
                "Use extend_agent or reconfigure_agent first."
            )
        if agent.id in c.runtime.inflight:
            return _err(f"Agent {agent.id} is already running; wait for it next cycle.")
        if not c.runtime.project_slot_free(task.project_id or agent.project_id):
            return _err(
                f"Project already has "
                f"{c.config.max_concurrent_sub_agents_per_project} sub-agents "
                f"running (its concurrency limit). There is no limit on how many "
                "agents you may create - they just cannot all run at once. "
                "Dispatch in the background and the scheduler starts it as soon "
                "as a slot frees, or work a card in another project."
            )

        _reactivate(c, agent)
        inputs = _project_inputs(c.store, task)
        if action.review:
            message = prompts.review_assignment_prompt(_task_dict(task),
                                                       input_files=inputs)
        else:
            message = prompts.worker_assignment_prompt(
                _task_dict(task), turns_remaining=remaining,
                prior=agent.last_report, input_files=inputs
            )
        if action.instructions:
            message += "\n\n## Additional instructions\n" + action.instructions

        if action.background:
            try:
                c.runtime.submit(
                    agent.id,
                    lambda: _dispatch(agent, task, message, remaining,
                                      review=action.review))
            except RuntimeError as exc:
                return _err(str(exc))
            return _ok(
                f"Dispatched {agent.role} ({agent.id}) in the BACKGROUND on task "
                f"{task.id} with {remaining} turns. It moves to 'review' when done; "
                "dispatch other ready work now."
            )
        return _dispatch(agent, task, message, remaining, review=action.review)


class ResumeAgentAction(Action):
    agent_id: str
    instructions: str = Field(description="What to do on this continuation.")
    task_id: str | None = None


class ResumeAgentExecutor(ToolExecutor):
    def __call__(self, action: ResumeAgentAction, conversation=None) -> Observation:
        c = ctx()
        agent = c.store.agents.get(action.agent_id)
        if agent is None:
            return _err(f"Unknown agent_id {action.agent_id}.")
        task = c.store.tasks.get(action.task_id or agent.task_id or "")
        if frozen := _frozen_reason(
                c.store, getattr(task, "project_id", None) or agent.project_id):
            return _err(frozen)
        remaining = agent.turn_limit - agent.turns_used
        if remaining <= 0:
            return _err(
                f"Agent {agent.id} has no turns left. Use extend_agent to grant more."
            )
        _reactivate(c, agent)
        c.store.flush()
        return _dispatch(agent, task, action.instructions, remaining)


def _reactivate(c, agent: AgentSpec) -> None:
    """Put an archived agent back on the live hierarchy.

    Assigning to or dispatching an archived specialist is an explicit decision
    to use it again, so it has to reappear on the tree; otherwise the UI shows
    an agent working while the hierarchy says it is gone.
    """
    if agent.archived_at is None:
        return
    agent.archived_at = None
    agent.retirement_reason = None
    if agent.status == "done":
        agent.status = "idle"
    if agent.parent_agent_id and agent.parent_agent_id in c.store.agents:
        parent = c.store.agents[agent.parent_agent_id]
        if agent.id not in parent.child_agent_ids:
            parent.child_agent_ids.append(agent.id)
            c.store.upsert_agent(parent)


def _dispatch(agent: AgentSpec, task: TaskSpec | None, message: str,
              remaining: int, review: bool = False) -> Observation:
    """Run an agent synchronously and reflect the outcome on the board."""
    c = ctx()
    if task is not None:
        # Keep the task<->agent links so budget exhaustion is always attributable.
        # A reviewer must not take over the worker link: the card's turn counters
        # follow the worker, and the store syncs from the assigned agent only.
        if review:
            task.reviewer_agent_id = agent.id
            if not agent.is_reviewer:
                agent.is_reviewer = True
                c.store.upsert_agent(agent)
            if task.assigned_agent_id == agent.id:
                # The orchestrator assigned this reviewer as the worker before
                # dispatching it as a reviewer; hand the card back to the real
                # worker so its counters stay attributable.
                worker = next(
                    (a for a in c.store.project_agents(agent.project_id or "")
                     if a.id != agent.id and a.task_id == task.id
                     and not a.is_reviewer),
                    None,
                )
                task.assigned_agent_id = worker.id if worker else None
        else:
            task.assigned_agent_id = agent.id
            # The orchestrator can dispatch with an explicit task_id without
            # ever calling assign_task; without the reverse link the live turn
            # visualizer and the store sync have nothing to write through.
            if agent.task_id != task.id:
                agent.task_id = task.id
                c.store.upsert_agent(agent)
    # Directives are injected into this dispatch's prompt, so they are consumed
    # here; leaving them queued would replay stale instructions forever.
    if task is not None and task.user_directives:
        task.user_directives = []
    if task is not None and task.column in ("backlog", "ready"):
        task.column = "in_progress"
        task.started_at = task.started_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds")
        task.touch()

    tool_names = agent.tools_override or (EXEC_TOOLS + WORKER_TOOLS)
    outcome = c.runtime.run(agent, [Tool(name=t) for t in tool_names], message,
                            turn_limit=agent.turn_limit)

    if task is not None:
        if not review:
            task.turns_used = agent.turns_used
        task.runs += 1
        task.result_summary = outcome.final_report or task.result_summary
        if outcome.budget_exhausted:
            task.column = "blocked"
            task.blocked_reason = (
                f"agent {agent.id} used all {agent.turn_limit} turns"
            )
            task.blocked_cause = "budget"
        elif outcome.status == "llm_unavailable":
            task.column = "blocked"
            task.blocked_reason = (
                "LLM endpoint returned no output; request timeout is now "
                f"{c.runtime.llm_timeout}s"
            )
            task.blocked_cause = "llm"
        elif outcome.status == "failed" or outcome.error:
            task.column = "blocked"
            task.blocked_cause = "agent_failed"
            task.blocked_reason = (
                f"agent {agent.id} failed: {outcome.error or outcome.status}"
            )
        elif not review and task.column in ("blocked", "rework", "in_progress"):
            # A run that did not fail or run out of budget means whatever was
            # holding the card (a block, review feedback) is no longer holding
            # it. Move it forward and clear the stale reason; leaving a card in
            # 'blocked' after a clean run is what makes the board lie. Reviewer
            # dispatches are excluded: their verdict moves the card via
            # record_review.
            task.column = "review"
            if task.blocked_reason:
                c.store.add_event(
                    "task_unblocked",
                    f"Task '{task.title}' ran clean; cleared block "
                    f"'{task.blocked_reason}'.",
                    project_id=task.project_id, task_id=task.id,
                    agent_id=agent.id,
                )
            task.blocked_reason = None
            task.blocked_cause = None
        task.touch()

    c.store.add_event(
        "agent_dispatched",
        f"{agent.role} ran task {task.id if task else '-'}: "
        f"status={outcome.status}, turns={agent.turns_used}/{agent.turn_limit}.",
        project_id=agent.project_id, agent_id=agent.id,
        task_id=task.id if task else None,
    )
    c.store.flush()

    if outcome.budget_exhausted:
        return _ok(prompts.budget_exhausted_briefing(_agent_dict(agent),
                                                    _task_dict(task) if task else None))
    text = outcome.final_report or "(no report returned)"
    if outcome.error:
        return _err(f"Agent run errored: {outcome.error}\n\nPartial report:\n{text}")
    return _ok(
        f"Agent {agent.id} finished ({outcome.status}), used {agent.turns_used}/"
        f"{agent.turn_limit} turns.\n\nREPORT:\n{text}"
    )


class ExtendAgentAction(Action):
    agent_id: str
    extra_turns: int = Field(default=100, description="Turns to add to the budget.")
    decision: str = Field(description="Why extending is the right call.")


class ExtendAgentExecutor(ToolExecutor):
    def __call__(self, action: ExtendAgentAction, conversation=None) -> Observation:
        c = ctx()
        agent = c.store.agents.get(action.agent_id)
        if agent is None:
            return _err(f"Unknown agent_id {action.agent_id}.")
        agent.turn_limit += action.extra_turns
        agent.status = "idle"
        agent.touch()
        if agent.task_id and agent.task_id in c.store.tasks:
            task = c.store.tasks[agent.task_id]
            task.turn_limit = agent.turn_limit
            # Granting turns resolves a budget block, so the stale reason goes
            # with it; any other cause still stands.
            if task.column == "blocked" and task.blocked_cause == "budget":
                task.column = "ready"
                task.blocked_reason = None
                task.blocked_cause = None
            task.touch()
        c.store.add_event(
            "budget_extended",
            f"Extended {agent.role} by {action.extra_turns} turns. Reason: {action.decision}",
            project_id=agent.project_id, agent_id=agent.id,
        )
        c.store.flush()
        return _ok(
            f"Budget for {agent.id} is now {agent.turn_limit} turns "
            f"(used {agent.turns_used}). Resume it with resume_agent or run_agent."
        )


class ReconfigureAgentAction(Action):
    agent_id: str
    decision: str = Field(description="What was wrong and why this fixes it.")
    system_prompt: str | None = None
    specialty: str | None = None
    mission: str | None = None
    tools: list[str] | None = None
    turn_limit: int | None = Field(default=None, description="New total turn budget.")
    fresh_conversation: bool = Field(
        default=True,
        description="Start the agent in a new conversation so the new prompt and "
                    "tool set actually take effect. Its prior report is carried over.",
    )


class ReconfigureAgentExecutor(ToolExecutor):
    def __call__(self, action: ReconfigureAgentAction, conversation=None) -> Observation:
        c = ctx()
        agent = c.store.agents.get(action.agent_id)
        if agent is None:
            return _err(f"Unknown agent_id {action.agent_id}.")
        if agent.id in c.runtime.inflight:
            return _err(f"Agent {agent.id} is running; retire or wait before reconfiguring.")

        changed_config = any(
            v is not None
            for v in (action.system_prompt, action.specialty, action.mission, action.tools)
        )
        if action.system_prompt:
            agent.system_prompt = action.system_prompt
        if action.specialty:
            agent.specialty = action.specialty
        if action.mission:
            agent.mission = action.mission
        if action.tools:
            invalid = [t for t in action.tools
                       if t not in EXEC_TOOLS + WORKER_TOOLS + ORCHESTRATOR_TOOLS]
            if invalid:
                return _err(f"Unknown tools: {invalid}.")
            agent.tools_override = action.tools
        if action.turn_limit:
            agent.turn_limit = action.turn_limit

        if changed_config and action.fresh_conversation:
            c.runtime.reset_conversation(agent)
        else:
            agent.turns_used = 0
            agent.status = "idle"
            agent.touch()
        if agent.task_id and agent.task_id in c.store.tasks:
            task = c.store.tasks[agent.task_id]
            # Only the worker drives the card's counters; reconfiguring a
            # reviewer would otherwise reset the card to 0/new-limit.
            if not agent.is_reviewer and task.assigned_agent_id == agent.id:
                task.turns_used = agent.turns_used
                task.turn_limit = agent.turn_limit
            # Reconfiguring the agent resolves blocks that were about the agent
            # itself (budget, LLM, its own report); a dependency or user-input
            # block still stands, so its reason stays visible.
            if task.column == "blocked" and task.blocked_cause in (
                    "budget", "llm", "worker"):
                task.column = "ready"
                task.blocked_reason = None
                task.blocked_cause = None
            task.touch()

        c.store.add_event(
            "agent_reconfigured",
            f"Reconfigured {agent.role}: {action.decision}",
            project_id=agent.project_id, agent_id=agent.id,
        )
        c.store.flush()
        return _ok(
            f"Agent {agent.id} reconfigured; budget {agent.turn_limit} turns "
            f"(used {agent.turns_used}). Resume it with resume_agent."
        )


class RetireAgentAction(Action):
    agent_id: str
    reason: str


class RetireAgentExecutor(ToolExecutor):
    def __call__(self, action: RetireAgentAction, conversation=None) -> Observation:
        c = ctx()
        agent = c.store.agents.get(action.agent_id)
        if agent is None:
            return _err(f"Unknown agent_id {action.agent_id}.")
        c.runtime.close(agent.id)
        agent.status = "done"
        agent.task_id = None
        # Archived rather than deleted: cost, tokens, and the transcript stay
        # queryable, but the agent leaves the live hierarchy so the tree does
        # not fill up with finished specialists. resume_agent un-archives it.
        agent.archived_at = now_iso()
        agent.retirement_reason = action.reason
        agent.touch()
        if agent.parent_agent_id and agent.parent_agent_id in c.store.agents:
            parent = c.store.agents[agent.parent_agent_id]
            parent.child_agent_ids = [a for a in parent.child_agent_ids if a != agent.id]
            c.store.upsert_agent(parent)
        c.store.add_event("agent_retired", f"Retired {agent.role}: {action.reason}",
                          project_id=agent.project_id, agent_id=agent.id)
        c.store.flush()
        return _ok(f"Agent {agent.id} retired and removed from the hierarchy. "
                   "Use resume_agent to bring it back.")


class ListAgentsAction(Action):
    project_id: str | None = None


class ListAgentsExecutor(ToolExecutor):
    def __call__(self, action: ListAgentsAction, conversation=None) -> Observation:
        c = ctx()
        agents = (c.store.project_agents(action.project_id) if action.project_id
                  else list(c.store.agents.values()))
        return _ok(json.dumps([_agent_dict(a) for a in agents], indent=2))


# --------------------------------------------------------------------------
# review coordination
# --------------------------------------------------------------------------

class RequestReviewAction(Action):
    task_id: str
    reviewer_expertise: str | None = Field(
        default=None, description="Expertise the reviewer should have; defaults to "
                                  "'code-reviewer' plus the task's own expertise.")


class RequestReviewExecutor(ToolExecutor):
    def __call__(self, action: RequestReviewAction, conversation=None) -> Observation:
        c = ctx()
        task = c.store.tasks.get(action.task_id)
        if task is None:
            return _err(f"Unknown task_id {action.task_id}.")
        if frozen := _frozen_reason(c.store, task.project_id):
            return _err(frozen)
        task.column = "review"
        task.touch()
        caller = c.current_agent(conversation)
        if caller:
            task.assigned_agent_id = task.assigned_agent_id or caller.id
        c.store.add_event("review_requested", f"Review requested for '{task.title}'.",
                          project_id=task.project_id, task_id=task.id)
        c.store.flush()
        expertise = action.reviewer_expertise or f"reviewer-{task.expertise}"
        return _ok(
            f"Task {task.id} moved to 'review'. Coordinate the review: create a "
            f"'{expertise}' agent with the reviewer mission, then run it with "
            f"review=true on task {task.id}."
        )


class RecordReviewAction(Action):
    task_id: str
    verdict: str = Field(description="'approved' or 'changes_requested'.")
    feedback: str = Field(description="Specific, actionable findings with file paths.")
    criteria_met: list[str] = Field(default_factory=list)
    criteria_failed: list[str] = Field(default_factory=list)


class RecordReviewExecutor(ToolExecutor):
    def __call__(self, action: RecordReviewAction, conversation=None) -> Observation:
        c = ctx()
        task = c.store.tasks.get(action.task_id)
        if task is None:
            return _err(f"Unknown task_id {action.task_id}.")
        if action.verdict not in ("approved", "changes_requested"):
            return _err("verdict must be 'approved' or 'changes_requested'.")

        reviewer = c.current_agent(conversation)
        task.reviewer_agent_id = reviewer.id if reviewer else task.reviewer_agent_id
        task.review_verdict = action.verdict
        task.review_feedback = action.feedback
        task.review_rounds += 1

        if action.verdict == "approved":
            task.column = "done"
            task.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            # A card that just landed cannot also be blocked; leaving the old
            # reason behind makes a done card advertise a blocker it resolved.
            task.blocked_reason = None
            task.blocked_cause = None
        else:
            task.column = "rework"
        task.touch()
        c.store.add_event(
            "review_recorded",
            f"Review of '{task.title}': {action.verdict} (round {task.review_rounds}).",
            project_id=task.project_id, task_id=task.id,
            agent_id=reviewer.id if reviewer else None,
        )
        c.store.revise_project(
            task.project_id,
            f"review: '{task.title}' {action.verdict} (round {task.review_rounds})",
        )
        return _ok(
            f"Review recorded. Task {task.id} is now '{task.column}'. "
            + ("Nothing further needed for this task."
               if task.column == "done"
               else "Re-dispatch it with the feedback, or escalate if rounds keep failing.")
        )


# --------------------------------------------------------------------------
# delivery + reporting
# --------------------------------------------------------------------------

class SignalDeliveryAction(Action):
    project_id: str
    summary: str = Field(description="What was delivered and where to find it.")
    evidence: list[str] = Field(default_factory=list,
                                description="Paths / commands that prove it works.")


class SignalDeliveryExecutor(ToolExecutor):
    def __call__(self, action: SignalDeliveryAction, conversation=None) -> Observation:
        c = ctx()
        project = c.store.projects.get(action.project_id)
        if project is None:
            return _err(f"Unknown project_id {action.project_id}.")
        if project.status == "paused":
            return _err(_frozen_reason(c.store, project.id))
        open_tasks = [t for t in c.store.project_tasks(project.id) if t.column != "done"]
        if open_tasks:
            return _err(
                "Cannot deliver: " + ", ".join(f"{t.id}({t.column})" for t in open_tasks)
            )
        was_ready = (project.status == "ready_for_delivery"
                     and project.delivery_signaled)
        project.status = "ready_for_delivery"
        project.delivery_signaled = True
        project.ready_at = now_iso()
        project.ready_generation += 1
        project.touch()
        # The orchestrator may already have auto-detected readiness this cycle;
        # only count a revision when the signal itself is new information.
        if not was_ready:
            c.store.revise_project(project.id,
                                   f"delivery signaled: {action.summary[:120]}")
        c.store.add_event("delivery_signaled",
                          f"PROJECT READY FOR DELIVERY: {project.name} - {action.summary}",
                          project_id=project.id)
        c.store.flush()
        return _ok(f"Delivery signaled for {project.id}; the user has been notified.")


class ReportCycleAction(Action):
    summary: str = Field(description="2-4 sentences: what you did and what happens next.")
    needs_user: str | None = Field(
        default=None, description="Question or decision only the user can resolve.")


class ReportCycleExecutor(ToolExecutor):
    def __call__(self, action: ReportCycleAction, conversation=None) -> Observation:
        c = ctx()
        c.store.add_event("orchestrator_report", action.summary)
        if action.needs_user:
            c.store.add_event("needs_user", f"Needs the user: {action.needs_user}")
        c.store.flush()
        return _ok("Cycle report recorded.")


class ReportToParentAction(Action):
    status: str = Field(description="'complete', 'partial', or 'blocked'.")
    summary: str = Field(description="What you did, with file paths.")
    deliverables: list[str] = Field(default_factory=list,
                                    description="Files you created or changed.")
    concerns: str | None = Field(default=None, description="Risks, unknowns, next steps.")
    blocker: str | None = Field(
        default=None,
        description="If status is 'blocked': what is blocking you NOW, and what would "
                    "unblock it. If you resolved the blocker you were dispatched with, "
                    "leave this empty and report 'complete' or 'partial' so the board "
                    "clears the stale block.")


class ReportToParentExecutor(ToolExecutor):
    def __call__(self, action: ReportToParentAction, conversation=None) -> Observation:
        c = ctx()
        agent = c.current_agent(conversation)
        if agent is None:
            return _err("Could not identify the reporting agent.")
        agent.last_report = (
            f"[{action.status}] {action.summary}"
            + (f"\nDeliverables: {', '.join(action.deliverables)}" if action.deliverables else "")
            + (f"\nConcerns: {action.concerns}" if action.concerns else "")
        )
        agent.touch()
        if agent.task_id and agent.task_id in c.store.tasks:
            task = c.store.tasks[agent.task_id]
            task.result_summary = agent.last_report
            if action.deliverables:
                task.deliverables = list(dict.fromkeys(task.deliverables + action.deliverables))
            # The worker on the card is the authority on whether it is still
            # blocked: it just checked. Restate a new reason, or clear the old
            # one once the blocker is confirmed gone.
            if action.status == "blocked":
                task.column = "blocked"
                task.blocked_reason = action.blocker or (
                    f"{agent.role} reported blocked without a reason"
                )
                task.blocked_cause = "worker"
            elif task.column == "blocked":
                c.store.add_event(
                    "task_unblocked",
                    f"Task '{task.title}' confirmed unblocked by {agent.role} "
                    f"(was blocked: {task.blocked_reason or 'no reason recorded'}).",
                    project_id=task.project_id, task_id=task.id, agent_id=agent.id,
                )
                task.column = "in_progress"
                task.blocked_reason = None
                task.blocked_cause = None
            task.touch()
            c.store.upsert_task(task)
        c.store.add_event(
            "agent_report", f"{agent.role} reported {action.status}: {action.summary[:400]}",
            project_id=agent.project_id, agent_id=agent.id, task_id=agent.task_id,
        )
        c.store.flush()
        if action.status == "blocked":
            return _ok(
                "Report filed and the card is marked blocked with your reason. "
                "Now call `finish` with the same summary so the orchestrator can act."
            )
        return _ok(
            "Report filed. Now call `finish` with the same summary - your turn budget "
            "is reserved for implementation, not re-explaining."
        )


class ReadFileAction(Action):
    path: str = Field(description="Absolute path, or path relative to the project workspace.")
    max_lines: int = Field(default=200, description="Lines to return from the top.")


class ReadFileExecutor(ToolExecutor):
    def __call__(self, action: ReadFileAction, conversation=None) -> Observation:
        c = ctx()
        agent = c.current_agent(conversation)
        base = Path(agent.workspace).resolve() if (agent and agent.workspace) else c.config.root_dir
        path = Path(action.path)
        if not path.is_absolute():
            path = (base / path).resolve()
        if c.config.sandbox_workspaces and agent and agent.depth > 0:
            try:
                path.relative_to(base)
            except ValueError:
                return _err(f"Path {path} is outside your workspace {base}.")
        if not path.exists():
            return _err(f"No such file: {path}")
        if path.is_dir():
            entries = sorted(p.name for p in path.iterdir())[:200]
            return _ok(f"{path} is a directory:\n" + "\n".join(entries))
        lines = path.read_text(errors="replace").splitlines()[: action.max_lines]
        return _ok(f"{path} ({len(lines)} lines shown)\n" + "\n".join(lines))


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

_REGISTERED = False


def register_harness_tools() -> None:
    """Register every harness tool exactly once per process."""
    global _REGISTERED
    if _REGISTERED:
        return

    # Importing these registers the canonical SDK exec tools.
    TerminalTool, FileEditorTool, TaskTrackerTool  # noqa: F841

    pairs: list[tuple[str, str, type[Action], ToolExecutor]] = [
        ("create_project", "Create a new project with a goal and workspace.",
         CreateProjectAction, CreateProjectExecutor()),
        ("list_projects", "List all projects with board counts.",
         ListProjectsAction, ListProjectsExecutor()),
        ("create_task", "Add a task to a project board.",
         CreateTaskAction, CreateTaskExecutor()),
        ("update_task", "Change or move an existing task.",
         UpdateTaskAction, UpdateTaskExecutor()),
        ("list_tasks", "List tasks for a project, optionally one column.",
         ListTasksAction, ListTasksExecutor()),
        ("create_sub_agent", "Dynamically create a specialist sub-agent for a task.",
         CreateSubAgentAction, CreateSubAgentExecutor()),
        ("assign_task", "Assign a task to an agent.",
         AssignTaskAction, AssignTaskExecutor()),
        ("run_agent", "Dispatch a task to an agent now (uses its turn budget).",
         RunAgentAction, RunAgentExecutor()),
        ("resume_agent", "Continue an existing agent with new instructions.",
         ResumeAgentAction, ResumeAgentExecutor()),
        ("extend_agent", "Grant an agent another turn budget.",
         ExtendAgentAction, ExtendAgentExecutor()),
        ("reconfigure_agent", "Change an agent's prompt/tools/budget and reset it.",
         ReconfigureAgentAction, ReconfigureAgentExecutor()),
        ("retire_agent", "Retire an agent that is no longer needed.",
         RetireAgentAction, RetireAgentExecutor()),
        ("list_agents", "List agents and their statuses.",
         ListAgentsAction, ListAgentsExecutor()),
        ("request_review", "Move a task to review and name the reviewer expertise.",
         RequestReviewAction, RequestReviewExecutor()),
        ("record_review", "Record a review verdict for a task.",
         RecordReviewAction, RecordReviewExecutor()),
        ("signal_delivery", "Signal a project is ready to deliver to the user.",
         SignalDeliveryAction, SignalDeliveryExecutor()),
        ("report_cycle", "Summarise what you did this cycle.",
         ReportCycleAction, ReportCycleExecutor()),
        ("report_to_parent", "File your result with your parent/orchestrator.",
         ReportToParentAction, ReportToParentExecutor()),
        ("read_file", "Read a file or list a directory in your workspace.",
         ReadFileAction, ReadFileExecutor()),
    ]
    for name, description, action_type, executor in pairs:
        _register(name, description, action_type, executor)
    _REGISTERED = True
