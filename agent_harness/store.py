"""Durable state for the harness: projects, tasks, agents, and an event log.

Everything the kanban board renders lives here, so the web UI and the
orchestrator loop share one source of truth that survives restarts.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

KANBAN_COLUMNS = [
    "backlog",
    "ready",
    "in_progress",
    "review",
    "rework",
    "done",
    "blocked",
]

Column = Literal[
    "backlog", "ready", "in_progress", "review", "rework", "done", "blocked"
]
AgentStatus = Literal[
    "idle", "running", "waiting_budget", "budget_exhausted", "paused", "failed", "done"
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TaskSpec(BaseModel):
    """A unit of work on the board."""

    id: str = Field(default_factory=lambda: new_id("task"))
    project_id: str
    parent_task_id: str | None = None
    title: str
    objective: str
    expertise: str = Field(default="generalist")
    column: Column = Field(default="backlog")
    depends_on: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)

    assigned_agent_id: str | None = None
    reviewer_agent_id: str | None = None

    turns_used: int = 0
    turn_limit: int = 100
    runs: int = 0
    review_rounds: int = 0

    # Why the task sits in 'blocked', and any direction the user has supplied
    # since. Directives are queued so the orchestrator cannot miss one that
    # arrives mid-cycle.
    blocked_reason: str | None = None
    user_directives: list[dict] = Field(default_factory=list)

    result_summary: str | None = None
    review_verdict: str | None = None
    review_feedback: str | None = None
    orchestrator_note: str | None = None

    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    started_at: str | None = None
    finished_at: str | None = None

    def touch(self) -> None:
        self.updated_at = now_iso()


class AgentSpec(BaseModel):
    """A specialist agent in the hierarchy (orchestrator is the root)."""

    id: str = Field(default_factory=lambda: new_id("agent"))
    project_id: str | None = None
    parent_agent_id: str | None = None
    child_agent_ids: list[str] = Field(default_factory=list)

    role: str
    specialty: str
    mission: str
    system_prompt: str
    status: AgentStatus = "idle"
    depth: int = 0

    conversation_id: str | None = None
    persistence_dir: str | None = None
    workspace: str | None = None
    tools_override: list[str] | None = None

    task_id: str | None = None
    created_by: str = "orchestrator"
    # Set when the agent was dispatched with review=true. Reviewers never drive
    # a task card's turn counters, even if they also carry task_id.
    is_reviewer: bool = False

    turns_used: int = 0
    turn_limit: int = 100
    runs: int = 0
    tokens: int = 0
    cost_usd: float = 0.0

    last_report: str | None = None
    last_error: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    def touch(self) -> None:
        self.updated_at = now_iso()


class Project(BaseModel):
    """A long-running project owned by the orchestrator."""

    id: str = Field(default_factory=lambda: new_id("proj"))
    name: str
    goal: str
    constraints: str = ""
    workspace: str
    status: Literal["active", "ready_for_delivery", "delivered", "paused"] = "active"
    delivery_signaled: bool = False
    delivered_at: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    def touch(self) -> None:
        self.updated_at = now_iso()


class HarnessEvent(BaseModel):
    """Append-only log line surfaced in the UI activity feed."""

    id: str = Field(default_factory=lambda: new_id("ev"))
    at: str = Field(default_factory=now_iso)
    kind: str
    project_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    message: str


class Store:
    """Thread-safe JSON-backed store for all harness state."""

    def __init__(self, data_dir: str | os.PathLike[str]):
        self.root = Path(data_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self._lock = threading.RLock()

        self.projects: dict[str, Project] = {}
        self.tasks: dict[str, TaskSpec] = {}
        self.agents: dict[str, AgentSpec] = {}
        self.recent_events: list[HarnessEvent] = []
        self.meta: dict[str, Any] = {}
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        self.projects = {p["id"]: Project.model_validate(p) for p in data.get("projects", [])}
        self.tasks = {t["id"]: TaskSpec.model_validate(t) for t in data.get("tasks", [])}
        self.agents = {a["id"]: AgentSpec.model_validate(a) for a in data.get("agents", [])}
        self.meta = data.get("meta", {})
        if self.events_path.exists():
            lines = self.events_path.read_text().splitlines()[-400:]
            for line in lines:
                try:
                    self.recent_events.append(HarnessEvent.model_validate(json.loads(line)))
                except Exception:
                    continue

    def flush(self) -> None:
        with self._lock:
            payload = {
                "projects": [p.model_dump() for p in self.projects.values()],
                "tasks": [t.model_dump() for t in self.tasks.values()],
                "agents": [a.model_dump() for a in self.agents.values()],
                "meta": self.meta,
            }
            tmp = self.state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
            os.replace(tmp, self.state_path)

    # ---------- mutations ----------

    def add_event(self, kind: str, message: str, project_id: str | None = None,
                  agent_id: str | None = None, task_id: str | None = None) -> HarnessEvent:
        ev = HarnessEvent(kind=kind, message=message, project_id=project_id,
                          agent_id=agent_id, task_id=task_id)
        with self._lock:
            self.recent_events.append(ev)
            del self.recent_events[:-400]
            with self.events_path.open("a") as fh:
                fh.write(json.dumps(ev.model_dump()) + "\n")
        return ev

    def create_project(self, name: str, goal: str, workspace: str,
                       constraints: str = "") -> Project:
        proj = Project(name=name, goal=goal, workspace=workspace, constraints=constraints)
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.projects[proj.id] = proj
        self.add_event("project_created", f"Project '{name}' created.", project_id=proj.id)
        self.flush()
        return proj

    def upsert_task(self, task: TaskSpec) -> TaskSpec:
        with self._lock:
            self.tasks[task.id] = task
        task.touch()
        return task

    def upsert_agent(self, agent: AgentSpec) -> AgentSpec:
        with self._lock:
            self.agents[agent.id] = agent
        agent.touch()
        self._sync_task_budget(agent)
        return agent

    def _sync_task_budget(self, agent: AgentSpec) -> None:
        """Mirror an agent's turn counters onto the task it is working.

        The card shows the task's counters but the agent owns them, so syncing
        here (rather than at each call site) is what keeps the two from
        drifting. Gated on the worker link: a reviewer also carries task_id,
        and its turns must not overwrite the worker's.
        """
        task = self.tasks.get(agent.task_id or "")
        if task is None or agent.is_reviewer or task.assigned_agent_id != agent.id:
            return
        task.turns_used = agent.turns_used
        task.turn_limit = agent.turn_limit
        task.touch()

    # ---------- queries ----------

    def project_tasks(self, project_id: str) -> list[TaskSpec]:
        return sorted(
            (t for t in self.tasks.values() if t.project_id == project_id),
            key=lambda t: t.created_at,
        )

    def project_agents(self, project_id: str) -> list[AgentSpec]:
        return sorted(
            (a for a in self.agents.values() if a.project_id == project_id),
            key=lambda a: (a.depth, a.created_at),
        )

    def board(self, project_id: str) -> dict[str, list[TaskSpec]]:
        board: dict[str, list[TaskSpec]] = {c: [] for c in KANBAN_COLUMNS}
        for t in self.project_tasks(project_id):
            board.setdefault(t.column, []).append(t)
        return board

    def column_counts(self, project_id: str) -> dict[str, int]:
        return {c: len(v) for c, v in self.board(project_id).items()}

    def tasks_in_columns(self, columns: tuple[str, ...],
                         project_id: str | None = None) -> list[TaskSpec]:
        return sorted(
            (
                t
                for t in self.tasks.values()
                if t.column in columns and (project_id is None or t.project_id == project_id)
            ),
            key=lambda t: t.created_at,
        )

    def agents_in_status(self, statuses: tuple[str, ...]) -> list[AgentSpec]:
        return [a for a in self.agents.values() if a.status in statuses]

    def agent_children(self, agent_id: str) -> list[AgentSpec]:
        agent = self.agents.get(agent_id)
        if not agent:
            return []
        return [self.agents[c] for c in agent.child_agent_ids if c in self.agents]

    def active_project_ids(self) -> list[str]:
        return [p.id for p in self.projects.values() if p.status == "active"]
