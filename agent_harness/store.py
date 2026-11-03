"""Durable state for the harness: projects, tasks, agents, and an event log.

Everything the kanban board renders lives here, so the web UI and the
orchestrator loop share one source of truth that survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

KANBAN_COLUMNS = [
    "backlog",
    "ready",
    "in_progress",
    "review",
    "rework",
    "done",
    "blocked",
]

# How many revision entries to keep per project. Long-running projects revise
# often; the log is for recent history, not a full audit trail.
REVISION_LOG_LIMIT = 50

Column = Literal[
    "backlog", "ready", "in_progress", "review", "rework", "done", "blocked"
]
AgentStatus = Literal[
    "idle", "running", "waiting_budget", "budget_exhausted", "paused", "failed",
    "done", "llm_unavailable",
]
# Derived from the Literal so the repair path in Store._load can never drift
# out of sync with the model.
AGENT_STATUSES: tuple[str, ...] = get_args(AgentStatus)
ProjectStatus = Literal["active", "ready_for_delivery", "delivered", "paused",
                        "removed"]
PROJECT_STATUSES: tuple[str, ...] = get_args(ProjectStatus)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class _Model(BaseModel):
    """Base for persisted models.

    validate_assignment matters here: the store flushes these objects to JSON
    and re-validates them on load. Without it, an invalid field assignment is
    accepted in memory, written to disk, and only surfaces as a crash on the
    *next* startup — long after the code that caused it ran.
    """

    model_config = ConfigDict(validate_assignment=True)


class TaskSpec(_Model):
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
    # How many cycles in a row this card has been blocked for the same reason.
    # The orchestrator gets a nudge that escalates with the count: one block is
    # a hiccup, several identical blocks mean the current plan is not working.
    block_strikes: int = 0
    # The blocked_reason the current strike run is counting, so a *changed*
    # blocker restarts the count instead of inheriting the old one.
    block_strike_reason: str | None = None

    # Why the task sits in 'blocked', and any direction the user has supplied
    # since. Directives are queued so the orchestrator cannot miss one that
    # arrives mid-cycle.
    blocked_reason: str | None = None
    # Machine-readable class of `blocked_reason`. Lets reconcile() tell a block
    # that resolves itself (upstream deps landed) from one needing judgement
    # (budget, LLM outage) without pattern-matching prose.
    blocked_cause: str | None = None
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


class AgentSpec(_Model):
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
    # Spend/tokens from conversations this agent already closed. A fresh
    # conversation restarts the SDK's accumulated metrics at 0, so without this
    # baseline reconfigure_agent(fresh_conversation=True) would silently erase
    # every dollar the agent had already spent.
    prior_cost_usd: float = 0.0
    prior_tokens: int = 0

    last_report: str | None = None
    last_error: str | None = None
    # Set when the agent was retired out of the live hierarchy: its work is
    # finished and it will not be dispatched again. The record is kept (cost,
    # tokens, transcript) but the tree hides it, so a long-running project does
    # not accumulate an ever-growing wall of finished specialists.
    archived_at: str | None = None
    retirement_reason: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    def touch(self) -> None:
        self.updated_at = now_iso()


class Project(_Model):
    """A long-running project owned by the orchestrator."""

    id: str = Field(default_factory=lambda: new_id("proj"))
    name: str
    goal: str
    constraints: str = ""
    workspace: str
    status: ProjectStatus = "active"
    delivery_signaled: bool = False
    delivered_at: str | None = None
    # When the project became ready_for_delivery. ready_generation increments on
    # each transition so the delivery alert re-fires for a genuinely new
    # delivery, but not for unrelated touches.
    ready_at: str | None = None
    ready_generation: int = 0
    # Set when the user removes the project. 'hidden' kept its files, 'archived'
    # moved them, 'purged' deleted them.
    removed_at: str | None = None
    removal_mode: str | None = None
    # Spend is summed over the project's agents, but removal deletes those
    # records, so the total is snapshotted here to keep cost history.
    final_cost_usd: float | None = None
    # Project version, starting at 0.1. Minor increments on every revision
    # (work landed, review reopened a task, scope changed); major increments
    # when the project is delivered, which is a release.
    version_major: int = 0
    version_minor: int = 1
    # Short audit trail of what each version changed. Kept on the project (not
    # derived from the event log, which rotates) so the dashboard can show a
    # version history even long after the events scrolled off.
    revision_log: list[dict] = Field(default_factory=list)
    # Free-text the user has added since the last cycle: new information, a
    # changed mind, an extra request. Queued rather than applied directly
    # because only the orchestrator can turn it into tasks and agents. Consumed
    # by the cycle that briefs on it, so nothing is missed or replayed.
    user_messages: list[dict] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    @property
    def version(self) -> str:
        return f"{self.version_major}.{self.version_minor}"

    def touch(self) -> None:
        self.updated_at = now_iso()


class HarnessEvent(_Model):
    """Append-only log line surfaced in the UI activity feed."""

    id: str = Field(default_factory=lambda: new_id("ev"))
    at: str = Field(default_factory=now_iso)
    kind: str
    project_id: str | None = None
    agent_id: str | None = None
    task_id: str | None = None
    message: str


def _one_line(exc: Exception) -> str:
    """Compact a pydantic ValidationError for a single log line.

    str(exc) is multi-line and gets truncated by log shippers, which is exactly
    when an operator needs to see which field was rejected.
    """
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            parts = []
            for e in errors():
                loc = ".".join(str(x) for x in (e.get("loc") or ())) or "?"
                parts.append(f"{loc}: {e.get('msg')} (got {e.get('input')!r})")
            if parts:
                return "; ".join(parts)
        except Exception:  # noqa: BLE001
            pass
    return " ".join(str(exc).split())


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

    def _load_records(self, raw: list[dict], model: type[_Model], label: str,
                      repair) -> dict[str, Any]:
        """Validate records, repairing or dropping bad ones instead of dying.

        A single malformed record must not brick startup: the harness is a
        long-running service whose whole board lives in one JSON file, and an
        unhandled error here means the container crash-loops with no way to
        recover through the UI.
        """
        out: dict[str, Any] = {}
        for record in raw:
            try:
                out[record["id"]] = model.model_validate(record)
                continue
            except Exception as exc:  # noqa: BLE001 - startup must not raise
                first_error = exc
            repaired = repair(record)
            if repaired is None:
                logger.error("Dropping unreadable %s record %r: %s",
                             label, record.get("id"), _one_line(first_error))
                continue
            try:
                obj = model.model_validate(repaired)
            except Exception as exc2:  # noqa: BLE001
                logger.error("Dropping unreadable %s record %r: %s",
                             label, record.get("id"), _one_line(exc2))
                continue
            logger.warning("Repaired %s record %s: %s",
                           label, obj.id, _one_line(first_error))
            out[obj.id] = obj
        return out

    @staticmethod
    def _repair_agent(record: dict) -> dict | None:
        """Coerce an unknown agent status to 'failed', keeping the record.

        Statuses are the field most likely to go stale across versions, and
        'failed' is already surfaced as an alert, so a repaired agent stays
        visible instead of silently vanishing from the board.
        """
        if "status" in record and record["status"] not in AGENT_STATUSES:
            fixed = dict(record)
            fixed["status"] = "failed"
            fixed["last_error"] = (
                f"(repaired on load: status {record['status']!r} is not a known "
                "agent status)")
            return fixed
        return None

    @staticmethod
    def _repair_project(record: dict) -> dict | None:
        if "status" in record and record["status"] not in PROJECT_STATUSES:
            fixed = dict(record)
            fixed["status"] = "active"
            return fixed
        return None

    @staticmethod
    def _repair_task(record: dict) -> dict | None:
        if "column" in record and record["column"] not in KANBAN_COLUMNS:
            fixed = dict(record)
            fixed["column"] = "blocked"
            fixed["blocked_reason"] = (
                f"repaired on load: unknown column {record['column']!r}")
            return fixed
        return None

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text())
        self.projects = self._load_records(
            data.get("projects", []), Project, "project", self._repair_project)
        self.tasks = self._load_records(
            data.get("tasks", []), TaskSpec, "task", self._repair_task)
        self.agents = self._load_records(
            data.get("agents", []), AgentSpec, "agent", self._repair_agent)
        self._normalize_loaded_tasks()
        self.meta = data.get("meta", {})
        # Which build wrote this file, kept separate from the running build so a
        # version mismatch stays visible after meta is overwritten on save.
        self.meta["harness_version_stored"] = self.meta.get("harness_version")
        if self.events_path.exists():
            lines = self.events_path.read_text().splitlines()[-400:]
            for line in lines:
                try:
                    self.recent_events.append(HarnessEvent.model_validate(json.loads(line)))
                except Exception:
                    continue

    def _normalize_loaded_tasks(self) -> None:
        """Clear block fields on tasks that are no longer blocked.

        Older builds approved a review without clearing blocked_reason, leaving
        done cards advertising a blocker they had already resolved. The data is
        valid, so _repair_task never sees it; this sweep runs on every load.
        """
        for task in self.tasks.values():
            if task.column != "blocked" and task.blocked_reason:
                logger.warning("Cleared stale block on %s task %s: %s",
                               task.column, task.id, task.blocked_reason)
                task.blocked_reason = None
                task.blocked_cause = None

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
        self.add_event("project_created",
                       f"Project '{name}' created at version {proj.version}.",
                       project_id=proj.id)
        self.flush()
        return proj

    def revise_project(self, project_id: str, reason: str,
                       release: bool = False) -> Project | None:
        """Bump the project version after a material revision.

        Minor for every revision (work landed, review reopened a task, the plan
        changed); major for a delivery, which is a release. Returns None when
        the project is unknown, so callers can treat it as best-effort.
        """
        with self._lock:
            project = self.projects.get(project_id)
            if project is None:
                return None
            before = project.version
            if release:
                project.version_major += 1
                project.version_minor = 0
            else:
                project.version_minor += 1
            project.touch()
            project.revision_log.append({
                "version": project.version, "reason": reason, "at": project.updated_at,
            })
            del project.revision_log[:-REVISION_LOG_LIMIT]
        self.add_event(
            "project_revised",
            f"Project '{project.name}' {before} -> {project.version}: {reason}",
            project_id=project_id,
        )
        self.flush()
        return project

    def upsert_task(self, task: TaskSpec) -> TaskSpec:
        # finished_at is stamped here rather than at each move site so the
        # Done column can be ordered by when work actually closed. Cards moved
        # back out of done (rework) clear it and get a new one on re-entry.
        if task.column == "done":
            if task.finished_at is None:
                task.finished_at = now_iso()
        else:
            task.finished_at = None
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

    def project_agents(self, project_id: str,
                       include_archived: bool = False) -> list[AgentSpec]:
        return sorted(
            (a for a in self.agents.values()
             if a.project_id == project_id
             and (include_archived or a.archived_at is None)),
            key=lambda a: (a.depth, a.created_at),
        )

    def board(self, project_id: str) -> dict[str, list[TaskSpec]]:
        board: dict[str, list[TaskSpec]] = {c: [] for c in KANBAN_COLUMNS}
        for t in self.project_tasks(project_id):
            board.setdefault(t.column, []).append(t)
        # Done cards are newest first: the UI keeps only the most recent ones
        # visible, so the ordering has to put them at the front.
        board["done"] = sorted(
            board.get("done", []),
            key=lambda t: t.finished_at or t.updated_at, reverse=True)
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

    def agents_in_status(self, statuses: tuple[str, ...],
                         exclude_removed: bool = True) -> list[AgentSpec]:
        out = []
        for a in self.agents.values():
            if a.status not in statuses:
                continue
            if exclude_removed and a.role != "orchestrator":
                # The orchestrator is global; its project_id is just the last
                # project it touched, so removing that one must not hide it.
                proj = self.projects.get(a.project_id or "")
                if proj is not None and proj.status == "removed":
                    continue
            out.append(a)
        return out

    def agent_children(self, agent_id: str) -> list[AgentSpec]:
        agent = self.agents.get(agent_id)
        if not agent:
            return []
        return [self.agents[c] for c in agent.child_agent_ids if c in self.agents]

    def active_project_ids(self) -> list[str]:
        return [p.id for p in self.projects.values() if p.status == "active"]

    def visible_projects(self) -> list[Project]:
        """Projects the dashboard and orchestrator should still act on."""
        return sorted((p for p in self.projects.values() if p.status != "removed"),
                      key=lambda p: p.created_at)

    def project_cost(self, project_id: str) -> float:
        """Spend attributable to one project, excluding shared orchestrator spend.

        The orchestrator is global but carries the project_id of the last
        project it touched, so counting it per project would both double-count
        across projects and orphan its spend when that project is purged.

        Removal deletes the project's agents, so a removed project reports the
        total snapshotted at removal time instead of collapsing to zero.
        """
        project = self.projects.get(project_id)
        if project is not None and project.final_cost_usd is not None:
            return project.final_cost_usd
        return sum(a.cost_usd for a in self.agents.values()
                   if a.project_id == project_id and a.role != "orchestrator")

    def orchestrator_cost(self) -> float:
        """Spend by the shared orchestrator(s) - harness overhead, not project spend."""
        return sum(a.cost_usd for a in self.agents.values()
                   if a.role == "orchestrator")

    def harness_cost(self) -> float:
        """Everything actually spent: every agent still in the store.

        Summing agents (rather than projects) means spend can never be orphaned
        by a project_id pointing at a removed project. The orphan sweep is what
        keeps this equal to sum(project_cost) + orchestrator_cost.
        """
        return sum(a.cost_usd for a in self.agents.values())

    def purge_project(self, project_id: str) -> None:
        """Delete a project's records outright (the 'purge' removal mode)."""
        with self._lock:
            self.projects.pop(project_id, None)
            for tid in [t.id for t in self.tasks.values()
                        if t.project_id == project_id]:
                self.tasks.pop(tid, None)
            self.delete_project_agents(project_id)

    def delete_project_agents(self, project_id: str) -> list[str]:
        """Delete every agent belonging to a project. See delete_agents()."""
        doomed = [a.id for a in self.agents.values()
                  if a.project_id == project_id and a.role != "orchestrator"]
        return self.delete_agents(doomed)

    def delete_agents(self, agent_ids: list[str]) -> list[str]:
        """Delete these agents and scrub all references to them.

        Returns the ids removed. The orchestrator is shared across projects and
        is never deleted.

        child_agent_ids must be scrubbed as well as the records deleted: the
        delegation limit counts that list, so leaving dead ids on a parent
        blocks it from staffing any later project.
        """
        with self._lock:
            doomed = {aid for aid in agent_ids
                      if aid in self.agents
                      and self.agents[aid].role != "orchestrator"}
            if not doomed:
                return []
            for aid in doomed:
                self.agents.pop(aid, None)
            self._scrub_agent_refs(doomed)
            return sorted(doomed)

    def _scrub_agent_refs(self, doomed: set[str]) -> None:
        """Drop references to deleted agents from parents and task cards."""
        for agent in self.agents.values():
            if any(c in doomed for c in agent.child_agent_ids):
                agent.child_agent_ids = [
                    c for c in agent.child_agent_ids if c not in doomed
                ]
                agent.touch()
        for task in self.tasks.values():
            if task.assigned_agent_id in doomed:
                task.assigned_agent_id = None
            if task.reviewer_agent_id in doomed:
                task.reviewer_agent_id = None

    def orphaned_agents(self) -> dict[str, list[str]]:
        """Agents that can never run again, keyed by reason.

        'missing_project' - the project record is gone or was removed.
        'missing_parent'  - parent_agent_id points at an agent that no longer
                            exists, so the agent is detached from the hierarchy.
        'dangling_child'  - a surviving agent still lists a child id that no
                            longer exists; these ids count toward
                            max_children_per_agent, so they leak slots.

        The orchestrator is excluded: it is global and its project_id is only
        the last project it touched.
        """
        out: dict[str, list[str]] = {
            "missing_project": [], "missing_parent": [], "dangling_child": [],
        }
        for agent in self.agents.values():
            if agent.role == "orchestrator":
                continue
            project = self.projects.get(agent.project_id or "")
            if agent.project_id is None or project is None \
                    or project.status == "removed":
                out["missing_project"].append(agent.id)
            elif agent.parent_agent_id and agent.parent_agent_id not in self.agents:
                out["missing_parent"].append(agent.id)
        for agent in self.agents.values():
            if any(c not in self.agents for c in agent.child_agent_ids):
                out["dangling_child"].append(agent.id)
        return out

    def remove_project(self, project_id: str, mode: str) -> Project:
        """Drop a project from the board.

        'hidden' leaves every artifact in place; 'archived' moves the workspace
        under the archive dir; 'purged' deletes the workspace and conversation
        logs. In all three modes the project record is kept with status
        'removed' so cost history and the activity feed survive.

        The project's agents are deleted in every mode: they can never be
        dispatched again, and their ids would otherwise keep counting toward
        max_children_per_agent on their parents.
        """
        project = self.projects[project_id]
        # Snapshot before the status flips and the agent records go away.
        project.final_cost_usd = self.project_cost(project_id)
        project.status = "removed"
        project.removed_at = now_iso()
        project.removal_mode = mode
        project.touch()
        self.delete_project_agents(project_id)
        self.add_event(
            "project_removed", f"Project '{project.name}' removed ({mode}).",
            project_id=project_id,
        )
        self.flush()
        return project
