"""The orchestrator agent and the forever-running scheduler loop.

Every ``tick_seconds`` the scheduler:
  1. runs deterministic housekeeping (budget-exhaustion, review queue, delivery
     readiness, kanban refresh),
  2. hands the orchestrator a briefing of what needs attention,
  3. lets the orchestrator act through its board tools.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from openhands.sdk import Tool
from openhands.sdk.logger import get_logger

from . import prompts, tools as harness_tools
from .config import HarnessConfig
from .runtime import AgentRuntime
from .store import AgentSpec, Store, new_id, now_iso

logger = get_logger(__name__)


@dataclass
class CycleReport:
    project_id: str
    attention: list[str]
    actions: list[str]
    orchestrator_status: str
    error: str | None = None


class Orchestrator:
    """Root agent that plans, delegates, reviews, and delivers."""

    def __init__(self, config: HarnessConfig, store: Store, runtime: AgentRuntime):
        self.config = config
        self.store = store
        self.runtime = runtime
        self._stop = threading.Event()
        harness_tools.init_context(config, store, runtime)
        self.spec = self._load_or_create_spec()

    # ---------- identity ----------

    def _load_or_create_spec(self) -> AgentSpec:
        existing = [a for a in self.store.agents.values() if a.role == "orchestrator"]
        if existing:
            return sorted(existing, key=lambda a: a.created_at)[0]
        spec = AgentSpec(
            id=new_id("orch"),
            role="orchestrator",
            specialty="project orchestration",
            mission="Decompose projects, create specialists, coordinate reviews, "
                    "manage turn budgets, and drive projects to delivery.",
            system_prompt=prompts.orchestrator_prompt(),
            depth=0,
            status="idle",
            turn_limit=self.config.orchestrator_turn_limit,
            created_by="system",
            workspace=str(self.config.root_dir),
        )
        self.store.upsert_agent(spec)
        self.store.flush()
        return spec

    # ---------- cycle ----------

    def reconcile(self) -> None:
        """Deterministic board transitions that need no judgement."""
        for project in self.store.visible_projects():
            for task in self.store.project_tasks(project.id):
                # A backlog task with satisfied dependencies is ready to staff.
                if task.column == "backlog" and all(
                    d in self.store.tasks and self.store.tasks[d].column == "done"
                    for d in task.depends_on
                ):
                    task.column = "ready"
                    task.touch()
                    self.store.add_event(
                        "task_unblocked",
                        f"Task '{task.title}' dependencies satisfied -> ready.",
                        project_id=project.id, task_id=task.id,
                    )

            tasks = self.store.project_tasks(project.id)
            if tasks and all(t.column == "done" for t in tasks) \
                    and project.status == "active":
                project.status = "ready_for_delivery"
                project.ready_at = now_iso()
                project.ready_generation += 1
                project.touch()
                self.store.revise_project(
                    project.id, "all tasks done - ready for delivery")
                self.store.add_event(
                    "delivery_candidate",
                    f"Project '{project.name}' looks ready to deliver.",
                    project_id=project.id,
                )
        self.store.flush()

    def detect(self) -> dict[str, list[str]]:
        """Flag everything the orchestrator must act on this cycle."""
        attention: dict[str, list[str]] = {pid: [] for pid in self.store.projects}
        inflight = self.runtime.inflight

        def note(project_id: str | None, message: str) -> None:
            attention.setdefault(project_id, []).append(message)

        for project in self.store.visible_projects():
            board = self.store.board(project.id)

            for task in board["review"]:
                note(project.id,
                     f"Task {task.id} '{task.title}' is WAITING FOR REVIEW - coordinate a "
                     f"review now (create a 'reviewer-{task.expertise}' specialist and "
                     "run it with review=true).")

            for task in board["blocked"]:
                agent = self.store.agents.get(task.assigned_agent_id or "")
                if task.user_directives:
                    queued = "; ".join(
                        f"[{d.get('at', '')}] {d.get('text', '')}"
                        for d in task.user_directives
                    )
                    note(project.id,
                         f"Task {task.id} '{task.title}' is BLOCKED and the user has "
                         f"supplied direction - apply it and re-dispatch: {queued}")
                elif agent and agent.status == "budget_exhausted":
                    note(project.id, prompts.budget_exhausted_briefing(
                        {"id": agent.id, "role": agent.role,
                         "turns_used": agent.turns_used,
                         "turn_limit": agent.turn_limit,
                         "last_report": agent.last_report},
                        {"id": task.id, "title": task.title,
                         "objective": task.objective},
                    ))
                elif agent and agent.status == "llm_unavailable":
                    note(project.id,
                         f"Task {task.id} '{task.title}' is blocked because the LLM "
                         f"endpoint returned nothing for {agent.id} ({agent.role}). "
                         f"Timeout is now {self.runtime.llm_timeout}s - re-dispatch "
                         "the same agent, or pause the project if the endpoint is down.")
                elif agent and agent.status == "failed":
                    note(project.id,
                         f"Task {task.id} '{task.title}' is blocked after agent "
                         f"{agent.id} FAILED: {agent.last_error} - reconfigure or "
                         "replace the agent, then re-dispatch.")
                else:
                    note(project.id,
                         f"Task {task.id} '{task.title}' is BLOCKED"
                         + (f": {task.blocked_reason}" if task.blocked_reason else "")
                         + " - decide whether to re-dispatch, reconfigure, split it, "
                         "or ask the user for direction.")

            for task in board["rework"]:
                note(project.id,
                     f"Task {task.id} '{task.title}' needs REWORK after review round "
                     f"{task.review_rounds}. Feedback: {task.review_feedback}")
                if task.review_rounds >= self.config.review_attempts_before_escalation:
                    note(project.id,
                         f"Task {task.id} has failed review {task.review_rounds} times - "
                         "escalate: reconfigure the specialist, split the task, or mark "
                         "it blocked for the user.")

            for task in board["ready"]:
                if task.assigned_agent_id in inflight:
                    continue
                note(project.id,
                     f"Task {task.id} '{task.title}' is READY and unstaffed - create a "
                     f"'{task.expertise}' specialist and dispatch it.")

            for task in board["in_progress"]:
                if task.assigned_agent_id not in inflight:
                    note(project.id,
                         f"Task {task.id} '{task.title}' shows in_progress but its agent "
                         "is not running - re-dispatch it or move it to 'rework'.")

            for task in board["backlog"]:
                unmet = [d for d in task.depends_on
                         if d in self.store.tasks
                         and self.store.tasks[d].column != "done"]
                if unmet:
                    note(project.id,
                         f"Task {task.id} '{task.title}' is waiting on "
                         f"{', '.join(unmet)} - finish those first.")

            if self.store.projects[project.id].status == "ready_for_delivery" \
                    and not project.delivery_signaled:
                note(project.id,
                     f"ALL tasks for '{project.name}' are done - verify the deliverables "
                     "with read_file, then call signal_delivery so the user is notified.")

            if not self.store.project_tasks(project.id):
                note(project.id,
                     f"Project '{project.name}' has no tasks - decompose the goal into "
                     "tasks with acceptance criteria now.")

        for agent in self.store.agents_in_status(("budget_exhausted",)):
            task = self.store.tasks.get(agent.task_id or "")
            if task is not None and task.column == "blocked" \
                    and task.assigned_agent_id == agent.id:
                continue  # already surfaced via the blocked-task briefing above
            note(agent.project_id,
                 f"Agent {agent.id} ({agent.role}) is BUDGET EXHAUSTED at "
                 f"{agent.turns_used}/{agent.turn_limit} turns - extend, reconfigure, "
                 "or split its work.")

        for agent in self.store.agents_in_status(("failed",)):
            note(agent.project_id,
                 f"Agent {agent.id} ({agent.role}) FAILED: {agent.last_error} - "
                 "reconfigure or replace it.")

        for agent in self.store.agents_in_status(("llm_unavailable",)):
            task = self.store.tasks.get(agent.task_id or "")
            if task is not None and task.column == "blocked" \
                    and task.assigned_agent_id == agent.id:
                continue  # already surfaced via the blocked-task briefing above
            note(agent.project_id,
                 f"Agent {agent.id} ({agent.role}) made no progress: the LLM endpoint "
                 f"returned nothing. Request timeout is now {self.runtime.llm_timeout}s "
                 f"({self.runtime.llm_empty_responses} empty responses seen) - retry it "
                 "with resume_agent rather than treating the task as failed.")

        if self.runtime.llm_empty_responses or self.runtime.llm_transient_errors:
            # Global endpoint health: attached to every project, since run_cycle
            # only forwards attention keyed by a real project id.
            health = (
                f"LLM endpoint health: {self.runtime.llm_empty_responses} empty "
                f"responses, {self.runtime.llm_transient_errors} transient errors, "
                f"timeout widened {self.runtime.llm_timeouts_widened}x to "
                f"{self.runtime.llm_timeout}s. If this keeps growing the endpoint is "
                "overloaded - reduce parallelism or wait before dispatching more agents."
            )
            for project_id in attention:
                attention[project_id].append(health)

        return attention

    def housekeeping(self) -> tuple[dict[str, list[str]], list[str]]:
        """Back-compat wrapper: reconcile then detect."""
        self.reconcile()
        return self.detect(), []

    def sweep_orphans(self) -> list[str]:
        """Forget agents that can never run again.

        Removing a project deletes its agents, but a removal that happened in
        an earlier process - or a hand-edited state file - can leave agents
        whose project is gone, agents detached from a deleted parent, and
        parents still holding dead child ids. Those dead ids count toward
        max_children_per_agent, so they permanently steal delegation slots.

        Returns a human-readable line per thing cleaned up.
        """
        orphans = self.store.orphaned_agents()
        lines: list[str] = []

        doomed = sorted(set(orphans["missing_project"]) | set(orphans["missing_parent"]))
        if doomed:
            roles = {a: self.store.agents[a].role for a in doomed
                     if a in self.store.agents}
            logs = self.runtime.delete_agent_artifacts(doomed)
            self.store.delete_agents(doomed)
            for aid in doomed:
                lines.append(
                    f"Deleted orphaned agent {aid} ({roles.get(aid, '?')})."
                )
            if logs:
                lines.append(f"Deleted {len(logs)} orphaned conversation log dir(s).")

        # Dead ids left on surviving agents leak delegation slots even when the
        # agent record itself is fine.
        for aid in orphans["dangling_child"]:
            parent = self.store.agents.get(aid)
            if parent is None:
                continue
            dead = [c for c in parent.child_agent_ids if c not in self.store.agents]
            parent.child_agent_ids = [c for c in parent.child_agent_ids
                                      if c in self.store.agents]
            self.store.upsert_agent(parent)
            lines.append(
                f"Released {len(dead)} stale child id(s) on {aid} "
                f"({parent.role}): {', '.join(dead)}."
            )

        if lines:
            self.store.add_event(
                "orphans_cleaned",
                f"Orphan sweep: {'; '.join(lines)}",
            )
            self.store.flush()
        return lines

    def run_cycle(self) -> list[CycleReport]:
        """One scheduler tick: sweep, reconcile, detect, then let the orchestrator act."""
        self.sweep_orphans()
        self.reconcile()
        attention_map = self.detect()
        reports: list[CycleReport] = []

        for project in self.store.visible_projects():
            if project.status in ("paused", "delivered"):
                continue
            attention = attention_map.get(project.id, [])
            if not attention:
                continue

            reports.append(self._run_project_cycle(project, attention))

        self._record_cycle_end("ok")
        return reports

    def _record_cycle_end(self, status: str) -> None:
        """Stamp cycle metadata.

        last_tick alone is written only on success, so a harness whose cycles
        keep failing shows a timestamp frozen at the first completed cycle -
        which reads like 'last cycle never updates'. The status and failure
        fields make the stall visible instead.
        """
        now = int(time.time())
        meta = self.store.meta
        started = meta.get("cycle_started_at")
        meta["cycle_count"] = int(meta.get("cycle_count", 0)) + 1
        meta["cycle_status"] = status
        meta["cycle_duration"] = (now - int(started)) if started else 0
        if status == "ok":
            meta["last_tick"] = now
            meta["last_cycle_completed_at"] = now
        meta["last_cycle_attempt_at"] = now
        meta["loop_started_at"] = meta.get("loop_started_at") or now

    def _run_project_cycle(self, project, attention: list[str]) -> CycleReport:
        board = {col: [self._task_to_dict(t) for t in items]
                 for col, items in self.store.board(project.id).items()}
        agents = [self._agent_to_dict(a) for a in self.store.project_agents(project.id)]
        briefing = prompts.cycle_briefing(
            {"id": project.id, "name": project.name, "goal": project.goal,
             "constraints": project.constraints, "workspace": project.workspace,
             "status": project.status},
            board, agents, attention,
        )

        # Refresh the orchestrator's per-cycle budget; its conversation persists.
        self.spec.turn_limit = self.spec.turns_used + self.config.orchestrator_turn_limit
        self.spec.task_id = None
        self.spec.project_id = project.id
        self.spec.touch()

        tool_specs = [Tool(name=n) for n in harness_tools.ORCHESTRATOR_TOOLS]
        outcome = self.runtime.run(self.spec, tool_specs, briefing,
                                   turn_limit=self.spec.turn_limit)

        actions = [e.message for e in self.store.recent_events[-25:]
                   if e.kind in ("task_created", "agent_created", "agent_dispatched",
                                 "review_recorded", "budget_extended",
                                 "agent_reconfigured", "delivery_signaled")]
        return CycleReport(
            project_id=project.id, attention=attention, actions=actions,
            orchestrator_status=outcome.status, error=outcome.error,
        )

    @staticmethod
    def _task_to_dict(task) -> dict:
        return {
            "id": task.id, "title": task.title, "objective": task.objective,
            "expertise": task.expertise, "column": task.column,
            "depends_on": task.depends_on,
            "acceptance_criteria": task.acceptance_criteria,
            "deliverables": task.deliverables,
            "assigned_agent_id": task.assigned_agent_id,
            "turns_used": task.turns_used, "turn_limit": task.turn_limit,
            "review_rounds": task.review_rounds,
            "result_summary": task.result_summary,
            "review_verdict": task.review_verdict,
            "review_feedback": task.review_feedback,
        }

    @staticmethod
    def _agent_to_dict(agent) -> dict:
        return {
            "id": agent.id, "role": agent.role, "specialty": agent.specialty,
            "status": agent.status, "turns_used": agent.turns_used,
            "turn_limit": agent.turn_limit, "task_id": agent.task_id,
            "parent_agent_id": agent.parent_agent_id,
        }

    # ---------- loop ----------

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        logger.info("Orchestrator loop starting (tick=%ss)", self.config.tick_seconds)
        self.store.add_event("loop_started",
                             f"Orchestrator loop started (tick {self.config.tick_seconds}s).")
        self.store.meta["loop_started_at"] = int(time.time())
        self.store.flush()
        while not self._stop.is_set():
            started = time.time()
            self.store.meta["cycle_started_at"] = int(started)
            self.store.meta["cycle_status"] = "running"
            self.store.flush()
            try:
                reports = self.run_cycle()
                for r in reports:
                    logger.info("cycle %s: %d attention items, status=%s",
                                r.project_id, len(r.attention), r.orchestrator_status)
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.exception("Orchestrator cycle failed")
                self.store.add_event("loop_error", f"Cycle failed: {type(exc).__name__}: {exc}")
                self._record_cycle_end("failed")
            elapsed = time.time() - started
            wait = max(1.0, self.config.tick_seconds - elapsed)
            self._stop.wait(wait)
        self.runtime.close_all()
