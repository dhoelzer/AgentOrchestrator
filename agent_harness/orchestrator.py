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
from .store import AgentSpec, Store, new_id

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
        for project in self.store.projects.values():
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
                project.touch()
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

        for project in self.store.projects.values():
            board = self.store.board(project.id)

            for task in board["review"]:
                note(project.id,
                     f"Task {task.id} '{task.title}' is WAITING FOR REVIEW - coordinate a "
                     f"review now (create a 'reviewer-{task.expertise}' specialist and "
                     "run it with review=true).")

            for task in board["blocked"]:
                agent = self.store.agents.get(task.assigned_agent_id or "")
                if agent and agent.status == "budget_exhausted":
                    note(project.id, prompts.budget_exhausted_briefing(
                        {"id": agent.id, "role": agent.role,
                         "turns_used": agent.turns_used,
                         "turn_limit": agent.turn_limit,
                         "last_report": agent.last_report},
                        {"id": task.id, "title": task.title,
                         "objective": task.objective},
                    ))

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

        return attention

    def housekeeping(self) -> tuple[dict[str, list[str]], list[str]]:
        """Back-compat wrapper: reconcile then detect."""
        self.reconcile()
        return self.detect(), []

    def run_cycle(self) -> list[CycleReport]:
        """One scheduler tick: reconcile, detect, then let the orchestrator act."""
        self.reconcile()
        attention_map = self.detect()
        reports: list[CycleReport] = []

        for project in sorted(self.store.projects.values(), key=lambda p: p.created_at):
            if project.status in ("paused", "delivered"):
                continue
            attention = attention_map.get(project.id, [])
            if not attention:
                continue

            reports.append(self._run_project_cycle(project, attention))

        self.store.meta["last_tick"] = int(time.time())
        self.store.flush()
        return reports

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
        self.store.flush()
        while not self._stop.is_set():
            started = time.time()
            try:
                reports = self.run_cycle()
                for r in reports:
                    logger.info("cycle %s: %d attention items, status=%s",
                                r.project_id, len(r.attention), r.orchestrator_status)
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.exception("Orchestrator cycle failed")
                self.store.add_event("loop_error", f"Cycle failed: {type(exc).__name__}: {exc}")
                self.store.flush()
            elapsed = time.time() - started
            wait = max(1.0, self.config.tick_seconds - elapsed)
            self._stop.wait(wait)
        self.runtime.close_all()
