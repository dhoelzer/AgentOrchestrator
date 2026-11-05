"""Deterministic agent dispatch: the part of orchestration that needs no model.

The orchestrator decides *which* agent owns a card. Launching that agent is not
a decision - it is the mechanical consequence of one. Doing it through the LLM
anyway costs a full round trip through a saturated endpoint before anything
starts, which is how an idle endpoint ends up with nothing running: the card is
staffed, the agent is waiting, and the only thing standing between them is an
orchestrator turn that has to queue for a slot itself.

So the split is:

* orchestrator (LLM) - judgement: decompose, staff, re-specify, review, deliver
* scheduler (no LLM) - consequence: start what is already staffed and unblocked

The scheduler deliberately does not make decisions. It only acts on cards the
orchestrator has already attached an agent to, which is why `assigned_agent_id`
is the gate: an unstaffed card is left alone for the briefing, and a card the
orchestrator chose not to staff stays unstaffed.
"""

from __future__ import annotations

import logging

from .config import HarnessConfig
from .prompts import review_assignment_prompt, worker_assignment_prompt
from .store import AgentSpec, Store, TaskSpec
from .tools import _dispatch, _project_inputs, _reactivate, _task_dict, ctx

logger = logging.getLogger(__name__)

# Columns a worker run moves forward on its own. `review` is excluded: a card
# there has finished its worker stretch, and its verdict comes from record_review.
DISPATCHABLE = ("ready", "rework")

# A card still in 'backlog' is normally unstaffed, and reconcile() promotes it
# to 'ready' once its dependencies land. But an explicit assignment is itself
# the staffing decision, so a backlog card that already has an agent attached is
# fair game. This matters most when cycles keep getting refused: reconcile()
# runs inside a cycle, so a harness with a stuck orchestrator prompt would never
# promote anything and the board would freeze in 'backlog' forever.
ASSIGNABLE = DISPATCHABLE + ("backlog",)

# A block the scheduler must not try to work through by starting an agent.
HARD_BLOCKS = ("llm", "manual")


class Scheduler:
    """Starts already-staffed work as soon as capacity allows."""

    def __init__(self, config: HarnessConfig, store: Store, runtime) -> None:
        self.config = config
        self.store = store
        self.runtime = runtime
        # Cards the scheduler started, so a cycle briefing can tell the
        # orchestrator its assignment is already moving rather than waiting.
        self.started_total = 0

    # ---------------- selection ----------------

    def _blocked_for_llm(self, task: TaskSpec, agent: AgentSpec) -> bool:
        """True when starting this agent would just hit a known-dead endpoint."""
        if task.blocked_cause in HARD_BLOCKS:
            return True
        if agent.status == "llm_unavailable" or agent.llm_failures >= 1:
            return True
        return False

    def _ready(self) -> list[tuple[TaskSpec, AgentSpec, bool]]:
        """Staffed, unblocked cards that can start right now.

        Oldest first, so a backlog drains in the order the orchestrator created
        it rather than by dict order.
        """
        out: list[tuple[TaskSpec, AgentSpec, bool]] = []
        for project in self.store.visible_projects():
            # Frozen projects are not scheduled: a pause means the runs already
            # in flight may finish, but nothing new starts until the user
            # resumes.
            if project.is_frozen:
                continue
            for task in self.store.project_tasks(project.id):
                review = task.column == "review"
                if not review and task.column not in ASSIGNABLE:
                    continue
                if review and not self._needs_review(task):
                    continue
                agent_id = (task.reviewer_agent_id if review
                            else task.assigned_agent_id)
                if not agent_id:
                    # Unstaffed: that is the orchestrator's call, not ours.
                    continue
                agent = self.store.agents.get(agent_id)
                if agent is None:
                    continue
                if agent.archived_at is not None:
                    continue
                if agent.id in self.runtime.inflight:
                    continue
                if agent.turn_limit - agent.turns_used <= 0:
                    continue
                if self._blocked_for_llm(task, agent):
                    continue
                if not all(self.store.tasks.get(d) is not None
                           and self.store.tasks[d].column == "done"
                           for d in task.depends_on):
                    continue
                out.append((task, agent, review))
        return sorted(out, key=lambda ta: (ta[0].created_at or "", ta[0].id))

    @staticmethod
    def _needs_review(task: TaskSpec) -> bool:
        """A card in 'review' that is still waiting for its first verdict.

        A reviewer run does not move its own card - record_review does - so
        without this a reviewer that has already reported would be dispatched
        again on the next tick, forever.
        """
        return task.review_verdict is None and task.review_rounds == 0

    # ---------------- execution ----------------

    def sweep(self) -> list[tuple[str | None, str]]:
        """Start what can start. Returns (project_id, note) for the briefing."""
        if not self.config.auto_dispatch:
            return []
        notes: list[tuple[str | None, str]] = []
        for task, agent, review in self._ready():
            cap = self.runtime.capacity()
            free = cap["slots_free"]
            if not self.runtime.project_slot_free(task.project_id):
                notes.append((
                    task.project_id,
                    f"SCHEDULER: {task.project_id} is at its sub-agent "
                    f"concurrency limit "
                    f"({cap['max_concurrent_sub_agents_per_project']} running). "
                    f"{task.id} '{task.title}' is staffed and waiting on a slot, "
                    "not on a decision. It starts on its own when one frees; "
                    "dispatch work in another project meanwhile."))
                continue
            if free <= 0:
                notes.append((
                    task.project_id,
                    f"SCHEDULER: no free LLM slot; {task.id} '{task.title}' is "
                    "staffed and waiting on capacity, not on a decision. Do not "
                    "run_agent it; it starts on its own once a slot frees."))
                break
            if self._start(task, agent, review):
                self.started_total += 1
                notes.append((
                    task.project_id,
                    f"SCHEDULER: started {agent.role} ({agent.id}) on task "
                    f"{task.id} '{task.title}'"
                    + (" as reviewer" if review else "")
                    + f" with {agent.turn_limit - agent.turns_used} turns. "
                    "Do not run_agent it again this cycle."))
        return notes

    def _start(self, task: TaskSpec, agent: AgentSpec, review: bool) -> bool:
        """Dispatch one card in the background. False if the runtime refused."""
        remaining = agent.turn_limit - agent.turns_used
        _reactivate(ctx, agent)
        inputs = _project_inputs(self.store, task)
        if review:
            message = review_assignment_prompt(_task_dict(task), input_files=inputs)
        else:
            message = worker_assignment_prompt(
                _task_dict(task), turns_remaining=remaining,
                prior=agent.last_report, input_files=inputs)
        try:
            self.runtime.submit(
                agent.id,
                lambda: _dispatch(agent, task, message, remaining,
                                  review=review))
        except RuntimeError as exc:
            # Usually "already running": a race with the orchestrator's own
            # dispatch. Not an error worth escalating - the work is starting.
            logger.info("Scheduler skipped %s on %s: %s", agent.id, task.id, exc)
            return False
        self.store.add_event(
            "agent_dispatched",
            f"Scheduler started {agent.role} ({agent.id}) on task {task.id} "
            f"'{task.title}' without an orchestrator turn.",
            project_id=agent.project_id, agent_id=agent.id, task_id=task.id)
        self.store.flush()
        return True
