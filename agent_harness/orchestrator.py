"""The orchestrator agent and the forever-running scheduler loop.

Every ``tick_seconds`` the scheduler:
  1. runs deterministic housekeeping (budget-exhaustion, review queue, delivery
     readiness, kanban refresh),
  2. hands the orchestrator a briefing of what needs attention,
  3. lets the orchestrator act through its board tools.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import deque
from dataclasses import dataclass

from openhands.sdk import Tool
from openhands.sdk.logger import get_logger

from . import prompts, tools as harness_tools
from .config import HarnessConfig, resolve_config_path
from .runtime import AgentRuntime
from .scheduler import Scheduler
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

    def __init__(self, config: HarnessConfig, store: Store, runtime: AgentRuntime,
                 config_path: str | None = None):
        self.config = config
        self.store = store
        self.runtime = runtime
        self.config_path = resolve_config_path(config_path)
        # Fingerprint of config.json as last seen, so an edit made outside the
        # API (or by another process) is noticed on the next tick.
        self._config_fp = self._file_fingerprint()
        self._stop = threading.Event()
        # Only one cycle may run at a time. The scheduler loop is sequential,
        # but /api/tick starts a cycle from the web thread, and a manual tick
        # landing on a running cycle enqueues a second orchestrator LLM request
        # while the first one is still holding the conversation.
        self._cycle_lock = threading.Lock()
        # Starts work the orchestrator has already assigned, on its own tick, so
        # a staffed card does not wait for another orchestrator turn (and
        # another trip through a saturated endpoint) before anything happens.
        self.scheduler = Scheduler(config, store, runtime)
        self._scheduler_notes: list[tuple[str | None, str]] = []
        # Recent cycle durations, and the interval the loop is currently using.
        self._cycle_durations: deque[float] = deque(maxlen=12)
        # Set when something happens that should not wait out a long idle
        # interval: a new project, new user input, or an agent finishing.
        self._wake = threading.Event()
        # Whether the most recent cycle had anything to act on. Drives the
        # idle backoff in _next_tick.
        self._cycle_had_work = True
        self._last_skip_event_at: float | None = None
        self.effective_tick = float(max(1, config.tick_seconds))
        self.config_notes: list[str] = []
        harness_tools.init_context(config, store, runtime)
        self.spec = self._load_or_create_spec()

    # ---------- config reload ----------

    def _file_fingerprint(self) -> str:
        """Identity of config.json as it exists on disk right now.

        Hashed content rather than mtime/size: filesystem timestamp granularity
        is coarse enough that a write can land with an unchanged mtime, and an
        edit like max_parallel_workers 2 -> 6 leaves the byte count identical.
        The file is small and this runs once per tick, so hashing is cheap.
        """
        try:
            return hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        except OSError:
            return "missing"

    def note_config_written(self) -> None:
        """Called after the API patches config.json.

        The live config was already updated in place by the patch, so the next
        tick must not treat its own write as an external edit and emit a
        spurious reload event.
        """
        self._config_fp = self._file_fingerprint()

    def reload_config_if_changed(self) -> list[str]:
        """Re-read config.json at the start of a tick if it changed.

        Returns a human-readable list of what was reconfigured (empty if
        nothing changed). The live config object is mutated in place rather
        than replaced: the API, the tool context and the runtime all hold this
        same object, so rebinding it here would leave them on stale settings.
        """
        fp = self._file_fingerprint()
        if fp == self._config_fp:
            return []
        self._config_fp = fp

        try:
            fresh = HarnessConfig.load(self.config_path)
        except Exception as exc:  # noqa: BLE001 - a bad edit must not stop the loop
            self.store.add_event(
                "config_reload_failed",
                f"config.json changed but could not be loaded; keeping the "
                f"previous settings: {type(exc).__name__}: {exc}")
            logger.exception("Config reload failed")
            return []

        before = self._config_snapshot()
        # In-place update: keep every holder of this object in sync.
        for field in type(self.config).model_fields:
            setattr(self.config, field, getattr(fresh, field))

        notes = self._apply_config_changes(before)
        if notes:
            self.store.add_event(
                "config_reloaded",
                "Config changed on disk; reconfigured: " + "; ".join(notes))
            logger.info("Config reloaded: %s", "; ".join(notes))
        return notes

    def _config_snapshot(self) -> dict:
        cfg = self.config
        return {
            "tick_seconds": cfg.tick_seconds,
            "max_parallel_workers": cfg.max_parallel_workers,
            "llm_queue_depth": cfg.llm_queue_depth,
            "sub_agent_turn_limit": cfg.sub_agent_turn_limit,
            "orchestrator_turn_limit": cfg.orchestrator_turn_limit,
            "max_sub_agent_depth": cfg.max_sub_agent_depth,
            "max_children_per_agent": cfg.max_children_per_agent,
            "request_timeout": cfg.llm.request_timeout,
            "server_concurrency": cfg.llm_server_concurrency,
            "router_selection": cfg.router_selection,
            "servers": {s.name: s.fingerprint() for s in cfg.servers()},
        }

    def _apply_config_changes(self, before: dict) -> list[str]:
        """Push changed settings into the live runtime, describing each move."""
        cfg = self.config
        notes: list[str] = []

        if cfg.max_parallel_workers != before["max_parallel_workers"]:
            old, new = before["max_parallel_workers"], cfg.max_parallel_workers
            self.runtime.set_worker_slots(new)
            if new > old:
                notes.append(
                    f"worker slots {old} -> {new}: {new - old} more agents "
                    "can run at once, so schedule more of the ready work now")
            else:
                notes.append(
                    f"worker slots {old} -> {new}: running agents are left "
                    f"to drain down; do not start more than {new} at a time")

        if (cfg.max_parallel_workers != before["max_parallel_workers"]
                or cfg.llm_queue_depth != before["llm_queue_depth"]):
            self.runtime.sync_pool_size()

        if cfg.router_selection != before["router_selection"]:
            notes.append(
                "endpoint selection %s -> %s: the router now picks endpoints by %s"
                % (before["router_selection"], cfg.router_selection,
                   "their measured token rate"
                   if cfg.router_selection == "throughput" else "load alone"))

        if cfg.llm.request_timeout != before["request_timeout"]:
            self.runtime.set_llm_timeout(cfg.llm.request_timeout)
            notes.append(f"LLM request timeout -> {cfg.llm.request_timeout}s")

        if cfg.llm_server_concurrency != before["server_concurrency"]:
            notes.append(
                f"requests per LLM endpoint "
                f"{before['server_concurrency']} -> {cfg.llm_server_concurrency}")

        after = {s.name: s.fingerprint() for s in cfg.servers()}
        added = sorted(set(after) - set(before["servers"]))
        removed = sorted(set(before["servers"]) - set(after))
        changed = sorted(n for n in set(after) & set(before["servers"])
                         if after[n] != before["servers"][n])
        if added or removed or changed:
            notes.append(
                "LLM servers now: " + ", ".join(after)
                + (f" (added {', '.join(added)})" if added else "")
                + (f" (removed {', '.join(removed)})" if removed else "")
                + (f" (reconfigured {', '.join(changed)})" if changed else ""))
            # The router is the only thing that talks to an endpoint, so a
            # server change is applied there and nowhere else: no conversation
            # has to be re-pointed, because none of them hold an endpoint.
            if self.runtime.router is not None:
                self.runtime.router.sync_servers()
                self.runtime.router.probe_all()
            self.store.meta["llm_servers_active"] = list(after)

        for attr, label in (("tick_seconds", "tick"),
                            ("sub_agent_turn_limit", "sub-agent turn limit"),
                            ("orchestrator_turn_limit", "orchestrator turn limit"),
                            ("max_sub_agent_depth", "max sub-agent depth"),
                            ("max_children_per_agent", "max children per agent")):
            if getattr(cfg, attr) != before[attr]:
                notes.append(f"{label} {before[attr]} -> {getattr(cfg, attr)}")

        return notes

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
            # A paused project is frozen: its board stops moving, so a card does
            # not silently advance (or get declared delivery-ready) while the
            # user has the halt on. Runs already in flight still write their own
            # results through the tools, which is the queue draining.
            if project.is_frozen:
                continue
            for task in self.store.project_tasks(project.id):
                deps_met = all(
                    d in self.store.tasks and self.store.tasks[d].column == "done"
                    for d in task.depends_on
                )
                # A backlog task with satisfied dependencies is ready to staff.
                if task.column == "backlog" and deps_met:
                    task.column = "ready"
                    task.touch()
                    self.store.add_event(
                        "task_unblocked",
                        f"Task '{task.title}' dependencies satisfied -> ready.",
                        project_id=project.id, task_id=task.id,
                    )
                # A card blocked on upstream work is no longer blocked once that
                # work lands - another agent resolved it, so re-queue it rather
                # than waiting on judgement. Blocks the harness itself wrote for
                # a reason that dependencies cannot fix (budget, LLM outage) stay
                # put for the cycle briefing.
                elif (task.column == "blocked" and deps_met and task.depends_on
                      and task.blocked_cause not in ("budget", "llm", "manual")):
                    task.column = "ready"
                    task.blocked_reason = None
                    task.blocked_cause = None
                    task.touch()
                    self.store.add_event(
                        "task_unblocked",
                        f"Task '{task.title}' unblocked: upstream dependencies "
                        "are now done.",
                        project_id=project.id, task_id=task.id,
                    )

                # A card that is not blocked carries no grudge: reset its strike
                # count so a later, unrelated block starts from one.
                if task.column != "blocked" and (
                        task.block_strikes or task.block_strike_reason):
                    task.block_strikes = 0
                    task.block_strike_reason = None
                    task.touch()

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
            # Nothing to coordinate on a frozen project: no cycle is run for it,
            # so attention would be computed, counted, and thrown away - and a
            # 'waiting for review' line in the UI would promise a review that is
            # never going to be scheduled.
            if project.is_frozen:
                continue
            board = self.store.board(project.id)

            # Anything the user typed into the project box. This is the highest
            # priority attention item: it is a direct human request, and the
            # only thing that can turn it into work is the orchestrator.
            for msg in project.user_messages:
                note(project.id,
                     f"NEW USER INPUT for this project (at {msg.get('at', '')}): "
                     f"{msg.get('text', '')} - decide what task cards this implies, "
                     "then create them and either assign an existing agent or "
                     "create a new specialist for each. If it changes the goal or "
                     "scope, say so in your report.")

            for task in board["review"]:
                note(project.id,
                     f"Task {task.id} '{task.title}' is WAITING FOR REVIEW - coordinate a "
                     f"review now (create a 'reviewer-{task.expertise}' specialist and "
                     "run it with review=true).")

            for task in board["blocked"]:
                agent = self.store.agents.get(task.assigned_agent_id or "")
                # Every blocked card is evaluated on every tick, so count how
                # long it has sat blocked on the same reason. A changed blocker
                # is a new problem and restarts the count.
                strikes = self._count_strike(task)
                if task.user_directives:
                    queued = "; ".join(
                        f"[{d.get('at', '')}] {d.get('text', '')}"
                        for d in task.user_directives
                    )
                    note(project.id,
                         f"Task {task.id} '{task.title}' is BLOCKED and the user has "
                         f"supplied direction - apply it and re-dispatch: {queued}")
                # An LLM outage is not a problem the orchestrator can solve: it
                # cannot restart an endpoint, and re-dispatching into a dead one
                # every tick is how a project looks 'stuck'. The harness already
                # blocked the card and alerted the user, so the instruction here
                # is to hold unless the endpoint is demonstrably back. A user
                # directive still wins: it may say the endpoint is fixed.
                elif task.blocked_cause == "llm" or (
                        agent is not None and agent.status == "llm_unavailable"):
                    note(project.id, self._llm_outage_note(task, agent))
                    continue
                elif agent and agent.status == "budget_exhausted":
                    note(project.id, prompts.budget_exhausted_briefing(
                        {"id": agent.id, "role": agent.role,
                         "turns_used": agent.turns_used,
                         "turn_limit": agent.turn_limit,
                         "last_report": agent.last_report},
                        {"id": task.id, "title": task.title,
                         "objective": task.objective},
                    ))
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
                note(project.id, self._recovery_menu(task, agent, strikes))

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

    def _count_strike(self, task) -> int:
        """Record that this card was blocked again for the same reason.

        Returns the count including this cycle. The escalation is what stops a
        card cycling through the same failed plan forever: the orchestrator is
        told plainly when repeating the same move has already not worked.
        """
        reason = task.blocked_reason or ""
        if task.block_strike_reason != reason:
            task.block_strike_reason = reason
            task.block_strikes = 1
        else:
            task.block_strikes += 1
        self.store.upsert_task(task)
        return task.block_strikes

    def _llm_outage_note(self, task, agent) -> str:
        failures = agent.llm_failures if agent else 0
        return (
            f"LLM OUTAGE on task {task.id} '{task.title}' - DO NOT ACT ON IT. "
            f"The endpoint failed {failures} run(s) in a row"
            + (f" for {agent.id} ({agent.role}): {agent.llm_failure_error}"
               if agent else "")
            + ". The harness already blocked this card and alerted the user; "
            "an unreachable endpoint is not an agent failure, so do not "
            "re-dispatch, reconfigure, extend, or re-specify it. Leave it "
            "blocked - it returns to 'ready' when the user acknowledges the "
            "alert after fixing the endpoint.")

    @staticmethod
    def _recovery_menu(task, agent, strikes: int) -> str:
        """The three ways out of a block, escalated by how long it has lasted.

        The orchestrator must pick one deliberately rather than re-dispatching
        the same agent on the same instructions and hoping for a different
        result.
        """
        turns = f"{task.turns_used}/{task.turn_limit}"
        head = (f"RECOVERY DECISION for blocked task {task.id} '{task.title}' "
                f"(turns {turns}, blocked {strikes} cycle(s) in a row"
                + (f", cause {task.blocked_cause}" if task.blocked_cause else "")
                + "). Choose exactly one:")
        options = [
            "1. MORE TURNS: if the agent was making real progress and simply ran "
            "out of budget, call `extend_agent` (it moves the card back to ready) "
            "then `resume_agent` with the same agent, or move the card to backlog "
            "with `update_task` to re-queue it.",
            "2. RE-SPECIFY: if the agent was confused, drifting out of scope, or "
            "misreading the objective, call `update_task` to rewrite the title, "
            "objective, acceptance criteria and deliverables, and `reconfigure_agent` "
            "to change its prompt/tools/model. Then re-dispatch with explicit "
            "`instructions`.",
            "3. SPLIT: if the work is simply too big for one agent, call "
            "`update_task` to reduce the parent to the first part, then "
            "`create_task` with parent_task_id set to this card for each remaining "
            "part, and staff each with an existing agent or a new specialist.",
        ]
        if strikes >= 3:
            options.append(
                f"ESCALATION: this card has been blocked on the same reason for "
                f"{strikes} cycles. Repeating the previous fix has not worked - "
                "prefer option 3 (split) or option 2 (re-specify), and record your "
                "reasoning in the `note`/`decision` field.")
        if agent is not None and agent.status == "budget_exhausted":
            options.append(
                f"Note: {agent.id} ({agent.role}) is budget_exhausted at "
                f"{agent.turns_used}/{agent.turn_limit} turns; its last report is "
                "the best evidence for which option fits.")
        return head + "\n   " + "\n   ".join(options)

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

    def sweep_finished_agents(self) -> list[str]:
        """Archive sub-agents whose work is finished out of the live hierarchy.

        A specialist that has done its job stays in the tree forever otherwise,
        so a long-running project ends up with a wall of finished agents that
        hides the ones actually doing work. Archiving (not deleting) keeps cost,
        tokens, and the transcript queryable and leaves the agent available to
        `resume_agent`, while freeing the parent's delegation slot.

        An agent is only archived when it is genuinely finished with the
        project: it was retired, or the task it was assigned to is done; it is
        not running, it has no live children, and it is not the reviewer of
        record on a card that is still open. A specialist created but not yet
        used also stays - it exists for work that is still coming.
        """
        # Archiving a child can finish its parent, so repeat until the tree
        # settles; otherwise a depth-3 subtree takes one tick per level to
        # clear out.
        lines: list[str] = []
        for _ in range(10):
            made = self._archive_finished_once()
            if not made:
                break
            lines.extend(made)
        if lines:
            self.store.flush()
        return lines

    def _archive_finished_once(self) -> list[str]:
        lines: list[str] = []
        inflight = self.runtime.inflight
        for agent in list(self.store.agents.values()):
            if agent.role == "orchestrator" or agent.archived_at is not None:
                continue
            if agent.id in inflight or agent.child_agent_ids:
                continue
            if agent.status not in ("done", "idle"):
                continue

            # An agent the orchestrator created but has not used yet stays on
            # the tree: it was created for work that is still coming.
            if agent.task_id is None and agent.status != "done":
                continue
            task = self.store.tasks.get(agent.task_id or "")
            if task is not None and task.column != "done":
                continue
            if any(t.reviewer_agent_id == agent.id and t.column != "done"
                   for t in self.store.tasks.values()):
                continue
            reason = ("retired after finishing its work" if agent.status == "done"
                      else "assigned task is done")

            agent.archived_at = now_iso()
            agent.retirement_reason = reason
            agent.touch()
            if agent.parent_agent_id and agent.parent_agent_id in self.store.agents:
                parent = self.store.agents[agent.parent_agent_id]
                parent.child_agent_ids = [
                    c for c in parent.child_agent_ids if c != agent.id]
                self.store.upsert_agent(parent)
            self.store.add_event(
                "agent_archived",
                f"Archived {agent.role} ({agent.id}) from the hierarchy: {reason}.",
                project_id=agent.project_id, agent_id=agent.id)
            lines.append(f"Archived {agent.role} ({agent.id}): {reason}.")
        return lines

    def run_cycle(self) -> list[CycleReport] | None:
        """Run one scheduler cycle, unless an orchestrator is already busy.

        Returns None when the cycle was skipped. There is deliberately no way
        to queue a cycle behind a running one: a queued cycle would run a
        briefing built from the board as it looked *before* the running cycle
        changed it, so it would act on stale state and then write over the
        fresher decisions. Refusing is correct; waiting is not.

        The lock acquire is the atomic claim, so the queue check runs while it
        is held: no window opens between "is one running?" and "start one".
        """
        if not self._cycle_lock.acquire(blocking=False):
            self._note_cycle_skip("another cycle was still running")
            return None
        try:
            busy = self.runtime.llm_busy("orchestrator")
            if busy is not None:
                self._note_cycle_skip(
                    f"an orchestrator prompt is already {busy['status']} "
                    f"({busy['id']})")
                return None
            return self._run_cycle()
        finally:
            self._cycle_lock.release()

    def _run_scheduler(self) -> list[tuple[str | None, str]]:
        try:
            return self.scheduler.sweep()
        except Exception:  # noqa: BLE001 - dispatch is not worth killing a tick
            logger.exception("Scheduler sweep failed")
            return []

    def _note_cycle_skip(self, why: str) -> None:
        started = int(self.store.meta.get("cycle_started_at") or time.time())
        self.store.meta["cycle_skips"] = int(
            self.store.meta.get("cycle_skips", 0)) + 1
        # How long the in-flight cycle has been holding the loop, and why the
        # new one was refused. The dashboard turns this into an alert: a loop
        # that refuses cycle after cycle is the "nothing is moving" symptom, and
        # a bare skip counter does not say what is blocking it.
        self.store.meta["cycle_stuck_seconds"] = int(time.time() - started)
        self.store.meta["cycle_skip_reason"] = why
        # Coalesced: a long cycle makes the loop retry many times, and one
        # event per retry buries the feed in identical noise. The counter
        # still increments every time.
        last = self._last_skip_event_at
        if last is None or time.time() - last >= 60:
            self._last_skip_event_at = time.time()
            self.store.add_event(
                "cycle_skipped",
                f"Cycle skipped: {why} (previous started "
                f"{int(time.time() - started)}s ago).")
        logger.info("Skipping cycle: %s", why)

    def cycle_running(self) -> bool:
        """True while a cycle holds the loop."""
        return self._cycle_lock.locked()

    def orchestrator_busy(self) -> str | None:
        """Why an orchestrator cycle cannot start now, or None if it can.

        Covers both halves of the user's rule: one executing, and one queued
        waiting on an LLM slot. The queued half matters because a prompt that
        has been written but not yet dispatched is already committed to a
        briefing that the next cycle would supersede.
        """
        if self._cycle_lock.locked():
            return "a cycle is already running"
        busy = self.runtime.llm_busy("orchestrator")
        if busy is not None:
            return (f"an orchestrator prompt is already {busy['status']} "
                    f"for {int(time.time() - busy['queued_at'])}s")
        return None

    def _run_cycle(self) -> list[CycleReport]:
        """One scheduler tick: reload config, sweep, reconcile, detect, act."""
        # First thing every tick: settings may have been edited through the UI
        # or by hand, and the whole point is that the harness adapts without a
        # restart. The notes describe the new capacity so the briefing can tell
        # the orchestrator to use it (or hold back).
        self.config_notes = self.reload_config_if_changed()
        self.sweep_orphans()
        self.sweep_finished_agents()
        self.reconcile()
        attention_map = self.detect()
        for project_id, notes in self._config_attention().items():
            attention_map.setdefault(project_id, []).extend(notes)
        for project_id, note in self._scheduler_notes:
            if project_id:
                attention_map.setdefault(project_id, []).append(note)
        reports: list[CycleReport] = []

        for project in self.store.visible_projects():
            if project.is_frozen:
                continue
            attention = attention_map.get(project.id, [])
            if not attention:
                continue

            reports.append(self._run_project_cycle(project, attention))

        # Idle = nothing needed a decision and nothing is in flight. Only that
        # combination justifies a long sleep: with agents running, a finishing
        # run can unblock a card, and the loop has to notice reasonably soon.
        self._cycle_had_work = bool(reports) or bool(self.runtime.inflight)
        self._record_cycle_end("ok")
        return reports

    def _record_cycle_end(self, status: str,
                          error: str | None = None) -> None:
        """Stamp cycle metadata.

        last_tick alone is written only on success, so a harness whose cycles
        keep failing shows a timestamp frozen at the first completed cycle -
        which reads like 'last cycle never updates'. The status and failure
        fields make the stall visible instead.
        """
        now = int(time.time())
        meta = self.store.meta
        started = meta.get("cycle_started_at")
        # A cycle that ran ends the stall: leaving the skip counters set keeps
        # the alert up after the problem has cleared.
        if int(meta.get("cycle_skips", 0)):
            meta["cycle_skips"] = 0
            meta["cycle_stuck_seconds"] = 0
            meta["cycle_skip_reason"] = None
        meta["cycle_count"] = int(meta.get("cycle_count", 0)) + 1
        meta["cycle_status"] = status
        meta["cycle_duration"] = (now - int(started)) if started else 0
        meta["cycle_error"] = error
        if status == "ok":
            meta["last_tick"] = now
            meta["last_cycle_completed_at"] = now
        meta["last_cycle_attempt_at"] = now
        meta["loop_started_at"] = meta.get("loop_started_at") or now

    def _config_attention(self) -> dict[str, list[str]]:
        """Turn config changes into instructions the orchestrator must act on.

        Broadcast to every project: capacity is shared, so a new LLM server or
        an extra slot is relevant to each one's queue.
        """
        notes = getattr(self, "config_notes", [])
        if not notes:
            # Capacity is in every briefing (cycle_briefing's capacity section),
            # so it must NOT be an attention item: attention is what decides
            # whether a project gets an orchestrator cycle at all, and making it
            # always-present would run a paid cycle per project per tick even
            # when there is nothing to decide.
            return {}
        cap = self.runtime.capacity()
        lines = [f"CONFIG CHANGED (applied this tick): {n}" for n in notes]
        lines.append(
            f"CAPACITY NOW: {cap['slots_free']} of {cap['worker_slots']} "
            f"sub-agent slots free ({cap['workers_active']} running), "
            f"{cap['queued']} action(s) already queued, max "
            f"{cap['max_concurrent_sub_agents_per_project']} running per "
            f"project, {len(cap['llm_servers'])} LLM server(s) in rotation "
            f"({', '.join(cap['llm_servers'])}). Creating sub-agents is "
            "unlimited; running them at once is not.")
        if cap["slots_free"] > 0:
            lines.append(
                "More capacity than before: dispatch additional ready cards in "
                "the background now instead of leaving them in ready/in_progress.")
        return {pid: lines for pid in self.store.projects}

    def _run_project_cycle(self, project, attention: list[str]) -> CycleReport:
        board = {col: [self._task_to_dict(t) for t in items]
                 for col, items in self.store.board(project.id).items()}
        agents = [self._agent_to_dict(a) for a in self.store.project_agents(project.id)]
        archived = [
            {"id": a.id, "role": a.role, "specialty": a.specialty,
             "turns": f"{a.turns_used}/{a.turn_limit}", "why": a.retirement_reason}
            for a in self.store.project_agents(project.id, include_archived=True)
            if a.archived_at is not None
        ]
        briefing = prompts.cycle_briefing(
            {"id": project.id, "name": project.name, "goal": project.goal,
             "constraints": project.constraints, "workspace": project.workspace,
             "status": project.status,
             "user_messages": project.user_messages,
             "input_files": project.input_files},
            board, agents, attention, archived=archived,
            capacity=self.runtime.capacity(),
        )

        # Refresh the orchestrator's per-cycle budget; its conversation persists.
        self.spec.turn_limit = self.spec.turns_used + self.config.orchestrator_turn_limit
        self.spec.task_id = None
        self.spec.project_id = project.id
        self.spec.touch()

        # The queued user input is now in the briefing, so it is consumed here:
        # keeping it queued would replay the same request every cycle. If the
        # cycle fails, the messages go back so the request is never silently
        # dropped.
        queued = list(project.user_messages)
        if queued:
            project.user_messages = []
            self.store.flush()

        tool_specs = [Tool(name=n) for n in harness_tools.ORCHESTRATOR_TOOLS]
        try:
            outcome = self.runtime.run(self.spec, tool_specs, briefing,
                                       turn_limit=self.spec.turn_limit,
                                       kind="orchestrator")
        except Exception:
            if queued:
                project.user_messages = queued + project.user_messages
                self.store.flush()
            raise
        # A run that never reached the model (endpoint down, agent failed before
        # acting) consumed the briefing without acting on it. Put the user's
        # request back so it is re-briefed next cycle instead of vanishing.
        if queued and outcome.status in ("llm_unavailable", "failed"):
            project.user_messages = queued + project.user_messages
            self.store.flush()

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
            "parent_task_id": task.parent_task_id,
            "turns_used": task.turns_used, "turn_limit": task.turn_limit,
            "review_rounds": task.review_rounds,
            "blocked_cause": task.blocked_cause,
            "block_strikes": task.block_strikes,
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
        # The loop may be parked in a long idle wait; wake it so shutdown is
        # prompt regardless of the current interval.
        self._wake.set()

    def wake(self, reason: str = "") -> None:
        """Cut the idle wait short so new work is picked up promptly.

        An idle loop deliberately sleeps for a long time; without this, a
        project created by the user could sit untouched for the whole idle
        interval.
        """
        self._cycle_durations.clear()
        self._wake.set()
        if reason:
            logger.info("Loop woken early: %s", reason)

    def _next_tick(self, elapsed: float, had_work: bool = True) -> float:
        """Re-derive the scheduler interval from how long the last cycle took.

        A fixed interval cannot be right: a cycle that runs long makes the loop
        fall behind (and a manual tick then queues an orchestrator request on
        top of a conversation that is still busy), while a cycle that finishes
        quickly leaves the harness idle for the rest of the interval. The
        measured duration is turned into an interval that keeps the loop about
        `tick_duty_cycle` busy, bounded by min/max_tick_seconds.

        An empty cycle costs no LLM call - detect() found nothing to act on and
        the orchestrator was never asked - so letting the interval fall to the
        floor when the harness is idle is what makes new work get picked up
        promptly. The deque keeps the slowest recent cycle, so one expensive
        cycle holds the interval wide for several ticks rather than collapsing
        after a run of cheap ones.
        """
        cfg = self.config
        # min_tick_seconds bounds how aggressively the auto-tuner speeds up; it
        # must not override an explicitly configured faster tick, so the floor
        # is the smaller of the two.
        floor = max(1, min(cfg.min_tick_seconds, cfg.tick_seconds))
        ceiling = max(floor, cfg.max_tick_seconds)
        if not cfg.dynamic_tick:
            return float(max(floor, min(cfg.tick_seconds, ceiling)))

        # Nothing to act on and nothing running: there is no value in coming
        # back quickly, and the cycle cost no LLM call either way. Back off
        # toward idle_tick_seconds instead of letting the duty-cycle maths pull
        # the interval to the floor.
        current = self.effective_tick
        if not had_work:
            ceiling = max(floor, cfg.idle_tick_seconds, cfg.max_tick_seconds)
            idle = float(max(floor, min(ceiling, cfg.idle_tick_seconds)))
            if current < idle:
                step = current * max(1.0, cfg.idle_backoff_factor)
                self.effective_tick = float(min(idle, max(step, floor + 1)))
            else:
                self.effective_tick = idle
            self._cycle_durations.clear()
            return self.effective_tick

        self._cycle_durations.append(max(0.05, elapsed))
        # Keep the longest recent cycle in the average: one slow cycle is a
        # better signal of the real cost than several empty ones.
        # deque has no slice syntax; list() first.
        recent = max(list(self._cycle_durations)[-6:])
        target = recent / max(0.05, min(1.0, cfg.tick_duty_cycle))
        # Asymmetric: grow quickly when falling behind, shrink slowly so a
        # burst of cheap cycles does not make the loop spin.
        if target > current:
            new = target
        else:
            new = current + (target - current) * 0.35
        new = float(max(floor, min(ceiling, new)))
        self.effective_tick = new
        return new

    def run_forever(self) -> None:
        self.effective_tick = float(max(1, self.config.tick_seconds))
        logger.info("Orchestrator loop starting (tick=%ss)", self.config.tick_seconds)
        self.store.add_event("loop_started",
                             f"Orchestrator loop started (tick {self.config.tick_seconds}s).")
        self.store.meta["loop_started_at"] = int(time.time())
        self.store.flush()
        while not self._stop.is_set():
            # Cleared at the top of the iteration, not before the sleep: a wake
            # that arrives mid-cycle must still shorten the following wait.
            self._wake.clear()
            started = time.time()
            self.store.meta["cycle_started_at"] = int(started)
            self.store.meta["cycle_status"] = "running"
            self.store.flush()
            # Before the cycle, and regardless of whether one is allowed to
            # run: this is the part of orchestration that needs no model, so it
            # must not be gated on an orchestrator slot. A cycle that keeps
            # getting skipped because one orchestrator prompt is stuck should
            # still start work that is already staffed and unblocked.
            self._scheduler_notes = self._run_scheduler()
            reports = None
            try:
                reports = self.run_cycle()
                for r in (reports or []):
                    logger.info("cycle %s: %d attention items, status=%s",
                                r.project_id, len(r.attention), r.orchestrator_status)
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.exception("Orchestrator cycle failed")
                detail = f"{type(exc).__name__}: {exc}"
                self.store.add_event("loop_error", f"Cycle failed: {detail}")
                self._record_cycle_end("failed", detail)
            elapsed = time.time() - started
            # A skipped cycle costs ~0s. Feeding that to the tuner collapses the
            # interval to the floor, which then makes the loop retry (and skip)
            # every floor-length tick while one long cycle is in flight.
            wait = (self._next_tick(elapsed, self._cycle_had_work)
                    if reports is not None else self.effective_tick)
            self.store.meta["effective_tick"] = round(wait, 1)
            self.store.meta["last_cycle_seconds"] = round(elapsed, 1)
            self.store.flush()
            # Interruptible: a new project or new user input should not have to
            # wait out an idle-length interval before anyone looks at it.
            self._wake.wait(timeout=max(1.0, wait))
        self.runtime.close_all()
