"""Runs agent conversations under a hard turn budget.

A budget is enforced by capping ``max_iteration_per_run`` for a single run and
counting cumulative agent turns across runs. Because conversations persist, a
sub-agent that runs out of budget can be resumed later with a new budget, which
is exactly what the orchestrator decides on when it is told the budget ran out.

Agents do not talk to an LLM endpoint. Every agent -- orchestrator, sub-agent,
and condenser -- is built against the LLM router, which owns endpoint choice,
request formatting, headers, and load balancing. See ``llm_router.py``.

Slow local endpoints are handled by widening the request timeout on each
timeout (shared per endpoint, so one slow reply helps every agent) and by
retrying runs that make no progress.
"""

from __future__ import annotations

import contextlib
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import openai
from openhands.sdk import Agent, Conversation, LLM, LLMSummarizingCondenser
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.conversation.visualizer import ConversationVisualizerBase
from openhands.sdk.event.base import Event
from openhands.sdk.event.llm_convertible import ActionEvent, MessageEvent
from openhands.sdk.logger import get_logger
from openhands.sdk.security.confirmation_policy import AlwaysConfirm, NeverConfirm

from .config import HarnessConfig
from .store import AGENT_STATUSES, AgentSpec, Store, new_id

logger = get_logger(__name__)

# Errors that mean "the endpoint was too slow / unreachable / overloaded", not
# "the agent did something wrong". These trigger a retry. Deliberately excludes
# BadRequestError / AuthenticationError (both APIStatusError subclasses):
# retrying those just burns time on a permanent failure. APITimeoutError
# subclasses APIConnectionError, so it is covered.
TRANSIENT_LLM_ERRORS = (
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
)

# Only these justify concluding "the endpoint needs a longer timeout". A rate
# limit needs a backoff, not a bigger deadline.
TIMEOUT_LIKE_ERRORS = (openai.APIConnectionError,)


class ConcurrencyGate:
    """Bounds how many threads may hold the LLM at once; limit is adjustable.

    Slots are held per thread, and a thread that already holds one never takes
    a second. A sub-agent dispatched synchronously from inside its parent's
    tool call runs on the parent's thread, so re-entrant acquisition is what
    prevents a limit of 1 from deadlocking a nested dispatch.
    """

    def __init__(self, limit: int):
        self._limit = max(1, limit)
        self._holders: dict[int, int] = {}
        self._cond = threading.Condition()

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    @property
    def active(self) -> int:
        with self._cond:
            return len(self._holders)

    def set_limit(self, limit: int) -> None:
        with self._cond:
            self._limit = max(1, limit)
            self._cond.notify_all()

    def acquire(self) -> None:
        ident = threading.get_ident()
        with self._cond:
            if ident in self._holders:
                self._holders[ident] += 1
                return
            while len(self._holders) >= self._limit:
                self._cond.wait()
            self._holders[ident] = 1

    def release(self) -> None:
        ident = threading.get_ident()
        with self._cond:
            count = self._holders.get(ident)
            if count is None:
                return
            if count <= 1:
                del self._holders[ident]
            else:
                self._holders[ident] = count - 1
            self._cond.notify()

    @contextlib.contextmanager
    def slot(self):
        self.acquire()
        try:
            yield
        finally:
            self.release()


@dataclass
class TurnCounter:
    """Counts agent turns by walking conversation events."""

    seen_response_ids: set[str] = field(default_factory=set)

    @staticmethod
    def count(conversation: Conversation) -> int:
        """Total agent turns in a conversation.

        One turn == one LLM response. ``ActionEvent`` always carries the
        ``llm_response_id`` of the response that produced it; a plain-text
        ``MessageEvent`` may not, so it is counted on its own.
        """
        return len(_response_ids(conversation.state.events)) + _standalone_messages(
            conversation.state.events
        )


def _response_ids(events) -> set[str]:
    ids: set[str] = set()
    for event in events:
        if event.source != "agent":
            continue
        # Not every agent-sourced event is an LLM turn: SystemPromptEvent and
        # condenser events carry no llm_response_id at all.
        rid = getattr(event, "llm_response_id", None)
        if rid or isinstance(event, ActionEvent):
            ids.add(str(rid or event.id))
    return ids


def _standalone_messages(events) -> int:
    return sum(
        1
        for event in events
        if event.source == "agent"
        and isinstance(event, MessageEvent)
        and not getattr(event, "llm_response_id", None)
    )


class LiveTurnVisualizer(ConversationVisualizerBase):
    """Publishes turn usage to the store while a run is still in progress.

    Without this, a long sub-agent run shows a stale count until it finishes.
    The count is recomputed from the live event store on each event, which
    stays correct across resume and condensation without needing to seed or
    deduplicate ids.
    """

    def __init__(self, spec: AgentSpec, store: Store):
        super().__init__()
        self.spec = spec
        self.store = store
        self._updated_at = 0.0
        self._flushed_at = 0.0

    def on_event(self, event: Event) -> None:
        if self._state is None or event.source != "agent":
            return

        now = time.time()
        # Recomputing walks the whole event store, so throttle it; the final
        # count is reconciled by run() regardless.
        if now - self._updated_at < 1.0:
            return
        self._updated_at = now

        events = list(self._state.events)
        total = len(_response_ids(events)) + _standalone_messages(events)
        # Never regress: a condenser can prune events out of the store.
        if total > self.spec.turns_used:
            self.spec.turns_used = total
            self.store.upsert_agent(self.spec)
            if now - self._flushed_at >= 5.0:
                self._flushed_at = now
                self.store.flush()


@dataclass
class RunOutcome:
    turns_used: int
    turns_this_run: int
    status: str
    final_report: str
    budget_exhausted: bool
    tokens: int
    cost_usd: float
    error: str | None = None
    # Set when the run was retried because the endpoint timed out or returned
    # nothing, so the endpoint can be seen as unhealthy rather than treating it
    # as an agent failure.
    llm_retries: int = 0
    llm_health: str | None = None
    # Consecutive LLM transport failures for this agent, including this run.
    # Drives the harness's own response (block the card, alert the user) so an
    # unreachable endpoint never becomes the orchestrator's problem to solve.
    llm_failures: int = 0
    llm_failure_error: str | None = None


@dataclass
class LLMQueueEntry:
    """One agent action waiting for, or currently using, the LLM.

    The dashboard shows these as a queue: entries are created when a run is
    requested and removed when it finishes, so 'queued' means the prompt is
    written and waiting on a concurrency slot, and 'executing' means the LLM
    has it. The prompt is kept here so the user can read exactly what the
    harness is about to send.
    """

    id: str
    seq: int
    agent_id: str
    agent_role: str
    project_id: str | None
    task_id: str | None
    kind: str            # orchestrator | worker | reviewer
    depth: int
    prompt: str
    queued_at: float
    started_at: float | None = None
    status: str = "queued"   # queued | executing


class AgentRuntime:
    """Creates, resumes and runs agent conversations with budget enforcement."""

    def __init__(self, config: HarnessConfig, store: Store, router=None):
        self.config = config
        self.store = store
        # Every LLM request goes through this. None only in tests that never
        # make a real request.
        self.router = router
        self._conversations: dict[str, Conversation] = {}
        # conversation_id -> agent_id, populated as conversations open.
        self.conversation_index: dict[str, str] = {}
        # Guards the _conversations / conversation_index maps only. Runs are
        # serialised per agent, not globally, so background dispatch is parallel.
        self._lock = threading.RLock()
        self._agent_locks: dict[str, threading.Lock] = {}
        self._pool = ThreadPoolExecutor(
            # The pool must be larger than the gate. If they are the same size,
            # every dispatched thread immediately wins a gate slot, so the
            # backlog hides in the pool's own work queue and the dashboard's
            # 'queued' lane is always empty.
            max_workers=self._pool_size(config),
            thread_name_prefix="worker",
        )
        # Refcount, not a set: submit() marks a background dispatch and the run()
        # it eventually calls marks the same agent again, so a plain set would be
        # cleared by the inner unmark while the outer one is still in flight.
        self._inflight: dict[str, int] = {}
        # Gate slots currently held by an orchestrator run. The orchestrator
        # holds one for the whole cycle, so capacity() can report how many
        # slots are genuinely left for workers.
        self._orchestrator_slots = 0
        # Live LLM tuning, shared by every agent: one slow reply widens the
        # timeout for all of them.
        self._llm_lock = threading.RLock()
        self._llms: dict[str, list] = {}
        # A widened timeout is kept in the store so a restart does not go back
        # to a value the endpoint already proved too small.
        self.llm_timeout = int(store.meta.get(
            "llm_timeout", config.llm.request_timeout) or config.llm.request_timeout)
        self.llm_timeouts_widened = int(store.meta.get("llm_timeouts_widened", 0))
        self.llm_empty_responses = int(store.meta.get("llm_empty_responses", 0))
        self.llm_transient_errors = int(store.meta.get("llm_transient_errors", 0))
        self.gate = ConcurrencyGate(self._gate_limit(config))
        # Agent actions awaiting the LLM, in request order. Entries live only
        # for the duration of a run; the dashboard renders them as a queue.
        self._queue_lock = threading.Lock()
        self._llm_queue: list[LLMQueueEntry] = []
        self._queue_seq = 0

    @staticmethod
    def _gate_limit(config: HarnessConfig) -> int:
        """Gate slots: one per worker, plus one reserved for the orchestrator.

        The orchestrator's own run holds a slot for the entire cycle, and a
        worker dispatched from inside its tool call reuses that slot only while
        it runs on the orchestrator's thread. Background dispatches get their
        own thread and therefore their own slot, so the orchestrator has to
        keep one reserved or it can be starved by its own children and never
        finish the cycle that dispatched them.
        """
        return max(1, config.max_parallel_workers) + 1

    @staticmethod
    def _pool_size(config: HarnessConfig) -> int:
        """Threads the pool may run: gate slots plus the visible queue backlog."""
        return AgentRuntime._gate_limit(config) + max(0, config.llm_queue_depth)

    def orchestrator_slots(self) -> int:
        """Gate slots currently held by an orchestrator run."""
        with self._lock:
            return self._orchestrator_slots

    def _note_orchestrator_slot(self, delta: int) -> None:
        with self._lock:
            self._orchestrator_slots = max(0, self._orchestrator_slots + delta)

    def set_worker_slots(self, workers: int) -> int:
        """Re-derive the gate limit from a new worker count."""
        limit = max(1, int(workers)) + 1
        self.gate.set_limit(limit)
        return limit

    def capacity(self) -> dict:
        """How much LLM capacity exists right now, for scheduling decisions."""
        with self._queue_lock:
            queued = sum(1 for e in self._llm_queue if e.status == "queued")
        limit = self.gate.limit
        active = self.gate.active
        # One gate slot is reserved for the orchestrator and is not worker
        # capacity whether or not the orchestrator happens to be holding it
        # right now. Reporting the raw gate number is what made the harness
        # under-dispatch by one.
        orchestrator_holds = self.orchestrator_slots()
        reserved = max(1, orchestrator_holds)
        worker_slots = max(0, limit - reserved)
        workers_active = max(0, active - orchestrator_holds)
        return {
            "llm_slots": limit,
            "worker_slots": worker_slots,
            "workers_active": workers_active,
            "slots_in_use": active,
            "slots_free": max(0, worker_slots - workers_active),
            "queued": queued,
            "pool_size": self._pool._max_workers,
            # Config truth, so scheduling advice holds even before the router
            # has probed anything.
            "llm_servers": [srv.name for srv in self.config.servers()],
            "router_selection": self.config.router_selection,
        }

    def sync_pool_size(self) -> None:
        """Re-derive the pool ceiling from the current config."""
        self.set_pool_size(self._pool_size(self.config))

    def set_pool_size(self, limit: int) -> None:
        """Raise the pool ceiling when the gate limit or queue depth is patched.

        Grow-only: a stdlib worker thread only exits on a shutdown sentinel, so
        lowering the ceiling cannot reclaim threads that already exist. That is
        harmless here — surplus threads park on the work queue, and the gate,
        not the pool, is what bounds real LLM concurrency.
        """
        size = max(1, limit)
        if size > self._pool._max_workers:
            self._pool._max_workers = size

    @property
    def inflight(self) -> set[str]:
        """Agent ids with a run in progress right now.

        Authoritative for every run, including the orchestrator's own, unlike
        ``status == "running"``, which is a persisted field that can be left
        stale by a crash mid-run.
        """
        with self._lock:
            return {aid for aid, n in self._inflight.items() if n > 0}

    def _mark_inflight(self, agent_id: str) -> None:
        with self._lock:
            self._inflight[agent_id] = self._inflight.get(agent_id, 0) + 1

    def _unmark_inflight(self, agent_id: str) -> None:
        with self._lock:
            left = self._inflight.get(agent_id, 0) - 1
            if left > 0:
                self._inflight[agent_id] = left
            else:
                self._inflight.pop(agent_id, None)

    def submit(self, agent_id: str, fn) -> None:
        """Run a worker stretch off the loop thread."""
        # Check and mark in one lock: the guard is what stops a task being
        # dispatched twice, so it must not be racy.
        with self._lock:
            if self._inflight.get(agent_id, 0) > 0:
                raise RuntimeError(f"Agent {agent_id} is already running")
            self._inflight[agent_id] = 1

        def guarded() -> None:
            try:
                fn()
            finally:
                self._unmark_inflight(agent_id)

        self._pool.submit(guarded)

    # ---------- construction ----------

    def _persistence_dir(self, agent: AgentSpec) -> str:
        if agent.persistence_dir:
            return agent.persistence_dir
        scope = agent.project_id or "global"
        return str(self.config.root_dir / "conversations" / scope / agent.id)

    def build_agent(self, spec: AgentSpec, tools: list,
                    server_name: str | None = None) -> Agent:
        """Build the SDK Agent for a stored spec.

        The agent is pointed at the LLM router, never at an endpoint: it supplies
        a prompt and the router decides where that prompt goes. The attribution
        headers are what make the router's per-request cards meaningful, and
        they travel with every request this conversation makes.
        """
        llm = self._router_llm(spec, spec.id)
        condenser_llm = self._router_llm(spec, f"{spec.id}-condenser")
        condenser = LLMSummarizingCondenser(
            llm=condenser_llm,
            max_size=120,
            keep_first=2,
        )
        # The SDK keeps our LLM objects by reference, so mutating their timeout
        # later retunes a live conversation without rebuilding or resetting it.
        # Keyed by agent so closing a conversation drops its LLMs again.
        with self._llm_lock:
            self._llms[spec.id] = [llm, condenser_llm]
        return Agent(
            llm=llm,
            tools=list(tools),
            system_prompt=spec.system_prompt,
            security_policy_filename="",
            condenser=condenser,
            include_default_tools=["FinishTool", "ThinkTool"],
        )

    def _router_llm(self, spec: AgentSpec, usage_id: str) -> LLM:
        """An SDK LLM whose only destination is the router.

        `model` is a placeholder the router rewrites per endpoint: an Ollama
        server needs `qwen2.5:7b` where an OpenAI-style gateway needs
        `openai/qwen`, and only the router knows which endpoint is being used.
        It still has to carry the `openai/` prefix: litellm picks the provider
        off the model string before it ever opens a connection, so an unprefixed
        name fails locally with "LLM Provider NOT provided".
        Without a router (unit tests that never send a request) the agent keeps
        its configured endpoint.
        """
        cfg = self.config.llm
        if self.router is None:
            return LLM(**cfg.build_kwargs(
                usage_id, self.llm_timeout, cfg.num_retries, cfg.drop_params))
        from .llm_router import (
            H_ATTR_AGENT, H_ATTR_KIND, H_ATTR_PROJECT, H_ATTR_ROLE, H_ATTR_TASK)
        headers = {
            H_ATTR_AGENT: spec.id,
            H_ATTR_ROLE: spec.role or "",
            H_ATTR_KIND: self._classify_run(spec, []),
        }
        if spec.project_id:
            headers[H_ATTR_PROJECT] = spec.project_id
        if spec.task_id:
            headers[H_ATTR_TASK] = spec.task_id
        return LLM(
            model="openai/harness-routed",
            api_key="router",
            base_url=self.router.base_url,
            api_mode="chat",
            timeout=self.llm_timeout,
            stream_idle_timeout=float(self.llm_timeout),
            num_retries=0,
            drop_params=cfg.drop_params,
            caching_prompt=False,
            usage_id=usage_id,
            extra_headers=headers,
            max_input_tokens=cfg.max_input_tokens,
            max_output_tokens=cfg.max_output_tokens,
            input_cost_per_token=cfg.input_cost_per_token,
            output_cost_per_token=cfg.output_cost_per_token,
        )

    # ---------- live LLM tuning ----------

    def set_llm_timeout(self, seconds: int) -> int:
        """Apply a new request timeout to every live agent LLM."""
        cfg = self.config.llm
        bounded = int(max(cfg.min_request_timeout,
                          min(seconds, cfg.max_request_timeout)))
        with self._llm_lock:
            self.llm_timeout = bounded
            for agent_id, llms in list(self._llms.items()):
                for llm in llms:
                    try:
                        llm.timeout = bounded
                        llm.stream_idle_timeout = float(bounded)
                    except Exception:  # noqa: BLE001 - LLM torn down under us
                        logger.debug("Dropping stale LLM for %s", agent_id)
        self.llm_timeouts_widened += 1
        self._persist_llm_health()
        logger.warning("LLM request timeout adjusted to %ss", bounded)
        return bounded

    def widen_llm_timeout(self, reason: str) -> int:
        """Grow the timeout geometrically after a slow/unreachable endpoint."""
        cfg = self.config.llm
        with self._llm_lock:
            current = self.llm_timeout
        if not cfg.adaptive_timeout or current >= cfg.max_request_timeout:
            return current
        grown = max(int(current * cfg.timeout_growth),
                    current + cfg.min_request_timeout)
        return self.set_llm_timeout(grown)

    def _persist_llm_health(self) -> None:
        self.store.meta.update({
            "llm_timeout": self.llm_timeout,
            "llm_timeouts_widened": self.llm_timeouts_widened,
            "llm_empty_responses": self.llm_empty_responses,
            "llm_transient_errors": self.llm_transient_errors,
        })

    def _agent_lock(self, agent_id: str) -> threading.Lock:
        with self._lock:
            return self._agent_locks.setdefault(agent_id, threading.Lock())

    def open_conversation(self, spec: AgentSpec, tools: list) -> Conversation:
        """Open (or reuse) the persistent conversation for an agent spec."""
        with self._lock:
            existing = self._conversations.get(spec.id)
            if existing is not None:
                return existing

            persistence_dir = self._persistence_dir(spec)
            if spec.conversation_id:
                conv_id = uuid.UUID(spec.conversation_id)
            else:
                conv_id = uuid.uuid4()
                spec.conversation_id = str(conv_id)
                spec.persistence_dir = persistence_dir
                self.store.upsert_agent(spec)

            conv = Conversation(
                agent=self.build_agent(spec, tools),
                workspace=spec.workspace or str(self.config.root_dir),
                persistence_dir=persistence_dir,
                conversation_id=conv_id,
                visualizer=LiveTurnVisualizer(spec, self.store),
                stuck_detection=True,
                delete_on_close=False,
                max_iteration_per_run=self._budget_for(spec),
            )
            policy = AlwaysConfirm() if self.config.confirm_tools else NeverConfirm()
            conv.set_confirmation_policy(policy)
            self._conversations[spec.id] = conv
            self.conversation_index[str(conv.id)] = spec.id
            return conv

    def _budget_for(self, spec: AgentSpec) -> int:
        """Iterations allowed for the next run of this agent."""
        remaining = max(0, spec.turn_limit - spec.turns_used)
        return max(1, remaining)

    # ---------- LLM action queue ----------

    def _enqueue_llm(self, spec: AgentSpec, message: str, kind: str) -> LLMQueueEntry:
        # No endpoint is chosen here. The router picks one when it actually
        # dispatches, so a run waiting on a slot is not silently pinned to a
        # server that may be busy or down by the time it moves.
        entry = LLMQueueEntry(
            id=new_id("llm"), seq=self._next_seq(), agent_id=spec.id,
            agent_role=spec.role, project_id=spec.project_id,
            task_id=spec.task_id, kind=kind, depth=spec.depth,
            prompt=message, queued_at=time.time(),
        )
        with self._queue_lock:
            self._llm_queue.append(entry)
        return entry

    def _next_seq(self) -> int:
        with self._queue_lock:
            self._queue_seq += 1
            return self._queue_seq

    def _begin_llm(self, entry: LLMQueueEntry) -> None:
        """Mark the entry as holding the LLM (it has its concurrency slot)."""
        with self._queue_lock:
            entry.status = "executing"
            entry.started_at = time.time()

    def _drop_llm(self, entry: LLMQueueEntry) -> None:
        with self._queue_lock:
            try:
                self._llm_queue.remove(entry)
            except ValueError:
                pass

    def llm_queue(self) -> list[dict]:
        """Queue snapshot in request order: oldest first, newest last.

        The UI renders this reversed (newest on the left) with the executing
        entry pinned to the right, so the order here is the source of truth and
        the display is a presentation choice.
        """
        with self._queue_lock:
            ordered = sorted(self._llm_queue, key=lambda e: e.seq)
            return [{
                "id": e.id, "seq": e.seq, "agent_id": e.agent_id,
                "agent_role": e.agent_role, "project_id": e.project_id,
                "task_id": e.task_id, "kind": e.kind, "depth": e.depth,
                "status": e.status, "queued_at": e.queued_at,
                "started_at": e.started_at,
                "waiting_seconds": round(
                    (e.started_at or time.time()) - e.queued_at, 1),
                "running_seconds": round(
                    time.time() - e.started_at, 1) if e.started_at else None,
                "prompt_chars": len(e.prompt),
                "prompt_preview": e.prompt[:400],
            } for e in ordered]

    def llm_busy(self, kind: str) -> dict | None:
        """The queued-or-executing LLM entry of a given kind, oldest first.

        Used to refuse a new orchestrator cycle while an orchestrator prompt is
        already committed: a second one would carry a briefing built from a
        board the first is still changing.
        """
        with self._queue_lock:
            for e in sorted(self._llm_queue, key=lambda x: x.seq):
                if e.kind == kind:
                    return {"id": e.id, "status": e.status, "kind": e.kind,
                            "agent_id": e.agent_id, "queued_at": e.queued_at,
                            "started_at": e.started_at}
        return None

    def llm_prompt(self, entry_id: str) -> str | None:
        with self._queue_lock:
            for e in self._llm_queue:
                if e.id == entry_id:
                    return e.prompt
        return None

    def run(self, spec: AgentSpec, tools: list, message: str,
            turn_limit: int | None = None, kind: str | None = None) -> RunOutcome:
        """Run one budgeted stretch of an agent's conversation.

        A run that produces no agent turn at all — an empty completion, or an
        endpoint that timed out before answering — is retried with a widened
        timeout instead of being reported as an agent failure.

        The run is queued before the concurrency gate is acquired, so the
        dashboard can show a prompt that is written and waiting on a slot as
        distinct from one the LLM is currently working on.
        """
        entry = self._enqueue_llm(
            spec, message, kind or self._classify_run(spec, tools))
        try:
            with self._agent_lock(spec.id):
                with self.gate.slot():
                    return self._run_with_slot(spec, entry, tools, message,
                                               turn_limit)
        finally:
            self._drop_llm(entry)

    def _run_with_slot(self, spec: AgentSpec, entry: LLMQueueEntry, tools: list,
                       message: str, turn_limit: int | None) -> RunOutcome:
        """Run one stretch while holding a gate slot."""
        is_orchestrator = spec.role == "orchestrator"
        if is_orchestrator:
            self._note_orchestrator_slot(1)
        try:
            self._begin_llm(entry)
            # Tracked here rather than only in submit() so the orchestrator's
            # own run, and any synchronous run_agent, also show as active.
            self._mark_inflight(spec.id)
            try:
                return self._run_guarded(spec, tools, message, turn_limit)
            finally:
                self._unmark_inflight(spec.id)
        finally:
            if is_orchestrator:
                self._note_orchestrator_slot(-1)

    @staticmethod
    def _classify_run(spec: AgentSpec, tools: list) -> str:
        """Label the queue card by what the agent is being asked to do."""
        if spec.role == "orchestrator":
            return "orchestrator"
        names = {getattr(t, "name", "") for t in tools}
        if spec.is_reviewer or "record_review" in names:
            return "reviewer"
        return "worker"

    def _run_guarded(self, spec: AgentSpec, tools: list, message: str,
                     turn_limit: int | None) -> RunOutcome:
        cfg = self.config.llm
        if turn_limit is not None:
            spec.turn_limit = turn_limit

        before_turns = spec.turns_used
        conv = self.open_conversation(spec, tools)
        events_before = len(list(conv.state.events))
        conv.max_iteration_per_run = self._budget_for(spec)

        spec.status = "running"
        spec.runs += 1
        spec.touch()
        self.store.upsert_agent(spec)
        self.store.add_event(
            "agent_run_started",
            f"{spec.role} started run {spec.runs} "
            f"(budget {spec.turn_limit - before_turns} turns).",
            project_id=spec.project_id, agent_id=spec.id, task_id=spec.task_id,
        )

        attempts = 1 + max(0, cfg.empty_response_retries)
        error: str | None = None
        retries = 0
        health: list[str] = []
        retryable = True

        for attempt in range(attempts):
            error = None
            try:
                # A retry resubmits the *same* prompt. Telling the agent about
                # the transport failure ("the model returned nothing", "the
                # request timed out") makes it react to a problem that is not
                # its own - it apologises, re-plans, or burns turns on an
                # endpoint it cannot fix. The prompt that failed is simply sent
                # again, unchanged.
                conv.send_message(message)
                conv.run()
            except TRANSIENT_LLM_ERRORS as exc:
                error = f"{type(exc).__name__}: {exc}"
                self.llm_transient_errors += 1
                self._persist_llm_health()
                # Endpoint health is the router's business now: it parks a
                # failing endpoint and moves on. The runtime only still owns the
                # request timeout, which the router reads live.
                health.append(f"transient:{type(exc).__name__}")
                logger.warning("Agent %s hit a transient LLM error: %s",
                               spec.id, error)
                if isinstance(exc, TIMEOUT_LIKE_ERRORS):
                    timeout = self.widen_llm_timeout(type(exc).__name__)
                    health.append(f"timeout->{timeout}s")
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                error = f"{type(exc).__name__}: {exc}"
                retryable = False  # auth/bad-request/tool faults: retrying wastes budget
                logger.exception("Agent %s run failed", spec.id)

            # max(): a condenser that prunes events can make the count regress,
            # which would silently hand the agent turns back. Never regress.
            turns_now = max(before_turns, TurnCounter.count(conv))
            spec.turns_used = turns_now

            if error is None and turns_now > before_turns:
                break  # the endpoint answered and the agent made progress
            if not retryable or attempt + 1 >= attempts:
                break

            if error is None:
                # Ran clean but produced zero agent turns: an empty completion.
                self.llm_empty_responses += 1
                self._persist_llm_health()
                health.append("empty_response")
                logger.warning(
                    "Agent %s produced no turns on run %s (attempt %d/%d); "
                    "endpoint likely unhealthy", spec.id, spec.runs,
                    attempt + 1, attempts,
                )
            retries += 1
            if cfg.adaptive_timeout:
                self.widen_llm_timeout("no_progress")
            self.store.add_event(
                "llm_retry",
                f"Retrying {spec.role} (attempt {attempt + 2}/{attempts}) after "
                f"{'no output' if error is None else error}; "
                f"timeout now {self.llm_timeout}s.",
                project_id=spec.project_id, agent_id=spec.id, task_id=spec.task_id,
            )
            self.store.flush()

        turns_now = max(before_turns, spec.turns_used)
        turns_this_run = turns_now - before_turns
        spec.turns_used = turns_now

        metrics = conv.state.stats.get_combined_metrics()
        usage = metrics.accumulated_token_usage
        # Metrics are per-conversation: a fresh conversation restarts them at 0,
        # so the agent's lifetime total is this conversation plus the baseline
        # rolled over by reset_conversation.
        spec.tokens = spec.prior_tokens + (
            (usage.prompt_tokens + usage.completion_tokens
             + usage.cache_read_tokens + usage.cache_write_tokens)
            if usage else 0
        )
        spec.cost_usd = spec.prior_cost_usd + float(metrics.accumulated_cost or 0.0)

        status = str(conv.state.execution_status.value
                     if hasattr(conv.state.execution_status, "value")
                     else conv.state.execution_status)
        limit_code = self._run_limit_code(conv, events_before)
        budget_exhausted = limit_code in ("MaxIterationsReached", "MaxBudgetReached")
        no_progress = turns_this_run == 0 and not budget_exhausted

        final_report = self._final_report(conv)
        spec.last_report = final_report or spec.last_report
        spec.last_error = error
        # 'retryable' is what separates an endpoint fault (connection, timeout,
        # rate limit) from an agent fault (auth, bad request, tool crash): only
        # the former is transient, so only the former is infrastructure.
        llm_transport = error is not None and retryable
        spec.status = self._map_status(status, budget_exhausted, error,
                                       no_progress, llm_transport)
        # LLM transport failures are counted here, on the agent, so the harness
        # can stop retrying an endpoint that is simply down. A run that reached
        # the model clears the streak even if the agent itself then failed -
        # that is a different problem with a different fix.
        llm_broken = llm_transport or (
            error is None and no_progress and not budget_exhausted)
        if llm_broken:
            spec.llm_failures += 1
            spec.llm_failure_error = error or "endpoint returned no output"
        else:
            spec.llm_failures = 0
            spec.llm_failure_error = None
        spec.touch()
        self.store.upsert_agent(spec)

        self.store.add_event(
            "agent_run_finished",
            f"{spec.role} finished run {spec.runs}: status={spec.status}, "
            f"turns={spec.turns_used}/{spec.turn_limit}"
            + (f", llm_retries={retries}" if retries else "") + ".",
            project_id=spec.project_id, agent_id=spec.id, task_id=spec.task_id,
        )
        self.store.flush()

        return RunOutcome(
            turns_used=spec.turns_used,
            turns_this_run=turns_this_run,
            status=spec.status,
            final_report=final_report,
            budget_exhausted=budget_exhausted,
            tokens=spec.tokens,
            cost_usd=spec.cost_usd,
            error=error,
            llm_retries=retries,
            llm_health=",".join(health) if health else None,
            llm_failures=spec.llm_failures,
            llm_failure_error=spec.llm_failure_error,
        )

    @staticmethod
    def _map_status(status: str, budget_exhausted: bool, error: str | None,
                    no_progress: bool = False,
                    llm_transport: bool = False) -> str:
        """Map an SDK execution status onto a valid AgentStatus.

        Total by construction: AgentSpec validates on assignment, so returning
        anything outside AGENT_STATUSES would raise inside a live run. Unknown
        values become 'failed' rather than being passed through.

        `llm_transport` marks an error raised by the endpoint (connection,
        timeout, rate limit) rather than by the agent. Those read as
        'llm_unavailable' so the board, the alerts, and the recovery path all
        treat an unreachable endpoint as infrastructure, not a broken agent.
        """
        if budget_exhausted:
            return "budget_exhausted"
        if error:
            return "llm_unavailable" if llm_transport else "failed"
        if no_progress:
            # The endpoint answered with nothing; this is an infrastructure
            # problem, not an agent that ran out of ideas.
            return "llm_unavailable"
        mapped = {
            "finished": "idle", "idle": "idle",
            "running": "running",
            "paused": "paused",
            # Only reachable with confirm_tools enabled.
            "waiting_for_confirmation": "paused",
            "error": "failed", "stuck": "failed", "deleting": "failed",
        }.get(status)
        return mapped if mapped in AGENT_STATUSES else "failed"

    @staticmethod
    def _run_limit_code(conv: Conversation, since: int) -> str | None:
        """Run-limit code emitted during *this* run, if any."""
        for event in list(conv.state.events)[since:]:
            code = getattr(event, "code", None)
            if code in ("MaxIterationsReached", "MaxBudgetReached"):
                return str(code)
        return None

    @staticmethod
    def _final_report(conv: Conversation) -> str:
        events = list(conv.state.events)
        for event in reversed(events):
            if isinstance(event, ActionEvent) and event.tool_name == "finish":
                message = getattr(getattr(event, "action", None), "message", None)
                if message:
                    return str(message)
            if isinstance(event, MessageEvent) and event.source == "agent":
                texts = [c.text for c in event.llm_message.content if getattr(c, "text", None)]
                if texts:
                    return "\n".join(texts)
        return ""

    # ---------- lifecycle ----------

    def reset_conversation(self, spec: AgentSpec) -> None:
        """Give an agent a brand-new conversation.

        The SDK keeps the persisted ``SystemPromptEvent`` (system prompt *and*
        tool schema) when resuming, so prompt/tool changes only take effect in a
        fresh conversation. Prior progress is carried over through the agent's
        ``last_report``, which the dispatcher injects into the next assignment.
        """
        self.close(spec.id)
        with self._lock:
            spec.conversation_id = str(uuid.uuid4())
            spec.persistence_dir = None
            spec.turns_used = 0
            spec.runs = 0
            # The closing conversation's spend is exactly (cost_usd - prior),
            # since run() keeps cost_usd at prior + current conversation. So the
            # new baseline is just cost_usd - not prior += cost_usd, which would
            # double-count if a reset ever ran twice.
            spec.prior_cost_usd = spec.cost_usd
            spec.prior_tokens = spec.tokens
            spec.status = "idle"
            spec.touch()
            self.store.upsert_agent(spec)

    def close(self, agent_id: str) -> None:
        with self._llm_lock:
            self._llms.pop(agent_id, None)
        with self._lock:
            conv = self._conversations.pop(agent_id, None)
            if conv is not None:
                try:
                    conv.close()
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to close conversation for %s", agent_id)

    def handle_agent_logs(self, agents: list[AgentSpec], *, keep: bool,
                          archive_to: Path | None = None,
                          prune_project_dirs: bool = False) -> tuple[list[str], str]:
        """Close conversations, then keep / move / delete each agent's log dir.

        Conversation state is keyed by agent id under the project's dir, which
        is also where the orchestrator's own transcript can live. Only the
        listed agents' dirs are touched, so a live orchestrator never loses its
        history. Returns (paths moved or deleted, human-readable note).

        prune_project_dirs additionally clears leftovers in those dirs (state
        from agents that no longer exist), but never a dir belonging to an agent
        still in the store.
        """
        paths: list[str] = []
        for spec in agents:
            target = Path(spec.persistence_dir) if spec.persistence_dir \
                else Path(self._persistence_dir(spec))
            self.close(spec.id)
            if not target.exists():
                continue
            if keep:
                continue
            if archive_to is not None:
                archive_to.mkdir(parents=True, exist_ok=True)
                dest = archive_to / f"{spec.id}-logs"
                shutil.move(str(target), str(dest))
                paths.append(str(dest))
            else:
                shutil.rmtree(target, ignore_errors=True)
                paths.append(str(target))

        if prune_project_dirs:
            paths.extend(self._prune_project_log_dirs(agents))

        if keep:
            note = "Conversation logs kept."
        elif archive_to is not None:
            note = (f"Moved {len(paths)} conversation log dir(s) to {archive_to}."
                    if paths else "No conversation logs to move.")
        else:
            note = (f"Deleted {len(paths)} conversation log dir(s)."
                    if paths else "No conversation logs to delete.")
        return paths, note

    def _prune_project_log_dirs(self, agents: list[AgentSpec]) -> list[str]:
        """Delete leftovers in the affected projects' log dirs.

        A dir is kept if its name is still an agent in the store - notably the
        orchestrator, whose transcript lives under whichever project it last
        touched and must survive that project's removal.
        """
        removed: list[str] = []
        project_dirs = {
            (Path(spec.persistence_dir) if spec.persistence_dir
             else Path(self._persistence_dir(spec))).parent
            for spec in agents
        }
        for parent in project_dirs:
            if not parent.is_dir():
                continue
            for entry in parent.iterdir():
                if entry.name in self.store.agents:
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    with contextlib.suppress(OSError):
                        entry.unlink()
                removed.append(str(entry))
            with contextlib.suppress(OSError):
                parent.rmdir()  # only succeeds once genuinely empty
        return removed

    def delete_agent_artifacts(self, agent_ids: list[str]) -> list[str]:
        """Close conversations and delete persisted state for these agents.

        Used by the orphan sweep, where the agent record may already be gone.
        """
        removed: list[str] = []
        for agent_id in agent_ids:
            spec = self.store.agents.get(agent_id)
            target = None
            if spec is not None:
                target = Path(spec.persistence_dir) if spec.persistence_dir \
                    else Path(self._persistence_dir(spec))
            self.close(agent_id)
            if target is not None and target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed.append(str(target))
        return removed

    def close_all(self) -> None:
        for agent_id in list(self._conversations):
            self.close(agent_id)
