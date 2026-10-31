"""Runs agent conversations under a hard turn budget.

A budget is enforced by capping ``max_iteration_per_run`` for a single run and
counting cumulative agent turns across runs. Because conversations persist, a
sub-agent that runs out of budget can be resumed later with a new budget, which
is exactly what the orchestrator decides on when it is told the budget ran out.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from openhands.sdk import Agent, Conversation, LLMSummarizingCondenser
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event.llm_convertible import ActionEvent, MessageEvent
from openhands.sdk.logger import get_logger
from openhands.sdk.security.confirmation_policy import AlwaysConfirm, NeverConfirm

from .config import HarnessConfig
from .store import AgentSpec, Store

logger = get_logger(__name__)


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
        response_ids: set[str] = set()
        message_events = 0
        for event in conversation.state.events:
            if event.source != "agent":
                continue
            if isinstance(event, ActionEvent):
                response_ids.add(str(event.llm_response_id))
            elif isinstance(event, MessageEvent):
                if event.llm_response_id:
                    response_ids.add(str(event.llm_response_id))
                else:
                    message_events += 1
        return len(response_ids) + message_events


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


class AgentRuntime:
    """Creates, resumes and runs agent conversations with budget enforcement."""

    def __init__(self, config: HarnessConfig, store: Store):
        self.config = config
        self.store = store
        self._conversations: dict[str, Conversation] = {}
        # conversation_id -> agent_id, populated as conversations open.
        self.conversation_index: dict[str, str] = {}
        # Guards the _conversations / conversation_index maps only. Runs are
        # serialised per agent, not globally, so background dispatch is parallel.
        self._lock = threading.RLock()
        self._agent_locks: dict[str, threading.Lock] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, config.max_parallel_workers),
            thread_name_prefix="worker",
        )
        self._inflight: set[str] = set()

    @property
    def inflight(self) -> set[str]:
        with self._lock:
            return set(self._inflight)

    def submit(self, agent_id: str, fn) -> None:
        """Run a worker stretch off the loop thread."""
        with self._lock:
            if agent_id in self._inflight:
                raise RuntimeError(f"Agent {agent_id} is already running")
            self._inflight.add(agent_id)

        def guarded() -> None:
            try:
                fn()
            finally:
                with self._lock:
                    self._inflight.discard(agent_id)

        self._pool.submit(guarded)

    # ---------- construction ----------

    def _persistence_dir(self, agent: AgentSpec) -> str:
        if agent.persistence_dir:
            return agent.persistence_dir
        scope = agent.project_id or "global"
        return str(self.config.root_dir / "conversations" / scope / agent.id)

    def build_agent(self, spec: AgentSpec, tools: list) -> Agent:
        """Build the SDK Agent for a stored spec."""
        llm = self.config.llm.build_llm(usage_id=spec.id)
        condenser = LLMSummarizingCondenser(
            llm=self.config.llm.build_llm(usage_id=f"{spec.id}-condenser"),
            max_size=120,
            keep_first=2,
        )
        return Agent(
            llm=llm,
            tools=list(tools),
            system_prompt=spec.system_prompt,
            security_policy_filename="",
            condenser=condenser,
            include_default_tools=["FinishTool", "ThinkTool"],
        )

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
                visualizer=None,
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

    # ---------- execution ----------

    def run(self, spec: AgentSpec, tools: list, message: str,
            turn_limit: int | None = None) -> RunOutcome:
        """Run one budgeted stretch of an agent's conversation."""
        with self._agent_lock(spec.id):
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

            error: str | None = None
            try:
                conv.send_message(message)
                conv.run()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                error = f"{type(exc).__name__}: {exc}"
                logger.exception("Agent %s run failed", spec.id)

            # max(): a condenser that prunes events can make the count regress,
            # which would silently hand the agent turns back. Never regress.
            turns_now = max(before_turns, TurnCounter.count(conv))
            turns_this_run = turns_now - before_turns
            spec.turns_used = turns_now

            metrics = conv.state.stats.get_combined_metrics()
            usage = metrics.accumulated_token_usage
            spec.tokens = (
                usage.prompt_tokens + usage.completion_tokens
                + usage.cache_read_tokens + usage.cache_write_tokens
            )
            spec.cost_usd = float(metrics.accumulated_cost or 0.0)

            status = str(conv.state.execution_status.value
                         if hasattr(conv.state.execution_status, "value")
                         else conv.state.execution_status)
            limit_code = self._run_limit_code(conv, events_before)
            budget_exhausted = limit_code in ("MaxIterationsReached", "MaxBudgetReached")

            final_report = self._final_report(conv)
            spec.last_report = final_report or spec.last_report
            spec.last_error = error
            spec.status = self._map_status(status, budget_exhausted, error)
            spec.touch()
            self.store.upsert_agent(spec)

            self.store.add_event(
                "agent_run_finished",
                f"{spec.role} finished run {spec.runs}: status={spec.status}, "
                f"turns={spec.turns_used}/{spec.turn_limit}.",
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
            )

    @staticmethod
    def _map_status(status: str, budget_exhausted: bool, error: str | None) -> str:
        if budget_exhausted:
            return "budget_exhausted"
        if error:
            return "failed"
        if status in ("finished", "idle"):
            return "idle"
        if status in ("error", "stuck"):
            return "failed"
        return status

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
            spec.tokens = 0
            spec.cost_usd = 0.0
            spec.status = "idle"
            spec.touch()
            self.store.upsert_agent(spec)

    def close(self, agent_id: str) -> None:
        with self._lock:
            conv = self._conversations.pop(agent_id, None)
            if conv is not None:
                try:
                    conv.close()
                except Exception:  # noqa: BLE001
                    logger.debug("Failed to close conversation for %s", agent_id)

    def close_all(self) -> None:
        for agent_id in list(self._conversations):
            self.close(agent_id)
