"""Runtime resilience, turn-count sync, blocked-task resume, live config edits."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from agent_harness.config import HarnessConfig, resolve_config_path
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime, ConcurrencyGate, TurnCounter
from agent_harness.store import (
    AGENT_STATUSES,
    KANBAN_COLUMNS,
    AgentSpec,
    Store,
    TaskSpec,
)
from pydantic import ValidationError
from agent_harness.tools import (
    AssignTaskAction,
    AssignTaskExecutor,
    CreateProjectAction,
    CreateProjectExecutor,
    CreateSubAgentAction,
    CreateSubAgentExecutor,
    CreateTaskAction,
    CreateTaskExecutor,
    UpdateTaskAction,
    UpdateTaskExecutor,
    register_harness_tools,
)
from fastapi.testclient import TestClient

from agent_harness.web.api import _apply_config_patch, build_api


class FakeConv:
    def __init__(self, cid="conv-1"):
        self.id = cid


def make(tmp: str) -> tuple[HarnessConfig, Store, AgentRuntime, Orchestrator]:
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    cfg.projects_dir = str(Path(tmp) / "projects")
    cfg.tick_seconds = 60
    register_harness_tools()
    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    return cfg, store, runtime, Orchestrator(cfg, store, runtime)


def test_concurrency_gate_serialises_below_limit():
    import threading
    import time

    gate = ConcurrencyGate(2)
    peak = 0
    lock = threading.Lock()

    def worker():
        nonlocal peak
        with gate.slot():
            with lock:
                peak = max(peak, gate.active)
            time.sleep(0.05)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak <= 2, peak
    assert gate.active == 0

    gate.set_limit(6)
    assert gate.limit == 6
    print("concurrency gate ok: peak", peak)


def test_turn_counter_counts_unique_response_ids():
    # Real SDK events, not fakes: the counter keys off llm_response_id and the
    # ActionEvent/MessageEvent types, so fakes would not test the real logic.
    from openhands.sdk.event.llm_convertible.action import ActionEvent
    from openhands.sdk.event.llm_convertible.message import MessageEvent
    from openhands.sdk.llm import Message, MessageToolCall, TextContent

    def msg_event(rid=None):
        return MessageEvent(
            source="agent",
            llm_message=Message(role="assistant",
                                content=[TextContent(text="hi")]),
            llm_response_id=rid,
        )

    def action_event(rid):
        return ActionEvent(
            thought=[TextContent(text="thinking")],
            tool_name="finish", tool_call_id="call-1",
            tool_call=MessageToolCall(id="call-1", name="finish",
                                      arguments='{"message": "done"}',
                                      origin="completion"),
            action=None, llm_response_id=rid,
        )

    class State:
        events = [
            action_event("r1"),
            action_event("r1"),      # same LLM response -> one turn
            action_event("r2"),
            msg_event("r3"),         # has an id -> counted by id
            msg_event(None),         # standalone message -> counted alone
            MessageEvent(source="user",
                         llm_message=Message(role="user",
                                             content=[TextContent(text="u")])),
            ActionEvent(thought=[TextContent(text="thinking")],
                        tool_name="finish", tool_call_id="call-9",
                        tool_call=MessageToolCall(
                            id="call-9", name="finish",
                            arguments='{"message": "done"}',
                            origin="completion"),
                        action=None, llm_response_id="r-user-src",
                        source="user"),
        ]

    class Conv:
        state = State()

    # r1, r2, r3 = 3 id-keyed turns + 1 standalone message
    assert TurnCounter.count(Conv()) == 4, TurnCounter.count(Conv())
    print("turn counter ok")


def test_task_turns_track_agent_turns():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")

        CreateProjectExecutor()(CreateProjectAction(
            name="Sync", goal="turn counts must match", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Build", objective="x",
            expertise="backend", acceptance_criteria=["works"],
            deliverables=["src/app.py"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]

        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build it", project_id=pid), conv)
        agent_id = [a.id for a in store.project_agents(pid) if a.depth == 1][0]
        AssignTaskExecutor()(AssignTaskAction(task_id=task_id, agent_id=agent_id), conv)

        # Simulate a run that consumed 37 turns.
        spec = store.agents[agent_id]
        spec.turns_used = 37
        store.upsert_agent(spec)
        assert store.tasks[task_id].turns_used == 37, store.tasks[task_id].turns_used

        # A second agent on the same task must not regress the count.
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-reviewer", specialty="backend",
            mission="review it", project_id=pid), conv)
        other = [a for a in store.project_agents(pid)
                 if a.role == "backend-reviewer"][0]
        other.turns_used = 5
        store.upsert_agent(other)
        assert store.tasks[task_id].turns_used == 37, "count regressed"

        # Unassigning stops the sync but keeps the history: an unstaffed card
        # still shows the turns the previous worker burned.
        task = store.tasks[task_id]
        task.assigned_agent_id = None
        store.upsert_task(task)
        spec.turns_used = 99
        store.upsert_agent(spec)
        assert store.tasks[task_id].turns_used == 37, "sync should be inert"

        # Re-staffing a fresh agent re-links the counters to the new worker.
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev-2", specialty="backend",
            mission="finish it", project_id=pid), conv)
        fresh_id = [a.id for a in store.project_agents(pid)
                    if a.role == "backend-dev-2"][0]
        AssignTaskExecutor()(AssignTaskAction(task_id=task_id, agent_id=fresh_id), conv)
        fresh = store.agents[fresh_id]
        fresh.turns_used = 4
        store.upsert_agent(fresh)
        assert store.tasks[task_id].turns_used == 4, store.tasks[task_id].turns_used
        print("task/agent turn sync ok")


def test_blocked_task_requires_reason_and_is_resumed():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")

        CreateProjectExecutor()(CreateProjectAction(
            name="Blocked", goal="resume blocked work", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Flaky", objective="x", expertise="qa",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]

        # Manual block without a reason is rejected: the cycle needs to know
        # what to act on.
        obs = UpdateTaskExecutor()(UpdateTaskAction(
            task_id=task_id, column="blocked"), conv)
        assert obs.is_error and "blocked_reason" in obs.text, obs.text

        obs = UpdateTaskExecutor()(UpdateTaskAction(
            task_id=task_id, column="blocked",
            blocked_reason="missing credentials"), conv)
        assert not obs.is_error, obs.text
        assert store.tasks[task_id].blocked_reason == "missing credentials"

        attention = orch.detect()[pid]
        assert any("missing credentials" in m for m in attention), attention
        assert any("BLOCKED" in m for m in attention)

        # A user directive must survive a restart and reach the next briefing.
        store.tasks[task_id].user_directives.append(
            {"text": "use env vars", "at": "now"})
        store.tasks[task_id].column = "ready"
        store.flush()
        fresh = Store(cfg.data_dir)
        assert fresh.tasks[task_id].user_directives[0]["text"] == "use env vars"
        print("blocked task detect/resume ok")


def test_llm_unavailable_agent_surfaces():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="SlowLLM", goal="handle empty responses", constraints=""), conv)
        pid = list(store.projects)[0]

        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Stalled", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        agent_id = [a.id for a in store.project_agents(pid) if a.depth == 1][0]

        agent = store.agents[agent_id]
        agent.status = "llm_unavailable"
        store.upsert_agent(agent)
        runtime.llm_empty_responses = 3
        runtime.llm_timeout = 600

        attention = orch.detect()[pid]
        assert any("returned nothing" in m for m in attention), attention
        assert any("600s" in m for m in attention), attention
        print("llm_unavailable surfaced ok")


def test_live_config_patch_and_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)

        changed = _apply_config_patch(cfg, orch, {
            "llm.base_url": "http://other-host:1234/v1",
            "llm.model": "openai/other-model",
            "llm.request_timeout": 900,
            "max_parallel_workers": 5,
            "sub_agent_turn_limit": 250,
            "llm.input_cost_per_token": 0.000001,
        })
        keys = dict(changed)
        assert cfg.llm.base_url == "http://other-host:1234/v1"
        assert cfg.llm.model == "openai/other-model"
        assert runtime.llm_timeout == 900, runtime.llm_timeout
        assert runtime.gate.limit == 5
        assert cfg.sub_agent_turn_limit == 250
        assert cfg.llm.input_cost_per_token == 0.000001
        assert "llm.base_url" in keys

        # A new agent picks up the new endpoint and the live timeout.
        llm = cfg.llm.build_llm(usage_id="x", timeout=runtime.llm_timeout)
        assert llm.base_url == "http://other-host:1234/v1"
        assert llm.timeout == 900
        assert llm.input_cost_per_token == 0.000001

        try:
            _apply_config_patch(cfg, orch, {"llm.request_timeout": "not-a-number"})
            raise AssertionError("bad type should raise")
        except Exception as exc:
            assert "must be int" in str(exc), exc

        try:
            _apply_config_patch(cfg, orch, {"not_a_setting": 1})
            raise AssertionError("unknown key should raise")
        except Exception as exc:
            assert "Unknown setting" in str(exc), exc
        print("live config patch ok")


def test_adaptive_timeout_growth_is_bounded():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        cfg.llm.request_timeout = 100
        cfg.llm.max_request_timeout = 500
        runtime.llm_timeout = 100

        assert runtime.widen_llm_timeout("t") == 200
        assert runtime.widen_llm_timeout("t") == 400
        assert runtime.widen_llm_timeout("t") == 500  # clamped
        assert runtime.widen_llm_timeout("t") == 500  # stays at the cap

        # Survives a restart: the store keeps the widened value.
        store.flush()
        _, _, runtime2, _ = make(tmp)
        assert runtime2.llm_timeout == 500, runtime2.llm_timeout

        runtime2.config.llm.adaptive_timeout = False
        runtime2.llm_timeout = 100
        assert runtime2.widen_llm_timeout("t") == 100
        print("adaptive timeout ok")


class FlakyConv:
    """Conversation whose runs fail or produce nothing, to exercise retry."""

    def __init__(self, outcomes):
        self.id = "flaky-conv"
        self._outcomes = list(outcomes)
        self.messages = []
        self.max_iteration_per_run = 0
        self._events = []
        self._state = None

    def send_message(self, text):
        self.messages.append(text)

    def run(self):
        outcome = self._outcomes.pop(0) if self._outcomes else None
        if isinstance(outcome, Exception):
            raise outcome
        for _ in range(outcome or 0):
            self._events.append(ACTION_EVENT("resp-%d" % len(self._events)))

    @property
    def state(self):
        if self._state is None:
            self._state = _FakeState(self._events)
        return self._state


class _FakeMetrics:
    accumulated_token_usage = None
    accumulated_cost = 0.0


class _FakeState:
    def __init__(self, events):
        self._events = events
        self.execution_status = "idle"

        class _Stats:
            @staticmethod
            def get_combined_metrics():
                return _FakeMetrics()

        self.stats = _Stats()

    @property
    def events(self):
        return list(self._events)


def _action_event(rid):
    from openhands.sdk.event.llm_convertible.action import ActionEvent
    from openhands.sdk.llm import MessageToolCall, TextContent
    return ActionEvent(
        source="agent", thought=[TextContent(text="t")],
        tool_name="finish", tool_call_id="c",
        tool_call=MessageToolCall(id="c", name="finish",
                                  arguments="{}", origin="completion"),
        action=None, llm_response_id=rid,
    )


ACTION_EVENT = _action_event


def test_run_retries_empty_and_transient_failures():
    import litellm

    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Retry", goal="retry flaky endpoint", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Work", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        agent_id = [a.id for a in store.project_agents(pid) if a.depth == 1][0]
        spec = store.agents[agent_id]

        # 1st attempt times out, 2nd returns nothing, 3rd makes progress.
        flaky = FlakyConv([
            litellm.exceptions.Timeout(
                message="timed out", model="qwen", llm_provider="openai"),
            0,
            2,
        ])
        runtime.open_conversation = lambda *a, **k: flaky

        outcome = runtime.run(spec, [], "do the work")

        assert outcome.turns_this_run == 2, outcome
        assert outcome.llm_retries == 2, outcome
        assert outcome.status == "idle", outcome
        assert len(flaky.messages) == 3, flaky.messages
        # Only the first attempt carries the assignment; retries just nudge.
        assert flaky.messages[0] == "do the work"
        assert "timed out" in flaky.messages[1].lower()
        # The timeout grew because the endpoint timed out.
        assert runtime.llm_timeout > cfg.llm.request_timeout, runtime.llm_timeout
        assert runtime.llm_transient_errors >= 1
        assert runtime.llm_empty_responses >= 1
        assert spec.turns_used == 2
        print("retry ok: timeout widened to %ss, %d retries"
              % (runtime.llm_timeout, outcome.llm_retries))


def test_run_reports_llm_unavailable_when_endpoint_stays_down():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        cfg.llm.empty_response_retries = 1
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Down", goal="endpoint stays down", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Work", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        agent_id = [a.id for a in store.project_agents(pid) if a.depth == 1][0]
        spec = store.agents[agent_id]

        flaky = FlakyConv([0, 0])
        runtime.open_conversation = lambda *a, **k: flaky
        outcome = runtime.run(spec, [], "do the work")

        assert outcome.turns_this_run == 0
        assert outcome.status == "llm_unavailable", outcome.status
        assert outcome.error is None, "empty output is not an agent error"
        assert "llm_unavailable" in orch.detect()[pid][0] or any(
            "returned nothing" in m for m in orch.detect()[pid]), orch.detect()[pid]
        print("llm_unavailable reported ok")


def test_llm_unavailable_survives_a_flush_and_reload():
    """Regression: an invalid status used to crash startup, not the write.

    Pydantic does not validate on assignment by default, so a bad status was
    accepted in memory, flushed to state.json, and only blew up as a
    ValidationError inside Store._load() on the *next* start - which under
    `restart: unless-stopped` meant a crash loop with no UI to recover from.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Persist", goal="status must round-trip", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        agent_id = [a.id for a in store.project_agents(pid) if a.depth == 1][0]

        store.agents[agent_id].status = "llm_unavailable"
        store.flush()

        on_disk = json.loads((Path(tmp) / "state.json").read_text())
        saved = {a["id"]: a["status"] for a in on_disk["agents"]}
        assert saved[agent_id] == "llm_unavailable", saved

        reloaded = Store(cfg.data_dir)
        assert reloaded.agents[agent_id].status == "llm_unavailable"
        assert reloaded.agents[agent_id].turns_used == 0
        print("llm_unavailable round-trips through reload ok")


def test_bad_agent_status_is_repaired_on_load_not_fatal():
    """A record the model rejects must not brick startup.

    Simulates state written by an older/newer build: the status is something
    this build does not know, so it is coerced to 'failed' (which already
    surfaces as an alert) instead of dropping the agent or raising.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Legacy", goal="unknown status in state", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        agent_id = [a.id for a in store.project_agents(pid) if a.depth == 1][0]
        store.agents[agent_id].turns_used = 41
        store.flush()

        state_path = Path(tmp) / "state.json"
        data = json.loads(state_path.read_text())
        for a in data["agents"]:
            if a["id"] == agent_id:
                a["status"] = "quantum_leaking"
        state_path.write_text(json.dumps(data))

        reloaded = Store(cfg.data_dir)
        kept = reloaded.agents[agent_id]
        assert kept.status == "failed", kept.status
        assert kept.turns_used == 41, "repair must not lose the turn count"
        assert "quantum_leaking" in (kept.last_error or ""), kept.last_error
        # Still visible on the board and still alerting.
        assert reloaded.project_agents(pid), "repaired agent vanished"
        alerts = TestClient(build_api(cfg, reloaded, Orchestrator(
            cfg, reloaded, AgentRuntime(cfg, reloaded)))).get("/api/state").json()["alerts"]
        assert any("backend-dev is failed" in a["message"]
                   and "41/100" in a["message"] for a in alerts), alerts
        print("unknown agent status repaired on load ok")


def test_unknown_project_status_and_task_column_are_repaired_on_load():
    """Same guard for the other two record kinds, without losing their state."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Repair", goal="bad enums in state", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Card", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]
        store.projects[pid].version_minor = 7
        store.tasks[task_id].turns_used = 12
        store.flush()

        state_path = Path(tmp) / "state.json"
        data = json.loads(state_path.read_text())
        data["projects"][0]["status"] = "melting"
        data["tasks"][0]["column"] = "sideways"
        state_path.write_text(json.dumps(data))

        reloaded = Store(cfg.data_dir)
        assert reloaded.projects[pid].status == "active", reloaded.projects[pid].status
        assert reloaded.projects[pid].version == "0.7", reloaded.projects[pid].version
        task = reloaded.tasks[task_id]
        assert task.column == "blocked", task.column
        assert task.turns_used == 12, task.turns_used
        assert "sideways" in (task.blocked_reason or ""), task.blocked_reason
        # A repaired task in blocked must carry a reason or the UI shows a
        # reasonless blocker; the orchestrator resumes it on the next cycle.
        assert task.column in KANBAN_COLUMNS
        print("unknown project status / task column repaired ok")


def test_assignment_of_invalid_status_raises_at_the_write():
    """validate_assignment turns a silent disk-poisoning write into a loud one."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        spec = AgentSpec(role="w", specialty="x", mission="m", system_prompt="p")
        store.agents[spec.id] = spec
        try:
            spec.status = "not_a_real_status"
        except ValidationError:
            pass
        else:
            raise AssertionError("invalid status assignment was accepted")
        # Same guard on the other persisted models.
        proj = store.create_project("G", "goal", str(Path(tmp) / "ws"))
        try:
            proj.status = "on_fire"
        except ValidationError:
            pass
        else:
            raise AssertionError("invalid project status assignment was accepted")
        task = TaskSpec(project_id=proj.id, title="t", objective="o",
                        expertise="e", acceptance_criteria=["a"],
                        deliverables=["d"])
        store.tasks[task.id] = task
        try:
            task.column = "shredded"
        except ValidationError:
            pass
        else:
            raise AssertionError("invalid task column assignment was accepted")
        print("invalid assignments rejected at the write ok")


def test_map_status_covers_every_sdk_execution_status():
    """_map_status must be total: it feeds a field that now validates.

    An unmapped SDK status would otherwise raise inside a live run.
    """
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    for status in ConversationExecutionStatus:
        mapped = AgentRuntime._map_status(status.value, False, None)
        assert mapped in AGENT_STATUSES, (status.value, mapped)
    # Unknown/future values degrade to 'failed' rather than passing through.
    assert AgentRuntime._map_status("deleting", False, None) == "failed"
    assert AgentRuntime._map_status("brand_new", False, None) == "failed"
    assert AgentRuntime._map_status("brand_new", True, None) == "budget_exhausted"
    assert AgentRuntime._map_status("brand_new", False, "boom") == "failed"
    assert AgentRuntime._map_status("brand_new", False, None, True) == "llm_unavailable"
    print("map_status total over", len(list(ConversationExecutionStatus)),
          "sdk statuses ok")


def test_auth_errors_are_not_retried():
    import litellm

    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Auth", goal="bad key must not loop", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Work", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        agent_id = [a.id for a in store.project_agents(pid) if a.depth == 1][0]
        spec = store.agents[agent_id]

        flaky = FlakyConv([litellm.exceptions.AuthenticationError(
            message="invalid api key", model="qwen", llm_provider="openai")])
        runtime.open_conversation = lambda *a, **k: flaky
        outcome = runtime.run(spec, [], "do the work")

        assert outcome.llm_retries == 0, "auth failure must not be retried"
        assert len(flaky.messages) == 1, flaky.messages
        assert outcome.status == "failed", outcome.status
        assert "AuthenticationError" in (outcome.error or "")
        print("auth error not retried ok")


def test_reviewer_does_not_steal_the_worker_link():
    """The card's turn counters follow the worker, not the reviewer."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Links", goal="reviewer must not own the card", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Build", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]

        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        worker_id = [a.id for a in store.project_agents(pid)
                     if a.role == "backend-dev"][0]
        AssignTaskExecutor()(AssignTaskAction(task_id=task_id, agent_id=worker_id), conv)

        # A reviewer run must land on reviewer_agent_id, leaving the worker link.
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-reviewer", specialty="backend",
            mission="review", project_id=pid), conv)
        reviewer_id = [a.id for a in store.project_agents(pid)
                       if a.role == "backend-reviewer"][0]

        worker, reviewer = store.agents[worker_id], store.agents[reviewer_id]
        worker.turn_limit, worker.turns_used = 30, 12
        reviewer.turn_limit, reviewer.turns_used = 20, 7
        store.upsert_agent(worker)

        from agent_harness.tools import _dispatch
        runtime.open_conversation = lambda *a, **k: FlakyConv([7])
        _dispatch(reviewer, store.tasks[task_id], "review it", 20, review=True)

        task = store.tasks[task_id]
        assert task.assigned_agent_id == worker_id, "worker link was clobbered"
        assert task.reviewer_agent_id == reviewer_id, "reviewer not recorded"
        assert task.turns_used == 12, (
            f"card must show the worker's 12 turns, got {task.turns_used}")
        assert task.turn_limit == 30, task.turn_limit
        print("reviewer/worker link separation ok")


def test_dispatch_without_assign_task_still_links_counters():
    """run_agent(task_id=...) must link the agent even if assign_task was skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="NoAssign", goal="link created at dispatch", constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Build", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend",
            mission="build", project_id=pid), conv)
        worker_id = [a.id for a in store.project_agents(pid)
                     if a.role == "backend-dev"][0]
        assert store.agents[worker_id].task_id is None, "should start unlinked"

        from agent_harness.tools import RunAgentExecutor, RunAgentAction
        runtime.open_conversation = lambda *a, **k: FlakyConv([9])
        obs = RunAgentExecutor()(RunAgentAction(
            agent_id=worker_id, task_id=task_id, background=False), conv)
        assert not obs.is_error, obs.text

        task, worker = store.tasks[task_id], store.agents[worker_id]
        assert worker.task_id == task_id, "reverse link not created"
        assert task.assigned_agent_id == worker_id
        assert task.turns_used == 9 == worker.turns_used, (task.turns_used, worker.turns_used)
        assert task.turn_limit == worker.turn_limit

        # A reviewer dispatched the same way must not take the card over.
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="reviewer", specialty="backend", mission="review",
            project_id=pid), conv)
        reviewer_id = [a.id for a in store.project_agents(pid)
                       if a.role == "reviewer"][0]
        runtime.open_conversation = lambda *a, **k: FlakyConv([13])
        obs = RunAgentExecutor()(RunAgentAction(
            agent_id=reviewer_id, task_id=task_id, review=True,
            background=False), conv)
        assert not obs.is_error, obs.text

        task = store.tasks[task_id]
        assert store.agents[reviewer_id].is_reviewer, "reviewer flag not set"
        assert task.assigned_agent_id == worker_id, "worker link stolen"
        assert task.turns_used == 9, (
            f"card must keep the worker's 9 turns, got {task.turns_used}")
        assert task.turn_limit == 100
        print("dispatch-time linking + reviewer isolation ok")


def test_reconfigure_and_assign_keep_card_counters_on_the_worker():
    """Reconfiguring/assigning a reviewer must not reset or steal the card."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        conv = FakeConv("orch-conv")
        CreateProjectExecutor()(CreateProjectAction(
            name="Reconf", goal="reviewer must not reset the card",
            constraints=""), conv)
        pid = list(store.projects)[0]
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Build", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["out"]), conv)
        task_id = [t.id for t in store.project_tasks(pid)][0]

        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="backend-dev", specialty="backend", mission="build",
            project_id=pid), conv)
        worker_id = [a.id for a in store.project_agents(pid)
                     if a.role == "backend-dev"][0]
        AssignTaskExecutor()(AssignTaskAction(task_id=task_id, agent_id=worker_id), conv)

        from agent_harness.tools import (RunAgentExecutor, RunAgentAction,
                                         ReconfigureAgentAction, ReconfigureAgentExecutor)
        runtime.open_conversation = lambda *a, **k: FlakyConv([6])
        RunAgentExecutor()(RunAgentAction(
            agent_id=worker_id, task_id=task_id, background=False), conv)
        assert store.tasks[task_id].turns_used == 6

        # Reviewer is assigned first (as the orchestrator sometimes does), then
        # dispatched with review=true.
        CreateSubAgentExecutor()(CreateSubAgentAction(
            role="reviewer", specialty="backend", mission="review",
            project_id=pid), conv)
        reviewer_id = [a.id for a in store.project_agents(pid)
                       if a.role == "reviewer"][0]
        AssignTaskExecutor()(AssignTaskAction(task_id=task_id, agent_id=reviewer_id), conv)
        assert store.tasks[task_id].assigned_agent_id == worker_id, (
            "assign_task handed the worker link to a reviewer")

        runtime.open_conversation = lambda *a, **k: FlakyConv([4])
        ReconfigureAgentExecutor()(ReconfigureAgentAction(
            agent_id=reviewer_id, decision="reviewer prompt was too loose",
            mission="review harder", turn_limit=12,
            fresh_conversation=False), conv)
        task = store.tasks[task_id]
        assert task.turns_used == 6, (
            f"reconfiguring the reviewer reset the card to {task.turns_used}")
        assert task.turn_limit == 100, task.turn_limit

        _dispatch_review(store, runtime, reviewer_id, task_id)
        task = store.tasks[task_id]
        assert task.assigned_agent_id == worker_id, "worker link lost after review"
        assert task.reviewer_agent_id == reviewer_id
        assert task.turns_used == 6, task.turns_used
        print("reconfigure/assign reviewer isolation ok")


def _dispatch_review(store, runtime, reviewer_id, task_id):
    conv = FakeConv("orch-conv")
    from agent_harness.tools import RunAgentExecutor, RunAgentAction
    runtime.open_conversation = lambda *a, **k: FlakyConv([4])
    RunAgentExecutor()(RunAgentAction(
        agent_id=reviewer_id, task_id=task_id, review=True, background=False), conv)


def test_config_edits_persist_to_the_loaded_file():
    """UI config edits must write back to the file the process loaded from."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_file = Path(tmp) / "config.json"
        cfg_file.write_text(json.dumps({
            "llm": {"base_url": "http://x/v1/", "model": "openai/m",
                    "api_key": "sekret", "request_timeout": 300},
            "data_dir": tmp, "projects_dir": str(Path(tmp) / "projects"),
            "tick_seconds": 60,
        }))
        assert resolve_config_path(cfg_file) == cfg_file
        cfg = HarnessConfig.load(cfg_file)
        assert cfg.llm.request_timeout == 300

        store = Store(tmp)
        runtime = AgentRuntime(cfg, store)
        orch = Orchestrator(cfg, store, runtime)
        app = build_api(cfg, store, orch, config_path=str(cfg_file))
        client = TestClient(app)

        r = client.post("/api/config", json={"llm.request_timeout": 777})
        assert r.status_code == 200, r.text
        on_disk = json.loads(cfg_file.read_text())
        assert on_disk["llm"]["request_timeout"] == 777, on_disk["llm"]
        # The key must never be echoed to the browser.
        got = client.get("/api/config").json()["config"]
        assert got["llm"]["api_key"] != "sekret"
        assert "sekret" not in r.text
        print("config persistence + redaction ok")


def test_env_overrides_are_not_baked_into_the_config_file():
    """Patching one setting must not rewrite values that came from the environment.

    A harness started with HARNESS_DATA_DIR/HARNESS_PORT must not persist those
    overrides into config.json, or the next plain start inherits them.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg_file = Path(tmp) / "config.json"
        cfg_file.write_text(json.dumps({
            "llm": {"base_url": "http://x/v1/", "model": "openai/m"},
            "data_dir": str(Path(tmp) / "disk-data"),
            "projects_dir": str(Path(tmp) / "disk-data" / "projects"),
            "tick_seconds": 60,
            "port": 8090,
        }))
        cfg = HarnessConfig.load(cfg_file)
        # Simulate an environment override on top of the file.
        cfg.data_dir = str(Path(tmp) / "env-data")
        cfg.port = 8099
        cfg.host = "0.0.0.0"

        store = Store(cfg.data_dir)
        runtime = AgentRuntime(cfg, store)
        orch = Orchestrator(cfg, store, runtime)
        app = build_api(cfg, store, orch, config_path=str(cfg_file))
        client = TestClient(app)

        r = client.post("/api/config", json={"tick_seconds": 90})
        assert r.status_code == 200, r.text

        on_disk = json.loads(cfg_file.read_text())
        assert on_disk["tick_seconds"] == 90, on_disk
        assert on_disk["data_dir"] == str(Path(tmp) / "disk-data"), on_disk
        assert on_disk["port"] == 8090, on_disk
        assert on_disk["llm"]["base_url"] == "http://x/v1/", on_disk
        assert "env-data" not in cfg_file.read_text(), cfg_file.read_text()
        print("env overrides kept out of config file ok")


def test_qwen_cost_defaults():
    cfg = HarnessConfig()
    assert cfg.llm.input_cost_per_token == 0.000002
    assert cfg.llm.output_cost_per_token == 0.000015
    llm = cfg.llm.build_llm(usage_id="qwen")
    assert llm.input_cost_per_token == 0.000002
    assert llm.output_cost_per_token == 0.000015
    print("qwen token costs ok")


def test_inflight_refcount_survives_nested_submit_and_run():
    """submit() marks an agent, and the run() it calls marks it again.

    A plain set would let the inner unmark clear the agent while the outer
    dispatch is still going, flickering the dashboard's active light and
    letting remove-project slip past the running-agent guard.
    """
    import threading
    import time

    from agent_harness.runtime import RunOutcome
    from agent_harness.store import AgentSpec

    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        agent = AgentSpec(role="w", specialty="x", mission="m",
                          system_prompt="p", project_id="p1")
        store.upsert_agent(agent)

        entered = threading.Event()
        release = threading.Event()
        seen: list[int] = []

        def fake_guarded(spec, tools, message, turn_limit, **_kw):
            # submit() already marked it, so run()'s mark makes this depth 2.
            seen.append(runtime._inflight.get(agent.id, 0))
            entered.set()
            release.wait(10)
            return RunOutcome(turns_used=0, turns_this_run=0, status="idle",
                              final_report="", budget_exhausted=False,
                              tokens=0, cost_usd=0.0)

        runtime._run_guarded = fake_guarded
        runtime.submit(agent.id, lambda: runtime.run(agent, [], "go"))
        try:
            assert entered.wait(5), "run never started"
            assert agent.id in runtime.inflight, "lost the mark mid-run"
        finally:
            release.set()

        deadline = time.time() + 5
        while agent.id in runtime.inflight and time.time() < deadline:
            time.sleep(0.05)
        assert agent.id not in runtime.inflight, "refcount leaked after both exited"

        hold = threading.Event()
        runtime.submit(agent.id, lambda: hold.wait(10))
        try:
            try:
                runtime.submit(agent.id, lambda: None)
                raise AssertionError("double submit must be refused")
            except RuntimeError:
                pass
        finally:
            hold.set()
        assert seen == [2], seen
        print("inflight refcount nesting ok")


if __name__ == "__main__":
    test_concurrency_gate_serialises_below_limit()
    test_turn_counter_counts_unique_response_ids()
    test_task_turns_track_agent_turns()
    test_blocked_task_requires_reason_and_is_resumed()
    test_llm_unavailable_agent_surfaces()
    test_live_config_patch_and_persistence()
    test_adaptive_timeout_growth_is_bounded()
    test_run_retries_empty_and_transient_failures()
    test_run_reports_llm_unavailable_when_endpoint_stays_down()
    test_llm_unavailable_survives_a_flush_and_reload()
    test_bad_agent_status_is_repaired_on_load_not_fatal()
    test_unknown_project_status_and_task_column_are_repaired_on_load()
    test_assignment_of_invalid_status_raises_at_the_write()
    test_map_status_covers_every_sdk_execution_status()
    test_auth_errors_are_not_retried()
    test_reviewer_does_not_steal_the_worker_link()
    test_dispatch_without_assign_task_still_links_counters()
    test_reconfigure_and_assign_keep_card_counters_on_the_worker()
    test_config_edits_persist_to_the_loaded_file()
    test_env_overrides_are_not_baked_into_the_config_file()
    test_qwen_cost_defaults()
    test_inflight_refcount_survives_nested_submit_and_run()
    print("\nALL RUNTIME/CONFIG TESTS PASSED")
