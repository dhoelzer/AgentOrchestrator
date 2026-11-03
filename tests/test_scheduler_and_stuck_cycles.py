"""The deterministic scheduler, and the stuck-cycle signature it fixes.

Symptom this covers: one orchestrator prompt sits 'executing' for 900s, another
sits 'AWAITING', the LLM endpoints look idle, and no other task moves. Two things
cause it together - an upstream call that can never be reclaimed (covered in
test_llm_router.py) and the fact that *starting* an already-staffed card requires
another orchestrator turn, which has to queue for capacity itself.

The split that fixes the second half: the orchestrator (LLM) decides which agent
owns a card; the scheduler (no LLM) starts it. A card the orchestrator has
already assigned must not wait on a model call before anything happens.

Run: python3 tests/test_scheduler_and_stuck_cycles.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

from agent_harness.config import HarnessConfig, LLMConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime, RunOutcome
from agent_harness.store import Store
from tests.test_runtime_and_config import FlakyConv
from agent_harness.tools import (
    AssignTaskAction,
    AssignTaskExecutor,
    CreateProjectAction,
    CreateProjectExecutor,
    CreateSubAgentAction,
    CreateSubAgentExecutor,
    CreateTaskAction,
    CreateTaskExecutor,
    register_harness_tools,
)


class FakeConv:
    def __init__(self, cid="conv-1"):
        self.id = cid


def seed(tmp: str):
    register_harness_tools()
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    cfg.projects_dir = str(Path(tmp) / "projects")
    cfg.tick_seconds = 60
    cfg.max_parallel_workers = 3
    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    orch = Orchestrator(cfg, store, runtime)

    conv = FakeConv("orch-conv")
    CreateProjectExecutor()(CreateProjectAction(
        name="Widget", goal="ship a widget", constraints=""), conv)
    pid = list(store.projects)[0]
    CreateTaskExecutor()(CreateTaskAction(
        project_id=pid, title="Build the API", objective="expose the API",
        expertise="backend", acceptance_criteria=["tests pass"],
        deliverables=["api.py"]), conv)
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role="backend-dev", specialty="backend", mission="build the API",
        project_id=pid), conv)
    store.flush()
    task = store.project_tasks(pid)[0]
    agent = [a for a in store.project_agents(pid) if a.depth == 1][0]
    return cfg, store, runtime, orch, pid, task, agent


def _stub_run(runtime, *, turns: int = 5, status: str = "idle") -> list:
    """Replace the LLM call, keeping the board mechanics real.

    _dispatch is what moves a card, and it drives the card off the RunOutcome,
    so stubbing at runtime.run keeps every transition the scheduler is
    responsible for under test without spending a real agent run.
    """
    calls: list[dict] = []

    def fake_run(spec, tools, message, turn_limit=None):
        calls.append({"agent_id": spec.id, "message": message,
                      "turn_limit": turn_limit})
        spec.turns_used = min(spec.turn_limit, spec.turns_used + turns)
        return RunOutcome(turns_used=spec.turns_used, turns_this_run=turns,
                          status=status, final_report="done",
                          budget_exhausted=False, tokens=10, cost_usd=0.0)

    runtime.run = fake_run
    return calls


def _wait_idle(runtime, timeout=5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not runtime.inflight:
            return True
        time.sleep(0.02)
    return False


# ------------------------------------------------- scheduler: what it may start


def test_assigned_card_starts_without_an_orchestrator_turn():
    """The core of the fix: assign_task is a decision, so launching it is a
    consequence and must not need another LLM round trip."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        _stub_run(runtime)
        conv = FakeConv("orch-conv")
        AssignTaskExecutor()(AssignTaskAction(task_id=task.id,
                                             agent_id=agent.id), conv)

        started: list[str] = []
        runtime.submit = lambda aid, fn: (started.append(aid),
                                          fn())[1]  # run inline, no thread
        notes = orch.scheduler.sweep()

        assert started == [agent.id], started
        assert notes and "started backend-dev" in notes[0][1], notes
        assert notes[0][0] == pid, notes
        # A scheduler start is a real dispatch, not just a note: the card ran
        # and, on a clean finish, moved on to review like any worker stretch.
        done = store.tasks[task.id]
        assert done.column == "review", done.column
        assert done.runs == 1 and done.turns_used == 5, done
        print("assigned card starts with no orchestrator turn ok")


def test_unstaffed_card_is_left_for_the_orchestrator():
    """Staffing is judgement. The scheduler only acts on cards the orchestrator
    has already attached an agent to."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        called: list[str] = []
        runtime.submit = lambda aid, fn: called.append(aid)
        notes = orch.scheduler.sweep()
        assert called == [], called
        assert notes == [], notes
        assert store.tasks[task.id].assigned_agent_id is None
        print("unstaffed card left to the orchestrator ok")


def test_scheduler_never_double_starts_a_running_agent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        conv = FakeConv("orch-conv")
        AssignTaskExecutor()(AssignTaskAction(task_id=task.id,
                                             agent_id=agent.id), conv)
        # Keep submit and its inflight bookkeeping real; make the run itself
        # block. Replacing submit would bypass the guard being tested.
        started = threading.Event()
        release = threading.Event()

        def slow_run(spec, tools, message, turn_limit=None):
            started.set()
            release.wait(timeout=10)
            return RunOutcome(turns_used=spec.turns_used, turns_this_run=0,
                              status="idle", final_report="done",
                              budget_exhausted=False, tokens=1, cost_usd=0.0)

        runtime.run = slow_run
        orch.scheduler.sweep()
        assert started.wait(timeout=5), "first start never began"
        # Still in flight: the second sweep must not start it again.
        assert agent.id in runtime.inflight, runtime.inflight
        second = orch.scheduler.sweep()
        assert second == [], second
        release.set()
        assert _wait_idle(runtime), "agent never finished"
        print("scheduler does not double-start ok")


def test_blocked_for_llm_card_is_not_re_dispatched():
    """Re-dispatching into a dead endpoint is what makes a project look stuck:
    turns burn every tick and nothing lands. Those cards wait for the user."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        conv = FakeConv("orch-conv")
        AssignTaskExecutor()(AssignTaskAction(task_id=task.id,
                                             agent_id=agent.id), conv)
        task = store.tasks[task.id]
        task.column = "blocked"
        task.blocked_cause = "llm"
        task.blocked_reason = "LLM endpoint returned no output"
        store.upsert_task(task)
        called: list[str] = []
        runtime.submit = lambda aid, fn: called.append(aid)
        assert orch.scheduler.sweep() == []
        assert called == [], called

        # Same for an agent whose own run reported the endpoint down.
        task = store.tasks[task.id]
        task.column, task.blocked_cause, task.blocked_reason = (
            "ready", None, None)
        store.upsert_task(task)
        agent = store.agents[agent.id]
        agent.llm_failures = 1
        agent.status = "llm_unavailable"
        store.upsert_agent(agent)
        assert orch.scheduler.sweep() == []
        assert called == [], called
        print("llm-blocked card not re-dispatched ok")


def test_review_card_is_reviewed_once_not_forever():
    """A reviewer run does not move its own card - record_review does. Without a
    guard, the scheduler would re-dispatch the same reviewer every tick."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        _stub_run(runtime)
        conv = FakeConv("orch-conv")
        task = store.tasks[task.id]
        task.column = "review"
        task.reviewer_agent_id = agent.id
        store.upsert_task(task)

        called: list[str] = []
        runtime.submit = lambda aid, fn: called.append(aid)
        notes = orch.scheduler.sweep()
        assert called == [agent.id], called
        assert any("as reviewer" in n[1] for n in notes), notes

        # Verdict recorded: the card is no longer waiting on a review.
        task = store.tasks[task.id]
        task.review_verdict = "changes_requested"
        task.review_rounds = 1
        store.upsert_task(task)
        called.clear()
        assert orch.scheduler.sweep() == []
        assert called == [], called
        print("review card reviewed once ok")


def test_dependency_order_and_capacity_backpressure():
    """A card whose dependency is unfinished must not start, and when no slot
    is free the scheduler must say so instead of queueing silently."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        conv = FakeConv("orch-conv")
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="Integrate", objective="wire it up",
            expertise="backend", acceptance_criteria=["works"],
            deliverables=["main.py"], depends_on=[task.id]), conv)
        dependent = [t for t in store.project_tasks(pid)
                     if t.title == "Integrate"][0]
        AssignTaskExecutor()(AssignTaskAction(task_id=dependent.id,
                                             agent_id=agent.id), conv)
        called: list[str] = []
        runtime.submit = lambda aid, fn: called.append(aid)
        assert orch.scheduler.sweep() == []
        assert called == [], called

        # Capacity: with every worker slot held, nothing may start, and the
        # briefing has to be told the card is waiting on capacity - not on a
        # decision - so the orchestrator does not burn a turn on it.
        AssignTaskExecutor()(AssignTaskAction(task_id=task.id,
                                             agent_id=agent.id), conv)
        # Slots are held per thread, so saturating the gate needs real threads
        # - one thread can never occupy more than one slot.
        hold = threading.Event()
        entered = threading.Event()
        count = runtime.gate.limit

        def hold_slot():
            runtime.gate.acquire()
            entered.set()
            hold.wait(timeout=10)
            runtime.gate.release()

        threads = [threading.Thread(target=hold_slot, daemon=True)
                   for _ in range(count)]
        for t in threads:
            t.start()
        assert entered.wait(timeout=5)
        time.sleep(0.2)
        assert runtime.capacity()["slots_free"] == 0, runtime.capacity()
        try:
            notes = orch.scheduler.sweep()
            assert called == [], called
            assert notes and "waiting on capacity" in notes[0][1], notes
        finally:
            hold.set()
            for t in threads:
                t.join(timeout=5)
        print("dependency order and capacity backpressure ok")


def test_auto_dispatch_can_be_turned_off():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        cfg.auto_dispatch = False
        conv = FakeConv("orch-conv")
        AssignTaskExecutor()(AssignTaskAction(task_id=task.id,
                                             agent_id=agent.id), conv)
        called: list[str] = []
        runtime.submit = lambda aid, fn: called.append(aid)
        assert orch.scheduler.sweep() == []
        assert called == [], called
        print("auto_dispatch=false honoured ok")


# ------------------------------------- the stuck-cycle signature and its fixes


def test_scheduler_runs_when_the_cycle_is_refused():
    """The reported state: one orchestrator prompt is executing, so every cycle
    is refused, and nothing else moves.

    Refusing a stacked cycle is correct. But the scheduler must still run,
    because it needs no model: work that is already staffed and unblocked should
    start even while the orchestrator is blocked on its own prompt.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        _stub_run(runtime)
        conv = FakeConv("orch-conv")
        AssignTaskExecutor()(AssignTaskAction(task_id=task.id,
                                             agent_id=agent.id), conv)

        # Simulate an orchestrator cycle holding the loop.
        assert orch._cycle_lock.acquire(blocking=False)
        try:
            assert orch.run_cycle() is None, "cycle should be refused"
            assert orch.orchestrator_busy() is not None
            started: list[str] = []
            runtime.submit = lambda aid, fn: started.append(aid)
            notes = orch._run_scheduler()
            assert started == [agent.id], started
            assert notes and "started backend-dev" in notes[0][1], notes
        finally:
            orch._cycle_lock.release()
        print("scheduler runs while cycles are refused ok")


def test_stuck_orchestrator_prompt_is_visible_and_bounded():
    """An orchestrator prompt that has been sitting 'awaiting' for 900s must be
    reported as the reason cycles are skipped, with the elapsed time attached."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        entry = runtime._enqueue_llm(orch.spec, "the briefing", "orchestrator")
        runtime._begin_llm(entry)
        why = orch.orchestrator_busy()
        assert why is not None and "executing" in why, why
        assert orch.run_cycle() is None, "cycle should be refused"
        runtime._drop_llm(entry)
        assert orch.orchestrator_busy() is None
        print("stuck orchestrator prompt surfaced ok")


def test_empty_completion_does_not_widen_the_timeout():
    """An empty completion means the endpoint answered cleanly and the model
    said nothing. Widening the deadline for that is backwards, and it is how a
    harness drifts to a 3600s timeout that then holds a slot for an hour."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        cfg.llm.request_timeout = 300
        cfg.llm.empty_response_retries = 2
        runtime.llm_timeout = 300

        # Three clean runs that produce zero agent turns: an empty completion.
        conv = FlakyConv([0, 0, 0])
        runtime.open_conversation = lambda *a, **k: conv
        spec = store.agents[agent.id]
        runtime.run(spec, [], "do the work")

        assert runtime.llm_empty_responses >= 1, runtime.llm_empty_responses
        assert runtime.llm_timeouts_widened == 0, runtime.llm_timeouts_widened
        assert runtime.llm_timeout == 300, runtime.llm_timeout
        print("empty completion leaves the timeout alone ok")


def test_timeout_decays_back_after_clean_runs():
    """Widening without a way back is permanent damage: the value lives in
    store.meta and survives restarts, so one bad afternoon lets every later
    request hang for an hour."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, task, agent = seed(tmp)
        cfg.llm.request_timeout = 300
        cfg.llm.max_request_timeout = 900
        cfg.llm.timeout_decay_after_runs = 2

        runtime.set_llm_timeout(900)
        assert runtime.llm_timeout == 900
        # Not yet: a single clean run should not undo a widening the next slow
        # request may still need.
        assert runtime.decay_llm_timeout() == 900
        # Two clean runs: halve, not snap, so an intermittently slow endpoint
        # keeps some headroom.
        assert runtime.decay_llm_timeout() == 450
        assert runtime.decay_llm_timeout() == 450
        assert runtime.decay_llm_timeout() == 300
        # And it stops at the configured value rather than going lower.
        for _ in range(4):
            assert runtime.decay_llm_timeout() == 300
        print("timeout decays back to the configured value ok")


def test_persisted_timeout_is_clamped_on_startup():
    """A state file written by an older build can carry a runaway value. It must
    not be trusted back in."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = HarnessConfig()
        cfg.data_dir = tmp
        cfg.projects_dir = str(Path(tmp) / "projects")
        cfg.llm.request_timeout = 300
        cfg.llm.max_request_timeout = 900
        store = Store(cfg.data_dir)
        store.meta["llm_timeout"] = 86400
        store.flush()
        runtime = AgentRuntime(cfg, store)
        assert runtime.llm_timeout == 900, runtime.llm_timeout

        # An explicitly configured timeout above the widening ceiling is the
        # user's choice, not runaway drift, and must survive.
        cfg2 = HarnessConfig()
        cfg2.data_dir = tempfile.mkdtemp()
        cfg2.llm = LLMConfig(request_timeout=3600, max_request_timeout=900)
        runtime2 = AgentRuntime(cfg2, Store(cfg2.data_dir))
        assert runtime2.llm_timeout == 3600, runtime2.llm_timeout
        print("persisted timeout clamped on startup ok")


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"--- {t.__name__}")
        t()
    print(f"\nALL SCHEDULER / STUCK-CYCLE TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
