"""LLM action queue, project-level user input, and blocked-card recovery.

Run: python3 tests/test_queue_input_recovery.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from agent_harness.config import HarnessConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime, RunOutcome
from agent_harness.store import AgentSpec, Store, TaskSpec
from agent_harness.tools import (
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
from agent_harness.web.api import build_api


class FakeConv:
    def __init__(self, cid="conv-1"):
        self.id = cid


def make(tmp: str):
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    cfg.projects_dir = str(Path(tmp) / "projects")
    cfg.tick_seconds = 60
    register_harness_tools()
    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    orch = Orchestrator(cfg, store, runtime)
    return cfg, store, runtime, orch


def seed(tmp: str, name: str = "Widget"):
    cfg, store, runtime, orch = make(tmp)
    conv = FakeConv()
    CreateProjectExecutor()(CreateProjectAction(
        name=name, goal="ship a widget", constraints=""), conv)
    pid = list(store.projects)[0]
    CreateTaskExecutor()(CreateTaskAction(
        project_id=pid, title="Build it", objective="x", expertise="backend",
        acceptance_criteria=["works"], deliverables=["artifact.py"]), conv)
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role="backend-dev", specialty="backend", mission="build",
        project_id=pid), conv)
    return cfg, store, runtime, orch


def outcome(status="idle", turns=1) -> RunOutcome:
    return RunOutcome(turns_used=turns, turns_this_run=turns, status=status,
                      final_report="done", budget_exhausted=False,
                      tokens=10, cost_usd=0.01)


# ------------------------------------------------------- LLM action queue


def test_run_queues_then_executes_then_clears():
    """A run is visible as 'queued' before it gets a slot, 'executing' after."""
    cfg, store, runtime, orch = make(tempfile.mkdtemp())
    agent = AgentSpec(role="python-engineer", specialty="py", mission="m",
                      system_prompt="p", project_id="p1", depth=1)
    store.upsert_agent(agent)

    entered = threading.Event()
    release = threading.Event()
    states: list[list[tuple[str, str]]] = []

    def snapshot() -> list[tuple[str, str]]:
        return [(e["agent_role"], e["status"]) for e in runtime.llm_queue()]

    def fake_guarded(spec, tools, message, turn_limit, **_kw):
        states.append(snapshot())
        entered.set()
        release.wait(10)
        return outcome()

    runtime._run_guarded = fake_guarded
    runtime.gate.set_limit(1)

    # Occupy the only slot from another thread so the run must queue first.
    hold = threading.Event()
    blocker = threading.Thread(
        target=lambda: (runtime.gate.acquire(), hold.set(),
                        release.wait(10), runtime.gate.release()))
    blocker.start()
    assert hold.wait(5), "blocker never took the slot"

    worker = threading.Thread(target=lambda: runtime.run(agent, [], "write tests"))
    worker.start()
    deadline = time.time() + 5
    while not runtime.llm_queue() and time.time() < deadline:
        time.sleep(0.02)
    queued = runtime.llm_queue()
    assert len(queued) == 1, queued
    assert queued[0]["status"] == "queued", "run should wait, not execute"
    assert queued[0]["agent_role"] == "python-engineer"
    assert queued[0]["prompt_preview"] == "write tests"
    assert queued[0]["kind"] == "worker", queued[0]["kind"]

    release.set()
    assert entered.wait(10), "run never reached the LLM"
    assert states[-1][0][1] == "executing", states[-1]

    worker.join(10)
    hold.set()
    blocker.join(10)
    assert runtime.llm_queue() == [], "entry leaked after the run finished"
    print("queue lifecycle queued->executing->cleared ok")


def test_queue_orders_by_request_and_keeps_full_prompt():
    cfg, store, runtime, orch = make(tempfile.mkdtemp())
    a1 = AgentSpec(role="first", specialty="x", mission="m", system_prompt="p")
    a2 = AgentSpec(role="second", specialty="x", mission="m", system_prompt="p")
    store.upsert_agent(a1)
    store.upsert_agent(a2)

    seen = threading.Event()
    release = threading.Event()

    def fake_guarded(spec, tools, message, turn_limit, **_kw):
        seen.set()
        release.wait(10)
        return outcome()

    runtime._run_guarded = fake_guarded
    runtime.gate.set_limit(1)

    hold = threading.Event()
    threading.Thread(target=lambda: (runtime.gate.acquire(), hold.set(),
                                     release.wait(10),
                                     runtime.gate.release())).start()
    hold.wait(5)

    threading.Thread(target=lambda: runtime.run(a1, [], "prompt one")).start()
    deadline = time.time() + 5
    while len(runtime.llm_queue()) < 1 and time.time() < deadline:
        time.sleep(0.02)
    threading.Thread(target=lambda: runtime.run(a2, [], "prompt two")).start()
    deadline = time.time() + 5
    while len(runtime.llm_queue()) < 2 and time.time() < deadline:
        time.sleep(0.02)

    order = [e["agent_role"] for e in runtime.llm_queue()]
    assert order == ["first", "second"], (
        f"API must return request order; UI reverses it: {order}")
    seqs = [e["seq"] for e in runtime.llm_queue()]
    assert seqs == sorted(seqs), seqs

    newest = runtime.llm_queue()[-1]
    assert runtime.llm_prompt(newest["id"]) == "prompt two"
    assert runtime.llm_prompt("llm-nope") is None

    release.set()
    time.sleep(0.4)
    print("queue ordering and prompt retrieval ok")


def test_queue_endpoints_and_state_exposure():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    agent = store.project_agents(pid)[0]
    agent.project_id = pid

    hold = threading.Event()
    release = threading.Event()

    def fake_guarded(spec, tools, message, turn_limit, **_kw):
        release.wait(10)
        return outcome()

    runtime._run_guarded = fake_guarded
    threading.Thread(target=lambda: runtime.run(agent, [], "the real prompt")).start()
    deadline = time.time() + 5
    while not runtime.llm_queue() and time.time() < deadline:
        time.sleep(0.02)

    app = build_api(cfg, store, orch)
    client = TestClient(app)

    entry = runtime.llm_queue()[0]
    q = client.get(f"/api/projects/{pid}/queue").json()["queue"]
    assert [e["id"] for e in q] == [entry["id"]], q
    assert q[0]["agent_role"] == agent.role

    body = client.get(f"/api/queue/{entry['id']}/prompt").json()
    assert body["prompt"] == "the real prompt", body

    state = client.get("/api/state").json()
    assert [e["id"] for e in state["llm_queue"]] == [entry["id"]]

    missing = client.get("/api/queue/llm-gone/prompt")
    assert missing.status_code == 404, missing.status_code
    assert client.get("/api/projects/nope/queue").status_code == 404

    release.set()
    time.sleep(0.4)
    print("queue endpoints ok")


def test_orchestrator_cycle_shows_as_orchestrator_kind():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    kinds: list[str] = []

    def fake_guarded(spec, tools, message, turn_limit, **_kw):
        kinds.append(runtime.llm_queue()[0]["kind"])
        return outcome()

    runtime._run_guarded = fake_guarded
    orch._run_project_cycle(store.projects[list(store.projects)[0]], ["do work"])
    assert kinds == ["orchestrator"], kinds
    print("orchestrator run labelled ok")


# --------------------------------------------------- project user input


def test_project_message_endpoint_queues_and_reopens():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    app = build_api(cfg, store, orch)
    client = TestClient(app)

    r = client.post(f"/api/projects/{pid}/messages",
                    json={"text": "also add a CSV export"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["messages"] == 1 and body["reopened"] is False
    assert store.projects[pid].user_messages[0]["text"] == "also add a CSV export"

    assert client.post(f"/api/projects/{pid}/messages", json={"text": "  "}).status_code == 400
    assert client.post(f"/api/projects/nope/messages",
                       json={"text": "x"}).status_code == 404

    # A delivered project would be skipped by the loop, so a new request has to
    # reopen it or the button would do nothing after delivery.
    store.projects[pid].status = "delivered"
    store.flush()
    r2 = client.post(f"/api/projects/{pid}/messages",
                     json={"text": "now add rate limiting"})
    assert r2.json()["reopened"] is True
    assert store.projects[pid].status == "active"

    state = client.get("/api/state").json()
    proj = [p for p in state["projects"] if p["id"] == pid][0]
    assert [m["text"] for m in proj["user_messages"]] == [
        "also add a CSV export", "now add rate limiting"]
    print("project message endpoint ok")


def test_user_input_reaches_attention_and_briefing():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    store.projects[pid].user_messages.append({"text": "add CSV export", "at": "now"})

    attention = orch.detect()[pid]
    joined = "\n".join(attention)
    assert "NEW USER INPUT" in joined, attention
    assert "add CSV export" in joined

    captured: list[str] = []

    def fake_guarded(spec, tools, message, turn_limit, **_kw):
        captured.append(message)
        return outcome()

    runtime._run_guarded = fake_guarded
    orch._run_project_cycle(store.projects[pid], attention)
    assert '"user_input_pending"' in captured[0], captured[0][:400]
    assert "add CSV export" in captured[0]
    print("user input reaches briefing ok")


def test_cycle_consumes_messages_and_restores_them_on_failure():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    store.projects[pid].user_messages.append({"text": "add CSV export", "at": "now"})
    store.flush()

    runtime._run_guarded = lambda *a, **k: outcome()
    orch._run_project_cycle(store.projects[pid], ["x"])
    assert store.projects[pid].user_messages == [], "message replayed every cycle"

    # An LLM outage means the briefing was never acted on: the request must come
    # back rather than vanish.
    store.projects[pid].user_messages.append({"text": "keep me", "at": "now"})
    runtime._run_guarded = lambda *a, **k: outcome(status="llm_unavailable")
    orch._run_project_cycle(store.projects[pid], ["x"])
    texts = [m["text"] for m in store.projects[pid].user_messages]
    assert texts == ["keep me"], texts

    # A hard crash must not lose it either.
    def boom(*a, **k):
        raise RuntimeError("exploded")

    runtime._run_guarded = boom
    try:
        orch._run_project_cycle(store.projects[pid], ["x"])
    except RuntimeError:
        pass
    texts = [m["text"] for m in store.projects[pid].user_messages]
    assert texts == ["keep me"], texts
    print("message consumption/restore ok")


# ------------------------------------------------- blocked-card recovery


def blocked_task(store, pid, reason="agent used all 100 turns", cause="budget"):
    task = TaskSpec(project_id=pid, title="Big job", objective="do the big thing",
                    expertise="backend", column="blocked",
                    blocked_reason=reason, blocked_cause=cause,
                    turns_used=100, turn_limit=100)
    store.upsert_task(task)
    return task


def test_every_blocked_card_is_evaluated_each_tick():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    blocked_task(store, pid)
    blocked_task(store, pid, reason="worker said no", cause="worker")

    attention = orch.detect()[pid]
    joined = "\n".join(attention)
    assert joined.count("RECOVERY DECISION") == 2, attention
    for opt in ("1. MORE TURNS", "2. RE-SPECIFY", "3. SPLIT"):
        assert opt in joined, opt
    print("blocked cards evaluated with 3 options ok")


def test_strikes_escalate_and_reset():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    task = blocked_task(store, pid)

    counts = [orch.detect()[pid] for _ in range(3)]
    assert task.block_strikes == 3, task.block_strikes
    assert task.block_strike_reason == task.blocked_reason
    assert "blocked 3 cycle(s) in a row" in "\n".join(counts[2])
    assert "ESCALATION" in "\n".join(counts[2]), "no escalation at 3 strikes"
    assert "ESCALATION" not in "\n".join(counts[0]), "escalated too early"

    # A different blocker is a different problem: the count restarts.
    task.blocked_reason = "endpoint is down"
    task.blocked_cause = "llm"
    store.upsert_task(task)
    orch.detect()
    assert task.block_strikes == 1, task.block_strikes

    # Landing the card clears the grudge so a later block starts fresh.
    task.column = "done"
    store.upsert_task(task)
    orch.reconcile()
    assert task.block_strikes == 0 and task.block_strike_reason is None
    print("strike escalation/reset ok")


def test_recovery_menu_names_the_exhausted_agent():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    task = blocked_task(store, pid)
    agent = store.project_agents(pid)[0]
    agent.status = "budget_exhausted"
    agent.turns_used, agent.turn_limit = 100, 100
    agent.task_id = task.id
    store.upsert_agent(agent)
    task.assigned_agent_id = agent.id
    store.upsert_task(task)

    joined = "\n".join(orch.detect()[pid])
    assert f"{agent.id} ({agent.role}) is budget_exhausted" in joined, joined
    print("recovery menu cites the exhausted agent ok")


def test_split_children_link_to_the_parent_card():
    cfg, store, runtime, orch = seed(tempfile.mkdtemp())
    pid = list(store.projects)[0]
    parent = blocked_task(store, pid)
    conv = FakeConv()

    CreateTaskExecutor()(CreateTaskAction(
        project_id=pid, title="Part two", objective="second half",
        expertise="backend", parent_task_id=parent.id), conv)
    child = [t for t in store.tasks.values() if t.title == "Part two"][0]
    assert child.parent_task_id == parent.id

    bad = CreateTaskExecutor()(CreateTaskAction(
        project_id=pid, title="Orphan", objective="x",
        parent_task_id="task-nope"), conv)
    assert "Unknown parent_task_id" in str(bad), bad

    # The parent's own re-specification is what update_task already supports.
    UpdateTaskExecutor()(UpdateTaskAction(
        task_id=parent.id, objective="first half only",
        column="backlog", note="split into two"), conv)
    assert store.tasks[parent.id].column == "backlog"
    assert store.tasks[parent.id].blocked_reason is None

    app = build_api(cfg, store, orch)
    d = TestClient(app).get("/api/state").json()
    proj = [p for p in d["projects"] if p["id"] == pid][0]
    cards = {t["id"]: t for c in proj["board"].values() for t in c}
    assert cards[child.id]["parent_task_id"] == parent.id
    print("split traceability ok")


def test_ui_assets_wire_the_queue_and_message_box():
    js = (Path(__file__).resolve().parents[1]
          / "agent_harness" / "web" / "static" / "app.js").read_text()
    html = (Path(__file__).resolve().parents[1]
            / "agent_harness" / "web" / "templates" / "index.html").read_text()
    assert "function renderQueue" in js, "queue strip missing"
    assert 'data-queue=' in js, "queue cards not clickable"
    assert "/api/queue/" in js, "prompt fetch missing"
    assert 'status === "executing"' in js, "executing lane missing"
    assert ".reverse()" in js, "newest-left ordering missing"
    assert "renderMessageBar" in js, "project message box missing"
    assert "/messages" in js, "message POST missing"
    # The strip must be right on first paint, not only after the fast poll.
    assert "d.llm_queue" in js, "refresh() must seed queues from /api/state"
    for sel in (".qcard", ".qlane.exec", ".msgform"):
        assert sel in html, f"{sel} style missing"
    print("queue/message UI contract ok")


def main() -> None:
    for fn in (test_run_queues_then_executes_then_clears,
               test_queue_orders_by_request_and_keeps_full_prompt,
               test_queue_endpoints_and_state_exposure,
               test_orchestrator_cycle_shows_as_orchestrator_kind,
               test_project_message_endpoint_queues_and_reopens,
               test_user_input_reaches_attention_and_briefing,
               test_cycle_consumes_messages_and_restores_them_on_failure,
               test_every_blocked_card_is_evaluated_each_tick,
               test_strikes_escalate_and_reset,
               test_recovery_menu_names_the_exhausted_agent,
               test_split_children_link_to_the_parent_card,
               test_ui_assets_wire_the_queue_and_message_box):
        print(f"--- {fn.__name__}")
        fn()
    print("\nALL QUEUE/INPUT/RECOVERY TESTS PASSED")


if __name__ == "__main__":
    main()
