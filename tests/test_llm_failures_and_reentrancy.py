"""Orchestrator re-entrancy, LLM-failure alerting, and the done column.

These cover who owns a failure. An LLM endpoint that is down or timing out is
the user's problem: the harness reports it and stops. An agent that ran out of
budget or crashed is the orchestrator's problem. Getting that boundary wrong
shows up as a project that looks 'stuck' - the orchestrator quietly re-dispatches
into a dead endpoint every tick, burning turns and producing nothing.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from agent_harness.config import HarnessConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime
from agent_harness.store import Store
from agent_harness.tools import (
    CreateProjectAction,
    CreateProjectExecutor,
    CreateSubAgentAction,
    CreateSubAgentExecutor,
    CreateTaskAction,
    CreateTaskExecutor,
    register_harness_tools,
)
from agent_harness.web.api import build_api


class FakeConv:
    def __init__(self, cid="conv-1"):
        self.id = cid


def seed(tmp: str, name: str = "Widget"):
    register_harness_tools()
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    cfg.projects_dir = str(Path(tmp) / "projects")
    cfg.tick_seconds = 60
    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    orch = Orchestrator(cfg, store, runtime)

    conv = FakeConv("orch-conv")
    CreateProjectExecutor()(CreateProjectAction(
        name=name, goal="ship a widget", constraints=""), conv)
    pid = list(store.projects)[0]
    CreateTaskExecutor()(CreateTaskAction(
        project_id=pid, title="Build the API", objective="expose the API",
        expertise="backend", acceptance_criteria=["tests pass"],
        deliverables=["api.py"]), conv)
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role="backend-dev", specialty="backend", mission="build the API",
        project_id=pid), conv)
    store.flush()
    return cfg, store, runtime, orch


def _llm_blocked(store, pid):
    """Put the seeded card into the state a dead endpoint leaves behind."""
    task = store.project_tasks(pid)[0]
    agent = [a for a in store.project_agents(pid) if a.depth == 1][0]
    agent.status = "llm_unavailable"
    agent.llm_failures = 2
    agent.llm_failure_error = "APIConnectionError: connection refused"
    agent.task_id = task.id
    task.column = "blocked"
    task.blocked_reason = "LLM endpoint returned no output"
    task.blocked_cause = "llm"
    task.assigned_agent_id = agent.id
    store.upsert_agent(agent)
    store.upsert_task(task)
    store.flush()
    return task, agent


# ----------------------------------------------- orchestrator re-entrancy


def test_orchestrator_never_queues_behind_itself():
    """A second cycle is refused while one runs, not queued behind it.

    A queued cycle carries a briefing built from the board as it looked before
    the running cycle moved anything, so it would act on stale state and then
    overwrite the fresher decisions.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        started = threading.Event()
        release = threading.Event()

        def slow_cycle():
            started.set()
            release.wait(timeout=5)
            return ["ran"]

        orch._run_cycle = slow_cycle
        t = threading.Thread(target=orch.run_cycle, daemon=True)
        t.start()
        assert started.wait(timeout=3), "cycle never started"

        assert orch.run_cycle() is None, "second cycle was not refused"
        assert orch.run_cycle() is None, "third cycle was not refused"
        assert orch.orchestrator_busy() == "a cycle is already running"

        release.set()
        t.join(timeout=3)
        assert orch.orchestrator_busy() is None
        assert orch.run_cycle() == ["ran"], "must run again once free"
        print("orchestrator refuses to queue ok")


def test_queued_orchestrator_prompt_blocks_a_new_cycle():
    """'Queued to execute' counts as busy, not just 'currently executing'.

    A prompt that has been written and is waiting on an LLM slot is already
    committed to a briefing; starting another cycle would supersede it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        entry = runtime._enqueue_llm(orch.spec, "briefing", "orchestrator")
        assert "queued" in (orch.orchestrator_busy() or ""), "queued prompt ignored"
        assert orch.run_cycle() is None, "cycle started over a queued prompt"

        # A worker prompt, even an in-flight one, is not the orchestrator.
        runtime._drop_llm(entry)
        worker = [a for a in store.project_agents(list(store.projects)[0])
                  if a.depth == 1][0]
        runtime._enqueue_llm(worker, "work", "worker")
        assert orch.orchestrator_busy() is None, "worker blocked the cycle"
        print("queued orchestrator prompt blocks a cycle ok")


def test_api_tick_refuses_to_stack_cycles():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        client = TestClient(build_api(cfg, store, orch))
        orch._cycle_lock.acquire()
        try:
            body = client.post("/api/tick").json()
            assert body["started"] is False, body
            assert "not queued" in body["reason"], body
        finally:
            orch._cycle_lock.release()
        print("api tick refuses to stack ok")


# ------------------------------------------------------ llm failure paths


def test_llm_error_alerts_the_user_and_names_the_failure():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task, _ = _llm_blocked(store, pid)

        alerts = TestClient(build_api(cfg, store, orch)).get(
            "/api/state").json()["alerts"]
        llm = [a for a in alerts if "LLM error" in a["message"]]
        assert len(llm) == 1, alerts
        assert llm[0]["level"] == "error", llm
        assert "connection refused" in llm[0]["message"], llm
        assert "2 run(s) in a row" in llm[0]["message"], llm
        assert task.title in llm[0]["message"], "the card is not named"
        print("llm error alert ok")


def test_acknowledging_an_llm_alert_releases_the_blocked_card():
    """The ack is the user's signal that the endpoint was dealt with.

    Without it, a card the harness blocked for an LLM outage would sit blocked
    forever: the orchestrator is told not to touch it, and nothing else can
    move it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task, agent = _llm_blocked(store, pid)

        client = TestClient(build_api(cfg, store, orch))
        alerts = client.get("/api/state").json()["alerts"]
        target = [a for a in alerts if "LLM error" in a["message"]][0]["id"]

        r = client.post("/api/alerts/ack", json={"alert_id": target})
        assert r.status_code == 200, r.text
        assert r.json()["released_agents"] == [agent.id], r.json()

        assert task.column == "ready", task.column
        assert task.blocked_cause is None and task.blocked_reason is None
        assert store.agents[agent.id].llm_failures == 0
        assert store.agents[agent.id].status == "idle"
        assert not [a for a in client.get("/api/state").json()["alerts"]
                    if "LLM error" in a["message"]], "alert still up"
        assert any(e.kind == "llm_block_released" for e in store.recent_events)
        print("ack releases llm block ok")


def test_llm_blocked_card_is_not_sent_to_the_orchestrator():
    """The orchestrator is told to stand down, not to fix the endpoint.

    Re-dispatching into a dead endpoint every tick is how a project looks
    'stuck': it burns turns and produces nothing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task, _ = _llm_blocked(store, pid)

        joined = "\n".join(orch.detect()[pid])
        assert "DO NOT ACT ON IT" in joined, joined
        assert "connection refused" in joined, joined
        assert "RECOVERY DECISION" not in joined, joined
        for verb in ("reconfigure or replace the agent", "extend it",
                     "re-dispatch the same agent", "split it into smaller"):
            assert verb not in joined, verb

        # A user directive still wins: it may be the news that the endpoint is
        # back, and only the orchestrator can turn that into work.
        task.user_directives = [{"text": "endpoint is fixed, carry on",
                                 "at": "now"}]
        store.upsert_task(task)
        joined = "\n".join(orch.detect()[pid])
        assert "supplied direction" in joined, joined
        print("llm block kept off the orchestrator ok")


def test_llm_outage_is_not_reported_as_an_agent_failure():
    """'failed' sends the orchestrator off reconfiguring an agent that works."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task, agent = _llm_blocked(store, pid)
        agent.status = "failed"
        agent.llm_failures = 3
        agent.llm_failure_error = "APITimeoutError: timed out"
        store.upsert_agent(agent)

        alerts = TestClient(build_api(cfg, store, orch)).get(
            "/api/state").json()["alerts"]
        assert any("LLM error" in a["message"] for a in alerts), alerts
        assert not any(agent.id in a["message"] and "FAILED" in a["message"]
                       for a in alerts), alerts
        print("llm outage not labelled an agent failure ok")


def test_cycle_failure_alerts_the_user():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        store.meta.update({
            "cycle_status": "failed",
            "cycle_error": "RuntimeError: orchestrator LLM unreachable",
            "cycle_count": 7,
        })
        store.flush()
        alerts = TestClient(build_api(cfg, store, orch)).get(
            "/api/state").json()["alerts"]
        cyc = [a for a in alerts if "Orchestrator cycle failed" in a["message"]]
        assert len(cyc) == 1 and cyc[0]["level"] == "error", alerts
        assert "unreachable" in cyc[0]["message"], cyc
        print("cycle failure alert ok")


def test_all_endpoints_down_raises_one_error_alert():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)

        class FakeRouter:
            def status(self):
                return {"servers": [{"name": "primary", "reachable": False},
                                    {"name": "backup", "reachable": False}]}

        runtime.router = FakeRouter()
        client = TestClient(build_api(cfg, store, orch))
        down = [a for a in client.get("/api/state").json()["alerts"]
                if "unreachable" in a["message"]]
        assert len(down) == 1 and down[0]["level"] == "error", down
        assert "primary, backup" in down[0]["message"], down

        # A reachable endpoint clears it.
        runtime.router = None
        assert not [a for a in client.get("/api/state").json()["alerts"]
                    if "unreachable" in a["message"]]
        print("all-endpoints-down alert ok")


def test_removed_projects_raise_no_llm_alerts():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        _llm_blocked(store, pid)
        client = TestClient(build_api(cfg, store, orch))
        assert any("LLM error" in a["message"]
                   for a in client.get("/api/state").json()["alerts"])
        store.remove_project(pid, "hide")
        assert not any("LLM error" in a["message"]
                       for a in client.get("/api/state").json()["alerts"])
        print("removed projects raise no llm alerts ok")


# ------------------------------------------------------------- done column


def test_done_column_collapses_to_a_summary_card_at_twenty():
    """The 20 most recent cards render; the rest fold into one summary card."""
    js = (Path(__file__).resolve().parents[1]
          / "agent_harness" / "web" / "static" / "app.js").read_text()
    assert "const DONE_VISIBLE = 20;" in js, "done column must show 20"
    assert 'class="task done summary"' in js, "summary must be a card, not a button"
    assert "earlier done card" in js, "summary must state the count"
    assert "hidden" in js, "summary must state how many are hidden"
    # The summary card holds inner spans, so the click must be matched on the
    # container or a click on the tag falls through to the task-detail handler.
    assert 'closest("[data-more-done]")' in js, "summary click handler missing"
    print("done column summary card ok")


def main() -> None:
    test_orchestrator_never_queues_behind_itself()
    test_queued_orchestrator_prompt_blocks_a_new_cycle()
    test_api_tick_refuses_to_stack_cycles()
    test_llm_error_alerts_the_user_and_names_the_failure()
    test_acknowledging_an_llm_alert_releases_the_blocked_card()
    test_llm_blocked_card_is_not_sent_to_the_orchestrator()
    test_llm_outage_is_not_reported_as_an_agent_failure()
    test_cycle_failure_alerts_the_user()
    test_all_endpoints_down_raises_one_error_alert()
    test_removed_projects_raise_no_llm_alerts()
    test_done_column_collapses_to_a_summary_card_at_twenty()
    print("\nALL LLM-FAILURE / RE-ENTRANCY TESTS PASSED")


if __name__ == "__main__":
    main()
