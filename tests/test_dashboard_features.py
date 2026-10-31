"""Project removal modes, cost surfacing, cycle freshness, alert acks."""

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
    """Create a project with a workspace file, a task and two agents."""
    cfg, store, runtime, orch = make(tmp)
    conv = FakeConv()
    CreateProjectExecutor()(CreateProjectAction(
        name=name, goal="ship a widget", constraints=""), conv)
    pid = list(store.projects)[0]
    Path(store.projects[pid].workspace, "artifact.py").write_text("print(1)\n")
    CreateTaskExecutor()(CreateTaskAction(
        project_id=pid, title="Build it", objective="x", expertise="backend",
        acceptance_criteria=["works"], deliverables=["artifact.py"]), conv)
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role="backend-dev", specialty="backend", mission="build",
        project_id=pid), conv)
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role="reviewer", specialty="review", mission="review",
        project_id=pid), conv)
    for a in store.project_agents(pid):
        a.cost_usd = 1.25
        a.tokens = 1000
    store.flush()
    return cfg, store, runtime, orch


# ---------------------------------------------------------------- removal


def test_remove_hide_keeps_files_and_stops_work():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        workspace = Path(store.projects[pid].workspace)
        assert (workspace / "artifact.py").exists()

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        r = client.post(f"/api/projects/{pid}/remove",
                        json={"mode": "hide", "confirm": "Widget"})
        assert r.status_code == 200, r.text

        assert (workspace / "artifact.py").exists(), "hide must not touch files"
        assert store.projects[pid].status == "removed"
        assert store.projects[pid].removal_mode == "hide"

        # Gone from the dashboard and from the orchestrator's workload.
        assert pid not in [p["id"] for p in client.get("/api/state").json()["projects"]]
        assert pid not in store.visible_projects()
        orch.reconcile()
        assert orch.detect().get(pid, []) == [] or pid not in orch.detect()
        print("remove/hide ok")


def test_remove_archives_workspace_and_keeps_record():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        workspace = Path(store.projects[pid].workspace)
        conv_dir = cfg.root_dir / "conversations" / pid
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "log.json").write_text("{}")

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        r = client.post(f"/api/projects/{pid}/remove",
                        json={"mode": "archive", "confirm": "Widget"})
        assert r.status_code == 200, r.text

        assert not workspace.exists(), "archive must move the workspace out"
        archive = cfg.root_dir / "archived-projects"
        moved = list(archive.glob("Widget-*"))
        assert moved, f"expected an archived workspace under {archive}"
        assert (moved[0] / "artifact.py").exists()
        assert list(archive.glob(f"{pid}-*")), "conversation logs not archived"
        assert not conv_dir.exists()

        # Record survives so cost history and the feed remain queryable.
        assert store.projects[pid].status == "removed"
        assert store.projects[pid].removal_mode == "archive"
        print("remove/archive ok")


def test_remove_purge_deletes_everything():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        workspace = Path(store.projects[pid].workspace)
        conv_dir = cfg.root_dir / "conversations" / pid
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "log.json").write_text("{}")

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        r = client.post(f"/api/projects/{pid}/remove",
                        json={"mode": "purge", "confirm": "Widget"})
        assert r.status_code == 200, r.text

        assert not workspace.exists(), "purge must delete the workspace"
        assert not conv_dir.exists(), "purge must delete conversation logs"
        assert pid not in store.projects, "purge must drop the record"
        assert not [t for t in store.tasks.values() if t.project_id == pid]
        assert not [a for a in store.agents.values() if a.project_id == pid]

        # A reload must not resurrect it.
        reloaded = Store(cfg.data_dir)
        assert pid not in reloaded.projects
        print("remove/purge ok")


def test_remove_requires_name_confirmation_and_bad_mode_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        app = build_api(cfg, store, orch)
        client = TestClient(app)

        r = client.post(f"/api/projects/{pid}/remove",
                        json={"mode": "purge", "confirm": "wrong name"})
        assert r.status_code == 400, r.text
        assert store.projects[pid].status != "removed"

        r = client.post(f"/api/projects/{pid}/remove",
                        json={"mode": "yolo", "confirm": "Widget"})
        assert r.status_code == 400, r.text
        assert store.projects[pid].status != "removed"

        r = client.post("/api/projects/nope/remove",
                        json={"mode": "hide", "confirm": "x"})
        assert r.status_code == 404
        print("remove guardrails ok")


def test_remove_refuses_while_agents_are_running():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        agent = store.project_agents(pid)[0]
        hold = threading.Event()
        runtime.submit(agent.id, lambda: hold.wait(10))
        deadline = time.time() + 5
        while agent.id not in runtime.inflight and time.time() < deadline:
            time.sleep(0.05)
        assert agent.id in runtime.inflight, "worker never registered as inflight"
        try:
            app = build_api(cfg, store, orch)
            client = TestClient(app)
            r = client.post(f"/api/projects/{pid}/remove",
                            json={"mode": "purge", "confirm": "Widget"})
            assert r.status_code == 409, r.text
            assert store.projects[pid].status != "removed", "must not remove mid-run"
            assert (Path(store.projects[pid].workspace) / "artifact.py").exists()
        finally:
            hold.set()
            deadline = time.time() + 5
            while agent.id in runtime.inflight and time.time() < deadline:
                time.sleep(0.05)
        print("remove refuses running agents ok")


# ---------------------------------------------------------------- costs


def test_costs_surface_on_agents_and_projects():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        assert store.project_cost(pid) == 2.5

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        d = client.get("/api/state").json()
        proj = [p for p in d["projects"] if p["id"] == pid][0]
        assert proj["cost_usd"] == 2.5, proj["cost_usd"]
        assert d["total_cost_usd"] == 2.5, d["total_cost_usd"]

        # Every node in the tree carries a cost, including zero-cost agents.
        def walk(n):
            assert "cost_usd" in n, n["role"]
            [walk(c) for c in n.get("children", [])]
        [walk(n) for n in proj["tree"]]

        # A zero-cost agent still reports a numeric field, so the UI can render it.
        fresh = store.project_agents(pid)[0]
        fresh.cost_usd = 0.0
        store.upsert_agent(fresh)
        d2 = client.get("/api/state").json()
        p2 = [p for p in d2["projects"] if p["id"] == pid][0]
        assert p2["cost_usd"] == 1.25
        print("cost surfacing ok")


def test_removed_projects_do_not_count_toward_total_cost():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        store.remove_project(pid, "hide")
        app = build_api(cfg, store, orch)
        client = TestClient(app)
        d = client.get("/api/state").json()
        assert d["total_cost_usd"] == 0.0, d["total_cost_usd"]
        print("removed projects excluded from total ok")


# ---------------------------------------------------------------- cycle freshness


def test_cycle_metadata_advances_even_when_cycles_fail():
    """The header must not look frozen when every cycle raises."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)

        def boom():
            raise RuntimeError("cycle blew up")

        orch.detect = boom  # type: ignore[method-assign]
        orch._stop = threading.Event()
        cfg.tick_seconds = 1

        t = threading.Thread(target=orch.run_forever, daemon=True)
        t.start()
        time.sleep(2.5)
        orch.stop()
        t.join(timeout=3)

        meta = store.meta
        assert meta.get("cycle_count", 0) >= 2, meta
        assert meta.get("last_cycle_attempt_at"), meta
        assert meta.get("last_tick") is None, "failed cycles must not claim success"
        assert meta.get("cycle_status") == "failed", meta
        print(f"cycle metadata ok (count={meta['cycle_count']}, "
              f"status={meta['cycle_status']})")


def test_cycle_metadata_records_successes_and_duration():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        orch._stop = threading.Event()
        cfg.tick_seconds = 1
        t = threading.Thread(target=orch.run_forever, daemon=True)
        t.start()
        time.sleep(2.2)
        orch.stop()
        t.join(timeout=3)

        meta = store.meta
        assert meta.get("last_tick"), meta
        assert meta.get("last_cycle_completed_at"), meta
        assert meta.get("cycle_status") == "ok", meta
        assert meta.get("cycle_count", 0) >= 2, meta

        app = build_api(cfg, store, orch)
        cyc = client_cycle(app)
        assert cyc["running"] is False
        assert cyc["stale"] is False
        assert cyc["last_completed_at"] == meta["last_cycle_completed_at"]
        print("cycle success metadata ok")


def client_cycle(app) -> dict:
    return TestClient(app).get("/api/state").json()["meta"]["cycle"]


def test_cycle_pill_reports_running_state():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        store.meta["cycle_started_at"] = int(time.time()) - 30
        store.meta["cycle_status"] = "running"
        store.meta["last_cycle_attempt_at"] = int(time.time()) - 30
        app = build_api(cfg, store, orch)
        cyc = client_cycle(app)
        assert cyc["running"] is True, cyc
        assert cyc["started_at"], cyc
        print("running cycle reported ok")


def test_stale_flag_set_when_loop_has_not_completed_lately():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = make(tmp)
        cfg.tick_seconds = 60
        store.meta["cycle_status"] = "failed"
        store.meta["last_cycle_attempt_at"] = int(time.time()) - 600
        app = build_api(cfg, store, orch)
        cyc = client_cycle(app)
        assert cyc["stale"] is True, cyc
        print("stale flag ok")


# ---------------------------------------------------------------- alerts


def test_alerts_have_stable_ids_and_can_be_acknowledged():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task = store.project_tasks(pid)[0]
        task.column = "review"
        store.upsert_task(task)
        store.flush()

        app = build_api(cfg, store, orch)
        client = TestClient(app)

        alerts = client.get("/api/state").json()["alerts"]
        assert alerts, "expected a review alert"
        ids = [a["id"] for a in alerts]
        assert len(set(ids)) == len(ids)

        # Re-fetch: same ids (stable across polls).
        again = client.get("/api/state").json()["alerts"]
        assert [a["id"] for a in again] == ids

        target = alerts[0]["id"]
        r = client.post("/api/alerts/ack", json={"alert_id": target})
        assert r.status_code == 200, r.text
        left = client.get("/api/state").json()["alerts"]
        assert target not in [a["id"] for a in left], left
        print("alert ack ok")


def test_ack_is_per_occurrence_not_per_alert_type():
    """Acknowledging a review must not silence the next review round."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task = store.project_tasks(pid)[0]
        task.column = "review"
        task.review_rounds = 0
        store.upsert_task(task)
        store.flush()

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        first = client.get("/api/state").json()["alerts"][0]

        client.post("/api/alerts/ack", json={"alert_id": first["id"]})
        assert client.get("/api/state").json()["alerts"] == []

        # A new review round is a new occurrence: it must resurface.
        task = store.tasks[task.id]
        task.review_rounds = 1
        store.upsert_task(task)
        store.flush()
        after = client.get("/api/state").json()["alerts"]
        assert after, "new review round should alert again"
        assert after[0]["id"] != first["id"]
        print("per-occurrence ack ok")


def test_ack_all_clears_every_alert_and_can_be_reset():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task = store.project_tasks(pid)[0]
        task.column = "blocked"
        task.blocked_reason = "needs input"
        store.upsert_task(task)
        store.flush()

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        assert client.get("/api/state").json()["alerts"]

        r = client.post("/api/alerts/ack", json={"all": True})
        assert r.status_code == 200, r.text
        assert client.get("/api/state").json()["alerts"] == []

        r = client.delete("/api/alerts/ack")
        assert r.status_code == 200, r.text
        assert client.get("/api/state").json()["alerts"], "reset should bring them back"
        print("ack all / reset ok")


def test_delivery_alert_does_not_resurface_on_unrelated_touches():
    """Acking 'ready to deliver' must not come back because the project was touched."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        for t in store.project_tasks(pid):
            t.column = "done"
            store.upsert_task(t)
        orch.reconcile()
        assert store.projects[pid].status == "ready_for_delivery"
        assert store.projects[pid].ready_at, "reconcile must stamp ready_at"

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        alerts = client.get("/api/state").json()["alerts"]
        delivery = [a for a in alerts if "ready to deliver" in a["message"]]
        assert delivery, alerts

        client.post("/api/alerts/ack", json={"alert_id": delivery[0]["id"]})
        assert client.get("/api/state").json()["alerts"] == []

        # An unrelated touch must not re-fire the acknowledged alert.
        proj = store.projects[pid]
        proj.touch()
        store.flush()
        assert client.get("/api/state").json()["alerts"] == [], "alert re-fired"

        # But becoming ready a second time (after rework) is a new occurrence.
        proj.status = "active"
        store.flush()
        orch.reconcile()
        assert client.get("/api/state").json()["alerts"], "second delivery should alert"
        print("delivery alert stability ok")


def test_tree_keeps_retired_subagents_under_their_parent():
    """Retiring prunes the parent's child list; the tree must still show the agent."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        orch_agent = [a for a in store.agents.values() if a.role == "orchestrator"][0]
        worker = [a for a in store.project_agents(pid) if a.role == "backend-dev"][0]
        reviewer = [a for a in store.project_agents(pid) if a.role == "reviewer"][0]
        reviewer.parent_agent_id = worker.id
        store.upsert_agent(reviewer)

        # Simulate what retire_agent does to the parent's child list.
        orch_agent.child_agent_ids = []
        worker.child_agent_ids = []
        store.upsert_agent(orch_agent)
        store.upsert_agent(worker)

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        proj = [p for p in client.get("/api/state").json()["projects"]
                if p["id"] == pid][0]
        (root,) = proj["tree"]
        assert root["role"] == "orchestrator"
        roles = {c["role"] for c in root["children"]}
        assert roles == {"backend-dev"}, roles
        (worker_node,) = [c for c in root["children"] if c["role"] == "backend-dev"]
        assert [c["role"] for c in worker_node["children"]] == ["reviewer"]
        print("tree keeps retired subagents ok")


def test_removing_a_project_never_touches_the_shared_orchestrator():
    """The orchestrator is global; its project_id is just the last one it touched."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        orch.spec.project_id = pid          # what a real cycle does
        store.upsert_agent(orch.spec)
        orch.spec.cost_usd = 1.25
        orch.spec.turns_used = 7
        store.upsert_agent(orch.spec)

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        r = client.post(f"/api/projects/{pid}/remove",
                        json={"mode": "purge", "confirm": "Widget"})
        assert r.status_code == 200, r.text

        assert orch.spec.id in store.agents, "purge deleted the orchestrator"
        kept = store.agents[orch.spec.id]
        assert kept.turns_used == 7 and kept.cost_usd == 1.25, kept
        assert kept.conversation_id == orch.spec.conversation_id

        # A restart must reuse the same orchestrator, not mint a new one.
        reloaded = Store(cfg.data_dir)
        runtime2 = AgentRuntime(cfg, reloaded)
        orch2 = Orchestrator(cfg, reloaded, runtime2)
        assert orch2.spec.id == orch.spec.id, "restart created a second orchestrator"

        # And an unhealthy orchestrator must still alert even with no projects.
        orch.spec.status = "llm_unavailable"
        reloaded.agents[orch.spec.id].status = "llm_unavailable"
        app2 = build_api(cfg, reloaded, orch2)
        alerts = TestClient(app2).get("/api/state").json()["alerts"]
        assert any("llm_unavailable" in a["message"] for a in alerts), alerts
        print("shared orchestrator survives removal ok")


def test_removed_projects_raise_no_alerts():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        task = store.project_tasks(pid)[0]
        task.column = "blocked"
        store.upsert_task(task)
        agent = store.project_agents(pid)[0]
        agent.status = "budget_exhausted"
        store.upsert_agent(agent)
        store.flush()

        app = build_api(cfg, store, orch)
        client = TestClient(app)
        assert client.get("/api/state").json()["alerts"]

        store.remove_project(pid, "hide")
        assert client.get("/api/state").json()["alerts"] == []
        print("removed projects silent ok")


def main() -> None:
    test_remove_hide_keeps_files_and_stops_work()
    test_remove_archives_workspace_and_keeps_record()
    test_remove_purge_deletes_everything()
    test_remove_requires_name_confirmation_and_bad_mode_rejected()
    test_remove_refuses_while_agents_are_running()
    test_costs_surface_on_agents_and_projects()
    test_removed_projects_do_not_count_toward_total_cost()
    test_cycle_metadata_advances_even_when_cycles_fail()
    test_cycle_metadata_records_successes_and_duration()
    test_cycle_pill_reports_running_state()
    test_stale_flag_set_when_loop_has_not_completed_lately()
    test_alerts_have_stable_ids_and_can_be_acknowledged()
    test_ack_is_per_occurrence_not_per_alert_type()
    test_ack_all_clears_every_alert_and_can_be_reset()
    test_delivery_alert_does_not_resurface_on_unrelated_touches()
    test_tree_keeps_retired_subagents_under_their_parent()
    test_removing_a_project_never_touches_the_shared_orchestrator()
    test_removed_projects_raise_no_alerts()
    print("\nALL DASHBOARD FEATURE TESTS PASSED")


if __name__ == "__main__":
    main()
