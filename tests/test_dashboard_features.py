"""Project removal modes, cost surfacing, cycle freshness, alert acks."""

from __future__ import annotations

import json
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
from agent_harness.store import Store, REVISION_LOG_LIMIT
from agent_harness.tools import (
    CreateProjectAction,
    CreateProjectExecutor,
    CreateSubAgentAction,
    CreateSubAgentExecutor,
    CreateTaskAction,
    CreateTaskExecutor,
    RecordReviewAction,
    RecordReviewExecutor,
    SignalDeliveryAction,
    SignalDeliveryExecutor,
    UpdateTaskAction,
    UpdateTaskExecutor,
    register_harness_tools,
)
from agent_harness.store import AgentSpec
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
        # Real layout: conversations/<project_id>/<agent_id>/
        conv_dir = cfg.root_dir / "conversations" / pid
        agent = [a for a in store.agents.values() if a.role == "backend-dev"][0]
        (conv_dir / agent.id).mkdir(parents=True, exist_ok=True)
        (conv_dir / agent.id / "log.json").write_text("{}")
        agent.persistence_dir = str(conv_dir / agent.id)
        store.upsert_agent(agent)

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
        assert list(archive.glob(f"{agent.id}-logs")), "agent logs not archived"
        assert not (conv_dir / agent.id).exists()

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


def _staffed(tmp: str, name: str = "Widget", limit: int = 2):
    """Project with two sub-agents, a fake transcript each, and a low child cap."""
    cfg, store, runtime, orch = seed(tmp, name)
    cfg.max_children_per_agent = limit
    pid = list(store.projects)[0]
    conv = FakeConv()
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role="backend-dev", specialty="backend", mission="build", project_id=pid), conv)
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role="reviewer", specialty="review", mission="review", project_id=pid), conv)
    oid = [a.id for a in store.agents.values() if a.role == "orchestrator"][0]
    for aid in [a.id for a in store.agents.values()]:
        d = Path(cfg.root_dir) / "conversations" / pid / aid
        d.mkdir(parents=True, exist_ok=True)
        (d / "log.json").write_text("{}")
        store.agents[aid].persistence_dir = str(d)
        store.upsert_agent(store.agents[aid])
    return cfg, store, runtime, orch, pid, oid


def test_removal_frees_delegation_slots_in_every_mode():
    """Removed agents must stop counting against max_children_per_agent."""
    for mode in ("hide", "archive", "purge"):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, store, runtime, orch, pid, oid = _staffed(tmp)
            conv = FakeConv()
            client = TestClient(build_api(cfg, store, orch))
            r = client.post(f"/api/projects/{pid}/remove",
                            json={"mode": mode, "confirm": "Widget"})
            assert r.status_code == 200, r.text
            assert r.json()["agents_removed"] == 2, r.json()

            assert store.agents[oid].child_agent_ids == [], (mode, "ids leaked")
            left = [a.role for a in store.agents.values()]
            assert left == ["orchestrator"], (mode, left)

            # The real symptom: a new project must be staffable afterwards.
            CreateProjectExecutor()(CreateProjectAction(
                name="Next", goal="g", constraints=""), conv)
            new_pid = [p for p in store.visible_projects() if p.name == "Next"][0].id
            res = CreateSubAgentExecutor()(CreateSubAgentAction(
                role="dev", specialty="s", mission="m", project_id=new_pid), conv)
            text = " ".join(c.text for c in res.content)
            assert "created" in text, (mode, text)
            assert store.orphaned_agents() == {
                "missing_project": [], "missing_parent": [], "dangling_child": []
            }, mode
            print(f"removal frees slots ({mode}) ok")


def test_removal_keeps_cost_history_after_agents_are_deleted():
    """Agents are deleted, so the project total must be snapshotted."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, oid = _staffed(tmp)
        for a in store.agents.values():
            if a.role != "orchestrator":
                a.cost_usd = 1.25
                store.upsert_agent(a)
        assert store.project_cost(pid) == 2.5
        TestClient(build_api(cfg, store, orch)).post(
            f"/api/projects/{pid}/remove", json={"mode": "hide", "confirm": "Widget"})
        assert store.project_cost(pid) == 2.5, store.project_cost(pid)
        print("cost survives agent deletion ok")


def test_removal_never_touches_the_orchestrator_transcript():
    """The orchestrator's transcript lives under the project dir it last touched."""
    for mode, logs_kept in (("hide", True), ("archive", False), ("purge", False)):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, store, runtime, orch, pid, oid = _staffed(tmp)
            orch_dir = Path(cfg.root_dir) / "conversations" / pid / oid
            assert orch_dir.exists()
            TestClient(build_api(cfg, store, orch)).post(
                f"/api/projects/{pid}/remove",
                json={"mode": mode, "confirm": "Widget"})
            assert (orch_dir / "log.json").exists(), (mode, "orchestrator log lost")

            sub_dirs = [d for d in (Path(cfg.root_dir) / "conversations" / pid)
                        .iterdir() if d.name != oid]
            if logs_kept:
                assert len(sub_dirs) == 2, (mode, sub_dirs)
            else:
                assert sub_dirs == [], (mode, sub_dirs)
            print(f"orchestrator transcript preserved ({mode}) ok")


def test_orphan_sweep_cleans_agents_left_by_an_older_build():
    """State written before agent cleanup existed must self-heal on the next cycle."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, oid = _staffed(tmp)
        # Simulate the old behaviour: project removed, agents and ids left behind.
        store.projects[pid].status = "removed"
        store.projects[pid].removal_mode = "hide"
        store.flush()
        store.agents["ghost-proj"] = AgentSpec(
            id="ghost-proj", role="zombie", specialty="s", mission="m",
            system_prompt="p", project_id="no-such-project")
        store.agents[oid].child_agent_ids = ["agent-vanished-1", "agent-vanished-2"]
        store.upsert_agent(store.agents[oid])

        found = store.orphaned_agents()
        assert "ghost-proj" in found["missing_project"], found
        assert oid in found["dangling_child"], found

        lines = orch.sweep_orphans()
        assert lines, "sweep reported nothing"
        assert "ghost-proj" not in store.agents
        assert store.agents[oid].child_agent_ids == []
        assert store.orphaned_agents() == {
            "missing_project": [], "missing_parent": [], "dangling_child": []
        }
        assert orch.sweep_orphans() == [], "sweep not idempotent"
        print("orphan sweep cleans legacy state ok")


def test_orphan_sweep_deletes_agents_detached_from_a_deleted_parent():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, oid = _staffed(tmp)
        worker = [a for a in store.agents.values() if a.role == "backend-dev"][0]
        reviewer = [a for a in store.agents.values() if a.role == "reviewer"][0]
        reviewer.parent_agent_id = "agent-deleted-long-ago"
        store.upsert_agent(reviewer)

        assert reviewer.id in store.orphaned_agents()["missing_parent"]
        orch.sweep_orphans()
        assert reviewer.id not in store.agents, "detached agent survived"
        assert worker.id in store.agents, "healthy agent was deleted"
        print("orphan sweep deletes detached agents ok")


def test_sweep_runs_every_cycle_and_frees_slots_without_llm_help():
    """The sweep is deterministic: it must run even if the orchestrator never acts."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, oid = _staffed(tmp)
        store.projects[pid].status = "removed"
        store.flush()
        assert store.agents[oid].child_agent_ids, "precondition: stale ids present"

        orch.run_cycle()
        assert store.agents[oid].child_agent_ids == []
        assert [a.role for a in store.agents.values()] == ["orchestrator"]
        print("sweep runs from the cycle loop ok")


def test_purging_a_project_never_orchestrates_away_spend():
    """The orchestrator's project_id drifts; purging that project must not
    make its spend disappear from the harness total."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch = seed(tmp)
        pid = list(store.projects)[0]
        oid = [a.id for a in store.agents.values() if a.role == "orchestrator"][0]
        store.agents[oid].cost_usd = 4.0
        # The orchestrator last worked on a project that is about to vanish.
        store.agents[oid].project_id = pid
        store.upsert_agent(store.agents[oid])

        client = TestClient(build_api(cfg, store, orch))
        before = client.get("/api/state").json()
        assert before["total_cost_usd"] == 6.5, before["total_cost_usd"]
        assert before["orchestrator_cost_usd"] == 4.0
        assert before["project_cost_usd"] == 2.5

        client.post(f"/api/projects/{pid}/remove",
                    json={"mode": "purge", "confirm": "Widget"})
        after = client.get("/api/state").json()
        # Purge deletes the project record, so its spend is no longer
        # attributable to a project - but the orchestrator's $4 must survive.
        assert after["orchestrator_cost_usd"] == 4.0, after["orchestrator_cost_usd"]
        assert after["total_cost_usd"] == 4.0, after["total_cost_usd"]
        print("purge does not orphan orchestrator spend ok")


def test_harness_total_reconciles_with_project_plus_orchestrator():
    """total == sum(project costs) + orchestrator overhead, while nothing is orphaned."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, oid = _staffed(tmp)
        for a in store.agents.values():
            a.cost_usd = 1.0
            store.upsert_agent(a)
        d = TestClient(build_api(cfg, store, orch)).get("/api/state").json()
        assert abs(d["total_cost_usd"]
                   - (d["project_cost_usd"] + d["orchestrator_cost_usd"])) < 1e-9, d
        # The project card total must equal the sum of its non-shared tree nodes.
        proj = [p for p in d["projects"] if p["id"] == pid][0]
        tree_cost = 0.0
        def walk(n):
            nonlocal tree_cost
            if not n.get("shared"):
                tree_cost += n["cost_usd"]
            [walk(c) for c in n.get("children", [])]
        [walk(n) for n in proj["tree"]]
        assert abs(proj["cost_usd"] - tree_cost) < 1e-9, (proj["cost_usd"], tree_cost)
        print("cost reconciliation ok")


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


def _one_project(tmp: str, name: str = "Widget", goal: str = "ship a widget"):
    """A store with exactly one project and no tasks (version 0.1)."""
    cfg, store, runtime, orch = make(tmp)
    CreateProjectExecutor()(CreateProjectAction(
        name=name, goal=goal, constraints=""), FakeConv())
    return cfg, store, runtime, orch, list(store.projects)[0]


def test_projects_start_at_version_0_1():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _one_project(tmp, name="Fresh")
        assert store.projects[pid].version == "0.1", store.projects[pid].version
        d = TestClient(build_api(cfg, store, orch)).get("/api/state").json()
        proj = [p for p in d["projects"] if p["id"] == pid][0]
        assert proj["version"] == "0.1", proj["version"]
        # Decomposing the project into tasks is itself the first revision.
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="A", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["a.py"]), FakeConv())
        assert store.projects[pid].version == "0.2", store.projects[pid].version
        print("projects start at 0.1")


def test_material_revisions_bump_the_minor_version():
    """Adding a task and recording a review are revisions; the minor climbs."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _one_project(tmp, name="Versioned", goal="ship it")
        conv = FakeConv()
        assert store.projects[pid].version == "0.1"

        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="A", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["a.py"]), conv)
        assert store.projects[pid].version == "0.2", store.projects[pid].version

        tid = list(store.tasks)[0]
        RecordReviewExecutor()(RecordReviewAction(
            task_id=tid, verdict="changes_requested", feedback="fix it"), conv)
        assert store.projects[pid].version == "0.3", store.projects[pid].version
        RecordReviewExecutor()(RecordReviewAction(
            task_id=tid, verdict="approved", feedback="good"), conv)
        assert store.projects[pid].version == "0.4", store.projects[pid].version

        # Each version records why it changed.
        reasons = [r["reason"] for r in store.projects[pid].revision_log]
        assert any("plan:" in r for r in reasons), reasons
        assert any("review:" in r for r in reasons), reasons
        print("minor version climbs on material revisions")


def test_routine_board_moves_are_not_revisions():
    """Moving a card between columns is not a project revision."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _one_project(tmp, name="Moves", goal="ship it")
        conv = FakeConv()
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="A", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["a.py"]), conv)
        tid = list(store.tasks)[0]
        before = store.projects[pid].version

        for col in ("ready", "in_progress", "review", "rework", "blocked"):
            args = dict(task_id=tid, column=col)
            if col == "blocked":
                args["blocked_reason"] = "waiting on a dependency"
            UpdateTaskExecutor()(UpdateTaskAction(**args), conv)
        assert store.projects[pid].version == before, (before, store.projects[pid].version)

        # Redefining the task's acceptance criteria IS a revision.
        UpdateTaskExecutor()(UpdateTaskAction(
            task_id=tid, acceptance_criteria=["works", "is fast"]), conv)
        assert store.projects[pid].version != before, store.projects[pid].version
        print("board moves do not bump the version; scope edits do")


def test_delivery_is_a_major_release():
    """Delivery bumps the major and resets the minor."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _one_project(tmp, name="Releaser", goal="ship it")
        conv = FakeConv()
        CreateTaskExecutor()(CreateTaskAction(
            project_id=pid, title="A", objective="x", expertise="backend",
            acceptance_criteria=["works"], deliverables=["a.py"]), conv)
        tid = list(store.tasks)[0]
        RecordReviewExecutor()(RecordReviewAction(
            task_id=tid, verdict="approved", feedback="good"), conv)
        minor_before = store.projects[pid].version_minor
        assert minor_before >= 2

        SignalDeliveryExecutor()(SignalDeliveryAction(
            project_id=pid, summary="all done"), conv)
        api = TestClient(build_api(cfg, store, orch))
        out = api.post(f"/api/projects/{pid}/mark-delivered").json()
        proj = store.projects[pid]
        assert proj.version_major == 1, proj.version
        assert proj.version_minor == 0, proj.version
        assert out["version"] == "1.0", out

        # Post-release work keeps climbing from the reset minor.
        api.post(f"/api/projects/{pid}/revise", json={"goal": "a second version"})
        assert store.projects[pid].version == "1.1", store.projects[pid].version
        print("delivery is a major release")


def test_editing_scope_bumps_the_version_and_is_reported():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _one_project(tmp)
        api = TestClient(build_api(cfg, store, orch))

        # No change -> no revision.
        out = api.post(f"/api/projects/{pid}/revise",
                       json={"goal": "ship a widget"}).json()
        assert out["changed"] == [], out
        assert store.projects[pid].version == "0.1", store.projects[pid].version

        out = api.post(f"/api/projects/{pid}/revise",
                       json={"goal": "ship a bigger widget"}).json()
        assert out["changed"] == ["goal"], out
        assert out["previous_version"] == "0.1", out
        assert out["version"] == "0.2", out
        assert store.projects[pid].goal == "ship a bigger widget"

        # Constraints can be set and cleared; a blank goal or name cannot.
        assert api.post(f"/api/projects/{pid}/revise",
                        json={"constraints": "under 20 lines"}).json()["changed"] == ["constraints"]
        assert store.projects[pid].constraints == "under 20 lines"
        assert api.post(f"/api/projects/{pid}/revise",
                        json={"constraints": ""}).json()["changed"] == ["constraints"]
        assert store.projects[pid].constraints == ""
        assert api.post(f"/api/projects/{pid}/revise",
                        json={"goal": "  "}).json()["changed"] == [], "blank goal ignored"
        assert api.post(f"/api/projects/{pid}/revise",
                        json={"name": ""}).json()["changed"] == [], "blank name ignored"
        assert store.projects[pid].goal == "ship a bigger widget"
        print("scope edits bump the version")


def test_version_survives_a_restart_and_old_state_gets_a_default():
    """Pre-versioning state files load with 0.1 rather than crashing."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _one_project(tmp)
        store.revise_project(pid, "work landed")
        store.revise_project(pid, "review closed")
        assert store.projects[pid].version == "0.3"
        store.flush()

        # Strip version fields from the on-disk state to simulate an upgrade
        # from a build that had no versioning at all.
        state_path = Path(cfg.data_dir) / "state.json"
        data = json.loads(state_path.read_text())
        for proj in data["projects"]:
            for key in ("version_major", "version_minor", "revision_log"):
                proj.pop(key, None)
        state_path.write_text(json.dumps(data))

        reloaded = Store(cfg.data_dir)
        assert reloaded.projects[pid].version == "0.1", reloaded.projects[pid].version
        assert reloaded.projects[pid].revision_log == []

        # And a normal restart keeps the real version.
        store.flush()
        again = Store(cfg.data_dir)
        assert again.projects[pid].version == "0.3", again.projects[pid].version
        assert len(again.projects[pid].revision_log) == 2
        print("version persists; legacy state defaults to 0.1")


def test_revision_log_is_capped():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _one_project(tmp)
        for i in range(REVISION_LOG_LIMIT + 25):
            store.revise_project(pid, f"revision {i}")
        proj = store.projects[pid]
        assert len(proj.revision_log) == REVISION_LOG_LIMIT, len(proj.revision_log)
        # Oldest entries drop off; the newest is retained.
        assert proj.revision_log[-1]["reason"] == f"revision {REVISION_LOG_LIMIT + 24}"
        assert proj.version_minor == REVISION_LOG_LIMIT + 26, proj.version
        print("revision log capped")


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
    test_removal_frees_delegation_slots_in_every_mode()
    test_removal_keeps_cost_history_after_agents_are_deleted()
    test_removal_never_touches_the_orchestrator_transcript()
    test_orphan_sweep_cleans_agents_left_by_an_older_build()
    test_orphan_sweep_deletes_agents_detached_from_a_deleted_parent()
    test_sweep_runs_every_cycle_and_frees_slots_without_llm_help()
    test_purging_a_project_never_orchestrates_away_spend()
    test_harness_total_reconciles_with_project_plus_orchestrator()
    test_removed_projects_raise_no_alerts()
    test_projects_start_at_version_0_1()
    test_material_revisions_bump_the_minor_version()
    test_routine_board_moves_are_not_revisions()
    test_delivery_is_a_major_release()
    test_editing_scope_bumps_the_version_and_is_reported()
    test_version_survives_a_restart_and_old_state_gets_a_default()
    test_revision_log_is_capped()
    print("\nALL DASHBOARD FEATURE TESTS PASSED")


if __name__ == "__main__":
    main()
