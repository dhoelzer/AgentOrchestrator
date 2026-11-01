"""Blocked-task re-checks, per-card cost, and harness version - no LLM needed."""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

sys.path.insert(0, ".")

from agent_harness import version as harness_version
from agent_harness.config import HarnessConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime, RunOutcome
from agent_harness.store import Store
from agent_harness.tools import (
    AssignTaskAction,
    AssignTaskExecutor,
    CreateProjectAction,
    CreateProjectExecutor,
    CreateSubAgentAction,
    CreateSubAgentExecutor,
    CreateTaskAction,
    CreateTaskExecutor,
    ExtendAgentAction,
    ExtendAgentExecutor,
    ReportToParentAction,
    ReportToParentExecutor,
    RunAgentAction,
    RunAgentExecutor,
    UpdateTaskAction,
    UpdateTaskExecutor,
    register_harness_tools,
)


class FakeConv:
    def __init__(self, cid: str):
        self.id = cid


def _outcome(agent, status="idle", error=None, budget=False):
    return RunOutcome(
        turns_used=agent.turns_used, turns_this_run=3, status=status,
        final_report="worked", budget_exhausted=budget, tokens=100,
        cost_usd=0.25, error=error)


def main() -> int:
    shutil.rmtree("/tmp/h5", ignore_errors=True)
    cfg = HarnessConfig.load()
    cfg.data_dir = "/tmp/h5"
    cfg.projects_dir = "/tmp/h5/projects"
    cfg.ensure_dirs()

    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    register_harness_tools()
    orch = Orchestrator(cfg, store, runtime)
    orch_conv = FakeConv("orch-conv")

    def call(executor, action, conv=orch_conv):
        return executor(action, conv)

    call(CreateProjectExecutor(), CreateProjectAction(name="Blocks", goal="g"))
    project = next(iter(store.projects.values()))
    call(CreateTaskExecutor(), CreateTaskAction(
        project_id=project.id, title="Blocked card", objective="o",
        expertise="backend", acceptance_criteria=["ok"], deliverables=["x"]))
    task = next(iter(store.tasks.values()))
    call(CreateSubAgentExecutor(), CreateSubAgentAction(
        role="backend-dev", specialty="backend", mission="build",
        project_id=project.id))
    worker = next(a for a in store.project_agents(project.id) if a.depth == 1)
    call(AssignTaskExecutor(), AssignTaskAction(task_id=task.id, agent_id=worker.id))

    # ---- 1. a clean re-dispatch must clear a stale block ----
    task.column = "blocked"
    task.blocked_reason = "agent ran out of turns"
    task.blocked_cause = "budget"
    store.upsert_task(task)

    runtime.run = lambda spec, tools, msg, turn_limit=None, **k: _outcome(spec)
    obs = call(RunAgentExecutor(), RunAgentAction(
        agent_id=worker.id, task_id=task.id, background=False))
    assert not obs.is_error, obs.text
    task = store.tasks[task.id]
    assert task.column == "review", task.column
    assert task.blocked_reason is None, task.blocked_reason
    assert task.blocked_cause is None, task.blocked_cause
    print("clean re-dispatch cleared block ->", task.column)

    # ---- 2. the worker can restate a NEW reason for blocking ----
    runtime.run = lambda spec, tools, msg, turn_limit=None, **k: _outcome(spec)
    runtime.conversation_index["w-conv"] = worker.id
    obs = call(ReportToParentExecutor(), ReportToParentAction(
        status="blocked", summary="cannot proceed",
        blocker="need credentials from the user"), FakeConv("w-conv"))
    assert not obs.is_error, obs.text
    task = store.tasks[task.id]
    assert task.column == "blocked"
    assert task.blocked_reason == "need credentials from the user"
    assert task.blocked_cause == "worker"
    print("worker restated block ->", task.blocked_reason)

    # ---- 3. the worker confirms the block is gone ----
    obs = call(ReportToParentExecutor(), ReportToParentAction(
        status="complete", summary="did it", deliverables=["x.py"]), FakeConv("w-conv"))
    assert not obs.is_error, obs.text
    task = store.tasks[task.id]
    assert task.column == "in_progress", task.column
    assert task.blocked_reason is None and task.blocked_cause is None
    print("worker confirmed unblocked ->", task.column)
    runtime.conversation_index.pop("w-conv", None)

    # ---- 4. extend_agent clears a budget block but not a user-input block ----
    task.column = "blocked"
    task.blocked_reason = "agent used all turns"
    task.blocked_cause = "budget"
    store.upsert_task(task)
    obs = call(ExtendAgentExecutor(), ExtendAgentAction(
        agent_id=worker.id, extra_turns=50, decision="real progress"))
    assert not obs.is_error, obs.text
    task = store.tasks[task.id]
    assert task.column == "ready" and task.blocked_reason is None
    print("extend cleared budget block ->", task.column)

    task.column = "blocked"
    task.blocked_reason = "waiting on user credentials"
    task.blocked_cause = "manual"
    store.upsert_task(task)
    call(ExtendAgentExecutor(), ExtendAgentAction(
        agent_id=worker.id, extra_turns=50, decision="more turns"))
    task = store.tasks[task.id]
    assert task.column == "blocked", "user-input block must survive extend"
    assert task.blocked_reason == "waiting on user credentials"
    print("manual block survived extend ->", task.blocked_reason)

    # ---- 5. reconcile re-queues a card whose upstream deps just landed ----
    call(CreateTaskExecutor(), CreateTaskAction(
        project_id=project.id, title="Upstream", objective="u", expertise="backend"))
    upstream = next(t for t in store.project_tasks(project.id)
                    if t.title == "Upstream")
    task.column = "blocked"
    task.blocked_reason = "waiting on upstream interface"
    task.blocked_cause = "dependency"
    task.depends_on = [upstream.id]
    store.upsert_task(task)
    upstream.column = "done"
    store.upsert_task(upstream)
    orch.reconcile()
    task = store.tasks[task.id]
    assert task.column == "ready", task.column
    assert task.blocked_reason is None
    print("deps resolved by another agent ->", task.column)

    # a budget block must NOT be auto-cleared just because deps are done
    task.column = "blocked"
    task.blocked_reason = "agent used all turns"
    task.blocked_cause = "budget"
    store.upsert_task(task)
    orch.reconcile()
    assert store.tasks[task.id].column == "blocked"
    print("budget block not auto-cleared by deps")

    # ---- 6. failed runs block the card with a cause ----
    runtime.run = lambda spec, tools, msg, turn_limit=None, **k: _outcome(
        spec, status="failed", error="boom")
    task.column = "in_progress"
    task.blocked_reason = None
    store.upsert_task(task)
    obs = call(RunAgentExecutor(), RunAgentAction(
        agent_id=worker.id, task_id=task.id, background=False))
    assert obs.is_error, "a failed run must surface as an error to the orchestrator"
    task = store.tasks[task.id]
    assert task.column == "blocked" and task.blocked_cause == "agent_failed"
    assert "boom" in task.blocked_reason
    print("failed run ->", task.column, task.blocked_cause)

    # ---- 7. per-card cost comes from the assigned agent ----
    from agent_harness.web.api import _task_json
    worker.cost_usd = 5.9197
    store.upsert_agent(worker)
    card = _task_json(store.tasks[task.id], agents=store.agents)
    assert card["cost_usd"] == 5.9197, card["cost_usd"]
    assert card["agent_role"] == "backend-dev"
    unassigned = next(t for t in store.project_tasks(project.id)
                      if t.title == "Upstream")
    assert _task_json(unassigned, agents=store.agents)["cost_usd"] is None
    print("card cost ->", card["cost_usd"])

    # ---- 8. user direction clears the block it was answering ----
    from agent_harness.web.api import build_api
    from fastapi.testclient import TestClient
    client = TestClient(build_api(cfg, store, orch))

    task.column = "blocked"
    task.blocked_reason = "waiting on user credentials"
    task.blocked_cause = "manual"
    store.upsert_task(task)
    resp = client.post(f"/api/projects/{project.id}/tasks/{task.id}/directives",
                       json={"text": "use the vault credentials"})
    assert resp.status_code == 200, resp.text
    task = store.tasks[task.id]
    assert task.column == "ready", task.column
    assert task.blocked_reason is None and task.blocked_cause is None
    assert task.user_directives[-1]["text"] == "use the vault credentials"
    print("user direction cleared block ->", task.column)

    # ---- 9. harness version is exposed and stamped into state ----
    assert harness_version.__version__.count(".") == 2
    assert harness_version.version_string().startswith(harness_version.__version__)
    health = client.get("/health").json()
    assert health["harness_version"].startswith(harness_version.__version__)
    state = client.get("/api/state").json()
    assert state["meta"]["harness_version"].startswith(harness_version.__version__)
    assert client.get("/docs").status_code == 200
    print("harness version ->", health["harness_version"])

    # ---- 10. cost survives a fresh conversation (reset_conversation) ----
    agent = store.agents[worker.id]
    agent.cost_usd = 3.0
    agent.tokens = 1000
    agent.prior_cost_usd = 0.0
    agent.prior_tokens = 0
    store.upsert_agent(agent)
    runtime.reset_conversation(agent)
    agent = store.agents[worker.id]
    assert agent.prior_cost_usd == 3.0, agent.prior_cost_usd
    assert agent.prior_tokens == 1000
    assert agent.cost_usd == 3.0, f"cost wiped by reset: {agent.cost_usd}"
    print("reset kept lifetime spend ->", agent.cost_usd,
          "(baseline", agent.prior_cost_usd, ")")

    # A second reset must not double-count: cost_usd already contains prior.
    runtime.reset_conversation(agent)
    agent = store.agents[worker.id]
    assert agent.prior_cost_usd == 3.0, agent.prior_cost_usd
    assert agent.cost_usd == 3.0, agent.cost_usd
    print("second reset did not double-count ->", agent.cost_usd)

    # The store's cost rollup must still see the rolled-over spend, so the
    # dashboard does not drop to $0 between the reset and the next run.
    assert store.project_cost(project.id) >= 3.0, store.project_cost(project.id)
    print("project cost kept across reset ->", store.project_cost(project.id))

    # ---- 11. a state file written by an older build is detectable ----
    # The stack bind-mounts its data volume, so an un-rebuilt container can serve
    # an old build against new data. The mismatch is visible on the first load of
    # such a file; after this build writes it once, it no longer is.
    old_dir = "/tmp/h5_versionstamp"
    shutil.rmtree(old_dir, ignore_errors=True)
    pathlib.Path(old_dir).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(old_dir) / "state.json").write_text(json.dumps({
        "projects": [], "tasks": [], "agents": [],
        "meta": {"harness_version": "0.9.0"},
    }))
    reloaded = Store(old_dir)
    assert reloaded.meta["harness_version"] == "0.9.0"
    assert reloaded.meta["harness_version_stored"] == "0.9.0"
    assert reloaded.meta["harness_version_stored"] != harness_version.version_string()
    reloaded.meta["harness_version"] = harness_version.version_string()
    reloaded.flush()
    # Re-loading a file this build already wrote must NOT report a stale mismatch:
    # the stored value is re-derived from the file, not carried over from meta.
    again = Store(old_dir)
    assert again.meta["harness_version"] == harness_version.version_string()
    assert again.meta["harness_version_stored"] == harness_version.version_string()
    print("stored build -> 0.9.0 on first load, cleared after upgrade")

    print("\nALL BLOCK/COST/VERSION TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
