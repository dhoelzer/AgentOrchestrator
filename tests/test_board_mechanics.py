"""Board mechanics without an LLM: project -> task -> sub-agent -> review -> delivery."""

from __future__ import annotations

import shutil
import sys

sys.path.insert(0, ".")

from harness.config import HarnessConfig
from harness.orchestrator import Orchestrator
from harness.runtime import AgentRuntime
from harness.store import Store
from harness.tools import (
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
    ListProjectsAction,
    ListProjectsExecutor,
    RecordReviewAction,
    RecordReviewExecutor,
    RequestReviewAction,
    RequestReviewExecutor,
    SignalDeliveryAction,
    SignalDeliveryExecutor,
    UpdateTaskAction,
    UpdateTaskExecutor,
    register_harness_tools,
)


class FakeConv:
    """Stand-in for LocalConversation: executors only read .id."""

    def __init__(self, cid: str):
        self.id = cid


def main() -> int:
    shutil.rmtree("/tmp/h2", ignore_errors=True)
    cfg = HarnessConfig.load()
    cfg.data_dir = "/tmp/h2"
    cfg.projects_dir = "/tmp/h2/projects"
    cfg.ensure_dirs()

    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    register_harness_tools()
    orch = Orchestrator(cfg, store, runtime)

    orch_conv = FakeConv("orch-conv")

    def call(executor, action, conv=orch_conv):
        obs = executor(action, conv)
        return obs

    # 1. project
    obs = call(CreateProjectExecutor(), CreateProjectAction(
        name="Widget API", goal="A FastAPI widget service with tests."))
    assert not obs.is_error, obs.text
    project = next(iter(store.projects.values()))
    print("project:", project.id, project.workspace)

    # 2. tasks
    call(CreateTaskExecutor(), CreateTaskAction(
        project_id=project.id, title="Create app skeleton",
        objective="app.py exposes /health", expertise="python-engineer",
        acceptance_criteria=["GET /health returns 200"], deliverables=["app.py"]))
    call(CreateTaskExecutor(), CreateTaskAction(
        project_id=project.id, title="Add tests", objective="pytest passes",
        expertise="test-engineer"))
    assert len(store.tasks) == 2
    first, second = list(store.tasks.values())
    print("tasks:", first.id, second.id)

    # 3. dependency auto-advance: done dependency -> second becomes ready
    call(UpdateTaskExecutor(), UpdateTaskAction(task_id=first.id, column="done"))
    attention, _ = orch.housekeeping()
    assert second.column == "ready", f"expected ready, got {second.column}"
    print("auto-advance ok:", second.id, "->", second.column)
    assert any("READY and unstaffed" in a for a in attention[project.id])

    # 4. sub-agent creation (depth + parent linkage + default budget)
    obs = call(CreateSubAgentExecutor(), CreateSubAgentAction(
        role="python-engineer", specialty="FastAPI", mission="Build the skeleton.",
        project_id=project.id))
    assert not obs.is_error, obs.text
    worker = next(a for a in store.agents.values() if a.role == "python-engineer")
    assert worker.depth == 1
    assert worker.parent_agent_id == orch.spec.id
    assert worker.turn_limit == 100
    print("sub-agent:", worker.id, "depth", worker.depth, "budget", worker.turn_limit)

    # nested sub-agent created by the worker itself
    runtime.conversation_index["worker-conv"] = worker.id
    obs = call(CreateSubAgentExecutor(), CreateSubAgentAction(
        role="devops-engineer", specialty="docker", mission="Write a Dockerfile.",
        project_id=project.id), FakeConv("worker-conv"))
    assert not obs.is_error, obs.text
    child = next(a for a in store.agents.values() if a.role == "devops-engineer")
    assert child.parent_agent_id == worker.id and child.depth == 2
    print("nested sub-agent:", child.id, "depth", child.depth, "parent", child.parent_agent_id)

    # depth limit
    runtime.conversation_index["child-conv"] = child.id
    obs = call(CreateSubAgentExecutor(), CreateSubAgentAction(
        role="deep", specialty="x", mission="y", project_id=project.id),
        FakeConv("child-conv"))
    deep = next(a for a in store.agents.values() if a.role == "deep")
    assert deep.depth == 3
    runtime.conversation_index["deep-conv"] = deep.id
    obs = call(CreateSubAgentExecutor(), CreateSubAgentAction(
        role="toodeep", specialty="x", mission="y", project_id=project.id),
        FakeConv("deep-conv"))
    assert obs.is_error and "depth limit" in obs.text, obs.text
    print("depth limit enforced:", obs.text[:70])
    for key in ("worker-conv", "child-conv", "deep-conv"):
        runtime.conversation_index.pop(key, None)

    # 5. assign
    obs = call(AssignTaskExecutor(), AssignTaskAction(task_id=second.id, agent_id=worker.id))
    assert not obs.is_error, obs.text
    assert store.tasks[second.id].assigned_agent_id == worker.id
    assert store.agents[worker.id].task_id == second.id
    print("assign ok")

    # 6. review flow
    obs = call(RequestReviewExecutor(), RequestReviewAction(task_id=second.id))
    assert store.tasks[second.id].column == "review", store.tasks[second.id].column
    runtime.conversation_index["rev-conv"] = child.id
    obs = call(RecordReviewExecutor(), RecordReviewAction(
        task_id=second.id, verdict="changes_requested", feedback="Missing error handling."),
        FakeConv("rev-conv"))
    assert not obs.is_error, obs.text
    assert store.tasks[second.id].column == "rework"
    assert store.tasks[second.id].reviewer_agent_id == child.id
    print("review round 1 -> rework")

    attention, _ = orch.housekeeping()
    assert any("REWORK" in a for a in attention[project.id])

    obs = call(RecordReviewExecutor(), RecordReviewAction(
        task_id=second.id, verdict="approved", feedback="All criteria met."),
        FakeConv("rev-conv"))
    assert store.tasks[second.id].column == "done"
    assert store.tasks[second.id].review_rounds == 2
    print("review round 2 -> done")
    runtime.conversation_index.pop("rev-conv", None)

    # bad verdict rejected
    obs = call(RecordReviewExecutor(), RecordReviewAction(
        task_id=second.id, verdict="maybe", feedback="x"))
    assert obs.is_error
    print("invalid verdict rejected")

    # 7. delivery gating: an open task must block delivery
    third = call(CreateTaskExecutor(), CreateTaskAction(
        project_id=project.id, title="Write README", objective="README.md exists",
        expertise="docs-writer"))
    third_id = [t.id for t in store.tasks.values() if t.title == "Write README"][0]
    obs = call(SignalDeliveryExecutor(), SignalDeliveryAction(
        project_id=project.id, summary="done"))
    assert obs.is_error, "delivery should be blocked while a task is open"
    print("delivery correctly blocked:", obs.text[:70])

    call(UpdateTaskExecutor(), UpdateTaskAction(task_id=third_id, column="done"))
    orch.reconcile()
    attention = orch.detect()
    assert store.projects[project.id].status == "ready_for_delivery"
    assert any("signal_delivery" in a for a in attention[project.id])
    obs = call(SignalDeliveryExecutor(), SignalDeliveryAction(
        project_id=project.id, summary="Widget API shipped."))
    assert not obs.is_error, obs.text
    assert store.projects[project.id].delivery_signaled
    print("delivery signaled")

    # 8. budget extension after exhaustion
    worker = store.agents[worker.id]
    worker.turns_used = 100
    worker.status = "budget_exhausted"
    store.flush()
    obs = call(ExtendAgentExecutor(), ExtendAgentAction(
        agent_id=worker.id, extra_turns=100, decision="Real progress, ran out of time."))
    assert not obs.is_error, obs.text
    assert store.agents[worker.id].turn_limit == 200
    assert store.agents[worker.id].status == "idle"
    print("budget extended ->", store.agents[worker.id].turn_limit)

    # 9. persistence round-trip
    reloaded = Store(cfg.data_dir)
    assert len(reloaded.projects) == 1 and len(reloaded.tasks) == 3
    assert len(reloaded.agents) == len(store.agents)
    print("persistence ok:", len(reloaded.agents), "agents,",
          len(reloaded.recent_events), "events")

    # 10. read_file sandbox
    runtime.conversation_index["w-conv"] = worker.id
    from harness.tools import ReadFileAction, ReadFileExecutor
    obs = call(ReadFileExecutor(), ReadFileAction(path="/etc/passwd"), FakeConv("w-conv"))
    assert obs.is_error and "outside your workspace" in obs.text
    print("read_file sandbox enforced")
    runtime.conversation_index.pop("w-conv", None)

    obs = call(ListProjectsExecutor(), ListProjectsAction())
    assert not obs.is_error
    runtime.close_all()
    print("\nALL MECHANICS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
