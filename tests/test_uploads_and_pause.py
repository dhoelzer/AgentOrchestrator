"""User file uploads and paused-project enforcement.

Run: python3 tests/test_uploads_and_pause.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from agent_harness import prompts
from agent_harness.config import HarnessConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime, RunOutcome
from agent_harness.store import AgentSpec, Store, TaskSpec
from agent_harness.tools import (
    AssignTaskAction,
    AssignTaskExecutor,
    CreateProjectAction,
    CreateProjectExecutor,
    CreateSubAgentAction,
    CreateSubAgentExecutor,
    CreateTaskAction,
    CreateTaskExecutor,
    ReadFileAction,
    ReadFileExecutor,
    RequestReviewAction,
    RequestReviewExecutor,
    ResumeAgentAction,
    ResumeAgentExecutor,
    RunAgentAction,
    RunAgentExecutor,
    SignalDeliveryAction,
    SignalDeliveryExecutor,
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
    return cfg, store, runtime, orch, build_api(cfg, store, orch)


def outcome(status="idle", turns=1) -> RunOutcome:
    return RunOutcome(turns_used=turns, turns_this_run=turns, status=status,
                      final_report="done", budget_exhausted=False,
                      tokens=10, cost_usd=0.01)


def multi(*specs: tuple[str, bytes]) -> list[tuple[str, tuple[str, bytes]]]:
    return [("files", (name, data)) for name, data in specs]


# --------------------------------------------------------------- uploads


def test_upload_with_project_create_lands_in_workspace():
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    client = TestClient(api)
    r = client.post(
        "/api/projects",
        data={"name": "Spec Driven", "goal": "implement the spec",
              "constraints": "python only"},
        files=multi(("spec.md", b"# Spec\nbuild exactly this"),
                    ("data.csv", b"a,b\n1,2")),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    pid = body["id"]
    inputs = Path(body["workspace"]) / "inputs"
    assert sorted(p.name for p in inputs.iterdir()) == ["data.csv", "spec.md"]
    assert (inputs / "spec.md").read_bytes() == b"# Spec\nbuild exactly this"
    assert body["input_files"] == ["spec.md", "data.csv"]

    project = store.projects[pid]
    # The goal names the material, so the very first cycle sees it exists.
    assert "inputs/spec.md" in project.goal and "inputs/data.csv" in project.goal
    # ...and the state payload exposes it to the UI.
    state = client.get("/api/state").json()
    listed = next(p for p in state["projects"] if p["id"] == pid)
    assert [f["name"] for f in listed["input_files"]] == ["spec.md", "data.csv"]
    runtime.close_all()
    print("upload on create ok")


def test_upload_on_existing_project_via_message_and_endpoint():
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    client = TestClient(api)
    pid = client.post("/api/projects",
                      json={"name": "Plain", "goal": "g"}).json()["id"]

    r = client.post(f"/api/projects/{pid}/messages",
                    data={"text": "use the attached log"},
                    files=multi(("run.log", b"stack trace here")))
    assert r.status_code == 200, r.text
    assert r.json()["input_files"] == ["run.log"]
    msg = store.projects[pid].user_messages[-1]
    assert "inputs/run.log" in msg["text"] and msg["files"] == ["run.log"]

    r = client.post(f"/api/projects/{pid}/uploads", files=multi(("cfg.yaml", b"x: 1")))
    assert r.status_code == 200, r.text
    assert r.json()["input_files"] == ["run.log", "cfg.yaml"]

    ws = Path(store.projects[pid].workspace)
    assert (ws / "inputs" / "run.log").read_bytes() == b"stack trace here"
    assert (ws / "inputs" / "cfg.yaml").read_bytes() == b"x: 1"

    # A file-only message is still valid: the upload itself is the direction.
    r = client.post(f"/api/projects/{pid}/messages", data={"text": ""},
                    files=multi(("only.txt", b"hi")))
    assert r.status_code == 200, r.text
    runtime.close_all()
    print("upload on existing project ok")


def test_upload_names_are_contained_and_limited():
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    client = TestClient(api)
    pid = client.post("/api/projects", json={"name": "Limits", "goal": "g"}).json()["id"]
    ws = Path(store.projects[pid].workspace)

    canary = Path(cfg.data_dir) / "canary.txt"
    canary.write_text("do not touch")
    r = client.post(f"/api/projects/{pid}/uploads",
                    files=multi(("../canary.txt", b"OVERWRITE"),
                                ("/etc/passwd", b"OVERWRITE")))
    assert r.status_code == 200, r.text
    assert canary.read_text() == "do not touch"
    names = [f["name"] for f in store.projects[pid].input_files]
    assert names == ["canary.txt", "passwd"], names
    assert all((ws / "inputs" / n).exists() for n in names)

    cfg.max_upload_bytes = 16
    r = client.post(f"/api/projects/{pid}/uploads", files=multi(("big.bin", b"x" * 64)))
    assert r.status_code == 400 and "over the" in r.json()["detail"]
    cfg.max_upload_bytes = cfg.model_fields["max_upload_bytes"].default

    cfg.max_upload_files = 1
    r = client.post(f"/api/projects/{pid}/uploads",
                    files=multi(("a.txt", b"a"), ("b.txt", b"b")))
    assert r.status_code == 400 and "Too many files" in r.json()["detail"]
    cfg.max_upload_files = cfg.model_fields["max_upload_files"].default

    # Same name twice does not clobber the first upload.
    client.post(f"/api/projects/{pid}/uploads", files=multi(("dup.txt", b"one")))
    client.post(f"/api/projects/{pid}/uploads", files=multi(("dup.txt", b"two")))
    dupes = sorted(p.name for p in (ws / "inputs").glob("dup*"))
    assert dupes == ["dup-1.txt", "dup.txt"], dupes
    assert (ws / "inputs" / "dup.txt").read_bytes() == b"one"
    runtime.close_all()
    print("upload containment + limits ok")


def test_agents_can_read_uploaded_inputs():
    """The whole point: an agent's own workspace-relative read finds the file."""
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    client = TestClient(api)
    r = client.post("/api/projects", data={"name": "Readable", "goal": "g"},
                    files=multi(("spec.md", b"THE SPEC BODY")))
    pid, ws = r.json()["id"], r.json()["workspace"]

    worker = AgentSpec(role="backend-dev", specialty="backend", mission="m",
                       system_prompt="p", project_id=pid, depth=1, workspace=ws)
    store.upsert_agent(worker)
    store.upsert_agent(AgentSpec(role="orchestrator", specialty="o", mission="m",
                                 system_prompt="p", project_id=pid, depth=0,
                                 workspace=ws))
    store.flush()
    runtime.conversation_index["w-conv"] = worker.id

    obs = ReadFileExecutor()(ReadFileAction(path="inputs/spec.md"), FakeConv("w-conv"))
    assert not obs.is_error, obs.text
    assert "THE SPEC BODY" in obs.text
    runtime.conversation_index.pop("w-conv", None)
    runtime.close_all()
    print("agent read of inputs/ ok")


def test_inputs_reach_briefing_and_assignment_prompts():
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    client = TestClient(api)
    r = client.post("/api/projects", data={"name": "Briefed", "goal": "goal text"},
                    files=multi(("spec.md", b"x" * 40)))
    pid = r.json()["id"]
    task = TaskSpec(project_id=pid, title="card", objective="obj", expertise="backend",
                    column="ready")
    store.upsert_task(task)

    board = {col: [orch._task_to_dict(t) for t in items]
             for col, items in store.board(pid).items()}
    briefing = prompts.cycle_briefing(
        {"id": pid, "name": "Briefed", "goal": "goal text", "constraints": "",
         "workspace": store.projects[pid].workspace, "status": "active",
         "user_messages": [], "input_files": store.projects[pid].input_files},
        board, [], ["go"])
    assert "inputs/spec.md" in briefing, briefing

    assignment = prompts.worker_assignment_prompt(
        {"id": task.id, "title": "card", "objective": "obj"},
        turns_remaining=100,
        input_files=[{"path": "inputs/spec.md", "size": 40}])
    assert "inputs/spec.md" in assignment
    assert "source material" in assignment

    review = prompts.review_assignment_prompt(
        {"id": task.id, "title": "card"},
        input_files=[{"path": "inputs/spec.md", "size": 40}])
    assert "inputs/spec.md" in review
    runtime.close_all()
    print("inputs in briefing + prompts ok")


def test_dispatch_prompts_carry_input_files():
    """Both dispatch paths (tool + scheduler) attach the inputs block."""
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    client = TestClient(api)
    r = client.post("/api/projects", data={"name": "Dispatched", "goal": "g"},
                    files=multi(("spec.md", b"body")))
    pid = r.json()["id"]
    conv = FakeConv()
    CreateTaskExecutor()(CreateTaskAction(project_id=pid, title="card",
                                          objective="o", expertise="backend"), conv)
    task = store.project_tasks(pid)[0]
    worker = AgentSpec(role="backend-dev", specialty="backend", mission="m",
                       system_prompt="p", project_id=pid, depth=1,
                       workspace=store.projects[pid].workspace, turn_limit=50)
    store.upsert_agent(worker)
    store.upsert_agent(AgentSpec(role="orchestrator", specialty="o", mission="m",
                                 system_prompt="p", project_id=pid, depth=0,
                                 workspace=store.projects[pid].workspace))
    runtime.conversation_index["conv-1"] = next(
        a.id for a in store.agents.values() if a.role == "orchestrator")

    seen: list[str] = []
    runtime.run = lambda spec, tools, message, turn_limit=None, **kw: (
        seen.append(message), outcome())[1]
    RunAgentExecutor()(RunAgentAction(agent_id=worker.id, task_id=task.id), conv)
    assert seen and "inputs/spec.md" in seen[0], seen
    runtime.conversation_index.pop("conv-1", None)
    runtime.close_all()
    print("dispatch prompt carries inputs ok")


def test_ui_exposes_file_inputs():
    """A bare <input type=file> is invisible in this theme; the affordance the
    user can actually see is the styled label, so assert on that."""
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    client = TestClient(api)
    html = client.get("/").text
    js = client.get("/static/app.js").text

    assert html.count('class="filepick"') >= 1, "create-project picker missing"
    assert "Attach files" in html, "no visible attach affordance on create form"
    assert "Attach files" in js, "no visible attach affordance on message bar"
    assert ".filepick" in html, "picker has no styling (renders invisible)"
    assert 'name="files"' in js, "message-bar file input missing"
    assert "input_files" in js, "uploaded inputs not rendered"
    assert "new FormData(e.target)" in js, "submissions not multipart"
    runtime.close_all()
    print("upload UI contract ok")


def test_picked_files_survive_the_polling_rerender():
    """The board is rebuilt from innerHTML every 5s. Without a stash, a file the
    user picked (or typed text) vanishes before they can press submit."""
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    js = TestClient(api).get("/static/app.js").text
    assert "capturePendingForms()" in js and "restorePendingForms()" in js
    assert "state.pendingFiles" in js and "state.pendingForms" in js
    # A FileList cannot be assigned directly; it has to be rebuilt via DT.
    assert "new DataTransfer()" in js
    runtime.close_all()
    print("pending-form preservation ok")


def test_collapsed_project_stays_collapsed():
    """The old code re-added every project to openProjects on every poll, so a
    collapse was undone by the next 5s refresh."""
    cfg, store, runtime, orch, api = make(tempfile.mkdtemp())
    js = TestClient(api).get("/static/app.js").text

    # Seeding must be one-shot per project id, not per refresh.
    assert "seenProjects" in js, "open state is re-seeded on every poll"
    assert "state.seenProjects.has(p.id)" in js, "seed is not guarded by first-sight"
    reseed = 'if (!state.openProjects.has(p.id) && p.status !== "delivered")'
    assert reseed not in js, "unconditional re-open still present"

    # Only the summary toggles; a click in the card body (e.g. the message
    # input) used to flip the open state too.
    assert 't.closest("summary")' in js, "toggle is not scoped to the summary"
    assert 'sum.closest("details.card[data-project]")' in js
    # Controls inside the summary must not collapse the card they live in.
    guard = 't.closest("summary") &&'
    assert guard in js, "no summary-control guard"
    tail = js[js.index(guard):js.index(guard) + 260]
    assert "e.preventDefault()" in tail, "summary buttons still toggle the card"
    runtime.close_all()
    print("collapse persistence ok")


# ---------------------------------------------------------------- pause


def seed_paused(tmp: str):
    """A project with a staffed ready card, a review card, and a blocked card."""
    cfg, store, runtime, orch, api = make(tmp)
    conv = FakeConv()
    CreateProjectExecutor()(CreateProjectAction(name="Halted", goal="ship it",
                                                constraints=""), conv)
    pid = list(store.projects)[0]
    CreateTaskExecutor()(CreateTaskAction(project_id=pid, title="Build",
                                          objective="x", expertise="backend"), conv)
    CreateSubAgentExecutor()(CreateSubAgentAction(role="backend-dev",
                                                  specialty="backend", mission="b",
                                                  project_id=pid), conv)
    worker = next(a for a in store.agents.values() if a.role == "backend-dev")
    task = store.project_tasks(pid)[0]
    AssignTaskExecutor()(AssignTaskAction(agent_id=worker.id, task_id=task.id), conv)
    return cfg, store, runtime, orch, api, pid, task, worker


def test_pause_stops_scheduler_dispatch():
    cfg, store, runtime, orch, api, pid, task, worker = seed_paused(tempfile.mkdtemp())
    submitted: list[str] = []
    runtime.submit = lambda aid, fn, **kw: submitted.append(aid)

    assert orch.scheduler.sweep(), "active project should dispatch"
    assert submitted == [worker.id]

    submitted.clear()
    store.projects[pid].status = "paused"
    store.flush()
    assert not orch.scheduler.sweep(), "paused project must not dispatch"
    assert submitted == []
    runtime.close_all()
    print("pause stops scheduler ok")


def test_pause_freezes_the_board():
    cfg, store, runtime, orch, api, pid, task, worker = seed_paused(tempfile.mkdtemp())
    store.projects[pid].status = "paused"
    task.column = "backlog"
    task.depends_on = []
    orch.reconcile()
    assert task.column == "backlog", "paused board must not advance on its own"

    task.column = "blocked"
    task.blocked_cause = "dependency"
    orch.reconcile()
    assert task.column == "blocked"

    for t in store.project_tasks(pid):
        t.column = "done"
    orch.reconcile()
    assert store.projects[pid].status == "paused", \
        "a paused project must not flip to ready_for_delivery"
    runtime.close_all()
    print("pause freezes board ok")


def test_pause_produces_no_attention():
    cfg, store, runtime, orch, api, pid, task, worker = seed_paused(tempfile.mkdtemp())
    task.column = "review"
    task.review_verdict = None
    task.review_rounds = 0
    assert store.projects[pid].id == pid
    assert orch.detect().get(pid), "active project should raise attention"

    store.projects[pid].status = "paused"
    attention = orch.detect()
    assert not attention.get(pid), "paused project must not ask for coordination"
    runtime.close_all()
    print("pause silences detect ok")


def test_pause_blocks_new_work_from_tools():
    cfg, store, runtime, orch, api, pid, task, worker = seed_paused(tempfile.mkdtemp())
    conv = FakeConv()
    orch_id = next(a.id for a in store.agents.values() if a.role == "orchestrator")
    runtime.conversation_index["conv-1"] = orch_id
    store.projects[pid].status = "paused"
    store.flush()

    def expect_refused(executor, action, why):
        obs = executor()(action, conv)
        assert obs.is_error, f"{why} should be refused while paused"
        assert "paused" in obs.text, obs.text
        return obs

    expect_refused(CreateTaskExecutor, CreateTaskAction(
        project_id=pid, title="new", objective="o", expertise="backend"),
        "create_task")
    expect_refused(CreateSubAgentExecutor, CreateSubAgentAction(
        role="tester", specialty="testing", mission="t", project_id=pid),
        "create_sub_agent")
    expect_refused(AssignTaskExecutor, AssignTaskAction(
        agent_id=worker.id, task_id=task.id), "assign_task")
    expect_refused(RunAgentExecutor, RunAgentAction(
        agent_id=worker.id, task_id=task.id), "run_agent")
    expect_refused(ResumeAgentExecutor, ResumeAgentAction(
        agent_id=worker.id, task_id=task.id, instructions="keep going"),
        "resume_agent")
    expect_refused(RequestReviewExecutor, RequestReviewAction(
        task_id=task.id), "request_review")
    for t in store.project_tasks(pid):
        t.column = "done"
    expect_refused(SignalDeliveryExecutor, SignalDeliveryAction(
        project_id=pid, summary="all done"), "signal_delivery")

    # Nothing new was created by the refused calls.
    assert len(store.project_tasks(pid)) == 1
    assert not [a for a in store.agents.values() if a.role == "tester"]
    assert task.column != "review"
    runtime.conversation_index.pop("conv-1", None)
    runtime.close_all()
    print("pause blocks tool-level work ok")


def test_pause_lets_in_flight_work_finish():
    """The queue drains: a finishing agent can still report and record results."""
    cfg, store, runtime, orch, api, pid, task, worker = seed_paused(tempfile.mkdtemp())
    conv = FakeConv("w-conv")
    runtime.conversation_index["w-conv"] = worker.id
    task.column = "in_progress"
    store.projects[pid].status = "paused"
    store.flush()

    from agent_harness.tools import ReportToParentAction, ReportToParentExecutor
    obs = ReportToParentExecutor()(ReportToParentAction(
        status="complete", summary="finished while pausing",
        deliverables=["artifact.py"]), conv)
    assert not obs.is_error, obs.text
    finished = store.project_tasks(pid)[0]
    assert finished.result_summary, "the finishing report must be recorded"
    assert finished.column == "in_progress", "a paused board does not advance itself"
    runtime.conversation_index.pop("w-conv", None)
    runtime.close_all()
    print("pause drains in-flight work ok")


def test_resume_reopens_the_project():
    cfg, store, runtime, orch, api, pid, task, worker = seed_paused(tempfile.mkdtemp())
    client = TestClient(api)
    store.projects[pid].status = "paused"
    submitted: list[str] = []
    runtime.submit = lambda aid, fn, **kw: submitted.append(aid)
    orch.scheduler.sweep()
    assert submitted == []

    # The pause endpoint toggles, which is how the UI resumes a project.
    r = client.post(f"/api/projects/{pid}/pause")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "active"
    assert orch.scheduler.sweep(), "resumed project must be schedulable again"
    assert submitted == [worker.id]
    runtime.close_all()
    print("resume ok")


def main() -> None:
    for fn in (test_upload_with_project_create_lands_in_workspace,
               test_upload_on_existing_project_via_message_and_endpoint,
               test_upload_names_are_contained_and_limited,
               test_agents_can_read_uploaded_inputs,
               test_inputs_reach_briefing_and_assignment_prompts,
               test_dispatch_prompts_carry_input_files,
               test_ui_exposes_file_inputs,
               test_picked_files_survive_the_polling_rerender,
               test_collapsed_project_stays_collapsed,
               test_pause_stops_scheduler_dispatch,
               test_pause_freezes_the_board,
               test_pause_produces_no_attention,
               test_pause_blocks_new_work_from_tools,
               test_pause_lets_in_flight_work_finish,
               test_resume_reopens_the_project):
        print(f"--- {fn.__name__}")
        fn()
    print("\nALL UPLOAD + PAUSE TESTS PASSED")


if __name__ == "__main__":
    main()
