"""Sub-agent fan-out limits, and how the loop paces itself.

Two things this covers:

1. Creating sub-agents is unlimited. The ceiling that should govern fan-out is
   how many may *run at once* inside a project, not how many may exist. A hard
   creation cap turns "spawn the expert this card needs" into a refusal, which
   is the wrong constraint.

2. The orchestrator has no lifetime turn budget, and the loop should not run
   frequently when there is nothing to do. A turn count that reads like a
   countdown makes the model triage instead of plan, and a floor-length tick on
   an idle harness buys nothing.

Run: python3 tests/test_concurrency_and_tick.py
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

from agent_harness import prompts
from agent_harness.config import HarnessConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import (AgentRuntime, ProjectConcurrencyLimit,
                           RunOutcome)
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
    RunAgentAction,
    RunAgentExecutor,
    register_harness_tools,
)

class FakeConv:

    def __init__(self, cid="conv-1"):
        self.id = cid

def drain(runtime, timeout=10.0) -> None:
    """Let parked runs finish before the temp dir is torn down.

    The stubbed runs release on gate.set(); without waiting, the still-running
    threads write into a temp dir that is being removed.
    """
    deadline = time.monotonic() + timeout
    while runtime.inflight and time.monotonic() < deadline:
        time.sleep(0.05)

def _ok(obs) -> bool:
    return not obs.is_error

def seed(tmp: str, **overrides):
    register_harness_tools()
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    cfg.projects_dir = str(Path(tmp) / "projects")
    cfg.tick_seconds = 60
    for k, v in overrides.items():
        setattr(cfg, k, v)
    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    orch = Orchestrator(cfg, store, runtime)
    conv = FakeConv("orch-conv")
    CreateProjectExecutor()(CreateProjectAction(
        name="Widget", goal="ship a widget", constraints=""), conv)
    pid = list(store.projects)[0]
    return cfg, store, runtime, orch, pid, conv

def add_card(store, pid, title):
    conv = FakeConv("orch-conv")
    CreateTaskExecutor()(CreateTaskAction(
        project_id=pid, title=title, objective=f"do {title}",
        expertise="backend", acceptance_criteria=["done"],
        deliverables=[f"{title}.py"]), conv)
    return store.project_tasks(pid)[-1]

def add_agent(store, runtime, pid, role):
    conv = FakeConv("orch-conv")
    CreateSubAgentExecutor()(CreateSubAgentAction(
        role=role, specialty="backend", mission=f"work on {role}",
        project_id=pid), conv)
    return [a for a in store.project_agents(pid) if a.role == role][0]

def blocking_run(runtime):
    """Stub the model call with a run that parks until released.

    Stubbed at _run_guarded, one level below run(), so the real run() still
    executes: the project-slot claim, the concurrency gate, and the inflight
    bookkeeping are exactly what these tests are about. Stubbing run() itself
    would bypass the enforcement under test.
    """
    gate = threading.Event()

    def fake_guarded(spec, tools, message, turn_limit):
        gate.wait(timeout=10)
        return RunOutcome(turns_used=1, turns_this_run=1, status="idle",
                          final_report="done", budget_exhausted=False,
                          tokens=5, cost_usd=0.0)

    runtime._run_guarded = fake_guarded
    return gate

# ---------------------------------------- creation is unlimited, running is not

def test_creating_many_sub_agents_is_unlimited():
    """No cap on how many specialists exist; the old default of 6 per parent
    refused the tenth expert a project needed."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, conv = seed(tmp)
        add_card(store, pid, "Build the API")
        errors = []
        for i in range(25):
            obs = CreateSubAgentExecutor()(CreateSubAgentAction(
                role=f"expert-{i}", specialty="backend",
                mission=f"job {i}", project_id=pid), conv)
            if not _ok(obs):
                errors.append((i, obs.text))
        assert not errors, f"creation refused at: {errors[:2]}"
        created = [a for a in store.project_agents(pid) if a.depth == 1]
        assert len(created) == 25, len(created)
        runtime.close_all()
        print("unlimited sub-agent creation ok")

def test_creation_cap_is_opt_in():
    """0 means unlimited; a non-zero value still works as a guard."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, conv = seed(tmp, max_children_per_agent=2)
        for i in range(2):
            CreateSubAgentExecutor()(CreateSubAgentAction(
                role=f"a{i}", specialty="backend", mission="m",
                project_id=pid), conv)
        obs = CreateSubAgentExecutor()(CreateSubAgentAction(
            role="a3", specialty="backend", mission="m", project_id=pid), conv)
        assert "limit 2" in obs.text, obs.text
        runtime.close_all()
        print("opt-in creation cap ok")

def test_project_concurrency_limit_refuses_the_extra_run():
    """The real limit: how many sub-agents one project runs at once."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, conv = seed(
            tmp, max_concurrent_sub_agents_per_project=2, max_parallel_workers=5)
        release = blocking_run(runtime)
        add_card(store, pid, "Build the API")
        agents = [add_agent(store, runtime, pid, f"w{i}") for i in range(3)]
        for a in agents[:2]:
            t = store.project_tasks(pid)[0]
            AssignTaskExecutor()(AssignTaskAction(task_id=t.id, agent_id=a.id), conv)
            threading.Thread(
                target=lambda ag=a: RunAgentExecutor()(
                    RunAgentAction(agent_id=ag.id, background=False), conv),
                daemon=True).start()
        deadline = time.monotonic() + 5
        while runtime.project_run_counts().get(pid, 0) < 2 \
                and time.monotonic() < deadline:
            time.sleep(0.02)
        assert runtime.project_run_counts().get(pid) == 2, runtime.project_run_counts()
        assert not runtime.project_slot_free(pid)

        t = store.project_tasks(pid)[0]
        AssignTaskExecutor()(AssignTaskAction(task_id=t.id, agent_id=agents[2].id), conv)
        obs = RunAgentExecutor()(RunAgentAction(agent_id=agents[2].id,
                                               background=False), conv)
        assert "concurrency limit" in obs.text, obs.text
        assert "no limit on how many" in obs.text.lower(), obs.text
        release.set()
        drain(runtime)
        runtime.close_all()
        print("per-project concurrency limit ok")

def test_limit_raises_when_dispatched_anyway():
    """The runtime is the enforcement point, not just the tool's courtesy check."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, conv = seed(
            tmp, max_concurrent_sub_agents_per_project=1, max_parallel_workers=5)
        release = blocking_run(runtime)
        add_card(store, pid, "Build the API")
        a1, a2 = add_agent(store, runtime, pid, "w1"), add_agent(store, runtime, pid, "w2")
        threading.Thread(target=lambda: runtime.run(a1, [], "go"),
                         daemon=True).start()
        deadline = time.monotonic() + 5
        while runtime.project_slot_free(pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not runtime.project_slot_free(pid), "first run never claimed a slot"
        raised = False
        try:
            runtime.run(a2, [], "go")
        except ProjectConcurrencyLimit:
            raised = True
        assert raised, "runtime allowed a project to exceed its concurrency limit"
        release.set()
        drain(runtime)
        runtime.close_all()
        print("runtime enforcement ok")

def test_orchestrator_does_not_consume_a_project_slot():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid, conv = seed(
            tmp, max_concurrent_sub_agents_per_project=1)
        spec = orch._load_or_create_spec()
        assert runtime._claim_project_slot(spec) is True
        assert runtime.project_run_counts() == {}, runtime.project_run_counts()
        runtime.close_all()
        print("orchestrator excluded from project slots ok")

# ------------------------------------------------------- no lifetime budget

def test_orchestrator_prompt_denies_a_lifetime_budget():
    p = prompts.orchestrator_prompt()
    assert "no lifetime limit" in p
    assert "running out of turns" in p  # named in order to be forbidden
    assert "fresh slice of turns" in p

def test_briefing_does_not_show_a_countdown_for_the_orchestrator():
    """'3272/3320' was read as a lifetime countdown, and the orchestrator
    started reporting it was down to its last cycle."""
    text = prompts.cycle_briefing(
        {"id": "p1", "name": "P", "goal": "g", "constraints": "",
         "workspace": "/tmp", "status": "active", "user_messages": [],
         "input_files": []},
        board={"ready": [], "in_progress": []},
        agents=[{"id": "o1", "role": "orchestrator", "specialty": "orchestration",
                 "status": "idle", "turns_used": 3272, "turn_limit": 3320,
                 "task_id": None, "parent_agent_id": None}],
        attention=[])
    assert "3272/3320" not in text, text
    assert "3272 lifetime (no limit; 3320 per cycle)" in text, text
    # A sub-agent keeps its real, finite budget.
    text2 = prompts.cycle_briefing(
        {"id": "p1", "name": "P", "goal": "g", "constraints": "",
         "workspace": "/tmp", "status": "active", "user_messages": [],
         "input_files": []},
        board={"ready": [], "in_progress": []},
        agents=[{"id": "w1", "role": "worker", "specialty": "backend",
                 "status": "running", "turns_used": 40, "turn_limit": 100,
                 "task_id": None, "parent_agent_id": "o1"}],
        attention=[])
    assert "40/100" in text2

def test_briefing_states_concurrency_not_creation():
    text = prompts.cycle_briefing(
        {"id": "p1", "name": "P", "goal": "g", "constraints": "",
         "workspace": "/tmp", "status": "active", "user_messages": [],
         "input_files": []},
        board={"ready": [], "in_progress": []}, agents=[], attention=[],
        capacity={"slots_free": 3, "worker_slots": 3, "workers_active": 0,
                  "max_concurrent_sub_agents_per_project": 3})
    assert "Capacity now" in text
    assert "3 of 3" in text
    assert "Creating agents is unlimited" in text

# ------------------------------------------------------------- idle tick

def _tick_orch(tmp, **over):
    cfg, store, runtime, orch, pid, conv = seed(tmp, **over)
    return cfg, store, runtime, orch, pid

def test_idle_cycles_back_off_to_the_idle_tick():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _tick_orch(
            tmp, tick_seconds=60, min_tick_seconds=10, idle_tick_seconds=300)
        orch.effective_tick = 60.0
        seen = []
        for _ in range(8):
            seen.append(orch._next_tick(0.05, had_work=False))
        assert all(s > 60 for s in seen[1:]), seen
        assert seen[-1] == 300.0, seen
        runtime.close_all()
        print("idle backoff ok:", seen)

def test_work_still_uses_measured_cycle_duration():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _tick_orch(
            tmp, tick_seconds=60, min_tick_seconds=10, idle_tick_seconds=300,
            tick_duty_cycle=0.5)
        orch.effective_tick = 60.0
        # A 40s cycle at a 0.5 duty cycle wants an 80s interval.
        got = orch._next_tick(40.0, had_work=True)
        assert abs(got - 80.0) < 1.0, got
        runtime.close_all()
        print("busy tick still duty-cycle driven ok")

def test_idle_does_not_collapse_the_tick_to_the_floor():
    """The old behaviour: an empty cycle costs ~0s, so the duty-cycle maths
    pulled the interval to the floor and the loop retried every floor tick."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _tick_orch(
            tmp, tick_seconds=60, min_tick_seconds=10, idle_tick_seconds=300)
        orch.effective_tick = 60.0
        for _ in range(4):
            orch._next_tick(0.05, had_work=False)
        assert orch.effective_tick > 60, orch.effective_tick
        runtime.close_all()
        print("no floor collapse on idle ok")

def test_wake_shortens_the_idle_wait():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _tick_orch(tmp, idle_tick_seconds=300)
        assert not orch._wake.is_set()
        orch.wake("project created")
        assert orch._wake.is_set()
        # The loop sleeps on the event, so a wake means it stops waiting.
        started = time.monotonic()
        orch._wake.wait(timeout=300)
        assert time.monotonic() - started < 1.0
        runtime.close_all()
        print("wake ok")

def test_stop_wakes_a_parked_loop():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _tick_orch(tmp, idle_tick_seconds=300)
        orch.stop()
        assert orch._wake.is_set() and orch._stop.is_set()
        runtime.close_all()
        print("stop wakes loop ok")

def test_idle_flag_requires_both_no_attention_and_nothing_running():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, store, runtime, orch, pid = _tick_orch(tmp)

        def no_llm(spec, tools, message, turn_limit=None, kind=None):
            return RunOutcome(turns_used=1, turns_this_run=1, status="idle",
                              final_report="ok", budget_exhausted=False,
                              tokens=5, cost_usd=0.0)

        runtime.run = no_llm
        orch._cycle_had_work = False
        runtime._mark_inflight("some-agent")
        orch._run_cycle()
        # An agent is in flight, so the loop must keep watching at working pace.
        assert orch._cycle_had_work is True
        runtime._unmark_inflight("some-agent")

        # Delivered project, nothing running: nothing to coordinate, so the
        # cycle should register as idle and let the loop back off. (A project
        # with no tasks is NOT idle - that genuinely needs orchestrator
        # attention, and detect() flags it.)
        store.projects[pid].status = "delivered"
        orch._cycle_had_work = True
        orch._run_cycle()
        assert orch._cycle_had_work is False, "idle cycle did not register as idle"
        runtime.close_all()
        print("idle detection ok")

def test_settings_form_exposes_the_new_knobs():
    """A limit the user cannot see is a limit they cannot reason about. The
    concurrency ceiling in particular must read as concurrency, not creation."""
    from agent_harness.web.api import TEMPLATES
    html = (TEMPLATES / "index.html").read_text()
    js = (Path(__file__).resolve().parent.parent
          / "agent_harness" / "web" / "static" / "app.js").read_text()
    for name in ("max_concurrent_sub_agents_per_project",
                 "max_children_per_agent", "idle_tick_seconds"):
        assert f'name="{name}"' in html, name
    assert "no cap on how many may be created" in html
    # The JS must know the keys or the form never sends them.
    for name in ("max_concurrent_sub_agents_per_project", "idle_tick_seconds"):
        assert name in js, name


def test_new_config_fields_are_patchable():
    from agent_harness.web.api import _PATCHABLE
    assert "max_concurrent_sub_agents_per_project" in _PATCHABLE
    assert "idle_tick_seconds" in _PATCHABLE

if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            print(f"--- {name}")
            fn()
    print("\nALL CONCURRENCY + TICK TESTS PASSED")
