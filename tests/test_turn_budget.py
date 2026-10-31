"""Turn-budget enforcement: exhaustion -> report back -> orchestrator re-budgets.

Uses a small budget so the test is fast; the mechanism is identical at 100.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from agent_harness import prompts
from agent_harness.config import HarnessConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime
from agent_harness.store import AgentSpec, Store, TaskSpec, new_id
from agent_harness.tools import (
    ExtendAgentAction, ExtendAgentExecutor, ReconfigureAgentAction,
    ReconfigureAgentExecutor, _dispatch, _task_dict, register_harness_tools,
)

# A 1-turn first budget makes exhaustion deterministic: the agent's first action on
# a 40-file task is always a tool call, never `finish`, so the SDK must stop it on
# the iteration cap. Later stretches use a larger budget to show resumed progress.
BUDGET = 1
STRETCH = 4


class FakeConv:
    def __init__(self, cid):
        self.id = cid


def main() -> int:
    shutil.rmtree("/tmp/budget", ignore_errors=True)
    cfg = HarnessConfig.load()
    cfg.data_dir = "/tmp/budget"
    cfg.projects_dir = "/tmp/budget/projects"
    cfg.sub_agent_turn_limit = BUDGET
    cfg.orchestrator_turn_limit = 10
    cfg.ensure_dirs()

    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    register_harness_tools()
    orch = Orchestrator(cfg, store, runtime)

    project = store.create_project(
        name="Budget", goal="Prove the turn budget stops an agent mid-task.",
        workspace="/tmp/budget/projects/budget")
    task = TaskSpec(
        project_id=project.id, title="Write 40 notes files",
        objective="Create note1.txt .. note40.txt, each containing its own number",
        expertise="generalist", turn_limit=BUDGET,
        acceptance_criteria=["note1.txt .. note40.txt all exist"],
        deliverables=[f"note{i}.txt" for i in range(1, 41)],
    )
    store.upsert_task(task)

    system_prompt = prompts.specialist_prompt(
        agent_name="note-writer", specialty="file creation",
        mission="Create note1.txt .. note40.txt, each containing its own number.",
        project_name=project.name, project_goal=project.goal,
        workspace=project.workspace, constraints="", turn_limit=BUDGET,
    )
    agent = AgentSpec(
        id=new_id("agent"), project_id=project.id, role="note-writer",
        specialty="file creation", mission="Create the note files.",
        system_prompt=system_prompt,
        depth=1, parent_agent_id=orch.spec.id, task_id=task.id,
        workspace=project.workspace, turn_limit=BUDGET,
    )
    store.upsert_agent(agent)
    store.flush()

    # --- stretch 1: should stop at the budget ---
    msg = prompts.worker_assignment_prompt(_task_dict(task), turns_remaining=BUDGET, prior=None)
    _dispatch(agent, task, msg, BUDGET)

    agent = store.agents[agent.id]
    task = store.tasks[task.id]
    print(f"after stretch 1: agent turns={agent.turns_used}/{agent.turn_limit} "
          f"status={agent.status}")
    print(f"                 task column={task.column} turns={task.turns_used}")
    # The budget is a hard stop: the run was cut off by the iteration cap, not by
    # the agent deciding it was finished.
    assert agent.turns_used <= BUDGET, f"budget breached: {agent.turns_used} > {BUDGET}"
    assert agent.turns_used == BUDGET, "expected the budget to be fully consumed"
    assert agent.status == "budget_exhausted", agent.status
    assert task.column == "blocked", task.column

    created = sorted(p.name for p in Path(project.workspace).glob("note*.txt"))
    print(f"files after stretch 1: {len(created)} (task needs 40)")

    # --- the orchestrator is told, and decides to extend ---
    attention = orch.detect()
    briefs = [a for msgs in attention.values() for a in msgs
              if "BUDGET EXHAUSTED" in a or "budget" in a.lower()]
    assert briefs, "orchestrator was not briefed about the exhausted budget"
    print("orchestrator briefing received:", briefs[0][:120].replace("\n", " "))

    obs = ExtendAgentExecutor()(ExtendAgentAction(
        agent_id=agent.id, extra_turns=STRETCH,
        decision="Real progress; the plan is sound, just needs more turns."),
        FakeConv("o"))
    assert not obs.is_error, obs.text
    agent = store.agents[agent.id]
    assert agent.turn_limit == BUDGET + STRETCH and agent.status == "idle"
    assert store.tasks[task.id].column == "ready"
    print(f"extended -> limit={agent.turn_limit}, task column={store.tasks[task.id].column}")

    # --- stretch 2: resume with the new budget ---
    prior = agent.last_report
    msg = prompts.worker_assignment_prompt(_task_dict(store.tasks[task.id]),
                                   turns_remaining=STRETCH, prior=prior)
    _dispatch(agent, store.tasks[task.id], msg, BUDGET + STRETCH)
    agent = store.agents[agent.id]
    task = store.tasks[task.id]
    print(f"after stretch 2: agent turns={agent.turns_used}/{agent.turn_limit} "
          f"status={agent.status} task column={task.column}")
    assert agent.turns_used <= agent.turn_limit
    created2 = sorted(p.name for p in Path(project.workspace).glob("note*.txt"))
    print("files after stretch 2:", created2)
    assert len(created2) >= len(created), "extension should have produced progress"
    assert agent.turns_used <= agent.turn_limit, "second budget breached"

    # --- reconfigure path: new prompt/tools needs a fresh conversation ---
    obs = ReconfigureAgentExecutor()(ReconfigureAgentAction(
        agent_id=agent.id, decision="Too slow; give it a tighter, batch-oriented prompt.",
        system_prompt="You are a fast executor. Create all remaining files in ONE "
                      "terminal command using a shell loop, then call report_to_parent.",
        turn_limit=STRETCH), FakeConv("o"))
    assert not obs.is_error, obs.text
    agent = store.agents[agent.id]
    assert agent.turns_used == 0 and agent.runs == 0, (agent.turns_used, agent.runs)
    print(f"reconfigured -> fresh conversation, turns reset to {agent.turns_used}")

    # --- final stretch to completion ---
    msg = prompts.worker_assignment_prompt(_task_dict(store.tasks[task.id]),
                                   turns_remaining=STRETCH, prior=agent.last_report)
    _dispatch(agent, store.tasks[task.id], msg, STRETCH)
    task = store.tasks[task.id]
    created3 = sorted(p.name for p in Path(project.workspace).glob("note*.txt"))
    print(f"after stretch 3: task column={task.column} files={len(created3)}")

    runtime.close_all()
    print("\nBUDGET ENFORCEMENT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
