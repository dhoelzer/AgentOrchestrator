"""End-to-end run against the configured LLM.

Creates a project, runs orchestrator cycles, and asserts the full loop:
task decomposition -> dynamic specialist creation -> dispatch -> review ->
delivery signal, with real files written to the project workspace.
"""

from __future__ import annotations

import shutil
import sys

sys.path.insert(0, ".")

from agent_harness.config import HarnessConfig
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime
from agent_harness.store import Store
from agent_harness.tools import register_harness_tools

CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def main() -> int:
    shutil.rmtree("/tmp/e2e", ignore_errors=True)
    cfg = HarnessConfig.load()
    cfg.data_dir = "/tmp/e2e"
    cfg.projects_dir = "/tmp/e2e/projects"
    cfg.tick_seconds = 5
    cfg.orchestrator_turn_limit = 25
    cfg.sub_agent_turn_limit = 100
    cfg.ensure_dirs()

    store = Store(cfg.data_dir)
    runtime = AgentRuntime(cfg, store)
    register_harness_tools()
    orch = Orchestrator(cfg, store, runtime)

    store.create_project(
        name="Greeter",
        goal="A file greeter.py in the project workspace that defines greet(name) "
             "returning 'hello <name>' and prints it when run as a script.",
        workspace="/tmp/e2e/projects/greeter",
        constraints="Standard library only. No tests framework needed.",
    )
    print("project created:", [p.id for p in store.projects.values()])

    for i in range(CYCLES):
        print(f"\n===== CYCLE {i + 1} =====")
        reports = orch.run_cycle()
        for r in reports:
            print(f"  project={r.project_id} status={r.orchestrator_status} "
                  f"attention={len(r.attention)}")
            if r.error:
                print("  ERROR:", r.error)
        counts = store.column_counts(next(iter(store.projects)))
        print("  board:", {k: v for k, v in counts.items() if v})
        agents = store.project_agents(next(iter(store.projects)))
        print("  agents:", [(a.role, a.status, f"{a.turns_used}/{a.turn_limit}")
                            for a in agents])

    project = next(iter(store.projects.values()))
    print("\n===== FINAL =====")
    print("project status:", project.status, "delivery_signaled:",
          project.delivery_signaled)
    for t in store.project_tasks(project.id):
        print(f"  task {t.id} [{t.column}] {t.title} "
              f"turns={t.turns_used}/{t.turn_limit} verdict={t.review_verdict}")
    for a in store.project_agents(project.id):
        print(f"  agent {a.id} {a.role} depth={a.depth} status={a.status} "
              f"turns={a.turns_used}/{a.turn_limit} parent={a.parent_agent_id}")

    import pathlib
    ws = pathlib.Path(project.workspace)
    files = [str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file()]
    print("workspace files:", files)

    kinds = [e.kind for e in store.recent_events]
    print("event kinds:", {k: kinds.count(k) for k in set(kinds)})

    runtime.close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
