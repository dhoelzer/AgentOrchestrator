# Repository notes

Agent harness built on the OpenHands Software Agent SDK (`openhands-sdk` / `openhands-tools`).

## Run

```bash
pip install -r requirements.txt
python -m harness.main          # web UI on :8090 + forever orchestrator loop
```

## Tests

```bash
python tests/test_board_mechanics.py   # no LLM needed; fast, run this first
python tests/test_e2e_llm.py 3         # needs the configured LLM endpoint
python tests/test_turn_budget.py       # needs the configured LLM endpoint
```

`test_board_mechanics.py` covers the whole board/delegation/review/delivery surface
without an LLM. Prefer extending it over the LLM-backed tests when changing mechanics —
the LLM tests take minutes and are non-deterministic.

## Non-obvious SDK facts

- `ToolDefinition.name` is a `ClassVar`. Passing `name=` per instance is silently
  ignored — set the name on the `ToolDefinition` class or register by action type.
- `Conversation` resume **replays the persisted system prompt and tool schema**.
  Changing an agent's prompt/tools requires a fresh `conversation_id`
  (`AgentRuntime.reset_conversation`), not just a new `Agent`.
- A run stopped by a limit emits `ConversationErrorEvent` with `code` of
  `MaxIterationsReached` or `MaxBudgetReached`. Detect exhaustion by that code, not by
  matching `detail` text. Scope the scan to events added during the current run, or a
  stale code from an earlier run leaks into the next one.
- `max_iteration_per_run` caps a single `run()`, not cumulative turns. Track cumulative
  turns yourself and set the cap to `limit - used` per run.
- Turn counts derived from `state.events` can regress if a condenser prunes events;
  `AgentRuntime.run` clamps with `max(before, count)` so budgets cannot leak.
- `include_default_tools=["FinishTool", "ThinkTool"]` keeps `finish`/`think` while
  supplying an explicit tool list.
- Unmapped local model names (e.g. `openai/qwen`) emit harmless litellm
  "Cost calculation failed" warnings; cost metrics stay 0.

## Harness invariants

- One process per `data_dir`. `Store` is in-memory with a JSON flush; two processes
  will clobber each other's `state.json`.
- Runs are serialised **per agent**, not globally, so `run_agent` background dispatch
  is genuinely parallel. Lock order is agent lock → store lock; do not invert it.
- `reconfigure_agent` must be rejected while the agent is in `runtime.inflight`.
- A dispatched task always keeps `assigned_agent_id` set, otherwise budget exhaustion
  on a `blocked` task becomes unattributable and the orchestrator is never briefed.
