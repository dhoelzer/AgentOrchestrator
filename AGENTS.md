# Repository notes

Agent harness built on the OpenHands Software Agent SDK (`openhands-sdk` / `openhands-tools`).

## Naming: package vs data dir

The Python package is **`agent_harness`**. The top-level `harness/` directory is
reserved for the bind-mounted runtime data (config, state, conversations, project
workspaces) and is gitignored. Do not rename the package back to `harness` — it
would collide with that mount both on the host and inside the image.

## Run

```bash
pip install -r requirements.txt
python -m agent_harness.main          # web UI on :8090 + forever orchestrator loop

docker compose up -d --build          # containerized + Docker-in-Docker sidecar
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
- The SDK strips only a fixed set of sensitive vars (`SESSION_API_KEY`,
  `OH_SECRET_KEY`, `OH_SESSION_API_KEYS_*`, OTEL headers) from the agent shell
  environment. `DOCKER_HOST` and `HARNESS_LLM_API_KEY` do reach agent commands —
  which is what makes DinD work, and also means agents can read the LLM key.
- In the `docker:cli` image the CLI plugins live in
  `/usr/local/libexec/docker/cli-plugins`, **not** `/usr/local/lib/docker/...`.

## Container

- `compose.yaml` runs the app privileged with a `docker:29-dind` sidecar and
  binds `./harness` -> `/data`. `privileged` is required for the inner daemon to
  mount overlayfs; it is also root-equivalent on the host.
- The entrypoint seeds `/data/config.json` from the image default on first run
  and `chmod 666`s it, since the repo's `config.json` is mode 600 and the
  container writes as root.
- `HARNESS_HOST` is read by `config.py`; `HARNESS_PORT`, `HARNESS_DATA_DIR`,
  `HARNESS_TICK_SECONDS`, `HARNESS_SUB_AGENT_TURN_LIMIT`,
  `HARNESS_MAX_PARALLEL_WORKERS` and the three `HARNESS_LLM_*` vars also
  override `config.json`.
- This sandbox cannot build or run containers (nested overlayfs mounts fail with
  `invalid argument`), so image builds cannot be verified here. Validate with
  `docker compose config` and `docker buildx build --check` instead.

## Harness invariants

- One process per `data_dir`. `Store` is in-memory with a JSON flush; two processes
  will clobber each other's `state.json`.
- Runs are serialised **per agent**, not globally, so `run_agent` background dispatch
  is genuinely parallel. Lock order is agent lock → store lock; do not invert it.
- `reconfigure_agent` must be rejected while the agent is in `runtime.inflight`.
- A dispatched task always keeps `assigned_agent_id` set, otherwise budget exhaustion
  on a `blocked` task becomes unattributable and the orchestrator is never briefed.
