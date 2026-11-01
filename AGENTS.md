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
python tests/test_runtime_and_config.py  # no LLM needed
python tests/test_dashboard_features.py  # no LLM needed; removal, orphans, costs
python tests/test_blocks_cost_version.py # no LLM needed; block causes, card cost, version
python tests/test_e2e_llm.py 3         # needs the configured LLM endpoint
python tests/test_turn_budget.py       # needs the configured LLM endpoint
```

`test_board_mechanics.py` covers the whole board/delegation/review/delivery surface
without an LLM. Prefer extending it over the LLM-backed tests when changing mechanics —
the LLM tests take minutes and are non-deterministic.

The LLM-backed tests print a `ConversationRunError: cannot schedule new futures
after shutdown` traceback for one agent near the end. That is litellm's thread
pool being torn down at interpreter exit while a background run is still
finishing — it is logged by `_run_guarded`, does not fail the run, and the test
still exits 0. Judge these tests by exit code, not by absence of ERROR lines.

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
  "Cost calculation failed" warnings and report $0 cost. Pass
  `input_cost_per_token` / `output_cost_per_token` explicitly to price them;
  `LLMConfig` defaults them to the Qwen rates (2e-06 / 1.5e-05).
- `state.events` contains agent-sourced events that are **not** LLM turns
  (`SystemPromptEvent`, condenser events) and they have no `llm_response_id`
  attribute at all — read it with `getattr`, never as a plain attribute.
- `metrics.accumulated_token_usage` is `None` until the first billed response.
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
  `_dispatch` creates that link (and the reverse `agent.task_id`) itself, because the
  orchestrator can call `run_agent(task_id=...)` without ever calling `assign_task`.
- `assigned_agent_id` is the **worker** link and drives the card's turn counters;
  reviewers go on `reviewer_agent_id` and set `AgentSpec.is_reviewer`. Never let a
  review run write `assigned_agent_id` or `task.turns_used` — the card would show the
  reviewer's budget instead of the worker's. `Store._sync_task_budget` is gated on
  both, so a reviewer cannot overwrite the worker's count.
- `is_reviewer` is only known once something marks it: `assign_task` infers it from
  `task.column == "review"` (a task already in review has finished its worker
  stretch), and `_dispatch(review=True)` sets it. Three paths had to be gated this
  way — `_dispatch`, `assign_task`, and `reconfigure_agent` — because the
  orchestrator reaches all of them with a reviewer attached to a task.

## Blocked tasks: cause, and who may clear it

- `TaskSpec.blocked_reason` is prose for humans; `TaskSpec.blocked_cause` is the
  machine-readable class the harness branches on. **Always set both together**,
  and clear both together. A card left in `blocked` with a stale reason is the
  bug this section exists to prevent.
- Causes: `budget` (turn limit hit), `llm` (endpoint returned nothing),
  `agent_failed`, `worker` (the sub-agent reported blocked), `manual`
  (orchestrator blocked it, usually awaiting the user), `dependency`.
- Each cause has exactly one resolver, and it must clear the reason:
  `budget` -> `extend_agent`; `budget`/`llm`/`worker` -> `reconfigure_agent`;
  `dependency` -> `reconcile()` when upstream deps reach `done`; any -> a clean
  re-dispatch through `_dispatch`, or a user directive via the directives
  endpoint. `_dispatch` moves `blocked`/`rework`/`in_progress` to `review` on a
  clean run — that is deliberate, a clean run means the blocker is gone.
- `reconcile()` only auto-clears when deps are satisfied AND the cause is not
  `budget`/`llm`/`manual`: those need judgement, so they stay for the cycle
  briefing. Do not widen that set without a briefing-side equivalent.
- `report_to_parent` gained a `blocker` field: a worker reporting `blocked`
  restates the CURRENT reason; reporting `complete`/`partial` on a blocked card
  clears it. The worker on the card is the authority — it just re-checked.
- Reviewer dispatches are excluded from the `_dispatch` clean-run transition;
  their verdict moves the card through `record_review`.
- `BLOCK_FIX` in `app.js` mirrors the cause list for the UI. Update it when you
  add a cause, or cards render the generic fallback.

## Dashboard: removal, cost, cycle freshness

- Removing a project has three modes, all of which set `Project.status = "removed"`:
  `hide` (files untouched), `archive` (workspace + conversation logs moved to
  `data_dir/archived-projects/`), `purge` (workspace, conversation logs and the
  store records all deleted). Only purge drops the record, so hide/archive keep
  cost history queryable. Removal is refused with 409 while any of the project's
  agents are in `runtime.inflight`, and requires `confirm == project.name`.
- `Store.visible_projects()` is the only correct way to iterate projects for the
  board, alerts, reconcile and detect; `store.projects` still contains removed
  ones. `agents_in_status()` filters removed projects by default.
- Cost is stored per agent (`AgentSpec.cost_usd`) and summed on demand by
  `Store.project_cost()`. There is no cached project total, so the per-agent
  figures always reconcile to the project total and the harness total.
- SDK metrics are **per conversation**: a fresh conversation restarts
  `accumulated_cost` at 0. Keep the invariant `cost_usd == prior_cost_usd +
  current conversation spend`, which is what `run()` writes. `prior_*` is the
  spend from already-closed conversations. `reset_conversation()` sets
  `prior_* = cost_usd` (idempotent) and leaves `cost_usd` alone, so the board
  never shows a `$0` agent between the reset and the next run. Zeroing
  `cost_usd` on reset silently deletes an agent's spend — the bug these fields
  exist to prevent. Do not write `prior += cost_usd`: `cost_usd` already
  contains `prior`, so that double-counts on a second reset.
- Kanban cards show cost by mirroring the assigned agent (`_task_json(...,
  agents=store.agents)`). Cost is never stored on the card; if you add a cost
  field, derive it, do not persist it. Unassigned cards render "no spend" rather
  than `$0.00`, which reads as a bug.
- Cycle metadata lives in `store.meta`: `cycle_started_at`/`cycle_status` are
  written *before* a cycle runs, `_record_cycle_end()` writes `last_tick`,
  `last_cycle_completed_at`, `last_cycle_attempt_at`, `cycle_count` and
  `cycle_duration` after it, on success **and** failure. `last_tick` alone only
  advances on success, which is why it used to look frozen when every cycle
  raised; the header pill now reports attempt vs. completion separately and
  flags `stale`.
- Alert ids are `sha1(key)` where the key embeds a generation counter
  (`task.review_rounds`, `blocked_reason`, `agent.turns_used`,
  `project.ready_generation`). Acknowledging dismisses that occurrence only; a
  new occurrence gets a new id. `ready_generation` increments on every
  transition into `ready_for_delivery` — do not key the delivery alert on
  `updated_at`, which changes on unrelated touches.
- The agent tree is built from `parent_agent_id`, never `child_agent_ids`:
  `retire_agent` prunes the parent's child list to free a delegation slot, which
  would otherwise erase retired agents from the dashboard. The orchestrator is
  grafted in as the root because it is global and carries only one `project_id`.
- **Removing a project must delete its agents.** `Store.remove_project` calls
  `delete_project_agents`, which also scrubs the deleted ids out of every
  surviving parent's `child_agent_ids`. Skipping the scrub leaves stale ids that
  still count against `max_children_per_agent`, so the orchestrator is silently
  blocked from creating agents forever. `Orchestrator.sweep_orphans` runs at the
  top of every `run_cycle` to clean agents whose project or parent is gone and
  dangling child ids left by older builds or hand-edited `state.json`.
- `runtime.close()` on a `Conversation` is safe per-agent — `TmuxPanePool` is
  per-conversation, not process-global — but never close the orchestrator's
  live conversation. The orchestrator's transcript lives under
  `conversations/<its current project_id>/<orch id>`, so removal modes must
  exclude the orchestrator from doomed-agent cleanup. `_prune_project_log_dirs`
  protects it by keeping any log entry whose name is still a key in
  `store.agents`.
- Removal refuses with `409` while any of the project's agents are in
  `runtime.inflight`. Do not relax this: closing a conversation out from under a
  running agent produces `TmuxPanePool is not initialized or already closed`.
- **Cost semantics.** `harness_cost()` sums every agent in the store, so spend
  can never be orphaned by a `project_id` pointing at a purged project.
  `project_cost()` deliberately **excludes** the orchestrator: it is shared
  across projects and its `project_id` drifts to whichever project it last
  touched, so charging it per-project double-counts it and loses it on purge.
  The UI marks the orchestrator node `shared` and the header pill shows
  `total = projects + orchestrator`. `project.final_cost_usd` is snapshotted via
  `project_cost()` at removal, so hidden/archived projects keep a consistent total.

## Persisted-model validation (state.json)

- All persisted models inherit `_Model`, which sets
  `model_config = ConfigDict(validate_assignment=True)`. This is load-bearing:
  the store flushes these objects to JSON and re-validates on load, so without
  it an invalid assignment is accepted in memory, written to disk, and only
  crashes on the **next** startup — a crash loop under `restart: unless-stopped`
  with no UI to recover through. This exact bug shipped when `llm_unavailable`
  was added to the runtime but not to the `AgentStatus` Literal.
- `Store._load` never raises for a bad record: `_load_records` tries the model,
  then a per-kind `_repair_*` (unknown agent status -> `failed`, unknown project
  status -> `active`, unknown task column -> `blocked` + a `blocked_reason`),
  and drops the record with a logged error only if repair is impossible. Repair
  preserves turns/cost/version, so a bad enum never loses work.
- `AGENT_STATUSES` / `PROJECT_STATUSES` are derived with `get_args()` from the
  Literals — never hand-maintain a second copy of the status list.
- `AgentRuntime._map_status` must stay **total**: it feeds a validating field, so
  an unmapped SDK `ConversationExecutionStatus` value would raise inside a live
  run. Unknown values degrade to `failed`. If the SDK adds statuses, add a
  mapping (the test asserts coverage over every enum member).
- `_one_line()` compacts a `ValidationError` for logs; `str(exc)` is multi-line
  and gets truncated by log shippers exactly when the rejected field matters.

## Project versions and revisions

- `Project.version_major`/`version_minor` render as `major.minor` through the
  `Project.version` property. New projects start `0.1`; `revise_project()` is the
  **only** mutator — minor +1 normally, major +1 with minor reset when
  `release=True`. Never write the version fields directly.
- A revision is recorded only for *material* change: task creation, a task's
  content fields (title/objective/expertise/criteria/deliverables/depends_on),
  a review verdict, readiness transitions, and scope edits. **Kanban column
  moves are not revisions** — `update_task` distinguishes content change from a
  bare move via a content-diff check, so keep that split if you touch the tool.
- `signal_delivery` bumps the minor only; the *release* is the user accepting it
  (`mark-delivered`, `release=True`). It also skips the revision when the loop
  already flagged readiness this cycle (`was_ready`), otherwise every cycle
  would inflate the version.
- `revision_log` is trimmed to `REVISION_LOG_LIMIT` (50) on append. Entries are
  `{"version", "reason", "at"}` where `at` is the project's ISO `updated_at` —
  note `agoLabel()` in the UI must accept both ISO strings (store records) and
  epoch ints (cycle metadata); they are not the same format.
- Old state files have no version fields; `Store` loads them at `0.1`/empty log.
  Keep the `Project` field defaults rather than making them required, or
  upgrading breaks state loading.

## Harness version (distinct from project version)

- `agent_harness/version.py::__version__` is the **build** version of the harness
  itself. Bump it on every change to the harness: patch = fixes/copy, minor = new
  tools, endpoints, or backwards-compatible persisted fields, major = breaking
  change to config, state schema, or the board model.
- `version_string()` appends the git SHA from a checkout (`1.2.0+abc1234`) and is
  bare inside an image. Set `HARNESS_GIT_SHA` to pin it in a build.
- It surfaces in the dashboard header pill, `/health`, `/api/state`
  (`meta.harness_version`), and FastAPI's OpenAPI info. This matters because the
  stack bind-mounts its data volume: a container that was not rebuilt serves an
  old build against new data, and that is otherwise invisible.
- `Store._load()` copies the on-disk `meta.harness_version` into
  `meta.harness_version_stored` before it is overwritten, so the header tooltip
  can name the build that last wrote `state.json`. Do not read
  `meta.harness_version` for that comparison — it is the running build.
