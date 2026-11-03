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
python tests/test_queue_input_recovery.py # no LLM needed; LLM queue, project input, recovery
python tests/test_llm_failures_and_reentrancy.py # no LLM needed; LLM alerting, orchestrator re-entrancy
python tests/test_e2e_llm.py 3         # needs the configured LLM endpoint
python tests/test_turn_budget.py       # needs the configured LLM endpoint
```

`test_board_mechanics.py` covers the whole board/delegation/review/delivery surface
without an LLM. Prefer extending it over the LLM-backed tests when changing mechanics —
the LLM tests take minutes and are non-deterministic.

`test_e2e_llm.py` drains `runtime.inflight` before it prints its final report.
Sub-agent runs go to a thread pool, so the last cycle can return while a worker
is still mid-LLM-call; exiting then tears down litellm's thread pool under the
run and raises `ConversationRunError: cannot schedule new futures after
shutdown`. That is a test-harness race, not a harness bug — if you see it again,
the drain is missing or the new code leaves something in `inflight` forever.

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

## LLM failures belong to the user, not the orchestrator

An endpoint that is down, timing out, or rate-limited is infrastructure. The
orchestrator cannot restart it, and re-dispatching into it every tick is exactly
what makes a project look "stuck": turns burn, nothing lands. So:

- `AgentRuntime.run` classifies the outcome with `llm_transport = error is not
  None and retryable` and passes it to `_map_status`. Transport faults (and a
  run that reached zero turns with no error) map to `llm_unavailable`; an agent
  fault maps to `failed`. Mislabelling an outage `failed` sends the orchestrator
  off reconfiguring an agent that was never broken.
- `AgentSpec.llm_failures` is the consecutive-streak counter and
  `llm_failure_error` the message. Any run that reached the model resets both,
  even if the agent then failed for its own reasons.
- Retries **resubmit the same prompt verbatim**. Never inject "the model timed
  out" / "returned nothing" into the agent conversation: the agent reacts to an
  infrastructure fault it cannot fix (apologises, re-plans, spends budget on the
  endpoint). The old nudge-message behaviour is asserted against in
  `test_retry_resubmits_the_same_prompt_and_never_mentions_the_error`.
- Alerts: `llm:<agent_id>:<failures>` is `level: error` and names the failure,
  the streak, and the blocked card. It replaces the generic
  `<role> is llm_unavailable` warning for the same event (that warning is
  skipped when `llm_failures >= 1`) — two alerts for one outage reads as two
  problems. `cycle:<n>:<err>` fires when the orchestrator cycle itself raised,
  and `endpoints_down:<names>` when every configured server is unreachable.
- Acking an `llm:` alert is the user's "I fixed the endpoint" signal:
  `_release_llm_block` resets the streak, sets the agent `idle`, and returns the
  card to `ready`. Without it the card is blocked forever, because the
  orchestrator is told not to touch it.
- `Orchestrator.detect()` skips `blocked_cause == "llm"` cards (and
  `llm_unavailable` agents) with a "DO NOT ACT ON IT" note instead of the
  recovery menu — no re-dispatch/reconfigure/extend/split guidance. A user
  directive still outranks this (it may be the news the endpoint is back), so
  the outage branch is an `elif` after the directive branch, not before it.
- `llm_failures` is deliberately separate from `last_error`: a run that reached
  the model and then failed keeps `last_error` but clears the LLM streak.

## Orchestrator re-entrancy

There is no such thing as a queued orchestrator cycle. `run_cycle()` claims
`_cycle_lock` with `acquire(blocking=False)` and returns `None` if it cannot,
then re-checks `runtime.llm_busy("orchestrator")` **while holding the lock** so
no window opens between the check and the start. Both halves of the rule matter:
a running cycle, and a prompt that is queued waiting for an LLM slot, is already
committed to a briefing built from the board as it looked earlier — a second
cycle would act on stale state and then overwrite the fresher decisions.
`orchestrator_busy()` exposes the reason for `/api/tick`, which refuses rather
than stacking. Do not "fix" a missed cycle by queueing one: the next tick
re-detects from current state, which is the point.

## Kanban done column

`DONE_VISIBLE = 20` in `app.js`; anything older folds into a summary **card**
(`.task.done.summary`, not a bare button) that states how many are hidden. Its
click must be matched with `closest("[data-more-done]")` — the card holds inner
spans, and `t.dataset.moreDone` would miss a click on one and fall through to
the task-detail handler with no task id.

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

## Active lights (task cards + agent hierarchy)

- `runtime.inflight` is the **only** authoritative "is this agent running right
  now" signal. Do not use `agent.status == "running"`: it is a persisted field
  that a crash mid-run leaves stale, which would pin a card's light on forever.
- `AgentRuntime._inflight` is a **refcount map, not a set**. `submit()` marks an
  agent and the `run()` it dispatches marks the same agent again; with a set the
  inner unmark would clear it while the outer dispatch is still going, flickering
  the light and letting `remove project` slip past the running-agent guard. The
  nesting depth is asserted in `test_inflight_refcount_survives_nested_submit_and_run`.
- `run()` does the marking (not just `submit()`), so the orchestrator's own cycle
  and any synchronous `run_agent` also light up. `submit()` still does its own
  check-and-mark inside one lock — that guard is what prevents double dispatch.
- API exposes it as a boolean `active` on `_task_json`, `_agent_json`, and every
  `_agent_tree` node. A card's light follows its `assigned_agent_id`, so one agent
  working lights every card it is linked to. Unassigned cards stay dark.
- The frontend contract is asserted by `test_ui_assets_wire_the_light_and_collapsed_done_cards`
  (it greps `app.js`/`index.html`). If you rename `active` or the `.light` classes,
  that test fails on purpose — update it deliberately.

## Agent archival (finished specialists leave the hierarchy)

- `AgentSpec.archived_at` marks a specialist that finished its work. It is
  **archived, not deleted**: cost, tokens, and transcript stay queryable, and
  assigning it a new card (`_reactivate()` in `tools.py`) puts it back on the
  live tree instead of cloning a fresh specialist for the same expertise.
- `Orchestrator.sweep_finished_agents()` runs every tick. It archives an agent
  only when it is genuinely finished: retired, or its assigned card is `done`;
  not in flight; no live children; not the reviewer of record on an open card;
  and not a specialist created but not yet used (that exists for work still
  coming). An agent linked to no card at all is left alone unless it is `done`.
- The sweep **cascades**: it repeats until a pass archives nothing, because
  archiving the last child is what makes its parent finish. Without the loop a
  depth-3 subtree takes one tick per level to clear.
- `Store.project_agents(pid, include_archived=...)` is the single filter point.
  The live tree hides archived agents; `/api/state?show_archived=1` and the
  header "Archived" toggle reveal them, and each project reports
  `archived_agents` so the UI can say how many are hidden.
- Rationale: otherwise a long-running project accumulates a wall of finished
  agents that hides the few actually working.

## Done-column cards

- `TaskSpec.finished_at` is stamped inside `Store.upsert_task()` on the
  transition *into* `done` and cleared on any move out of it, so rework gets a
  fresh stamp. Do not stamp it at the individual move sites — there are a dozen.
- `Store.board()` returns the `done` column **newest-first** so the UI can keep
  just the recent ones. `app.js` shows `DONE_VISIBLE = 10` with a
  "Show N earlier of M done" / "Collapse done" toggle (`state.showAllDone`).
- Done cards render **title only** (no expertise/turns/cost meta) plus their light.
  Rationale: a landed card has no live counters worth scanning, and the column is
  usually the busiest one. Detail view still has everything; the card stays clickable.
- A card must never carry `blocked_reason` while not in the `blocked` column.
  `record_review` clears the block fields on approval, and `Store._normalize_loaded_tasks()`
  sweeps legacy state on every load (older builds left the reason behind, so 14 of
  21 done cards advertised blockers they had resolved). The sweep runs after
  `_load_records`, not in `_repair_task` — that hook only fires when validation
  *fails*, and this data is valid, just inconsistent.


## LLM action queue

- `AgentRuntime._llm_queue` holds one `LLMQueueEntry` per agent action, created
  **before** the concurrency gate is acquired. That ordering is the whole point:
  an action that has a prompt but no slot is visible as `queued`, and flips to
  `executing` only once it holds the slot and is actually in the model.
- Entries are removed when the run returns, so the queue is live state only —
  never persisted, never historical. `llm_queue()` returns **request order**
  (ascending `seq`); the newest-left display is a frontend choice
  (`.reverse()` in `renderQueue`), not the API contract.
- The executing lane can hold more than one entry: the gate limit is
  `max_concurrent_agents`, so several agents can be in the model at once.
- The strip polls `/api/projects/{id}/queue` every 1.5 s and patches its own
  container. Do not fold it into `/api/state`'s 5 s poll: the point is to show
  what the model is doing *right now*, and a 5 s board poll is too coarse.
  `/api/state` still carries `llm_queue` for the initial paint.

## Project-level user input

- `Project.user_messages` is the queue for "add information / new request" from
  the dashboard. It is separate from `TaskSpec.user_directives`: a directive
  steers an existing card's worker, a project message creates new scope.
- The cycle **consumes** the messages when it builds the briefing, and puts them
  back if the run did not reach the model (`llm_unavailable` / `failed`) or
  raised. Consuming without restoring replays nothing; restoring on success would
  re-ask the orchestrator to do the same work every tick.
- Posting a message to a `ready_for_delivery` / `delivered` project must reopen
  it (`status = "active"`, `delivery_signaled = False`) — `run_cycle()` skips
  finished projects, so without the reopen the button silently does nothing
  after delivery.
- `prompts.orchestrator_briefing` passes them as `user_input_pending`, newest
  first, because the orchestrator reads top-down and the latest intent wins.

## Blocked cards are re-evaluated every tick

- Every card in `blocked` gets a `RECOVERY DECISION` briefing line on every
  cycle, whatever wrote the block — not just budget exhaustion. It names the
  three moves: more turns, re-specify the card/agent, or split the card.
- `block_strikes` counts consecutive cycles blocked on the **same**
  `blocked_reason`; `block_strike_reason` is what the run is counting, so a
  changed blocker restarts at 1 and leaving the column clears both
  (`Orchestrator.reconcile`). At 3+ the briefing escalates and tells the
  orchestrator the previous fix already failed.
- Splits are traceable via `TaskSpec.parent_task_id` (settable on
  `create_task`). Without it a split loses the audit trail of why three small
  cards replaced one big one.

## The queued lane needs a pool bigger than the gate

- `submit()` puts work on the `ThreadPoolExecutor`; the LLM queue entry is only
  created inside `run()`, once the gate slot is held. If the pool ceiling equals
  the gate limit, every dispatched run immediately gets a pool thread and then
  blocks *inside the pool's own `_work_queue`*, where nothing can see it — the
  dashboard's `queued` lane stays empty no matter how much work is waiting.
- Fix: pool ceiling = `gate.limit + llm_queue_depth`. The surplus threads sit on
  the gate and are visible as `status == "queued"`. The gate, not the pool, is
  what bounds real LLM concurrency.
- `max_parallel_workers` counts **worker** slots only. The gate limit is
  `max_parallel_workers + 1`: the extra slot is reserved for the orchestrator,
  which holds one for the whole cycle while it decides what to dispatch.
  Counting the orchestrator in the config value silently costs a worker slot and
  caps real concurrency at `max_parallel_workers - 1`.
- Because of that, anything that reads or writes the limit must go through
  `AgentRuntime.set_worker_slots()` / `capacity()["worker_slots"]`, never
  `gate.limit` directly. The API once reported `gate.limit` as
  `max_parallel_workers`, so each settings round-trip drifted the limit by one.
  `/api/config` now reports `max_parallel_workers` (config truth) and
  `llm_slots` (gate truth) as separate fields.
- `capacity()["worker_slots"]` excludes the reserved slot whether or not the
  orchestrator currently holds it — otherwise the number the orchestrator plans
  against changes mid-cycle.
- Pool resizing is **grow-only**: a CPython worker thread only exits on the
  shutdown sentinel and never re-checks `_max_workers`, so lowering the ceiling
  cannot reclaim live threads. Surplus threads parked on the queue are harmless.

## Multiple LLM servers (`llm_servers`)

- `HarnessConfig.servers()` returns `llm` first, then `llm_servers`, deduped by
  `(base_url, route())`. Primary-first means a config with no `llm_servers`
  needs no special case anywhere.
- `LLMServerConfig.model` is the **literal model key** for that endpoint, and
  two different strings are derived from it. Do not conflate them:
  * `wire_model()` → the value of `model` in the request body. Always verbatim
    (`qwen3:30b`), because every endpoint is addressed over the OpenAI chat
    completions protocol and Ollama matches that string against its own model
    list: a prefixed key is a hard 404, `model 'ollama_chat/qwen3:30b' not found`.
  * `route()` → a litellm model string (`ollama_chat/qwen3:30b`), used only when
    an SDK `LLM` addresses an endpoint **directly** (the no-router fallback),
    where litellm picks its client off the prefix. Never a wire value.
  `extra_body` is merged into the body the router forwards, and into
  `litellm_extra_body` for the direct path (Ollama knobs: `num_ctx`, `keep_alive`).
- Every LLM request goes through `LLMRouter` (`agent_harness/llm_router.py`), a
  continually running internal HTTP service on a loopback port the OS picks
  (`router_port: 0`). Agents never address an endpoint directly: `build_agent`
  points `llm.base_url` at `router.base_url` for both the agent and its
  condenser. There is no rotator and no per-conversation server binding.
- The agent's model string is the placeholder `openai/harness-routed`. It MUST
  keep the `openai/` prefix: litellm resolves the provider from the model string
  before it opens a connection, so an unprefixed name fails locally with
  "LLM Provider NOT provided" and the request never reaches the router. The
  router rewrites `body["model"]` to `srv.wire_model()` per endpoint, merges
  that endpoint's `extra_body`, and strips keys the endpoint cannot accept
  (`body_keys_to_drop()`). The SDK adds `reasoning_effort: "high"` to every
  request; Ollama's OpenAI layer maps it to its own `think` option and answers
  400 `think value "high" is not supported for this model`, so it is dropped for
  `ollama`/`ollama_chat` unless that server sets `reasoning_effort` itself.
  `drop_body_keys` overrides the rule for any endpoint.
- Dispatch is availability-based, never pre-assigned. A request sits in the
  queue with `server: None` until an endpoint is free; the tie-break between
  equally-idle endpoints is "least recently assigned" (stable, spreads
  sequential load). Concurrency per endpoint is `llm_server_concurrency`.
  Because nothing is bound up front, a fast endpoint naturally takes more load
  than a slow one.
- Failover split, and why: a **transport** fault (connect/timeout) is absorbed
  by the router, which re-queues the request for the next free endpoint
  (`failed_over: true`). An **HTTP status >= 400** is relayed verbatim instead,
  because the caller's client must see the real code to do its own handling
  (429 -> litellm rate-limit path). Only the endpoint is parked. Retrying a
  request that already streamed bytes would corrupt the response, so failover
  stops once `req.responded` is set.
- A 429 does not park an endpoint (slow down, not down). 5xx / unreachable does,
  for `min(max(15, upstream_timeout), 600)`s.
- Attribution travels in request headers (`x-harness-agent`, `-role`, `-kind`,
  `-project`, `-task`) so router cards can be scoped to a project. The response
  carries `x-harness-router: <endpoint>`.
- Clock discipline: all cooldown/in-flight math uses `time.monotonic()`;
  `time.time()` is only for display timestamps. Mixing them produced a tok/s
  rate off by the epoch offset.
- `test_prompt_config` deliberately bypasses the queue (an operator who just
  added a server wants an answer about *that* endpoint, not a fleet slot) and
  accepts a config that is not saved yet. An unsaved definition gets only its
  own typed key — `resolved_api_key(use_env=False)` — so the harness-wide key is
  never sent to a URL someone merely typed into a form.
- The configured primary is always in the router fleet; `servers()` puts `llm`
  first, so a config with no `llm_servers` needs no special case.

## Live reconfiguration at the start of every tick

- `Orchestrator.run_cycle()` calls `reload_config_if_changed()` **first**, before
  any scheduling. The live `HarnessConfig` object is mutated field-by-field,
  never rebound: the API, `tools.init_context` and the runtime all hold that same
  instance.
- Change detection is a **sha256 of the file contents**, not mtime/size. FS
  timestamp granularity is coarse enough for a write to keep the same mtime, and
  `max_parallel_workers: 2 -> 6` leaves the byte count identical — both miss the
  edit.
- `note_config_written()` is called by the API after it patches config.json.
  Without it the loop reads its own write as an external edit and emits a
  spurious `config_reloaded` every tick.
- Shrinking `max_parallel_workers` drains: the gate limit drops for new runs, but
  in-flight agents are never interrupted.
- Server changes are pushed into the router with `router.sync_servers()`; no
  conversation is touched, because none of them holds an endpoint. This removed
  `rebind_stale_servers()` entirely, along with the idle-vs-inflight drain
  distinction. `server_concurrency` is part of the config fingerprint, so
  changing it alone still triggers a reload.
- A malformed config.json must not stop the loop: keep the previous settings,
  emit `config_reload_failed`, carry on.
- Config changes reach the model as `CONFIG CHANGED` / `CAPACITY NOW` lines in
  the per-project attention list (broadcast to every project — capacity is
  shared), plus a `config_reloaded` event in the feed.

## Test boundaries

- Stub at the **conversation** (`open_conversation` returning a fake with
  `send_message`/`run`/`state.events`), not `_run_guarded`, when testing retry,
  timeout-widening, or server-health behaviour — that logic lives *inside*
  `_run_guarded`, so stubbing it bypasses the code under test.
- Test stubs for `_run_guarded` take `**_kw`: `server_name` is keyword-only
  specifically so existing 4-positional stubs keep working.
