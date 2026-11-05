const $ = (s, r = document) => r.querySelector(s);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { data: null, openProjects: new Set(), seenProjects: new Set(),
                tab: "board", queues: {}, pendingFiles: {}, pendingForms: {},
                router: null, showArchived: false, showAllDone: false };

const money = (v) => "$" + (Number(v || 0)).toFixed(4).replace(/0+$/, "").replace(/\.$/, ".00");

// A card with no worker has no spend to report; showing $0.00 there reads as a
// bug, so distinguish "nobody has worked on this" from "worked for free".
const cardCost = (t) =>
  t.assigned_agent_id ? money(t.cost_usd) : '<span class="muted">no spend</span>';

function agoLabel(ts) {
  if (!ts) return "never";
  // Cycle timestamps are epoch seconds; store records (revisions, events) are ISO.
  const then = typeof ts === "number" ? ts : Math.floor(new Date(ts).getTime() / 1000);
  if (!Number.isFinite(then)) return "unknown";
  const secs = Math.max(0, Math.floor(Date.now() / 1000) - then);
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

function turnsClass(used, limit) {
  if (used >= limit) return "turns full";
  if (used >= limit * 0.8) return "turns hot";
  return "turns";
}

function renderHeader(d) {
  const ab = $("#archivedBtn");
  if (ab) {
    const hidden = (d.projects || []).reduce((n, x) => n + (x.archived_agents || 0), 0);
    ab.textContent = "Archived: " + (state.showArchived ? "shown" : "hidden")
      + (hidden ? ` (${hidden})` : "");
    ab.classList.toggle("active", state.showArchived);
  }
  const ver = d.meta.harness_version || "unknown";
  const stored = d.meta.harness_version_stored;
  $("#versionPill").textContent = "v" + ver;
  $("#versionPill").title = `Harness build ${ver}` +
    (stored && stored !== ver ? ` — state.json last written by ${stored}` : "");
  $("#modelPill").innerHTML = `model: <b>${esc(d.config.model)}</b>`;
  $("#tickPill").innerHTML = `tick: <b>${d.config.tick_seconds}s</b>`;
  $("#budgetPill").innerHTML = `sub-agent budget: <b>${d.config.sub_agent_turn_limit} turns</b>`;
  const inflight = d.meta.inflight || [];
  const cyc = d.meta.cycle || {};
  let cycleTxt;
  if (cyc.running) {
    cycleTxt = `cycle <b>running</b> (${agoLabel(cyc.started_at)} in)`;
  } else if (cyc.last_completed_at) {
    cycleTxt = `last cycle <b>${agoLabel(cyc.last_completed_at)}</b>`;
  } else {
    cycleTxt = "last cycle <b>never</b>";
  }
  if (cyc.count) cycleTxt += ` · #${cyc.count}`;
  // Only call out the attempt when it diverges from the last success: that gap
  // is the signature of a loop that is running but failing.
  const diverged = cyc.last_attempt_at && cyc.last_completed_at
    ? cyc.last_attempt_at - cyc.last_completed_at > 2
    : !!cyc.last_attempt_at;
  if (diverged) {
    cycleTxt += ` · last attempt ${agoLabel(cyc.last_attempt_at)}`;
  }
  const pill = $("#runPill");
  pill.innerHTML = `running: <b>${inflight.length}</b> · ${cycleTxt}`;
  pill.classList.toggle("running", !!cyc.running);
  pill.classList.toggle("stale", !!cyc.stale);
  pill.title =
    `Loop started ${agoLabel(cyc.loop_started_at)} · tick ${cyc.tick_seconds}s\n` +
    `Last attempt ${agoLabel(cyc.last_attempt_at)} · last success ${agoLabel(cyc.last_completed_at)}` +
    (cyc.stale ? "\nWARNING: the loop has not completed a cycle recently." : "");
  $("#costPill").innerHTML =
    `cost: <b>${money(d.total_cost_usd)}</b>` +
    ` <span class="muted">(${money(d.project_cost_usd)} projects` +
    ` + ${money(d.orchestrator_cost_usd)} orchestrator)</span>`;
  $("#costPill").title =
    `Harness total = every agent still in the store.\n` +
    `Projects: ${money(d.project_cost_usd)}\n` +
    `Shared orchestrator overhead: ${money(d.orchestrator_cost_usd)}\n` +
    `Removed projects keep their spend in the activity feed.`;
  const llm = d.meta.llm || {};
  const unhealthy = (llm.empty_responses || 0) + (llm.transient_errors || 0) > 0;
  const router = d.meta.llm_router || {};
  const fleet = router.servers || [];
  const cap = d.meta.capacity || {};
  const rc = router.counts || {};
  const down = fleet.filter((s) => s.reachable === false || s.cooling_down);
  $("#llmPill").innerHTML =
    `llm: <b>${llm.timeout ?? "?"}s timeout</b>` +
    ` · <b>${fleet.length - down.length}/${fleet.length}</b> llms` +
    (rc.queued ? ` · <b>${rc.queued}</b> req queued` : "") +
    (unhealthy
      ? ` · <b>${(llm.empty_responses || 0) + (llm.transient_errors || 0)}</b> retried`
      : "");
  $("#llmPill").classList.toggle("warn", unhealthy || down.length > 0);
  $("#llmPill").title =
    `LLM router: agents only produce a prompt; the router picks the endpoint.\n` +
    `Selection: ${(cap.router_selection || "throughput")} - ` +
    (cap.router_selection === "round_robin"
      ? "endpoints share the load evenly.\n\n"
      : "among endpoints with a free slot, the one that should answer soonest "
        + "wins, so a fast endpoint takes more of the load.\n\n") +
    fleet.map((s) => `  ${s.name} · ${s.model} @ ${s.base_url}\n` +
      `    reachable=${s.reachable === null ? "unprobed" : s.reachable}` +
      ` · ${s.tokens_per_second ?? "?"} tok/s · ${s.total_tokens ?? 0} tokens` +
      ` · ${s.in_flight ?? 0} in flight` +
      (s.cooling_down ? ` · cooling ${s.cooldown_seconds}s` : "")).join("\n") +
    `\nWorker slots: ${cap.workers_active ?? 0}/${cap.worker_slots ?? 0} working` +
    ` (${cap.slots_free ?? 0} free; the orchestrator holds one slot of its own)` +
    `\nRouter requests: ${rc.queued ?? 0} queued,` +
    ` ${(rc.awaiting ?? 0) + (rc.generating ?? 0)} sent`;
}

/* The LLM router: one card per request it is holding.
   `queued`   -> waiting for any endpoint to free up (no endpoint chosen yet)
   `awaiting` -> sent; waiting on the model to start answering
   `generating` -> tokens are actively arriving
   Endpoint cards are pinned to the far right, one per configured LLM. */
function renderRouter(rt, projectId) {
  const all = (rt && rt.requests) || [];
  const allRecent = (rt && rt.recent) || [];
  const servers = (rt && rt.servers) || [];
  const counts = (rt && rt.counts) || {};
  // Request cards are scoped to the project the strip sits under; the endpoint
  // cards on the right are the whole fleet, since the router is shared.
  const mine = (r) =>
    !projectId || r.project_id === projectId || r.project_id === null;
  const reqs = all.filter(mine);
  const recent = allRecent.filter(mine);
  const STATUS = { queued: "queued", awaiting: "awaiting",
                   generating: "generating", failed: "failed" };
  const card = (r) => `<div class="rcard ${esc(r.status)}" data-req="${esc(r.id)}"
      title="${esc(r.agent_role || "?")} · ${esc(r.kind || "?")} · ${
        r.status === "queued"
          ? `waiting ${r.queued_seconds}s for any LLM to free up`
          : `on ${esc(r.server || "?")} for ${r.running_seconds ?? 0}s`}">
    <span class="rtstat">${esc(STATUS[r.status] || r.status)}${
      r.failed_over ? " · failed over" : ""}</span>
    <span class="rtwho">${esc(r.agent_role || r.agent_id || "request")}</span>
    <span class="rtmeta">${r.status === "queued"
        ? `waiting ${r.queued_seconds}s`
        : `${esc(r.server || "?")} · ${r.running_seconds ?? 0}s`} · ${
        r.prompt_chars} chars${
        r.completion_tokens ? ` · ${r.completion_tokens} tok` : ""}</span>
  </div>`;
  const scard = (s) => {
    const bad = s.reachable === false;
    const cls = bad ? "down" : (s.cooling_down ? "cooling" : "");
    const dot = bad ? "bad" : (s.cooling_down ? "warn" : "on");
    return `<div class="scard ${cls}" title="${esc(s.base_url)} · model key ${
        esc(s.model)}\n${s.requests_ok ?? 0} ok / ${s.requests_failed ?? 0} failed` +
      `${s.last_error ? `\nlast error: ${esc(s.last_error)}` : ""}\n` +
      `${s.prompt_tokens ?? 0} input + ${s.total_tokens ?? 0} output tokens over ${
        s.generation_seconds ?? 0}s of generation">
      <span class="shead"><span class="light ${dot}"></span>
        <span class="sname">${esc(s.name)}</span></span>
      <span class="srate">${s.tokens_per_second ?? "—"} tok/s</span>
      <span class="stok">${fmtTokens(s.total_tokens)} tokens · ${
        s.in_flight ?? 0} in flight${
        s.cooling_down ? ` · cooling ${s.cooldown_seconds}s` : ""}</span>
    </div>`;
  };
  const pending = reqs.filter((r) => r.status === "queued").reverse();
  const sent = reqs.filter((r) => r.status !== "queued");
  const finished = recent.slice(-4).reverse();
  return `<div class="rtbar">
    <div class="rtlane">
      <span class="rtlabel">router queued</span>
      <div class="rrow">${pending.map(card).join("")
        || '<span class="muted">none</span>'}</div>
    </div>
    <div class="rtlane">
      <span class="rtlabel">sent / generating</span>
      <div class="rrow">${sent.map(card).join("")
        || '<span class="muted">idle</span>'}</div>
    </div>
    ${finished.length ? `<div class="rtlane"><span class="rtlabel">just done</span>
      <div class="rrow">${finished.map((r) =>
        `<div class="rcard ${esc(r.status)}" data-req="${esc(r.id)}"
           title="${esc(r.error || `answered by ${r.server || "?"}`)}">
           <span class="rtstat">${esc(r.status)}</span>
           <span class="rtwho">${esc(r.agent_role || "request")}</span>
           <span class="rtmeta">${r.completion_tokens ?? 0} tok · ${
             r.running_seconds ?? 0}s</span></div>`).join("")}</div></div>` : ""}
    <div class="rtlane srv">
      <span class="rtlabel">llms</span>
      <div class="rrow">${servers.map(scard).join("")
        || '<span class="muted">none configured</span>'}</div>
    </div>
  </div>
  <div class="muted" style="font-size:10px;margin-top:3px">
    ${counts.queued ?? 0} waiting on an endpoint · ${(counts.awaiting ?? 0) + (counts.generating ?? 0)} sent
    · ${counts.server_slots_in_use ?? 0}/${counts.server_slots ?? 0} endpoint slots busy
    · ${counts.served ?? 0} served since start</div>`;
}

function fmtTokens(n) {
  if (n === null || n === undefined) return "—";
  if (n >= 1000000) return (n / 1000000).toFixed(1) + "M";
  if (n >= 1000) return (n / 1000).toFixed(1) + "k";
  return String(n);
}

function renderAlerts(d) {
  const alerts = d.alerts || [];
  $("#alertsHead").hidden = alerts.length === 0;
  $("#alerts").innerHTML = alerts
    .map(
      (a) => `<div class="alert ${a.level}"><div class="row">
        <span class="msg">${esc(a.message)}</span>
        <button data-ack="${esc(a.id)}" title="Dismiss this alert">Acknowledge</button>
      </div></div>`
    )
    .join("");
}

// What will clear each kind of block, so a blocked card tells the user whether
// they need to act or the harness will resolve it on the next cycle.
const BLOCK_FIX = {
  budget: "grant more turns to re-run",
  llm: "acknowledge the LLM alert to put this card back in the queue",
  worker: "the worker will re-check the blocker when re-dispatched",
  agent_failed: "re-dispatch, or reconfigure the agent",
  dependency: "clears itself when upstream tasks finish",
  manual: "needs your direction — add one below",
};

function taskCard(t) {
  const done = t.column === "done";
  return `<div class="task${done ? " done" : ""}${t.active ? " active" : ""}" data-task="${t.id}">
    <div class="t"><span class="light ${t.active ? "on" : "off"}" title="${
      t.assigned_agent_id
        ? `${esc(t.agent_role || t.assigned_agent_id)} is ${t.active ? "working on this now" : "not running"}`
        : "No worker assigned"}"></span>${esc(t.title)}</div>
    ${done ? "" : `<div class="m">
      <span class="tag">${esc(t.expertise)}</span>
      <span class="${turnsClass(t.turns_used, t.turn_limit)}">${t.turns_used}/${t.turn_limit} turns</span>
      ${t.review_rounds ? `<span class="tag">review ×${t.review_rounds}</span>` : ""}
      ${t.review_verdict ? `<span class="tag">${esc(t.review_verdict)}</span>` : ""}
      ${t.assigned_agent_id ? `<span class="tag">🤖 ${esc(t.agent_role || t.assigned_agent_id)}</span>` : ""}
      <span class="kv cost" title="${t.assigned_agent_id
        ? `Spend by ${esc(t.agent_role || t.assigned_agent_id)} on this card`
        : "No worker assigned yet"}">${cardCost(t)}</span>
    </div>`}
    ${t.blocked_reason ? `<div class="blocked-reason">⚠ ${esc(t.blocked_reason)}
      <span class="fix">${esc(BLOCK_FIX[t.blocked_cause] || "will be re-checked next cycle")}</span>
    </div>` : ""}
  </div>`
}

function renderTree(nodes) {
  if (!nodes.length) return '<div class="muted">No agents yet.</div>';
  const walk = (n, root = false) => {
    const kids = (n.children || []).map((c) => walk(c)).join("");
    return `<div class="node ${root ? "root" : ""}">
      <div class="row" data-agent="${n.id}">
        <span class="light ${n.active ? "on" : "off"}" title="${
          n.active ? "Running right now" : "Not running"}"></span>
        <span class="dot ${esc(n.status)}" title="status: ${esc(n.status)}"></span>
        <span class="role">${esc(n.role)}</span>
        <span class="spec">${esc(n.specialty)}</span>
        <span class="kv">${n.turns_used}/${n.turn_limit} turns · ${esc(n.status)}</span>
        ${n.task_id ? `<span class="tag">task ${esc(n.task_id)}</span>` : ""}
        ${n.tokens ? `<span class="kv">${(n.tokens / 1000).toFixed(1)}k tok</span>` : ""}
        <span class="kv cost" title="${n.shared ? "Shared orchestrator spend — counted as harness overhead, not in this project's total" : ""}">${money(n.cost_usd)}${n.shared ? " (shared)" : ""}</span>
        ${n.archived ? `<span class="tag" title="Finished and off the live hierarchy; assign it work again to bring it back">archived</span>` : ""}
      </div>
      ${kids}
    </div>`;
  };
  return `<div class="tree">${nodes.map((n) => walk(n, true)).join("")}</div>`;
}

/* Agent actions waiting on the LLM. The API returns request order (oldest
   first); the strip shows newest on the left and pins the executing action to
   the right, so reading left-to-right reads as "just queued ... in flight". */
/* Where the user adds information or a new request to a running project. The
   orchestrator reads it on its next tick and turns it into cards and agents. */
function renderMessageBar(p) {
  const pending = (p.user_messages || []).slice(-3).reverse();
  const inputs = (p.input_files || []);
  return `<div class="msgbar">
    <form class="msgform" data-msg="${esc(p.id)}">
      <input name="text" placeholder="Add information or a new request — the orchestrator will plan it next tick" autocomplete="off">
      <label class="filepick" title="Attach files to this project. They go into the workspace under inputs/ where the project's agents can read them.">
        <input type="file" name="files" multiple data-pick="${esc(p.id)}">
        <span class="fp-btn">Attach files</span>
        <span class="fp-names" data-pick-names="${esc(p.id)}">none</span>
      </label>
      <button type="submit">Add to project</button>
    </form>
    ${inputs.length ? `<div class="msgpending">${inputs.map((f) =>
        `<span class="tag" title="uploaded ${esc(f.at || '')}">input: inputs/${esc(f.name)}</span>`
      ).join("")}</div>` : ""}
    ${pending.length ? `<div class="msgpending">${pending.map((m) =>
        `<span class="tag pending" title="${esc(m.at)}">pending: ${esc(m.text.slice(0, 80))}</span>`
      ).join("")}</div>` : ""}
  </div>`;
}

function renderQueue(queue) {
  const items = queue || [];
  if (!items.length)
    return '<div class="queue-empty muted">Nothing waiting on the LLM.</div>';
  const executing = items.filter((e) => e.status === "executing");
  const waiting = items.filter((e) => e.status !== "executing").reverse();
  const card = (e) => `<div class="qcard ${esc(e.status)}" data-queue="${esc(e.id)}"
      title="${esc(e.agent_role)} · ${esc(e.kind)} · ${
        e.status === "executing"
          ? `sending to the LLM for ${e.running_seconds ?? 0}s`
          : `waiting ${e.waiting_seconds}s for a slot`}">
    <span class="qkind ${esc(e.kind)}">${esc(e.kind)}</span>
    <span class="qrole">${esc(e.agent_role)}</span>
    <span class="qmeta">${e.status === "executing"
      ? `running ${e.running_seconds ?? 0}s`
      : `queued ${e.waiting_seconds}s`} · ${e.prompt_chars} chars</span>
  </div>`;
  return `<div class="queue">
    <div class="qlane">
      <span class="qlabel">queued</span>
      <div class="qrow">${waiting.map(card).join("") || '<span class="muted">none</span>'}</div>
    </div>
    <div class="qlane exec">
      <span class="qlabel">executing</span>
      <div class="qrow">${executing.map(card).join("") || '<span class="muted">idle</span>'}</div>
    </div>
  </div>`;
}

async function showQueuePrompt(entryId) {
  const res = await fetch(`/api/queue/${encodeURIComponent(entryId)}/prompt`);
  const body = await res.json();
  if (!res.ok) {
    // The entry only lives while the run is pending or in flight, so it is
    // normal for a click to arrive just after the LLM finished with it.
    $("#detailBody").innerHTML = `<h3>Action already dispatched</h3>
      <div class="muted">${esc(body.detail || "That action is no longer in the queue.")}</div>`;
  } else {
    $("#detailBody").innerHTML = `<h3>Prompt waiting for the LLM</h3>
      <div class="muted">${esc(body.id)}</div>
      <div class="sec"><h5>Prompt</h5><pre>${esc(body.prompt)}</pre></div>`;
  }
  $("#detail").showModal();
}

// A long-running project accumulates done cards without bound, and they all
// look alike. Show the 20 most recent and fold the rest into a summary card
// that states how many are hidden.
const DONE_VISIBLE = 20;

function doneSummaryCard(hidden, total) {
  return `<div class="task done summary" data-more-done="1"
      title="Click to ${state.showAllDone ? "hide" : "show"} the older done cards">
    <div class="t"><span class="light off"></span>${
      state.showAllDone ? "Collapse done" : `${hidden} earlier done card${
        hidden === 1 ? "" : "s"}`}</div>
    <div class="m"><span class="tag">${
      state.showAllDone ? `all ${total} shown` : `${hidden} of ${total} hidden`
    }</span><span class="kv">${state.showAllDone
      ? "click to collapse" : "click to expand"}</span></div>
  </div>`
}

function renderBoard(p) {
  return `<div class="kanban">${p.columns_order
    .map((col) => {
      const items = p.board[col] || [];
      const overflow = col === "done" && items.length > DONE_VISIBLE;
      const shown = overflow && !state.showAllDone
        ? items.slice(0, DONE_VISIBLE) : items;
      const summary = overflow
        ? doneSummaryCard(items.length - DONE_VISIBLE, items.length) : "";
      return `<div class="col"><h4><span>${col}</span><span>${items.length}</span></h4>
        ${shown.map(taskCard).join("")}${summary}</div>`;
    })
    .join("")}</div>`;
}

function renderRevisions(p) {
  const revs = p.revisions || [];
  const items = revs.length
    ? `<ol>${revs.slice(0, 12).map((r) =>
        `<li><b>${esc(r.version)}</b> · ${esc(r.reason)} <span class="kv">${esc(agoLabel(r.at))}</span></li>`).join("")}</ol>`
    : '<div class="muted">No revisions recorded yet.</div>';
  return `<div class="revisions">
    <h5 style="margin:12px 0 6px;font-size:12px;text-transform:uppercase;color:var(--dim)">
      Version history — current ${esc(p.version)}</h5>
    ${items}
    <div style="margin-top:8px">
      <button data-edit="${p.id}">Edit scope (bumps version)</button>
    </div>
  </div>`;
}

function renderProjects(d) {
  capturePendingForms();
  const html = d.projects
    .map((p) => {
      const open = state.openProjects.has(p.id) ? "open" : "";
      return `<details class="card" ${open} data-project="${p.id}">
        <summary>
          <span class="pname">${esc(p.name)}</span>
          <span class="ver" title="Project version — starts at 0.1, increments on every revision, major bumps on delivery">v${esc(p.version)}</span>
          <span class="badge ${esc(p.status)}">${esc(p.status)}</span>
          <span class="goal">${esc(p.goal)}</span>
          <span class="spacer"></span>
          <span class="kv cost" title="This project's own agents, excluding shared orchestrator overhead">project ${money(p.cost_usd)}</span>
          <span class="kv">${esc(p.id)}</span>
          <button data-pause="${p.id}">${p.status === "paused" ? "Resume" : "Pause"}</button>
          ${p.status === "ready_for_delivery" ? `<button data-deliver="${p.id}">Mark delivered</button>` : ""}
          <button class="danger" data-remove="${p.id}">Remove…</button>
        </summary>
        <div class="body">
          <div class="tabs">
            <button class="tab ${state.tab === "board" ? "active" : ""}" data-tab="board">Kanban</button>
            <button class="tab ${state.tab === "agents" ? "active" : ""}" data-tab="agents">Agent hierarchy</button>
            <span class="spacer"></span>
            <span class="kv">${esc(p.workspace)}</span>
          </div>
          ${renderMessageBar(p)}
          <div id="queue-${p.id}" class="qwrap">${renderQueue(state.queues[p.id])}</div>
          <div id="router-${p.id}" class="rtwrap">${renderRouter(state.router, p.id)}</div>
          ${p.archived_agents ? `<div class="kv" style="margin:6px 0">
            ${p.archived_agents} archived specialist(s) hidden from the hierarchy —
            use the "Archived" button in the header to show them.</div>` : ""}
          <div class="grid2">
            <div>${state.tab === "board" ? renderBoard(p) : renderTree(p.tree)}</div>
            <div>
              <h5 style="margin:12px 0 6px;font-size:12px;text-transform:uppercase;color:var(--dim)">
                ${state.tab === "board" ? "Agent hierarchy" : "Kanban"}</h5>
              ${state.tab === "board" ? renderTree(p.tree) : renderBoard(p)}
            </div>
          </div>
          ${renderRevisions(p)}
        </div>
      </details>`;
    })
    .join("");
  $("#projects").innerHTML =
    html || '<div class="muted">No projects yet — create one above.</div>';
  restorePendingForms();
}

/* The board is re-rendered from scratch every poll, which would otherwise wipe
   what the user just typed or the file they just picked — a picked FileList
   cannot be set programmatically, so it is stashed and re-attached by index. */
function capturePendingForms() {
  document.querySelectorAll("form[data-msg]").forEach((f) => {
    const text = f.querySelector("input[name=text]");
    const file = f.querySelector("input[type=file]");
    const key = f.dataset.msg;
    if (text && text.value.trim()) state.pendingForms[key] = text.value;
    if (file && file.files.length) state.pendingFiles[key] = Array.from(file.files);
  });
}

function restorePendingForms() {
  document.querySelectorAll("form[data-msg]").forEach((f) => {
    const key = f.dataset.msg;
    const text = f.querySelector("input[name=text]");
    if (text && state.pendingForms[key]) text.value = state.pendingForms[key];
    const file = f.querySelector("input[type=file]");
    if (file && state.pendingFiles[key]) {
      const dt = new DataTransfer();
      state.pendingFiles[key].forEach((fl) => dt.items.add(fl));
      file.files = dt.files;
    }
    updatePickNames(f.querySelector("[data-pick-names]"), file);
  });
}

function updatePickNames(span, input) {
  if (!span || !input) return;
  const n = input.files ? input.files.length : 0;
  if (!n) { span.textContent = "none"; return; }
  const names = Array.from(input.files).map((f) => f.name);
  span.textContent = names.length > 2
    ? `${names.slice(0, 2).join(", ")} +${names.length - 2}`
    : names.join(", ");
}

function renderFeed(d) {
  $("#feed").innerHTML = d.events
    .map(
      (e) => `<div class="ev"><span class="ts">${new Date(e.at).toLocaleTimeString()}</span>
        <span class="k">${esc(e.kind)}</span>${esc(e.message)}</div>`
    )
    .join("");
}

async function refresh() {
  const res = await fetch("/api/state?show_archived=" + (state.showArchived ? "1" : "0"));
  const d = await res.json();
  state.data = d;
  // Seed the queue from the state payload so the strip is right on first paint;
  // the faster per-project poll keeps it current between refreshes.
  const q = d.llm_queue || [];
  d.projects.forEach((p) => {
    p.columns_order = d.columns;
    state.queues[p.id] = q.filter(
      (e) => e.project_id === p.id || e.project_id === null);
    // Default a project open the first time we see it, then never touch its
    // open state again: re-seeding on every poll would pop a collapsed card
    // back open a few seconds after the user closed it.
    if (!state.seenProjects.has(p.id)) {
      state.seenProjects.add(p.id);
      if (p.status !== "delivered") state.openProjects.add(p.id);
    }
  });
  state.router = d.meta.llm_router || null;
  renderHeader(d);
  renderAlerts(d);
  renderProjects(d);
  renderFeed(d);
}

async function showTask(projectId, taskId) {
  const res = await fetch(`/api/projects/${projectId}/tasks/${taskId}`);
  const t = await res.json();
  $("#detailBody").innerHTML = `
    <h3>${esc(t.title)}</h3>
    <div class="muted">${esc(t.id)} · column <b>${esc(t.column)}</b> · ${t.turns_used}/${t.turn_limit} turns</div>
    <div class="sec"><h5>Objective</h5><pre>${esc(t.objective)}</pre></div>
    <div class="sec"><h5>Acceptance criteria</h5><pre>${esc((t.acceptance_criteria || []).join("\n") || "—")}</pre></div>
    <div class="sec"><h5>Deliverables</h5><pre>${esc((t.deliverables || []).join("\n") || "—")}</pre></div>
    <div class="sec"><h5>Worker report</h5><pre>${esc(t.result_summary || "—")}</pre></div>
    <div class="sec"><h5>Review (${esc(t.review_verdict || "pending")}, round ${t.review_rounds})</h5>
      <pre>${esc(t.review_feedback || "—")}</pre></div>
    <div class="sec"><h5>Orchestrator note</h5><pre>${esc(t.orchestrator_note || "—")}</pre></div>
    <div class="sec"><h5>Blocked</h5><pre>${t.blocked_reason
      ? `${esc(t.blocked_reason)}\ncause: ${esc(t.blocked_cause || "unknown")}\n` +
        `resolution: ${esc(BLOCK_FIX[t.blocked_cause] || "re-checked next cycle")}`
      : "not blocked"}</pre></div>
    <div class="sec"><h5>Direction for the next run</h5>
      <pre>${esc((t.user_directives || []).map((x) => `[${x.at}] ${x.text}`).join("\n") || "—")}</pre>
      <div style="display:flex;gap:8px;margin-top:8px">
        <input id="directiveText" placeholder="Tell the worker what to change…"
               style="flex:1;background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px">
        <button id="directiveSend" data-project="${esc(projectId)}" data-task="${esc(taskId)}">Send</button>
      </div></div>`;
  $("#detail").showModal();
}

async function showAgent(projectId, agentId) {
  const res = await fetch(`/api/projects/${projectId}/agents/${agentId}`);
  const a = await res.json();
  $("#detailBody").innerHTML = `
    <h3>${esc(a.role)}</h3>
    <div class="muted">${esc(a.id)} · ${esc(a.specialty)} · depth ${a.depth} · ${esc(a.status)}</div>
    <div class="sec"><h5>Turn budget</h5><pre>${a.turns_used}/${a.turn_limit} turns · ${a.runs} runs · ${a.tokens} tokens</pre></div>
    <div class="sec"><h5>Total cost</h5><pre class="cost">${money(a.cost_usd)}</pre></div>
    <div class="sec"><h5>Mission</h5><pre>${esc(a.mission)}</pre></div>
    <div class="sec"><h5>Last report</h5><pre>${esc(a.last_report || "—")}</pre></div>
    <div class="sec"><h5>Last error</h5><pre>${esc(a.last_error || "none")}</pre></div>
    <div class="sec"><h5>System prompt</h5><pre>${esc(a.system_prompt)}</pre></div>`;
  $("#detail").showModal();
}

document.addEventListener("submit", async (e) => {
  const msgForm = e.target.dataset ? e.target.dataset.msg : null;
  if (msgForm) {
    e.preventDefault();
    const text = (e.target.querySelector("input[name=text]").value || "").trim();
    const files = e.target.querySelector("input[type=file]");
    if (!text && !(files && files.files.length)) return;
    const fd = new FormData(e.target);
    const res = await fetch(`/api/projects/${msgForm}/messages`, {
      method: "POST", body: fd,
    });
    if (res.ok) {
      e.target.reset();
      delete state.pendingForms[msgForm];
      delete state.pendingFiles[msgForm];
      updatePickNames(e.target.querySelector("[data-pick-names]"),
                      e.target.querySelector("input[type=file]"));
      await refresh();
    } else {
      alert("Could not add that: " + (await res.text()));
    }
    return;
  }
  if (e.target.id !== "projForm") return;
  e.preventDefault();
  const res = await fetch("/api/projects", {
    method: "POST",
    body: new FormData(e.target),
  });
  if (res.ok) {
    e.target.reset();
    updatePickNames($("#projFileNames"), $("#projFiles"));
    await refresh();
  } else {
    alert("Could not create project: " + (await res.text()));
  }
});

// The visible "N selected" text lives in the label, not the hidden input.
document.addEventListener("change", (e) => {
  if (e.target.matches("input[type=file]")) {
    const box = e.target.closest("form[data-msg]");
    updatePickNames(
      box ? box.querySelector("[data-pick-names]") : $("#projFileNames"), e.target);
  }
});

document.addEventListener("click", async (e) => {
  const t = e.target;
  // Pause / delivered / Remove live inside <summary>, and a click on a summary
  // toggles the card natively. Suppress that so pressing Pause does not also
  // collapse the project the user is looking at.
  if (t.closest && t.closest("summary") &&
      (t.closest("button") || t.closest("input") || t.closest("select"))) {
    e.preventDefault();
  }
  if (t.id === "tickBtn") {
    await fetch("/api/tick", { method: "POST" });
    setTimeout(refresh, 1500);
    return;
  }
  if (t.id === "settingsBtn") {
    await openSettings();
    return;
  }
  if (t.id === "settingsSave") {
    await saveSettings();
    return;
  }
  if (t.id === "addServer") {
    syncServerList();
    serverList.push({ name: "", provider: "ollama_chat", base_url: "", model: "" });
    $("#serverRows").innerHTML = serverList.map(serverRow).join("");
    populateTestServers();
    return;
  }
  const rowTest = t.closest && t.closest("[data-srvtest]");
  if (rowTest) {
    await runTestPrompt(Number(rowTest.dataset.srvtest));
    return;
  }
  if (t.id === "probeServers") {
    $("#testMsg").textContent = "re-checking reachability…";
    const res = await fetch("/api/llm/router/probe", { method: "POST" });
    const body = await res.json();
    $("#testMsg").textContent = (body.servers || [])
      .map((x) => `${x.name}: ${x.reachable ? "reachable" : "unreachable"}`).join(" · ");
    return;
  }
  if (t.id === "runTest") {
    await runTestPrompt();
    return;
  }
  if (t.dataset.srvdel !== undefined) {
    syncServerList();
    serverList.splice(Number(t.dataset.srvdel), 1);
    $("#serverRows").innerHTML = serverList.map(serverRow).join("")
      || '<p class="muted">No extra servers.</p>';
    return;
  }
  if (t.id === "ackAllBtn") {
    await fetch("/api/alerts/ack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ all: true }),
    });
    await refresh();
    return;
  }
  const queueCard = t.closest && t.closest("[data-queue]");
  if (queueCard) {
    await showQueuePrompt(queueCard.dataset.queue);
    return;
  }
  const reqCard = t.closest && t.closest("[data-req]");
  if (reqCard) {
    await showRouterPrompt(reqCard.dataset.req);
    return;
  }
  if (t.dataset.ack) {
    await fetch("/api/alerts/ack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ alert_id: t.dataset.ack }),
    });
    await refresh();
    return;
  }
  if (t.dataset.remove) {
    openRemove(t.dataset.remove);
    return;
  }
  if (t.id === "removeGo") {
    await doRemove();
    return;
  }
  if (t.id === "removeCancel") {
    $("#remove").close();
    return;
  }
  if (t.id === "directiveSend") {
    const input = $("#directiveText");
    const text = (input?.value || "").trim();
    if (!text) return;
    const res = await fetch(
      `/api/projects/${t.dataset.project}/tasks/${t.dataset.task}/directives`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
    if (res.ok) {
      $("#detail").close();
      await refresh();
    } else {
      alert("Could not send direction: " + (await res.text()));
    }
    return;
  }
  if (t.dataset.pause) {
    await fetch(`/api/projects/${t.dataset.pause}/pause`, { method: "POST" });
    refresh();
    return;
  }
  if (t.dataset.deliver) {
    await fetch(`/api/projects/${t.dataset.deliver}/mark-delivered`, { method: "POST" });
    refresh();
    return;
  }
  if (t.dataset.edit) {
    const p = (state.data?.projects || []).find((x) => x.id === t.dataset.edit);
    if (!p) return;
    const goal = window.prompt("Goal (leaving it unchanged records no revision):", p.goal);
    if (goal === null) return;
    const constraints = window.prompt("Constraints:", p.constraints || "");
    if (constraints === null) return;
    const res = await fetch(`/api/projects/${p.id}/revise`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, constraints }),
    });
    if (!res.ok) {
      alert("Could not update scope: " + (await res.text()));
    }
    refresh();
    return;
  }
  // The summary card is a .task div, so match on the container: a click that
  // lands on its inner tag span must not fall through to the task-detail
  // handler with no task id.
  if (t.closest && t.closest("[data-more-done]")) {
    state.showAllDone = !state.showAllDone;
    renderProjects(state.data);
    return;
  }
  if (t.id === "archivedBtn") {
    state.showArchived = !state.showArchived;
    refresh();
    return;
  }
  if (t.dataset.tab) {
    state.tab = t.dataset.tab;
    renderProjects(state.data);
    return;
  }
  const card = t.closest(".task");
  if (card) {
    const proj = t.closest("[data-project]");
    showTask(proj.dataset.project, card.dataset.task);
    return;
  }
  const node = t.closest("[data-agent]");
  if (node) {
    const proj = t.closest("[data-project]");
    showAgent(proj.dataset.project, node.dataset.agent);
    return;
  }
  const sum = t.closest("summary");
  if (sum) {
    const det = sum.closest("details.card[data-project]");
    if (det) {
      if (det.open) state.openProjects.delete(det.dataset.project);
      else state.openProjects.add(det.dataset.project);
    }
  }
});

let removeTarget = null;

function openRemove(projectId) {
  const p = (state.data?.projects || []).find((x) => x.id === projectId);
  if (!p) return;
  removeTarget = { id: projectId, name: p.name };
  $("#removeSub").textContent =
    `Project "${p.name}" (${projectId}) · ${p.status} · ${money(p.cost_usd)} spent`;
  $("#removeConfirm").value = "";
  $("#removeMsg").textContent = "";
  $("#remove").showModal();
}

async function doRemove() {
  if (!removeTarget) return;
  const mode = document.querySelector('input[name="removalMode"]:checked')?.value;
  const res = await fetch(`/api/projects/${removeTarget.id}/remove`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, confirm: $("#removeConfirm").value.trim() }),
  });
  if (res.ok) {
    $("#remove").close();
    state.openProjects.delete(removeTarget.id);
    await refresh();
  } else {
    const body = await res.text();
    $("#removeMsg").textContent = "Rejected: " + body.slice(0, 200);
  }
}

const SETTING_KEYS = [
  "llm.model", "llm.base_url", "llm.api_key", "llm.request_timeout",
  "llm.max_request_timeout", "llm.empty_response_retries", "tick_seconds",
  "sub_agent_turn_limit", "max_parallel_workers", "llm_queue_depth",
  "llm_server_concurrency", "router_stall_timeout", "router_max_call_timeout",
  "max_sub_agent_depth", "max_concurrent_sub_agents_per_project",
  "max_children_per_agent", "idle_tick_seconds", "llm.input_cost_per_token",
  "llm.output_cost_per_token",
];

let serverList = [];
let serversBaseline = "[]";

function serverRow(srv, i) {
  const s = srv || { name: "", provider: "ollama_chat", base_url: "", model: "" };
  const cell = (f, ph) => `<input data-srv="${i}" data-field="${f}"
      value="${esc(s[f] ?? "")}" placeholder="${ph}">`;
  return `<div class="serverrow">
    ${cell("name", "name")}
    <select data-srv="${i}" data-field="provider">
      ${["openai", "ollama", "ollama_chat"].map((p) =>
        `<option ${s.provider === p ? "selected" : ""}>${p}</option>`).join("")}
    </select>
    ${cell("base_url", "http://host:11434/v1")}
    ${cell("model", "model key sent to the server")}
    ${cell("api_key", "(unchanged)")}
    <button class="ghost" type="button" data-srvtest="${i}"
      title="Send one prompt to exactly this endpoint, without saving it">Test</button>
    <button class="ghost" type="button" data-srvdel="${i}">Remove</button>
  </div>`;
}

function populateTestServers() {
  const sel = $("#testServer");
  if (!sel) return;
  const current = sel.value;
  // __primary__ is built from the form, so an unsaved primary endpoint can be
  // tested too. Each server row also has its own Test button.
  const opts = ["__primary__", ...serverList.filter((x) => x.name).map((x) => x.name)];
  sel.innerHTML = opts.map((n) => n === "__primary__"
    ? `<option value="__primary__">(primary endpoint, from the form)</option>`
    : `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  if (opts.includes(current)) sel.value = current;
}

async function runTestPrompt(rowIndex) {
  const msg = $("#testMsg");
  const out = $("#testOut");
  const chosen = $("#testServer") ? $("#testServer").value : "__primary__";
  const prompt = $("#testPrompt") ? $("#testPrompt").value.trim() : "";
  const form = $("#settingsForm");
  const payload = { prompt, max_tokens: 64 };
  if (rowIndex !== undefined) {
    // The row's own Test button: send to exactly what is typed in that row,
    // whether or not it has been saved.
    const row = readRow(rowIndex);
    if (!row.base_url || !row.model) {
      msg.textContent = "fill in base URL and model for that row first";
      return;
    }
    payload.server_config = row;
  } else if (chosen === "__primary__") {
    payload.server_config = {
      name: "primary", provider: "openai",
      base_url: form.elements["llm.base_url"].value.trim(),
      model: form.elements["llm.model"].value.trim(),
      api_key: form.elements["llm.api_key"].value.trim(),
    };
  } else {
    payload.server = chosen;
  }
  if (!payload.server_config && !payload.server) {
    msg.textContent = "give the server a name first";
    return;
  }
  msg.textContent = "sending…";
  out.hidden = true;
  try {
    const res = await fetch("/api/llm/router/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();
    if (!res.ok) {
      msg.textContent = body.detail || "the test failed";
      return;
    }
    if (body.ok) {
      msg.textContent = `${body.server}: ok · ${body.seconds}s · `
        + `${body.completion_tokens ?? 0} tok · ${body.tokens_per_second ?? "?"} tok/s`;
      out.textContent = `model sent: ${body.model}\n`
        + `reply: ${body.reply || "(no final answer)"}`
        + (body.reasoning_preview
            ? `\nreasoning: ${body.reasoning_preview}` : "")
        + (body.note ? `\n\n${body.note}` : "");
    } else {
      msg.textContent = `${body.server}: failed`;
      out.textContent = `error: ${body.error}`
        + (body.status ? ` (HTTP ${body.status})` : "");
    }
    out.hidden = false;
  } catch (err) {
    msg.textContent = "request to the harness failed: " + err;
  }
}

function readRow(i) {
  // One row's currently typed values, so a brand-new endpoint can be tested
  // before it is saved.
  const out = { ...(serverList[i] || {}) };
  for (const el of document.querySelectorAll(
      `[data-srv="${i}"][data-field]`)) {
    out[el.dataset.field] = el.value.trim();
  }
  return out;
}

function syncServerList() {
  // Capture what is currently typed before a re-render replaces the inputs.
  serverList = collectServers();
}

function renderServers(cfg, health) {
  serverList = (cfg.llm_servers || []).map((s) => ({ ...s }));
  serversBaseline = JSON.stringify(serverList);
  $("#serverRows").innerHTML = serverList.map(serverRow).join("")
    || '<p class="muted">No extra servers — every request goes to the primary '
     + 'endpoint above.</p>';
  const primary = { name: cfg.llm?.name || "primary", model: cfg.llm?.model,
                    base_url: cfg.llm?.base_url, provider: "openai" };
  const rows = [primary, ...(health || [])];
  $("#serverHealth").innerHTML = rows.map((h) => `<span class="tag ${
      h.cooling_down ? "bad" : "ok"}" title="${esc(h.base_url || "")}">${
      esc(h.name)} · ${esc(h.model)}${
      h.cooling_down ? ` · cooling ${h.cooldown_seconds}s` : ""}</span>`).join("");
}

async function openSettings() {
  const res = await fetch("/api/config");
  const d = await res.json();
  const flat = { ...d.config };
  for (const [k, v] of Object.entries(d.config.llm || {})) flat[`llm.${k}`] = v;
  // The live timeout is the adaptive one, not the configured starting value.
  flat["llm.request_timeout"] = d.runtime.llm_timeout;
  const form = $("#settingsForm");
  for (const key of SETTING_KEYS) {
    const input = form.elements[key];
    if (!input) continue;
    input.value = key === "llm.api_key" ? "" : (flat[key] ?? "");
  }
  renderServers(d.config, d.llm_servers);
  populateTestServers();
  $("#settingsMsg").textContent = "";
  $("#settings").showModal();
}

function collectServers() {
  // Merge onto the loaded server objects: fields with no input here
  // (extra_body, token caps, api_key_env) must survive an unrelated edit.
  const out = serverList.map((s) => ({ ...s }));
  for (const el of document.querySelectorAll("[data-srv][data-field]")) {
    const i = Number(el.dataset.srv);
    const row = out[i] || (out[i] = {});
    const v = el.value.trim();
    if (v) row[el.dataset.field] = v;
  }
  return out.filter((s) => s.name && s.base_url && s.model);
}

async function saveSettings() {
  const form = $("#settingsForm");
  const patch = {};
  for (const key of SETTING_KEYS) {
    const input = form.elements[key];
    if (!input || input.value === "") continue;
    patch[key] = input.value;
  }
  const servers = collectServers();
  if (JSON.stringify(servers) !== serversBaseline) {
    patch.llm_servers = servers;
  }
  if (!Object.keys(patch).length) {
    $("#settingsMsg").textContent = "Nothing to change.";
    return;
  }
  const res = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (res.ok) {
    const out = await res.json();
    $("#settingsMsg").textContent =
      `Applied ${Object.keys(out.changed).length} setting(s).`;
    await refresh();
    setTimeout(() => $("#settings").close(), 900);
  } else {
    $("#settingsMsg").textContent = "Rejected: " + (await res.text());
  }
}

refresh();
/* The queue is live runtime state, so it is polled faster than the board and
   patched into its own container rather than re-rendering the whole page. */
async function showRouterPrompt(requestId) {
  const res = await fetch(`/api/llm/router/requests/${encodeURIComponent(requestId)}`);
  const body = await res.json();
  if (!res.ok) {
    // Requests are pruned shortly after they finish, so a click can arrive just
    // after the card it came from disappears.
    $("#detailBody").innerHTML = `<h3>Request already cleared</h3>
      <div class="muted">${esc(body.detail || "That request is no longer in the router.")}</div>`;
  } else {
    $("#detailBody").innerHTML = `<h3>Prompt held by the LLM router</h3>
      <div class="muted">${esc(body.id)}</div>
      <div class="sec"><h5>Prompt</h5><pre>${esc(body.prompt)}</pre></div>`;
  }
  $("#detail").showModal();
}

async function refreshQueues() {
  const projects = (state.data && state.data.projects) || [];
  // One router snapshot per poll: it is a single shared service, so every
  // project's strip reads from the same object.
  try {
    const res = await fetch("/api/llm/router");
    if (res.ok) {
      state.router = await res.json();
      projects.forEach((p) => {
        const host = document.getElementById(`router-${p.id}`);
        if (host) host.innerHTML = renderRouter(state.router, p.id);
      });
    }
  } catch { /* harness mid-restart; next tick retries */ }
  await Promise.all(projects.map(async (p) => {
    try {
      const res = await fetch(`/api/projects/${p.id}/queue`);
      if (!res.ok) return;
      const body = await res.json();
      state.queues[p.id] = body.queue;
      const host = document.getElementById(`queue-${p.id}`);
      if (host) host.innerHTML = renderQueue(body.queue);
    } catch { /* harness mid-restart; next tick retries */ }
  }));
}

setInterval(refresh, 5000);
setInterval(refreshQueues, 1500);
