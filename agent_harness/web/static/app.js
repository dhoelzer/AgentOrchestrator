const $ = (s, r = document) => r.querySelector(s);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { data: null, openProjects: new Set(), tab: "board", queues: {} };

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
  $("#llmPill").innerHTML =
    `llm: <b>${llm.timeout ?? "?"}s timeout</b>` +
    (unhealthy
      ? ` · <b>${(llm.empty_responses || 0) + (llm.transient_errors || 0)}</b> retried`
      : "");
  $("#llmPill").classList.toggle("warn", unhealthy);
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
  llm: "retries automatically once the endpoint recovers",
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
  return `<div class="msgbar">
    <form class="msgform" data-msg="${esc(p.id)}">
      <input name="text" placeholder="Add information or a new request — the orchestrator will plan it next tick" autocomplete="off">
      <button type="submit">Add to project</button>
    </form>
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

function renderBoard(p) {
  return `<div class="kanban">${p.columns_order
    .map((col) => {
      const items = p.board[col] || [];
      return `<div class="col"><h4><span>${col}</span><span>${items.length}</span></h4>
        ${items.map(taskCard).join("")}</div>`;
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
  const res = await fetch("/api/state");
  const d = await res.json();
  state.data = d;
  d.projects.forEach((p) => {
    p.columns_order = d.columns;
    if (!state.openProjects.has(p.id) && p.status !== "delivered") state.openProjects.add(p.id);
  });
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
    if (!text) return;
    const res = await fetch(`/api/projects/${msgForm}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (res.ok) {
      e.target.reset();
      await refresh();
    } else {
      alert("Could not add that: " + (await res.text()));
    }
    return;
  }
  if (e.target.id !== "projForm") return;
  e.preventDefault();
  const fd = new FormData(e.target);
  const res = await fetch("/api/projects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: fd.get("name"),
      goal: fd.get("goal"),
      constraints: fd.get("constraints") || "",
    }),
  });
  if (res.ok) {
    e.target.reset();
    await refresh();
  } else {
    alert("Could not create project: " + (await res.text()));
  }
});

document.addEventListener("click", async (e) => {
  const t = e.target;
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
  const det = t.closest("details.card[data-project]");
  if (det && !t.closest("button")) {
    if (det.open) state.openProjects.delete(det.dataset.project);
    else state.openProjects.add(det.dataset.project);
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
  "sub_agent_turn_limit", "max_parallel_workers", "max_sub_agent_depth",
  "llm.input_cost_per_token", "llm.output_cost_per_token",
];

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
  $("#settingsMsg").textContent = "";
  $("#settings").showModal();
}

async function saveSettings() {
  const form = $("#settingsForm");
  const patch = {};
  for (const key of SETTING_KEYS) {
    const input = form.elements[key];
    if (!input || input.value === "") continue;
    patch[key] = input.value;
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
async function refreshQueues() {
  const projects = (state.data && state.data.projects) || [];
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
