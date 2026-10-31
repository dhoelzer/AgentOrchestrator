const $ = (s, r = document) => r.querySelector(s);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { data: null, openProjects: new Set(), tab: "board" };

function turnsClass(used, limit) {
  if (used >= limit) return "turns full";
  if (used >= limit * 0.8) return "turns hot";
  return "turns";
}

function renderHeader(d) {
  $("#modelPill").innerHTML = `model: <b>${esc(d.config.model)}</b>`;
  $("#tickPill").innerHTML = `tick: <b>${d.config.tick_seconds}s</b>`;
  $("#budgetPill").innerHTML = `sub-agent budget: <b>${d.config.sub_agent_turn_limit} turns</b>`;
  const inflight = d.meta.inflight || [];
  const last = d.meta.last_tick ? new Date(d.meta.last_tick * 1000).toLocaleTimeString() : "never";
  $("#runPill").innerHTML = `running: <b>${inflight.length}</b> · last cycle ${last}`;
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
  $("#alerts").innerHTML = d.alerts
    .map((a) => `<div class="alert ${a.level}">${esc(a.message)}</div>`)
    .join("");
}

function taskCard(t) {
  return `<div class="task" data-task="${t.id}">
    <div class="t">${esc(t.title)}</div>
    <div class="m">
      <span class="tag">${esc(t.expertise)}</span>
      <span class="${turnsClass(t.turns_used, t.turn_limit)}">${t.turns_used}/${t.turn_limit} turns</span>
      ${t.review_rounds ? `<span class="tag">review ×${t.review_rounds}</span>` : ""}
      ${t.review_verdict ? `<span class="tag">${esc(t.review_verdict)}</span>` : ""}
      ${t.assigned_agent_id ? `<span class="tag">🤖 ${esc(t.assigned_agent_id)}</span>` : ""}
    </div>
    ${t.blocked_reason ? `<div class="blocked-reason">⚠ ${esc(t.blocked_reason)}</div>` : ""}
  </div>`;
}

function renderTree(nodes) {
  if (!nodes.length) return '<div class="muted">No agents yet.</div>';
  const walk = (n, root = false) => {
    const kids = (n.children || []).map((c) => walk(c)).join("");
    return `<div class="node ${root ? "root" : ""}">
      <div class="row" data-agent="${n.id}">
        <span class="dot ${esc(n.status)}"></span>
        <span class="role">${esc(n.role)}</span>
        <span class="spec">${esc(n.specialty)}</span>
        <span class="kv">${n.turns_used}/${n.turn_limit} turns · ${esc(n.status)}</span>
        ${n.task_id ? `<span class="tag">task ${esc(n.task_id)}</span>` : ""}
        ${n.tokens ? `<span class="kv">${(n.tokens / 1000).toFixed(1)}k tok</span>` : ""}
        ${n.cost_usd ? `<span class="kv">$${n.cost_usd.toFixed(4)}</span>` : ""}
      </div>
      ${kids}
    </div>`;
  };
  return `<div class="tree">${nodes.map((n) => walk(n, true)).join("")}</div>`;
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

function renderProjects(d) {
  const html = d.projects
    .map((p) => {
      const open = state.openProjects.has(p.id) ? "open" : "";
      return `<details class="card" ${open} data-project="${p.id}">
        <summary>
          <span class="pname">${esc(p.name)}</span>
          <span class="badge ${esc(p.status)}">${esc(p.status)}</span>
          <span class="goal">${esc(p.goal)}</span>
          <span class="spacer"></span>
          <span class="kv">${esc(p.id)}</span>
          <button data-pause="${p.id}">${p.status === "paused" ? "Resume" : "Pause"}</button>
          ${p.status === "ready_for_delivery" ? `<button data-deliver="${p.id}">Mark delivered</button>` : ""}
        </summary>
        <div class="body">
          <div class="tabs">
            <button class="tab ${state.tab === "board" ? "active" : ""}" data-tab="board">Kanban</button>
            <button class="tab ${state.tab === "agents" ? "active" : ""}" data-tab="agents">Agent hierarchy</button>
            <span class="spacer"></span>
            <span class="kv">${esc(p.workspace)}</span>
          </div>
          <div class="grid2">
            <div>${state.tab === "board" ? renderBoard(p) : renderTree(p.tree)}</div>
            <div>
              <h5 style="margin:12px 0 6px;font-size:12px;text-transform:uppercase;color:var(--dim)">
                ${state.tab === "board" ? "Agent hierarchy" : "Kanban"}</h5>
              ${state.tab === "board" ? renderTree(p.tree) : renderBoard(p)}
            </div>
          </div>
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
    <div class="sec"><h5>Blocked</h5><pre>${esc(t.blocked_reason || "not blocked")}</pre></div>
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
    <div class="sec"><h5>Turn budget</h5><pre>${a.turns_used}/${a.turn_limit} turns · ${a.runs} runs · ${a.tokens} tokens · $${(a.cost_usd ?? 0).toFixed(4)}</pre></div>
    <div class="sec"><h5>Mission</h5><pre>${esc(a.mission)}</pre></div>
    <div class="sec"><h5>Last report</h5><pre>${esc(a.last_report || "—")}</pre></div>
    <div class="sec"><h5>Last error</h5><pre>${esc(a.last_error || "none")}</pre></div>
    <div class="sec"><h5>System prompt</h5><pre>${esc(a.system_prompt)}</pre></div>`;
  $("#detail").showModal();
}

document.addEventListener("submit", async (e) => {
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
setInterval(refresh, 5000);
