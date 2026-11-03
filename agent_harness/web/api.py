"""FastAPI web interface: create projects and watch the kanban + agent tree."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import HarnessConfig
from ..orchestrator import Orchestrator
from ..store import KANBAN_COLUMNS, AgentSpec, Store
from ..version import __version__, version_string

TEMPLATES = Path(__file__).parent / "templates"


class ProjectForm(BaseModel):
    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    constraints: str = ""


def build_api(config: HarnessConfig, store: Store, orchestrator: Orchestrator,
              config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Orchestrator Harness", version=__version__)
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")),
              name="static")

    # ---------------- views ----------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        # Cache-bust the bundle by mtime so a harness upgrade is picked up by
        # dashboards that are already open.
        app_js = Path(__file__).parent / "static" / "app.js"
        html = (TEMPLATES / "index.html").read_text().replace(
            "/static/app.js", f"/static/app.js?v={int(app_js.stat().st_mtime)}")
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "harness_version": version_string(),
            "model": config.llm.model,
            "base_url": config.llm.base_url,
            "tick_seconds": config.tick_seconds,
            "sub_agent_turn_limit": config.sub_agent_turn_limit,
            "last_tick": store.meta.get("last_tick"),
            "inflight": sorted(orchestrator.runtime.inflight),
        }

    @app.get("/api/state")
    def state(show_archived: bool = False) -> dict:
        projects = []
        for project in store.visible_projects():
            active = orchestrator.runtime.inflight
            board = {col: [_task_json(t, agents=store.agents, active=active)
                           for t in items]
                     for col, items in store.board(project.id).items()}
            revisions = list(reversed(project.revision_log))
            projects.append({
                "id": project.id,
                "name": project.name,
                "goal": project.goal,
                "constraints": project.constraints,
                "status": project.status,
                "workspace": project.workspace,
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "delivery_signaled": project.delivery_signaled,
                "version": project.version,
                "version_major": project.version_major,
                "version_minor": project.version_minor,
                "revisions": revisions,
                "cost_usd": store.project_cost(project.id),
                "board": board,
                "counts": store.column_counts(project.id),
                "tree": _agent_tree(store, project.id, active=active,
                                    include_archived=show_archived),
                "archived_agents": sum(
                    1 for a in store.project_agents(project.id,
                                                     include_archived=True)
                    if a.archived_at is not None),
                # Queued user input the orchestrator has not acted on yet, so
                # the UI can show a request as 'pending' until a cycle consumes it.
                "user_messages": project.user_messages,
            })
        return {
            "config": {
                "model": config.llm.model,
                "base_url": config.llm.base_url,
                "tick_seconds": config.tick_seconds,
                "sub_agent_turn_limit": config.sub_agent_turn_limit,
                "orchestrator_turn_limit": config.orchestrator_turn_limit,
            },
            "meta": {"harness_version": version_string(),
                     # Build that last wrote state.json: differs from the running
                     # build when a container was not rebuilt against new data.
                     "harness_version_stored": store.meta.get("harness_version_stored"),
                     "last_tick": store.meta.get("last_tick"),
                     "inflight": sorted(orchestrator.runtime.inflight),
                     "llm": _llm_health(orchestrator),
                     # Everything the router strip needs: pending/queued LLM
                     # requests and one entry per configured endpoint.
                     "llm_router": _router_state(orchestrator),
                     "capacity": orchestrator.runtime.capacity(),
                     "cycle": _cycle_status(store, orchestrator)},
            "total_cost_usd": store.harness_cost(),
            "project_cost_usd": sum(store.project_cost(p.id)
                                    for p in store.visible_projects()),
            "orchestrator_cost_usd": store.orchestrator_cost(),
            "columns": KANBAN_COLUMNS,
            "llm_queue": orchestrator.runtime.llm_queue(),
            "projects": projects,
            "events": [e.model_dump() for e in store.recent_events[-60:]][::-1],
            "alerts": [{k: v for k, v in a.items() if k != "key"}
                       for a in _alerts(store, orchestrator.runtime)
                       if not a["acknowledged"]],
        }

    # ---------------- mutations ----------------

    @app.post("/api/projects")
    def create_project(form: ProjectForm) -> dict:
        slug = form.name.strip().replace(" ", "-")[:40]
        if not slug:
            raise HTTPException(400, "Project name is required")
        workspace = str(Path(config.projects_dir) / slug)
        project = store.create_project(
            name=form.name.strip(), goal=form.goal.strip(),
            constraints=form.constraints.strip(), workspace=workspace,
        )
        return {"id": project.id, "workspace": project.workspace}

    @app.post("/projects")
    def create_project_form(name: str = Form(...), goal: str = Form(...),
                            constraints: str = Form("")) -> RedirectResponse:
        slug = name.strip().replace(" ", "-")[:40]
        workspace = str(Path(config.projects_dir) / slug)
        store.create_project(name=name.strip(), goal=goal.strip(),
                             constraints=constraints.strip(), workspace=workspace)
        return RedirectResponse("/", status_code=303)

    @app.post("/api/projects/{project_id}/pause")
    def pause_project(project_id: str) -> dict:
        project = _get_project(project_id)
        project.status = "paused" if project.status != "paused" else "active"
        project.touch()
        store.flush()
        return {"id": project.id, "status": project.status}

    @app.post("/api/projects/{project_id}/mark-delivered")
    def mark_delivered(project_id: str) -> dict:
        project = _get_project(project_id)
        project.status = "delivered"
        project.delivered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        project.touch()
        store.add_event("delivered", f"User marked '{project.name}' as delivered.",
                        project_id=project.id)
        # A delivery is a release: bump the major and reset the minor.
        revised = store.revise_project(project_id, "delivered", release=True)
        return {"id": project.id, "status": project.status,
                "version": revised.version if revised else project.version}

    @app.post("/api/projects/{project_id}/revise")
    def revise_project(project_id: str, body: dict) -> dict:
        """Edit a project's goal/constraints; a scope change is a new revision."""
        project = _get_project(project_id)
        before = project.version
        changed: list[str] = []
        for field in ("goal", "constraints", "name"):
            if field not in body or body[field] is None:
                continue
            value = str(body[field]).strip()
            # Empty is only a valid value for constraints (clearing them);
            # a blank name or goal would make the project unusable.
            if not value and field != "constraints":
                continue
            if value != getattr(project, field):
                setattr(project, field, value)
                changed.append(field)
        if not changed:
            return {"id": project.id, "version": project.version, "changed": []}
        project.touch()
        revised = store.revise_project(
            project_id, "scope: updated " + ", ".join(changed))
        return {"id": project.id, "version": revised.version if revised else project.version,
                "previous_version": before, "changed": changed}

    @app.post("/api/projects/{project_id}/remove")
    def remove_project(project_id: str, body: dict) -> dict:
        """Remove a project. mode: archive | hide | purge.

        archive - move the workspace to the archive dir, keep harness state.
        hide    - drop it from the interface, leave every file untouched.
        purge   - delete workspace + conversation logs + harness state.
        """
        project = _get_project(project_id)
        mode = (body.get("mode") or "").strip().lower()
        if mode not in REMOVAL_MODES:
            raise HTTPException(400, f"mode must be one of {sorted(REMOVAL_MODES)}")
        if body.get("confirm") != project.name:
            raise HTTPException(400,
                                "Confirmation failed: send the project name as 'confirm'.")

        # The orchestrator is shared across projects: never close or block on it.
        # Archived agents are included: removal has to account for their
        # conversation logs and cost even though the hierarchy hides them.
        project_agents = [a for a in store.project_agents(project_id,
                                                          include_archived=True)
                          if a.role != "orchestrator"]

        running = sorted(a.id for a in project_agents
                         if a.id in orchestrator.runtime.inflight)
        if running:
            raise HTTPException(409, f"Agents still running: {running}. "
                                     "Pause the project or wait for them.")

        # Conversation logs are keyed by agent id, and the orchestrator's own
        # transcript can live under this project's dir (its project_id is the
        # last project it touched). Handle each agent's dir individually so the
        # orchestrator's is never moved or deleted out from under a live loop.
        doomed_agents = [a for a in store.project_agents(project_id,
                                                          include_archived=True)
                         if a.role != "orchestrator"]
        _, log_note = orchestrator.runtime.handle_agent_logs(
            doomed_agents,
            keep=mode == "hide",
            archive_to=(_archive_root(config) if mode == "archive" else None),
            prune_project_dirs=mode == "purge",
        )

        detail = _remove_artifacts(config, store, project, mode)
        # remove_project snapshots cost and deletes the agents in every mode;
        # purge additionally drops the project and task records.
        store.remove_project(project_id, mode)
        if mode == "purge":
            store.purge_project(project_id)

        store.add_event(
            "project_removed",
            f"Project '{project.name}' removed ({mode}). {detail} {log_note} "
            f"Released {len(doomed_agents)} agent(s).",
            project_id=project_id,
        )
        store.flush()
        return {"id": project_id, "mode": mode, "detail": detail,
                "agents_removed": len(doomed_agents)}

    @app.get("/api/llm/router")
    def router_state() -> dict:
        """Live router view: queued/sent LLM requests and per-endpoint health."""
        return _router_state(orchestrator)

    @app.get("/api/llm/router/requests/{request_id}")
    def router_request(request_id: str) -> dict:
        prompt = orchestrator.runtime.router.request_prompt(request_id) \
            if orchestrator.runtime.router else None
        if prompt is None:
            raise HTTPException(404, "That request is no longer in the router.")
        return {"id": request_id, "prompt": prompt}

    @app.post("/api/llm/router/test")
    def router_test(body: dict) -> dict:
        """Send one prompt straight to a named endpoint and report the result.

        Used by the "Test prompt" button when adding an LLM, so a new endpoint
        can be proven before it ever receives real agent work.
        """
        router = orchestrator.runtime.router
        if router is None:
            raise HTTPException(503, "The LLM router is not running.")
        prompt = (body.get("prompt") or "").strip() or (
            "Reply with exactly the word READY and nothing else.")
        try:
            max_tokens = int(body.get("max_tokens") or 64)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "max_tokens must be a number") from exc
        max_tokens = max(1, min(max_tokens, 512))
        # An inline definition lets a new endpoint be proven before it is saved
        # into the config and starts receiving real agent work.
        inline = body.get("server_config")
        if inline:
            server = _validate_server(inline)
            result = router.test_prompt_config(server, prompt, max_tokens)
        else:
            server_name = (body.get("server") or "").strip()
            if not server_name:
                raise HTTPException(400, "A server name or server_config is required.")
            result = router.test_prompt(server_name, prompt, max_tokens)
        store.add_event(
            "llm_test_prompt",
            f"Test prompt to LLM '{result.get('server')}': "
            + ("ok" if result.get("ok") else f"failed ({result.get('error')})"),
        )
        store.flush()
        return result

    @app.post("/api/llm/router/probe")
    def router_probe() -> dict:
        router = orchestrator.runtime.router
        if router is None:
            raise HTTPException(503, "The LLM router is not running.")
        return {"servers": router.probe_all()}

    @app.get("/api/config")
    def get_config() -> dict:
        # The API key is never echoed to the UI; it is only ever written to.
        redacted = config.model_dump()
        redacted["llm"]["api_key"] = "***" if config.llm.resolved_api_key() else ""
        for srv in redacted.get("llm_servers", []):
            srv["api_key"] = "***" if srv.get("api_key") else ""
        return {
            "config": redacted,
            "llm_servers": _router_state(orchestrator)["servers"],
            "runtime": {
                "llm_timeout": orchestrator.runtime.llm_timeout,
                "llm_timeouts_widened": orchestrator.runtime.llm_timeouts_widened,
                "llm_empty_responses": orchestrator.runtime.llm_empty_responses,
                "llm_transient_errors": orchestrator.runtime.llm_transient_errors,
                "max_parallel_workers": config.max_parallel_workers,
                "llm_slots": orchestrator.runtime.gate.limit,
                "active_llm_slots": orchestrator.runtime.gate.active,
                "inflight": sorted(orchestrator.runtime.inflight),
            },
        }

    @app.post("/api/config")
    def update_config(patch: dict) -> dict:
        """Apply a runtime config patch and hot-reload it where possible.

        Takes effect immediately for new agents and, for LLM timeouts and
        parallelism, for conversations that are already open.
        """
        changed = _apply_config_patch(config, orchestrator, patch)
        if not changed:
            raise HTTPException(400, "Patch contained no changes.")
        if persist := (patch.get("persist") or config_path):
            try:
                _write_config(config, persist, changed)
            except OSError as exc:
                raise HTTPException(500, f"Could not write config: {exc}") from exc
            # The live config is already updated; tell the loop so it does not
            # read its own write back as an external edit next tick.
            orchestrator.note_config_written()
        store.add_event(
            "config_updated",
            "Runtime config updated: " + ", ".join(
                f"{k}={_summarize(k, v)}" for k, v in changed),
        )
        store.flush()
        return {"changed": dict(changed),
                "runtime": {
                    "llm_timeout": orchestrator.runtime.llm_timeout,
                    "max_parallel_workers": config.max_parallel_workers,
                    "llm_slots": orchestrator.runtime.gate.limit,
                }}

    @app.post("/api/alerts/ack")
    def ack_alert(body: dict) -> dict:
        """Dismiss an alert. Pass alert_id to dismiss one, all=true for all.

        Acknowledging an LLM-outage alert is also the user's signal that the
        endpoint has been dealt with: the agent's failure streak resets and its
        card goes back to 'ready', so the next cycle can dispatch it. Without
        that, a card blocked by an outage would sit blocked forever with nobody
        able to unblock it.
        """
        alert_id = (body.get("alert_id") or "").strip()
        acked = store.meta.setdefault("alert_acks", {})
        live = _alerts(store, orchestrator.runtime)
        released: list[str] = []
        targets = live if body.get("all") else [
            a for a in live if a["id"] == alert_id]
        if not targets and not body.get("all") and not alert_id:
            raise HTTPException(400, "Provide alert_id or all=true.")
        for a in targets:
            acked[a["id"]] = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
            if a["key"].startswith("llm:"):
                released.append(a["key"].split(":")[1])
        for agent_id in released:
            _release_llm_block(store, agent_id)

        # Ack ids are per-occurrence, so entries for alerts that have since
        # resolved are dead weight; keep only what is still on screen.
        still_live = {a["id"] for a in _alerts(store, orchestrator.runtime)}
        store.meta["alert_acks"] = {k: v for k, v in acked.items()
                                    if k in still_live}
        store.flush()
        return {"acknowledged": len(store.meta["alert_acks"]),
                "released_agents": released}

    @app.delete("/api/alerts/ack")
    def clear_alert_acks() -> dict:
        store.meta.pop("alert_acks", None)
        store.flush()
        return {"acknowledged": 0}

    @app.post("/api/projects/{project_id}/tasks/{task_id}/directives")
    def add_directive(project_id: str, task_id: str, body: dict) -> dict:
        task = store.tasks.get(task_id)
        if task is None or task.project_id != project_id:
            raise HTTPException(404, "Unknown task")
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "Directive text is required")
        task.user_directives.append({
            "text": text,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        # A user speaking up is exactly the event that should put a stalled task
        # back in play; the next cycle's briefing will carry the directive. The
        # block is answered, so its reason goes with it - otherwise the card
        # keeps advertising a blocker the user has already resolved.
        if task.column == "blocked":
            cleared = task.blocked_reason
            task.column = "ready"
            task.blocked_reason = None
            task.blocked_cause = None
            if cleared:
                store.add_event(
                    "task_unblocked",
                    f"Task '{task.title}' unblocked by user direction "
                    f"(was blocked: {cleared}).",
                    project_id=project_id, task_id=task_id)
        task.touch()
        store.add_event("user_directive", f"User direction on '{task.title}': {text}",
                        project_id=project_id, task_id=task_id)
        store.flush()
        return {"id": task.id, "column": task.column,
                "directives": len(task.user_directives)}

    @app.post("/api/projects/{project_id}/messages")
    def add_project_message(project_id: str, body: dict) -> dict:
        """Add information or a new request for the orchestrator to act on.

        The orchestrator turns it into task cards and agents on the next tick;
        the harness cannot guess what new scope means for the plan.
        """
        project = _get_project(project_id)
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "Message text is required")
        project.user_messages.append({
            "text": text,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        # A delivered/ready project would otherwise be skipped by the loop, so
        # the user's new request would sit unread. Reopening it is what makes
        # the button work after delivery.
        reopened = project.status in ("ready_for_delivery", "delivered")
        if reopened:
            project.status = "active"
            project.delivery_signaled = False
            store.add_event(
                "project_reopened",
                f"Project '{project.name}' reopened by a new user request.",
                project_id=project_id)
        project.touch()
        store.add_event("user_message",
                        f"User added direction to '{project.name}': {text[:300]}",
                        project_id=project_id)
        store.flush()
        return {"id": project.id, "messages": len(project.user_messages),
                "status": project.status, "reopened": reopened}

    @app.post("/api/tick")
    def force_tick() -> dict:
        """Start one orchestrator cycle now, unless one is already committed.

        Refuses rather than queues: a cycle queued behind a running one would
        run a briefing built from the board as it looked before that cycle
        moved anything. run_cycle() re-checks under the lock, so this is only
        the friendly pre-check.
        """
        busy = orchestrator.orchestrator_busy()
        if busy:
            return {"started": False, "running": True,
                    "reason": f"not queued: {busy}"}
        threading.Thread(target=orchestrator.run_cycle, daemon=True).start()
        return {"started": True, "running": True}

    @app.get("/api/projects/{project_id}/tasks/{task_id}")
    def task_detail(project_id: str, task_id: str) -> dict:
        task = store.tasks.get(task_id)
        if task is None or task.project_id != project_id:
            raise HTTPException(404, "Unknown task")
        return _task_json(task, full=True, agents=store.agents,
                        active=orchestrator.runtime.inflight)

    @app.get("/api/projects/{project_id}/queue")
    def project_queue(project_id: str) -> dict:
        """Agent actions awaiting the LLM, oldest first.

        Kept out of the card payloads because it is live runtime state, not
        board state: the strip above the kanban polls it on its own cadence.
        """
        _get_project(project_id)
        return {"queue": [e for e in orchestrator.runtime.llm_queue()
                          if e["project_id"] in (project_id, None)]}

    @app.get("/api/queue/{entry_id}/prompt")
    def queue_prompt(entry_id: str) -> dict:
        """The full prompt an queued action is waiting to send to the LLM."""
        prompt = orchestrator.runtime.llm_prompt(entry_id)
        if prompt is None:
            raise HTTPException(404, "Queue entry already dispatched or gone")
        return {"id": entry_id, "prompt": prompt}

    @app.get("/api/projects/{project_id}/agents/{agent_id}")
    def agent_detail(project_id: str, agent_id: str) -> dict:
        agent = store.agents.get(agent_id)
        if agent is None or agent.project_id != project_id:
            raise HTTPException(404, "Unknown agent")
        return _agent_json(agent, full=True,
                        active=orchestrator.runtime.inflight)

    def _get_project(project_id: str):
        project = store.projects.get(project_id)
        if project is None:
            raise HTTPException(404, "Unknown project")
        return project

    return app


def _llm_health(orchestrator: Orchestrator) -> dict:
    runtime = orchestrator.runtime
    return {
        "timeout": runtime.llm_timeout,
        "timeouts_widened": runtime.llm_timeouts_widened,
        "empty_responses": runtime.llm_empty_responses,
        "transient_errors": runtime.llm_transient_errors,
        "adaptive": orchestrator.config.llm.adaptive_timeout,
        "parallel_limit": runtime.gate.limit,
        "active_slots": runtime.gate.active,
    }


def _task_json(task, full: bool = False, agents: dict | None = None,
               active: set | None = None) -> dict:
    # Cost lives on the agent, not the card; mirror it here so the kanban card
    # shows what its worker has actually spent.
    agent = (agents or {}).get(task.assigned_agent_id or "")
    # The card's light follows the worker assigned to it. It uses the runtime's
    # in-flight set rather than the persisted status, which a crash mid-run can
    # leave stuck on "running".
    is_active = bool(task.assigned_agent_id) and bool(
        active and task.assigned_agent_id in active)
    data = {
        "id": task.id, "title": task.title, "expertise": task.expertise,
        "column": task.column, "assigned_agent_id": task.assigned_agent_id,
        "parent_task_id": task.parent_task_id,
        "block_strikes": task.block_strikes,
        "reviewer_agent_id": task.reviewer_agent_id,
        "turns_used": task.turns_used, "turn_limit": task.turn_limit,
        "review_rounds": task.review_rounds,
        "depends_on": task.depends_on,
        "review_verdict": task.review_verdict,
        "blocked_reason": task.blocked_reason,
        "blocked_cause": task.blocked_cause,
        "cost_usd": agent.cost_usd if agent else None,
        "agent_role": agent.role if agent else None,
        "agent_status": agent.status if agent else None,
        "active": is_active,
        "updated_at": task.updated_at,
    }
    if full:
        data.update({
            "objective": task.objective,
            "acceptance_criteria": task.acceptance_criteria,
            "deliverables": task.deliverables,
            "result_summary": task.result_summary,
            "review_feedback": task.review_feedback,
            "orchestrator_note": task.orchestrator_note,
            "user_directives": task.user_directives,
            "created_at": task.created_at, "started_at": task.started_at,
            "finished_at": task.finished_at, "runs": task.runs,
        })
    return data


def _agent_json(agent, full: bool = False, active: set | None = None) -> dict:
    data = {
        "id": agent.id, "role": agent.role, "specialty": agent.specialty,
        "status": agent.status, "depth": agent.depth,
        "active": bool(active and agent.id in active),
        "turns_used": agent.turns_used, "turn_limit": agent.turn_limit,
        "task_id": agent.task_id, "tokens": agent.tokens,
        "cost_usd": agent.cost_usd,
        "children": agent.child_agent_ids,
        # The orchestrator is shared across projects, so its spend is harness
        # overhead, not part of any single project's cost.
        "shared": agent.role == "orchestrator",
        "archived": agent.archived_at is not None,
    }
    if full:
        data.update({
            "mission": agent.mission, "system_prompt": agent.system_prompt,
        "archived_at": agent.archived_at,
        "retirement_reason": agent.retirement_reason,
            "last_report": agent.last_report, "last_error": agent.last_error,
            "runs": agent.runs, "cost_usd": agent.cost_usd,
            "conversation_id": agent.conversation_id,
            "created_by": agent.created_by, "created_at": agent.created_at,
        })
    return data


def _agent_tree(store: Store, project_id: str,
                active: set | None = None,
                include_archived: bool = False) -> list[dict]:
    """Nested agent hierarchy for the project, orchestrator first.

    Children come from parent_agent_id, not child_agent_ids: retiring an agent
    prunes the parent's child list to free a delegation slot, but the retired
    agent still belongs in the tree with a status of its own.

    Archived agents - finished specialists the sweep took off the live tree -
    are hidden unless asked for, so the tree shows who is working rather than
    every agent a long-running project has ever spawned.
    """
    agents = store.project_agents(project_id, include_archived=include_archived)
    # The orchestrator is global but carries a single project_id, so it is not
    # always in the project's own agent list; graft it in as the root.
    orchestrator = next((a for a in store.agents.values()
                         if a.role == "orchestrator"), None)
    if orchestrator is not None:
        agents = [orchestrator] + agents

    by_id = {a.id: a for a in agents}
    kids: dict[str, list[AgentSpec]] = {}
    roots: list[AgentSpec] = []
    for a in agents:
        if a is orchestrator:
            continue
        if a.parent_agent_id and a.parent_agent_id in by_id:
            kids.setdefault(a.parent_agent_id, []).append(a)
        elif orchestrator is not None:
            kids.setdefault(orchestrator.id, []).append(a)
        else:
            roots.append(a)
    if orchestrator is not None:
        roots = [orchestrator]

    def node(agent: AgentSpec, seen: set[str]) -> dict:
        children = []
        for child in kids.get(agent.id, []):
            if child.id not in seen:
                seen.add(child.id)
                children.append(node(child, seen))
        return {**_agent_json(agent, active=active), "children": children}

    return [node(a, {a.id}) for a in roots]


def _alert_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _release_llm_block(store: Store, agent_id: str) -> None:
    """Put an agent blocked by an LLM outage back in play."""
    agent = store.agents.get(agent_id)
    if agent is None:
        return
    agent.llm_failures = 0
    agent.llm_failure_error = None
    if agent.status == "llm_unavailable":
        agent.status = "idle"
    agent.touch()
    store.upsert_agent(agent)
    task = store.tasks.get(agent.task_id or "")
    if task is not None and task.column == "blocked" \
            and task.blocked_cause == "llm":
        task.column = "ready"
        task.blocked_reason = None
        task.blocked_cause = None
        task.block_strikes = 0
        task.block_strike_reason = None
        task.touch()
        store.upsert_task(task)
    store.add_event(
        "llm_block_released",
        f"User acknowledged the LLM alert for {agent.role} ({agent.id}); "
        "failure streak reset and the card returned to ready.",
        project_id=agent.project_id, agent_id=agent.id,
        task_id=task.id if task else None)


def _alerts(store: Store, runtime=None) -> list[dict]:
    """Dashboard alerts, each with a stable id the user can acknowledge.

    The id embeds a generation counter (review round, blocked reason, turn
    count) so acknowledging a problem does not silence the *next* occurrence of
    it - only this one.
    """
    alerts: list[dict] = []
    for project in store.visible_projects():
        if project.status == "ready_for_delivery":
            alerts.append({
                "level": "success", "project_id": project.id,
                "key": f"delivery:{project.id}:{project.ready_generation}",
                "message": f"'{project.name}' is ready to deliver.",
            })
        for task in store.project_tasks(project.id):
            if task.column == "review":
                alerts.append({
                    "level": "info", "project_id": project.id,
                    "key": f"review:{task.id}:{task.review_rounds}",
                    "message": f"'{task.title}' is waiting for review.",
                })
            elif task.column == "blocked":
                alerts.append({
                    "level": "warn", "project_id": project.id,
                    "key": f"blocked:{task.id}:{task.blocked_reason or ''}",
                    "message": f"'{task.title}' is blocked"
                               + (f": {task.blocked_reason}" if task.blocked_reason
                                  else "."),
                })
    for agent in store.agents_in_status(
        ("budget_exhausted", "failed", "llm_unavailable")
    ):
        if agent.llm_failures >= 1:
            # An endpoint fault, not an agent fault: the error-level alert
            # below names the actual failure, and a second warning for the same
            # event just reads as two problems.
            continue
        alerts.append({
            "level": "warn", "project_id": agent.project_id,
            "key": f"agent:{agent.id}:{agent.status}:{agent.turns_used}",
            "message": f"{agent.role} is {agent.status} "
                       f"({agent.turns_used}/{agent.turn_limit} turns).",
        })

    # LLM transport failures are the user's to fix, not the orchestrator's: the
    # harness cannot repair an unreachable endpoint, so it stops retrying and
    # says so here. The key carries the failure count, so each new failure
    # re-alerts instead of being silenced by acknowledging the first one.
    for agent in store.agents.values():
        if agent.llm_failures < 1:
            continue
        if agent.role != "orchestrator":
            proj = store.projects.get(agent.project_id or "")
            if proj is not None and proj.status == "removed":
                continue
        alerts.append({
            "level": "error", "project_id": agent.project_id,
            "key": f"llm:{agent.id}:{agent.llm_failures}",
            "message": (
                f"LLM error while running {agent.role} "
                f"({agent.llm_failures} run(s) in a row): "
                f"{agent.llm_failure_error or agent.last_error}"
                + (f" - task '{store.tasks[agent.task_id].title}' is blocked."
                   if agent.task_id in store.tasks else "")
            ),
        })

    # The orchestrator itself could not run. Nothing on the board moves while
    # this is true, so it is the most important thing on the page.
    cycle_error = store.meta.get("cycle_error")
    if store.meta.get("cycle_status") == "failed" and cycle_error:
        alerts.append({
            "level": "error", "project_id": None,
            "key": f"cycle:{store.meta.get('cycle_count', 0)}:{cycle_error[:80]}",
            "message": f"Orchestrator cycle failed: {cycle_error}",
        })

    # Every configured endpoint down at once: no agent can make progress, and
    # only the user can bring one back.
    servers = (runtime.router.status()["servers"]
               if runtime is not None and runtime.router else [])
    if servers and all(s.get("reachable") is False for s in servers):
        down = ", ".join(s["name"] for s in servers)
        alerts.append({
            "level": "error", "project_id": None,
            "key": f"endpoints_down:{down}",
            "message": f"All LLM endpoints are unreachable ({down}). "
                       "No agent can run until one is back.",
        })

    # A loop that keeps refusing cycles is the "nothing is moving" symptom, and
    # without this the only clue is a counter nobody reads. The reason matters:
    # a cycle refused because an orchestrator prompt has been sitting for many
    # minutes is a different problem from a loop that is simply busy, and the
    # scheduler keeps starting already-staffed work in the meantime, so the
    # user should be told which it is.
    skips = int(store.meta.get("cycle_skips", 0))
    stuck_for = int(store.meta.get("cycle_stuck_seconds", 0))
    if skips >= 3 and stuck_for >= 120:
        alerts.append({
            "level": "warn", "project_id": None,
            "key": f"cycle_stuck:{skips}:{store.meta.get('cycle_started_at', 0)}",
            "message": (
                f"Orchestrator cycles have been refused {skips} times; the "
                f"current orchestrator prompt has been waiting "
                f"{stuck_for}s ({store.meta.get('cycle_skip_reason') or 'unknown'}). "
                "Work that is already staffed still starts on its own; new "
                "planning is what is blocked."),
        })

    acked = store.meta.get("alert_acks", {})
    for a in alerts:
        # The key survives alongside the id: the ack endpoint needs it to tell
        # an LLM-outage alert (which releases a blocked card once the user has
        # been told) apart from an ordinary one.
        a["id"] = _alert_id(a["key"])
        a["acknowledged"] = a["id"] in acked
    return alerts


def _cycle_status(store: Store, orchestrator) -> dict:
    """How fresh the orchestrator loop is, for the header pill.

    The interval is the one the loop is actually using, not the configured
    default: with dynamic tick enabled the two diverge, and showing the static
    value makes a healthy loop look stale (or a lagging one look fine).
    """
    meta = store.meta
    now = int(time.time())
    attempt = meta.get("last_cycle_attempt_at")
    completed = meta.get("last_cycle_completed_at") or meta.get("last_tick")
    status = meta.get("cycle_status", "idle")
    tick_seconds = int(round(float(meta.get("effective_tick")
                                   or orchestrator.effective_tick
                                   or orchestrator.config.tick_seconds)))
    running = orchestrator.cycle_running() or status == "running"
    return {
        "status": status,
        "count": int(meta.get("cycle_count", 0)),
        "running": running,
        "started_at": meta.get("cycle_started_at"),
        "last_attempt_at": attempt,
        "last_completed_at": completed,
        "last_tick": meta.get("last_tick"),
        "duration": meta.get("cycle_duration", 0),
        "loop_started_at": meta.get("loop_started_at"),
        "tick_seconds": tick_seconds,
        "configured_tick_seconds": orchestrator.config.tick_seconds,
        "dynamic_tick": orchestrator.config.dynamic_tick,
        "skips": int(meta.get("cycle_skips", 0)),
        "stale": bool(attempt and not running
                      and now - int(attempt) > max(120, tick_seconds * 3)),
    }


REMOVAL_MODES = {"archive", "hide", "purge"}


def _archive_root(config: HarnessConfig) -> Path:
    return config.root_dir / "archived-projects"


def _remove_artifacts(config: HarnessConfig, store: Store, project,
                      mode: str) -> str:
    """Carry out the workspace side of a removal, returning a readable note.

    Conversation logs are handled per agent (see handle_agent_logs), not here:
    the project's log dir can also contain the orchestrator's live transcript,
    so moving or deleting it wholesale would break a running loop.
    """
    workspace = Path(project.workspace).expanduser() if project.workspace else None

    if mode == "hide":
        return "Files left untouched in " + str(workspace or "(no workspace)") + "."

    if mode == "archive":
        if not workspace or not workspace.exists():
            return "No workspace to archive."
        archive_root = _archive_root(config)
        archive_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = archive_root / f"{workspace.name}-{stamp}"
        shutil.move(str(workspace), str(target))
        return f"Workspace moved to {target}."

    # purge
    if workspace and workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
        return f"Deleted workspace {workspace}."
    return "Deleted nothing (workspace already gone)."



# Settings that can be changed while the harness runs. Each value is coerced to
# the declared type, so a bad type is rejected rather than silently corrupting
# config.
_PATCHABLE = {
    "tick_seconds": ("runtime", int),
    "dynamic_tick": ("runtime", bool),
    "min_tick_seconds": ("runtime", int),
    "max_tick_seconds": ("runtime", int),
    "tick_duty_cycle": ("runtime", float),
    "sub_agent_turn_limit": ("runtime", int),
    "orchestrator_turn_limit": ("runtime", int),
    "max_parallel_workers": ("gate", int),
    "llm_queue_depth": ("pool", int),
    "llm_server_concurrency": ("router", int),
    "router_selection": ("router", str),
    # Read per upstream call, so a patch takes effect on the next request with
    # no restart and no conversation being touched.
    "router_stall_timeout": ("router", int),
    "router_max_call_timeout": ("router", int),
    "auto_dispatch": ("runtime", bool),
    "llm_servers": ("servers", list),
    "max_sub_agent_depth": ("runtime", int),
    "max_children_per_agent": ("runtime", int),
    "review_attempts_before_escalation": ("runtime", int),
    "confirm_tools": ("runtime", bool),
    "llm.model": ("llm", str),
    "llm.base_url": ("llm", str),
    "llm.api_key": ("llm", str),
    "llm.request_timeout": ("llm_timeout", int),
    "llm.min_request_timeout": ("llm", int),
    "llm.max_request_timeout": ("llm", int),
    "llm.timeout_growth": ("llm", float),
    "llm.num_retries": ("llm", int),
    "llm.empty_response_retries": ("llm", int),
    "llm.adaptive_timeout": ("llm", bool),
    "llm.max_input_tokens": ("llm", int),
    "llm.max_output_tokens": ("llm", int),
    "llm.input_cost_per_token": ("llm", float),
    "llm.output_cost_per_token": ("llm", float),
}


def _router_state(orchestrator: Orchestrator) -> dict:
    """Router snapshot, or an empty one when running without a router."""
    router = orchestrator.runtime.router
    if router is None:
        return {"requests": [], "recent": [], "servers": [],
                "counts": {"queued": 0, "awaiting": 0, "generating": 0,
                           "served": 0, "retries": 0, "servers": 0,
                           "server_slots": 0, "server_slots_in_use": 0},
                "uptime_seconds": 0.0}
    return router.status()


def _apply_config_patch(config: HarnessConfig, orchestrator: Orchestrator,
                        patch: dict) -> list[tuple[str, object]]:
    """Validate and apply a dotted-key patch, returning what changed."""
    changed: list[tuple[str, object]] = []

    for key, value in patch.items():
        if key == "persist":
            continue
        if key not in _PATCHABLE:
            raise HTTPException(400, f"Unknown setting {key!r}. "
                                     f"Allowed: {sorted(_PATCHABLE)}")
        _, expected_type = _PATCHABLE[key]
        try:
            coerced = expected_type(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"{key} must be {expected_type.__name__}") from exc

        if key == "llm_servers":
            coerced = _validate_servers(value)
            config.llm_servers = coerced
        elif key.startswith("llm."):
            setattr(config.llm, key.split(".", 1)[1], coerced)
        else:
            setattr(config, key, coerced)
        changed.append((key, coerced))

    # Applied last: these need the validated values above.
    if any(k == "max_parallel_workers" for k, _ in changed):
        # Re-derive the gate from the worker count; setting the gate directly
        # would drop the orchestrator's reserved slot.
        orchestrator.runtime.set_worker_slots(config.max_parallel_workers)
    timeout_patch = next((v for k, v in changed if k == "llm.request_timeout"), None)
    if timeout_patch is not None:
        orchestrator.runtime.set_llm_timeout(int(timeout_patch))
    if any(k in ("max_parallel_workers", "llm_queue_depth") for k, _ in changed):
        orchestrator.runtime.sync_pool_size()
    # Server list and per-endpoint concurrency take effect in the router, which
    # is the only thing that talks to an endpoint.
    router = orchestrator.runtime.router
    if router is not None and any(
            k in ("llm_servers", "llm_server_concurrency", "llm.base_url",
                  "llm.model") for k, _ in changed):
        router.sync_servers()
        router.probe_all()

    return changed


def _validate_server(value: dict):
    """Validate one posted server definition."""
    from agent_harness.config import LLMServerConfig

    if not isinstance(value, dict):
        raise HTTPException(400, "server_config must be an object.")
    try:
        return LLMServerConfig.model_validate(value)
    except Exception as exc:  # noqa: BLE001 - surface pydantic detail to the UI
        raise HTTPException(400, f"Invalid server_config: {exc}") from exc


def _validate_servers(value: list) -> list:
    """Validate a posted server list before it can affect live routing."""
    from agent_harness.config import LLMServerConfig

    if not isinstance(value, list):
        raise HTTPException(400, "llm_servers must be a list of server objects.")
    try:
        servers = [LLMServerConfig.model_validate(v) for v in value]
    except Exception as exc:  # noqa: BLE001 - surface pydantic detail to the UI
        raise HTTPException(400, f"Invalid llm_servers: {exc}") from exc
    names = [s.name for s in servers]
    if len(set(names)) != len(names):
        raise HTTPException(400, f"Server names must be unique; got {names}.")
    return servers


def _summarize(key: str, value):
    """Readable form for the event log; persistence still uses the real value."""
    if key == "llm_servers":
        return [f"{srv.name} ({srv.route()} @ {srv.base_url})" for srv in value]
    if key == "llm.api_key":
        return "***"
    return value


def _write_config(config: HarnessConfig, path: str, changed: list) -> None:
    """Persist the live config so a restart keeps the change.

    Only the patched keys are written. Dumping config.model_dump() would also
    bake in values that came from HARNESS_* environment variables, so a run
    started with e.g. HARNESS_DATA_DIR would silently rewrite the file's
    data_dir for everyone who starts it without that variable.
    """
    target = Path(path).expanduser()
    try:
        on_disk = json.loads(target.read_text()) if target.exists() else {}
    except json.JSONDecodeError:
        on_disk = config.model_dump()

    for key, value in changed:
        parts = key.split(".")
        node = on_disk
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        # Nested pydantic models (the server list) need plain dicts on disk.
        if key == "llm_servers":
            value = [srv.model_dump(exclude_none=True) for srv in value]
        node[parts[-1]] = value

    payload = json.dumps(on_disk, indent=2, sort_keys=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, target)
