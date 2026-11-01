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
    def state() -> dict:
        projects = []
        for project in store.visible_projects():
            board = {col: [_task_json(t, agents=store.agents) for t in items]
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
                "tree": _agent_tree(store, project.id),
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
                     "cycle": _cycle_status(store, config.tick_seconds)},
            "total_cost_usd": store.harness_cost(),
            "project_cost_usd": sum(store.project_cost(p.id)
                                    for p in store.visible_projects()),
            "orchestrator_cost_usd": store.orchestrator_cost(),
            "columns": KANBAN_COLUMNS,
            "projects": projects,
            "events": [e.model_dump() for e in store.recent_events[-60:]][::-1],
            "alerts": [a for a in _alerts(store) if not a["acknowledged"]],
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
        project_agents = [a for a in store.project_agents(project_id)
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
        doomed_agents = [a for a in store.project_agents(project_id)
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

    @app.get("/api/config")
    def get_config() -> dict:
        # The API key is never echoed to the UI; it is only ever written to.
        redacted = config.model_dump()
        redacted["llm"]["api_key"] = "***" if config.llm.resolved_api_key() else ""
        return {
            "config": redacted,
            "runtime": {
                "llm_timeout": orchestrator.runtime.llm_timeout,
                "llm_timeouts_widened": orchestrator.runtime.llm_timeouts_widened,
                "llm_empty_responses": orchestrator.runtime.llm_empty_responses,
                "llm_transient_errors": orchestrator.runtime.llm_transient_errors,
                "max_parallel_workers": orchestrator.runtime.gate.limit,
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
        store.add_event(
            "config_updated",
            "Runtime config updated: " + ", ".join(f"{k}={v}" for k, v in changed),
        )
        store.flush()
        return {"changed": dict(changed),
                "runtime": {
                    "llm_timeout": orchestrator.runtime.llm_timeout,
                    "max_parallel_workers": orchestrator.runtime.gate.limit,
                }}

    @app.post("/api/alerts/ack")
    def ack_alert(body: dict) -> dict:
        """Dismiss an alert. Pass alert_id to dismiss one, all=true for all."""
        alert_id = (body.get("alert_id") or "").strip()
        acked = store.meta.setdefault("alert_acks", {})
        if body.get("all"):
            for a in _alerts(store):
                acked[a["id"]] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        elif alert_id:
            acked[alert_id] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        else:
            raise HTTPException(400, "Provide alert_id or all=true.")

        # Ack ids are per-occurrence, so entries for alerts that have since
        # resolved are dead weight; keep only what is still on screen.
        live = {a["id"] for a in _alerts(store)}
        store.meta["alert_acks"] = {k: v for k, v in acked.items() if k in live}
        store.flush()
        return {"acknowledged": len(store.meta["alert_acks"])}

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

    @app.post("/api/tick")
    def force_tick() -> dict:
        """Run one orchestrator cycle in the background, immediately."""
        threading.Thread(target=orchestrator.run_cycle, daemon=True).start()
        return {"started": True}

    @app.get("/api/projects/{project_id}/tasks/{task_id}")
    def task_detail(project_id: str, task_id: str) -> dict:
        task = store.tasks.get(task_id)
        if task is None or task.project_id != project_id:
            raise HTTPException(404, "Unknown task")
        return _task_json(task, full=True, agents=store.agents)

    @app.get("/api/projects/{project_id}/agents/{agent_id}")
    def agent_detail(project_id: str, agent_id: str) -> dict:
        agent = store.agents.get(agent_id)
        if agent is None or agent.project_id != project_id:
            raise HTTPException(404, "Unknown agent")
        return _agent_json(agent, full=True)

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


def _task_json(task, full: bool = False, agents: dict | None = None) -> dict:
    # Cost lives on the agent, not the card; mirror it here so the kanban card
    # shows what its worker has actually spent.
    agent = (agents or {}).get(task.assigned_agent_id or "")
    data = {
        "id": task.id, "title": task.title, "expertise": task.expertise,
        "column": task.column, "assigned_agent_id": task.assigned_agent_id,
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


def _agent_json(agent, full: bool = False) -> dict:
    data = {
        "id": agent.id, "role": agent.role, "specialty": agent.specialty,
        "status": agent.status, "depth": agent.depth,
        "turns_used": agent.turns_used, "turn_limit": agent.turn_limit,
        "task_id": agent.task_id, "tokens": agent.tokens,
        "cost_usd": agent.cost_usd,
        "children": agent.child_agent_ids,
        # The orchestrator is shared across projects, so its spend is harness
        # overhead, not part of any single project's cost.
        "shared": agent.role == "orchestrator",
    }
    if full:
        data.update({
            "mission": agent.mission, "system_prompt": agent.system_prompt,
            "last_report": agent.last_report, "last_error": agent.last_error,
            "runs": agent.runs, "cost_usd": agent.cost_usd,
            "conversation_id": agent.conversation_id,
            "created_by": agent.created_by, "created_at": agent.created_at,
        })
    return data


def _agent_tree(store: Store, project_id: str) -> list[dict]:
    """Nested agent hierarchy for the project, orchestrator first.

    Children come from parent_agent_id, not child_agent_ids: retiring an agent
    prunes the parent's child list to free a delegation slot, but the retired
    agent still belongs in the tree with a status of its own.
    """
    agents = store.project_agents(project_id)
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
        return {**_agent_json(agent), "children": children}

    return [node(a, {a.id}) for a in roots]


def _alert_id(key: str) -> str:
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _alerts(store: Store) -> list[dict]:
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
        alerts.append({
            "level": "warn", "project_id": agent.project_id,
            "key": f"agent:{agent.id}:{agent.status}:{agent.turns_used}",
            "message": f"{agent.role} is {agent.status} "
                       f"({agent.turns_used}/{agent.turn_limit} turns).",
        })

    acked = store.meta.get("alert_acks", {})
    for a in alerts:
        a["id"] = _alert_id(a.pop("key"))
        a["acknowledged"] = a["id"] in acked
    return alerts


def _cycle_status(store: Store, tick_seconds: int) -> dict:
    """How fresh the orchestrator loop is, for the header pill."""
    meta = store.meta
    now = int(time.time())
    attempt = meta.get("last_cycle_attempt_at")
    completed = meta.get("last_cycle_completed_at") or meta.get("last_tick")
    status = meta.get("cycle_status", "idle")
    return {
        "status": status,
        "count": int(meta.get("cycle_count", 0)),
        "running": status == "running",
        "started_at": meta.get("cycle_started_at"),
        "last_attempt_at": attempt,
        "last_completed_at": completed,
        "last_tick": meta.get("last_tick"),
        "duration": meta.get("cycle_duration", 0),
        "loop_started_at": meta.get("loop_started_at"),
        "tick_seconds": tick_seconds,
        "stale": bool(attempt and status != "running"
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
    "sub_agent_turn_limit": ("runtime", int),
    "orchestrator_turn_limit": ("runtime", int),
    "max_parallel_workers": ("gate", int),
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

        if key.startswith("llm."):
            setattr(config.llm, key.split(".", 1)[1], coerced)
        else:
            setattr(config, key, coerced)
        changed.append((key, coerced))

    # Applied last: these need the validated values above.
    if any(k == "max_parallel_workers" for k, _ in changed):
        orchestrator.runtime.gate.set_limit(config.max_parallel_workers)
    timeout_patch = next((v for k, v in changed if k == "llm.request_timeout"), None)
    if timeout_patch is not None:
        orchestrator.runtime.set_llm_timeout(int(timeout_patch))

    return changed


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
        node[parts[-1]] = value

    payload = json.dumps(on_disk, indent=2, sort_keys=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, target)
