"""FastAPI web interface: create projects and watch the kanban + agent tree."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import HarnessConfig
from ..orchestrator import Orchestrator
from ..store import KANBAN_COLUMNS, AgentSpec, Store

TEMPLATES = Path(__file__).parent / "templates"


class ProjectForm(BaseModel):
    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    constraints: str = ""


def build_api(config: HarnessConfig, store: Store, orchestrator: Orchestrator,
              config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Orchestrator Harness", version="1.0")
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")),
              name="static")

    # ---------------- views ----------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (TEMPLATES / "index.html").read_text()

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
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
        for project in sorted(store.projects.values(), key=lambda p: p.created_at):
            board = {col: [_task_json(t) for t in items]
                     for col, items in store.board(project.id).items()}
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
            "meta": {"last_tick": store.meta.get("last_tick"),
                     "inflight": sorted(orchestrator.runtime.inflight),
                     "llm": _llm_health(orchestrator)},
            "columns": KANBAN_COLUMNS,
            "projects": projects,
            "events": [e.model_dump() for e in store.recent_events[-60:]][::-1],
            "alerts": _alerts(store),
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
        store.flush()
        return {"id": project.id, "status": project.status}

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
                _write_config(config, persist)
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
        # back in play; the next cycle's briefing will carry the directive.
        if task.column == "blocked":
            task.column = "ready"
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
        return _task_json(task, full=True)

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


def _task_json(task, full: bool = False) -> dict:
    data = {
        "id": task.id, "title": task.title, "expertise": task.expertise,
        "column": task.column, "assigned_agent_id": task.assigned_agent_id,
        "reviewer_agent_id": task.reviewer_agent_id,
        "turns_used": task.turns_used, "turn_limit": task.turn_limit,
        "review_rounds": task.review_rounds,
        "depends_on": task.depends_on,
        "review_verdict": task.review_verdict,
        "blocked_reason": task.blocked_reason,
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
    """Nested agent hierarchy for the project, orchestrator first."""
    agents = store.project_agents(project_id)
    by_id = {a.id: a for a in agents}
    roots = [a for a in agents if a.parent_agent_id is None or a.parent_agent_id not in by_id]

    def node(agent: AgentSpec, seen: set[str]) -> dict:
        children = []
        for child_id in agent.child_agent_ids:
            child = by_id.get(child_id)
            if child and child_id not in seen:
                seen.add(child_id)
                children.append(node(child, seen))
        return {**_agent_json(agent), "children": children}

    return [node(a, {a.id}) for a in roots]


def _alerts(store: Store) -> list[dict]:
    alerts: list[dict] = []
    for project in store.projects.values():
        if project.status == "ready_for_delivery":
            alerts.append({
                "level": "success", "project_id": project.id,
                "message": f"'{project.name}' is ready to deliver.",
            })
        for task in store.project_tasks(project.id):
            if task.column == "review":
                alerts.append({"level": "info", "project_id": project.id,
                               "message": f"'{task.title}' is waiting for review."})
            elif task.column == "blocked":
                alerts.append({
                    "level": "warn", "project_id": project.id,
                    "message": f"'{task.title}' is blocked"
                               + (f": {task.blocked_reason}" if task.blocked_reason
                                  else "."),
                })
    for agent in store.agents_in_status(
        ("budget_exhausted", "failed", "llm_unavailable")
    ):
        alerts.append({
            "level": "warn", "project_id": agent.project_id,
            "message": f"{agent.role} is {agent.status} "
                       f"({agent.turns_used}/{agent.turn_limit} turns).",
        })
    return alerts


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


def _write_config(config: HarnessConfig, path: str) -> None:
    """Persist the live config so a restart keeps the change."""
    target = Path(path).expanduser()
    payload = json.dumps(config.model_dump(), indent=2, sort_keys=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(payload)
    os.replace(tmp, target)
