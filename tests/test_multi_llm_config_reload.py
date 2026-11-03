"""Multi-server LLM config, the LLM router, and live config reload.

Run: python3 tests/test_multi_llm_config_reload.py

No LLM endpoint is contacted: runs are stubbed at _run_guarded, which is the
boundary the harness itself uses.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, ".")

from fastapi.testclient import TestClient

from agent_harness.config import HarnessConfig, LLMServerConfig
from agent_harness.llm_router import make_router
from agent_harness.orchestrator import Orchestrator
from agent_harness.runtime import AgentRuntime, RunOutcome
from agent_harness.store import AgentSpec, Store
from agent_harness.tools import register_harness_tools
from agent_harness.web.api import build_api

register_harness_tools()


def _outcome(turns=1) -> RunOutcome:
    return RunOutcome(turns_used=turns, turns_this_run=turns, status="idle",
                      final_report="done", budget_exhausted=False,
                      tokens=0, cost_usd=0.0)


def _runtime(with_router: bool = False,
             **overrides) -> tuple[HarnessConfig, Store, AgentRuntime]:
    tmp = tempfile.mkdtemp()
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    for k, v in overrides.items():
        setattr(cfg, k, v)
    store = Store(tmp)
    router = make_router(cfg, store) if with_router else None
    return cfg, store, AgentRuntime(cfg, store, router=router)


def _spec(store: Store, role: str = "worker", **kw) -> AgentSpec:
    spec = AgentSpec(role=role, specialty="x", mission="m", system_prompt="p",
                     project_id="p1", depth=1, **kw)
    return store.upsert_agent(spec)


def _servers() -> list[LLMServerConfig]:
    return [
        LLMServerConfig(name="openai-gw", provider="openai",
                        base_url="http://gw:8080/v1/", model="openai/qwen"),
        LLMServerConfig(name="ollama-a", provider="ollama_chat",
                        base_url="http://a:11434/v1", model="qwen2.5:7b"),
    ]


# ---------- the queued lane must actually fill up ----------

def test_pool_is_larger_than_the_gate():
    """The bug: pool == gate meant a dispatched run always had a free slot, so
    the backlog hid in the thread pool and the queue strip stayed empty."""
    cfg, _, rt = _runtime(max_parallel_workers=1, llm_queue_depth=4)
    # 1 worker slot + 1 reserved for the orchestrator.
    assert rt.gate.limit == 2
    assert rt._pool._max_workers == 2 + 4, rt._pool._max_workers
    print("pool 6 vs gate 2 ok")


def test_saturated_gate_shows_waiting_agents_as_queued():
    cfg, store, rt = _runtime(max_parallel_workers=1, llm_queue_depth=4)
    hold = threading.Event()
    specs = [_spec(store, role=f"worker{i}") for i in range(3)]

    def slow(spec, tools, message, turn_limit, *, server_name=None):
        hold.wait(10)
        return _outcome()

    rt._run_guarded = slow
    for s in specs:
        rt.submit(s.id, lambda s=s: rt.run(s, [], "go"))

    deadline = time.time() + 5
    while len(rt.llm_queue()) < 3 and time.time() < deadline:
        time.sleep(0.05)
    q = rt.llm_queue()
    assert len(q) == 3, f"only {len(q)} visible; backlog is hiding in the pool"
    executing = [e for e in q if e["status"] == "executing"]
    queued = [e for e in q if e["status"] == "queued"]
    assert len(executing) == rt.gate.limit, \
        f"gate limit is {rt.gate.limit}, so that many may execute"
    assert len(queued) == len(specs) - rt.gate.limit, \
        "the runs beyond the gate must show as queued"
    assert rt.gate.active == rt.gate.limit
    hold.set()
    print("queued lane fills under backpressure ok")


def test_capacity_report_for_scheduling():
    _, store, rt = _runtime(max_parallel_workers=2, llm_queue_depth=3)
    cap = rt.capacity()
    assert cap["worker_slots"] == 2 and cap["slots_free"] == 2
    assert cap["llm_slots"] == 3, "one slot is reserved for the orchestrator"
    assert cap["queued"] == 0 and cap["pool_size"] == 3 + 3
    assert cap["llm_servers"] == ["primary"]
    print("capacity report ok")


# ---------- server config: openai + ollama style ----------

def test_model_key_is_sent_verbatim_per_server():
    """Two different strings, and conflating them breaks Ollama.

    `wire_model()` is what goes in the request body's `model` key: the literal
    name the endpoint knows, because every endpoint is spoken to over the OpenAI
    protocol and Ollama matches that string against its own model list.
    `route()` is a litellm model string used only when an SDK LLM is built to
    address an endpoint directly; prefixing the wire key makes Ollama 404.
    """
    openai_srv = LLMServerConfig(model="openai/qwen", base_url="http://gw/v1")
    ollama_srv = LLMServerConfig(model="qwen3:30b", provider="ollama_chat",
                                 base_url="http://a:11434/v1")
    # Wire: verbatim, no prefix, for every provider.
    assert openai_srv.wire_model() == "openai/qwen"
    assert ollama_srv.wire_model() == "qwen3:30b"
    assert LLMServerConfig(model="gpt-5", provider="openai").wire_model() == "gpt-5"
    # Route: provider-prefixed, for the direct-SDK fallback only.
    assert openai_srv.route() == "openai/qwen"
    assert ollama_srv.route() == "ollama_chat/qwen3:30b"
    assert LLMServerConfig(model="gpt-5", provider="openai").route() == "openai/gpt-5"
    assert ollama_srv.build_kwargs("u")["model"] == "ollama_chat/qwen3:30b"
    assert ollama_srv.build_kwargs("u")["base_url"] == "http://a:11434/v1"
    print("per-server model key ok")


def test_ollama_extra_body_reaches_the_request():
    srv = LLMServerConfig(name="o", provider="ollama_chat", base_url="http://a/v1",
                          model="qwen2.5:7b",
                          extra_body={"num_ctx": 32768, "keep_alive": "10m"})
    assert srv.build_kwargs("u")["litellm_extra_body"] == {
        "num_ctx": 32768, "keep_alive": "10m"}
    print("ollama extra_body ok")


def test_primary_llm_is_always_in_the_rotation():
    cfg = HarnessConfig()
    cfg.llm_servers = _servers()
    names = [s.name for s in cfg.servers()]
    assert names == ["primary", "openai-gw", "ollama-a"], names
    # No llm_servers -> single server, so pre-existing configs need no changes.
    assert [s.name for s in HarnessConfig().servers()] == ["primary"]
    print("primary always first ok")


def test_duplicate_endpoint_listed_twice_is_dropped():
    cfg = HarnessConfig()
    cfg.llm_servers = [
        LLMServerConfig(name="a", base_url="http://x/v1", model="openai/qwen"),
        LLMServerConfig(name="b", base_url="http://x/v1", model="openai/qwen"),
    ]
    names = [s.name for s in cfg.servers()]
    assert len(names) == 2 and names[0] == "primary", names
    assert set(names[1:]) & {"a", "b"} == {"a"} or set(names[1:]) == {"a"}, names
    print("duplicate endpoint deduped ok")


def test_duplicate_server_names_rejected():
    dup = [LLMServerConfig(name="x", base_url="http://one/v1", model="m1"),
           LLMServerConfig(name="x", base_url="http://two/v1", model="m2")]
    try:
        HarnessConfig(llm_servers=dup)
        raise AssertionError("duplicate names must be refused")
    except ValueError:
        pass
    # A name that collides with the implicit primary is refused too.
    try:
        HarnessConfig(llm_servers=[LLMServerConfig(
            name="primary", base_url="http://two/v1", model="m2")])
        raise AssertionError("primary is reserved")
    except ValueError:
        pass
    print("unique server names enforced ok")


def test_env_can_define_servers():
    import os
    os.environ["HARNESS_LLM_SERVERS"] = json.dumps([
        {"name": "env-ollama", "provider": "ollama", "base_url": "http://e:11434/v1",
         "model": "llama3"}])
    try:
        cfg = HarnessConfig.load(Path(tempfile.mkdtemp()) / "absent.json")
        assert [s.name for s in cfg.servers()] == ["primary", "env-ollama"]
        assert cfg.server_named("env-ollama").route() == "ollama/llama3"
    finally:
        del os.environ["HARNESS_LLM_SERVERS"]
    print("HARNESS_LLM_SERVERS ok")


# ---------- the router owns endpoint choice ----------

def test_agents_are_built_against_the_router_not_an_endpoint():
    """The core contract: an agent's LLM points at the router."""
    cfg, store, rt = _runtime(with_router=True,
                              llm_servers=_servers())
    spec = _spec(store, id="agent-route")
    agent = rt.build_agent(spec, [])
    assert agent.llm.base_url == rt.router.base_url, agent.llm.base_url
    assert agent.llm.base_url.endswith("/v1")
    assert agent.condenser.llm.base_url == rt.router.base_url
    # The model is a placeholder: only the router knows which endpoint answers.
    assert agent.llm.model == "openai/harness-routed"
    # Attribution headers are what make the router's cards meaningful.
    hdrs = agent.llm.extra_headers
    assert hdrs["x-harness-agent"] == "agent-route"
    assert hdrs["x-harness-project"] == "p1"
    print("agents point at the router ok")


def test_queue_entry_carries_no_server_until_dispatch():
    """Nothing is pre-assigned: the endpoint is chosen when it is free."""
    _, store, rt = _runtime(llm_servers=_servers())
    rt._enqueue_llm(_spec(store), "prompt", "worker")
    snap = rt.llm_queue()[0]
    assert "server" not in snap or snap.get("server") is None
    print("queue entry carries no pre-assigned server ok")


def _pick_and_release(rt) -> str | None:
    """Take the endpoint the router would use next, then free it again.

    Assignment has to update last_assigned the way the dispatcher does: the
    tie-break between equally-free endpoints is "least recently assigned".
    """
    with rt.router._lock:
        srv = rt.router._free_server_locked(time.monotonic())
        if srv is None:
            return None
        srv.in_flight += 1
        srv.last_assigned = time.monotonic()
        name = srv.name
    rt.router._release(name)
    return name


def test_router_spreads_requests_across_endpoints():
    """Equally-free endpoints are used in turn, not all piled on one."""
    cfg, store, rt = _runtime(with_router=True, llm_servers=_servers())
    rt.router.sync_servers()
    picks = [_pick_and_release(rt) for _ in range(7)]
    assert set(picks) == {"primary", "openai-gw", "ollama-a"}, picks
    print("router spread ok:", picks)


def test_router_parks_a_failing_endpoint_then_recovers():
    cfg, store, rt = _runtime(with_router=True, llm_servers=_servers())
    rt.router.sync_servers()
    rt.router._mark_failure("openai-gw", 500, "upstream refused")
    picks = {n for n in (_pick_and_release(rt) for _ in range(6)) if n}
    assert "openai-gw" not in picks, picks
    status = {s["name"]: s for s in rt.router.status()["servers"]}
    assert status["openai-gw"]["cooling_down"] is True
    assert status["primary"]["cooling_down"] is False
    rt.router._note_probe("openai-gw", True, None)
    picks2 = {n for n in (_pick_and_release(rt) for _ in range(6)) if n}
    assert "openai-gw" in picks2, picks2
    print("router parks a failing endpoint, then restores it ok")


def test_router_sends_every_configured_server_including_primary():
    cfg, store, rt = _runtime(with_router=True, llm_servers=_servers())
    rt.router.sync_servers()
    names = [s["name"] for s in rt.router.status()["servers"]]
    assert names == ["primary", "openai-gw", "ollama-a"], names
    print("primary always in the router ok")


    class FakeConv:
        """Stand-in for a Conversation: real retry/health code, no network.

        Stubbing at the conversation keeps the runtime's own retry and timeout
        logic on the path; endpoint health is no longer the runtime's job.
        """

        def __init__(self, raiser):
            self._raiser = raiser
            self.agent = type("A", (), {
                "llm": None, "condenser": None})()
            self.state = type("S", (), {"events": []})()
            self.max_iteration_per_run = 0
            self.turns = 0

        def send_message(self, message):
            pass

        def run(self):
            self._raiser()
            # A successful run appends an agent-sourced event per turn.
            self.turns += 1
            self.state.events.append(_agent_event(f"resp-{self.turns}"))

    def _agent_event(rid: str):
        return type("E", (), {"source": "agent", "llm_response_id": rid,
                              "id": rid})()

    def _run_with_error(rt, store, raiser):
        spec = _spec(store)
        rt.open_conversation = lambda spec_, tools: FakeConv(raiser)
        return rt.run(spec, [], "go")

    def test_connection_fault_widens_the_timeout_without_touching_endpoints():
        """Endpoint parking moved to the router; the runtime keeps the timeout."""
        import openai
        cfg, store, rt = _runtime(llm_servers=_servers())
        cfg.llm.empty_response_retries = 0

        def boom():
            raise openai.APIConnectionError(request=None)

        outcome = _run_with_error(rt, store, boom)
        assert "transient:APIConnectionError" in (outcome.llm_health or ""), \
            outcome.llm_health
        assert rt.llm_timeout > cfg.llm.min_request_timeout, "timeout must widen"
        print("connection fault widens timeout, endpoints left to the router ok")

    def test_rate_limit_does_not_widen_the_timeout():
        """A 429 means 'slow down', not 'the endpoint is slow or down'."""
        import openai
        cfg, store, rt = _runtime(llm_servers=_servers())
        cfg.llm.empty_response_retries = 0
        before = rt.llm_timeout

        def limited():
            raise openai.RateLimitError(message="slow down", response=None, body=None)

        _run_with_error(rt, store, limited)
        assert rt.llm_timeout == before
        print("rate limit does not widen timeout ok")


# ---------- live config reload at the start of a tick ----------

def _orchestrator_with_config_file() -> tuple[Path, HarnessConfig, Store,
                                              AgentRuntime, Orchestrator]:
    tmp = tempfile.mkdtemp()
    cfg_path = Path(tmp) / "config.json"
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    cfg.max_parallel_workers = 2
    cfg.llm_servers = [LLMServerConfig(name="ollama-a", provider="ollama_chat",
                                       base_url="http://a:11434/v1",
                                       model="qwen2.5:7b")]
    cfg_path.write_text(cfg.model_dump_json(indent=2))
    loaded = HarnessConfig.load(cfg_path)
    store = Store(tmp)
    router = make_router(loaded, store)
    rt = AgentRuntime(loaded, store, router=router)
    orch = Orchestrator(loaded, store, rt, config_path=str(cfg_path))
    return cfg_path, loaded, store, rt, orch


def test_unchanged_config_triggers_no_reload():
    _, _, _, _, orch = _orchestrator_with_config_file()
    assert orch.reload_config_if_changed() == []
    assert orch.reload_config_if_changed() == []
    print("no-op reload ok")


def test_added_server_and_more_slots_are_applied():
    cfg_path, _, store, rt, orch = _orchestrator_with_config_file()
    raw = json.loads(cfg_path.read_text())
    raw["llm_servers"].append({"name": "ollama-b", "provider": "ollama_chat",
                               "base_url": "http://b:11434/v1",
                               "model": "qwen2.5:14b"})
    raw["max_parallel_workers"] = 5
    cfg_path.write_text(json.dumps(raw, indent=2))
    time.sleep(0.01)

    notes = orch.reload_config_if_changed()
    # max_parallel_workers counts WORKER slots; the gate reserves one more for
    # the orchestrator so orchestrator cycles never consume worker capacity.
    assert rt.gate.limit == 5 + 1, "gate must follow the new parallel limit"
    assert rt.capacity()["worker_slots"] == 5, rt.capacity()
    assert rt._pool._max_workers == (5 + 1) + rt.config.llm_queue_depth
    assert [x["name"] for x in rt.router.status()["servers"]] == [
        "primary", "ollama-a", "ollama-b"]
    assert any("5" in n and "more agents can run" in n for n in notes), notes
    assert any("ollama-b" in n for n in notes), notes
    assert any(e.kind == "config_reloaded" for e in store.recent_events)
    print("added server + slots applied ok")


def test_reduced_slots_drain_rather_than_kill():
    cfg_path, _, _, rt, orch = _orchestrator_with_config_file()
    raw = json.loads(cfg_path.read_text())
    raw["max_parallel_workers"] = 1
    cfg_path.write_text(json.dumps(raw, indent=2))
    time.sleep(0.01)
    notes = orch.reload_config_if_changed()
    assert rt.gate.limit == 1 + 1
    assert any("drain" in n for n in notes), notes
    # The pool ceiling is grow-only: surplus threads idle, the gate does the
    # bounding, so in-flight runs are never interrupted.
    assert rt._pool._max_workers >= 1
    print("reduced slots drain ok")


def test_removed_server_stops_being_scheduled():
    cfg_path, _, _, rt, orch = _orchestrator_with_config_file()
    raw = json.loads(cfg_path.read_text())
    raw["llm_servers"] = []
    cfg_path.write_text(json.dumps(raw, indent=2))
    time.sleep(0.01)
    notes = orch.reload_config_if_changed()
    names = [x["name"] for x in rt.router.status()["servers"]]
    assert names == ["primary"], names
    picks = {n for n in (_pick_and_release(rt) for _ in range(5)) if n}
    assert "ollama-a" not in picks, picks
    assert any("removed ollama-a" in n for n in notes), notes
    print("removed server unscheduled ok")


def test_reconfigured_server_under_a_kept_name_is_caught():
    cfg_path, _, _, rt, orch = _orchestrator_with_config_file()
    raw = json.loads(cfg_path.read_text())
    raw["llm_servers"][0]["model"] = "qwen2.5:72b"
    cfg_path.write_text(json.dumps(raw, indent=2))
    time.sleep(0.01)
    notes = orch.reload_config_if_changed()
    assert rt.config.server_named("ollama-a").model == "qwen2.5:72b"
    assert any("reconfigured ollama-a" in n for n in notes), notes
    print("reconfigured server caught ok")


def test_server_changes_leave_conversations_pointing_at_the_router():
    """No rebinding step exists any more.

    The old design had to move conversations onto a live endpoint, and had to
    leave mid-run ones alone to drain. Agents never hold an endpoint now, so a
    server change lands in the router and every conversation is already correct.
    """
    cfg_path, _, store, rt, orch = _orchestrator_with_config_file()
    idle = store.upsert_agent(_spec(store, id="idle-1"))
    busy = store.upsert_agent(_spec(store, id="busy-1"))
    rt.open_conversation(idle, [])
    rt.open_conversation(busy, [])
    rt._mark_inflight("busy-1")
    before = {a: rt._conversations[a].agent.llm.base_url for a in ("idle-1", "busy-1")}

    raw = json.loads(cfg_path.read_text())
    raw["llm_servers"] = []
    cfg_path.write_text(json.dumps(raw, indent=2))
    time.sleep(0.01)
    notes = orch.reload_config_if_changed()

    assert [x["name"] for x in rt.router.status()["servers"]] == ["primary"]
    for agent_id, url in before.items():
        assert rt._conversations[agent_id].agent.llm.base_url == url
        assert url == rt.router.base_url, "must still address the router"
    assert not any("idle agent" in n for n in notes), notes
    print("server change needs no rebinding ok")


def test_broken_config_keeps_the_previous_settings():
    cfg_path, _, store, rt, orch = _orchestrator_with_config_file()
    cfg_path.write_text("{ not valid json")
    time.sleep(0.01)
    assert orch.reload_config_if_changed() == []
    assert rt.gate.limit == 3, "a bad edit must not change live settings"
    assert [x["name"] for x in rt.router.status()["servers"]] == ["primary", "ollama-a"]
    assert any(e.kind == "config_reload_failed" for e in store.recent_events)
    # And the loop keeps working: the next good edit still applies.
    cfg_path.write_text(json.dumps({"max_parallel_workers": 4}))
    time.sleep(0.01)
    orch.reload_config_if_changed()
    assert rt.gate.limit == 4 + 1
    print("bad config survives ok")


def test_tick_starts_with_a_config_check():
    """The reload must happen inside run_cycle, before any work is scheduled."""
    cfg_path, _, store, rt, orch = _orchestrator_with_config_file()
    calls: list[str] = []
    orch.reload_config_if_changed = lambda: calls.append("reload") or []
    orch.run_cycle()
    assert calls == ["reload"], "run_cycle must re-read config first"
    print("tick starts with a config check ok")


def test_config_change_reaches_the_orchestrator_briefing():
    cfg_path, _, store, rt, orch = _orchestrator_with_config_file()
    store.create_project("Payments", "charge/refund endpoints",
                         tempfile.mkdtemp())
    raw = json.loads(cfg_path.read_text())
    raw["max_parallel_workers"] = 6
    cfg_path.write_text(json.dumps(raw, indent=2))
    time.sleep(0.01)

    briefings: list[str] = []

    def capture(spec, tools, message, turn_limit=None, kind=None):
        briefings.append(message)
        return _outcome()

    rt.run = capture
    orch.run_cycle()
    assert briefings, "orchestrator was never briefed"
    text = briefings[0]
    assert "CONFIG CHANGED" in text, text[-800:]
    assert "CAPACITY NOW" in text
    # The edit raised max_parallel_workers to 6, and none of it is busy.
    assert "6 of 6 worker slots free" in text, text[-800:]
    print("config change reaches the briefing ok")


def test_api_patch_does_not_reload_its_own_write():
    cfg_path, cfg, store, rt, orch = _orchestrator_with_config_file()
    app = build_api(cfg, store, orch, config_path=str(cfg_path))
    client = TestClient(app)
    out = client.post("/api/config", json={"max_parallel_workers": 3})
    assert out.status_code == 200, out.text
    assert rt.gate.limit == 3 + 1
    assert orch.reload_config_if_changed() == [], (
        "the API's own write must not look like an external edit")
    print("API write not reloaded as external ok")


def test_api_can_configure_servers_and_reports_health():
    cfg_path, cfg, store, rt, orch = _orchestrator_with_config_file()
    app = build_api(cfg, store, orch, config_path=str(cfg_path))
    client = TestClient(app)

    got = client.get("/api/config").json()
    assert [s["name"] for s in got["llm_servers"]] == ["primary", "ollama-a"]
    assert got["config"]["llm_servers"][0]["api_key"] in ("***", "")

    out = client.post("/api/config", json={"llm_servers": [
        {"name": "ollama-b", "provider": "ollama_chat",
         "base_url": "http://b:11434/v1", "model": "qwen2.5:14b"},
    ], "persist": str(cfg_path)})
    assert out.status_code == 200, out.text
    assert [x["name"] for x in rt.router.status()["servers"]] == ["primary", "ollama-b"]
    # Persisted, and reloadable from disk.
    on_disk = json.loads(cfg_path.read_text())
    assert on_disk["llm_servers"][0]["name"] == "ollama-b"
    assert HarnessConfig.load(cfg_path).server_named("ollama-b").model == "qwen2.5:14b"
    print("API server config + health ok")


def test_api_rejects_bad_server_lists():
    _, cfg, store, rt, orch = _orchestrator_with_config_file()
    client = TestClient(build_api(cfg, store, orch))
    for bad in ([{"name": "dup"}, {"name": "dup"}],
                [{"name": "ok", "base_url": "http://x/v1", "model": "m"},
                 {"name": "ok", "base_url": "http://y/v1", "model": "m"}],
                "not a list",
                [{"provider": "not-a-provider"}]):
        res = client.post("/api/config", json={"llm_servers": bad})
        assert res.status_code == 400, (bad, res.status_code, res.text)
    assert [x["name"] for x in rt.router.status()["servers"]] == ["primary", "ollama-a"]
    print("bad server lists rejected ok")


def test_queue_depth_is_patchable():
    _, cfg, store, rt, orch = _orchestrator_with_config_file()
    client = TestClient(build_api(cfg, store, orch))
    before = rt._pool._max_workers
    res = client.post("/api/config", json={"llm_queue_depth": 20})
    assert res.status_code == 200, res.text
    assert rt.config.llm_queue_depth == 20
    assert rt._pool._max_workers == rt.gate.limit + 20
    print("queue depth patchable ok")


def test_old_config_without_servers_still_loads():
    """A config.json written before multi-server support must keep working."""
    tmp = tempfile.mkdtemp()
    cfg_path = Path(tmp) / "config.json"
    cfg_path.write_text(json.dumps({
        "llm": {"base_url": "http://old:8080/v1/", "model": "openai/legacy"},
        "max_parallel_workers": 2,
    }))
    cfg = HarnessConfig.load(cfg_path)
    assert cfg.llm_servers == []
    assert [s.name for s in cfg.servers()] == ["primary"]
    assert cfg.llm_queue_depth == 8, "new field must default, not break"
    assert cfg.server_named("primary").model == "openai/legacy"
    print("legacy config loads ok")


def test_ui_wires_servers_and_queue_server():
    root = Path(__file__).resolve().parents[1]
    js = (root / "agent_harness" / "web" / "static" / "app.js").read_text()
    html = (root / "agent_harness" / "web" / "templates" / "index.html").read_text()
    assert "function renderServers" in js, "server editor missing"
    assert "collectServers" in js and "llm_servers" in js
    assert 'data-srvdel' in js, "remove-server control missing"
    assert "id=\"serverRows\"" in html and "id=\"addServer\"" in html
    assert "llm_queue_depth" in html, "queue depth control missing"
    # The router strip: request cards plus one card per endpoint.
    assert "function renderRouter" in js, "router strip missing"
    assert 'data-req=' in js and "renderRouter(state.router" in js
    assert ".rcard" in html and ".scard" in html, "router card styles"
    # A Test button per server row, so a new LLM is proven before it is saved.
    assert "data-srvtest" in js and "/api/llm/router/test" in js
    assert "llm_server_concurrency" in html, "per-endpoint concurrency control"
    # The header pill must show the fleet without opening Settings.
    assert "llm_router" in js and "cooling_down" in js
    print("server UI contract ok")


def test_state_exposes_fleet_and_capacity():
    _, cfg, store, rt, orch = _orchestrator_with_config_file()
    client = TestClient(build_api(cfg, store, orch))
    meta = client.get("/api/state").json()["meta"]
    router = meta["llm_router"]
    assert [s["name"] for s in router["servers"]] == ["primary", "ollama-a"]
    assert set(router) >= {"requests", "recent", "servers", "counts"}, router.keys()
    assert router["counts"]["servers"] == 2
    # 2 worker slots + 1 reserved for the orchestrator.
    assert meta["capacity"]["llm_slots"] == 3
    assert meta["capacity"]["worker_slots"] == 2
    assert meta["capacity"]["slots_free"] == 2
    print("state exposes fleet + capacity ok")


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"--- {t.__name__}")
        t()
    print(f"\nALL MULTI-LLM/CONFIG-RELOAD TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
