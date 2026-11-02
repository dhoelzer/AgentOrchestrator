"""Multi-server LLM config, round-robin load balancing, and live config reload.

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


def _runtime(**overrides) -> tuple[HarnessConfig, Store, AgentRuntime]:
    tmp = tempfile.mkdtemp()
    cfg = HarnessConfig()
    cfg.data_dir = tmp
    for k, v in overrides.items():
        setattr(cfg, k, v)
    store = Store(tmp)
    return cfg, store, AgentRuntime(cfg, store)


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
    assert rt.gate.limit == 1
    assert rt._pool._max_workers == 5, rt._pool._max_workers
    print("pool 5 vs gate 1 ok")


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
    assert len(executing) == 1, "gate limit is 1, so exactly one may be executing"
    assert len(queued) == 2, "the two blocked runs must show as queued"
    assert rt.gate.active == 1
    hold.set()
    print("queued lane fills under backpressure ok")


def test_capacity_report_for_scheduling():
    _, store, rt = _runtime(max_parallel_workers=2, llm_queue_depth=3)
    cap = rt.capacity()
    assert cap["llm_slots"] == 2 and cap["slots_free"] == 2
    assert cap["queued"] == 0 and cap["pool_size"] == 5
    assert cap["llm_servers"] == ["primary"]
    print("capacity report ok")


# ---------- server config: openai + ollama style ----------

def test_model_key_is_sent_verbatim_per_server():
    """Each server names the literal model key for its request body."""
    openai_srv = LLMServerConfig(model="openai/qwen", base_url="http://gw/v1")
    ollama_srv = LLMServerConfig(model="qwen2.5:7b", provider="ollama_chat",
                                 base_url="http://a:11434/v1")
    assert openai_srv.route() == "openai/qwen"
    assert ollama_srv.route() == "ollama_chat/qwen2.5:7b"
    # An unprefixed model on an openai server still gets a usable route.
    assert LLMServerConfig(model="gpt-5", provider="openai").route() == "openai/gpt-5"
    assert ollama_srv.build_kwargs("u")["model"] == "ollama_chat/qwen2.5:7b"
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


# ---------- round-robin load balancing ----------

def test_requests_round_robin_across_servers():
    _, store, rt = _runtime()
    rt.config.llm_servers = _servers()
    picks = [rt.rotator.pick()[0].name for _ in range(7)]
    assert picks == ["primary", "openai-gw", "ollama-a", "primary",
                     "openai-gw", "ollama-a", "primary"], picks
    print("round-robin spread ok")


def test_each_run_is_bound_to_the_rotated_server():
    cfg, store, rt = _runtime()
    cfg.llm_servers = _servers()
    spec = _spec(store)
    seen: list[str] = []

    def fake(spec_, tools, message, turn_limit, *, server_name=None):
        seen.append(server_name)
        return _outcome()

    rt._run_guarded = fake
    for _ in range(3):
        rt.run(spec, [], "go")
    assert seen == ["primary", "openai-gw", "ollama-a"], seen
    # The queue card carries the server so the UI can show it.
    print("run bound to rotated server ok")


def test_queue_entry_exposes_server_and_failover():
    _, store, rt = _runtime()
    rt.config.llm_servers = _servers()
    entry = rt._enqueue_llm(_spec(store), "prompt", "worker")
    snap = rt.llm_queue()[0]
    assert snap["server"] == "primary"
    assert snap["skipped_servers"] is False
    print("queue entry carries server ok")


def test_dead_server_is_skipped_then_recovers():
    _, store, rt = _runtime()
    rt.config.llm_servers = _servers()
    rt.rotator.mark_failure("openai-gw", 60)
    picks = [rt.rotator.pick()[0].name for _ in range(6)]
    assert "openai-gw" not in picks, picks
    assert "primary" in picks and "ollama-a" in picks
    status = {s["name"]: s for s in rt.rotator.status()}
    assert status["openai-gw"]["cooling_down"] is True
    assert status["primary"]["cooling_down"] is False
    rt.rotator.mark_success("openai-gw")
    assert "openai-gw" in [rt.rotator.pick()[0].name for _ in range(3)]
    print("dead server skipped, then restored ok")


def test_all_servers_down_still_gets_the_primary():
    """A dead fleet must surface as a normal transient error, not 'no LLM'."""
    _, store, rt = _runtime()
    rt.config.llm_servers = _servers()
    for name in ("primary", "openai-gw", "ollama-a"):
        rt.rotator.mark_failure(name, 60)
    srv, skipped = rt.rotator.pick()
    assert srv.name == "primary" and skipped is True
    print("all-down fallback ok")


    class FakeConv:
        """Stand-in for a Conversation: real retry/health code, no network.

        Stubbing at the conversation (not _run_guarded) keeps the code under
        test - the transient-error handler that parks a server - on the path.
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
        rt.open_conversation = lambda spec_, tools, server=None: FakeConv(raiser)
        return rt.run(spec, [], "go")

    def test_connection_fault_parks_the_server_it_hit():
        import openai
        cfg, store, rt = _runtime()
        cfg.llm_servers = _servers()
        cfg.llm.empty_response_retries = 0

        def boom():
            raise openai.APIConnectionError(request=None)

        outcome = _run_with_error(rt, store, boom)  # lands on primary
        assert "server:primary:down" in (outcome.llm_health or ""), outcome.llm_health
        status = {s["name"]: s for s in rt.rotator.status()}
        assert status["primary"]["cooling_down"] is True, status
        assert status["primary"]["consecutive_failures"] == 1
        assert rt.rotator.pick()[0].name != "primary", "must rotate away from the fault"
        print("connection fault parks its server ok")

    def test_rate_limit_does_not_park_the_server():
        """A 429 means 'slow down', not 'endpoint is down' - no cooldown."""
        import openai
        cfg, store, rt = _runtime()
        cfg.llm_servers = _servers()
        cfg.llm.empty_response_retries = 0

        def limited():
            raise openai.RateLimitError(message="slow down", response=None, body=None)

        _run_with_error(rt, store, limited)
        assert all(not s["cooling_down"] for s in rt.rotator.status())
        print("rate limit does not park ok")



def test_rotation_cursor_survives_a_restart():
    tmp = tempfile.mkdtemp()
    cfg = HarnessConfig(); cfg.data_dir = tmp; cfg.llm_servers = _servers()
    store = Store(tmp)
    rt1 = AgentRuntime(cfg, store)
    rt1.rotator.pick(); rt1.rotator.pick(); rt1.rotator.persist()
    rt2 = AgentRuntime(cfg, Store(tmp))
    assert rt2.rotator.pick()[0].name == "ollama-a", "cursor lost across restart"
    print("rotation cursor persisted ok")


def test_agent_llm_is_built_from_its_server():
    _, store, rt = _runtime()
    rt.config.llm_servers = _servers()
    spec = _spec(store, id="agent-x")
    agent = rt.build_agent(spec, [], "ollama-a")
    assert agent.llm.model == "ollama_chat/qwen2.5:7b"
    assert agent.llm.base_url == "http://a:11434/v1"
    assert agent.llm.usage_id == "agent-x@ollama-a", "metrics must be per server"
    assert agent.condenser.llm.model == "ollama_chat/qwen2.5:7b"
    print("agent built from its server ok")


def test_switching_servers_repoints_a_live_conversation():
    _, store, rt = _runtime()
    rt.config.llm_servers = _servers()
    spec = _spec(store, id="agent-y")
    rt.open_conversation(spec, [], "primary")
    assert rt._conversations["agent-y"].agent.llm.base_url == rt.config.llm.base_url
    rt._bind_server(spec, "openai-gw")
    conv = rt._conversations["agent-y"]
    assert conv.agent.llm.base_url == "http://gw:8080/v1/"
    assert conv.agent.llm.usage_id == "agent-y@openai-gw"
    # Timeout retuning must still reach the swapped-in LLM.
    assert any(llm is conv.agent.llm for llm in rt._llms["agent-y"])
    print("live conversation re-pointed ok")


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
    rt = AgentRuntime(loaded, store)
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
    assert rt.gate.limit == 5, "gate must follow the new parallel limit"
    assert rt._pool._max_workers == 5 + rt.config.llm_queue_depth
    assert [s.name for s in rt.rotator.servers()] == [
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
    assert rt.gate.limit == 1
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
    assert [s.name for s in rt.rotator.servers()] == ["primary"]
    assert "ollama-a" not in [rt.rotator.pick()[0].name for _ in range(5)]
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


def test_idle_agents_move_off_a_removed_server_running_ones_drain():
    cfg_path, _, store, rt, orch = _orchestrator_with_config_file()
    idle = store.upsert_agent(_spec(store, id="idle-1"))
    busy = store.upsert_agent(_spec(store, id="busy-1"))
    rt.open_conversation(idle, [], "ollama-a")
    rt.open_conversation(busy, [], "ollama-a")
    rt._mark_inflight("busy-1")

    raw = json.loads(cfg_path.read_text())
    raw["llm_servers"] = []
    cfg_path.write_text(json.dumps(raw, indent=2))
    time.sleep(0.01)
    notes = orch.reload_config_if_changed()

    assert rt._conversations["idle-1"].agent.llm.usage_id == "idle-1@primary"
    assert rt._conversations["busy-1"].agent.llm.usage_id == "busy-1@ollama-a", (
        "a mid-run agent must finish on the endpoint it started with")
    assert any("1 idle agent" in n for n in notes), notes
    assert any("1 running agent" in n for n in notes), notes

    rt._unmark_inflight("busy-1")
    assert rt.rebind_stale_servers() == (1, 0)
    assert rt._conversations["busy-1"].agent.llm.usage_id == "busy-1@primary"
    print("idle rebound / running drain ok")


def test_broken_config_keeps_the_previous_settings():
    cfg_path, _, store, rt, orch = _orchestrator_with_config_file()
    cfg_path.write_text("{ not valid json")
    time.sleep(0.01)
    assert orch.reload_config_if_changed() == []
    assert rt.gate.limit == 2, "a bad edit must not change live settings"
    assert [s.name for s in rt.rotator.servers()] == ["primary", "ollama-a"]
    assert any(e.kind == "config_reload_failed" for e in store.recent_events)
    # And the loop keeps working: the next good edit still applies.
    cfg_path.write_text(json.dumps({"max_parallel_workers": 4}))
    time.sleep(0.01)
    orch.reload_config_if_changed()
    assert rt.gate.limit == 4
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
    assert "6 of 6 LLM slots free" in text
    print("config change reaches the briefing ok")


def test_api_patch_does_not_reload_its_own_write():
    cfg_path, cfg, store, rt, orch = _orchestrator_with_config_file()
    app = build_api(cfg, store, orch, config_path=str(cfg_path))
    client = TestClient(app)
    out = client.post("/api/config", json={"max_parallel_workers": 3})
    assert out.status_code == 200, out.text
    assert rt.gate.limit == 3
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
    assert [s.name for s in rt.rotator.servers()] == ["primary", "ollama-b"]
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
    assert [s.name for s in rt.rotator.servers()] == ["primary", "ollama-a"]
    print("bad server lists rejected ok")


def test_queue_depth_is_patchable():
    _, cfg, store, rt, orch = _orchestrator_with_config_file()
    client = TestClient(build_api(cfg, store, orch))
    before = rt._pool._max_workers
    res = client.post("/api/config", json={"llm_queue_depth": 20})
    assert res.status_code == 200, res.text
    assert rt.config.llm_queue_depth == 20
    assert rt._pool._max_workers == rt.config.max_parallel_workers + 20
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
    assert "qserver" in js and ".qcard .qserver" in html, "server on queue card"
    # The header pill must show the fleet without opening Settings.
    assert "llm_fleet" in js and "cooling_down" in js
    print("server UI contract ok")


def test_state_exposes_fleet_and_capacity():
    _, cfg, store, rt, orch = _orchestrator_with_config_file()
    client = TestClient(build_api(cfg, store, orch))
    meta = client.get("/api/state").json()["meta"]
    assert [s["name"] for s in meta["llm_fleet"]] == ["primary", "ollama-a"]
    assert meta["llm_fleet"][1]["provider"] == "ollama_chat"
    assert meta["capacity"]["llm_slots"] == 2
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
