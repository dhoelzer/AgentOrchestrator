"""The LLM router: the single place every agent's LLM request goes through.

Run: python3 tests/test_llm_router.py

Upstreams are local fake OpenAI-compatible servers, so the wire contract
(headers, model rewriting, SSE framing, chunked bodies) is exercised for real
without depending on a live model.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, ".")

import httpx

from agent_harness.config import HarnessConfig, LLMConfig, LLMServerConfig
from agent_harness.llm_router import make_router
from agent_harness.store import Store


class FakeUpstream:
    """Minimal OpenAI-compatible endpoint with realistic HTTP/1.1 framing."""

    def __init__(self, name: str, port: int, *, latency: float = 0.0,
                 tokens: int = 8, fail: tuple[int, str] | None = None,
                 stream_latency: float = 0.0,
                 known_models: set[str] | None = None):
        self.name, self.port = name, port
        self.latency, self.tokens, self.fail = latency, tokens, fail
        self.stream_latency = stream_latency
        # When set, the endpoint only answers for these exact model names and
        # 404s otherwise - how Ollama behaves, since it looks the string up in
        # its own model list rather than ignoring it.
        self.known_models = known_models
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                # The router probes /models to decide reachability; a fake that
                # 404s here would look like a dead endpoint even though it
                # answers chat requests fine.
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                names = sorted(outer.known_models or {outer.name})
                payload = json.dumps(
                    {"data": [{"id": m} for m in names]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append(body)
                outer.headers.append(dict(self.headers))
                if outer.known_models is not None \
                        and body.get("model") not in outer.known_models:
                    payload = json.dumps(
                        {"error": f"model '{body.get('model')}' not found"}).encode()
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if outer.fail:
                    code, msg = outer.fail
                    payload = json.dumps({"error": {"message": msg}}).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if outer.latency:
                    time.sleep(outer.latency)
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()

                    def emit(obj):
                        line = f"data: {json.dumps(obj)}\n\n".encode()
                        self.wfile.write(b"%X\r\n" % len(line) + line + b"\r\n")
                        self.wfile.flush()

                    emit({"id": "x", "object": "chat.completion.chunk",
                          "choices": [{"index": 0,
                                       "delta": {"content": "hi"},
                                       "finish_reason": None}]})
                    if outer.stream_latency:
                        time.sleep(outer.stream_latency)
                    emit({"id": "x", "object": "chat.completion.chunk",
                          "choices": [], "finish_reason": "stop"})
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(b"%X\r\n" % len(done) + done + b"\r\n")
                    # A chunked body is not finished until the zero-length chunk
                    # arrives; without it the reader waits for more and times out.
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                    return
                payload = json.dumps({
                    "id": "x", "object": "chat.completion",
                    "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": f"{name}-reply"}}],
                    "usage": {"prompt_tokens": 11,
                              "completion_tokens": outer.tokens,
                              "total_tokens": 11 + outer.tokens},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._http.daemon_threads = True
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def stop(self):
        self._http.shutdown()
        self._http.server_close()


def _router(**servers):
    cfg = HarnessConfig()
    cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="primary",
                        base_url="http://127.0.0.1:8901/v1", model="openai/prim")
    cfg.llm_servers = list(servers.values())
    r = make_router(cfg, Store(cfg.data_dir))
    r.start()
    return cfg, r


def test_ollama_server_gets_its_literal_model_key():
    """A provider prefix in the body makes Ollama 404: it looks the string up
    in its own model list. The wire key must be exactly what was configured."""
    for provider in ("ollama", "ollama_chat"):
        up = FakeUpstream("ollama", 8917, known_models={"qwen3:30b"})
        cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
        # The Ollama endpoint is the only one in the fleet, so the assertion is
        # about the key it receives, not about which endpoint was chosen.
        cfg.llm = LLMConfig(name="ollama-a", provider=provider,
                            base_url="http://127.0.0.1:8917/v1", model="qwen3:30b")
        cfg.llm_servers = []
        r = make_router(cfg, Store(cfg.data_dir)); r.start()
        resp = httpx.post(r.base_url + "/chat/completions",
                          json={"model": "openai/harness-routed",
                                "messages": [{"role": "user", "content": "hi"}]},
                          timeout=30)
        assert resp.status_code == 200, (provider, resp.text)
        sent = up.requests[-1]["model"]
        assert sent == "qwen3:30b", f"{provider}: upstream saw {sent!r}"
        assert "/" not in sent or sent.startswith("qwen"), sent
        r.stop(); up.stop()
    print("ollama literal model key ok (both providers)")


def test_ollama_base_url_is_used_verbatim():
    """No /v1 is added, stripped, or rewritten for an Ollama endpoint."""
    up = FakeUpstream("ollama", 8918, known_models={"qwen3:30b"})
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="ollama-a", provider="ollama_chat",
                        base_url="http://127.0.0.1:8918/v1", model="qwen3:30b")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "openai/harness-routed",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=30)
    assert resp.status_code == 200, resp.text
    assert up.requests[-1]["model"] == "qwen3:30b"
    srv = r.status()["servers"][0]
    assert srv["base_url"] == "http://127.0.0.1:8918/v1", srv["base_url"]
    assert srv["model"] == "qwen3:30b", srv["model"]
    r.stop(); up.stop()
    print("ollama base_url verbatim ok")


def test_endpoint_extra_body_reaches_the_wire():
    """Ollama knobs (num_ctx, keep_alive) are configured per server and must
    ride in the request body the router sends."""
    up = FakeUpstream("ollama", 8919, known_models={"qwen3:30b"})
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="ollama-a", provider="ollama_chat",
                        base_url="http://127.0.0.1:8919/v1", model="qwen3:30b",
                        extra_body={"num_ctx": 32768, "keep_alive": "10m"})
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "openai/harness-routed",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 64}, timeout=30)
    assert resp.status_code == 200, resp.text
    body = up.requests[-1]
    assert body.get("num_ctx") == 32768, body
    assert body.get("keep_alive") == "10m", body
    # The caller's own fields still win over endpoint defaults.
    assert body["max_tokens"] == 64, body
    r.stop(); up.stop()
    print("endpoint extra_body ok")


def test_router_deadline_follows_the_widened_timeout():
    """Adaptive widening lives in the store; the router must not cap upstream
    below the deadline the agent's own client is already waiting for."""
    up = FakeUpstream("a", 8926)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="a", base_url="http://127.0.0.1:8926/v1", model="m")
    cfg.llm.request_timeout = 300
    store = Store(cfg.data_dir)
    r = make_router(cfg, store); r.start()
    assert r.upstream_timeout == 300, r.upstream_timeout
    store.meta["llm_timeout"] = 1200
    assert r.upstream_timeout == 1200, r.upstream_timeout
    cfg.llm.request_timeout = 1800
    assert r.upstream_timeout == 1800, r.upstream_timeout
    r.stop(); up.stop()
    print("router deadline follows widening ok")


def test_ollama_endpoint_does_not_receive_reasoning_effort():
    """The SDK puts reasoning_effort in every request; Ollama maps it onto its
    own `think` option and 400s for models it does not mark reasoning-capable."""
    up = FakeUpstream("ollama", 8923, known_models={"qwen3:30b"})
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="ollama-a", provider="ollama_chat",
                        base_url="http://127.0.0.1:8923/v1", model="qwen3:30b")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "openai/harness-routed",
                            "reasoning_effort": "high",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=30)
    assert resp.status_code == 200, resp.text
    body = up.requests[-1]
    assert "reasoning_effort" not in body, body
    r.stop(); up.stop()
    print("reasoning_effort stripped for ollama ok")


def test_openai_endpoint_keeps_reasoning_effort():
    """The strip is endpoint-specific: a gateway that implements the option
    must still receive it."""
    up = FakeUpstream("gw", 8924)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="gw", provider="openai",
                        base_url="http://127.0.0.1:8924/v1", model="openai/gw")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "x", "reasoning_effort": "high",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=30)
    assert resp.status_code == 200, resp.text
    assert up.requests[-1].get("reasoning_effort") == "high", up.requests[-1]
    r.stop(); up.stop()
    print("reasoning_effort kept for openai ok")


def test_explicit_reasoning_effort_survives_on_an_ollama_endpoint():
    """If the operator asks for a reasoning effort on that endpoint, it is a
    deliberate choice and the router sends it."""
    up = FakeUpstream("ollama", 8925, known_models={"qwen3:30b"})
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="ollama-a", provider="ollama_chat",
                        base_url="http://127.0.0.1:8925/v1", model="qwen3:30b",
                        reasoning_effort="low")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "x", "reasoning_effort": "high",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=30)
    assert resp.status_code == 200, resp.text
    # The endpoint's own value wins over whatever the SDK injected.
    assert up.requests[-1].get("reasoning_effort") == "low", up.requests[-1]
    r.stop(); up.stop()
    print("explicit reasoning_effort honoured ok")


def test_mixed_fleet_each_endpoint_gets_its_own_model_key():
    """One placeholder from the agent; each endpoint receives the name it knows."""
    fast = FakeUpstream("openai-gw", 8920)
    oll = FakeUpstream("ollama", 8921, known_models={"qwen3:30b"})
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="openai-gw", base_url="http://127.0.0.1:8920/v1",
                        model="openai/gw-model")
    cfg.llm_servers = [LLMServerConfig(name="ollama-a", provider="ollama_chat",
                                       base_url="http://127.0.0.1:8921/v1",
                                       model="qwen3:30b")]
    cfg.llm_server_concurrency = 1
    # Round-robin, not the throughput default: this test wants every endpoint
    # to actually be used so each one's model key can be checked on the wire.
    cfg.router_selection = "round_robin"
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    seen = {}
    for i in range(6):
        resp = httpx.post(r.base_url + "/chat/completions",
                          json={"model": "openai/harness-routed",
                                "messages": [{"role": "user", "content": f"p{i}"}]},
                          timeout=30)
        assert resp.status_code == 200, resp.text
        seen[resp.headers["x-harness-router"]] = True
    assert set(seen) == {"openai-gw", "ollama-a"}, seen
    assert all(b["model"] == "openai/gw-model" for b in fast.requests), fast.requests
    assert all(b["model"] == "qwen3:30b" for b in oll.requests), oll.requests
    r.stop(); fast.stop(); oll.stop()
    print("mixed fleet model keys ok")


def test_test_prompt_sends_the_literal_key_to_an_ollama_row():
    """The settings Test button must exercise the endpoint the way real traffic
    will hit it, including an unsaved row."""
    up = FakeUpstream("ollama", 8922, known_models={"qwen3:30b"})
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="primary", base_url="http://127.0.0.1:8922/v1",
                        model="openai/prim")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    res = r.test_prompt_config(LLMServerConfig(
        name="not-saved-yet", provider="ollama_chat",
        base_url="http://127.0.0.1:8922/v1", model="qwen3:30b", api_key=""),
        "hi", max_tokens=16)
    assert res["ok"], res
    assert res["model"] == "qwen3:30b", res["model"]
    assert up.requests[-1]["model"] == "qwen3:30b", up.requests[-1]
    r.stop(); up.stop()
    print("test prompt on ollama row ok")


def test_router_rewrites_model_and_attaches_headers():
    """Agents send a placeholder model; the router puts the real one on the wire."""
    a = FakeUpstream("a", 8901)
    _, r = _router()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "whatever-the-agent-said",
                            "messages": [{"role": "user", "content": "hi"}]},
                      headers={"x-harness-agent": "ag-1", "x-harness-role": "worker"},
                      timeout=30)
    assert resp.status_code == 200, resp.text
    assert a.requests[-1]["model"] == "openai/prim", a.requests[-1]["model"]
    assert a.headers[-1]["Authorization"] == "Bearer local-no-key"
    # Attribution survives to the router's own view of the request.
    card = [x for x in r.status()["recent"] if x["agent_id"] == "ag-1"]
    assert card and card[0]["agent_role"] == "worker"
    assert resp.headers["x-harness-router"] == "primary"
    r.stop(); a.stop()
    print("model rewrite + header attach ok")


def test_router_streams_sse_through():
    a = FakeUpstream("a", 8902)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="primary", base_url="http://127.0.0.1:8902/v1",
                        model="openai/prim")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    with httpx.stream("POST", r.base_url + "/chat/completions",
                      json={"model": "x", "stream": True,
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=30) as resp:
        assert resp.status_code == 200
        lines = [l for l in resp.iter_lines() if l.startswith("data:")]
    assert len(lines) == 3, lines          # delta + finish + [DONE]
    assert a.requests[-1]["model"] == "openai/prim"
    r.stop(); a.stop()
    print("streaming relay ok")


def test_next_request_goes_to_whichever_endpoint_frees_up_first():
    """No pre-assignment: a fast endpoint takes more of the load than a slow one."""
    fast = FakeUpstream("fast", 8903, latency=0.02, tokens=5)
    slow = FakeUpstream("slow", 8904, latency=0.40, tokens=5)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="fast", base_url="http://127.0.0.1:8903/v1",
                        model="m-fast")
    cfg.llm_servers = [LLMServerConfig(name="slow", provider="openai",
                                       base_url="http://127.0.0.1:8904/v1",
                                       model="m-slow")]
    cfg.llm_server_concurrency = 1
    r = make_router(cfg, Store(cfg.data_dir)); r.start()

    served = []
    def call(i):
        resp = httpx.post(r.base_url + "/chat/completions",
                          json={"model": "x",
                                "messages": [{"role": "user", "content": f"p{i}"}]},
                          timeout=60)
        served.append(resp.headers.get("x-harness-router"))
    threads = [threading.Thread(target=call, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(fast.requests) > len(slow.requests), (
        f"fast={len(fast.requests)} slow={len(slow.requests)}: the free-first "
        "rule should send more work to the endpoint that frees up first")
    r.stop(); fast.stop(); slow.stop()
    print(f"availability-based dispatch ok (fast={len(fast.requests)}, "
          f"slow={len(slow.requests)})")


def test_all_busy_requests_show_as_queued_with_no_endpoint():
    """The dashboard requirement: queued means waiting, not pre-assigned."""
    hold = FakeUpstream("hold", 8905, latency=1.2)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="hold", base_url="http://127.0.0.1:8905/v1",
                        model="m")
    cfg.llm_server_concurrency = 1
    cfg.router_queue_timeout = 30
    r = make_router(cfg, Store(cfg.data_dir)); r.start()

    out: dict[int, int] = {}
    def call(i):
        resp = httpx.post(r.base_url + "/chat/completions",
                          json={"model": "x",
                                "messages": [{"role": "user", "content": f"q{i}"}]},
                          headers={"x-harness-agent": f"a{i}"}, timeout=60)
        out[i] = resp.status_code
    threads = [threading.Thread(target=call, args=(i,)) for i in range(3)]
    [t.start() for t in threads]
    time.sleep(0.5)
    snap = r.status()
    queued = [x for x in snap["requests"] if x["status"] == "queued"]
    sent = [x for x in snap["requests"] if x["status"] != "queued"]
    assert len(queued) == 2, snap["requests"]
    assert all(x["server"] is None for x in queued), "queued must not be bound"
    assert len(sent) == 1 and sent[0]["server"] == "hold"
    [t.join() for t in threads]
    assert set(out.values()) == {200}, out
    r.stop(); hold.stop()
    print("queued lane shows unbound requests ok")


def test_upstream_error_status_is_relayed_and_the_endpoint_parked():
    """An HTTP error is passed through verbatim, not silently retried.

    The caller's client has to see the real status: litellm only does its own
    rate-limit/backoff handling when a 429 actually arrives. The endpoint is
    parked, so the *next* request goes somewhere healthy.
    """
    dead = FakeUpstream("dead", 8906, fail=(500, "boom"))
    good = FakeUpstream("good", 8907)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="dead", base_url="http://127.0.0.1:8906/v1",
                        model="m-dead")
    cfg.llm_servers = [LLMServerConfig(name="good", provider="openai",
                                       base_url="http://127.0.0.1:8907/v1",
                                       model="m-good")]
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "x",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=60)
    assert resp.status_code == 500, resp.text
    assert "boom" in resp.text
    status = {s["name"]: s for s in r.status()["servers"]}
    assert status["dead"]["cooling_down"] is True
    assert status["dead"]["reachable"] is False
    # Next request: the parked endpoint is skipped entirely.
    resp2 = httpx.post(r.base_url + "/chat/completions",
                       json={"model": "x",
                             "messages": [{"role": "user", "content": "hi"}]},
                       timeout=60)
    assert resp2.status_code == 200, resp2.text
    assert resp2.headers["x-harness-router"] == "good", resp2.headers
    assert len(good.requests) == 1
    r.stop(); dead.stop(); good.stop()
    print("upstream error relayed, endpoint parked ok")


def test_unreachable_endpoint_fails_over_to_a_healthy_one():
    """A connection fault is the router's to absorb: the agent should not see it."""
    good = FakeUpstream("good", 8915)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    # port 8916 has nothing listening on it
    cfg.llm = LLMConfig(name="down", base_url="http://127.0.0.1:8916/v1",
                        model="m-down")
    cfg.llm_servers = [LLMServerConfig(name="good", provider="openai",
                                       base_url="http://127.0.0.1:8915/v1",
                                       model="m-good")]
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "x",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=60)
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-harness-router"] == "good", resp.headers
    card = [x for x in r.status()["recent"] if x["status"] == "completed"]
    assert card and card[-1]["failed_over"] is True, card
    r.stop(); good.stop()
    print("connection fault fails over ok")


def test_rate_limit_is_not_treated_as_endpoint_down():
    """A 429 means slow down; parking the endpoint would cascade the outage."""
    limited = FakeUpstream("limited", 8908, fail=(429, "slow down"))
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="limited", base_url="http://127.0.0.1:8908/v1",
                        model="m")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    resp = httpx.post(r.base_url + "/chat/completions",
                      json={"model": "x",
                            "messages": [{"role": "user", "content": "hi"}]},
                      timeout=30)
    assert resp.status_code == 429, resp.text
    assert r.status()["servers"][0]["cooling_down"] is False
    r.stop(); limited.stop()
    print("429 does not park the endpoint ok")


def test_throughput_and_token_totals_are_recorded():
    # 0.3s of real latency for 40 tokens: a rate far off ~133 tok/s means the
    # duration was measured across two different clocks.
    a = FakeUpstream("a", 8909, tokens=40, latency=0.3)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="a", base_url="http://127.0.0.1:8909/v1", model="m")
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    httpx.post(r.base_url + "/chat/completions",
               json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
               timeout=30)
    srv = r.status()["servers"][0]
    assert srv["total_tokens"] == 40, srv
    assert srv["prompt_tokens"] == 11, srv
    assert 40 < srv["tokens_per_second"] < 400, srv["tokens_per_second"]
    r.stop(); a.stop()
    print("throughput accounting ok")


def test_test_prompt_works_on_an_unsaved_endpoint():
    """The 'Test prompt' button must prove a new LLM before it is saved."""
    fresh = FakeUpstream("fresh", 8910, tokens=6)
    _, r = _router()
    result = r.test_prompt_config(
        LLMServerConfig(name="not-saved-yet", provider="openai",
                        base_url="http://127.0.0.1:8910/v1", model="m-fresh"),
        "Reply with exactly the word READY", 32)
    assert result["ok"] is True, result
    # The wire model is the literal key that was typed, never a provider-prefixed
    # litellm route: the endpoint is addressed over the OpenAI protocol, so it
    # must receive the name it knows.
    assert result["model"] == "m-fresh", result["model"]
    assert fresh.requests[-1]["model"] == "m-fresh", fresh.requests[-1]
    assert result["completion_tokens"] == 6
    assert result["reply"] == "fresh-reply"
    # An unsaved definition gets only the key typed for it, never the
    # harness-wide key from the environment.
    import os
    sentinel = "env-key-must-not-leak"
    os.environ["HARNESS_LLM_API_KEY"] = sentinel
    try:
        leaked = r.test_prompt_config(
            LLMServerConfig(name="also-unsaved", provider="openai",
                            base_url="http://127.0.0.1:8910/v1", model="m2",
                            api_key="typed-key"),
            "hi", 8)
        assert leaked["ok"] is True, leaked
        sent = fresh.headers[-1].get("Authorization")
        assert sent == "Bearer typed-key", sent
        assert sentinel not in str(fresh.headers[-1]), "env key leaked to a new URL"
    finally:
        os.environ.pop("HARNESS_LLM_API_KEY", None)
    r.stop(); fresh.stop()
    print("test prompt on an unsaved endpoint ok")


def test_config_reload_changes_the_router_fleet():
    a = FakeUpstream("a", 8911)
    b = FakeUpstream("b", 8912)
    cfg, r = _router()
    assert [s["name"] for s in r.status()["servers"]] == ["primary"]
    cfg.llm_servers = [LLMServerConfig(name="second", provider="openai",
                                       base_url="http://127.0.0.1:8912/v1",
                                       model="m-two")]
    added, removed = r.sync_servers()
    assert added == ["second"] and removed == []
    assert [s["name"] for s in r.status()["servers"]] == ["primary", "second"]
    names = set()
    for _ in range(4):
        httpx.post(r.base_url + "/chat/completions",
                   json={"model": "x",
                         "messages": [{"role": "user", "content": "hi"}]},
                   timeout=30)
        names.update(s["name"] for s in r.status()["servers"])
    assert names == {"primary", "second"}
    assert len(b.requests) > 0, "the newly added endpoint must be used"
    r.stop(); a.stop(); b.stop()
    print("live fleet change ok")


def test_per_server_headers_reach_the_wire():
    a = FakeUpstream("a", 8913)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="primary", base_url="http://127.0.0.1:8913/v1",
                        model="openai/prim",
                        headers={"X-Tenant": "acme"})
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    httpx.post(r.base_url + "/chat/completions",
               json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
               timeout=30)
    assert a.headers[-1].get("X-Tenant") == "acme", a.headers[-1]
    r.stop(); a.stop()
    print("per-server headers ok")


def test_capacity_report_admits_when_no_slot_is_free():
    hold = FakeUpstream("hold", 8914, latency=1.0)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="hold", base_url="http://127.0.0.1:8914/v1",
                        model="m")
    cfg.llm_server_concurrency = 1
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    t = threading.Thread(target=lambda: httpx.post(
        r.base_url + "/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        timeout=30), daemon=True)
    t.start()
    time.sleep(0.4)
    cap = r.capacity()
    assert cap["server_slots"] == 1
    assert cap["server_slots_free"] == 0, cap
    assert cap["queued_requests"] == 0
    t.join(10)
    r.stop(); hold.stop()
    print("capacity report ok")


def test_fast_endpoint_wins_even_when_the_slow_one_is_idle():
    """Throughput preference: load alone picks the wrong endpoint in a mixed fleet.

    A 500-token reply is ~8s at 60 tok/s and ~100s at 5 tok/s, so sending work
    to the idle slow endpoint because it has fewer requests in flight is worse
    than queueing it behind one request on the fast endpoint.
    """
    fast = FakeUpstream("fast", 8930, latency=0.02, tokens=5)
    slow = FakeUpstream("slow", 8931, latency=0.40, tokens=5)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="fast", base_url="http://127.0.0.1:8930/v1",
                        model="m-fast")
    cfg.llm_servers = [LLMServerConfig(name="slow", provider="openai",
                                       base_url="http://127.0.0.1:8931/v1",
                                       model="m-slow")]
    cfg.llm_server_concurrency = 2
    r = make_router(cfg, Store(cfg.data_dir)); r.start()

    def call(i):
        httpx.post(r.base_url + "/chat/completions",
                   json={"model": "x",
                         "messages": [{"role": "user", "content": f"p{i}"}]},
                   timeout=60)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    fast_rate = r._servers["fast"].tokens_per_second
    slow_rate = r._servers["slow"].tokens_per_second
    assert fast_rate and slow_rate and fast_rate > slow_rate * 3, (
        fast_rate, slow_rate)
    assert len(fast.requests) > len(slow.requests), (
        f"fast={len(fast.requests)} slow={len(slow.requests)}: the faster "
        "endpoint should take the larger share")
    r.stop(); fast.stop(); slow.stop()
    print(f"throughput preference ok (fast={len(fast.requests)}, "
          f"slow={len(slow.requests)})")


def test_unmeasured_endpoint_is_tried_once_then_ranked_on_its_own_rate():
    """A server added at runtime must not be starved for never having history."""
    known = FakeUpstream("known", 8932, latency=0.02, tokens=5)
    fresh = FakeUpstream("fresh", 8933, latency=0.02, tokens=5)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="known", base_url="http://127.0.0.1:8932/v1",
                        model="m-known")
    cfg.llm_servers = [LLMServerConfig(name="fresh", provider="openai",
                                       base_url="http://127.0.0.1:8933/v1",
                                       model="m-fresh")]
    cfg.llm_server_concurrency = 1
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    seen = {}
    for i in range(6):
        resp = httpx.post(r.base_url + "/chat/completions",
                          json={"model": "x",
                                "messages": [{"role": "user", "content": f"p{i}"}]},
                          timeout=30)
        assert resp.status_code == 200, resp.text
        seen[resp.headers["x-harness-router"]] = True
    assert seen == {"known": True, "fresh": True}, (
        f"{seen}: an endpoint with no samples must still get used, or it can "
        "never produce the rate needed to be ranked")
    r.stop(); known.stop(); fresh.stop()
    print("unmeasured endpoint sampled ok")


def test_round_robin_selection_spreads_regardless_of_speed():
    """The other policy: share the load evenly when endpoints are interchangeable."""
    fast = FakeUpstream("fast", 8934, latency=0.02, tokens=5)
    slow = FakeUpstream("slow", 8935, latency=0.02, tokens=5)
    cfg = HarnessConfig(); cfg.data_dir = tempfile.mkdtemp()
    cfg.llm = LLMConfig(name="fast", base_url="http://127.0.0.1:8934/v1",
                        model="m-fast")
    cfg.llm_servers = [LLMServerConfig(name="slow", provider="openai",
                                       base_url="http://127.0.0.1:8935/v1",
                                       model="m-slow")]
    cfg.llm_server_concurrency = 1
    cfg.router_selection = "round_robin"
    r = make_router(cfg, Store(cfg.data_dir)); r.start()
    for i in range(6):
        resp = httpx.post(r.base_url + "/chat/completions",
                          json={"model": "x",
                                "messages": [{"role": "user", "content": f"p{i}"}]},
                          timeout=30)
        assert resp.status_code == 200, resp.text
    assert len(fast.requests) == 3 and len(slow.requests) == 3, (
        f"fast={len(fast.requests)} slow={len(slow.requests)}: round_robin "
        "should split sequential requests evenly")
    r.stop(); fast.stop(); slow.stop()
    print("round_robin selection ok")


def main() -> None:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"--- {t.__name__}")
        t()
    print(f"\nALL LLM ROUTER TESTS PASSED ({len(tests)})")


if __name__ == "__main__":
    main()
