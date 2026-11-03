"""The LLM router: the single funnel every LLM request in the harness goes through.

Agents only produce a prompt. They are all pointed at this service, which owns
everything else about the request: choosing an endpoint, rewriting the model key
for that endpoint, attaching the credential and headers that endpoint needs,
sending it, and measuring what came back.

Two properties matter and are easy to get wrong:

* **Nothing is pre-assigned.** A request carries no server choice. The dispatcher
  hands the *oldest pending request* to *whichever endpoint frees up first*, so a
  slow endpoint slows only itself instead of stranding a request that was pinned
  to it at dispatch time.
* **It runs continually.** It is a real HTTP service on its own port, not a
  helper called from the agent loop, so requests from the orchestrator, every
  sub-agent, every condenser, and a manual test prompt all take the same path.

The inbound side speaks the OpenAI chat-completions contract, which is what the
OpenHands SDK (litellm + the OpenAI client) already emits. That means the router
receives a well-formed OpenAI body and only has to rewrite `model` and the
headers per upstream — it does not re-serialize prompts.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx

from .config import HarnessConfig, LLMServerConfig
from .store import Store, new_id

# Statuses a request passes through. `awaiting` means the bytes are on the wire
# and we are waiting on the model; `generating` means tokens are actively
# arriving. Both are "already sent", which is what the dashboard lane shows.
STATUSES = ("queued", "awaiting", "generating", "completed", "failed")

# Requests kept visible after they finish, so a card does not vanish the instant
# it lands and the user can see the result of the run they were watching.
RECENT_SECONDS = 12.0
RECENT_LIMIT = 8

H_ATTR_AGENT = "x-harness-agent"
H_ATTR_ROLE = "x-harness-role"
H_ATTR_KIND = "x-harness-kind"
H_ATTR_PROJECT = "x-harness-project"
H_ATTR_TASK = "x-harness-task"
H_ATTR_REQ = "x-harness-request"


@dataclass
class RouterRequest:
    """One chat-completion request travelling through the router."""

    id: str
    seq: int
    body: dict[str, Any]
    agent_id: str | None = None
    agent_role: str | None = None
    kind: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    status: str = "queued"
    server: str | None = None
    attempts: int = 0
    received_at: float = field(default_factory=time.time)
    sent_at: float | None = None
    # Monotonic twin of sent_at. Throughput must be measured on one clock:
    # mixing wall time with monotonic produces a nonsense duration.
    sent_mono: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    # Set when the request was moved off a failing endpoint onto another one.
    failed_over: bool = False
    # True once any byte of the reply has gone back to the caller. A response
    # cannot be replaced after that: writing a second status line into a body
    # that already started is a protocol error for the caller's client.
    responded: bool = False

    @property
    def active(self) -> bool:
        return self.status in ("queued", "awaiting", "generating")

    def prompt_preview(self) -> str:
        """The text the agent actually wants answered, for the card + drill-in.

        The body carries the whole conversation; the last non-system message is
        what the model is being asked to act on right now.
        """
        messages = self.body.get("messages") or []
        for msg in reversed(messages):
            if msg.get("role") in ("system", "assistant"):
                continue
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined
        return messages[-1].get("content", "") if messages else ""


@dataclass
class ServerState:
    """Live view of one configured endpoint: health, load, and throughput."""

    name: str
    provider: str
    model: str
    base_url: str
    # Endpoints in flight on this server right now.
    in_flight: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    consecutive_failures: int = 0
    # Monotonic deadline before which this endpoint is not offered.
    cooldown_until: float = 0.0
    reachable: bool | None = None      # None = not probed yet
    last_error: str | None = None
    last_checked_at: float | None = None
    last_ok_at: float | None = None
    # Throughput: summed so the average is a true mean, not a mean of means.
    output_tokens: int = 0
    prompt_tokens: int = 0
    generation_seconds: float = 0.0
    samples: int = 0
    last_rate: float | None = None
    # Monotonic timestamp of the last request assigned here. Used to spread
    # sequential requests across equally-free endpoints.
    last_assigned: float = 0.0

    def cooling(self, now: float) -> bool:
        return self.cooldown_until > now

    @property
    def tokens_per_second(self) -> float | None:
        if self.generation_seconds <= 0 or self.output_tokens <= 0:
            return None
        return self.output_tokens / self.generation_seconds

    def snapshot(self, now: float) -> dict[str, Any]:
        """`now` must be monotonic: cooldown_until is stored on that clock."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "in_flight": self.in_flight,
            "requests_ok": self.requests_ok,
            "requests_failed": self.requests_failed,
            "consecutive_failures": self.consecutive_failures,
            "cooling_down": self.cooling(now),
            "cooldown_seconds": max(0.0, round(self.cooldown_until - now, 1)),
            "reachable": self.reachable,
            "last_error": self.last_error,
            "last_checked_at": self.last_checked_at,
            "last_ok_at": self.last_ok_at,
            "total_tokens": self.output_tokens,
            "prompt_tokens": self.prompt_tokens,
            "tokens_per_second": (
                round(self.tokens_per_second, 1)
                if self.tokens_per_second is not None else None),
            "last_tokens_per_second": (
                round(self.last_rate, 1) if self.last_rate is not None else None),
            "generation_seconds": round(self.generation_seconds, 1),
            "samples": self.samples,
        }


class LLMRouter:
    """Queues every LLM request and dispatches it to the next free endpoint."""

    def __init__(self, config: HarnessConfig, store: Store,
                 host: str = "127.0.0.1", port: int = 0):
        self.config = config
        self.store = store
        self._lock = threading.RLock()
        # Freed when a request arrives or a server frees up: the dispatcher only
        # ever assigns a request when both are true at the same instant.
        self._cond = threading.Condition(self._lock)

        self._pending: queue.Queue[RouterRequest] = queue.Queue()
        self._requests: dict[str, RouterRequest] = {}
        self._recent: list[RouterRequest] = []
        self._seq = 0

        self._servers: dict[str, ServerState] = {}
        # Held because _sync_servers notifies the dispatcher, and notify
        # requires the condition's lock to be owned.
        with self._cond:
            self._sync_servers()

        self._http = httpx.Client(
            timeout=httpx.Timeout(self.upstream_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )

        self._host = host
        self._requested_port = port
        self._server = ThreadingHTTPServer((host, port), _Handler)
        # The handler resolves the router through the server object.
        self._server.router = self  # type: ignore[attr-defined]
        self._server.daemon_threads = True
        self._port = self._server.server_address[1]
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

        # Counts that outlive individual request cards.
        self.served = 0
        self.retries = 0
        self.started_at = time.time()

    @property
    def upstream_timeout(self) -> int:
        """Per-request deadline for one upstream call.

        Read from the live config each time, so changing the timeout in Settings
        applies to the next request rather than needing a restart.

        The runtime's adaptive widening is kept in the store, and the router has
        to follow it: if the deadline here stayed at the configured value while
        the agent's client waited longer, the router would answer the agent with
        a timeout it invented itself and the endpoint would never get the chance
        a slow model actually needs.
        """
        configured = int(getattr(self.config.llm, "request_timeout", 300) or 300)
        widened = int(self.store.meta.get("llm_timeout") or 0)
        return max(1, configured, widened)

    # ---------------- lifecycle ----------------

    def start(self) -> None:
        """Serve the proxy and run the dispatcher forever."""
        serve = threading.Thread(
            target=self._server.serve_forever,
            name="llm-router-http", daemon=True)
        dispatch = threading.Thread(
            target=self._dispatch_loop, name="llm-router-dispatch", daemon=True)
        probe = threading.Thread(
            target=self._probe_loop, name="llm-router-probe", daemon=True)
        self._threads = [serve, dispatch, probe]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._cond.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._http.close()

    @property
    def port(self) -> int:
        return self._port

    @property
    def base_url(self) -> str:
        """What an agent's LLM should use as `base_url`."""
        return f"http://{self._host}:{self._port}/v1"

    # ---------------- configuration ----------------

    def sync_servers(self) -> tuple[list[str], list[str]]:
        """Reconcile live server state with the config. Returns (added, removed)."""
        with self._lock:
            return self._sync_servers()

    def _sync_servers(self) -> tuple[list[str], list[str]]:
        configured = {s.name: s for s in self.config.servers()}
        added = [n for n in configured if n not in self._servers]
        removed = [n for n in self._servers if n not in configured]
        for name in removed:
            # A server whose endpoint changed is removed and re-added under the
            # same name, so carry throughput across but drop stale health.
            old = self._servers.pop(name)
            if name in configured:
                fresh = self._state_for(configured[name])
                fresh.output_tokens = old.output_tokens
                fresh.prompt_tokens = old.prompt_tokens
                fresh.generation_seconds = old.generation_seconds
                fresh.samples = old.samples
                self._servers[name] = fresh
        for name in added:
            self._servers[name] = self._state_for(configured[name])
        # Keep the config's order so the UI and tie-breaks are stable.
        self._servers = {n: self._servers[n] for n in configured}
        if added or removed:
            self._cond.notify_all()
        return (added, removed)

    def _state_for(self, srv: LLMServerConfig) -> ServerState:
        return ServerState(name=srv.name, provider=srv.provider,
                           model=srv.model, base_url=srv.base_url)

    def server_config(self, name: str) -> LLMServerConfig | None:
        return self.config.server_named(name)

    # ---------------- intake ----------------

    def submit(self, body: dict[str, Any], meta: dict[str, Any | None],
               client: Any = None) -> RouterRequest:
        """Accept a request from an agent and queue it. Never picks a server.

        `client` is the handler connection that must receive the reply. It is
        attached before the request becomes visible to the dispatcher: if the
        dispatcher could win the race, it would find nothing to write to and the
        agent would see a connection closed with no response at all.
        """
        with self._lock:
            self._seq += 1
            req = RouterRequest(
                id=new_id("req"), seq=self._seq, body=body,
                agent_id=meta.get("agent_id"), agent_role=meta.get("agent_role"),
                kind=meta.get("kind"), project_id=meta.get("project_id"),
                task_id=meta.get("task_id"),
            )
            req._client = client  # type: ignore[attr-defined]
            self._requests[req.id] = req
            # Appended before pruning: _prune_recent drops anything from
            # _requests that is not in _recent, so pruning first would delete
            # the request that was just accepted.
            self._recent.append(req)
            self._prune_recent()
        self._pending.put(req)
        with self._lock:
            self._cond.notify_all()
        return req

    def _prune_recent(self) -> None:
        """Drop finished requests once they have been visible long enough.

        `_requests` is keyed forever by id, so without this it grows for the life
        of the process. A request stays addressable while it is active or still
        shown as a recent card.
        """
        now = time.time()
        keep = [r for r in self._recent
                if r.active or (r.finished_at and now - r.finished_at < RECENT_SECONDS)]
        if len(keep) > RECENT_LIMIT:
            keep = keep[-RECENT_LIMIT:]
        self._recent = keep
        for rid in [r.id for r in list(self._requests.values()) if r not in keep]:
            del self._requests[rid]

    # ---------------- dispatch ----------------

    def _dispatch_loop(self) -> None:
        while not self._stop.is_set():
            self._expire_queued()
            req = self._await_assignment()
            if req is None:
                continue
            threading.Thread(
                target=self._deliver, args=(req,),
                name=f"llm-send-{req.seq}", daemon=True).start()

    def _expire_queued(self) -> None:
        """Fail requests that have waited past the router's queue timeout.

        Every endpoint being busy is a real condition the user should see as an
        error, not one that leaves an agent's request hanging indefinitely.
        """
        limit = max(1, self.config.router_queue_timeout)
        cutoff = time.time() - limit
        expired = []
        with self._lock:
            for r in list(self._requests.values()):
                if r.status == "queued" and r.received_at < cutoff:
                    r.status = "failed"
                    r.error = (f"no LLM endpoint became available within {limit}s")
                    r.finished_at = time.time()
                    expired.append(r)
        for r in expired:
            self._respond(r, 503, json.dumps({"error": {
                "message": f"LLM router: no endpoint free within {limit}s",
                "type": "router_queue_timeout",
            }}).encode(), "application/json")

    def _await_assignment(self) -> RouterRequest | None:
        """Block until a pending request and a free endpoint exist together.

        The pairing happens under the lock, so a request is never bound to an
        endpoint in advance: the endpoint that frees up first is the one that
        gets the oldest waiting request.
        """
        deadline = time.monotonic() + 1.0
        with self._cond:
            while not self._stop.is_set():
                srv = self._free_server_locked(time.monotonic())
                if srv is not None and not self._pending.empty():
                    try:
                        req = self._pending.get_nowait()
                    except queue.Empty:
                        continue
                    if req.status != "queued":
                        # Expired by _expire_queued between enqueue and here; it
                        # has already been answered with a 503.
                        continue
                    srv.in_flight += 1
                    srv.last_assigned = time.monotonic()
                    req.server = srv.name
                    req.status = "awaiting"
                    req.sent_at = time.time()
                    req.sent_mono = time.monotonic()
                    req.attempts += 1
                    return req
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
                if not self._pending.empty():
                    deadline = time.monotonic() + 1.0
        return None

    def _wait_estimate(self, srv: ServerState) -> float:
        """Roughly how long a request waits for service on this endpoint.

        Throughput mode ranks by this instead of by raw load: a mixed fleet is
        not symmetric, and a fast endpoint with one request already running
        still beats a slow idle one (a 500-token reply is 8s at 60 tok/s and
        100s at 5 tok/s, so "least loaded" would pick the wrong endpoint every
        time). With equal rates the estimate reduces to least-loaded, which is
        the behaviour the fleet needs when the endpoints are interchangeable.

        An endpoint that has never been served gets a negative estimate so it
        is tried once immediately: guessing its rate (however generously) makes
        its estimate tie with the measured endpoints, so it is never picked,
        never measured, and stays unusable forever. The negative value is
        self-limiting - it only applies until that first sample lands.
        """
        rate = srv.tokens_per_second
        if rate is None:
            return -1.0
        return (srv.in_flight + 1) / rate

    def _free_server_locked(self, now: float) -> ServerState | None:
        """The best healthy endpoint with a spare slot, or None.

        Endpoints in cooldown are skipped outright. Throughput mode then picks
        the endpoint that should serve the request soonest; round-robin mode
        ranks by load alone and lets the tie-break spread work evenly. The
        endpoint waiting longest for work breaks any remaining tie, so
        sequential requests still spread instead of piling onto whichever sorts
        first.

        A failed /models probe does NOT take an endpoint out of rotation. The
        probe is advisory - it is what the UI shows and it is cheap - but an
        endpoint that answers GET /models badly can still serve chat, and one
        that is genuinely down is parked by the cooldown its own failed request
        sets. Deprioritising on the probe instead would mean a flaky probe
        silently removes capacity and the failover path never runs.
        """
        limit = max(1, self.config.llm_server_concurrency)
        candidates = [s for s in self._servers.values()
                      if not s.cooling(now) and s.in_flight < limit]
        if not candidates:
            return None
        throughput = getattr(self.config, "router_selection", "throughput") \
            == "throughput"
        def rank(s: ServerState) -> tuple:
            if not throughput:
                return (s.in_flight, s.consecutive_failures)
            return (round(self._wait_estimate(s), 3), s.consecutive_failures)

        candidates.sort(key=rank)
        best = candidates[0]
        tied = [s for s in candidates if rank(s) == rank(best)]
        return min(tied, key=lambda s: s.last_assigned)

    def _release(self, server_name: str | None) -> None:
        if not server_name:
            return
        with self._lock:
            srv = self._servers.get(server_name)
            if srv is not None and srv.in_flight > 0:
                srv.in_flight -= 1
            self._cond.notify_all()

    def _deliver(self, req: RouterRequest) -> None:
        """Forward one request, failing over to another endpoint if needed."""
        servers = max(1, len(self._servers))
        while True:
            name = req.server
            srv_cfg = self.server_config(name) if name else None
            if srv_cfg is None:
                self._fail(req, "no LLM server is configured")
                self._release(name)
                return
            outcome = self._forward(req, srv_cfg)
            if outcome == "sent":
                self._release(name)
                return
            # Only retry while nothing has reached the client yet: once bytes of
            # a stream are on the wire, restarting would corrupt the response.
            if req.attempts >= servers or req.responded:
                self._fail(req, outcome)
                self._release(name)
                return
            self.retries += 1
            req.failed_over = True
            req.status = "queued"
            req.server = None
            req.sent_at = None
            req.sent_mono = None
            self._release(name)
            # Re-queue at the back: the request keeps its place in the funnel but
            # waits for the next free endpoint rather than hammering the one
            # that just failed.
            with self._lock:
                self._cond.notify_all()
            self._pending.put(req)
            return

    def _forward(self, req: RouterRequest, srv: LLMServerConfig) -> str:
        """Send to one endpoint. Returns 'sent' or an error string."""
        name = srv.name
        url = srv.chat_completions_url()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **srv.resolved_headers(),
        }
        api_key = srv.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        # The model key is per endpoint and must be the literal name that
        # endpoint knows: Ollama wants `qwen3:30b`, the primary wants
        # `openai/qwen`. A provider prefix here makes Ollama 404. Endpoint
        # extras (Ollama's num_ctx / keep_alive) ride in the same body.
        body = {**srv.extra_body, **req.body}
        body["model"] = srv.wire_model()
        # The SDK injects its own reasoning_effort for every request; an
        # endpoint-specific value overrides it, and an endpoint that does not
        # implement the option gets the key removed entirely.
        if srv.reasoning_effort:
            body["reasoning_effort"] = srv.reasoning_effort
        for key in srv.body_keys_to_drop():
            body.pop(key, None)
        timeout = self.upstream_timeout

        try:
            with self._http.stream(
                "POST", url, json=body, headers=headers,
                timeout=httpx.Timeout(timeout, connect=10.0),
            ) as upstream:
                if upstream.status_code >= 400:
                    detail = upstream.read().decode(errors="replace")[:400]
                    self._mark_failure(name, upstream.status_code, detail)
                    # Relay the upstream status verbatim so the caller's client
                    # raises its own correct error type (429 -> rate limit, etc).
                    self._respond(req, upstream.status_code, detail.encode(),
                                  "application/json")
                    return "sent"
                ctype = upstream.headers.get("content-type", "application/json")
                if "text/event-stream" in ctype:
                    self._relay_stream(req, name, upstream)
                else:
                    payload = upstream.read()
                    self._record_json(req, name, payload)
                    self._respond(req, upstream.status_code, payload, ctype)
            return "sent"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._mark_failure(name, None, detail)
            return detail

    def _relay_stream(self, req: RouterRequest, name: str,
                      upstream: httpx.Response) -> None:
        """Pass SSE through while measuring tokens and timing."""
        self._start_chunked(req)
        started = time.monotonic()
        usage: dict[str, Any] = {}
        chunks = 0
        for line in upstream.iter_lines():
            if not line:
                continue
            self._end_chunk(req, line + "\n\n")
            if req.first_token_at is None:
                req.first_token_at = time.monotonic()
                req.status = "generating"
                with self._lock:
                    srv = self._servers.get(name)
                    if srv is not None:
                        srv.reachable = True
                        srv.last_ok_at = time.time()
                        srv.last_error = None
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    chunks += 1
                    try:
                        parsed = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if parsed.get("usage"):
                        usage = parsed["usage"]
        self._finish_chunked(req)
        elapsed = time.monotonic() - started
        # Endpoints that omit a usage chunk are counted by SSE data frames,
        # which is a close proxy for chunks and keeps the rate meaningful.
        completion = int(usage.get("completion_tokens") or chunks)
        self._record_tokens(name, int(usage.get("prompt_tokens") or 0),
                            completion, elapsed)
        req.completion_tokens = completion
        req.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        req.status = "completed"
        req.finished_at = time.time()
        self.served += 1
        with self._lock:
            srv = self._servers.get(name)
            if srv is not None:
                srv.requests_ok += 1
                srv.consecutive_failures = 0

    def _record_json(self, req: RouterRequest, name: str, payload: bytes) -> None:
        started = req.sent_mono or time.monotonic()
        elapsed = time.monotonic() - started
        usage: dict[str, Any] = {}
        try:
            usage = json.loads(payload).get("usage") or {}
        except (json.JSONDecodeError, AttributeError):
            pass
        completion = int(usage.get("completion_tokens") or 0)
        self._record_tokens(name, int(usage.get("prompt_tokens") or 0),
                            completion, elapsed)
        req.completion_tokens = completion
        req.prompt_tokens = int(usage.get("prompt_tokens") or 0)
        req.status = "completed"
        req.finished_at = time.time()
        self.served += 1
        with self._lock:
            srv = self._servers.get(name)
            if srv is not None:
                srv.requests_ok += 1
                srv.consecutive_failures = 0
                srv.reachable = True
                srv.last_ok_at = time.time()
                srv.last_error = None

    def _record_tokens(self, name: str, prompt: int, completion: int,
                       seconds: float) -> None:
        with self._lock:
            srv = self._servers.get(name)
            if srv is None:
                return
            srv.prompt_tokens += prompt
            srv.output_tokens += completion
            srv.generation_seconds += max(0.0, seconds)
            srv.samples += 1
            if seconds > 0 and completion > 0:
                srv.last_rate = completion / seconds

    def _mark_failure(self, name: str, status: int | None, detail: str) -> None:
        """Park an endpoint that refused or could not be reached.

        A 429 is deliberately *not* parked: it means slow down, not down, and
        cooling the endpoint would push every request onto the rest of the fleet.
        """
        now = time.monotonic()
        with self._lock:
            srv = self._servers.get(name)
            if srv is None:
                return
            srv.requests_failed += 1
            srv.consecutive_failures += 1
            srv.last_error = f"HTTP {status}: {detail}" if status else detail
            srv.last_checked_at = time.time()
            if status is None or status >= 500:
                srv.reachable = False
                if status != 429:
                    cooldown = int(min(max(15, self.upstream_timeout), 600))
                    srv.cooldown_until = now + cooldown
            elif status and status < 500:
                srv.reachable = True   # it answered; the request was the problem

    def _fail(self, req: RouterRequest, reason: str) -> None:
        req.status = "failed"
        req.error = reason
        req.finished_at = time.time()
        self._respond(req, 502, json.dumps({"error": {
            "message": f"LLM router could not complete the request: {reason}",
            "type": "router_error",
        }}).encode(), "application/json")

    # ---------------- response writing ----------------

    def _start_chunked(self, req: RouterRequest) -> None:
        req.responded = True
        """Streamed relays have no Content-Length, so frame them as chunked.

        HTTP/1.1 with neither Content-Length nor chunked encoding leaves the body
        length ambiguous, and the OpenAI client can hang waiting for a body that
        has already been fully written.
        """
        client = getattr(req, "_client", None)
        if client is None:
            return
        client.send_response(200)
        client.send_header("Content-Type", "text/event-stream")
        client.send_header("Transfer-Encoding", "chunked")
        client.send_header("X-Harness-Router", req.server or "queued")
        client.end_headers()
        client.wfile.flush()

    def _end_chunk(self, req: RouterRequest, data) -> None:
        client = getattr(req, "_client", None)
        if client is None:
            return
        payload = data if isinstance(data, bytes) else data.encode()
        client.wfile.write(b"%X\r\n" % len(payload) + payload + b"\r\n")
        client.wfile.flush()

    def _finish_chunked(self, req: RouterRequest) -> None:
        client = getattr(req, "_client", None)
        if client is None:
            return
        client.wfile.write(b"0\r\n\r\n")
        client.wfile.flush()

    def _respond(self, req: RouterRequest, status: int, payload: bytes,
                 ctype: str) -> None:
        if req.responded:
            # The reply is already partly on the wire; the caller's client will
            # see the stream end, not a second response.
            return
        req.responded = True
        self._respond_head(req, status, ctype, len(payload))
        self._respond_chunk(req, payload)

    def _respond_head(self, req: RouterRequest, status: int, ctype: str,
                      length: int | None = None) -> None:
        client = getattr(req, "_client", None)
        if client is None:
            return
        client.send_response(status)
        client.send_header("Content-Type", ctype)
        if length is not None:
            client.send_header("Content-Length", str(length))
        client.send_header("X-Harness-Router", req.server or "queued")
        client.end_headers()
        client.wfile.flush()

    def _respond_chunk(self, req: RouterRequest, data) -> None:
        client = getattr(req, "_client", None)
        if client is None:
            return
        client.wfile.write(data if isinstance(data, bytes) else data.encode())
        client.wfile.flush()

    # ---------------- health probing ----------------

    def _probe_loop(self) -> None:
        while not self._stop.is_set():
            self.probe_all()
            self._stop.wait(max(5, self.config.llm_probe_seconds))

    def probe_all(self) -> list[dict[str, Any]]:
        """Check every endpoint is reachable, so the UI can show it."""
        results = []
        for name in list(self._servers):
            results.append(self.probe(name))
        return results

    def probe(self, name: str) -> dict[str, Any]:
        srv_cfg = self.server_config(name)
        with self._lock:
            state = self._servers.get(name)
        if srv_cfg is None or state is None:
            return {"name": name, "reachable": False, "error": "unknown server"}
        headers = dict(srv_cfg.resolved_headers())
        key = srv_cfg.resolved_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = srv_cfg.url_for("/models")
        try:
            resp = self._http.get(url, headers=headers, timeout=10.0)
            ok = resp.status_code < 400
            detail = None if ok else f"HTTP {resp.status_code}"
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        with self._lock:
            state.reachable = ok
            state.last_checked_at = time.time()
            state.last_error = detail
            if ok:
                state.last_ok_at = time.time()
                state.cooldown_until = 0.0
        return {"name": name, "reachable": ok, "error": detail}

    # ---------------- manual test prompt ----------------

    def test_prompt(self, name: str, prompt: str,
                    max_tokens: int = 64) -> dict[str, Any]:
        srv_cfg = self.server_config(name)
        if srv_cfg is None:
            return {"ok": False, "server": name,
                    "error": f"No LLM server named {name!r}."}
        return self.test_prompt_config(srv_cfg, prompt, max_tokens)

    def test_prompt_config(self, srv_cfg: LLMServerConfig, prompt: str,
                           max_tokens: int = 64) -> dict[str, Any]:
        """Send one prompt straight to an endpoint and report what happened.

        Deliberately bypasses the queue: when a user has just added a server they
        want an immediate answer about *that* endpoint, not a slot in a fleet.
        Accepts a config that is not in the server list yet, so a new endpoint
        can be proven before it is saved.
        """
        name = srv_cfg.name
        headers = {"Content-Type": "application/json",
                   "Accept": "application/json", **srv_cfg.resolved_headers()}
        # A definition that is not in the config yet gets only the key that was
        # typed for it, never the harness-wide one.
        tracked = name in self._servers
        key = srv_cfg.resolved_api_key(use_env=tracked)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {
            **srv_cfg.extra_body,
            "model": srv_cfg.wire_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if srv_cfg.reasoning_effort:
            body["reasoning_effort"] = srv_cfg.reasoning_effort
        for key in srv_cfg.body_keys_to_drop():
            body.pop(key, None)
        started = time.monotonic()
        try:
            resp = self._http.post(srv_cfg.chat_completions_url(), json=body,
                                   headers=headers, timeout=120.0)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self._note_probe(name, False, detail)
            return {"ok": False, "server": name, "error": detail,
                    "seconds": round(time.monotonic() - started, 2)}
        elapsed = time.monotonic() - started
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            data = {"_raw": resp.text[:400]}
        if resp.status_code >= 400:
            self._note_probe(name, False, f"HTTP {resp.status_code}")
            return {"ok": False, "server": name, "status": resp.status_code,
                    "error": json.dumps(data)[:400],
                    "seconds": round(elapsed, 2)}
        usage = data.get("usage") or {}
        completion = int(usage.get("completion_tokens") or 0)
        text, reasoning, finish = "", "", None
        for choice in data.get("choices") or []:
            msg = choice.get("message") or {}
            finish = choice.get("finish_reason") or finish
            text = text or (msg.get("content") or "")
            # Reasoning endpoints answer with content: null and put the whole
            # response in reasoning_content, so an empty reply is not a failure.
            reasoning = reasoning or (msg.get("reasoning_content") or "")
            if text:
                break
        note = None
        if not text:
            if reasoning and finish == "length":
                note = ("The endpoint spent the whole token budget reasoning and "
                        "returned no answer. It works, but raise the token budget "
                        "or use a non-reasoning model for short probes.")
            elif reasoning:
                note = "Endpoint returned reasoning only; no final answer."
            else:
                note = "Endpoint returned an empty reply."
        if tracked:
            self._record_tokens(name, int(usage.get("prompt_tokens") or 0),
                                completion, elapsed)
        self._note_probe(name, True, None)
        return {
            "ok": True, "server": name, "model": srv_cfg.wire_model(),
            "base_url": srv_cfg.base_url, "seconds": round(elapsed, 2),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion,
            "tokens_per_second": (
                round(completion / elapsed, 1) if elapsed > 0 and completion else None),
            "reply": text[:600],
            "reasoning_preview": reasoning[:300],
            "finish_reason": finish,
            "note": note,
        }

    def _note_probe(self, name: str, ok: bool, error: str | None) -> None:
        """Record a reachability result for a tracked endpoint."""
        with self._lock:
            state = self._servers.get(name)
            if state is None:
                return
            state.reachable = ok
            state.last_error = error
            state.last_checked_at = time.time()
            if ok:
                state.last_ok_at = time.time()
                state.cooldown_until = 0.0
                state.requests_ok += 1
                state.consecutive_failures = 0

    # ---------------- dashboard view ----------------

    def status(self) -> dict[str, Any]:
        now = time.time()
        mono = time.monotonic()
        with self._lock:
            self._prune_recent()
            active = [r for r in self._requests.values() if r.active]
            recent = list(self._recent)
            servers = [s.snapshot(mono) for s in self._servers.values()]
        pending = sorted((r for r in active if r.status == "queued"),
                         key=lambda r: r.seq)
        in_flight = sorted((r for r in active if r.status != "queued"),
                           key=lambda r: r.seq)
        return {
            "requests": [self._request_view(r, now) for r in pending + in_flight],
            "recent": [self._request_view(r, now) for r in recent
                       if not r.active],
            "servers": servers,
            "counts": {
                "queued": len(pending),
                "awaiting": sum(1 for r in in_flight if r.status == "awaiting"),
                "generating": sum(1 for r in in_flight if r.status == "generating"),
                "served": self.served,
                "retries": self.retries,
                "servers": len(servers),
                "server_slots": len(servers) * max(
                    1, self.config.llm_server_concurrency),
                "server_slots_in_use": sum(s["in_flight"] for s in servers),
            },
            "uptime_seconds": round(time.time() - self.started_at, 1),
        }

    def _request_view(self, r: RouterRequest, now: float) -> dict[str, Any]:
        return {
            "id": r.id, "seq": r.seq, "status": r.status,
            "agent_id": r.agent_id, "agent_role": r.agent_role,
            "kind": r.kind, "project_id": r.project_id, "task_id": r.task_id,
            "server": r.server, "attempts": r.attempts,
            "failed_over": r.failed_over,
            "received_at": r.received_at, "sent_at": r.sent_at,
            "finished_at": r.finished_at,
            "queued_seconds": round(
                (r.sent_at or now) - r.received_at, 1),
            "running_seconds": round(
                (r.finished_at or now) - (r.sent_at or now), 1)
            if r.sent_at else None,
            "prompt_chars": len(r.prompt_preview()),
            "prompt_preview": r.prompt_preview()[:400],
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "error": r.error,
            "messages": len(r.body.get("messages") or []),
            "stream": bool(r.body.get("stream")),
        }

    def request_prompt(self, request_id: str) -> str | None:
        with self._lock:
            req = self._requests.get(request_id)
        return req.prompt_preview() if req else None

    def capacity(self) -> dict[str, Any]:
        """Router-side capacity, for the orchestrator's scheduling decisions."""
        with self._lock:
            servers = list(self._servers.values())
            now = time.monotonic()
            live = [s for s in servers if not s.cooling(now)]
            limit = max(1, self.config.llm_server_concurrency)
            return {
                "llm_servers": [s.name for s in servers],
                "reachable_servers": [s.name for s in live],
                "server_slots": len(servers) * limit,
                "server_slots_free": sum(max(0, limit - s.in_flight)
                                         for s in live),
                "queued_requests": self._pending.qsize(),
            }


class _Handler(BaseHTTPRequestHandler):
    """Inbound OpenAI-compatible surface: accepts a prompt, returns a reply."""

    protocol_version = "HTTP/1.1"
    server_version = "HarnessLLMRouter"

    @property
    def router(self) -> LLMRouter:
        return self.server.router  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": f"Unknown path {self.path}"}})
            return
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self._json(400, {"error": {"message": f"Invalid JSON body: {exc}"}})
            return
        meta = {
            "agent_id": self.headers.get(H_ATTR_AGENT),
            "agent_role": self.headers.get(H_ATTR_ROLE),
            "kind": self.headers.get(H_ATTR_KIND),
            "project_id": self.headers.get(H_ATTR_PROJECT),
            "task_id": self.headers.get(H_ATTR_TASK),
        }
        # The handler thread must stay alive to write the response, so the
        # dispatcher writes back through this connection via the request.
        req = self.router.submit(body, meta, client=self)
        try:
            while req.active and not self.server.router._stop.is_set():
                time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            req.status = "failed"
            req.error = "client disconnected"
        except Exception:  # noqa: BLE001 - never kill the serving thread
            pass

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health"):
            self._json(200, {"ok": True, "service": "llm-router",
                             "servers": [s["name"] for s in
                                         self.router.status()["servers"]]})
        elif path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": s["model"], "object": "model", "owned_by": s["name"]}
                for s in self.router.status()["servers"]]})
        elif path in ("/v1/router/state", "/router/state"):
            self._json(200, self.router.status())
        else:
            self._json(404, {"error": {"message": f"Unknown path {self.path}"}})

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:  # silence the stdlib access log
        return


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_router(config: HarnessConfig, store: Store) -> LLMRouter:
    """Build the router on a loopback port the OS picks, so it never collides."""
    host = config.router_host or "127.0.0.1"
    return LLMRouter(config, store, host=host, port=config.router_port or 0)
