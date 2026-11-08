# LLM Router

The router is a **continually running HTTP service on its own port**. Every LLM request in the harness — orchestrator, every sub-agent at every depth, every context condenser, and a manual test prompt from the UI — is addressed to it. Agents only produce a prompt; the router owns everything else.

Source: `agent_harness/llm_router.py`.

## Why a service rather than a helper

Two properties that are easy to get wrong, and are the reason for the design:

- **Nothing is pre-assigned.** A request carries no server choice. The dispatcher hands the *oldest pending request* to *whichever endpoint frees up first*. A slow endpoint slows only itself, instead of stranding a request that was pinned to it at dispatch time.
- **It runs continually.** It is a real HTTP service, not a function called from the agent loop, so requests from every source take the same path and are measured the same way.

The inbound side speaks the OpenAI chat-completions contract, which is what the OpenHands SDK (litellm + the OpenAI client) already emits. The router receives a well-formed OpenAI body and only rewrites `model` and the headers per upstream — it does not re-serialize prompts.

## Wiring an agent to the router

`AgentRuntime._router_llm()` builds every agent's `LLM` pointed at the router:

```python
LLM(
    model="openai/harness-routed",   # placeholder; router rewrites per endpoint
    api_key="router",
    base_url=self.router.base_url,   # http://127.0.0.1:<port>/v1
    api_mode="chat",
    num_retries=0,                   # retries are the router's job
    extra_headers={                  # attribution, not routing
        "x-harness-agent":  spec.id,
        "x-harness-role":   spec.role,
        "x-harness-kind":   <orchestrator|worker|reviewer|...>,
        "x-harness-project": spec.project_id,
        "x-harness-task":   spec.task_id,
    },
)
```

The `openai/` prefix on the placeholder is required: litellm picks the provider off the model string *before* it opens a connection, so an unprefixed name fails locally with `LLM Provider NOT provided`. The value is then replaced per endpoint before anything leaves the process.

With no router (unit tests that never send a request) the agent keeps its configured endpoint directly.

## Request lifecycle

Statuses a request passes through:

```
queued ──► awaiting ──► generating ──► completed
   ▲                          │
   └──── failover ────────────┴──► failed
```

| Status | Meaning |
|---|---|
| `queued` | Bytes received, waiting for a free endpoint slot. |
| `awaiting` | On the wire; waiting on the model's first token. |
| `generating` | Tokens actively arriving. |
| `completed` / `failed` | Finished. Kept visible for `RECENT_SECONDS` (12s, max 8 cards) so a card does not vanish the instant it lands. |

The dispatcher wakes on a `Condition` — it assigns only when a request has arrived **and** an endpoint has a free slot at the same instant.

## Endpoint selection

`_free_server_locked()` picks the endpoint:

```python
candidates = [s for s in self._servers.values()
              if not s.cooling(now) and s.in_flight < limit]   # limit = llm_server_concurrency
if router_selection == "throughput":
    rank = (round(self._wait_estimate(s), 3), s.consecutive_failures)
else:                                    # "round_robin"
    rank = (s.in_flight, s.consecutive_failures)
candidates.sort(key=rank)
tied = [s for s in candidates if rank(s) == rank(best)]
return min(tied, key=lambda s: s.last_assigned)   # longest-waiting endpoint wins ties
```

### The wait estimate

```python
def _wait_estimate(self, srv):
    rate = srv.tokens_per_second
    if rate is None:
        return -1.0                      # never served: try it immediately
    return (srv.in_flight + 1) / rate
```

Throughput mode ranks by **estimated wait**, not by raw load. A mixed fleet is not symmetric: a fast endpoint with one request already running still beats a slow idle one — a 500-token reply is 8s at 60 tok/s and 100s at 5 tok/s, so "least loaded" would pick the wrong endpoint every time. With equal rates the estimate reduces to least-loaded, which is the right behaviour when endpoints are interchangeable.

**An endpoint that has never been served gets a negative estimate** so it is tried once immediately. Guessing its rate, however generously, makes its estimate tie with measured endpoints, so it is never picked, never measured, and stays unusable forever. The negative value is self-limiting: it applies only until the first sample lands.

### Throughput measurement

Per endpoint, summed rather than averaged, so the figure is a true mean and not a mean of means:

```
tokens_per_second = total_output_tokens / total_generation_seconds
```

`generation_seconds` is measured on one monotonic clock — mixing wall time with monotonic produces a nonsense duration.

### Probes are advisory

A failed `GET /models` probe does **not** take an endpoint out of rotation:

> an endpoint that answers `GET /models` badly can still serve chat, and one that is genuinely down is parked by the cooldown its own failed request sets. Deprioritising on the probe instead would mean a flaky probe silently removes capacity and the failover path never runs.

A successful probe clears the cooldown.

## Failover

```python
if req.attempts >= len(servers) or req.responded:
    self._fail(req, outcome)
    return
req.failed_over = True
req.status = "queued"; req.server = None
self._pending.put(req)      # back of the queue, not straight onto another endpoint
```

Two invariants:

- **`req.responded` is the hard line.** Once any byte of the reply has gone back to the caller, the response cannot be replaced — writing a second status line into a body that already started is a protocol error for the caller's client.
- **Re-queue at the back.** The request keeps its place in the funnel but waits for the next free endpoint rather than hammering the one that just failed.

## Streaming: why the router forces it upstream

The SDK asks for a complete JSON body (`stream` defaults to false). For a non-streaming call nothing arrives until the reply is finished — so **a slow-but-alive prefill and a half-open socket are indistinguishable**, and the only deadline that can separate them is the call ceiling.

So when the caller did not ask for a stream, the router requests one from the upstream anyway:

```python
body = {**body, "stream": True, "stream_options": {"include_usage": True}}
read_timeout = min(router_stall_timeout, upstream_timeout)
```

Output starts arriving as it is generated, so any long gap means the call is dead and its slot can be reclaimed. The caller still receives the single JSON body it asked for, reassembled by the router. Reassembling also restores failover a live relay cannot have: while the bytes are still buffered in the router, nothing has gone back to the caller, so a stalled request can be retried elsewhere.

If an endpoint refuses `stream=true` with a 4xx, that is a complaint about the option, not a dead endpoint — the router falls back to the plain call rather than parking a healthy server.

## Timeouts

Three different clocks, and they must not be confused:

| Knob | Default | Detects |
|---|---|---|
| `request_timeout` | 300s | The baseline deadline for one upstream call. |
| `router_stall_timeout` | 300s | A gap with **no bytes at all** — before the first token or between tokens. This is what catches a dead endpoint. |
| `router_max_call_timeout` | 900s | Hard ceiling on one upstream call, independent of adaptive widening. |

`upstream_timeout` is read from live config on every request, so a Settings change applies to the next request without a restart. It follows the runtime's adaptive widening but is clamped:

```python
ceiling = max(1, router_max_call_timeout, configured)
return max(1, min(ceiling, max(configured, widened)))
```

The ceiling bounds **automatic** growth. A larger `request_timeout` you wrote into `config.json` is a deliberate choice and is honoured — the runaway was widening drifting past what anyone asked for, not the configured value itself.

Why a ceiling is mandatory: a half-open socket (Ollama reloading a 30B model, or a generation that died without closing the connection) produces no bytes and no error. Without a ceiling the slot is never reclaimed and everything else queues behind it forever.

`router_stall_timeout` sits above realistic prefill time (a large briefing on a local 30B model can take minutes to emit its first token) and well below the call ceiling, so a stalled call is reclaimed in minutes rather than an hour.

## Health and cooldowns

`_mark_failure()`:

| Outcome | Effect |
|---|---|
| `status >= 500` or unreachable | `reachable = False`, cooldown of `min(max(15, upstream_timeout), 600)`s. |
| `429` | **Not cooled down.** A 429 means slow down, not down; cooling the endpoint would push every request onto the rest of the fleet. Still counted as a failure for ranking. |
| `4xx` | `reachable = True` — it answered; the request was the problem. |

Endpoints in cooldown are skipped outright by selection.

## Probing

A probe thread runs `GET <base_url>/models` against every endpoint every `llm_probe_seconds` (default 20s, floor 5s). It exists for the UI and for clearing cooldowns on success; it does not gate routing.

## Observability

`GET /api/llm/router` returns per-endpoint snapshots:

```json
{
  "name": "ollama-2", "provider": "ollama_chat", "model": "qwen3:30b",
  "base_url": "http://ollama:11434/v1",
  "in_flight": 1, "requests_ok": 42, "requests_failed": 3,
  "consecutive_failures": 0, "cooling_down": false, "cooldown_seconds": 0.0,
  "reachable": true, "last_error": null,
  "total_tokens": 18234, "prompt_tokens": 91200,
  "tokens_per_second": 31.4, "last_tokens_per_second": 28.9,
  "generation_seconds": 580.7, "samples": 42
}
```

Plus the live request cards (status, server, attempts, `failed_over`) and counters `served` / `retries` that outlive individual cards.

## Endpoints on the router itself

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | The OpenAI-compatible entry point every agent uses. |
| `GET` | `/v1/models` | Minimal model listing, so a client that validates the endpoint is satisfied. |

The router is bound to loopback by default (`router_host = 127.0.0.1`) and is not exposed outside the container.

## Gotchas

- **`router_port = 0` is the default** — the OS picks a free loopback port so a second stack on the same host never collides with the first. Read the real port from the startup log or `/api/llm/router`.
- **The model key is per endpoint.** `wire_model()` sends the literal configured name. Adding a provider prefix makes Ollama answer `model 'ollama_chat/qwen3:30b' not found`. See [Multiple LLMs and Selection](Multiple-LLMs-and-Selection).
- **`num_retries=0` on the agent side is intentional.** Retries belong to the router, which can move the request to a different endpoint; the SDK retrying would re-enter the same queue.

## Next

[Multiple LLMs and Selection](Multiple-LLMs-and-Selection) · [Configuration Reference](Configuration-Reference)
