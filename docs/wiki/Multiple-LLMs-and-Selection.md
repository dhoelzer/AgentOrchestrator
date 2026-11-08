# Multiple LLMs and Selection

The harness can be pointed at several OpenAI-compatible endpoints at once. The router then decides, per request, which endpoint serves it and how the request must be shaped for that endpoint.

Source: `agent_harness/config.py` (`LLMServerConfig`, `HarnessConfig.servers()`), selection in `agent_harness/llm_router.py`.

## The core problem

Different servers identify models differently, and both speak `/v1/chat/completions`:

| Server | Model key it expects |
|---|---|
| llama.cpp-style gateway | `openai/qwen` |
| Ollama | `qwen2.5:7b`, `qwen3:30b` |
| vLLM | `Qwen2.5-14B-Instruct` |
| OpenAI | `gpt-5.5` |

A single global model name cannot address all of them. So **each configured server names the literal model key it wants**, and the router rewrites the request body per endpoint.

## Configuration

```json
{
  "llm": {
    "base_url": "http://192.168.2.107:8080/v1/",
    "model": "openai/qwen",
    "api_key": "local-no-key"
  },
  "llm_servers": [
    {
      "name": "ollama-a",
      "provider": "ollama_chat",
      "base_url": "http://ollama-a:11434/v1",
      "model": "qwen2.5:7b",
      "extra_body": { "num_ctx": 32768, "keep_alive": "10m" }
    },
    {
      "name": "vllm-b",
      "provider": "openai",
      "base_url": "http://vllm-b:8000/v1",
      "model": "Qwen2.5-14B-Instruct"
    }
  ]
}
```

`llm` is always the primary. `llm_servers` adds to it.

## Ordering and de-duplication

```python
def servers(self):
    out = [self.llm]
    seen = {(self.llm.base_url, self.llm.wire_model())}
    for srv in self.llm_servers:
        key = (srv.base_url, srv.wire_model())
        if key in seen:
            continue          # same endpoint listed twice would skew the rotation
        seen.add(key)
        out.append(srv)
    return out
```

- **`llm` is always first**, so a config with no `llm_servers` needs no special case anywhere else, and the primary stays primary.
- **De-duplicated by `(base_url, wire_model())`.** Listing the same endpoint twice would give it double weight in the rotation.
- **Names must be unique** — enforced by a model validator: *"The name is the round-robin key and the metrics label, so duplicates cannot be told apart."*

## Three different "model" strings — do not confuse them

| Method | Value | Used for |
|---|---|---|
| `wire_model()` | the literal `model` string, verbatim | **The request body.** Always verbatim. |
| `route()` | `provider/model`, unless already prefixed | Building an SDK `LLM` that talks to an endpoint **directly** (the no-router fallback). litellm picks its client off this. |
| agent placeholder | `openai/harness-routed` | What every agent's `LLM` is configured with when a router exists. The router replaces it. |

> **Never add a provider prefix to the wire value.** Ollama matches the string against its own model list and 404s with `model 'ollama_chat/qwen3:30b' not found`.

`route()` leaves an already-prefixed model alone (`openai/qwen` stays `openai/qwen`), so existing configs keep working unchanged.

## Per-endpoint request shaping

The router builds the body per endpoint:

```python
body = {**srv.extra_body, **req.body}
body["model"] = srv.wire_model()
if srv.reasoning_effort:
    body["reasoning_effort"] = srv.reasoning_effort
for key in srv.body_keys_to_drop():
    body.pop(key, None)
```

Headers likewise:

```python
headers = {"Content-Type": "application/json", "Accept": "application/json",
           **srv.resolved_headers()}          # e.g. a gateway's X-Project-Slug
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"
```

Agents never set these. A sub-agent produces a prompt; the router attaches the credential and the headers that the chosen endpoint requires.

### `extra_body`

Merged into the request body. This is where Ollama's OpenAI-compatibility extensions live:

```json
"extra_body": { "num_ctx": 32768, "keep_alive": "10m" }
```

`num_ctx` sets the context window; `keep_alive` stops the model being unloaded between requests, which is what makes a second request fast instead of paying a full reload.

### `drop_body_keys`

Following the OpenAI standard is not the same as accepting every OpenAI extension. The SDK puts `reasoning_effort: "high"` in **every** request. Ollama's OpenAI layer maps that onto its own `think` option, which only understands `high`/`low` for models it marks as reasoning-capable — so it answers:

```
400 think value "high" is not supported for this model
```

The default rule:

```python
if provider in ("ollama", "ollama_chat") and not reasoning_effort:
    return ["reasoning_effort"]
return []
```

Omitting the key is what the standard allows; the endpoint then reasons however it is built to. An **explicitly configured** `reasoning_effort` means you want it on the wire for that endpoint, so it is not auto-dropped. Set `drop_body_keys: []` to force the key through, or list other keys to strip.

## Credentials per endpoint

```python
def resolved_api_key(self, use_env=True):
    if self.api_key_env:
        return os.environ.get(self.api_key_env) or self.api_key   # if use_env
    if not use_env:
        return self.api_key
    return os.environ.get("HARNESS_LLM_API_KEY") or self.api_key
```

Precedence for a saved server: `api_key_env` → `HARNESS_LLM_API_KEY` → `api_key`.

`use_env=False` is used for a server definition that has **not been saved yet** — a test prompt against a URL someone just typed into the UI. The harness-wide key must not be silently sent to an endpoint nobody has vetted.

## How the router selects

Full detail on the [LLM Router](LLM-Router) page. Summary:

1. Drop endpoints in cooldown, and endpoints at `llm_server_concurrency` in-flight.
2. Rank the rest:
   - `router_selection: "throughput"` (default) → by **estimated wait**, `(in_flight + 1) / tokens_per_second`.
   - `router_selection: "round_robin"` → by raw load.
3. Break ties by `consecutive_failures`, then by which endpoint has waited longest for work.
4. An endpoint with no samples yet gets a **negative** estimate so it is measured once immediately.

### Choosing a strategy

| Situation | Use |
|---|---|
| Mixed fleet — a 60 tok/s gateway and a 5 tok/s local model | `throughput`. They are not interchangeable; "least loaded" picks the wrong one every time. |
| Equally fast endpoints, sharing load | Either. With equal rates the wait estimate reduces to least-loaded. |
| You want predictable distribution for benchmarking | `round_robin`. |

## Live changes

The config is re-read at the start of every cycle, and the router reconciles its endpoint set:

- **Added server** → available immediately.
- **Removed or reconfigured server** → never scheduled again. Idle agents move to a live endpoint on their own; running ones finish where they are.
- A server whose endpoint changed is removed and re-added under the new identity, **carrying throughput across but dropping stale health** — a new endpoint has not earned a health record yet.

The orchestrator is told, in the briefing:

```
CONFIG CHANGED (applied this tick): LLM servers now: primary, ollama-a
CAPACITY NOW: ... 2 LLM server(s) in rotation (primary, ollama-a)
```

## Testing an endpoint before you commit to it

`POST /api/llm/router/test` runs a prompt against a named server, or against an unsaved definition you pass in (which is where `use_env=False` applies). `POST /api/llm/router/probe` re-runs the reachability probes immediately. See [Web Interface and API](Web-Interface-and-API).

## Gotchas

- **Ollama `base_url` must include `/v1`.** Ollama's OpenAI-compatible surface is `/v1/chat/completions`; the raw `/api/chat` route is a different protocol. The router appends `/chat/completions` to `base_url` exactly as the OpenAI client does.
- **`provider` does not change the wire model.** It only decides the litellm route prefix for the direct-SDK fallback.
- **Cost is per endpoint.** litellm cannot price an unknown custom model, so without `input_cost_per_token` / `output_cost_per_token` every run reports `$0.00`. Set them per server if you care about cost.
- **Concurrency is per endpoint.** `llm_server_concurrency` is the real bound on upstream load; adding a server raises total capacity only if that number is not already saturating it.

## Next

[Configuration Reference](Configuration-Reference) · [LLM Router](LLM-Router)
