"""LLM + runtime configuration for the orchestrator harness.

The LLM is swapped out purely by changing ``base_url`` + ``model`` (plus an API
key); every agent in the harness is built from an OpenAI-compatible endpoint.

Several endpoints can be configured (``llm_servers``). Each one names the exact
``model`` key it wants in the request body, which is what makes an Ollama-style
server (where the model id is ``qwen2.5:7b``, not a litellm route) usable
alongside an OpenAI-style one. Requests are then spread across the configured
servers round-robin.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from openhands.sdk import LLM

# Preconfigured default: the local Qwen endpoint. Override with env vars or
# config.json without touching any code.
DEFAULT_QWEN_BASE_URL = "http://192.168.2.107:8080/v1/"
DEFAULT_QWEN_MODEL = "openai/qwen"
DEFAULT_QWEN_API_KEY = "local-no-key"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def _truthy(value: str) -> bool:
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")


class LLMServerConfig(BaseModel):
    """One OpenAI-compatible endpoint.

    ``model`` is the literal string put in the request body's ``model`` key, so
    an Ollama server can be given ``qwen2.5:7b`` while an OpenAI-style gateway
    gets ``openai/qwen``. ``provider`` only decides the litellm route prefix;
    both speak the /v1/chat/completions protocol.
    """

    name: str = Field(default="primary")
    provider: Literal["openai", "ollama", "ollama_chat"] = Field(default="openai")
    base_url: str = Field(default=DEFAULT_QWEN_BASE_URL)
    model: str = Field(default=DEFAULT_QWEN_MODEL)
    api_key: str = Field(default=DEFAULT_QWEN_API_KEY)
    api_key_env: str | None = Field(
        default=None,
        description="Optional env var to read the API key from (overrides api_key).",
    )
    max_input_tokens: int | None = Field(default=131_072)
    max_output_tokens: int | None = Field(default=8_192)
    reasoning_effort: str | None = Field(default=None)
    # Extra keys merged into the request body, e.g. Ollama's
    # {"num_ctx": 32768, "keep_alive": "10m"}.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # Body keys this endpoint must NOT receive. The SDK adds
    # `reasoning_effort: "high"` to every request whether or not the endpoint
    # implements it; Ollama's OpenAI layer maps it onto its own `think` option
    # and answers 400 "think value \"high\" is not supported for this model".
    # None means decide from the provider - see `body_keys_to_drop`.
    drop_body_keys: list[str] | None = Field(default=None)
    # Extra HTTP headers this endpoint requires on every request, e.g. a
    # gateway's `X-Project-Slug`. The router attaches these; agents never do.
    headers: dict[str, str] = Field(default_factory=dict)

    # USD per token, used for cost accounting. Defaults price the preconfigured
    # Qwen endpoint ($2/M input, $15/M output); litellm cannot price an unknown
    # custom model on its own, so without these every run reports $0.00.
    input_cost_per_token: float | None = Field(default=0.000002)
    output_cost_per_token: float | None = Field(default=0.000015)

    def resolved_api_key(self, use_env: bool = True) -> str:
        """The key to send to this endpoint.

        ``use_env=False`` is for a server definition that has not been saved
        yet (a test prompt against a URL someone just typed): the harness-wide
        key must not be silently sent to an endpoint nobody has vetted.
        """
        if self.api_key_env:
            return (os.environ.get(self.api_key_env) or self.api_key) if use_env \
                else self.api_key
        if not use_env:
            return self.api_key
        return os.environ.get("HARNESS_LLM_API_KEY") or self.api_key

    def resolved_headers(self) -> dict[str, str]:
        """Headers the router must send to this endpoint, beyond the standard ones."""
        return dict(self.headers)

    def url_for(self, suffix: str) -> str:
        """Absolute URL for an OpenAI-style path fragment.

        Mirrors how the OpenAI client resolves paths: the base URL already
        carries the `/v1` prefix, so `/chat/completions` is appended to it.
        """
        return self.base_url.rstrip("/") + "/" + suffix.lstrip("/")

    def chat_completions_url(self) -> str:
        return self.url_for("/chat/completions")

    def wire_model(self) -> str:
        """The literal value for the request body's ``model`` key.

        Every endpoint in the fleet is spoken to over the OpenAI chat
        completions protocol, so the body must carry the model name the
        endpoint itself knows - ``qwen3:30b`` for Ollama, ``openai/qwen`` for
        the llama.cpp-style primary. Never add a provider prefix here: Ollama
        matches the string against its own model list and 404s with
        ``model 'ollama_chat/qwen3:30b' not found``.
        """
        return self.model

    def body_keys_to_drop(self) -> list[str]:
        """Body keys the router must strip before addressing this endpoint.

        Following the OpenAI API standard is not the same as accepting every
        OpenAI extension: the SDK puts ``reasoning_effort: "high"`` in every
        request, and Ollama's OpenAI layer maps that onto its own ``think``
        option, which only understands ``high``/``low`` for models it marks as
        reasoning-capable - so it answers 400. Omitting the key is what the
        standard allows; the endpoint then reasons however it is built to.
        """
        if self.drop_body_keys is not None:
            return list(self.drop_body_keys)
        # An explicitly configured reasoning_effort means the operator wants it
        # on the wire for this endpoint, so it is not auto-dropped.
        if self.provider in ("ollama", "ollama_chat") and not self.reasoning_effort:
            return ["reasoning_effort"]
        return []

    def route(self) -> str:
        """litellm model string: the provider prefix plus the server's model key.

        Only for building an SDK ``LLM`` that talks to an endpoint *directly*
        (the no-router fallback): litellm picks its client off this string.
        It is never a wire value - see ``wire_model``.

        An already-prefixed model (``openai/qwen``) is left alone, so existing
        configs keep working unchanged.
        """
        if "/" in self.model:
            return self.model
        return f"{self.provider}/{self.model}"

    def fingerprint(self) -> str:
        """Identity of the endpoint an LLM was built for.

        Used to notice that a live conversation is pointed at a stale endpoint
        after the config changed, without comparing secrets.
        """
        return json.dumps([
            self.name, self.route(), self.base_url,
            self.max_input_tokens, self.max_output_tokens,
            self.input_cost_per_token, self.output_cost_per_token,
            self.reasoning_effort, sorted(self.extra_body.items()),
            sorted(self.body_keys_to_drop()),
        ], sort_keys=True)

    def build_kwargs(self, usage_id: str, timeout: int | None = None,
                     num_retries: int = 3, drop_params: bool = True) -> dict[str, Any]:
        """Constructor arguments for an SDK LLM bound to this endpoint."""
        effective_timeout = timeout or 300
        kwargs: dict[str, Any] = {
            "model": self.route(),
            "api_key": self.resolved_api_key(),
            "base_url": self.base_url,
            "api_mode": "chat",
            "timeout": effective_timeout,
            # litellm derives stream_idle_timeout from `timeout`; set it
            # explicitly so a widened hard timeout also widens the idle one.
            "stream_idle_timeout": effective_timeout,
            "num_retries": num_retries,
            "drop_params": drop_params,
            "caching_prompt": False,
            "usage_id": usage_id,
            "retry_min_wait": 2,
            "retry_max_wait": 30,
        }
        if self.input_cost_per_token is not None:
            kwargs["input_cost_per_token"] = self.input_cost_per_token
        if self.output_cost_per_token is not None:
            kwargs["output_cost_per_token"] = self.output_cost_per_token
        if self.max_input_tokens:
            kwargs["max_input_tokens"] = self.max_input_tokens
        if self.max_output_tokens:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.extra_body:
            kwargs["litellm_extra_body"] = dict(self.extra_body)
        return kwargs


class LLMConfig(LLMServerConfig):
    """OpenAI-compatible endpoint settings shared by every agent.

    Also the implicit single server when ``llm_servers`` is not configured, and
    the home of the global tuning knobs that apply to every server.
    """

    name: str = Field(default="primary")
    request_timeout: int = Field(default=300)
    num_retries: int = Field(default=3)
    drop_params: bool = Field(default=True)
    native_tool_calling: bool = Field(default=True)

    # Local/self-hosted endpoints are often slow and occasionally return an
    # empty completion. Rather than fail the task, the runtime grows the
    # timeout on each timeout and retries empty replies.
    adaptive_timeout: bool = Field(default=True)
    min_request_timeout: int = Field(default=60)
    # Ceiling for the adaptive widening. Deliberately modest: widening is for
    # "this model is slow but alive", and a value this high turns a dead
    # endpoint into an hour of dead time per request. The router also caps any
    # single call at router_max_call_timeout regardless.
    max_request_timeout: int = Field(default=900)
    timeout_growth: float = Field(default=2.0)
    empty_response_retries: int = Field(default=2)
    # Consecutive clean runs after which the widened timeout starts shrinking
    # back toward request_timeout. Without this the widening is one-way and
    # permanent: one bad afternoon leaves every later request allowed to hang
    # for the widened duration, and the value survives a restart via store.meta.
    timeout_decay_after_runs: int = Field(default=5)

    def build_llm(self, usage_id: str, timeout: int | None = None) -> LLM:
        """Build an SDK LLM bound to this endpoint.

        ``usage_id`` keeps per-agent metrics separate so the UI can attribute
        token usage to a specific agent. ``timeout`` overrides the configured
        request timeout, which is how the runtime widens it after a slow reply.
        """
        return LLM(**self.build_kwargs(
            usage_id, timeout, self.num_retries, self.drop_params))


def resolve_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Where config.json lives, so UI edits can be written back to the same file."""
    return Path(path or os.environ.get("HARNESS_CONFIG") or REPO_ROOT / "config.json")


class HarnessConfig(BaseModel):
    """Top-level configuration loaded from config.json / environment."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    # Additional endpoints to spread requests across. When empty, `llm` is the
    # only server, which is how every pre-existing config behaves.
    llm_servers: list[LLMServerConfig] = Field(default_factory=list)
    data_dir: str = Field(default=str(DEFAULT_DATA_DIR))
    projects_dir: str = Field(default=str(DEFAULT_DATA_DIR / "projects"))
    tick_seconds: int = Field(default=60)
    # The scheduler interval is a guess about how long a cycle takes. A fixed
    # value is wrong in both directions: too short and cycles pile up behind
    # each other (an orchestrator queued while it is still running), too long
    # and idle capacity goes unused. The loop therefore re-derives the interval
    # from measured cycle duration, aiming to keep the loop this busy.
    dynamic_tick: bool = Field(default=True)
    min_tick_seconds: int = Field(default=10)
    max_tick_seconds: int = Field(default=600)
    # Fraction of the interval a cycle should occupy. 0.5 means "let a cycle
    # take about half the tick", so a 40s cycle settles at an 80s interval and
    # a 5s cycle at the floor.
    tick_duty_cycle: float = Field(default=0.5)
    sub_agent_turn_limit: int = Field(default=100)
    orchestrator_turn_limit: int = Field(default=60)
    # How many *worker* agents may hold the LLM at once. The orchestrator is
    # deliberately not counted: it holds a slot for the whole of a cycle while
    # it decides what to dispatch, so counting it here silently costs one
    # worker slot and caps real concurrency at max_parallel_workers - 1.
    max_parallel_workers: int = Field(default=3)
    # How many runs may be waiting on a concurrency slot at once. This must
    # exceed max_parallel_workers: with a pool the same size as the gate, a
    # dispatched run always finds a slot free, so nothing is ever *visibly*
    # queued — the backlog hides in the thread pool instead.
    llm_queue_depth: int = Field(default=8)
    max_sub_agent_depth: int = Field(default=3)
    max_children_per_agent: int = Field(default=6)
    review_attempts_before_escalation: int = Field(default=3)
    # Start already-staffed cards without asking the LLM. The orchestrator
    # decides *which* agent owns a card; once it has, launching that agent is a
    # mechanical step that needs no judgement, and making it wait for another
    # orchestrator turn is what leaves an idle endpoint with nothing running.
    auto_dispatch: bool = Field(default=True)
    # Uploads land in <project workspace>/inputs so every agent in the project
    # can read them with its own workspace-relative read_file.
    max_upload_bytes: int = Field(default=25 * 1024 * 1024)
    max_upload_files: int = Field(default=12)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8090)

    # ---- LLM router ----
    # Every LLM request in the harness goes through the router, which owns
    # endpoint choice, request formatting, headers, and load balancing.
    router_host: str = Field(default="127.0.0.1")
    # 0 lets the OS pick a free loopback port, so a second stack on the same
    # host never collides with the first.
    router_port: int = Field(default=0)
    # How many requests one endpoint may serve at once. The router hands the
    # oldest pending request to whichever endpoint has a free slot first, so
    # this is the real bound on upstream concurrency.
    llm_server_concurrency: int = Field(default=2)
    # How the router chooses among endpoints that all have a free slot.
    # "throughput" sends work to the fastest endpoint it has measured, which
    # minimises wall-clock time when the fleet is mixed (a 60 tok/s gateway and
    # a 5 tok/s local model are not interchangeable). "round_robin" spreads
    # requests evenly regardless of speed, which is what you want when the
    # endpoints are equally fast and you are sharing load rather than racing.
    router_selection: str = Field(default="throughput")
    # How long a request may sit unassigned before the router gives up with a
    # clean error instead of holding the agent's connection open forever.
    router_queue_timeout: int = Field(default=600)
    # Hard ceiling on one upstream call, independent of adaptive widening.
    # Widening exists because a big briefing legitimately takes a long time to
    # prefill on a local model; it must never be able to hold an endpoint slot
    # for an hour. A half-open socket (Ollama reloading a 30B model, or a
    # generation that died without closing the connection) produces no bytes and
    # no error, so without a ceiling the slot is never reclaimed and everything
    # else queues behind it forever.
    router_max_call_timeout: int = Field(default=900)
    # How long one upstream call may go without producing *any* output - before
    # the first token or between tokens. This is what detects a dead endpoint:
    # a half-open socket yields no bytes and no error, so without it the request
    # holds an endpoint slot until the call ceiling expires.
    #
    # One value covers both gaps because httpx has a single inter-read timeout;
    # it is set above realistic prefill time (a large briefing on a local 30B
    # model can take minutes to emit its first token) and well below the call
    # ceiling, so a stalled call is reclaimed in minutes rather than an hour.
    router_stall_timeout: int = Field(default=300)
    llm_probe_seconds: int = Field(default=20)
    # Agents run unattended, so tool calls are never gated on a human.
    confirm_tools: bool = Field(default=False)
    # Restrict worker/executor tool execution to each project's workspace.
    sandbox_workspaces: bool = Field(default=True)

    @property
    def root_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    def ensure_dirs(self) -> None:
        Path(self.data_dir).expanduser().mkdir(parents=True, exist_ok=True)
        Path(self.projects_dir).expanduser().mkdir(parents=True, exist_ok=True)

    def servers(self) -> list[LLMServerConfig]:
        """Every endpoint requests may be sent to, in round-robin order.

        `llm` is always first so the primary endpoint stays primary even when
        extra servers are configured, and so a config with no `llm_servers`
        needs no special case anywhere else.
        """
        out: list[LLMServerConfig] = [self.llm]
        seen = {(self.llm.base_url, self.llm.wire_model())}
        for srv in self.llm_servers:
            key = (srv.base_url, srv.wire_model())
            if key in seen:
                continue  # same endpoint listed twice would skew the rotation
            seen.add(key)
            out.append(srv)
        return out

    def server_named(self, name: str) -> LLMServerConfig | None:
        for srv in self.servers():
            if srv.name == name:
                return srv
        return None

    @model_validator(mode="after")
    def _unique_server_names(self) -> "HarnessConfig":
        names = [s.name for s in self.servers()]
        if len(set(names)) != len(names):
            raise ValueError(
                f"llm_servers names must be unique; got {names}. The name is the "
                "round-robin key and the metrics label, so duplicates cannot be "
                "told apart.")
        return self

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "HarnessConfig":
        """Load config.json, then apply environment overrides."""
        cfg_path = resolve_config_path(path)
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text())
            cfg = cls.model_validate(raw)
        else:
            cfg = cls()

        env = os.environ
        if env.get("HARNESS_LLM_BASE_URL"):
            cfg.llm.base_url = env["HARNESS_LLM_BASE_URL"]
        if env.get("HARNESS_LLM_MODEL"):
            cfg.llm.model = env["HARNESS_LLM_MODEL"]
        if env.get("HARNESS_LLM_API_KEY"):
            cfg.llm.api_key = env["HARNESS_LLM_API_KEY"]
        if env.get("HARNESS_LLM_SERVERS"):
            # JSON array of server objects; replaces the file's list so an env
            # based deployment is authoritative rather than merged.
            cfg.llm_servers = [
                LLMServerConfig.model_validate(s)
                for s in json.loads(env["HARNESS_LLM_SERVERS"])
            ]
        if env.get("HARNESS_LLM_QUEUE_DEPTH"):
            cfg.llm_queue_depth = int(env["HARNESS_LLM_QUEUE_DEPTH"])
        if env.get("HARNESS_DATA_DIR"):
            cfg.data_dir = env["HARNESS_DATA_DIR"]
            cfg.projects_dir = str(Path(cfg.data_dir) / "projects")
        for key, attr in (
            ("HARNESS_TICK_SECONDS", "tick_seconds"),
            ("HARNESS_SUB_AGENT_TURN_LIMIT", "sub_agent_turn_limit"),
            ("HARNESS_ORCHESTRATOR_TURN_LIMIT", "orchestrator_turn_limit"),
            ("HARNESS_MAX_PARALLEL_WORKERS", "max_parallel_workers"),
            ("HARNESS_MAX_SUB_AGENT_DEPTH", "max_sub_agent_depth"),
            ("HARNESS_ROUTER_MAX_CALL_TIMEOUT", "router_max_call_timeout"),
            ("HARNESS_ROUTER_STALL_TIMEOUT", "router_stall_timeout"),
            ("HARNESS_PORT", "port"),
        ):
            if env.get(key):
                setattr(cfg, attr, int(env[key]))
        for key, attr in (
            ("HARNESS_LLM_REQUEST_TIMEOUT", "request_timeout"),
            ("HARNESS_LLM_MAX_REQUEST_TIMEOUT", "max_request_timeout"),
            ("HARNESS_LLM_MIN_REQUEST_TIMEOUT", "min_request_timeout"),
            ("HARNESS_LLM_NUM_RETRIES", "num_retries"),
            ("HARNESS_LLM_EMPTY_RESPONSE_RETRIES", "empty_response_retries"),
        ):
            if env.get(key):
                setattr(cfg.llm, attr, int(env[key]))
        if env.get("HARNESS_LLM_TIMEOUT_GROWTH"):
            cfg.llm.timeout_growth = float(env["HARNESS_LLM_TIMEOUT_GROWTH"])
        if env.get("HARNESS_LLM_ADAPTIVE_TIMEOUT") is not None:
            cfg.llm.adaptive_timeout = _truthy(env["HARNESS_LLM_ADAPTIVE_TIMEOUT"])
        for key, attr in (
            ("HARNESS_LLM_INPUT_COST_PER_TOKEN", "input_cost_per_token"),
            ("HARNESS_LLM_OUTPUT_COST_PER_TOKEN", "output_cost_per_token"),
        ):
            if env.get(key):
                setattr(cfg.llm, attr, float(env[key]))
        if env.get("HARNESS_HOST"):
            cfg.host = env["HARNESS_HOST"]

        cfg.ensure_dirs()
        return cfg
