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

    # USD per token, used for cost accounting. Defaults price the preconfigured
    # Qwen endpoint ($2/M input, $15/M output); litellm cannot price an unknown
    # custom model on its own, so without these every run reports $0.00.
    input_cost_per_token: float | None = Field(default=0.000002)
    output_cost_per_token: float | None = Field(default=0.000015)

    def resolved_api_key(self) -> str:
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or self.api_key
        return os.environ.get("HARNESS_LLM_API_KEY") or self.api_key

    def route(self) -> str:
        """litellm model string: the provider prefix plus the server's model key.

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
    max_request_timeout: int = Field(default=3600)
    timeout_growth: float = Field(default=2.0)
    empty_response_retries: int = Field(default=2)

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
    sub_agent_turn_limit: int = Field(default=100)
    orchestrator_turn_limit: int = Field(default=60)
    max_parallel_workers: int = Field(default=3)
    # How many runs may be waiting on a concurrency slot at once. This must
    # exceed max_parallel_workers: with a pool the same size as the gate, a
    # dispatched run always finds a slot free, so nothing is ever *visibly*
    # queued — the backlog hides in the thread pool instead.
    llm_queue_depth: int = Field(default=8)
    max_sub_agent_depth: int = Field(default=3)
    max_children_per_agent: int = Field(default=6)
    review_attempts_before_escalation: int = Field(default=3)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8090)
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
        seen = {(self.llm.base_url, self.llm.route())}
        for srv in self.llm_servers:
            key = (srv.base_url, srv.route())
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
