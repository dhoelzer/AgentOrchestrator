"""LLM + runtime configuration for the orchestrator harness.

The LLM is swapped out purely by changing ``base_url`` + ``model`` (plus an API
key); every agent in the harness is built from an OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openhands.sdk import LLM

# Preconfigured default: the local Qwen endpoint. Override with env vars or
# config.json without touching any code.
DEFAULT_QWEN_BASE_URL = "http://192.168.2.107:8080/v1/"
DEFAULT_QWEN_MODEL = "openai/qwen"
DEFAULT_QWEN_API_KEY = "local-no-key"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"


class LLMConfig(BaseModel):
    """OpenAI-compatible endpoint settings shared by every agent."""

    base_url: str = Field(default=DEFAULT_QWEN_BASE_URL)
    model: str = Field(default=DEFAULT_QWEN_MODEL)
    api_key: str = Field(default=DEFAULT_QWEN_API_KEY)
    api_key_env: str | None = Field(
        default=None,
        description="Optional env var to read the API key from (overrides api_key).",
    )
    max_input_tokens: int | None = Field(default=131_072)
    max_output_tokens: int | None = Field(default=8_192)
    request_timeout: int = Field(default=300)
    num_retries: int = Field(default=3)
    drop_params: bool = Field(default=True)
    native_tool_calling: bool = Field(default=True)
    reasoning_effort: str | None = Field(default=None)

    def resolved_api_key(self) -> str:
        if self.api_key_env:
            return os.environ.get(self.api_key_env) or self.api_key
        return os.environ.get("HARNESS_LLM_API_KEY") or self.api_key

    def build_llm(self, usage_id: str) -> LLM:
        """Build an SDK LLM bound to this endpoint.

        ``usage_id`` keeps per-agent metrics separate so the UI can attribute
        token usage to a specific agent.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.resolved_api_key(),
            "base_url": self.base_url,
            "api_mode": "chat",
            "timeout": self.request_timeout,
            "num_retries": self.num_retries,
            "drop_params": self.drop_params,
            "caching_prompt": False,
            "usage_id": usage_id,
            "retry_min_wait": 2,
            "retry_max_wait": 30,
        }
        if self.max_input_tokens:
            kwargs["max_input_tokens"] = self.max_input_tokens
        if self.max_output_tokens:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return LLM(**kwargs)


class HarnessConfig(BaseModel):
    """Top-level configuration loaded from config.json / environment."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    data_dir: str = Field(default=str(DEFAULT_DATA_DIR))
    projects_dir: str = Field(default=str(DEFAULT_DATA_DIR / "projects"))
    tick_seconds: int = Field(default=60)
    sub_agent_turn_limit: int = Field(default=100)
    orchestrator_turn_limit: int = Field(default=60)
    max_parallel_workers: int = Field(default=3)
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

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "HarnessConfig":
        """Load config.json, then apply environment overrides."""
        cfg_path = Path(path or os.environ.get("HARNESS_CONFIG") or REPO_ROOT / "config.json")
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
        if env.get("HARNESS_DATA_DIR"):
            cfg.data_dir = env["HARNESS_DATA_DIR"]
            cfg.projects_dir = str(Path(cfg.data_dir) / "projects")
        for key, attr in (
            ("HARNESS_TICK_SECONDS", "tick_seconds"),
            ("HARNESS_SUB_AGENT_TURN_LIMIT", "sub_agent_turn_limit"),
            ("HARNESS_MAX_PARALLEL_WORKERS", "max_parallel_workers"),
            ("HARNESS_PORT", "port"),
        ):
            if env.get(key):
                setattr(cfg, attr, int(env[key]))
        if env.get("HARNESS_HOST"):
            cfg.host = env["HARNESS_HOST"]

        cfg.ensure_dirs()
        return cfg
