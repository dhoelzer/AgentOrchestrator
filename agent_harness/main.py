"""Entry point: start the web UI and the forever-running orchestrator loop.

    python -m agent_harness.main
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

import uvicorn

from .config import HarnessConfig, resolve_config_path
from .orchestrator import Orchestrator
from .runtime import AgentRuntime
from .store import Store
from .tools import register_harness_tools
from .version import version_string
from .web.api import build_api

logger = logging.getLogger("harness")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenHands orchestrator harness")
    parser.add_argument("--config", default=None, help="Path to config.json")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--tick", type=int, default=None, help="Seconds between cycles")
    parser.add_argument("--model", default=None, help="OpenAI-style model name")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key for that endpoint")
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit")
    return parser.parse_args(argv)


def apply_overrides(config: HarnessConfig, args: argparse.Namespace) -> HarnessConfig:
    if args.model:
        config.llm.model = args.model
    if args.base_url:
        config.llm.base_url = args.base_url
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.tick:
        config.tick_seconds = args.tick
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    config.ensure_dirs()
    return config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_overrides(HarnessConfig.load(args.config), args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    register_harness_tools()

    store = Store(config.data_dir)
    runtime = AgentRuntime(config, store)
    orchestrator = Orchestrator(config, store, runtime)

    logger.info("Orchestrator harness v%s", version_string())
    logger.info("LLM: model=%s base_url=%s", config.llm.model, config.llm.base_url)
    logger.info("Sub-agent turn limit: %s | tick: %ss",
                config.sub_agent_turn_limit, config.tick_seconds)
    logger.info("Data dir: %s", config.root_dir)
    stored = store.meta.get("harness_version_stored")
    if stored and stored != version_string():
        logger.warning("state.json was last written by harness %s (running %s)",
                       stored, version_string())
    store.meta["harness_version"] = version_string()
    store.flush()

    if args.once:
        reports = orchestrator.run_cycle()
        for r in reports:
            logger.info("cycle %s -> %s", r.project_id, r.orchestrator_status)
        store.flush()
        return 0

    app = build_api(config, store, orchestrator,
                    config_path=str(resolve_config_path(args.config)))

    loop_thread = threading.Thread(
        target=orchestrator.run_forever, name="orchestrator", daemon=True
    )
    loop_thread.start()

    server_config = uvicorn.Config(
        app, host=config.host, port=config.port, log_level="info", access_log=False
    )
    server = uvicorn.Server(server_config)

    def shutdown(signum, frame):  # noqa: ARG001
        logger.info("Signal %s received; shutting down.", signum)
        orchestrator.stop()
        server.should_exit = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info("Web UI: http://%s:%d", config.host, config.port)
    try:
        server.run()
    finally:
        orchestrator.stop()
        loop_thread.join(timeout=10)
        store.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
