# AgentOrchestrator

An **orchestrator harness** built on the [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk). A single root agent plans long-running software projects, creates specialist sub-agents on demand, reviews their output, and drives projects to delivery — unattended, forever.

Every LLM request in the harness, from every agent and every context condenser, goes through one continually running **LLM router** that owns endpoint selection, request formatting, headers, and health tracking. Agents only produce a prompt.

## What it does

- **Configuration swaps the model** by changing a base URL and a model name. Any OpenAI-compatible endpoint works. Preconfigured for a local Qwen endpoint.
- **Multiple LLM endpoints** can be configured at once. Each names the exact model key it wants, so an Ollama server (`qwen2.5:7b`) sits alongside an OpenAI-style gateway (`openai/qwen`).
- **Sub-agents are created dynamically** as specialists for the expertise the orchestrator identifies per task. Sub-agents may create their own sub-agents.
- **A web UI** creates projects and renders a kanban board plus the live agent hierarchy with a status per agent.
- **The loop runs forever.** Each cycle it looks for work needing an agent, coordinates anything waiting for review, flags projects ready for delivery, and refreshes the board.

## Architecture

```
                    ┌──────────────────────────────────────────┐
   browser ───────► │  Web UI + HTTP API (FastAPI, :8090)      │
                    └───────────────┬──────────────────────────┘
                                    │ state.json / config.json
                    ┌───────────────▼──────────────────────────┐
                    │  Orchestrator loop (runs forever)        │
                    │  reload config → sweep → reconcile →     │
                    │  detect → scheduler → LLM briefing       │
                    └───────┬──────────────────────┬───────────┘
                            │                      │
                  deterministic dispatch      judgement (LLM)
                  (Scheduler, no LLM)         (Orchestrator agent)
                            │                      │
                    ┌───────▼──────────────────────▼───────────┐
                    │  AgentRuntime                            │
                    │  gate slots · project slots · turn budget│
                    └───────────────────┬──────────────────────┘
                                        │ every request
                    ┌───────────────────▼──────────────────────┐
                    │  LLM router (HTTP service, own port)     │
                    │  queue · endpoint choice · headers ·     │
                    │  model-key rewrite · health · throughput │
                    └───────┬──────────────┬──────────────────┘
                            │              │
                    ┌───────▼───┐   ┌──────▼────┐   ┌─────────┐
                    │ primary   │   │ ollama-2  │ … │ gateway │
                    └───────────┘   └───────────┘   └─────────┘
```

## Pages

| Page | Contents |
|---|---|
| [Orchestrator](Orchestrator) | The forever loop, cycle phases, attention detection, briefings, turn budgets |
| [LLM Router](LLM-Router) | Request lifecycle, endpoint selection, health, cooldowns, timeouts, streaming |
| [Multiple LLMs and Selection](Multiple-LLMs-and-Selection) | Configuring a fleet of endpoints and how the router chooses among them |
| [Configuration Reference](Configuration-Reference) | Every option, what it does, worked examples |
| [Sub-Agents and Delegation](Sub-Agents-and-Delegation) | Hierarchy, turn budgets, depth, concurrency, budget exhaustion |
| [Tools Reference](Tools-Reference) | Every tool, which role may call it, guardrails |
| [Web Interface and API](Web-Interface-and-API) | The UI, kanban view, and every HTTP route |
| [Kanban and Project Lifecycle](Kanban-and-Project-Lifecycle) | Board columns, transitions, review, delivery, pausing |
| [Deployment and Operations](Deployment-and-Operations) | Docker/compose, environment variables, running it |

## Terminology

| Term | Meaning |
|---|---|
| **Project** | A goal plus its task board, workspace directory, and agents. |
| **Task card** | One unit of work with an objective, acceptance criteria, and deliverables. Lives in a kanban column. |
| **Specialist / sub-agent** | An agent created for one named expertise. Has a finite turn budget per assignment. |
| **Cycle** | One pass of the scheduler loop: housekeeping, then an orchestrator turn if anything needs judgement. |
| **Slot** | Permission to hold the LLM. Distinct from a thread and distinct from a project concurrency seat. |
| **Attention item** | A concrete thing the orchestrator must act on this cycle. No attention → no LLM call for that project. |

## Source layout

```
agent_harness/
  main.py           entry point: router, runtime, loop, web server
  config.py         HarnessConfig / LLMConfig / LLMServerConfig
  llm_router.py     the router service
  orchestrator.py   the root agent and the forever loop
  scheduler.py      deterministic dispatch (no LLM)
  runtime.py        AgentRuntime: slots, budgets, conversations
  store.py          persistence, board vocabulary
  tools.py          the agent-facing tool surface
  prompts.py        system prompts and briefing templates
  web/              FastAPI app, templates, static assets
tests/              standalone test scripts (no test framework required)
```

## Version

Reported by `GET /health` as `harness_version`, in the form `<version>+<git sha>`, e.g. `1.10.0+6cf4127`. See [Configuration Reference](Configuration-Reference) for the knobs that changed in 1.10.0.
