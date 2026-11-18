# `core/`

The agent runtime.

| File | Contents |
|---|---|
| `agent.py` | `RetrosynthesisAgent`: the ReACT loop, turn budgeting, completion gate, save/score bookkeeping |
| `agent_factory.py` | `build_agent()`: assembles config, LLM client, tools, services |
| `config.py` | base config, thinking presets, overlay merge, validation |
| `evaluation.py` | benchmark suite registry and delegation to `dataset/benchmark/<suite>/run_benchmark.py` |
| `llm_client.py` | OpenAI-compatible client (chat-completions and messages protocols, retries, output-budget handling) |
| `substrate_client.py` | adapter for a non-OpenAI gateway that needs a specific model alias |
| `session.py`, `iteration.py`, `route_text.py` | session state, iteration counting, route-to-text rendering |
| `prompt_loader.py` | loads the system prompt, JIT sections and nudge messages |
| `repl.py` | the interactive REPL behind `main.py repl` |

Configuration is data: `load_config(thinking=...)` returns a dict, and a
`--config` YAML is deep-merged over the preset.
