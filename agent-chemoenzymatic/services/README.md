# `services/`

Runtime services the loop uses; none of them talks to the model directly.

| File | Contents |
|---|---|
| `concurrent_tools.py` | parallel tool execution with per-tool timeouts |
| `compaction.py`, `autocompact.py` | context compaction (auto and micro thresholds) |
| `result_store.py` | persistent store for tool results referenced by id |
| `transcript.py` | event-level transcript writer (what the metric code reads) |
| `streaming.py` | streaming responses and auth headers |
| `anthropic_stream.py` | messages-protocol streaming adapter |
| `llm_models.py` | per-model request/response quirks |
| `config_schema.py`, `config_models.py` | schema validation for config overlays (the evaluation harness also validates the lean `agent` options here) |
| `precedent_db.py` | precedent route database behind `recall_similar_routes` |
| `reaction_ordering_review.py` | reaction-ordering review helper |
