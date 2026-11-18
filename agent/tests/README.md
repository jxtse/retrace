# `tests/`

Offline unit tests. No network, no GPU, no model inference: HTTP endpoints are
faked, LLM responses are canned, and anything that needs a live service is
skipped.

```bash
uv run pytest -q
```

`conftest.py` puts the harness root on `sys.path` and provides `FakeLLMClient`.
