# Tool description style guide (always-on layer)

> Audience: developers adding or editing the `description` field on a `Tool`
> subclass. The `description` is the **always-resident** string the LLM sees
> for that tool on every API call (it is included in the tool schema sent to
> the model on every turn — see `tools/base.py` `to_openai_tool` /
> `to_anthropic_tool`). Treat it as a contract, not marketing copy.

## Why this exists

Before this guide, descriptions ranged from 1 sentence (`analyze_molecule`)
to 8+ sentences (`neural_retro`). Some named their trigger condition, some
didn't. Specialist engines (e.g. `pistachio_ringbreaker`) were buried inside
parent-tool descriptions as parentheticals and never got called. The cost
was paid in agent quality — the LLM cannot use a tool whose trigger it does
not see.

## The 4-question template

Every tool description **must** answer these four questions, in this order:

```
Sentence 1 — WHAT it does (literal, no marketing).
Sentence 2 — WHEN to call / WHEN NOT to call (the trigger condition).
Sentence 3 — COST (latency in seconds; whether it consumes the iteration meter).
Sentence 4 — PAIRING (which tool typically runs in the same turn or next turn).
```

You may merge two questions into one sentence (e.g. `WHAT + WHEN`) when
that's cleaner. You may add a fifth sentence for a critical caveat. You
may **not** drop a question.

## Length budget

- **3–5 sentences** total.
- **≤ 600 chars** total (verification script enforces this; aim for ≤ 500).
- Anything that doesn't fit (parameter tuning, output schema, edge cases,
  worked examples) → push to the JIT section under
  `prompts/sections/tools/<tool_name>.md` and let the LLM pull it via
  `read_doc('tools/<tool_name>')`.

## Tone & format rules

- Imperative voice ("Call when…", "DO NOT call on…"), not declarative.
- Trigger conditions get explicit polarity: when to call **and** when not
  to call. The negative case is what stops the LLM from over-using the
  tool.
- Cost in **literal seconds** ("~30-60s/call"), not adjectives ("slow").
- "Counts toward the iteration budget" / "free" — name the budget
  status explicitly. Default: free. Iteration-billable tools must say so.
- Pairings name the **other tool by name** (`suggest_templates`,
  `explore_branches`, …) so the LLM can compose sequences from
  description alone.
- No emoji. No marketing words ("powerful", "advanced", "intelligent").
- Backticks around tool names and parameter names: `` `engine='ensemble'` ``.

## Example (canonical)

```python
description = (
    "Specialist neural one-step retro for RING-OPENING / RING-MODIFYING "
    "disconnections (Pistachio Ringbreaker model). Call when the target "
    "has a fused / bridged / spiro polycyclic core, a macrocycle, or a "
    "strained 3-/4-membered ring AND the regular `neural_retro` ensemble's "
    "top picks all keep the ring intact — the general engines are weakest "
    "there. DO NOT call on acyclic targets — wastes a slow HTTP round-trip. "
    "Cost: ~10-60s/call, counts toward the iteration budget. Pairs with "
    "`neural_retro` (run both, merge into one `explore_branches` call)."
)
```

5 sentences, 543 chars. Hits all four questions: WHAT (sentence 1) →
WHEN+WHEN-NOT (sentences 2-3) → COST (sentence 4) → PAIRING (sentence 5).

## What goes in the JIT section, not the description

- Full parameter reference (the JSON schema already carries names and
  types; only describe non-obvious behavior in the schema's per-property
  `description`).
- Output schema (field names, types, what each field means).
- Worked examples ("Given target X, expect output shaped like Y").
- Tuning notes ("Set `per_engine_topk=80` for terpene scaffolds").
- Failure modes ("Returns `[]` when the SMILES has no ring atoms").
- Cross-references to other JIT sections.

The description gives the LLM enough to **pick the tool and call it
correctly the first time**. Anything beyond that lives in the JIT section.

## Verification

Run the budget check from the plan's Verification section:

```bash
.venv/bin/python -c "
import importlib, inspect, glob
for path in glob.glob('tools/**/*.py', recursive=True):
    if any(s in path for s in ['_pycache', 'test_', '_subprocess', '_utils']): continue
    mod = importlib.import_module(path.replace('/', '.')[:-3])
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        if hasattr(cls, 'name') and hasattr(cls, 'description') and cls.__name__.endswith('Tool'):
            d = cls.description or ''
            n = d.count('.') + d.count('?') + d.count('!')
            print(f'{cls.name:30s} sentences={n} chars={len(d)}')
"
```

Expect every row to satisfy `3 ≤ sentences ≤ 7` and `chars ≤ 600`.
Multi-clause sentences with `, ` count as one for this purpose.
