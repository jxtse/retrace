# Chemistry audit (audit@1) and paired comparison

The audit is one blinded deep read of the rank-1 route by an LLM judge, returning
`VALID` / `SUSPECT` / `FAKE` with per-step reasons, on top of a deterministic
RDKit grounding layer that owns the mechanical facts (ring sizes, per-step carbon
balance, target reconstruction). A route marked `FAKE` gets a second adversarial
pass that tries to defend it before the verdict stands.

| File | Role |
|---|---|
| `AUDIT_SPEC.md` | the audit specification: prompt contract, verdict rules, aggregation, validation |
| `audit_driver.py` | two-stage driver: deep read → adversarial defence of `FAKE` verdicts; resumable (`verdicts.jsonl` keyed by uid), thread pool |
| `adapter.py` | source-bound single-worker adapter with integrity pins on the audit modules. Default action `verify` is fully offline (hash and schema checks); `summarize` and `replay` are offline; `run --execute-owner` is the only action that POSTs to the judge |
| `validators.py` | verdict/schema validators (extracted from `audit_driver.py`, pinned in `SOURCE_PINS.json`) |
| `grounding.py` | RDKit helpers (canonical SMILES, formula, ring sizes) |
| `compare_s3_base.py` | paired comparison of two arms audited under the same protocol |
| `recover_missing_verdicts.py` | schema-recovery pass: turns transport/schema `ERROR` records into a verdict only where the recorded raw response is unambiguous, under `schema_recovery_POLICY.json` |

## Usage

```
python adapter.py verify --manifest accepted40.json                  # offline
python adapter.py summarize --manifest accepted40.json
python adapter.py run --manifest accepted40.json --execute-owner     # real judge POSTs
python audit_driver.py --routes audit_routes.json --out verdicts.jsonl \
    [--limit-cases N] [--conds ...] [--workers N]
python compare_s3_base.py
```

The judge endpoint comes from `JUDGE_BASE_URL` (an OpenAI-compatible
`/v1/chat/completions` gateway). Transport and schema errors stay `ERROR`; they
are never silently counted as chemistry `FAKE`, and the metric code counts them
as failures in the denominator.

## Recovery policy

`schema_recovery_POLICY.json` records the enum-normalisation policy used for one
recovery pass, and `../metrics/METRICS_DEFINITIONS.md` explains how a recovered
record set is used in the metric computation. Recovery only rewrites JSON shape
where the underlying verdict text is unambiguous; it never generates routes and
never re-reads a route. An arm without a recovery layer keeps its raw campaign
record, so an `ERROR` there makes its audit rate a lower bound.
