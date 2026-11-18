# NOTE — the stage-1 sampling step

`rs_gen.py` and `rs_gen_facts.py` are the drivers that produced the
rejection-sampling pool from which the stage-1 rejection-sampling rows were
admitted. `finalize_rows.py` turns a raw sample into a training row.

**They are documentation, not runnable code in this repository.** They import
support packages from our internal tree that are not part of this release:

| Import | What it is | Status |
|---|---|---|
| `thinking_v2` (`native.py`, `common.py`, `data.py`, …) | token-exact renderer / tokenizer wrapper that makes swift, HF and vLLM agree on `input_ids` | not included (hard-wired to internal paths) |
| `rewrite_v3.data` | instruction text and the `teacher_messages()` builder used by the sampling prompts | not included |
| `thinking_sft.common.GateError` | shared exception type imported by `thinking_v2.common` | not included |
| `gen` | sampling loop + admission gates over a `thinking_v2` candidate | not included (see below) |
| `finalize_rows` | included here | — |

## The gates module

`gen.py` is not shipped. It was adapted from a third-party
"Finetuning-with-Sampling"-style sampler whose licence we could not establish from
the trees available to us, which is not a basis for releasing it. It also imports
the packages above, so it would not run here anyway.

The gate contract it enforced — and that `screening/apply_audit.py` reuses before
a row can enter training — is:

1. the sample is **complete**, with well-formed boundaries (`complete(ids)`);
2. **thinking and final answer are both non-empty** after decoding;
3. no **embedded control tokens** in either field;
4. no **explicit reference-dependency** phrases (`DEPENDENCY`); a plain *mention*
   is recorded as a flag, not a rejection (`MENTION`);
5. no **leak spans** (`LEAK` regex: hidden / reference / answer key / teacher /
   student / "for me only" …), with ±40 characters of context recorded per hit;
6. the parsed `KEEP/REJECT` decisions in the final answer must **equal** those
   parsed from the privileged reference (`reference_decisions_mismatch`
   otherwise), and any SMILES the model introduces must appear in the student
   question or the reference (no invented structures);
7. acceptance during sampling is lexicographic on `(complete, mean_logprob)` under
   the **student** prefix, with suffix budget `65536 − kept prefix length`.

Re-implementing (1)–(6) reproduces the admission behaviour of this step; no other
selection rule was applied here.

## Invocations as run

```
python rs_gen.py <service_url> <shard_index> <n_shards> [K] [limit]
python rs_gen_facts.py <url> <shard> <n_shards> <K> <ids.json> <out_dir>
python apply_audit.py pilot_inputs.jsonl pilot_audit.jsonl applied.jsonl   # screening/
```

`chain_rs_pilot.sh` shows the pilot wiring (two vLLM replicas, 32-way concurrency,
two shards).
