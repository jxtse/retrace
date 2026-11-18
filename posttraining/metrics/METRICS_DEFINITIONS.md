# Metric definitions

Four arms — base, S2, S3, teacher SFT — evaluated on the **same 40 held-out
Retro\*-190 targets**. `compute_metrics.py` reads frozen local artefacts
read-only, makes no model calls and no network requests, and emits `metrics.json`
plus `per_target_metrics.csv`.

## 1. Inputs

| Input | What it is | Used for |
|---|---|---|
| snapshots | `<SNAP>/<arm-snapshot-dir>/snapshots/<idx>/{result.json,transcript.json}` | outcome, route chemistry, behaviour, efficiency |
| audit campaigns | `<AUDIT>/campaign-<arm>/campaign.json` | verdicts and defect codes |
| recovery layers (optional) | `recovered-<arm>.json` plus its `POLICY.json` / `normalization-edits.json` | verdicts for arms whose stage-1 responses needed a schema re-parse |
| audit spec | `AUDIT_SPEC.md` next to the campaigns | definitions of VALID / SUSPECT / FAKE, audit@k, bootstrap |

Snapshot directories are named by three-digit target index (`snapshots/010` ↔
`idx=10`); `idx` is taken from `result.json.idx`, never parsed from the directory
name. Paths come from `RETRACE_SNAP`, `RETRACE_OUTPUTS` and the arm table at the
top of `compute_metrics.py`; edit `ARMS` to point at your own snapshot
directories.

### 1.1 Which audit record is authoritative

`campaign.json` per arm holds rank-1 records whose `route_verdict` may be
`ERROR` when a stage-1 response failed a schema check. A recovery pass can re-parse
the **already-saved** responses, map out-of-enum defect codes to `OTHER`, and
supply missing adversarial defences, without regenerating any route. When a
recovered set exists for an arm, its verdicts are used; when it does not, the
campaign record is used verbatim and an `ERROR` counts as a failure, which makes
that arm's audit rate a lower bound. Both values are exposed in `metrics.json`
(`A_outcome.audit_lenient_n` and `A_outcome.raw_campaign_lenient_if_no_recovery`),
so the effect of the recovery layer is visible rather than hidden.

## 2. Denominators

All rates use the fixed target count N for every arm. A target with no route, an
audit `ERROR` or a `NO_ROUTE` verdict counts as a failure and stays in the
denominator. Medians of per-route quantities (route length, evaluator score) use
only targets that have a rank-1 route, and the count used is printed next to the
value (`n=…`).

## 3. Metric definitions

### A. Outcome

| Field | Definition |
|---|---|
| `completion_n` | targets with `result.json.success == true`. `success` is the harness flag: a route reached the target and every leaf is buyable within the round budget. This is route completion. |
| `route_produced_n` | targets with a scored rank-1 route (`result.json.best_route.total_score` present). Separates "produced a route" from "completed". |
| `audit_lenient_n` | rank-1 verdict ∈ {VALID, SUSPECT} — audit@1 pass. |
| `audit_strict_n` | rank-1 verdict == VALID. |
| `audit_error_n` / `no_route_n` | unresolved audit record / no route to audit. Both count as a failure above. |
| `*_ci95` | percentile bootstrap over targets: resample N targets with replacement, B = 10,000, seed 42 (`random.Random(42)`, sequential), take the 2.5th and 97.5th percentiles (indices `int(0.025·B)` and `int(0.975·B)−1`). |
| `*_ci95_clopper_pearson` | exact binomial 95% CI by bisection on the binomial CDF; also reported because the percentile bootstrap degenerates for 100 % rates (`[1.000, 1.000]`). |

Paired tests against the base arm (`paired_vs_base`): targets are joined on `idx`.
**McNemar exact**, two-sided: `p = min(1, 2·Σ_{i≤min(b,c)} C(n,i)·0.5ⁿ)` over
discordant pairs. Paired Δ CI: bootstrap of `mean(arm) − mean(base)` resampling
target pairs (B = 10,000, seed 42).

### B. Chemistry of the rank-1 route

| Field | Definition |
|---|---|
| `verdict_distribution` | count of VALID / SUSPECT / FAKE / ERROR / NO_ROUTE per target. |
| `worst_defect_family_all` | family of the record's `worst_defect` (mapping below), one per audited route. |
| `step_fake_families` | the same families counted per **step** with `verdict == FAKE`, pooled over all audited routes of the arm (a route with several defects is counted several times). |
| `mean_fake_steps_per_audited_route` | mean `n_fake_steps` over audited routes (`n` printed). |
| route length | `n_audited_steps` = steps the audit actually scored; `route_len` = `len(best_route.steps)` in `result.json`. They can differ by one step because the saved route and the audited rendering are not identical objects. Both appear in `metrics.json` (`route_len_mean/median`, and the per-target `route_len`, `n_audited_steps`). |
| `evaluator_score_*` | `best_route.total_score` from the harness evaluator (0–1), median and mean over targets with a route; `feasibility` / `cost` / `greenness` are its sub-scores. These are harness scores, independent of the LLM audit: the audit judges chemistry, the score judges feasibility, cost and greenness. |

Defect-code → family mapping (unknown or legacy codes fall into `other`):

```
FABRICATED_OR_MISSING_SKELETAL_CARBON     → skeletal_carbon_fabricated_or_missing
WRONG_REGIO_OR_CONSTITUTIONAL             → wrong_regio_or_constitutional
WRONG_RING_SIZE                           → wrong_ring_size
IMPOSSIBLE_OR_NONSENSE_REACTION           → impossible_or_nonsense_reaction
WRONG_REAGENT_OR_CONDITIONS               → wrong_reagent_or_conditions
STEREO_ONLY                               → stereo_only
TARGET_NOT_PRODUCED, ERROR                → target_not_produced_or_parse
OTHER, CHEMOSELECTIVITY, UNSTABLE_INTERMEDIATE → other
NONE                                      → not a defect (VALID step/target)
```

### C. Agentic behaviour (from `transcript.json.events`, per-target means)

| Field | Definition (event-level, exact) |
|---|---|
| `turns` | count of `event == "llm_call"` (= iterations, one model turn each). Equals `stats.n_iterations`. |
| `tool_calls` | count of `event == "tool_call"`. |
| `tool_calls_by_name_*` | same, grouped by `name`. One-step expansion is `neural_retro`; template expansion is `suggest_templates`; `explore_branches` is a third expansion path; stock/buyability is `check_buyability`. |
| `rejected` | `tool_result` events whose `result` dict carries a non-empty `error` key. This is the harness's own rejection channel: unparsable SMILES from `validate_smiles`, `save_best_route` gate failures (`score_mismatch_rejected` / `ungrounded_route_rejected`), malformed `score_route` step JSON, empty `neural_retro` results or engine failures, `suggest_templates` on an invalid SMILES, unknown tool. |
| `invalid_smiles_rate` | `smiles_invalid / smiles_validation_calls`, where one validation is one `validate_smiles` tool result and it is invalid iff `result.valid is False`. Attempts that never reached the tool are not observable. |
| `duplicate_tool_calls` | calls whose `(name, canonical JSON arguments)` signature was already issued earlier in the same target (exact repeats only; semantically equivalent but textually different arguments are not counted). |
| `empty_turns` | `llm_call` with `n_tool_calls == 0` **and** `content_chars == 0`: a turn that produced neither prose nor a tool call. |
| `llm_error_turns` | `llm_call` events carrying `error` (for example "fixed output budget exhausted after 6 attempts … empty choices") with `finish_reason = null`. These coincide with empty turns. |
| `scored_with_warnings` / `warning_instances` | `score_route` results with a non-empty machine-readable `warnings` list (fabricated-carbon docks, stereo introduced without matching descriptors, etc.) / total warning strings. |
| `saved_with_warnings` | a `save_best_route` whose `arguments.score_route_result_id` resolves to a `score_route` result that carried at least one warning, i.e. the route was saved despite an explicit harness warning. Prose advice inside tool text is not machine-readable and is not used. |
| `rescore_in_place` | `save_best_route` whose result message starts with "Re-scored existing route in place" (the model re-scored and re-saved a route it had already saved). Primary revision/backtrack proxy. |
| `repeat_scored_fp` | extra scorings of an identical `route_fp` within a target (`Σ(count−1)` over fingerprints scored more than once). Second revision proxy. |
| `self_complete_ignored` | count of `llm_self_complete_ignored` events — the model declared completion and the harness suppressed it (premature self-termination attempts). |
| `end_reason.iteration_cap_40` | targets with `iterations >= config.agent.max_iterations` (40): ran the full budget. |
| `end_reason.wallclock_exceeded` | targets with at least one `wallclock_exceeded` event: stopped by the wall-clock budget before the round cap. |
| `end_reason.other_early_stop` | neither of the above. |

### D. Efficiency

| Field | Definition |
|---|---|
| `wall_seconds_*` | `result.json.wall_seconds` per target (median reported in hours). |
| `iterations_*` | `result.json.iterations`. |
| `saved_routes_*` | `len(result.json.all_routes)`; equal to `scratchpad.n_saved`. |
| `completion_tokens`, `reasoning_tokens` | **null — not derivable.** The snapshots carry no `usage` object and no `*_tokens` key; the harness logs `content_chars` per turn plus separate `thinking`/`reasoning` event text. |
| `content_chars_*`, `reasoning_chars_*` | measured character counts: visible content from `llm_call.content_chars`, reasoning from the `reasoning` event content length. Per-turn values divide by total turns. |
| `est_output_tokens_chars_div4` | **estimate**, `(content_chars + reasoning_chars) / 4` per target. Valid only as a relative cross-arm proxy; it is not a token count. |
| `finish_length_truncations` | count of `finish_reason == "length"`. |
| `budget_exhausted_calls` | `llm_call` events with `error` (empty choices after six attempts under the fixed output budget). |

## 4. What this data cannot answer

- Token accounting of any kind (see D). Any token figure must come from a wire
  log, which the snapshot does not contain.
- Whether the model read and acted on *prose* advice or warnings (only structured
  `warnings` lists exist).
- Run-to-run variance: each arm is one run per target, so all intervals are
  target-sampling intervals, not seed-variance intervals.
- Cost or GPU-time attribution: only `wall_seconds` per target and per-tool
  `elapsed_ms` are recorded.
- Correctness ground truth: the audit is an LLM judgement under `AUDIT_SPEC.md`;
  SUSPECT does not mean executable.
