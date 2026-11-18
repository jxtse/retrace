# Route audit pipeline (audit@k v1.0)

Frozen 2026-07-25. This specification, the prompt text and the code are versioned
together: any change requires a new version number and a re-run of every
condition.

## Metric definition

**audit@k**: for a target molecule t the system returns candidate routes ordered
by its internal evaluator, and the audit gives each route a final verdict
L ∈ {VALID, SUSPECT, FAKE}.

audit@k = (1/N) · Σ_t 1[at least one of the top-k routes passes]

- **Denominator N = every target in the dataset.** Targets that returned no route
  or whose run failed count as failures.
- **Two readings**: strict counts only VALID as a pass; lenient also counts
  SUSPECT when the adversarial pass did not overturn it.
- Unlike ML pass@k, audit@k is a success-within-the-top-k measure over a single
  ranked list, not an estimate over k independent samples; there is no
  combinatorial correction.
- If fewer than k candidates exist, the existing ones are used as-is.
- Uncertainty: bootstrap over targets, 95 % CI, B = 10,000, seed 42. This reflects
  target sampling uncertainty; each condition is one complete run.

## Pipeline

### Stage 0 — deterministic pre-check (RDKit only, zero variance)

For every step of a route (`product>>reactants` retro-SMILES):

1. **parse_ok**: every SMILES parses and canonicalises with RDKit;
2. **target_produced**: some step's product canonicalises to the target;
3. **annotations (not fatal)**: per-step carbon_delta (product carbons minus the
   sum over reactants) and ring_delta (ring-size change against the reactants).

**Fatal rules**: any_unparseable, or target_produced = false, gives the final
verdict FAKE (defect TARGET_NOT_PRODUCED / ERROR); the route does not reach later
stages.

A carbon-conservation violation (carbon_delta > 0) is **not** fatal at stage 0:
route notation allows omitting simple reagents (formaldehyde, CO2, protecting
groups), so a mechanical verdict there would produce false positives. Violations
are passed to stage 1 as structured annotations to check closely.

### Stage 1 — deep read (frozen LLM judge)

- judge: a frontier LLM through an OpenAI-compatible chat-completions endpoint
  (`JUDGE_BASE_URL`); temperature 0.2, max_tokens 32000, no tool calls, the route
  record inlined in the prompt
- planner identity is blinded: the prompt carries no condition label and no model
  name, and the route uid never appears in the prompt body
- output schema: route_verdict ∈ {VALID, SUSPECT, FAKE}; per-step verdict and
  defect_code; worst_defect_code
- defect taxonomy (12 classes): NONE, TARGET_NOT_PRODUCED, WRONG_RING_SIZE,
  FABRICATED_OR_MISSING_SKELETAL_CARBON, WRONG_REGIO_OR_CONSTITUTIONAL,
  IMPOSSIBLE_OR_NONSENSE_REACTION, WRONG_REAGENT_OR_CONDITIONS, STEREO_ONLY,
  NON_BUYABLE_LEAF, DISCONNECTED_TREE, OTHER, ERROR
- route-level aggregation (mechanical): any step FAKE → route FAKE; otherwise any
  step SUSPECT → SUSPECT; otherwise VALID

### Stage 2 — adversarial defence (same judge, opposite goal, independent context)

- **Trigger rule (fixed)**: fires when the stage-1 route_verdict is FAKE, and only
  then.
- The defender sees the route record plus the steps and reasons marked FAKE; it
  does not see the rest of the stage-1 output.
- Output: fake_upheld ∈ {true, false} plus corrected_verdict. If fake_upheld =
  false the final verdict becomes corrected_verdict; otherwise FAKE stands.

### Final aggregation (mechanical, no discretion)

Stage 0 fail → FAKE; stage 1 FAKE and stage 2 upheld → FAKE; stage 1 FAKE and the
defence succeeds → corrected_verdict; stage 1 SUSPECT → SUSPECT; stage 1 VALID →
VALID.

## Frozen settings

| Item | Value |
|---|---|
| judge model | a frontier LLM chosen per run; the served alias is recorded in the result header |
| temperature / max_tokens | 0.2 / 32000 |
| RDKit | the version installed on the host, recorded at run time |
| data | routes returned by the planner under one condition set, up to top-10 per target |
| ranking source | the scratchpad `routes` rank field (the system's internal evaluator; the audit does not re-rank) |
| prompt | `DEEPREAD_PROMPT` / `ADVERSARY_PROMPT` inside `audit_driver.py`, archived with the code |
| aggregation | as above, implemented in code |
| bootstrap | B = 10,000, seed 42, resampled over targets |

## Known limits

- The audit is an LLM judgement, not chemical ground truth. Reporting should be
  calibrated against chemist agreement: stratified sampling by verdict × defect
  class × condition, at least two chemists, double blind, human–human kappa
  reported before audit–human agreement, with per-planner consistency tests.
- If the judge and one of the planners share a vendor, state it as a limitation
  and report per-planner agreement.
- `starting_materials` is usually empty in this data, so the dangling-reactant
  check is disabled (`n_dangling = -1` means not applicable).
- Leaf purchasability is guaranteed by the runtime stock check at run exit; the
  audit does not repeat it.
