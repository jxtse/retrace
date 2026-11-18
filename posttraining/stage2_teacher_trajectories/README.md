# Stage 2 — teacher trajectories, route admission and turn-level masking

The stage-2 adapter is trained on teacher trajectories whose final route passed
the frozen audit, with supervision restricted to the turns a selector judged
worth keeping. Order of operations (each step fails closed on a digest mismatch):

1. **Split** — `split/freeze_split.py` → `SPLIT_STAGE2.json` (130 train / 20 dev /
   40 eval, Butina clusters on Morgan r=2, 2048-bit, Tanimoto 0.6, seed 42). The
   40 evaluation targets are never used for collection.
2. **Collection** — three replicates × 130 training targets through the planning
   harness with the teacher model (`collection/run_remote.py`;
   `append_rep3.py` appends the third replicate; `prepare_verify_remote.py`
   verifies the isolated remote copy).
3. **Route audit** — `audit/prepare.py` freezes each unique `(target, route)` for
   the unchanged audit v1 (`audit/AUDIT_SPEC.md`); `audit/run_audit.py` drives it
   single-worker with `audit/audit_driver.py` (deep read + adversarial FAKE
   defence), grounded by the deterministic RDKit facts from
   `audit/grounding_reference.py`. Only VALID trajectories are admitted;
   SUSPECT / FAKE / ERROR are excluded, and transport errors are never
   re-labelled as chemistry failures.
4. **Turn selection** — `selector/run_selection.py` + `selector/seg_select.py`
   score every assistant turn twice with `selector/prompt_v4.txt` (mean overall
   ≥ 7, risk veto ≥ 3). `selector/FROZEN_v4.json` pins the prompt, runner,
   compactor and mechanism-flag digests plus those thresholds.
5. **Rendering and masking** — `assemble/build_full_v1.py` re-renders each
   trajectory exactly as the deployed student sees it and marks per-turn spans;
   `assemble/assemble_a2_v1.py` keeps only VALID trajectories and intersects those
   spans with the selector decisions (`A2_VALUE_TH=7`; `A2_VALUE_TH=0` gives the
   risk-only ablation); `assemble/mech_flags_v1.py` supplies the deterministic
   exclusion features; `assemble/compact_traj.py` builds the review view.
   Supervision covers visible text, tool calls and the `<|im_end|>` of kept
   assistant turns only.
6. **Serialisation and parity** — `train/convert_and_check.py` serialises to
   ms-swift rows and checks token ids and the loss mask against the native
   template; `train/check_equiv.py` re-checks any row by uid;
   `train/convert_a1_medium.py` re-renders at `reasoning_effort=medium`
   (train/infer consistency) and asserts that the only difference versus `low` is
   the removed low-effort instruction block, which carries no supervised tokens.
7. **Training** — `train/run_a1v2.sh zero3`: ms-swift + DeepSpeed ZeRO-3 + 4-way
   sequence parallelism, `--adapters` = stage-1 adapter, 167 rows, global batch 4
   → 42 steps.

## Pins

`selector/FROZEN_v4.json` records the SHA-256 of the prompt, the selector, the
compactor and the mechanism-flag module as shipped here, so an edit to any of them
fails the `assemble/assemble_a2_v1.py` assertion. The selector reads its endpoint
from `ASTRA_BASE_URL`.

## Data you must supply

`full_v1/INDEX.jsonl` and `full_v1/tok/` (rendered trajectories and token ids),
`route_audit_v1/` (`verdicts.jsonl`, `route_members.json`) and
`selector/final_v4_s{1,2}/` (two evaluations per trajectory). These are produced by
the steps above; none is distributed here.
