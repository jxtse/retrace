# S3 existing-rollout first-pass screening v1

Scope: 260 existing S2 v2 medium train130 rollouts, two per target. No new rollout, inference service, training, external paid judge, source edits or repair. Primary goal: identify targets with a defensible final-route candidate vs need for supplementation, with unresolved evidence separated. This is exploratory model-assisted first-pass admission, NOT expert chemical certification, full reasoning-token approval, a frozen training export, or proof of learning benefit.

## Two levels

- Whole-trajectory final-route candidate admission. Native failure is NOT eligible as a complete-success trajectory; valid local segments are a separate deferred question.
- Process review of recorded visible narration, tool actions, scoring/saving and correction retention. Full hidden/reasoning-token quality is NOT exhaustively audited. Never imply all thinking tokens in an admitted candidate are safe for loss.

## Verdicts per trajectory

ADMIT_CANDIDATE: native success, final route structurally connected, no concrete substantial chemical defect found in its actual transformations after inspecting every final step, no unresolved serious stated-mechanism/graph conflict, and no known correction contradicted by final artifact. Plausibility is enough for this first pass; lack of literature/conditions alone is not exclusion. It remains provisional until training-span/mask review.
EXCLUDE: native failure OR a concrete substantial final-route defect OR a substantive agent mechanism error retained in the final result OR explicit recognition of a supported defect that remains in the selected route. Do not use native score, atom deficits, radical presence, nonidentical reference or unsupported general chemical intuitions alone as proof. Missing routine reagents/conditions can be underspecification, not impossible chemistry.
HOLD: substantial concern cannot be resolved from the provided data or the chemistry judgment is genuinely ambiguous. Review failure or incomplete reading is HOLD, never chemical EXCLUDE.

## Checks

1. Read every final step literal and final notes; check connectivity, reaction class, retained skeletal/heteroatom identity, key regio/stereochemical claims. Use toolkit facts as factual assistance, not chemical truth.
2. Look for agent-recognized errors; follow correction → revalidation → save → actual final selection. Warnings on rejected alternatives do not contaminate final route. Score warnings themselves are not an independent verdict.
3. Inspect process timeline. Distinguish productive recovery from warning-blind commits. Empty API retries are NOT automatic whole-trace exclusion. Few tokens/duration is not the ranking criterion. Hitting 40 iterations is not automatically bad; harness can force continued search.
4. If uncertainty requires reading more, use exact local sources/<key>/transcript.json events. You may use lightweight local Python/RDKit if already available, but no installs, inference, networking or SSH unless parent explicitly requests. Do not modify sources or facts.
5. For both successful repetitions of a target, choose the better ADMIT_CANDIDATE by stronger final chemical plausibility, evidence-backed correction retention and useful action progression. Ties may be declared; parent will preserve tie and use rep1 as a documented deterministic representative, not claim higher quality. Never rank solely by score, length or emptiness.

## Required record schema

One JSON object per assigned target in a JSON list, written to the worker-specific reviews/shard-NN.json. Fields:
- target_id: integer
- trajectories: [{key, verdict, confidence: high|medium|low, route_assessment_zh, process_assessment_zh, substantive_flags: [short codes], evidence: [{source: result|events, event_index: integer|null, step_number: integer|null, quote: exact literal contiguous source substring, explanation_zh}], reading_scope: {all_final_steps_read: bool, compact_process_read: bool, extra_event_indices: [int]}, followup_zh}]
- preferred_key: admitted key or null
- selection_reason_zh
- needs_additional_rollout: true if both excluded, false if admitted, null if unresolved hold without admission

Quote rules: result evidence quote must occur literally anywhere in that source result JSON's decoded string values (usually best_route.steps/notes). Events quote must occur in string values of the indexed original event, including arguments/result nested strings. No ellipses inserted. A structural interpretation is YOUR analysis; put it in explanation, not a fabricated original quote. All verdicts require at least one verifiable quote. For ADMIT provide at least one decisive representative step plus final artifact link via key; assertion of all-step reading is separately recorded. Prefer concise specific evidence over many long quotations.

## Independence / limitations

Workers use isolated contexts of the same model family. They do not see each other's judgments; this is independent reading, not independent model-family validation or expert gold. Three preselected overlapping targets will be reviewed by two workers for calibration; disagreement must be reconciled from sources, not majority vote. Reviewers may not close the parent task or launch follow-up sampling.
