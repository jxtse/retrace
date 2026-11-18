# Stage 1 — reference-guided chemical critique

The base model first learns to reason about routes while consulting a *reference*
explanation (a privileged context that the student prompt does not contain), then
produces its own on-policy samples that are screened for leakage and consistency.
Training uses the upstream Self-Distillation loop; this directory supplies the
data adapters around it.

```
sdft/       prompt/teacher-prompt pair construction, native-template rendering,
            per-turn supervision, wire-level admission (+ offline tests)
sampling/   rejection-sampling drivers and row finalisation (see sampling/NOTE.md)
screening/  leak-span and consistency screening, chemistry review export
train/      S1 dataset build + ms-swift SFT launcher
```

Flow: `sdft/` builds pairs → the upstream SDFT trainer runs → `sampling/` adds
on-policy samples → `screening/` admits or rejects them → `train/build_s1v2.py`
packs 819 rows → `train/run_s1v2.sh` trains the S1 adapter. `train/chain_s1v2_a1v2.sh`
chains S1 into stage 2 and checks the adapter.
