# score_route — schema and interpretation

`score_route(smiles, route_steps)` runs the multi-dimensional route verifier:
three transparent pillars, each backed by a published reference.

## Output (projected)

```
{
  "tool": "score_route",
  "total_score": 0.78,                  # USE THIS for save_best_route + completion gate
  "feasibility": {                      # weight 0.70
    "score": 0.82,
    "n_steps": 2,
    "pre_judge_score": 0.82,            # raw rater mean before the whole-route judge
    "warnings": [...]
  },
  "cost": {                             # weight 0.15
    "score": 0.71,
    "n_steps": 2,
    "step_efficiency": 0.78,            # exp(-n_steps/8.0)
    "leaf_buyability": 0.83,            # Enamine/molbloom/CSV chain
    "leaves": [{"smiles": "...", "source": "enamine", "catalog_id": "..."}],
    "scscore_target": 4.1,              # SAScore (1=easy, 10=hard)
    "scscore_leaves_mean": 2.4,
    "sc_decrease_score": 0.85           # sigmoid(target_sa - mean(leaf_sa))
  },
  "greenness": {                        # weight 0.15
    "score": 0.74,
    "per_step": [
      {"step": 0,
       "atom_economy": 0.81,            # Trost: max(MolWt(p)) / sum(reactant_mw)
       "solvent": {"name": "ethanol", "class": "recommended", "score": 1.0},
       "temperature_C": 25.0, "temperature_score": 1.0,
       "hazardous_reagents": [],
       "reagent_score": 1.0}
    ]
  },
  "weights_used": {"feasibility": 0.70, "cost": 0.15, "greenness": 0.15},
  "notes": [...]
}
```

`total_score = 0.70·feasibility + 0.15·cost + 0.15·greenness`. **Always pass
the top-level `total_score` as the `total_score` arg of `save_best_route`.**

## Sub-score → action

- low `feasibility.score` → the reaction_rater (discriminative classifier) is
  unconvinced by one or more steps. The step is mechanistically suspect; try an
  alternative disconnection.
- a `step N: invalid SMILES (...)` warning → a precursor or product could not be
  parsed/sanitized by RDKit (often a neural engine hallucination). That step is
  scored 0; re-derive the disconnection with a valid structure.
- a `step N: fabricated carbon (C: +k)` warning → the product has more skeletal
  carbons than every precursor combined, so a carbon was conjured from nowhere
  (a wrong reagent homolog at an H2O/NH3-only annulation, or a hallucinated
  skeleton). The step is docked; supply the precursor that actually carries
  those carbons.
- low `cost.step_efficiency` → too many steps. Look for convergent
  disconnections (2 reactants per step ideal).
- low `cost.leaf_buyability` → swap leaves for commercial SMs (use
  `suggest_templates`, `check_buyability`).
- low `cost.sc_decrease_score` → leaves aren't simpler than the target;
  pick decompositions that yield easier precursors.
- low `greenness.per_step[i].solvent.score` → CHEM21 flagged solvent
  (DCM, DMF, etc.). Try alternative conditions or enzymatic chemistry.
- low `greenness.per_step[i].temperature_score` → predicted T is high
  (>100 °C) or cryogenic (<–40 °C). Enzymatic steps usually score 1.0.
- non-empty `greenness.per_step[i].hazardous_reagents` → swap reagent.
