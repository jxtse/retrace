"""Multi-dimensional route verifier.

Three transparent pillars, each backed by a published reference:
  - Feasibility (0.70): discriminative reaction_rater per-step classifier (host-a)
                        — the sole per-step signal; raises if the endpoint is down
                        + RDKit SMILES sanity gate (unparseable step → scored 0)
                        + carbon-skeleton fabrication penalty (positive carbon_delta)
                        + stereo soft warning
                        + whole-route / per-step LLM judge (lift-don't-promote)
  - Cost        (0.15): step_efficiency = exp(-n_steps/8)
                        + leaf buyability (Enamine + molbloom + legacy CSVs)
                        + RDKit SAScore decrease sigmoid (Ertl & Schuffenhauer 2009)
  - Greenness   (0.15): atom_economy (Trost 1991)
                        + solvent class table (CHEM21/Prat 2016; optional overlay)
                        + ASKCOS NeuralNetContextRecommender (Gao 2018) for T/solvent
                          (optional, not bundled; see LICENSES.md)
                        + curated hazardous-reagent SMARTS table

Tool name remains ``score_route`` and the score field is ``total_score``.
"""

from .score_route import RouteVerifierTool

__all__ = ["RouteVerifierTool"]
