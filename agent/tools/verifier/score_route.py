"""Orchestrator for the multi-dimensional route verifier.

Three pillars:
  - Feasibility (default 0.70): reaction_rater per-step classifier + SMILES sanity + carbon-skeleton penalty + stereo
  - Cost        (default 0.15): step_efficiency + leaf economics (ChemPrice) + SCScore
  - Greenness   (default 0.15): atom_economy + hazardous-reagent scan + optional
                              solvent class (CHEM21 table) + optional predicted T
                              (ASKCOS condition model). Neither optional resource
                              is redistributed here; when absent the pillar reports
                              the missing component and degrades instead of failing
                              (see LICENSES.md).

Tool name is ``score_route``; the score field is ``total_score``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tools.base import Tool
from tools._route_fp import route_fingerprint

from .buyability import BuyabilityChain
from .carbon_skeleton import carbon_delta, forms_new_ring
# Optional local overlay: the CHEM21 solvent table is not redistributed (see
# LICENSES.md). Drop a module exposing ``classify_solvent(name) -> (class, score)``
# at this path to restore the solvent sub-score; otherwise it is reported as
# unavailable and excluded from the step average.
try:
    from .chem21_solvents import classify_solvent
except ImportError:  # pragma: no cover - depends on the optional local overlay
    classify_solvent = None  # type: ignore[assignment]
from .condition_prediction import ConditionPredictor, temperature_score
from .hazardous_reagents import reagent_score
from .llm_judge import (
    JudgeConfig,
    WholeRouteJudge,
    combine_step,
    combine_feasibility_and_judge,
    combine_uphold,
    should_invoke_judge,
    should_invoke_step_judge,
)
from .pricing import LeafPricer
from .scscore import sa_decrease_score, sc_score
from .smiles_sanity import check_step_smiles
from .stereo_check import warning_for_step
from ._reaction_rater_bridge import ReactionRaterClient

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env override; unset/blank → ``default``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def atom_economy(reactants: List[str], product: str) -> float:
    """Trost atom economy: max(MolWt(p)) / sum(MolWt(reactants)). Returns 0..1."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except Exception:
        return 0.0
    rmols = [m for m in (Chem.MolFromSmiles(s) for s in reactants) if m is not None]
    pmol = Chem.MolFromSmiles(product)
    if not rmols or pmol is None:
        return 0.0
    r_mw = sum(Descriptors.MolWt(m) for m in rmols)
    if r_mw <= 0:
        return 0.0
    p_mw = Descriptors.MolWt(pmol)
    return max(0.0, min(1.0, p_mw / r_mw))


def _step_to_rxn_smiles(reactants: List[str], product: str) -> str:
    return ".".join([r for r in reactants if r]) + ">>" + product


def _route_fp_from_steps(steps: List[Dict[str, Any]]) -> str:
    """Fingerprint score_route's ``{reactants:[...], product:...}`` steps the
    same way the scratchpad fingerprints its ``product>>reactants`` step
    strings, so the two ids are directly comparable for the save-time
    route-binding check. Note the *retro* direction: ``product>>reactants``.
    """
    retro_steps = [
        f"{s.get('product', '')}>>{'.'.join(r for r in (s.get('reactants') or []) if r)}"
        for s in steps
    ]
    return route_fingerprint(retro_steps)


def _identify_leaves(steps: List[Dict[str, Any]]) -> List[str]:
    """A leaf is a reactant SMILES that never appears as a product of an earlier step."""
    products: set[str] = set()
    leaves: List[str] = []
    seen_leaves: set[str] = set()
    try:
        from rdkit import Chem
        def canon(s: str) -> str:
            m = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(m) if m is not None else s
    except Exception:
        def canon(s: str) -> str:  # type: ignore[no-redef]
            return s

    for s in steps:
        for r in s.get("reactants", []) or []:
            cs = canon(r)
            if cs not in products and cs not in seen_leaves:
                leaves.append(cs)
                seen_leaves.add(cs)
        prod = s.get("product", "")
        if prod:
            products.add(canon(prod))
    return leaves


class RouteVerifierTool(Tool):
    """Multi-dimensional route scorer: feasibility + cost + greenness."""

    name = "score_route"
    description = (
        "Score a route across three pillars: feasibility (reaction_rater "
        "discriminative per-step classifier + carbon-skeleton fabrication "
        "penalty), cost (step efficiency + leaf buyability + SAScore decrease), "
        "greenness (atom economy + solvent class + predicted temperature + "
        "hazardous reagent scan; the solvent table and condition model are "
        "optional and reported as unavailable when absent). Returns "
        "`total_score` in [0,1] (the completion gate), the three pillar "
        "scores, and a `bottleneck` hint. Cost: ~30-200s/call, free; "
        "**per-case budget ~8 calls** — re-score only after a meaningfully-"
        "different candidate is built. Pairs with `save_best_route` (always "
        "save after scoring)."
    )
    tier = 3
    parameters = {
        "type": "object",
        "properties": {
            "smiles": {
                "type": "string",
                "description": "SMILES of the target molecule",
            },
            "route_steps": {
                "type": "string",
                "description": (
                    'JSON array of reaction steps, each with "reactants" '
                    '(list of SMILES) and "product" (SMILES). Example: '
                    '[{"reactants":["A","B"],"product":"C"},'
                    '{"reactants":["C","D"],"product":"E"}]'
                ),
            },
        },
        "required": ["smiles", "route_steps"],
    }

    DEFAULT_WEIGHTS = {"feasibility": 0.70, "cost": 0.15, "greenness": 0.15}

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        cache_dir: Optional[str] = None,
        rater_http_url: Optional[str] = None,
        rater_timeout_s: float = 30.0,
        rater_retries: int = 2,
        enamine_pkl: Optional[str] = None,
        buyability_db: Optional[str] = None,
        legacy_csv_dir: Optional[str] = None,
        condition_model_dir: Optional[str] = None,
        carbon_fabrication_penalty: float = 0.2,
        feasibility_dispersion_lambda: float = 0.3,
        chemprice_enabled: bool = True,
        chemprice_env_file: Optional[str] = None,
        chemprice_verifier_repo: Optional[str] = None,
        chemprice_timeout_s: float = 30.0,
        chemprice_cache_ttl_days: int = 30,
        judge_config: Optional[JudgeConfig] = None,
        llm_client: Any = None,
        result_store: Any = None,
        pillar_parallel: bool = True,
        feasibility_step_workers: int = 4,
    ):
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        
        # Validate that weights sum to 1.0 within tolerance
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-3:
            logger = logging.getLogger(__name__)
            logger.warning("Verifier weights sum to %.6f (expected 1.0); renormalizing", total)
        
        # Renormalize so weights always sum to 1
        total = total or 1.0
        self.weights = {k: v / total for k, v in self.weights.items()}

        self.cache_dir = cache_dir or os.path.join("runs", "_session_default")

        # Carbon-skeleton fabrication penalty. NOT atom/mass conservation —
        # that was removed (organic reactions are written unbalanced). This is
        # the one balance signal that IS diagnostic: a POSITIVE carbon_delta
        # (product has more skeletal carbons than every precursor combined)
        # means a carbon was fabricated, so we dock the step
        # `carbon_fabrication_penalty` per fabricated carbon (capped at the
        # step score). Negative delta (a fragment leaves) is benign and never
        # penalised. The LLM judge can still lift a step this docked. Set 0.0
        # to disable.
        self.carbon_fabrication_penalty = float(carbon_fabrication_penalty)

        # Feasibility dispersion penalty (PR, 2026-06-20). The feasibility
        # pillar stays MEAN-based on purpose — the system targets "almost
        # feasible" routes a chemist lightly fixes, so one weak step should not
        # zero a route the way a min/worst-step aggregator would. But a pure
        # mean barely separates chemist-VALID from chemist-FAKE routes
        # (AUC≈0.66 on the 3304-route RETRACE audit), because a route with 1-2
        # bad steps and several good ones still averages high. So we dock the
        # mean toward its worst surviving step:
        #     feas = mean - lambda * (mean - min)
        # lambda=0 is the old pure mean; lambda=1 is pure worst-step. This runs
        # AFTER the per-step LLM-judge lift, so a rater false-negative the judge
        # already recovered no longer drags `min` down — only a low step that
        # SURVIVED judge review (a genuine defect) is penalised. Calibrated on
        # the audit: lambda=0.3 is the within-target rank-1 optimum (non-FAKE
        # route ranked #1 on 72.7% of targets-with-a-better-option, up from
        # 69.8% at lambda=0) while keeping VALID routes near their mean
        # (mean_VALID 0.795→0.725). Set 0.0 to disable.
        self.feasibility_dispersion_lambda = max(
            0.0, min(1.0, float(feasibility_dispersion_lambda))
        )

        # Discriminative reaction_rater (host-a) — the SOLE per-step feasibility
        # signal (the ReactionT5v2-forward round-trip was removed 2026-06-09;
        # feasibility is now rater + LLM-judge safety net only). The rater is
        # REQUIRED: env override wins, then the constructor arg; if neither is
        # set we hard-fail rather than silently scoring every step 0. The
        # client itself stays fail-soft at request time (rate() returns None on
        # an endpoint blip) — the caller turns that into a loud RuntimeError.
        rater_url = os.environ.get("REACTION_RATER_URL") or rater_http_url or None
        if not rater_url:
            raise ValueError(
                "score_route requires a reaction_rater URL "
                "(verifier.feasibility.rater.http_url or env REACTION_RATER_URL); "
                "feasibility has no fallback scorer since ReactionT5 was removed."
            )
        self._rater = ReactionRaterClient(
            rater_url, timeout_s=rater_timeout_s, retries=rater_retries,
        )

        # Local fallback chain (Enamine pkl / molbloom / sqlite / legacy CSV).
        # Used when ChemPrice is unavailable or on timeout.
        self.buyability = BuyabilityChain(
            enamine_pkl=enamine_pkl,
            buyability_db=buyability_db,
            legacy_csv_dir=legacy_csv_dir,
        )

        # Primary leaf pricer (ChemPrice) — only initialised if enabled.
        if chemprice_enabled:
            price_cache_db = os.path.join(self.cache_dir, "price_cache.db")
            self.pricer: Optional[LeafPricer] = LeafPricer(
                cache_db=price_cache_db,
                env_file=chemprice_env_file,
                verifier_repo_path=chemprice_verifier_repo,
                timeout_s=chemprice_timeout_s,
                cache_ttl_days=chemprice_cache_ttl_days,
                buyability_chain=self.buyability,
            )
        else:
            self.pricer = None

        # Optional ASKCOS condition model (MPL-2.0, not redistributed; see
        # LICENSES.md). When the package or its weights are absent, the
        # greenness pillar records an "unavailable" error per step and scores
        # those steps 0.0 — the same per-step contract used for a failed
        # prediction — instead of refusing to build the tool.
        self.condition = None
        self.condition_unavailable: Optional[str] = None
        if condition_model_dir:
            try:
                self.condition = ConditionPredictor(condition_model_dir)
            except (ImportError, FileNotFoundError, OSError) as exc:
                self.condition_unavailable = str(exc)
                logger.warning("condition model unavailable: %s", exc)
        else:
            self.condition_unavailable = "condition_model_dir not configured"

        # Whole-route LLM judge — feasibility safety net. Built lazily so the
        # LLM client is only constructed when the judge actually fires.
        self.judge_cfg = judge_config or JudgeConfig()
        self._judge: Optional[WholeRouteJudge] = (
            WholeRouteJudge(self.judge_cfg, llm_client=llm_client)
            if self.judge_cfg.enabled
            else None
        )

        # Optional reverse-lookup channel into prior tool calls. Used by
        # the per-step judge to recover a reaction-class hint from
        # earlier ``suggest_templates`` results for the same product +
        # reactants. Same precedent as ExploreBranchesTool.
        self._result_store = result_store

        # Concurrency knobs. The three pillars are disjoint (feasibility →
        # rater/judge, cost → pricer/buyability, greenness → condition
        # predictor) and their per-step backends are independently
        # thread-safe (rater = pooled requests.Session), so they can run
        # concurrently. Env overrides let a batch launch tune or disable
        # parallelism without a config edit.
        self.pillar_parallel = _env_bool("SCORE_ROUTE_PILLAR_PARALLEL", pillar_parallel)
        self.feasibility_step_workers = _env_int(
            "SCORE_ROUTE_FEAS_WORKERS", feasibility_step_workers
        )

    # ──────────────────────────────────────────────────────────────────────
    # execute
    # ──────────────────────────────────────────────────────────────────────

    def execute(self, smiles: Optional[str] = None,
                route_steps: Optional[Any] = None, **kwargs) -> str:
        t0 = time.monotonic()
        # Defensive: LLM occasionally calls without required args. Return a
        # JSON error so the agent can recover and retry instead of bubbling a
        # TypeError up through the tool dispatcher.
        # Tolerate common alias keys the LLM has been observed to emit.
        if smiles is None:
            smiles = kwargs.get("target") or kwargs.get("smiles_target") or kwargs.get("product")
        if route_steps is None:
            route_steps = (kwargs.get("steps") or kwargs.get("route")
                           or kwargs.get("route_json"))
        if not smiles or not isinstance(smiles, str):
            return json.dumps({"error": "missing required argument: smiles (target SMILES string)"})
        if route_steps is None:
            return json.dumps({"error": "missing required argument: route_steps (JSON array of {reactants:[smi], product:smi})"})
        try:
            steps_raw = (
                json.loads(route_steps) if isinstance(route_steps, str) else route_steps
            )
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid route_steps JSON: {e}"})
        if not isinstance(steps_raw, list) or not steps_raw:
            return json.dumps({"error": "route_steps must be a non-empty JSON array"})

        # Structural route-grounding (pure topology, no chemistry): does the
        # route's reaction graph form a tree rooted at the target? Surfaced as
        # an informational block + warning, NOT folded into any pillar score —
        # grounding is orthogonal to per-step feasibility (a route can have
        # every step plausible yet not actually synthesise the target, or be a
        # disconnected graph). The agent's save_best_route gate hard-rejects an
        # ungrounded route; here we only report it so the LLM sees it pre-save.
        grounding: Dict[str, Any] = {}
        try:
            from tools.scratchpad import check_route_grounding
            g_products = [s.get("product", "") for s in steps_raw if isinstance(s, dict)]
            g_reactants = [list(s.get("reactants") or []) for s in steps_raw if isinstance(s, dict)]
            grounding = check_route_grounding(smiles, g_products, g_reactants)
        except Exception as exc:
            grounding = {"ok": True, "checked": False, "reason": f"grounding check error: {exc}"}

        notes: List[str] = []

        # Each pillar returns an internal `_full` dict (used to compute the
        # aggregate) plus a slim `summary` dict that's the only thing we
        # forward to the LLM. Keeping per-step internals out of context saves
        # tokens and the agent already has the route in its scratchpad.
        #
        # The three pillars are disjoint and their backends are independently
        # thread-safe, so run them concurrently when enabled. Each pillar gets
        # its OWN notes list (a shared list would race under threads), merged
        # back in deterministic pillar order afterwards so the final
        # `note` (notes[-1]) is reproducible regardless of completion timing.
        if getattr(self, "pillar_parallel", True):
            feas_notes: List[str] = []
            cost_notes: List[str] = []
            green_notes: List[str] = []
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="score_pillar") as ex:
                fut_feas = ex.submit(self._score_feasibility, steps_raw, feas_notes)
                fut_cost = ex.submit(self._score_cost, smiles, steps_raw, cost_notes)
                fut_green = ex.submit(self._score_greenness, steps_raw, green_notes)
                feas = fut_feas.result()
                cost = fut_cost.result()
                green = fut_green.result()
            # Deterministic merge: feasibility → cost → greenness.
            notes.extend(feas_notes)
            notes.extend(cost_notes)
            notes.extend(green_notes)
        else:
            feas = self._score_feasibility(steps_raw, notes=notes)
            cost = self._score_cost(smiles, steps_raw, notes=notes)
            green = self._score_greenness(steps_raw, notes=notes)

        total = (
            self.weights["feasibility"] * feas["score"]
            + self.weights["cost"] * cost["score"]
            + self.weights["greenness"] * green["score"]
        )

        # Bottleneck = the lowest-scoring pillar; surfaced as a one-liner the
        # LLM can act on instead of digging through per-step trace.
        pillar_scores = {
            "feasibility": feas["score"],
            "cost": cost["score"],
            "greenness": green["score"],
        }
        worst = min(pillar_scores, key=lambda k: pillar_scores[k])
        worst_summary = {
            "feasibility": feas["summary"],
            "cost": cost["summary"],
            "greenness": green["summary"],
        }[worst]
        worst_warnings = worst_summary.get("warnings") if isinstance(worst_summary, dict) else None
        bottleneck_hint = (
            worst_warnings[0] if isinstance(worst_warnings, list) and worst_warnings
            else f"{worst} score {pillar_scores[worst]:.2f}"
        )

        # Slim, LLM-facing payload: pillar scores as plain numbers, no
        # internal trace (pre_judge_score, judge_cap, per-step rater dicts).
        # Use `load_result` if you need the raw per-step data.
        out = {
            "total_score": round(total, 4),
            "feasibility_score": round(feas["score"], 4),
            "cost_score": round(cost["score"], 4),
            "greenness_score": round(green["score"], 4),
            "bottleneck": f"{worst}: {bottleneck_hint}",
            "n_steps": len(steps_raw),
            # Structural fingerprint of the scored route. save_best_route
            # recomputes this from the route being saved and rejects the
            # save if it disagrees — proving the cited score_route_result_id
            # scored THIS route, not a different higher-scoring one whose id
            # was reused. Must match tools._route_fp.route_fingerprint.
            "route_fp": _route_fp_from_steps(steps_raw),
        }
        # Surface the structural grounding verdict. INFORMATIONAL here
        # (save_best_route enforces it as a hard reject); the agent sees the
        # signal pre-save so it can repair a disconnected / wrong-rooted route
        # before spending a save slot on it.
        if grounding.get("checked"):
            out["route_grounding"] = {
                "ok": bool(grounding.get("ok")),
                "root_matches_target": bool(grounding.get("root_matches_target")),
                "n_dangling": int(grounding.get("n_dangling", 0) or 0),
            }
        # Surface up to 3 actionable warnings (atom imbalance, hazardous solvent,
        # extreme T) so the agent can fix them without a load_result round-trip.
        all_warnings: List[str] = []
        # Grounding failure is the most important warning — emit it first.
        if grounding.get("checked") and not grounding.get("ok"):
            all_warnings.append(
                "ungrounded route: " + str(grounding.get("reason", ""))[:140]
            )
        for pillar_name, pillar_dict in (("feasibility", feas), ("greenness", green)):
            w = pillar_dict.get("summary", {}).get("warnings") or []
            for entry in w:
                if entry not in all_warnings:
                    all_warnings.append(entry)
                if len(all_warnings) >= 3:
                    break
            if len(all_warnings) >= 3:
                break
        if all_warnings:
            out["warnings"] = all_warnings[:3]
        if notes:
            # Keep only the most-recent note (judge invocation, fallback).
            out["note"] = notes[-1][:160]

        # ── Audit-only per-step / per-pillar detail (traceability) ───────────
        # The slim `out` above is all the LLM ever sees (and all that lands in
        # the message stream). The rich per-step breakdown each pillar computed
        # is otherwise discarded; we attach it under the reserved `_detail`
        # key so ResultStore.persist writes it to disk (recoverable via
        # load_result / offline audit) WITHOUT bloating context — persist strips
        # `_detail` from the returned string and the projection allowlist omits
        # it from stubs. This answers "what did each step finally score, by
        # which backend (rater-only vs rater+judge), and what drove cost /
        # greenness?" after the run, which the scalar pillar scores cannot.
        out["_detail"] = {
            "weights": dict(self.weights),
            "feasibility": {
                "score": round(feas["score"], 4),
                # Full pillar summary (pre_judge_score, raw_mean, worst_step,
                # judge block when the whole-route judge fired, and the
                # step_judge_fired/upheld/lifts counters) — audit-only, needed
                # by the 2026-07 judge ablation to compute clean modification
                # rates without re-parsing notes.
                "summary": feas.get("summary", {}),
                # Each entry: rater confidence, optional carbon/ring warnings,
                # optional step_judge block (its presence = rater+judge backend),
                # and final_score (the post-judge value that fed the mean).
                "per_step": feas.get("per_step", []),
            },
            "cost": {
                "score": round(cost["score"], 4),
                "step_efficiency": cost.get("step_efficiency"),
                "leaf_economics": cost.get("leaf_economics"),
                "sc_decrease_score": cost.get("sc_decrease_score"),
                # Per-leaf: smiles, leaf_score, price_per_g, vendor.
                "leaves": cost.get("leaves", []),
            },
            "greenness": {
                "score": round(green["score"], 4),
                # Per-step: atom_economy, solvent{name,class,score},
                # temperature_C, temperature_score, hazardous_reagents.
                "per_step": green.get("per_step", []),
            },
        }
        return json.dumps(out, default=str)

    # ──────────────────────────────────────────────────────────────────────
    # Pillar 1: Feasibility
    # ──────────────────────────────────────────────────────────────────────

    def _score_one_feasibility_step(
        self, i: int, s: Dict[str, Any]
    ) -> tuple:
        """Score a single route step's feasibility — pure, no shared mutation.

        Returns ``(entry, step_score, step_warnings)``. Safe to run
        concurrently across steps: it only touches the (thread-safe) rater
        client, returning everything by value so the caller can reassemble in
        index order. Keeping this side-effect-free is what makes the per-step
        ThreadPoolExecutor fan-out correct.

        Two failure modes are deliberately distinguished:
          - a SMILES RDKit cannot sanitize ⇒ the ROUTE is broken ⇒ the step is
            scored 0 and the rater is skipped (a malformed disconnection is a
            real feasibility signal, not an outage).
          - the rater endpoint is unreachable (``rate()`` returns None) ⇒
            INFRASTRUCTURE is broken ⇒ we RAISE. Feasibility has no fallback
            scorer since ReactionT5 was removed, and silently scoring the step 0
            would let a dead endpoint masquerade as an infeasible route. The
            raise propagates out of the per-step pool and the pillar pool to
            surface as a loud per-case tool-error.
        """
        reactants: List[str] = list(s.get("reactants", []) or [])
        product: str = s.get("product", "")
        step_warnings: List[str] = []
        if not reactants or not product:
            return (
                {"step": i, "error": "missing reactants or product"},
                0.0,
                [f"step {i}: missing reactants or product"],
            )

        # ── Structural gate: every SMILES must sanitize ───────────────────
        # Runs BEFORE the rater so a malformed precursor/product (e.g. a
        # transformer hallucinating pentavalent carbon) is scored 0 instead of
        # being handed to the rater as if it were a real molecule. This is NOT a
        # rater outage — it is a broken route — so we score 0, we do not raise.
        ok, bad_smiles = check_step_smiles(reactants, product)
        if not ok:
            return (
                {"step": i, "error": "invalid SMILES", "invalid_smiles": bad_smiles},
                0.0,
                [f"step {i}: invalid SMILES ({bad_smiles!r}) — step scored 0"],
            )

        stereo = warning_for_step(reactants, product)

        rater_res = self._rater.rate(reactants, product)
        if rater_res is None:
            raise RuntimeError(
                f"step {i}: reaction_rater unavailable "
                f"({'.'.join(reactants)} >> {product}); feasibility cannot be "
                "scored without it (ReactionT5 fallback was removed)."
            )
        rater_conf = float(rater_res["confidence"])
        step_score = max(0.0, min(1.0, rater_conf))

        # Carbon-skeleton fabrication penalty. A POSITIVE carbon_delta means the
        # product carries more skeletal carbons than every precursor combined —
        # a carbon appeared from nowhere (a wrong reagent homolog at an
        # H2O/NH3-only annulation, or a hallucinated skeleton). Dock the rater
        # score by `carbon_fabrication_penalty` per fabricated carbon, capped at
        # the step score. A NEGATIVE delta (a fragment / leaving group departs)
        # is benign and never penalised — this is NOT mass balance.
        c_delta = carbon_delta(reactants, product)
        carbon_penalty = 0.0
        if c_delta > 0 and self.carbon_fabrication_penalty > 0:
            carbon_penalty = min(step_score, c_delta * self.carbon_fabrication_penalty)
            step_score = max(0.0, step_score - carbon_penalty)

        entry: Dict[str, Any] = {
            "step": i,
            "stereo_warning": stereo,
            "rater": {
                "confidence": round(rater_conf, 4),
                "feasible": rater_res["feasible"],
            },
        }
        if carbon_penalty > 0:
            entry["carbon_fabrication_penalty"] = round(carbon_penalty, 4)
            step_warnings.append(
                f"step {i}: fabricated carbon (C: +{c_delta}) "
                f"— docked {round(carbon_penalty, 4)}"
            )

        # Negative-carbon ring-forming warning — INFORMATIONAL ONLY, never
        # alters the score. A *bimolecular* step (>=2 precursors) that forms a
        # new ring yet LOSES skeletal carbon is mechanistically suspect: a
        # 100%-atom-economical cycloaddition (Diels–Alder, [4+2]) must conserve
        # carbon, so a negative delta here usually means a wrong, smaller
        # product was drawn (e.g. C5+C4 written as a C7 adduct) with no leaving
        # group to carry the missing carbon. We do NOT dock the step: a pure
        # structural rule has only ~51% precision against the chemist audit
        # (legitimate bimolecular ring closures — lactamisation, heteroannulation
        # — share the same signature). The warning is surfaced for the LLM judge
        # / human audit instead. See RETRACE_ERROR_ANALYSIS §2 and the
        # negative-delta "informational only" note in carbon_skeleton.py.
        if c_delta < 0 and len(reactants) >= 2 and forms_new_ring(reactants, product):
            entry["ring_carbon_warning"] = {
                "carbon_delta": c_delta,
                "n_precursors": len(reactants),
            }
            step_warnings.append(
                f"step {i}: bimolecular ring-forming step lost {abs(c_delta)} "
                "skeletal carbon(s) with no leaving group — possible wrong "
                "cycloaddition product (informational; score unaffected)"
            )

        # Stereo + a negative carbon_delta are INFORMATIONAL ONLY (not part of
        # the score) — organic reactions are routinely written without full mass
        # balance; only fabricated skeletal carbon (positive delta, above) is
        # penalised.
        if stereo:
            # `warning_for_step` returns a string (or None); historically we
            # treated it as a dict with a "warning" key, which crashed the
            # whole verifier on any step that produced a stereo flag.
            step_warnings.append(f"step {i}: {stereo}")
        return entry, step_score, step_warnings

    def _score_feasibility(self, steps: List[Dict[str, Any]],
                           notes: List[str]) -> Dict[str, Any]:
        # Per-step scoring is independent and its backend is thread-safe —
        # the rater client is a pooled requests.Session — so fan the steps out
        # and reassemble in index order. The LLM judge upgrade pass below stays
        # SERIAL on purpose — fanning judge calls out would reintroduce the
        # shared-tunnel retry herd. A step whose rater endpoint is down raises
        # RuntimeError; fut.result() re-raises it here, which propagates out of
        # the pillar pool to fail the whole score_route call loudly (the rater
        # is required — there is no fallback scorer).
        n = len(steps)
        results: List[Optional[tuple]] = [None] * n
        workers = max(1, min(getattr(self, "feasibility_step_workers", 4), n))
        if workers == 1 or n <= 1:
            for i, s in enumerate(steps):
                results[i] = self._score_one_feasibility_step(i, s)
        else:
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="score_feas_step"
            ) as ex:
                fut_to_i = {
                    ex.submit(self._score_one_feasibility_step, i, s): i
                    for i, s in enumerate(steps)
                }
                for fut in as_completed(fut_to_i):
                    i = fut_to_i[fut]
                    results[i] = fut.result()

        per_step: List[Dict[str, Any]] = []
        per_step_scores: List[float] = []
        warnings: List[str] = []
        for entry, step_score, step_warnings in results:  # type: ignore[misc]
            per_step.append(entry)
            per_step_scores.append(step_score)
            warnings.extend(step_warnings)

        # ── Per-step LLM judge upgrade pass ───────────────────────────────
        # Runs BEFORE the route average so the mean reflects judge-lifted
        # step scores. Narrower than the whole-route fallback below: fires
        # only for steps whose rater score is below the judge trigger — one LLM
        # call per genuinely-low step, not one for the whole route. Reaction-
        # class hint comes from prior ``suggest_templates`` results for this
        # exact disconnection (when present); pure-neural steps proceed without
        # a hint.
        n_step_judge_lifted = 0
        n_step_judge_fired = 0
        n_step_judge_upheld = 0
        if self._judge is not None:
            uphold_mode = bool(getattr(self.judge_cfg, "uphold_gate", False))
            for i, ps in enumerate(per_step):
                if not should_invoke_step_judge(
                    ps, self.judge_cfg, step_score=per_step_scores[i]
                ):
                    continue
                n_step_judge_fired += 1
                hint = self._lookup_reaction_class_hint(steps[i])
                before = per_step_scores[i]
                if uphold_mode:
                    # ── Arm C: uphold-by-default ─────────────────────────
                    try:
                        uv = self._judge.judge_step_uphold(
                            steps[i], before, hint=hint
                        )
                    except Exception as exc:
                        notes.append(f"step {i} judge failed: {exc}")
                        continue
                    lifted = combine_uphold(
                        before, uv, self.judge_cfg.step_judge_cap
                    )
                    upheld = bool(uv.get("uphold", True))
                    if upheld:
                        n_step_judge_upheld += 1
                    ps["step_judge"] = {
                        "mode": "uphold_gate",
                        "uphold": upheld,
                        "revised_score": uv.get("revised_score"),
                        "reason": uv.get("reason", ""),
                        "hint_used": hint is not None,
                        "lifted_from": round(before, 4),
                        "lifted_to": round(lifted, 4),
                    }
                    per_step_scores[i] = lifted
                    if lifted > before:
                        n_step_judge_lifted += 1
                    notes.append(
                        f"step {i} judge(uphold-gate): feas={before:.2f}→{lifted:.2f} "
                        f"(uphold={upheld}, hint={'yes' if hint else 'no'})"
                    )
                    continue
                # ── Legacy arm B: lift-don't-promote max() ───────────────
                try:
                    step_verdict = self._judge.judge_step(steps[i], hint=hint)
                except Exception as exc:
                    notes.append(f"step {i} judge failed: {exc}")
                    continue
                lifted = combine_step(
                    before, step_verdict["score"], self.judge_cfg.step_judge_cap,
                )
                ps["step_judge"] = {
                    "score": round(float(step_verdict["score"]), 4),
                    "verdict": step_verdict.get("verdict", ""),
                    "top_concern": step_verdict.get("top_concern", ""),
                    "hint_used": hint is not None,
                    "lifted_from": round(before, 4),
                    "lifted_to": round(lifted, 4),
                }
                per_step_scores[i] = lifted
                if lifted > before:
                    n_step_judge_lifted += 1
                notes.append(
                    f"step {i} judge: feas={before:.2f}→{lifted:.2f} "
                    f"(verdict={step_verdict.get('verdict','')}, "
                    f"hint={'yes' if hint else 'no'})"
                )

        # Thread each step's FINAL feasibility score (post rater + judge lift +
        # any penalty) back into its per_step entry. Until here the entry only
        # carried the raw rater confidence under `rater`; the authoritative
        # number that fed the route mean lived in the parallel `per_step_scores`
        # list. Copying it in makes the persisted `_detail` self-contained for
        # per-step audit ("what did this exact step finally score, and was it a
        # rater-only or rater+judge call?" — `step_judge` presence answers the
        # latter).
        for i, ps in enumerate(per_step):
            if i < len(per_step_scores):
                ps["final_score"] = round(float(per_step_scores[i]), 4)

        # Mean across (judge-lifted) per-step scores, then a gentle dispersion
        # dock toward the worst surviving step. See the constructor comment on
        # `feasibility_dispersion_lambda` for the rationale and calibration.
        raw_mean = (
            sum(per_step_scores) / len(per_step_scores) if per_step_scores else 0.0
        )
        disp_lambda = getattr(self, "feasibility_dispersion_lambda", 0.0)
        if per_step_scores and disp_lambda > 0:
            worst = min(per_step_scores)
            pre_judge_score = raw_mean - disp_lambda * (raw_mean - worst)
        else:
            pre_judge_score = raw_mean

        # ── LLM judge fallback (whole-route plausibility) ─────────────────
        judge_info: Optional[Dict[str, Any]] = None
        final_score = pre_judge_score
        if self._judge is not None and should_invoke_judge(pre_judge_score, per_step, self.judge_cfg):
            if bool(getattr(self.judge_cfg, "uphold_gate", False)):
                # ── Arm C: uphold-by-default ──────────────────────────────
                try:
                    uv = self._judge.judge_uphold(steps, pre_judge_score)
                    final_score = combine_uphold(
                        pre_judge_score, uv, self.judge_cfg.judge_cap
                    )
                    judge_info = {
                        "mode": "uphold_gate",
                        "uphold": bool(uv.get("uphold", True)),
                        "revised_score": uv.get("revised_score"),
                        "score": (uv.get("revised_score")
                                  if uv.get("revised_score") is not None
                                  else pre_judge_score),
                        "verdict": ("upheld" if uv.get("uphold", True)
                                    else "overridden"),
                        "top_concern": uv.get("reason", ""),
                    }
                    notes.append(
                        f"feasibility judge(uphold-gate) invoked "
                        f"(feas={pre_judge_score:.2f}→{final_score:.2f}, "
                        f"uphold={uv.get('uphold', True)})"
                    )
                except Exception as exc:
                    notes.append(f"feasibility judge failed: {exc}")
            else:
                # ── Legacy arm B: lift-don't-promote max() ────────────────
                try:
                    judge_info = self._judge.judge(steps)
                    final_score = combine_feasibility_and_judge(
                        pre_judge_score, judge_info["score"], self.judge_cfg.judge_cap
                    )
                    notes.append(
                        f"feasibility judge invoked (feas={pre_judge_score:.2f}, "
                        f"judge={judge_info['score']:.2f}→cap={self.judge_cfg.judge_cap}, "
                        f"verdict={judge_info.get('verdict','')})"
                    )
                except Exception as exc:
                    notes.append(f"feasibility judge failed: {exc}")

        # Slim summary the LLM actually gets back. No per-step internals; only
        # the things it needs to act on. `pre_judge_score` is the dispersion-
        # penalised mean before the whole-route judge could lift it; `raw_mean`
        # is the plain per-step mean; `score` is the final.
        summary = {
            "score": round(final_score, 4),
            "n_steps": len(steps),
            "pre_judge_score": round(pre_judge_score, 4),
            "raw_mean": round(raw_mean, 4),
            "warnings": warnings[:8],   # cap so a long bad route can't bloat context
        }
        if disp_lambda > 0 and per_step_scores:
            summary["worst_step"] = round(min(per_step_scores), 4)
        if n_step_judge_fired > 0:
            # Clean modification-rate instrumentation (2026-07 ablation):
            # fired = judge invocations; lifts = score actually changed;
            # upheld (uphold-gate mode only) = judge explicitly kept rater.
            summary["step_judge_fired"] = n_step_judge_fired
        if n_step_judge_upheld > 0:
            summary["step_judge_upheld"] = n_step_judge_upheld
        if n_step_judge_lifted > 0:
            summary["step_judge_lifts"] = n_step_judge_lifted
        if judge_info is not None:
            summary["judge"] = {
                "score": round(judge_info["score"], 4),
                "verdict": judge_info.get("verdict", ""),
                "top_concern": judge_info.get("top_concern", ""),
            }

        return {
            "score": round(final_score, 4),
            "summary": summary,
            "per_step": per_step,        # kept for local debugging / transcripts
        }

    def _lookup_reaction_class_hint(
        self, step: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return template-provenance evidence for a step, or None.

        Walks the agent's prior ``suggest_templates`` results via
        ``result_store.iter_by_tool`` (same precedent as
        ExploreBranchesTool) and returns the first suggestion whose
        canonical product AND reactants match this step. Pure-neural
        steps — where no template lookup was ever performed for this
        product — fall through to None and the caller proceeds without
        a hint.
        """
        rs = self._result_store
        if rs is None:
            return None
        try:
            from tools.builtins.explore_branches import _canon, _canon_set
        except Exception:
            return None
        product_canon = _canon(step.get("product", "") or "")
        if not product_canon:
            return None
        step_reactants = _canon_set(step.get("reactants") or [])
        if step_reactants is None:
            return None
        try:
            iterator = rs.iter_by_tool("suggest_templates")
        except Exception as exc:
            logger.debug("_lookup_reaction_class_hint iter_by_tool failed: %s", exc)
            return None
        for _, payload in iterator:
            if not isinstance(payload, dict):
                continue
            if _canon(payload.get("product", "") or "") != product_canon:
                continue
            for rank_i, s in enumerate(payload.get("suggestions") or []):
                if not isinstance(s, dict):
                    continue
                rxs = s.get("reactants_smiles") or s.get("reactants")
                if isinstance(rxs, str):
                    parts = [x for x in rxs.split(".") if x]
                elif isinstance(rxs, list):
                    parts = list(rxs)
                else:
                    continue
                cset = _canon_set(parts)
                if cset is None or cset != step_reactants:
                    continue
                return {
                    "template_source": s.get("template_source"),
                    "template_kind": s.get("template_kind"),
                    "rank": rank_i + 1,
                }
        return None

    # ──────────────────────────────────────────────────────────────────────
    # Pillar 2: Cost
    # ──────────────────────────────────────────────────────────────────────

    def _score_cost(self, target_smiles: str, steps: List[Dict[str, Any]],
                    notes: List[str]) -> Dict[str, Any]:
        n_steps = len(steps)
        step_efficiency = math.exp(-n_steps / 8.0)

        leaves = _identify_leaves(steps)

        # Leaf economics — ChemPrice primary, BuyabilityChain fallback. Only
        # the starting materials of the route get priced; intermediates are
        # produced in situ.
        if self.pricer is not None:
            price_eval = self.pricer.evaluate_leaves(leaves)
            leaf_economics = price_eval["score"]
            if price_eval["source_summary"] in ("fallback", "mixed"):
                notes.append(
                    f"leaf pricing: {price_eval['source_summary']} (ChemPrice "
                    f"unavailable for some leaves; using local BuyabilityChain 0/1)"
                )
            leaves_out = price_eval["leaves"]
            n_priced = price_eval["n_priced"]
            n_available = price_eval["n_available"]
        else:
            # ChemPrice disabled — pure 0/1 buyability fallback
            buy_eval = self.buyability.evaluate_leaves(leaves)
            leaf_economics = buy_eval["fraction"]
            leaves_out = buy_eval["leaves"]
            n_priced = 0
            n_available = buy_eval["n_buyable"]
            notes.append("leaf pricing: chemprice disabled; using 0/1 buyability")

        # SCScore decrease (Coley primary, vendored SAScore fallback).
        try:
            sa_dec = sa_decrease_score(target_smiles, leaves)
        except Exception as exc:
            logger.warning(
                "score_route: sa_decrease_score unavailable for %s: %s "
                "(falling back to neutral 0.5)",
                target_smiles, exc,
            )
            sa_dec = 0.5
            notes.append(f"sa_decrease fallback: {exc}")

        # Cost = mean of three terms (each in [0, 1])
        score = (step_efficiency + leaf_economics + sa_dec) / 3.0

        # Diagnostics
        try:
            sc_t: Optional[float] = sc_score(target_smiles)
        except Exception:
            sc_t = None
        sc_leaves = []
        for s in leaves:
            try:
                sc_leaves.append(sc_score(s))
            except Exception:
                continue
        sc_leaves_mean: Optional[float] = (
            sum(sc_leaves) / len(sc_leaves) if sc_leaves else None
        )

        return {
            "score": round(score, 4),
            "n_steps": n_steps,
            "step_efficiency": round(step_efficiency, 4),
            "leaf_economics": round(leaf_economics, 4),
            "leaves": leaves_out,
            "n_priced": n_priced,
            "n_available": n_available,
            "scscore_target": round(sc_t, 3) if sc_t is not None else None,
            "scscore_leaves_mean": round(sc_leaves_mean, 3) if sc_leaves_mean is not None else None,
            "sc_decrease_score": round(sa_dec, 4),
            "summary": {
                "score": round(score, 4),
                "n_steps": n_steps,
                "n_leaves": len(leaves),
                "n_available": n_available,
                "n_priced": n_priced,
                "step_efficiency": round(step_efficiency, 4),
                "leaf_economics": round(leaf_economics, 4),
                "sc_decrease": round(sa_dec, 4),
                # Brief leaves view: only smiles + score + price (no vendor strings).
                "leaves": [
                    {
                        "smiles": lf["smiles"],
                        "leaf_score": lf["leaf_score"],
                        "price_per_g": lf.get("price_per_g"),
                    }
                    for lf in leaves_out
                ],
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Pillar 3: Greenness
    # ──────────────────────────────────────────────────────────────────────

    def _score_greenness(self, steps: List[Dict[str, Any]],
                         notes: List[str]) -> Dict[str, Any]:
        per_step: List[Dict[str, Any]] = []
        per_step_scores: List[float] = []

        # Optional components. They are reported in the summary rather than
        # silently treated as perfect scores.
        degraded: List[str] = []
        if self.condition is None:
            reason = getattr(self, "condition_unavailable", None) or "not configured"
            degraded.append("condition_prediction")
            notes.append(
                f"greenness degraded: condition model unavailable ({reason}); "
                "predicted-temperature sub-score skipped and affected steps score 0"
            )
        if classify_solvent is None:
            degraded.append("chem21_solvent_table")
            notes.append(
                "greenness degraded: CHEM21 solvent table not installed "
                "(see LICENSES.md); solvent class is reported as 'unavailable'"
            )

        for i, s in enumerate(steps):
            reactants: List[str] = list(s.get("reactants", []) or [])
            product: str = s.get("product", "")
            if not reactants or not product:
                per_step.append({"step": i, "error": "missing reactants or product"})
                per_step_scores.append(0.0)
                continue

            ae = atom_economy(reactants, product)

            rxn_smi = _step_to_rxn_smiles(reactants, product)
            # Condition model is optional: when it is absent the
            # step is reported and scored 0 instead of crashing the pillar. A
            # failure raised by an installed model keeps propagating unchanged.
            if self.condition is None:
                per_step.append({"step": i, "atom_economy": round(ae, 4),
                                 "error": "condition prediction unavailable: "
                                          "optional ASKCOS condition model not installed"})
                per_step_scores.append(0.0)
                notes.append(f"step {i}: condition prediction unavailable "
                             "(optional ASKCOS condition model not installed)")
                continue
            cond = self.condition.predict_top1(rxn_smi)

            if classify_solvent is not None:
                solv_class, solv_score = classify_solvent(cond.get("solvent", "") or "")
            else:
                solv_class, solv_score = "unavailable", None
            # `.get("temperature", 25.0)` still returns None when the key is
            # present with a null value, and `float(None)` / temperature_score
            # would crash the whole greenness pillar. Coerce defensively.
            T_raw = cond.get("temperature", 25.0)
            try:
                T = float(T_raw) if T_raw is not None else 25.0
            except (TypeError, ValueError):
                T = 25.0
            T_score = temperature_score(T)

            # Hazardous reagents — scan reagent + catalyst (both are SMILES-ish)
            haz_inputs = [
                cond.get("reagent", "") or "",
                cond.get("catalyst", "") or "",
            ]
            r_score, r_matches = reagent_score(haz_inputs)

            # Step greenness = mean of the sub-scores that are available.
            if solv_score is None:
                components = [ae, T_score, r_score]
            else:
                components = [ae, solv_score, T_score, r_score]
            step_score = sum(components) / float(len(components))
            per_step_scores.append(step_score)
            per_step.append({
                "step": i,
                "atom_economy": round(ae, 4),
                "solvent": {
                    "name": cond.get("solvent", "") or "unknown",
                    "class": solv_class,
                    "score": solv_score,
                },
                "temperature_C": float(T),
                "temperature_score": round(T_score, 4),
                "hazardous_reagents": r_matches,
                "reagent_score": round(r_score, 4),
            })

        score = sum(per_step_scores) / len(per_step_scores) if per_step_scores else 0.0

        # Aggregate the noisy per-step structures into a slim summary the
        # LLM gets back. Per-step entries are kept on the orchestrator dict
        # for transcript / debugging only.
        ae_vals = [s["atom_economy"] for s in per_step if "atom_economy" in s]
        T_vals = [s["temperature_C"] for s in per_step if "temperature_C" in s]
        solv_classes: Dict[str, int] = {}
        haz_seen: List[str] = []
        green_warnings: List[str] = []
        for s in per_step:
            cls = (s.get("solvent") or {}).get("class")
            if cls:
                solv_classes[cls] = solv_classes.get(cls, 0) + 1
            for hz in s.get("hazardous_reagents", []) or []:
                tag = hz.get("name") or hz.get("smiles") or ""
                if tag and tag not in haz_seen:
                    haz_seen.append(tag)
            # Per-step actionable warnings: red solvent, high T, hazardous reagent
            i = s.get("step")
            if cls == "red":
                green_warnings.append(
                    f"step {i}: hazardous solvent '{(s.get('solvent') or {}).get('name','?')}'"
                )
            T = s.get("temperature_C")
            if T is not None and (T > 150 or T < -40):
                green_warnings.append(f"step {i}: extreme temperature {T:.0f}°C")

        summary = {
            "score": round(score, 4),
            "n_steps": len(steps),
            "atom_economy_mean": round(sum(ae_vals) / len(ae_vals), 4) if ae_vals else None,
            "temperature_C_mean": round(sum(T_vals) / len(T_vals), 1) if T_vals else None,
            "solvent_class_counts": solv_classes,
            "hazardous_reagents": haz_seen[:6],
            "warnings": green_warnings[:6],
            "degraded_components": degraded,
        }
        return {"score": round(score, 4), "summary": summary, "per_step": per_step}
