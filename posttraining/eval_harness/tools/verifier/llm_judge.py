"""Whole-route LLM-as-a-judge fallback for the feasibility pillar.

Why this exists
~~~~~~~~~~~~~~~
The discriminative ``reaction_rater`` is fast and covers the enzymatic /
cascade space, but it is not perfect: a chemically sound disconnection can
score low when the rater is unsure of an unusual but valid step. With
``feasibility`` weighted at 0.70, a single low rater score can sink an
otherwise valid route below the agent's completion gate and waste a search.

The judge takes the **whole route** in one shot (not per-step) and asks the
same LLM that drives the main agent to rate plausibility on a 0–1 scale.
The output is combined conservatively with the rater score so the judge can
*lift* a route the model thinks is fine but cannot *promote* a chemistry
disaster.

Trigger policy (set in :func:`should_invoke_judge`):

* feasibility (rater mean) below ``trigger_feasibility`` (default 0.4).

Combine policy::

    final = max(feasibility_score, min(judge_score, judge_cap))

Where ``judge_cap`` defaults to 0.7 — the judge alone cannot push a route
above 0.7 even if it claims 1.0. Real rater evidence is needed for a top score.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class JudgeConfig:
    enabled: bool = True
    base_url: Optional[str] = None        # default: same as main LLM
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 16000
    trigger_feasibility: float = 0.4      # invoke if rater-mean feasibility < this
    judge_cap: float = 0.7                # ceiling the whole-route judge alone can lift to
    step_judge_cap: float = 0.85          # ceiling the per-step judge alone can lift one step to
    timeout_s: float = 120.0
    # Pinned tier — by design, the judge cost stays predictable regardless
    # of the main agent's reasoning_effort. Override only via the
    # ``verifier.feasibility.judge.reasoning_effort`` config key.
    reasoning_effort: str = "medium"
    # Uphold-by-default gate (2026-07 ablation, arm C). When True the judge
    # is asked to FIRST decide whether the rater's low score is reasonable
    # (uphold -> score untouched) and only outputs a revised score when it
    # deems the rater an obvious false negative. When False (default) the
    # legacy lift-don't-promote max() combine applies. Config key:
    # ``verifier.feasibility.judge.uphold_gate``.
    uphold_gate: bool = False


# ──────────────────────────────────────────────────────────────────────────
# Trigger
# ──────────────────────────────────────────────────────────────────────────


def should_invoke_judge(
    feasibility: float,
    per_step: List[Dict[str, Any]],
    cfg: JudgeConfig,
) -> bool:
    if not cfg.enabled:
        return False
    return feasibility < cfg.trigger_feasibility


def should_invoke_step_judge(
    step: Dict[str, Any],
    cfg: JudgeConfig,
    step_score: Optional[float] = None,
) -> bool:
    """Per-step trigger: fire only on a genuinely low rater step score.

    Every step is now scored by the discriminative rater, so ``step_score``
    is always supplied by the caller; fire the judge when it falls below
    ``trigger_feasibility``. Confident-rater steps don't burn an LLM call.
    """
    if not cfg.enabled:
        return False
    if step_score is None:
        return False
    return step_score < cfg.trigger_feasibility


# ──────────────────────────────────────────────────────────────────────────
# Prompt + parser
# ──────────────────────────────────────────────────────────────────────────


def _build_prompt(steps: List[Dict[str, Any]]) -> str:
    """Compact, model-agnostic prompt. Whole route → one JSON verdict."""
    step_lines = []
    for i, s in enumerate(steps, start=1):
        reactants = ".".join(s.get("reactants", []) or [])
        product = s.get("product", "") or ""
        step_lines.append(f"{i}. {reactants} >> {product}")
    route_block = "\n".join(step_lines) if step_lines else "(empty route)"

    return (
        "You are an expert organic / chemoenzymatic synthesis reviewer. "
        "Rate the plausibility of the following retrosynthetic route as a "
        "whole. Be skeptical of mass-imbalance, missing leaving groups, "
        "wrong oxidation state, impossible regio/stereo selectivity, and "
        "non-existent transformations. Be charitable about minor solvent / "
        "reagent omissions and tautomer / charge-state mismatches.\n\n"
        "Route:\n"
        f"{route_block}\n\n"
        "Output STRICT JSON, single line, no markdown fences, exactly:\n"
        '{"score": <float in [0,1]>, "verdict": "<plausible|partially_plausible|implausible>", '
        '"top_concern": "<one short sentence or empty string>"}\n'
        "Anchors: 1.0 = textbook plausible end-to-end. 0.7 = mostly fine "
        "with a single fixable issue. 0.4 = serious doubts but not absurd. "
        "0.1 = clearly wrong chemistry."
    )


def _build_step_prompt(
    step: Dict[str, Any], hint: Optional[Dict[str, Any]] = None
) -> str:
    """Per-step variant of ``_build_prompt``. Optional ``hint`` carries
    reaction-class evidence from the agent's prior ``suggest_templates``
    calls so the judge can prior over the disconnection type.
    """
    reactants = ".".join(step.get("reactants", []) or [])
    product = step.get("product", "") or ""
    hint_block = ""
    if hint:
        # Evidence-style hint — does NOT claim the template is ground
        # truth, only that the agent saw this exact disconnection in a
        # template-library hit during planning.
        hint_block = (
            "\nReaction-class evidence from this run's `suggest_templates` "
            "(LLM had matched this exact disconnection to a curated template):\n"
            f"  template_source={hint.get('template_source', '?')}, "
            f"template_kind={hint.get('template_kind', '?')}, "
            f"rank={hint.get('rank', '?')}\n"
        )
    return (
        "You are an expert organic / chemoenzymatic synthesis reviewer. "
        "Rate the plausibility of THIS SINGLE retrosynthetic step. "
        "Be skeptical of mass-imbalance, missing leaving groups, wrong "
        "oxidation state, impossible regio/stereo selectivity, and "
        "non-existent transformations. Be charitable about minor solvent / "
        "reagent omissions and tautomer / charge-state mismatches.\n\n"
        f"Step: {reactants} >> {product}\n"
        f"{hint_block}\n"
        "Output STRICT JSON, single line, no markdown fences, exactly:\n"
        '{"score": <float in [0,1]>, "verdict": "<plausible|partially_plausible|implausible>", '
        '"top_concern": "<one short sentence or empty string>"}\n'
        "Anchors: 1.0 = textbook plausible. 0.7 = mostly fine with a single "
        "fixable issue. 0.4 = serious doubts but not absurd. 0.1 = clearly "
        "wrong chemistry."
    )


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _build_step_prompt_uphold(
    step: Dict[str, Any],
    rater_score: float,
    hint: Optional[Dict[str, Any]] = None,
) -> str:
    """Uphold-by-default per-step prompt (ablation arm C).

    Differs from ``_build_step_prompt`` in decision structure, not tone: the
    judge is shown the rater's low score and asked FIRST whether that verdict
    is defensible. Only an OBVIOUS false negative — chemistry a trained
    chemist would recognise as sound — justifies overriding. The default
    action is to uphold.
    """
    reactants = ".".join(step.get("reactants", []) or [])
    product = step.get("product", "") or ""
    hint_block = ""
    if hint:
        hint_block = (
            "\nReaction-class evidence from this run's `suggest_templates` "
            "(LLM had matched this exact disconnection to a curated template):\n"
            f"  template_source={hint.get('template_source', '?')}, "
            f"template_kind={hint.get('template_kind', '?')}, "
            f"rank={hint.get('rank', '?')}\n"
        )
    return (
        "You are an expert organic / chemoenzymatic synthesis reviewer "
        "auditing a fast discriminative reaction-feasibility model. The "
        "model flagged THIS SINGLE retrosynthetic step as low-feasibility "
        f"(score {rater_score:.2f} on [0,1]).\n\n"
        f"Step: {reactants} >> {product}\n"
        f"{hint_block}\n"
        "Your job is NOT to re-score the step from scratch. Decide whether "
        "the model's low score is a DEFENSIBLE verdict:\n"
        "- UPHOLD (default) if the step is questionable, unusual enough that "
        "doubt is reasonable, or you are uncertain. A low score for a "
        "genuinely debatable step is correct behaviour, not an error.\n"
        "- OVERRIDE only if this is an OBVIOUS false negative: a standard, "
        "well-precedented transformation (name reaction, textbook FGI, "
        "routine protection/deprotection, well-known enzymatic step) that "
        "any trained chemist would accept without hesitation.\n\n"
        "Be skeptical of mass-imbalance, missing leaving groups, wrong "
        "oxidation state, impossible regio/stereo selectivity, and "
        "non-existent transformations. Be charitable about minor solvent / "
        "reagent omissions and tautomer / charge-state mismatches.\n\n"
        "Output STRICT JSON, single line, no markdown fences, exactly:\n"
        '{"uphold": <true|false>, "revised_score": <float in [0,1] or null '
        'when uphold is true>, "reason": "<one short sentence>"}\n'
        "Anchors for revised_score (only when overriding): 0.85 = clearly "
        "sound, well-precedented. 0.6 = sound but with a fixable concern. "
        "Never override to a score below the model's own."
    )


def _build_prompt_uphold(steps: List[Dict[str, Any]], route_score: float) -> str:
    """Whole-route uphold-by-default prompt (ablation arm C analogue of
    ``_build_prompt``). Same decision structure as the per-step variant:
    uphold the low route-level feasibility unless it is an obvious false
    negative end-to-end."""
    step_lines = []
    for i, s in enumerate(steps, start=1):
        reactants = ".".join(s.get("reactants", []) or [])
        product = s.get("product", "") or ""
        step_lines.append(f"{i}. {reactants} >> {product}")
    route_block = "\n".join(step_lines) if step_lines else "(empty route)"

    return (
        "You are an expert organic / chemoenzymatic synthesis reviewer "
        "auditing a fast discriminative reaction-feasibility model. The "
        "model scored the following retrosynthetic route as low-feasibility "
        f"overall (route score {route_score:.2f} on [0,1]).\n\n"
        "Route:\n"
        f"{route_block}\n\n"
        "Your job is NOT to re-score the route from scratch. Decide whether "
        "the model's low score is a DEFENSIBLE verdict:\n"
        "- UPHOLD (default) if any step is questionable, the route is "
        "unusual enough that doubt is reasonable, or you are uncertain.\n"
        "- OVERRIDE only if the route is an OBVIOUS false negative: every "
        "step is a standard, well-precedented transformation that any "
        "trained chemist would accept without hesitation.\n\n"
        "Be skeptical of mass-imbalance, missing leaving groups, wrong "
        "oxidation state, impossible regio/stereo selectivity, and "
        "non-existent transformations. Be charitable about minor solvent / "
        "reagent omissions and tautomer / charge-state mismatches.\n\n"
        "Output STRICT JSON, single line, no markdown fences, exactly:\n"
        '{"uphold": <true|false>, "revised_score": <float in [0,1] or null '
        'when uphold is true>, "reason": "<one short sentence>"}\n'
        "Anchors for revised_score (only when overriding): 0.7 = clearly "
        "sound end-to-end. 0.5 = sound but with one fixable concern. Never "
        "override to a score below the model's own."
    )


def _parse_uphold_reply(text: str) -> Dict[str, Any]:
    """Parse the uphold-gate JSON verdict. Fail-closed: anything unparseable
    counts as uphold (score untouched) — the safety net must not lift on a
    malformed reply."""
    if not text:
        return {"uphold": True, "revised_score": None,
                "reason": "empty reply (fail-closed uphold)"}
    for blob in _JSON_RE.findall(text):
        try:
            obj = json.loads(blob)
            if "uphold" not in obj:
                continue
            uphold = bool(obj.get("uphold"))
            rs = obj.get("revised_score")
            revised: Optional[float] = None
            if not uphold and rs is not None:
                try:
                    revised = max(0.0, min(1.0, float(rs)))
                except (TypeError, ValueError):
                    # override without a usable score -> fail-closed uphold
                    uphold, revised = True, None
            if not uphold and revised is None:
                uphold = True
            return {
                "uphold": uphold,
                "revised_score": revised,
                "reason": str(obj.get("reason", ""))[:240],
            }
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return {"uphold": True, "revised_score": None,
            "reason": "unparseable reply (fail-closed uphold)"}


def _parse_judge_reply(text: str) -> Dict[str, Any]:
    if not text:
        return {"score": 0.0, "verdict": "implausible", "top_concern": "empty reply"}
    # Greedy {…} match first; fall back to regex scan if the model wrapped
    # the JSON in prose.
    candidates = _JSON_RE.findall(text)
    for blob in candidates:
        try:
            obj = json.loads(blob)
            if "score" in obj:
                score = float(obj.get("score", 0.0))
                return {
                    "score": max(0.0, min(1.0, score)),
                    "verdict": str(obj.get("verdict", "")) or "unknown",
                    "top_concern": str(obj.get("top_concern", ""))[:240],
                }
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return {"score": 0.0, "verdict": "implausible", "top_concern": "unparseable reply"}


# ──────────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────────


class WholeRouteJudge:
    """Lightweight LLM judge. Uses an LLMClient if injected, else builds one
    on demand from ``JudgeConfig``."""

    def __init__(self, cfg: JudgeConfig, llm_client: Any = None):
        self.cfg = cfg
        self._llm = llm_client
        # Per-process verdict cache. The judge runs at temperature=0.0, so an
        # identical prompt is a pure function of its input — `score_route` is
        # called ~15×/case on overlapping route variants that share sub-steps,
        # and `judge_step` re-poses the SAME `reactants >> product` prompt for
        # every shared step. Keying on the full prompt string captures the
        # whole-route text, the per-step text, AND the reaction-class hint
        # block verbatim, so a hit returns a byte-identical verdict to a live
        # call (score-preserving — does NOT change any route's total_score).
        # Scope is one process = one benchmark case subprocess, so it adds no
        # cross-case state and vanishes on exit; nothing touches NFS.
        self._verdict_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        # Lazy build so verifier import doesn't drag the LLM client unless
        # the judge actually fires.
        from core.llm_client import DEFAULT_BASE_URL, DEFAULT_MODEL
        from core.substrate_client import make_llm_client
        self._llm = make_llm_client(
            base_url=self.cfg.base_url or DEFAULT_BASE_URL,
            model=self.cfg.model or DEFAULT_MODEL,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            read_timeout=self.cfg.timeout_s,
            reasoning_effort=self.cfg.reasoning_effort,
        )
        return self._llm

    def judge(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._call(_build_prompt(steps))

    def judge_uphold(
        self, steps: List[Dict[str, Any]], route_score: float
    ) -> Dict[str, Any]:
        """Whole-route uphold-by-default verdict (ablation arm C). Same
        response shape and fail-closed semantics as ``judge_step_uphold``."""
        prompt = _build_prompt_uphold(steps, route_score)
        key = hashlib.sha256(("uphold:" + prompt).encode("utf-8")).hexdigest()
        with self._cache_lock:
            hit = self._verdict_cache.get(key)
            if hit is not None:
                self._cache_hits += 1
                return dict(hit)
            self._cache_misses += 1

        raw = self._call_uncached_raw(prompt)
        if raw is None:
            return {"uphold": True, "revised_score": None,
                    "reason": "judge call failed (fail-closed uphold)",
                    "transient_error": True}
        verdict = _parse_uphold_reply(raw)
        with self._cache_lock:
            self._verdict_cache[key] = dict(verdict)
        return verdict

    def judge_step(
        self, step: Dict[str, Any], hint: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Per-step plausibility verdict. Same response shape as ``judge``.

        ``hint`` (if given) injects a one-line reaction-class evidence
        block sourced from a matching ``suggest_templates`` suggestion
        for this step's product+reactants.
        """
        return self._call(_build_step_prompt(step, hint))

    def judge_step_uphold(
        self,
        step: Dict[str, Any],
        rater_score: float,
        hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Uphold-by-default per-step verdict (ablation arm C).

        Returns ``{"uphold": bool, "revised_score": float|None, "reason": str}``.
        Fail-closed: LLM errors and unparseable replies uphold the rater
        (score untouched) — the opposite failure default from ``judge_step``
        would silently lift on garbage.
        """
        prompt = _build_step_prompt_uphold(step, rater_score, hint)
        key = hashlib.sha256(("uphold:" + prompt).encode("utf-8")).hexdigest()
        with self._cache_lock:
            hit = self._verdict_cache.get(key)
            if hit is not None:
                self._cache_hits += 1
                return dict(hit)
            self._cache_misses += 1

        raw = self._call_uncached_raw(prompt)
        if raw is None:
            # transient LLM failure -> fail-closed uphold, NOT cached (retryable)
            return {"uphold": True, "revised_score": None,
                    "reason": "judge call failed (fail-closed uphold)",
                    "transient_error": True}
        verdict = _parse_uphold_reply(raw)
        with self._cache_lock:
            self._verdict_cache[key] = dict(verdict)
        return verdict

    def _call(self, prompt: str) -> Dict[str, Any]:
        """Shared LLM call + parse path used by both ``judge`` and
        ``judge_step``. Extracted so the chat_with_override / parse
        branches don't drift between the two callers.

        Verdicts are cached per-process keyed on the prompt (temperature=0.0
        → deterministic), so repeated scoring of routes that share sub-steps
        only pays the LLM once. A cache hit is byte-identical to a live call,
        so the cache never changes a score — it only removes a redundant
        tunnel round-trip. Error verdicts are NOT cached, so a transient
        judge failure can still succeed on a later re-score.
        """
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self._cache_lock:
            hit = self._verdict_cache.get(key)
            if hit is not None:
                self._cache_hits += 1
                return dict(hit)
            self._cache_misses += 1

        verdict = self._call_uncached(prompt)

        # Only cache real verdicts. An error verdict (verdict == "error") is a
        # transient failure (timeout / tunnel reset); caching it would pin the
        # failure for every later re-score of the same step. Let it retry.
        if verdict.get("verdict") != "error":
            with self._cache_lock:
                self._verdict_cache[key] = dict(verdict)
        return verdict

    def _call_uncached(self, prompt: str) -> Dict[str, Any]:
        raw = self._call_uncached_raw(prompt)
        if raw is None:
            return {
                "score": 0.0,
                "verdict": "error",
                "top_concern": "judge call failed",
            }
        return _parse_judge_reply(raw)

    def _call_uncached_raw(self, prompt: str) -> Optional[str]:
        """Raw LLM round-trip: returns the reply text, or None on transport
        failure. Shared by the legacy score path (``_call_uncached``) and the
        uphold path (``judge_step_uphold``) so the client handling / override
        plumbing can't drift between them."""
        messages = [
            {"role": "system",
             "content": "You return STRICT JSON only. No prose, no markdown."},
            {"role": "user", "content": prompt},
        ]
        try:
            llm = self._get_llm()
            # Use chat_with_override when available so the per-call temperature
            # is forced to 0 without mutating the client.
            if hasattr(llm, "chat_with_override"):
                resp = llm.chat_with_override(
                    messages,
                    tools=None,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    reasoning_effort=self.cfg.reasoning_effort,
                )
            else:
                resp = llm.chat(messages, tools=None)
            choices = resp.get("choices") or []
            content = ""
            for ch in choices:
                m = ch.get("message", {}) or {}
                if m.get("content"):
                    content = m["content"]
                    break
            return content
        except Exception as exc:
            logger.warning("LLM judge failed: %s", exc)
            return None


def combine_feasibility_and_judge(
    feasibility_score: float,
    judge_score: float,
    judge_cap: float,
) -> float:
    """``max(feasibility, min(judge, cap))``. Judge can lift but not promote a disaster."""
    capped = min(float(judge_score), float(judge_cap))
    return max(float(feasibility_score), capped)


def combine_step(
    feasibility_step_score: float,
    judge_step_score: float,
    step_judge_cap: float,
) -> float:
    """Per-step combine. Same lift-don't-promote rule as
    ``combine_feasibility_and_judge`` but applied to one step in isolation —
    ``step_judge_cap`` typically sits higher than the whole-route
    ``judge_cap`` because a single-step judgement is narrower (and so
    lower-variance) than a whole-route one.
    """
    capped = min(float(judge_step_score), float(step_judge_cap))
    return max(float(feasibility_step_score), capped)


def combine_uphold(
    rater_score: float,
    verdict: Dict[str, Any],
    cap: float,
) -> float:
    """Uphold-gate combine (ablation arm C), shared by the per-step and
    whole-route paths. ``uphold=True`` -> the rater score is returned
    UNTOUCHED (the judge explicitly declined to modify). ``uphold=False`` ->
    the same lift-don't-promote bound as the legacy path applies to the
    judge's revised score: ``max(rater, min(revised, cap))`` — the override
    can lift a suspected false negative but never demote below the rater and
    never promote past the cap without rater evidence.
    """
    if verdict.get("uphold", True):
        return float(rater_score)
    revised = verdict.get("revised_score")
    if revised is None:
        return float(rater_score)
    return max(float(rater_score), min(float(revised), float(cap)))
