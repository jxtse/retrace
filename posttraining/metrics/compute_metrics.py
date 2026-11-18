#!/usr/bin/env python3
"""
ReTRACE post-training metrics: four arms x 40 held-out Retro*-190 targets.

Read-only over:
  snapshots   : <root>/snapshots/<idx>/{result.json,transcript.json}
  frozen audit: <audit>/campaign-<arm>/campaign.json (+ schema-recovery layers)
Writes only to OUT.

No model calls, no network. stdlib only.
"""

import json, os, glob, math, random, csv, collections, statistics as st


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
SNAP = os.environ.get("RETRACE_SNAP", "/path/to/archive/S3-EndToEnd")
AUDIT = RETRACE_OUTPUTS + "/stage3-endtoend/test40-chemistry-v2"
OUT = RETRACE_OUTPUTS + "/posttrain-metrics"

ARMS = collections.OrderedDict([
    ("base",       ("Test40-Baseline-Medium-Snapshot-v1",   "base")),
    ("s2v2",       ("Test40-S2v2-Medium-Snapshot-v1",       "s2v2")),
    ("s3",         ("Test40-Final-Snapshot-v2",             "s3")),
    ("teachersft", ("Test40-TeacherSFT-Medium-Snapshot-v1", "teachersft")),
])

# Recovered / normalized record sets (verdict-level recovery only; route generation
# and stage-1 deep reads are unchanged, per the recovery POLICY.json files).
RECOVERED = {
    "base":       f"{AUDIT}/schema-recovery-v1/recovered-base.json",
    "s3":         f"{AUDIT}/schema-recovery-v1/recovered-s3.json",
    "s2v2":       f"{AUDIT}/schema-recovery-stage2/recovered-s2v2.json",
    "teachersft": None,   # no recovery layer -> campaign.json as-is (lower bound)
}

FAMILY = {
    "FABRICATED_OR_MISSING_SKELETAL_CARBON": "skeletal_carbon_fabricated_or_missing",
    "WRONG_REGIO_OR_CONSTITUTIONAL":         "wrong_regio_or_constitutional",
    "WRONG_RING_SIZE":                       "wrong_ring_size",
    "IMPOSSIBLE_OR_NONSENSE_REACTION":       "impossible_or_nonsense_reaction",
    "WRONG_REAGENT_OR_CONDITIONS":           "wrong_reagent_or_conditions",
    "STEREO_ONLY":                           "stereo_only",
    "TARGET_NOT_PRODUCED":                   "target_not_produced_or_parse",
    "ERROR":                                 "target_not_produced_or_parse",
    "OTHER":                                 "other",
    "CHEMOSELECTIVITY":                      "other",   # pre-normalization code
    "UNSTABLE_INTERMEDIATE":                 "other",   # pre-normalization code
}

B = 10000
SEED = 42


# ---------------------------------------------------------------- audit layer
def load_audit(arm):
    """Return {idx: record} for the arm, preferring the recovered record set."""
    rec = RECOVERED.get(arm)
    records = None
    src = None
    if rec and os.path.exists(rec):
        records = json.load(open(rec))
        src = os.path.relpath(rec, AUDIT)
    else:
        p = f"{AUDIT}/campaign-{arm}/campaign.json"
        records = json.load(open(p))["records"]
        src = os.path.relpath(p, AUDIT)
    out = {}
    for r in records:
        if r.get("rank", 1) != 1:
            continue
        out[int(r["idx"])] = r
    return out, src


def raw_lenient_n(arm, pre_normalization=True):
    """Lenient count straight from campaign.json, i.e. without the schema-recovery
    layer (ERROR kept as fail, pre-normalization defect codes). Documented so the
    difference against the recovered set is explicit rather than silent."""
    p = f"{AUDIT}/campaign-{arm}/campaign.json"
    records = json.load(open(p))["records"]
    n = sum(1 for r in records if r.get("rank", 1) == 1
            and r.get("route_verdict") in ("VALID", "SUSPECT"))
    return n


# ------------------------------------------------------------- snapshot layer
def load_snapshot(root):
    out = {}
    for sd in sorted(glob.glob(os.path.join(root, "snapshots", "*"))):
        rp, tp = os.path.join(sd, "result.json"), os.path.join(sd, "transcript.json")
        if not (os.path.exists(rp) and os.path.exists(tp)):
            continue
        res = json.load(open(rp))
        out[int(res["idx"])] = (res, tp)
    return out


def call_sig(name, args):
    return name + "\x00" + json.dumps(args, sort_keys=True, default=str)


def behaviour(tp):
    """Agentic-behaviour features from one transcript (operational definitions in
    METRICS_DEFINITIONS.md)."""
    d = json.load(open(tp))
    ev = d.get("events", [])
    conf = d.get("config", {}) or {}
    max_iter = (conf.get("agent", {}) or {}).get("max_iterations") or conf.get("max_iterations") or 40

    turns = tool_calls = 0
    by_tool = collections.Counter()
    rejected = 0
    rej_by_tool = collections.Counter()
    vs_attempt = vs_invalid = 0
    empty_turns = 0
    llm_error_turns = 0
    finish = collections.Counter()
    seen_sig = set()
    dup_calls = 0
    rescore_in_place = 0
    repeat_scored_fp = collections.Counter()
    scored_with_warnings = 0
    saved_with_warnings = 0
    score_by_id = {}
    self_complete = 0
    wallclock_exceeded = 0
    reasoning_chars = content_chars = 0
    warnings_total = 0

    for e in ev:
        k = e.get("event")
        if k == "llm_call":
            turns += 1
            content_chars += int(e.get("content_chars") or 0)
            ntc = e.get("n_tool_calls") or 0
            finish[e.get("finish_reason")] += 1
            if e.get("error"):
                llm_error_turns += 1
            if ntc == 0 and int(e.get("content_chars") or 0) == 0:
                empty_turns += 1
        elif k == "reasoning":
            reasoning_chars += len(e.get("content") or "")
        elif k == "tool_call":
            tool_calls += 1
            name = e.get("name")
            by_tool[name] += 1
            s = call_sig(name, e.get("arguments"))
            if s in seen_sig:
                dup_calls += 1
            else:
                seen_sig.add(s)
        elif k == "tool_result":
            name = e.get("name")
            r = e.get("result")
            if isinstance(r, dict) and r.get("error"):
                rejected += 1
                rej_by_tool[name] += 1
            if name == "validate_smiles" and isinstance(r, dict):
                vs_attempt += 1
                if r.get("valid") is False:
                    vs_invalid += 1
            if name == "score_route" and isinstance(r, dict):
                fp = r.get("route_fp")
                if fp:
                    repeat_scored_fp[fp] += 1
                w = r.get("warnings") or []
                if w:
                    scored_with_warnings += 1
                    warnings_total += len(w)
                score_by_id[e.get("tool_call_id")] = r
            if name == "save_best_route" and isinstance(r, dict):
                m = str(r.get("message") or "")
                if m.startswith("Re-scored existing route in place"):
                    rescore_in_place += 1
                sid = (e.get("arguments") or {}).get("score_route_result_id")
                sres = score_by_id.get(sid)
                if sres and (sres.get("warnings") or []):
                    saved_with_warnings += 1
            if name in ("save_best_route",) and isinstance(r, dict) and r.get("status") == "duplicate":
                pass
        elif k == "llm_self_complete_ignored":
            self_complete += 1
        elif k == "wallclock_exceeded":
            wallclock_exceeded += 1

    repeat_fp = sum(v - 1 for v in repeat_scored_fp.values() if v > 1)
    n_iter = (d.get("stats", {}) or {}).get("n_iterations")
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "tool_calls_by_name": dict(by_tool),
        "rejected": rejected,
        "rejected_by_tool": dict(rej_by_tool),
        "smiles_validations": vs_attempt,
        "smiles_invalid": vs_invalid,
        "empty_turns": empty_turns,
        "llm_error_turns": llm_error_turns,
        "finish_reason": dict(finish),
        "duplicate_tool_calls": dup_calls,
        "rescore_in_place": rescore_in_place,
        "repeat_scored_fp": repeat_fp,
        "scored_with_warnings": scored_with_warnings,
        "saved_with_warnings": saved_with_warnings,
        "warning_instances": warnings_total,
        "self_complete_ignored": self_complete,
        "wallclock_exceeded_events": wallclock_exceeded,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "n_iterations": n_iter,
        "max_iterations": max_iter,
        "hit_cap": (n_iter is not None and n_iter >= max_iter),
    }


# ------------------------------------------------------------------- helpers
def mean(xs):
    return sum(xs) / len(xs) if xs else None


def med(xs):
    return st.median(xs) if xs else None


def boot_ci(vals, rng_vals=None):
    """Percentile bootstrap 95% CI over targets. vals: list of 0/1 (or floats)."""
    n = len(vals)
    if n == 0:
        return None
    rnd = random.Random(SEED)
    means = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += vals[rnd.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * B)]
    hi = means[int(0.975 * B) - 1]
    return [round(lo, 4), round(hi, 4)]


def boot_ci_paired(a, b):
    """Bootstrap CI of mean(b)-mean(a) resampling targets in pairs."""
    n = len(a)
    rnd = random.Random(SEED)
    ds = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            i = rnd.randrange(n)
            s += b[i] - a[i]
        ds.append(s / n)
    ds.sort()
    return [round(ds[int(0.025 * B)], 4), round(ds[int(0.975 * B) - 1], 4)]


def mcnemar_exact(a, b):
    """Two-sided exact McNemar (binomial on discordant pairs)."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return 1.0, only_a, only_b
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail), only_a, only_b


def _binom_cdf_le(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI; reported alongside the percentile bootstrap so that
    degenerate arms (40/40, 0/40) still carry a finite interval."""
    if n == 0:
        return None
    lo, hi = 0.0, 1.0
    if k > 0:                      # solve P(X>=k | p_lo) = alpha/2 (increasing in p)
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            if 1 - _binom_cdf_le(k - 1, n, m) < alpha / 2:
                a = m               # need larger p
            else:
                b = m
        lo = (a + b) / 2
    if k < n:                      # solve P(X<=k | p_hi) = alpha/2
        a, b = 0.0, 1.0
        for _ in range(80):
            m = (a + b) / 2
            if _binom_cdf_le(k, n, m) > alpha / 2:
                a = m
            else:
                b = m
        hi = (a + b) / 2
    return [round(lo, 4), round(hi, 4)]



# ---------------------------------------------------------------------- main
def main():
    per_arm = {}
    per_target_rows = []
    audit_src = {}

    for arm, (dirname, audit_key) in ARMS.items():
        root = os.path.join(SNAP, dirname)
        snaps = load_snapshot(root)
        audit, src = load_audit(audit_key)
        audit_src[arm] = src

        rows = []
        for idx in sorted(snaps):
            res, tp = snaps[idx]
            bh = behaviour(tp)
            a = audit.get(idx)
            br = res.get("best_route") or {}
            steps = br.get("steps") or []
            sub = br.get("sub_scores") or {}
            verdict = (a or {}).get("route_verdict", "NO_ROUTE" if not a else None)
            n_fake = (a or {}).get("n_fake_steps")
            n_susp = (a or {}).get("n_suspect_steps")
            audited_steps = (a or {}).get("steps") or []
            row = {
                "arm": arm,
                "idx": idx,
                "success": bool(res.get("success")),
                "iterations": res.get("iterations"),
                "wall_seconds": res.get("wall_seconds"),
                "run_date": (res.get("started_at") or "")[:10],
                "verdict": verdict,
                "lenient": verdict in ("VALID", "SUSPECT"),
                "strict": verdict == "VALID",
                "audit_error": verdict == "ERROR",
                "no_route": verdict == "NO_ROUTE",
                "worst_defect": (a or {}).get("worst_defect"),
                "worst_family": FAMILY.get((a or {}).get("worst_defect") or "", None),
                "n_fake_steps": n_fake,
                "n_suspect_steps": n_susp,
                "n_audited_steps": len(audited_steps),
                "route_len": len(steps),
                "total_score": br.get("total_score"),
                "feasibility": sub.get("feasibility"),
                "cost": sub.get("cost"),
                "greenness": sub.get("greenness"),
                "saved_routes": len(res.get("all_routes") or []),
                "provenance": br.get("provenance"),
            }
            row.update(bh)
            rows.append(row)

        per_arm[arm] = rows

        # ---- aggregates -------------------------------------------------
        def col(k):
            return [r[k] for r in rows]

        def bools(k):
            return [1 if r[k] else 0 for r in rows]

        n = len(rows)
        verdicts = collections.Counter(r["verdict"] for r in rows)
        fams = collections.Counter(r["worst_family"] for r in rows if r["verdict"] not in ("VALID", None))
        fams_all = collections.Counter(r["worst_family"] for r in rows if r["worst_family"])
        audited = [r for r in rows if not r["no_route"]]
        with_route = [r for r in rows if r["route_len"]]
        score_rows = [r for r in rows if r["total_score"] is not None]

        # defect-code distribution across all audited steps
        step_codes = collections.Counter()
        step_fams = collections.Counter()
        fake_steps_per_route = []
        for r in audited:
            a = audit.get(r["idx"]) or {}
            for s in a.get("steps") or []:
                c = s.get("defect_code")
                step_codes[c] += 1
                if s.get("verdict") == "FAKE":
                    step_fams[FAMILY.get(c or "", "other")] += 1
            if a.get("n_fake_steps") is not None:
                fake_steps_per_route.append(a["n_fake_steps"])

        tool_names = sorted({k for r in rows for k in r["tool_calls_by_name"]})
        tool_means = {t: round(mean([r["tool_calls_by_name"].get(t, 0) for r in rows]), 3)
                      for t in tool_names}
        tool_totals = {t: sum(r["tool_calls_by_name"].get(t, 0) for r in rows) for t in tool_names}

        agg = {
            "n_targets": n,
            "audit_source": src,
            "run_date_distribution": dict(collections.Counter(r["run_date"] for r in rows)),
            "audit_resolved": sum(1 for r in rows if r["verdict"] not in ("ERROR", None)),
            "A_outcome": {
                "completion_n": sum(bools("success")),
                "completion_rate": round(mean(bools("success")), 4),
                "completion_ci95": boot_ci(bools("success")),
                "completion_ci95_clopper_pearson": clopper_pearson(sum(bools("success")), n),
                "audit_lenient_n": sum(bools("lenient")),
                "audit_lenient_rate": round(mean(bools("lenient")), 4),
                "audit_lenient_ci95": boot_ci(bools("lenient")),
                "audit_lenient_ci95_clopper_pearson": clopper_pearson(sum(bools("lenient")), n),
                "audit_strict_n": sum(bools("strict")),
                "audit_strict_rate": round(mean(bools("strict")), 4),
                "audit_strict_ci95": boot_ci(bools("strict")),
                "audit_strict_ci95_clopper_pearson": clopper_pearson(sum(bools("strict")), n),
                "audit_error_n": sum(bools("audit_error")),
                "no_route_n": sum(bools("no_route")),
                "route_produced_n": sum(1 for r in rows if r["total_score"] is not None),
                "route_produced_note": "rank-1 route exists and was scored, regardless of whether every leaf is buyable (native success additionally requires completion)",
                "lenient_counts_error_as_fail": True,
                "raw_campaign_lenient_if_no_recovery": raw_lenient_n(audit_key),
            },
            "B_chemistry": {
                "verdict_distribution": dict(verdicts),
                "worst_defect_distribution": dict(collections.Counter(
                    r["worst_defect"] for r in rows if r["worst_defect"])),
                "worst_defect_family_all": dict(fams_all),
                "worst_defect_family_nonvalid": dict(fams),
                "step_defect_codes": dict(step_codes),
                "step_fake_families": dict(step_fams),
                "mean_fake_steps_per_audited_route": round(mean(fake_steps_per_route), 3) if fake_steps_per_route else None,
                "median_fake_steps_per_audited_route": med(fake_steps_per_route),
                "audited_routes": len(fake_steps_per_route),
                "route_len_mean": round(mean([r["route_len"] for r in with_route]), 3) if with_route else None,
                "route_len_median": med([r["route_len"] for r in with_route]),
                "route_len_n": len(with_route),
                "evaluator_score_mean": round(mean([r["total_score"] for r in score_rows]), 4) if score_rows else None,
                "evaluator_score_median": round(med([r["total_score"] for r in score_rows]), 4) if score_rows else None,
                "feasibility_median": round(med([r["feasibility"] for r in score_rows if r["feasibility"] is not None]), 4) if score_rows else None,
                "cost_median": round(med([r["cost"] for r in score_rows if r["cost"] is not None]), 4) if score_rows else None,
                "greenness_median": round(med([r["greenness"] for r in score_rows if r["greenness"] is not None]), 4) if score_rows else None,
                "evaluator_score_n": len(score_rows),
            },
            "C_agentic": {
                "turns_mean": round(mean(col("turns")), 3),
                "turns_median": med(col("turns")),
                "tool_calls_mean": round(mean(col("tool_calls")), 3),
                "tool_calls_total": sum(col("tool_calls")),
                "tool_calls_by_name_mean": tool_means,
                "tool_calls_by_name_total": tool_totals,
                "rejected_mean": round(mean(col("rejected")), 3),
                "rejected_total": sum(col("rejected")),
                "rejected_by_tool_total": dict(collections.Counter(
                    k for r in rows for k, v in r["rejected_by_tool"].items() for _ in range(v))),
                "targets_with_rejection": sum(1 for r in rows if r["rejected"] > 0),
                "smiles_validation_calls": sum(col("smiles_validations")),
                "smiles_invalid": sum(col("smiles_invalid")),
                "invalid_smiles_rate": round(sum(col("smiles_invalid")) / sum(col("smiles_validations")), 4)
                if sum(col("smiles_validations")) else None,
                "duplicate_tool_calls_total": sum(col("duplicate_tool_calls")),
                "duplicate_tool_calls_mean": round(mean(col("duplicate_tool_calls")), 3),
                "targets_with_duplicates": sum(1 for r in rows if r["duplicate_tool_calls"] > 0),
                "empty_turns_total": sum(col("empty_turns")),
                "targets_with_empty_turn": sum(1 for r in rows if r["empty_turns"] > 0),
                "llm_error_turns_total": sum(col("llm_error_turns")),
                "targets_with_llm_error": sum(1 for r in rows if r["llm_error_turns"] > 0),
                "finish_reason_totals": dict(collections.Counter(
                    k for r in rows for k, v in r["finish_reason"].items() for _ in range(v))),
                "scored_with_warnings_total": sum(col("scored_with_warnings")),
                "warning_instances_total": sum(col("warning_instances")),
                "saved_with_warnings_total": sum(col("saved_with_warnings")),
                "targets_with_warnings": sum(1 for r in rows if r["scored_with_warnings"] > 0),
                "rescore_in_place_total": sum(col("rescore_in_place")),
                "repeat_scored_fp_total": sum(col("repeat_scored_fp")),
                "targets_with_rescore": sum(1 for r in rows if r["rescore_in_place"] > 0),
                "self_complete_attempts_total": sum(col("self_complete_ignored")),
                "targets_with_self_complete_attempt": sum(1 for r in rows if r["self_complete_ignored"] > 0),
                "hit_cap_n": sum(bools("hit_cap")),
                "ended_before_cap_n": sum(1 for r in rows if not r["hit_cap"]),
                "end_reason": {
                    "iteration_cap_40": sum(1 for r in rows if r["hit_cap"]),
                    "wallclock_exceeded": sum(1 for r in rows if r["wallclock_exceeded_events"] > 0),
                    "other_early_stop": sum(1 for r in rows
                                            if not r["hit_cap"] and r["wallclock_exceeded_events"] == 0),
                },
                "wallclock_exceeded_targets": sum(1 for r in rows if r["wallclock_exceeded_events"] > 0),
            },
            "D_efficiency": {
                "wall_seconds_mean": round(mean(col("wall_seconds")), 1),
                "wall_seconds_median": round(med(col("wall_seconds")), 1),
                "wall_hours_median": round(med(col("wall_seconds")) / 3600, 3),
                "wall_hours_mean": round(mean(col("wall_seconds")) / 3600, 3),
                "iterations_mean": round(mean(col("iterations")), 3),
                "iterations_median": med(col("iterations")),
                "saved_routes_mean": round(mean(col("saved_routes")), 3),
                "saved_routes_median": med(col("saved_routes")),
                "saved_routes_total": sum(col("saved_routes")),
                "completion_tokens": None,
                "reasoning_tokens": None,
                "tokens_unavailable_reason": "transcripts carry no usage/token fields (no prompt_tokens/completion_tokens/reasoning_tokens keys anywhere); only per-turn content_chars and reasoning character counts are recorded",
                "content_chars_mean": round(mean(col("content_chars")), 1),
                "reasoning_chars_mean": round(mean(col("reasoning_chars")), 1),
                "content_chars_per_turn": round(sum(col("content_chars")) / max(1, sum(col("turns"))), 1),
                "reasoning_chars_per_turn": round(sum(col("reasoning_chars")) / max(1, sum(col("turns"))), 1),
                "est_output_tokens_chars_div4": {
                    "value": round((sum(col("content_chars")) + sum(col("reasoning_chars"))) / 4 / n, 1),
                    "basis": "ESTIMATE ONLY (chars/4 heuristic) - measured tokens are absent from the frozen transcripts; use as a relative proxy across arms, not as an absolute token count",
                },
                "finish_length_truncations": sum(r["finish_reason"].get("length", 0) for r in rows),
                "budget_exhausted_calls": sum(col("llm_error_turns")),
            },
        }
        per_arm[arm] = (rows, agg)

    # ---- paired comparisons vs base ------------------------------------
    base_rows = per_arm["base"][0]
    base_by = {r["idx"]: r for r in base_rows}
    paired = {}
    for arm in ARMS:
        if arm == "base":
            continue
        rows = per_arm[arm][0]
        common = [r["idx"] for r in rows if r["idx"] in base_by]
        a_succ = [1 if base_by[i]["success"] else 0 for i in common]
        b_succ = [1 if next(r for r in rows if r["idx"] == i)["success"] else 0 for i in common]
        a_len = [1 if base_by[i]["lenient"] else 0 for i in common]
        b_len = [1 if next(r for r in rows if r["idx"] == i)["lenient"] else 0 for i in common]
        p_s, ba, bb = mcnemar_exact(a_succ, b_succ)
        p_l, la, lb = mcnemar_exact(a_len, b_len)
        paired[arm] = {
            "n_paired_targets": len(common),
            "completion_delta": round(mean(b_succ) - mean(a_succ), 4),
            "completion_mcnemar_exact_p": p_s,
            "completion_discordant": {"arm_only": bb, "base_only": ba},
            "completion_delta_ci95": boot_ci_paired(a_succ, b_succ),
            "audit_lenient_delta": round(mean(b_len) - mean(a_len), 4),
            "audit_lenient_mcnemar_exact_p": p_l,
            "audit_lenient_discordant": {"arm_only": lb, "base_only": la},
            "audit_lenient_delta_ci95": boot_ci_paired(a_len, b_len),
        }
    per_arm["base"][1]["A_outcome"]["mcnemar_vs_base"] = "reference"

    # ---- write outputs -------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    result = {
        "meta": {
            "dataset": "Retro*-190 held-out, 40-target test slice (same targets for all arms)",
            "arms": list(ARMS),
            "bootstrap": {"B": B, "seed": SEED, "method": "percentile over targets"},
            "audit_spec": "audit@1 two-stage LLM audit (AUDIT_SPEC.md); lenient = VALID or SUSPECT; strict = VALID",
            "audit_sources": audit_src,
            "snapshot_roots": {a: os.path.join(SNAP, d) for a, (d, _) in ARMS.items()},
            "notes": [
                "s2v2 ran with MolPort vendor lookup enabled; base/s3/teachersft had it off.",
                "s3 is the original single run, not best-of.",
                "teachersft audit has 1 unresolved ERROR (idx 182) and is a lower bound.",
                "Token counts are not recorded in the frozen transcripts.",
            ],
        },
        "arms": {a: per_arm[a][1] for a in ARMS},
        "paired_vs_base": paired,
    }
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=False)

    fields = ["arm", "idx", "success", "verdict", "lenient", "strict", "worst_defect",
              "worst_family", "n_fake_steps", "n_suspect_steps", "route_len", "n_audited_steps",
              "total_score", "feasibility", "cost", "greenness", "turns", "tool_calls",
              "rejected", "smiles_validations", "smiles_invalid", "duplicate_tool_calls",
              "empty_turns", "llm_error_turns", "scored_with_warnings", "saved_with_warnings",
              "rescore_in_place", "repeat_scored_fp", "self_complete_ignored",
              "wallclock_exceeded_events", "wall_seconds", "iterations", "hit_cap",
              "run_date",
              "saved_routes", "reasoning_chars", "content_chars", "provenance"]
    with open(f"{OUT}/per_target_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for arm in ARMS:
            for r in per_arm[arm][0]:
                w.writerow({k: (int(v) if isinstance(v, bool) else v) for k, v in r.items() if k in fields})

    print(json.dumps({a: per_arm[a][1]["A_outcome"] for a in ARMS}, indent=1))
    print(json.dumps(paired, indent=1))
    return result


if __name__ == "__main__":
    main()
