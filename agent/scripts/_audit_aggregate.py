#!/usr/bin/env python3
"""Aggregate the 4 quarter-workflow audit outputs into a top-1/3/5/10 quantified
audit report for both RETRACE conditions.

Input: the 4 workflow .output files (JSON arrays of per-route verdict records),
passed as argv, OR auto-discovered. Each record:
  {uid, cond, idx, rank, total_score, route_verdict (VALID|SUSPECT|FAKE|ERROR),
   worst_defect_code, n_fake_steps, n_suspect_steps, summary, verified, ...}

Two complementary top-K metrics per condition:
  (A) idx-level "solved@K": fraction of the 190 idx that have >=1 VALID route
      within their top-K ranks. This is the retrosynthesis success metric — the
      agent surfaces rank-1, but a chemist would accept the best VALID of the
      top-K, so solved@K measures "is a real route reachable in the top K".
  (B) route-level "valid-rate@K": of all top-K routes, the fraction that are
      VALID. Measures how much of the ranked list is trustworthy.

Also: defect taxonomy histogram, FAKE-rate by rank, score-vs-validity (are FAKE
routes scoring as high as VALID ones — i.e. is the verifier fooled?), and the
standard-vs-fast comparison at each K.

Writes final_results/_reports/RETRACE_CHEMIST_AUDIT.md + a machine-readable
RETRACE_CHEMIST_AUDIT.json.
"""
import json, os, sys, glob, statistics
from collections import defaultdict, Counter

REPO = os.environ.get("RETRACE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
OUT_MD = f"{REPO}/final_results/_reports/RETRACE_CHEMIST_AUDIT.md"
OUT_JSON = f"{REPO}/final_results/_reports/RETRACE_CHEMIST_AUDIT.json"
KS = [1, 3, 5, 10]


def load_outputs(paths):
    recs = []
    for p in paths:
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f"[warn] could not load {p}: {e}", file=sys.stderr)
            continue
        # workflow .output may wrap result in {result: [...]} or be the array
        if isinstance(data, dict) and "result" in data:
            data = data["result"]
        if isinstance(data, list):
            recs.extend(data)
    return recs


def dedupe(recs):
    """Keep one record per uid (prefer verified/non-ERROR)."""
    by_uid = {}
    for r in recs:
        uid = r.get("uid")
        if not uid:
            continue
        prev = by_uid.get(uid)
        if prev is None:
            by_uid[uid] = r
            continue
        # prefer non-ERROR, then verified
        def rank(x):
            return (x.get("route_verdict") != "ERROR", bool(x.get("verified")))
        if rank(r) > rank(prev):
            by_uid[uid] = r
    return by_uid


def main():
    paths = sys.argv[1:] or glob.glob("/tmp/audit_out_*.json")
    recs = load_outputs(paths)
    by_uid = dedupe(recs)
    print(f"[loaded] {len(recs)} raw records -> {len(by_uid)} unique uids")

    # The workflow output records don't carry total_score; join it back from
    # the payloads file by uid so the "is the verifier fooled?" score-vs-validity
    # section has the in-loop verifier score for each route.
    PAYLOADS = "/tmp/audit_payloads.json"
    try:
        score_by_uid = {p["uid"]: p.get("total_score")
                        for p in json.load(open(PAYLOADS))}
        n_joined = 0
        for uid, r in by_uid.items():
            if r.get("total_score") is None and uid in score_by_uid:
                r["total_score"] = score_by_uid[uid]
                n_joined += 1
        print(f"[joined] total_score onto {n_joined} records from {PAYLOADS}")
    except Exception as e:
        print(f"[warn] could not join total_score from {PAYLOADS}: {e}",
              file=sys.stderr)

    # organize: cond -> idx -> {rank: rec}
    data = defaultdict(lambda: defaultdict(dict))
    for uid, r in by_uid.items():
        data[r["cond"]][r["idx"]][r["rank"]] = r

    report = {"conditions": {}, "comparison": {}, "n_uids": len(by_uid)}
    L = []
    L.append("# Retro-190 RETRACE chemist-grade audit — top-1/3/5/10\n")
    L.append("Every top-10 route of both conditions audited per-step by an "
             "expert-chemist agent (VALID/SUSPECT/FAKE with defect code), with "
             "deterministic RDKit grounding (target-produced, carbon balance, ring "
             "sizes, dangling intermediates) pre-computed, and **every FAKE verdict "
             "adversarially re-verified** by an independent skeptic agent that tries "
             "to refute it. A route is FAKE if any step is mechanistically "
             "impossible; SUSPECT if any step is dubious but none impossible; VALID "
             "otherwise.\n")

    for cond in ["standard", "fast"]:
        idxs = sorted(data[cond])
        n_idx = len(idxs)
        cond_rec = {"n_idx": n_idx, "topk": {}}
        # route-level verdict counts overall
        all_recs = [data[cond][i][rk] for i in idxs for rk in data[cond][i]]
        vc = Counter(r["route_verdict"] for r in all_recs)
        # defect taxonomy (route worst_defect)
        defects = Counter(r.get("worst_defect_code", "NONE") for r in all_recs
                          if r["route_verdict"] == "FAKE")
        # FAKE rate by rank
        fake_by_rank = defaultdict(lambda: [0, 0])  # rank -> [fake, total]
        for r in all_recs:
            fake_by_rank[r["rank"]][1] += 1
            if r["route_verdict"] == "FAKE":
                fake_by_rank[r["rank"]][0] += 1
        # score vs validity
        valid_scores = [r["total_score"] for r in all_recs if r["route_verdict"] == "VALID" and r.get("total_score") is not None]
        fake_scores = [r["total_score"] for r in all_recs if r["route_verdict"] == "FAKE" and r.get("total_score") is not None]

        # top-K metrics
        for K in KS:
            solved = 0          # idx with >=1 VALID in top-K
            solved_or_suspect = 0
            rk_total = 0
            rk_valid = 0
            rank1_valid = 0
            for i in idxs:
                ranks = data[cond][i]
                topk = [ranks[rk] for rk in sorted(ranks) if rk <= K]
                if not topk:
                    continue
                verds = [t["route_verdict"] for t in topk]
                if "VALID" in verds:
                    solved += 1
                if "VALID" in verds or "SUSPECT" in verds:
                    solved_or_suspect += 1
                rk_total += len(topk)
                rk_valid += sum(1 for v in verds if v == "VALID")
            cond_rec["topk"][K] = {
                "solved_at_k": solved,
                "solved_rate": solved / n_idx if n_idx else 0,
                "solved_or_suspect": solved_or_suspect,
                "solved_or_suspect_rate": solved_or_suspect / n_idx if n_idx else 0,
                "route_valid_rate": rk_valid / rk_total if rk_total else 0,
                "n_routes_in_topk": rk_total,
            }
        cond_rec["verdict_counts"] = dict(vc)
        cond_rec["fake_defect_taxonomy"] = dict(defects)
        cond_rec["fake_by_rank"] = {rk: {"fake": v[0], "total": v[1]} for rk, v in sorted(fake_by_rank.items())}
        cond_rec["score_valid_mean"] = statistics.mean(valid_scores) if valid_scores else None
        cond_rec["score_fake_mean"] = statistics.mean(fake_scores) if fake_scores else None
        report["conditions"][cond] = cond_rec

    # ---- render markdown ----
    def pct(x):
        return f"{100*x:.1f}%"

    L.append("## (A) idx-level solved@K — fraction of 190 targets with ≥1 VALID route in top-K\n")
    L.append("This is the headline retrosynthesis-success metric: a chemist would "
             "accept the best scientifically-valid route among the top-K candidates.\n")
    L.append("| K | standard solved@K | fast solved@K | standard (incl SUSPECT) | fast (incl SUSPECT) |")
    L.append("|---|-------------------|---------------|-------------------------|---------------------|")
    for K in KS:
        s = report["conditions"]["standard"]["topk"][K]
        f = report["conditions"]["fast"]["topk"][K]
        L.append(f"| {K} | {s['solved_at_k']}/{report['conditions']['standard']['n_idx']} ({pct(s['solved_rate'])}) "
                 f"| {f['solved_at_k']}/{report['conditions']['fast']['n_idx']} ({pct(f['solved_rate'])}) "
                 f"| {pct(s['solved_or_suspect_rate'])} | {pct(f['solved_or_suspect_rate'])} |")
    L.append("")
    L.append("## (B) route-level valid-rate@K — of all top-K routes, fraction that are VALID\n")
    L.append("| K | standard valid-rate | fast valid-rate | standard n-routes | fast n-routes |")
    L.append("|---|---------------------|-----------------|-------------------|---------------|")
    for K in KS:
        s = report["conditions"]["standard"]["topk"][K]
        f = report["conditions"]["fast"]["topk"][K]
        L.append(f"| {K} | {pct(s['route_valid_rate'])} | {pct(f['route_valid_rate'])} "
                 f"| {s['n_routes_in_topk']} | {f['n_routes_in_topk']} |")
    L.append("")
    L.append("## Route-verdict distribution (all audited top-10 routes)\n")
    L.append("| condition | VALID | SUSPECT | FAKE | ERROR |")
    L.append("|-----------|-------|---------|------|-------|")
    for cond in ["standard", "fast"]:
        vc = report["conditions"][cond]["verdict_counts"]
        L.append(f"| {cond} | {vc.get('VALID',0)} | {vc.get('SUSPECT',0)} | {vc.get('FAKE',0)} | {vc.get('ERROR',0)} |")
    L.append("")
    L.append("## FAKE-rate by rank (does quality decay down the ranked list?)\n")
    L.append("| rank | standard FAKE/total | fast FAKE/total |")
    L.append("|------|---------------------|-----------------|")
    for rk in range(1, 11):
        s = report["conditions"]["standard"]["fake_by_rank"].get(rk, {"fake": 0, "total": 0})
        f = report["conditions"]["fast"]["fake_by_rank"].get(rk, {"fake": 0, "total": 0})
        sp = pct(s["fake"]/s["total"]) if s["total"] else "—"
        fp = pct(f["fake"]/f["total"]) if f["total"] else "—"
        L.append(f"| {rk} | {s['fake']}/{s['total']} ({sp}) | {f['fake']}/{f['total']} ({fp}) |")
    L.append("")
    L.append("## Is the verifier fooled? — mean total_score of VALID vs FAKE routes\n")
    L.append("If FAKE routes score as high as VALID ones, the in-loop verifier "
             "cannot distinguish them (the central finding of the feasibility-judge "
             "audit).\n")
    L.append("| condition | VALID mean score | FAKE mean score | Δ |")
    L.append("|-----------|------------------|-----------------|---|")
    for cond in ["standard", "fast"]:
        c = report["conditions"][cond]
        vm, fm = c["score_valid_mean"], c["score_fake_mean"]
        d = (vm - fm) if (vm is not None and fm is not None) else None
        L.append(f"| {cond} | {vm:.4f} | {fm:.4f} | {d:+.4f} |" if d is not None
                 else f"| {cond} | {vm} | {fm} | — |")
    L.append("")
    L.append("## FAKE-route defect taxonomy\n")
    L.append("| defect | standard | fast |")
    L.append("|--------|----------|------|")
    all_defects = set(report["conditions"]["standard"]["fake_defect_taxonomy"]) | set(report["conditions"]["fast"]["fake_defect_taxonomy"])
    for d in sorted(all_defects):
        s = report["conditions"]["standard"]["fake_defect_taxonomy"].get(d, 0)
        f = report["conditions"]["fast"]["fake_defect_taxonomy"].get(d, 0)
        L.append(f"| {d} | {s} | {f} |")
    L.append("")

    # comparison verdict
    s1 = report["conditions"]["standard"]["topk"][1]
    f1 = report["conditions"]["fast"]["topk"][1]
    report["comparison"]["solved_at_1_delta"] = s1["solved_rate"] - f1["solved_rate"]
    L.append("## Headline\n")
    L.append(f"- **Top-1 (rank-1) real-route rate**: standard {pct(s1['solved_rate'])} "
             f"vs fast {pct(f1['solved_rate'])} (Δ {100*(s1['solved_rate']-f1['solved_rate']):+.1f} pts).")
    s10 = report["conditions"]["standard"]["topk"][10]
    f10 = report["conditions"]["fast"]["topk"][10]
    L.append(f"- **Top-10 solved rate** (≥1 real route in top-10): standard "
             f"{pct(s10['solved_rate'])} vs fast {pct(f10['solved_rate'])}.")
    L.append("")

    open(OUT_MD, "w").write("\n".join(L) + "\n")
    json.dump(report, open(OUT_JSON, "w"), indent=2)
    print("\n".join(L))
    print(f"\n[written] {OUT_MD}\n[written] {OUT_JSON}")


if __name__ == "__main__":
    main()
