#!/usr/bin/env python3
"""Emit per-CASE chemist-audit reports into each RETRACE result folder.

Granularity is case-level (one file per target idx per condition), and within
each file every top-10 route is broken out with its per-STEP deterministic
chemistry facts, so the data can be hand-corrected later without re-running the
audit.

Two sources are joined (both verified 1:1, 3304↔3304, no gaps):
  - grounding export  /tmp/audit_routes_{cond}.json  — per-step reaction_name,
    reactants, product, carbon_delta, ring_delta, ring sizes (deterministic
    RDKit facts; objective and re-checkable).
  - merged verdicts   final_results/_reports/RETRACE_CHEMIST_AUDIT_records.json
    — route-level verdict + worst_defect + n_fake/n_suspect_steps + total_score
    + chemist `summary` + skeptic `verify_reason` + verified/fake_upheld flags.

NOTE on per-step verdicts: the workflow output kept only the route-level
verdict and the FAKE/SUSPECT step COUNTS (n_fake_steps / n_suspect_steps); the
structured per-step VALID/SUSPECT/FAKE labels the audit agent produced were not
persisted past the workflow boundary. The chemist `summary` names which step is
broken and why in prose. Each route therefore carries:
  - objective per-step facts (always present, re-checkable),
  - route-level verdict + counts + reasons (from the audit),
  - an empty `correction` block to fill in when revising a verdict.

Output: final_results/{Retro190_GPT55-<cond>_RETRACE}/chemist_audit/
          case-{idx:03d}.json        — one target, all its top-10 routes
          _INDEX.json                — case→verdict-summary roll-up
          README.md                  — schema + how to correct
"""
import json
import os
from collections import defaultdict

REPO = os.environ.get("RETRACE_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
RECORDS = f"{REPO}/final_results/_reports/RETRACE_CHEMIST_AUDIT_records.json"
GROUNDING = {
    "standard": "/tmp/audit_routes_standard.json",
    "fast": "/tmp/audit_routes_fast.json",
}
DEST = {
    "standard": f"{REPO}/final_results/Retro190_GPT55-standard_RETRACE/chemist_audit",
    "fast": f"{REPO}/final_results/Retro190_GPT55-fast_RETRACE/chemist_audit",
}


def load():
    verd = {x["uid"]: x for x in json.load(open(RECORDS))}
    routes = {}
    for cond, path in GROUNDING.items():
        for r in json.load(open(path)):
            routes[f"{cond}:{r['idx']}:r{r['rank']}"] = r
    # hard fail if the two sources disagree — we do not want a silently partial export
    gk, vk = set(routes), set(verd)
    assert not (vk - gk), f"verdict uids missing grounding: {sorted(vk - gk)[:5]}"
    assert not (gk - vk), f"grounding uids missing verdict: {sorted(gk - vk)[:5]}"
    return verd, routes


def step_record(st):
    """Objective, re-checkable per-step facts from the grounding export."""
    rd = st.get("ring_delta") or {}
    return {
        "i": st.get("i"),
        "reaction_name": st.get("reaction_name"),
        "reactants": st.get("reactants"),
        "product": st.get("product"),
        "parse_ok": st.get("parse_ok"),
        "carbon_delta": st.get("carbon_delta"),
        "ring_delta": {"gained": rd.get("gained") or [], "lost": rd.get("lost") or []},
        "ring_sizes_reactants": st.get("ring_sizes_reactants"),
        "ring_sizes_product": st.get("ring_sizes_product"),
    }


def route_record(uid, gr, vr):
    g = gr.get("grounding") or {}
    return {
        "uid": uid,
        "rank": gr["rank"],
        "total_score": vr.get("total_score"),
        "sub_scores": gr.get("sub_scores"),
        # --- audit verdict (correctable) ---
        "route_verdict": vr.get("route_verdict"),
        "worst_defect_code": vr.get("worst_defect_code"),
        "n_fake_steps": vr.get("n_fake_steps"),
        "n_suspect_steps": vr.get("n_suspect_steps"),
        "adversarially_verified": bool(vr.get("verified")),
        "fake_upheld": vr.get("fake_upheld"),
        "verify_failed": bool(vr.get("verify_failed")),
        "chemist_summary": vr.get("summary"),
        "skeptic_verify_reason": vr.get("verify_reason"),
        # --- deterministic per-step facts ---
        "num_steps": gr.get("num_steps"),
        "starting_materials": gr.get("starting_materials"),
        "route_grounding": {
            "target_produced": g.get("target_produced"),
            "any_unparseable": g.get("any_unparseable"),
            "n_dangling_intermediates": g.get("n_dangling_intermediates"),
            "dangling_intermediates": g.get("dangling_intermediates"),
        },
        "steps": [step_record(s) for s in (gr.get("steps") or [])],
        # --- left blank for hand-correction ---
        "correction": {
            "corrected_route_verdict": None,
            "corrected_step_verdicts": [],  # e.g. [{"i": 2, "verdict": "VALID", "note": "..."}]
            "corrected_by": None,
            "corrected_at": None,
            "note": None,
        },
    }


def main():
    verd, routes = load()
    # cond -> idx -> [route uids]
    by = defaultdict(lambda: defaultdict(list))
    for uid in routes:
        cond, idx, _ = uid.split(":")
        by[cond][int(idx)].append(uid)

    grand_index = {}
    for cond, idxmap in by.items():
        outdir = DEST[cond]
        os.makedirs(outdir, exist_ok=True)
        index = []
        for idx in sorted(idxmap):
            uids = sorted(idxmap[idx], key=lambda u: routes[u]["rank"])
            target = routes[uids[0]].get("target_smiles")
            rroutes = [route_record(u, routes[u], verd[u]) for u in uids]
            verds = [r["route_verdict"] for r in rroutes]
            case = {
                "idx": idx,
                "condition": cond,
                "target_smiles": target,
                "n_routes": len(rroutes),
                "solved": "VALID" in verds,            # ≥1 VALID route in top-10
                "rank1_verdict": rroutes[0]["route_verdict"],
                "verdict_counts": {
                    v: verds.count(v) for v in ("VALID", "SUSPECT", "FAKE", "ERROR")
                },
                "routes": rroutes,
            }
            path = f"{outdir}/case-{idx:03d}.json"
            json.dump(case, open(path, "w"), indent=1, ensure_ascii=False)
            index.append({
                "idx": idx,
                "file": f"case-{idx:03d}.json",
                "target_smiles": target,
                "solved": case["solved"],
                "rank1_verdict": case["rank1_verdict"],
                "verdict_counts": case["verdict_counts"],
            })
        json.dump({"condition": cond, "n_cases": len(index), "cases": index},
                  open(f"{outdir}/_INDEX.json", "w"), indent=1, ensure_ascii=False)
        open(f"{outdir}/README.md", "w").write(README.format(cond=cond, n=len(index)))
        grand_index[cond] = len(index)
        print(f"[{cond}] wrote {len(index)} case files -> {outdir}")
    print("done:", grand_index)


README = """# Per-case chemist audit — {cond} condition ({n} cases)

One JSON file per target: `case-NNN.json` (NNN = retro-190 idx, zero-padded).
`_INDEX.json` is the roll-up (idx → solved? / rank-1 verdict / verdict counts).

## Why this granularity
Each case file holds **all top-10 routes** for that target, and each route holds
its **per-step deterministic chemistry facts** (reaction name, reactants,
product, carbon_delta, ring_delta, ring sizes — all RDKit-derived and
re-checkable) alongside the **route-level audit verdict** and the **chemist /
skeptic reasoning**. This lets a verdict be revised later without re-running the
1000-agent audit workflow.

## Route record schema
- `route_verdict` — VALID / SUSPECT / FAKE (FAKE = ≥1 mechanistically impossible
  step; SUSPECT = dubious but none impossible; VALID = a chemist would accept).
- `worst_defect_code`, `n_fake_steps`, `n_suspect_steps` — from the audit.
- `adversarially_verified` / `fake_upheld` — every FAKE was screened by an
  independent skeptic told to refute it; `fake_upheld=false` means it was
  demoted. `verify_failed=true` means the skeptic hit a transient API error and
  the FAKE rests on the first auditor alone.
- `chemist_summary` / `skeptic_verify_reason` — the prose reasoning; the summary
  names which step is broken and why.
- `steps[]` — objective per-step facts (NOT a per-step verdict; the structured
  per-step labels were not persisted past the workflow boundary — see the
  summary prose for which step failed).
- `correction` — **left blank on purpose**. To revise a verdict, fill in
  `corrected_route_verdict` (and optionally `corrected_step_verdicts`,
  `corrected_by`, `corrected_at`, `note`). Downstream stats should prefer the
  corrected value when present.

## Provenance
Generated by `scripts/_audit_emit_case_reports.py` from the audit workflow
output (`RETRACE_CHEMIST_AUDIT_records.json`) joined with the grounding export.
See `final_results/_reports/RETRACE_CHEMIST_AUDIT.md` (aggregate) and
`RETRACE_ERROR_ANALYSIS.md` (failure-mode analysis).
"""


if __name__ == "__main__":
    main()
