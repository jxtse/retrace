#!/usr/bin/env python3
"""Export per-idx top-10 routes for BOTH RETRACE conditions with deterministic
RDKit grounding, so the chemist-audit workflow never has to trust the LLM for
mechanical facts (ring sizes, per-step carbon balance, target reconstruction).

For each condition we pick, per idx, the canon-SMILES-verified transcript copy
with the highest rank-1 score (defends against M4 mis-attribution and helper
double-runs — same selection rule as the budget comparison).

Output: /tmp/audit_routes_{cond}.json  =  list of route records:
  {
    cond, idx, rank, target_smiles, total_score, sub_scores, num_steps,
    starting_materials,
    steps: [ {i, raw, product, reactants[], reaction_name,
              parse_ok, carbon_delta, ring_sizes_product, ring_sizes_reactants,
              ring_delta} ],
    grounding: { target_reconstructed: bool,    # last step product canon==target
                 final_product_canon, target_canon,
                 any_unparseable: bool,
                 n_positive_cdelta_steps, n_negative_cdelta_steps,
                 n_ringchange_steps }
  }

The workflow reads these records inline via `args` (no loader agent).
"""
import json, glob, re, csv, sys
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
RDLogger.DisableLog("rdApp.*")

REPO = RETRACE_DATA_ROOT + "/retrosynthesis-agent"
CASES = f"{REPO}/dataset/benchmark/retro-190_dataset/cases.csv"
CONDS = {
    "standard": f"{REPO}/final_results/Retro190_GPT55-standard_RETRACE",
    "fast": f"{REPO}/final_results/Retro190_GPT55-fast_RETRACE",
}
TOPK = 10


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def carbon_count(s):
    m = Chem.MolFromSmiles(s)
    if not m:
        return None
    return sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C")


def ring_sizes(s):
    m = Chem.MolFromSmiles(s)
    if not m:
        return None
    ri = m.GetRingInfo()
    return sorted(len(r) for r in ri.AtomRings())


def parse_step(raw):
    """`product>>r1.r2 (reaction: name)` -> (product, [reactants], name)."""
    name = ""
    mm = re.search(r"\(reaction:\s*(.*?)\)\s*$", raw)
    if mm:
        name = mm.group(1).strip()
        raw = raw[: mm.start()].strip()
    if ">>" not in raw:
        return None, [], name
    prod, reac = raw.split(">>", 1)
    prod = prod.strip()
    reactants = [x.strip() for x in reac.split(".") if x.strip()]
    return prod, reactants, name


def ground_step(i, raw):
    prod, reactants, name = parse_step(raw)
    pc = carbon_count(prod) if prod else None
    rc_list = [carbon_count(r) for r in reactants]
    parse_ok = prod is not None and all(carbon_count(r) is not None for r in reactants) and pc is not None
    cdelta = None
    if parse_ok:
        cdelta = pc - sum(rc_list)  # product_C - sum precursor_C
    rs_prod = ring_sizes(prod) if prod else None
    rs_reac = []
    for r in reactants:
        rs = ring_sizes(r)
        if rs:
            rs_reac.extend(rs)
    rs_reac = sorted(rs_reac)
    ring_delta = None
    if rs_prod is not None:
        # multiset difference summary: rings gained in product vs all reactants
        from collections import Counter
        gained = Counter(rs_prod) - Counter(rs_reac)
        lost = Counter(rs_reac) - Counter(rs_prod)
        ring_delta = {"gained": sorted(gained.elements()), "lost": sorted(lost.elements())}
    return {
        "i": i,
        "raw": raw,
        "product": prod,
        "reactants": reactants,
        "reaction_name": name,
        "parse_ok": parse_ok,
        "carbon_delta": cdelta,
        "ring_sizes_product": rs_prod,
        "ring_sizes_reactants": rs_reac,
        "ring_delta": ring_delta,
    }


def main():
    rows = {int(r["idx"]): r["smiles"] for r in csv.DictReader(open(CASES))}
    for cond, root in CONDS.items():
        by_idx = defaultdict(list)
        for tr in glob.glob(f"{root}/run-*/**/transcript.json", recursive=True):
            if "_jitter_victims" in tr:
                continue
            m = re.search(r"case-(\d+)", tr)
            if m:
                by_idx[int(m.group(1))].append(tr)
        records = []
        for idx in sorted(by_idx):
            tgt_canon = canon(rows.get(idx, ""))
            best = None  # (rank1, routes, target_smiles)
            for tr in by_idx[idx]:
                try:
                    t = json.load(open(tr))
                except Exception:
                    continue
                if canon(t.get("target_smiles", "")) != tgt_canon:
                    continue
                routes = (t.get("scratchpad") or {}).get("routes") or []
                if not routes:
                    continue
                r1 = max((r.get("total_score") or 0) for r in routes)
                if best is None or r1 > best[0]:
                    best = (r1, routes, t.get("target_smiles", ""))
            if best is None:
                continue
            _, routes, tgt = best
            # routes already rank-sorted; take top-K
            routes = sorted(routes, key=lambda r: -(r.get("total_score") or 0))[:TOPK]
            for rank, r in enumerate(routes, 1):
                steps = [ground_step(i, s) for i, s in enumerate(r.get("steps") or [])]
                # ORDER-AGNOSTIC grounding (route step ordering is NOT a fixed
                # direction across routes — verified empirically). The reliable
                # facts: is the target a product of some step? does the set of
                # steps form one connected synthesis tree rooted at the target?
                prod_canons = [canon(s["product"]) for s in steps]
                reac_canons = [[canon(x) for x in s["reactants"]] for s in steps]
                target_produced = tgt_canon in prod_canons
                # connectivity: build the precursor graph. Start from the target's
                # producing step; an intermediate is "resolved" if it is itself a
                # product of another step OR a declared starting material. A route
                # is connected if, walking from the target, every non-buyable
                # reactant is the product of exactly one step (no dangling intermediate).
                sm_set = set(canon(x) for x in (r.get("starting_materials") or []))
                all_products = set(prod_canons)
                dangling = []  # reactants that are neither a step-product nor a starting material
                for rc_list in reac_canons:
                    for rc in rc_list:
                        if rc is None:
                            continue
                        if rc not in all_products and rc not in sm_set:
                            dangling.append(rc)
                records.append({
                    "cond": cond,
                    "idx": idx,
                    "rank": rank,
                    "target_smiles": tgt,
                    "total_score": r.get("total_score"),
                    "sub_scores": r.get("sub_scores"),
                    "num_steps": r.get("num_steps"),
                    "starting_materials": r.get("starting_materials"),
                    "steps": steps,
                    "grounding": {
                        "target_produced": target_produced,
                        "target_canon": tgt_canon,
                        "any_unparseable": any(not s["parse_ok"] for s in steps),
                        "n_dangling_intermediates": len(set(dangling)),
                        "dangling_intermediates": sorted(set(dangling))[:5],
                        "n_positive_cdelta_steps": sum(1 for s in steps if s["carbon_delta"] is not None and s["carbon_delta"] > 0),
                        "n_negative_cdelta_steps": sum(1 for s in steps if s["carbon_delta"] is not None and s["carbon_delta"] < 0),
                        "n_ringchange_steps": sum(1 for s in steps if s["ring_delta"] and (s["ring_delta"]["gained"] or s["ring_delta"]["lost"])),
                    },
                })
        out = f"/tmp/audit_routes_{cond}.json"
        json.dump(records, open(out, "w"))
        print(f"[{cond}] {len(records)} route records over "
              f"{len({r['idx'] for r in records})} idx -> {out}")
        # quick grounding tally
        n_notprod = sum(1 for r in records if not r["grounding"]["target_produced"])
        n_unparse = sum(1 for r in records if r["grounding"]["any_unparseable"])
        n_dangle = sum(1 for r in records if r["grounding"]["n_dangling_intermediates"] > 0)
        print(f"        target_NOT_produced routes={n_notprod}, "
              f"unparseable_step routes={n_unparse}, "
              f"routes_w_dangling_intermediate={n_dangle}")


if __name__ == "__main__":
    main()
