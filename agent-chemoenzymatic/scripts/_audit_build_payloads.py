#!/usr/bin/env python3
"""Flatten the two grounded route exports into ONE compact list of audit
payloads the Workflow can pass inline via `args`. Each payload is the minimal
self-contained unit an agent needs to audit ONE route:

  { uid, cond, idx, rank, total_score, target,
    steps: [ "<reaction_name>: <product>  <<  <r1>.<r2>   [Cdelta=+N, ringΔ gained=[..] lost=[..]]" ],
    flags: { target_produced, any_unparseable, n_dangling, dangling } }

uid = f"{cond}:{idx}:r{rank}" is the stable key for aggregation.
Writes /tmp/audit_payloads.json (a single flat list, both conditions).
"""
import json
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s
    except Exception:
        return s


def fmt_step(s):
    prod = canon(s["product"])
    reac = ".".join(canon(x) for x in s["reactants"])
    cd = s["carbon_delta"]
    rd = s["ring_delta"] or {"gained": [], "lost": []}
    tags = []
    if cd is not None and cd != 0:
        tags.append(f"Cdelta={cd:+d}")
    if rd["gained"] or rd["lost"]:
        tags.append(f"ringΔ gained={rd['gained']} lost={rd['lost']}")
    tagstr = ("   [" + ", ".join(tags) + "]") if tags else ""
    name = s["reaction_name"] or "(unnamed)"
    return f"{name}: {prod}  <<  {reac}{tagstr}"


def main():
    payloads = []
    for cond in ["standard", "fast"]:
        recs = json.load(open(f"/tmp/audit_routes_{cond}.json"))
        for r in recs:
            g = r["grounding"]
            payloads.append({
                "uid": f"{cond}:{r['idx']}:r{r['rank']}",
                "cond": cond,
                "idx": r["idx"],
                "rank": r["rank"],
                "total_score": r["total_score"],
                "target": canon(r["target_smiles"]),
                "steps": [fmt_step(s) for s in r["steps"]],
                "flags": {
                    "target_produced": g["target_produced"],
                    "any_unparseable": g["any_unparseable"],
                    "n_dangling": g["n_dangling_intermediates"],
                    "dangling": g["dangling_intermediates"],
                },
            })
    json.dump(payloads, open("/tmp/audit_payloads.json", "w"))
    print(f"{len(payloads)} payloads -> /tmp/audit_payloads.json")
    print(f"  standard: {sum(1 for p in payloads if p['cond']=='standard')}")
    print(f"  fast:     {sum(1 for p in payloads if p['cond']=='fast')}")
    # shard summary for workflow planning
    import statistics
    sizes = [len(json.dumps(p)) for p in payloads]
    print(f"  payload size median={statistics.median(sizes):.0f}B max={max(sizes)}B")


if __name__ == "__main__":
    main()
