#!/usr/bin/env python3
"""Build a blinded 10-target expert-ranking pack for ReTRACE.

Run from the ReTRACE repository root. Reads only existing FDA/JMC reference
routes, complete Agent routes embedded in transcripts, and verifier scores.
It does not alter benchmark outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import re
import statistics
import zipfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw

RDLogger.DisableLog("rdApp.error")
_FP_GENERATOR = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
_FP_CACHE: dict[str, Any] = {}

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "dataset/分子路线提取FDA_JMC_Chematica.zip"
SUITES = {
    "FDA": {
        "base": ROOT / "final_results/FDA_GPT55_RETRACE",
        "ref_member": "分子路线提取/2025FDA/fda_routes.json",
    },
    "JMC": {
        "base": ROOT / "final_results/JMC_GPT55_RETRACE",
        "ref_member": "分子路线提取/2025JMC/jmc_routes.json",
    },
}


def canonical(s: str) -> str:
    m = Chem.MolFromSmiles(s or "")
    return Chem.MolToSmiles(m, isomericSmiles=True) if m else (s or "")


def fp(s: str):
    if s in _FP_CACHE:
        return _FP_CACHE[s]
    m = Chem.MolFromSmiles(s or "")
    value = _FP_GENERATOR.GetFingerprint(m) if m else None
    _FP_CACHE[s] = value
    return value


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def route_features(route: dict[str, Any]) -> dict[str, Any]:
    reactions = []
    reactants, products, all_mols = set(), set(), set()
    for raw in route["steps"]:
        rxn = raw.split(" (reaction:", 1)[0].strip()
        reactions.append(rxn)
        if ">>" not in rxn:
            continue
        lhs, rhs = rxn.split(">>", 1)
        for s in lhs.split("."):
            if s:
                reactants.add(canonical(s)); all_mols.add(canonical(s))
        for s in rhs.split("."):
            if s:
                products.add(canonical(s)); all_mols.add(canonical(s))
    intermediates = products - {canonical(route.get("target_smiles", ""))}
    start = {canonical(s) for s in route.get("starting_materials", []) if s}
    return {"reactions": reactions, "reactants": reactants, "products": products,
            "intermediates": intermediates, "starting": start, "all_mols": all_mols}


def molset_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    af = [fp(x) for x in a]; bf = [fp(x) for x in b]
    af = [x for x in af if x is not None]; bf = [x for x in bf if x is not None]
    if not af or not bf:
        return 0.0
    ab = sum(max(DataStructs.TanimotoSimilarity(x, y) for y in bf) for x in af) / len(af)
    ba = sum(max(DataStructs.TanimotoSimilarity(y, x) for x in af) for y in bf) / len(bf)
    return (ab + ba) / 2


def route_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    fa, fb = a["features"], b["features"]
    return 0.45 * molset_similarity(fa["intermediates"], fb["intermediates"]) + \
           0.35 * molset_similarity(fa["starting"], fb["starting"]) + \
           0.20 * jaccard(set(fa["reactions"]), set(fb["reactions"]))


def extract_balanced_json(text: str, start: int) -> dict[str, Any] | None:
    i = text.find("{", start)
    if i < 0:
        return None
    depth = 0; in_string = False; escape = False
    for j in range(i, len(text)):
        c = text[j]
        if in_string:
            if escape: escape = False
            elif c == "\\": escape = True
            elif c == '"': in_string = False
        else:
            if c == '"': in_string = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try: return json.loads(text[i:j+1])
                    except json.JSONDecodeError: return None
    return None


def extract_ranked_routes(transcript: Path) -> list[dict[str, Any]]:
    if not transcript.exists(): return []
    text = transcript.read_text(errors="ignore")
    positions = [m.end() for m in re.finditer(r"\[tool_call\]\s*rank_routes\([^\n]*\)", text)]
    for pos in reversed(positions):
        search = pos
        while True:
            marker = text.find('{"schema_version": 1, "routes":', search)
            if marker < 0 or marker > pos + 120000: break
            obj = extract_balanced_json(text, marker)
            if obj and isinstance(obj.get("routes"), list): return obj["routes"]
            search = marker + 1
    return []


def load_refs() -> dict[str, list[dict[str, Any]]]:
    out = {}
    with zipfile.ZipFile(ZIP_PATH) as z:
        for suite, cfg in SUITES.items():
            out[suite] = json.loads(z.read(cfg["ref_member"]))["molecules"]
    return out


def load_scores(base: Path, suite: str) -> dict[tuple[str, str], dict[str, Any]]:
    p = base / "verifier_scores" / f"{suite.lower()}_rescored_verifier_scores.csv"
    out = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error") or not row.get("new_total"): continue
            out[(row["case_id"], row["label"])] = row
    return out


def reference_map(suite: str, refs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    if suite == "FDA":
        for r in refs:
            out[canonical(r["smiles"])] = r
    else:
        # JMC cases.csv is authoritative for case-to-reference target matching.
        cases = SUITES[suite]["base"].parent.parent / "dataset/benchmark/JMC/cases.csv"
        # fallback to repo path when base.parent.parent differs after relocation
        if not cases.exists(): cases = ROOT / "dataset/benchmark/JMC/cases.csv"
        by_target = {canonical(r["smiles"]): r for r in refs}
        with cases.open(encoding="utf-8-sig", newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                smi = canonical(row.get("smiles") or row.get("SMILES") or "")
                ref = by_target.get(smi)
                if ref: out[smi] = ref
    return out


def load_pool() -> list[dict[str, Any]]:
    refs = load_refs(); cases = []
    for suite, cfg in SUITES.items():
        base = cfg["base"]; ref_by_smi = reference_map(suite, refs[suite]); scores = load_scores(base, suite)
        for run_path in sorted((base / "runs").glob("*.json")):
            run = json.loads(run_path.read_text())
            cid = run_path.stem; target = canonical(run.get("smiles", "")); ref = ref_by_smi.get(target)
            if not ref or not ref.get("valid"): continue
            ranked = extract_ranked_routes(base / "transcripts" / f"{cid}.txt")
            run_all = {r.get("label"): r for r in (run.get("all_routes") or []) if r.get("label")}
            # Some FDA run summaries omit all_routes. Recover complete saved routes
            # from save_best_route calls in the transcript, then use rank_routes order/scores.
            transcript_path = base / "transcripts" / f"{cid}.txt"
            if transcript_path.exists():
                text = transcript_path.read_text(errors="ignore")
                for m in re.finditer(r"\[tool_call\]\s*save_best_route\(", text):
                    obj = extract_balanced_json(text, m.end())
                    if obj and obj.get("label") and obj.get("steps"):
                        run_all.setdefault(obj["label"], obj)
                # list_best_routes exposes labels/scores but not steps. Earlier
                # score_route calls contain the complete route_steps payload;
                # recover those by matching the score to the listed candidate.
                score_routes = []
                for m in re.finditer(r"\[tool_call\]\s*score_route\(", text):
                    obj = extract_balanced_json(text, m.end())
                    if not obj or not obj.get("route_steps"): continue
                    try:
                        steps_obj = json.loads(obj["route_steps"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    steps = []
                    for st in steps_obj:
                        react = ".".join(st.get("reactants", [])); prod = st.get("product", "")
                        if react and prod: steps.append(f"{prod}>>{react}")
                    if not steps: continue
                    # The adjacent tool result carries total_score.
                    res_at = text.find("========== TOOL ==========", m.end())
                    total = None
                    if res_at >= 0:
                        res = extract_balanced_json(text, res_at)
                        if res and res.get("total_score") is not None: total = float(res["total_score"])
                    if total is not None:
                        score_routes.append({"steps": steps, "starting_materials": [], "total_score": total})
                for rr in ranked:
                    label = rr.get("label")
                    if label in run_all or rr.get("total_score") is None: continue
                    if score_routes:
                        hit = min(score_routes, key=lambda x: abs(x["total_score"] - float(rr["total_score"])))
                        if abs(hit["total_score"] - float(rr["total_score"])) <= 5e-4:
                            run_all[label] = dict(hit, label=label, notes=rr.get("notes", ""),
                                                  starting_materials=rr.get("starting_materials", []))
            # Prefer the final rank_routes output. When absent or truncated,
            # all complete saved routes recovered above are still valid input.
            rank_by_label = {r.get("label"): r for r in ranked if r.get("label")}
            if not ranked or len(rank_by_label) < 4:
                ranked = []
                rank_by_label = {}
            for label, full in run_all.items():
                if label not in rank_by_label:
                    total = full.get("total_score") or (full.get("sub_scores") or {}).get("total_score")
                    if total is not None:
                        ranked.append({"label": label, "total_score": total,
                                       "starting_materials": full.get("starting_materials", []),
                                       "notes": full.get("notes", "")})
            routes = []
            for r in ranked:
                label = r.get("label", ""); sc = scores.get((cid, label)); full = run_all.get(label, {})
                total = float(sc["new_total"]) if sc else r.get("total_score")
                steps = full.get("steps") or r.get("steps")
                if not steps and r.get("first_step") and int(r.get("n_steps") or 0) == 1:
                    steps = [r["first_step"]]
                if total is None or not steps: continue
                x = {
                    "source": "agent", "label": label, "score": float(total),
                    "feasibility_score": float(sc["feasibility_score"]) if sc else (full.get("sub_scores") or {}).get("feasibility_score", r.get("feasibility")),
                    "cost_score": float(sc["cost_score"]) if sc else (full.get("sub_scores") or {}).get("cost_score", r.get("cost")),
                    "greenness_score": float(sc["greenness_score"]) if sc else (full.get("sub_scores") or {}).get("greenness_score", r.get("greenness")),
                    "steps": steps, "starting_materials": full.get("starting_materials") or r.get("starting_materials", []),
                    "notes": full.get("notes") or r.get("notes", ""), "target_smiles": target,
                    "score_source": "rescored_verifier" if sc else "rank_routes_fallback",
                }
                x["features"] = route_features(x); routes.append(x)
            # Deduplicate exact reaction sequences.
            unique = {}
            for r in routes:
                key = tuple(r["features"]["reactions"])
                if key not in unique or r["score"] > unique[key]["score"]: unique[key] = r
            routes = list(unique.values())
            if len(routes) < 4: continue
            ref_route = {
                "source": "reference", "label": "literature_reference", "score": None,
                "steps": [s["reaction_smiles"] for s in ref["steps"]],
                "starting_materials": [], "target_smiles": target,
                "reference_metadata": ref.get("reference", {}),
            }
            ref_route["features"] = route_features(ref_route)
            cases.append({"suite": suite, "case_id": cid, "target_smiles": target,
                          "target_name": run.get("name") or run.get("label") or ref.get("molecule"),
                          "reference": ref_route, "agents": routes})
    return cases


def select_four(case: dict[str, Any]) -> dict[str, Any] | None:
    routes = case["agents"]
    scores = [r["score"] for r in routes]
    global_range = max(scores) - min(scores)
    best = None
    # Exact combinatorial search is cheap (usually <= 15 choose 4).
    for combo in combinations(routes, 4):
        sims = [route_similarity(a, b) for a, b in combinations(combo, 2)]
        span = max(r["score"] for r in combo) - min(r["score"] for r in combo)
        lengths = {len(r["steps"]) for r in combo}
        # Diversity dominates; score span and route-length variety break ties.
        objective = 0.52 * (1 - statistics.mean(sims)) + 0.23 * (1 - max(sims)) + \
                    0.20 * (span / global_range if global_range > 1e-9 else 0) + \
                    0.05 * min(1.0, (len(lengths) - 1) / 2)
        rec = {"routes": list(combo), "mean_similarity": statistics.mean(sims),
               "max_similarity": max(sims), "min_similarity": min(sims),
               "score_span": span, "objective": objective,
               "lengths": sorted(lengths)}
        if best is None or objective > best["objective"]: best = rec
    return best


def choose_cases(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    candidates = []
    for c in pool:
        sel = select_four(c)
        if not sel: continue
        c = dict(c); c["selection"] = sel
        # Exclude near-clones; reward interpretable score spread without cherry-picking only extremes.
        if sel["max_similarity"] >= 0.93 or sel["score_span"] < 0.04: continue
        c["case_quality"] = sel["objective"] + 0.05 * min(len(c["agents"]), 15) / 15
        candidates.append(c)
    # Balance FDA/JMC if possible: 5 each, then fill globally.
    picked = []
    for suite in ("FDA", "JMC"):
        s = sorted((c for c in candidates if c["suite"] == suite), key=lambda x: x["case_quality"], reverse=True)
        picked.extend(s[: n // 2])
    if len(picked) < n:
        used = {(c["suite"], c["case_id"]) for c in picked}
        rest = sorted((c for c in candidates if (c["suite"], c["case_id"]) not in used), key=lambda x: x["case_quality"], reverse=True)
        picked.extend(rest[: n-len(picked)])
    return picked[:n]


def svg_for_smiles(smiles: str, legend: str = "") -> str:
    m = Chem.MolFromSmiles(smiles)
    if not m: return f"<div class='bad'>Invalid SMILES: {html.escape(smiles)}</div>"
    return Draw.MolsToGridImage([m], molsPerRow=1, subImgSize=(480, 260), legends=[legend], useSVG=True)


def write_outputs(selected: list[dict[str, Any]], outdir: Path, seed: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    blind_cases, key_cases, rows = [], [], []
    html_parts = ["""<!doctype html><meta charset='utf-8'><title>ReTRACE expert ranking</title>
<style>body{font:15px system-ui;max-width:1500px;margin:auto;padding:28px;color:#18212b}.case{page-break-before:always;border-top:3px solid #1f7a5a;padding-top:18px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{border:1px solid #d8dee5;border-radius:12px;padding:14px}.step{border-top:1px solid #eee;padding:8px 0;overflow-wrap:anywhere}.meta{color:#61707f}.rank{background:#f5f7f9;padding:12px;border-radius:8px}@media print{.case{break-before:page}.cards{display:block}.card{break-inside:avoid;margin:12px 0}}</style>
<h1>ReTRACE 逆合成路线专家盲评</h1><p>请仅依据路线内容排序。每个目标包含 5 条候选，候选来源和系统分数均已隐藏。</p>"""]
    for ci, c in enumerate(selected, 1):
        routes = [c["reference"]] + c["selection"]["routes"]
        rng.shuffle(routes)
        public = []
        private = []
        html_parts.append(f"<section class='case'><h2>目标 {ci:02d}：{html.escape(str(c['target_name']))}</h2>")
        html_parts.append(svg_for_smiles(c["target_smiles"], "Target"))
        html_parts.append("<div class='rank'><b>请填写：</b> 综合排序（1=最佳）：____ &gt; ____ &gt; ____ &gt; ____ &gt; ____<br>并可分别记录可行性、成本、绿色性评分及备注。</div><div class='cards'>")
        for ri, r in enumerate(routes):
            code = chr(ord("A") + ri)
            pub = {"candidate_code": code, "steps": r["steps"]}
            public.append(pub)
            private.append({"candidate_code": code, "source": r["source"], "label": r["label"],
                            "agent_score": r.get("score"), "feasibility_score": r.get("feasibility_score"),
                            "cost_score": r.get("cost_score"), "greenness_score": r.get("greenness_score"),
                            "score_source": r.get("score_source"),
                            "reference_metadata": r.get("reference_metadata")})
            html_parts.append(f"<article class='card'><h3>候选 {code} · {len(r['steps'])} 步</h3>")
            for si, step in enumerate(r["steps"], 1):
                rxn, _, desc = step.partition(" (reaction:")
                html_parts.append(f"<div class='step'><b>Step {si}</b><br><code>{html.escape(rxn)}</code>")
                if desc: html_parts.append(f"<div class='meta'>{html.escape(desc.rstrip(')'))}</div>")
                if ">>" in rxn:
                    product = rxn.split(">>",1)[0].split(".")[0]
                    html_parts.append(svg_for_smiles(product))
                html_parts.append("</div>")
            html_parts.append("</article>")
            rows.append({"target_id": f"T{ci:02d}", "target_name": c["target_name"], "candidate_code": code,
                         "overall_rank": "", "feasibility_1_5": "", "cost_1_5": "", "greenness_1_5": "", "comments": ""})
        html_parts.append("</div></section>")
        blind_cases.append({"target_id": f"T{ci:02d}", "target_name": c["target_name"], "target_smiles": c["target_smiles"], "candidates": public})
        key_cases.append({"target_id": f"T{ci:02d}", "suite": c["suite"], "case_id": c["case_id"],
                          "target_name": c["target_name"], "target_smiles": c["target_smiles"],
                          "selection_metrics": {k:v for k,v in c["selection"].items() if k != "routes"},
                          "candidate_key": private})
    html_parts.append("<h2>提交前检查</h2><ul><li>每个目标的 A–E 均已排序且无并列（如需并列请在备注说明）。</li><li>排序时未查看候选来源或系统分数。</li><li>对明显不可行、原子不守恒或立体化学错误的路线已注明。</li></ul>")
    (outdir / "expert_blind_routes.html").write_text("".join(html_parts))
    (outdir / "expert_blind_routes.json").write_text(json.dumps({"schema_version":1,"cases":blind_cases}, ensure_ascii=False, indent=2))
    (outdir / "private_answer_key.json").write_text(json.dumps({"schema_version":1,"seed":seed,"cases":key_cases}, ensure_ascii=False, indent=2))
    with (outdir / "annotation_template.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest={"n_targets":len(selected),"n_candidates_per_target":5,"reference_per_target":1,"agent_per_target":4,
              "source_suites":dict((s,sum(c["suite"]==s for c in selected)) for s in SUITES),
              "selection_policy":"exact 4-route search; prioritize low route similarity, then score span and length diversity; balanced FDA/JMC",
              "blind_seed":seed,"files":{}}
    for name in ["expert_blind_routes.html","expert_blind_routes.json","private_answer_key.json","annotation_template.csv"]:
        b=(outdir/name).read_bytes(); manifest["files"][name] = {"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b)}
    (outdir/"selection_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    readme="""# ReTRACE 专家路线排序标注包\n\n- `expert_blind_routes.html`: 给专家阅读/打印的盲评页面（不含来源与 Agent 分数）\n- `annotation_template.csv`: 排序与分项评分录入表\n- `expert_blind_routes.json`: 盲化后的机器可读数据\n- `private_answer_key.json`: **仅研究团队保留**，包含 reference 身份、Agent 标签及分数\n- `selection_manifest.json`: 数据规模、来源平衡与文件校验值\n\n注意：本包按路线多样性和分数跨度有目的地筛选，适合做小规模判别性/校准性专家研究；不能把得到的相关系数解释为对全部 ReTRACE 路线的无偏总体估计。\n"""
    (outdir/"README.md").write_text(readme)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=ROOT/"expert_annotation_10x5")
    ap.add_argument("--n",type=int,default=10); ap.add_argument("--seed",type=int,default=20260706)
    args=ap.parse_args(); pool=load_pool(); selected=choose_cases(pool,args.n)
    if len(selected)<args.n: raise SystemExit(f"Only {len(selected)} eligible cases found")
    write_outputs(selected,args.out,args.seed)
    print(json.dumps({"pool_cases":len(pool),"selected":len(selected),"out":str(args.out),
                      "cases":[{"suite":c["suite"],"case_id":c["case_id"],"target":c["target_name"],
                                "score_span":round(c["selection"]["score_span"],4),
                                "mean_similarity":round(c["selection"]["mean_similarity"],4),
                                "max_similarity":round(c["selection"]["max_similarity"],4)} for c in selected]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
