#!/usr/bin/env python
"""Render a chemist-friendly HTML view of every FDA-30 run.

For each target, produces `chemist_view/<idx>_<name>.html` with:
  - target card (image + key stats + agent self-score vs rescored score)
  - chronological tool-call timeline, each event rendered with chemistry-aware
    formatting (suggest_templates / neural_retro show top picks as reaction
    images; explore_branches / score_route / check_reaction get tabular
    breakdowns; check_buyability / save_best_route / list_best_routes get
    compact cards)
  - the agent's final COMPLETE ROUTE FOUND block with step-by-step reaction
    images

Plus a global `chemist_view/index.html` linking everything together.

Inspired by ../show_smiles.py (RDKit MolDraw2DCairo + base64 PNG inlining
used by the ChemBart UI).
"""
from __future__ import annotations
import base64, glob, html, io, json, os, re
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import MolDraw2DCairo

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "dataset/benchmark/FDA_small_mol"
RUNS_DIR = BENCH / "runs"
OUT_DIR = BENCH / "chemist_view"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESCORED = {e["idx"]: e for e in json.load(open(BENCH / "_rescored.json"))}

# ---------- molecule rendering ----------
_mol_cache: dict[str, str] = {}
_rxn_cache: dict[tuple, str] = {}

def mol_png(smi: str, w=240, h=160) -> str:
    key = f"{smi}|{w}x{h}"
    if key in _mol_cache:
        return _mol_cache[key]
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            _mol_cache[key] = ""; return ""
        d = MolDraw2DCairo(w, h)
        d.drawOptions().padding = 0.08
        d.DrawMolecule(mol)
        d.FinishDrawing()
        b = base64.b64encode(d.GetDrawingText()).decode()
        url = f"data:image/png;base64,{b}"
        _mol_cache[key] = url
        return url
    except Exception:
        _mol_cache[key] = ""; return ""


def rxn_png(reactants: list[str], products: list[str], w=560, h=190) -> str:
    """Reactants → product reaction image (mirrors show_smiles.py logic)."""
    key = (tuple(reactants), tuple(products), w, h)
    if key in _rxn_cache:
        return _rxn_cache[key]
    try:
        rxn = AllChem.ChemicalReaction()
        for s in reactants:
            m = Chem.MolFromSmiles(s)
            if m is not None: rxn.AddReactantTemplate(m)
        for s in products:
            m = Chem.MolFromSmiles(s)
            if m is not None: rxn.AddProductTemplate(m)
        if rxn.GetNumReactantTemplates()==0 and rxn.GetNumProductTemplates()==0:
            _rxn_cache[key] = ""; return ""
        d = MolDraw2DCairo(w, h)
        d.drawOptions().padding = 0.08
        d.DrawReaction(rxn, highlightByReactant=False)
        d.FinishDrawing()
        b = base64.b64encode(d.GetDrawingText()).decode()
        url = f"data:image/png;base64,{b}"
        _rxn_cache[key] = url
        return url
    except Exception:
        _rxn_cache[key] = ""; return ""


# ---------- HTML helpers ----------
def esc(s) -> str: return html.escape(str(s))

def img_or_code(img_url: str, fallback_code: str) -> str:
    if img_url:
        return f'<img src="{img_url}"/>'
    return f'<code>{esc(fallback_code)}</code>'


def fnum(v, n=3) -> str:
    if isinstance(v, (int, float)) and v == v: return f"{v:.{n}f}"
    return "—"


# ---------- per-tool renderers ----------
def render_analyze(r: dict) -> str:
    fg = ", ".join(f"{x.get('name')}×{x.get('count')}" for x in r.get("functional_groups", [])) or "—"
    rows = [
        ("Formula / MW", f"{r.get('atom_counts')} · MW {r.get('mw')}"),
        ("Functional groups", fg),
        ("Rings (aromatic)", f"{r.get('ring_count')} ({r.get('aromatic_rings')})"),
        ("Chiral centers", r.get("chiral_centers")),
        ("HBD / HBA / TPSA", f"{r.get('hbd')} / {r.get('hba')} / {r.get('tpsa')}"),
        ("logP / Rot. bonds", f"{r.get('logp')} / {r.get('rotatable_bonds')}"),
        ("SAScore / complexity", f"{r.get('sa_score')} / {r.get('complexity')}"),
    ]
    body = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
    return f"<table class='kv'>{body}</table>"


def render_validate(r: dict) -> str:
    return ("<table class='kv'>"
            f"<tr><th>valid</th><td>{esc(r.get('valid'))}</td></tr>"
            f"<tr><th>canonical</th><td><code>{esc(r.get('canonical'))}</code></td></tr>"
            f"<tr><th>formula / MW</th><td>{esc(r.get('formula'))} · {esc(r.get('mw'))}</td></tr>"
            f"<tr><th>InChIKey</th><td><code>{esc(r.get('inchikey'))}</code></td></tr>"
            "</table>")


def render_suggest(r: dict, target_smi: str | None, k: int = 6) -> str:
    sugs = r.get("suggestions", [])[:k]
    prod = r.get("product") or target_smi
    rows = []
    for i, s in enumerate(sugs):
        kind = s.get("template_kind","?")
        src = s.get("template_source","?")
        score = fnum(s.get("rank_score"))
        mwb = fnum(s.get("mw_balance"))
        reactants = s.get("reactants") or [s.get("reactants_smiles","")]
        if isinstance(reactants, str): reactants = [reactants]
        img = rxn_png(reactants, [prod] if prod else [])
        kind_cls = "enz" if kind == "enzymatic" else "org"
        rows.append(
            f"<tr><td>{i+1}</td><td><span class='kind {kind_cls}'>{esc(kind)}</span><br><small>{esc(src)}</small></td>"
            f"<td>{score}</td><td>{mwb}</td>"
            f"<td>{img_or_code(img, '.'.join(reactants))}</td></tr>"
        )
    note = (f"<p class='meta'>library {r.get('library_size')}, after pre-filter {r.get('candidates_after_prefilter')}, "
            f"showing top {len(sugs)} of {r.get('num_suggestions')}</p>")
    return note + (
        "<table class='tbl'><thead><tr><th>#</th><th>source</th><th>rank</th><th>MW bal</th><th>reaction</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_neural(r: dict, target_smi: str | None, k: int = 6) -> str:
    pe = r.get("per_engine_counts", {})
    pe_ms = r.get("per_engine_ms", {})
    head = "<p class='meta'>" + " · ".join(
        f"{eng}: {pe.get(eng)} preds in {pe_ms.get(eng)} ms" for eng in pe
    ) + f" · {r.get('n_unique_after_dedup')} unique after dedup</p>"
    preds = r.get("predictions", [])[:k]
    rows = []
    target = r.get("target") or target_smi
    for i, p in enumerate(preds):
        rs = p.get("reactants","")
        sup = p.get("support") or []
        sup_html = " ".join(f"<span class='eng eng-{e}'>{e}</span>" for e in sup)
        reactants = rs.split(".") if isinstance(rs, str) else rs
        img = rxn_png(reactants, [target] if target else [])
        rows.append(
            f"<tr><td>{i+1}</td><td>{sup_html}</td><td>{fnum(p.get('score'))}</td>"
            f"<td>bucket {p.get('bucket')}/sz {p.get('cluster_size')}</td>"
            f"<td>{img_or_code(img, rs)}</td></tr>"
        )
    return head + (
        "<table class='tbl'><thead><tr><th>#</th><th>support</th><th>score</th><th>cluster</th><th>reaction</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_explore(r: dict, target_smi: str | None) -> str:
    rows = []
    prod = r.get("product") or target_smi
    for i, b in enumerate(r.get("ranked", [])):
        reactants = b.get("reactants", [])
        img = rxn_png(reactants, [prod] if prod else [])
        rows.append(
            f"<tr><td>{i+1}</td><td><b>{esc(b.get('name'))}</b><br><small>{esc(b.get('notes',''))}</small></td>"
            f"<td>simp {fnum(b.get('simplicity'))}<br>buy ratio {fnum(b.get('buyable_ratio'))}<br>"
            f"score {fnum(b.get('score') or b.get('rank_score'))}</td>"
            f"<td>{img_or_code(img, '.'.join(reactants))}</td></tr>"
        )
    return ("<table class='tbl'><thead><tr><th>#</th><th>strategy</th><th>scores</th><th>reaction</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def render_check_buyability(r: dict) -> str:
    smi = r.get("input_smiles") or r.get("smiles","")
    img = mol_png(smi, 180, 120)
    buy = r.get("buyable")
    badge = "<span class='ok'>✓ buyable</span>" if buy else "<span class='no'>✗ not in stock</span>"
    stocks = ", ".join(r.get("found_in_stocks", []) or []) or "—"
    img_html = f'<img src="{img}"/>' if img else ''
    return (f"<div class='leaf'>{img_html}"
            f"<div><code>{esc(smi)}</code><br>{badge}<br><small>stocks: {esc(stocks)}</small></div></div>")


def render_check_reaction(r: dict) -> str:
    bal = r.get("balanced_with_byproduct") or r.get("balanced")
    badge = "<span class='ok'>balanced</span>" if bal else "<span class='no'>unbalanced</span>"
    bp = r.get("byproduct") or "—"
    return (f"<div>{badge} · byproduct: <b>{esc(bp)}</b><br>"
            f"<small>note: {esc(r.get('note',''))}</small><br>"
            f"<small>atom diff: {esc(r.get('atom_diff'))}</small></div>")


def render_score_route(r: dict) -> str:
    if "error" in r:
        return f"<div class='no'>error: {esc(r['error'])[:300]}</div>"
    fe = r.get("feasibility", {}); co = r.get("cost", {}); gr = r.get("greenness", {})
    notes = "<br>".join(esc(n) for n in r.get("notes", [])) or "—"
    judge_block = ""
    j = fe.get("judge") or {}
    if j:
        judge_block = (f"<div class='judge'>judge: <b>{esc(j.get('verdict'))}</b> ({fnum(j.get('score'),2)})"
                       f" — {esc((j.get('top_concern') or '')[:200])}</div>")
    leaves_block = ""
    leaves = co.get("leaves") or []
    if leaves:
        ls = "".join(
            f"<tr><td><code>{esc(l.get('smiles',''))}</code></td>"
            f"<td>{fnum(l.get('leaf_score'))}</td><td>{esc(l.get('price_per_g'))}</td></tr>"
            for l in leaves
        )
        leaves_block = ("<details><summary>leaves</summary><table class='tbl small'>"
                        "<thead><tr><th>SMILES</th><th>score</th><th>$/g</th></tr></thead>"
                        f"<tbody>{ls}</tbody></table></details>")
    fe_warn = "<br>".join(esc(w) for w in fe.get("warnings", [])) or ""
    return (
        f"<div class='score-grid'>"
        f"<div><h4>total</h4><div class='big'>{fnum(r.get('total_score'))}</div></div>"
        f"<div><h4>feasibility</h4><div class='big'>{fnum(fe.get('score'))}</div>"
        f"<small>pre={fnum(fe.get('pre_judge_score', fe.get('t5_score')))}, n_steps={fe.get('n_steps')}</small>"
        f"{('<div class=warn>'+fe_warn+'</div>') if fe_warn else ''}{judge_block}</div>"
        f"<div><h4>cost</h4><div class='big'>{fnum(co.get('score'))}</div>"
        f"<small>step_eff={fnum(co.get('step_efficiency'))}, leaf_eco={fnum(co.get('leaf_economics'))}, "
        f"sc_decr={fnum(co.get('sc_decrease'))}, leaves={co.get('n_leaves')} (avail {co.get('n_available')})</small>"
        f"{leaves_block}</div>"
        f"<div><h4>greenness</h4><div class='big'>{fnum(gr.get('score'))}</div>"
        f"<small>atom_eco={fnum(gr.get('atom_economy_mean'))}, T̄={gr.get('temperature_C_mean')}°C</small>"
        f"<br><small>solvents: {esc(gr.get('solvent_class_counts'))}</small>"
        f"<br><small>hazards: {esc(gr.get('hazardous_reagents') or '—')}</small></div>"
        f"</div>"
        f"<div class='notes'><b>weights:</b> {esc(r.get('weights_used'))} · "
        f"<b>elapsed:</b> {r.get('elapsed_ms')} ms<br>{notes}</div>"
    )


def render_save_route(r: dict) -> str:
    s = r.get("saved") or {}
    sub = s.get("sub_scores", {})
    return (
        f"<div><b>saved:</b> {esc(s.get('label'))} · total {fnum(s.get('total_score'))} "
        f"(feas {fnum(sub.get('feasibility'))}, cost {fnum(sub.get('cost'))}, green {fnum(sub.get('greenness'))}) · "
        f"{s.get('num_steps')} steps, {s.get('num_starting_materials')} starting materials"
        f"<br><small>{esc(s.get('notes_excerpt') or '')}</small></div>"
    )


def render_list_routes(r: dict) -> str:
    rows = []
    for i, rt in enumerate(r.get("routes", [])):
        sub = rt.get("sub_scores", {})
        rows.append(
            f"<tr><td>{i+1}</td><td><b>{esc(rt.get('label'))}</b></td>"
            f"<td>{fnum(rt.get('total_score'))}</td>"
            f"<td>{fnum(sub.get('feasibility'))}/{fnum(sub.get('cost'))}/{fnum(sub.get('greenness'))}</td>"
            f"<td>{rt.get('num_steps')}</td></tr>"
        )
    return ("<table class='tbl'><thead><tr><th>#</th><th>label</th><th>total</th>"
            "<th>feas/cost/green</th><th>n_steps</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def render_get_route(r: dict, target_smi: str | None) -> str:
    rt = r.get("route") or {}
    steps_html = ""
    for step_str in rt.get("steps", []):
        steps_html += "<li><code>" + esc(step_str) + "</code>"
        # try to extract reactants/product
        m = re.match(r"^\s*(.+?)\s*(?:>>|→|->)\s*(.+?)(?:\s*\(reaction:.*)?$", step_str)
        if m:
            lhs = [p.strip() for p in re.split(r"\s*\+\s*", m.group(1)) if p.strip()]
            lhs2 = []
            for p in lhs:
                lhs2.extend([x for x in p.split('.') if x])
            rhs = re.sub(r"\s*\(.*", "", m.group(2)).strip()
            img = rxn_png(lhs2, [rhs] if rhs else [])
            if img: steps_html += f"<br><img src='{img}'/>"
        steps_html += "</li>"
    sm = rt.get("starting_materials") or []
    sm_imgs = "".join(f"<div class='leaf'><img src='{mol_png(s,160,110)}'/><div><code>{esc(s)}</code></div></div>" for s in sm)
    return (f"<div><b>label:</b> {esc(rt.get('label'))} · total {fnum(rt.get('total_score'))}</div>"
            f"<ol class='steps'>{steps_html}</ol>"
            f"{('<h4>Starting materials</h4>'+sm_imgs) if sm_imgs else ''}")


def render_load_result(r: dict) -> str:
    return f"<small>load_result of <code>{esc(r.get('result_id'))}</code> ({esc(r.get('tool'))})</small>"


def render_compare(r: dict) -> str:
    return f"<pre>{esc(json.dumps(r, indent=1, default=str)[:1500])}</pre>"


def render_generic(r: Any) -> str:
    s = json.dumps(r, indent=1, default=str)
    if len(s) > 1500: s = s[:1500] + "\n... [truncated]"
    return f"<pre>{esc(s)}</pre>"


def render_tool_result(name: str, result: Any, target_smi: str | None) -> str:
    if not isinstance(result, dict):
        return render_generic(result)
    if name == "analyze_molecule": return render_analyze(result)
    if name == "validate_smiles":  return render_validate(result)
    if name == "suggest_templates":return render_suggest(result, target_smi)
    if name == "neural_retro":     return render_neural(result, target_smi)
    if name == "explore_branches": return render_explore(result, target_smi)
    if name == "check_buyability": return render_check_buyability(result)
    if name == "check_reaction":   return render_check_reaction(result)
    if name == "score_route":      return render_score_route(result)
    if name == "save_best_route":  return render_save_route(result)
    if name == "list_best_routes": return render_list_routes(result)
    if name == "get_best_route":   return render_get_route(result, target_smi)
    if name == "load_result":      return render_load_result(result)
    if name == "compare_molecules":return render_compare(result)
    return render_generic(result)


def render_tool_call_args(args: Any) -> str:
    s = json.dumps(args, default=str)
    if len(s) > 200: s = s[:200] + "…"
    return f"<code>{esc(s)}</code>"


# ---------- per-target page ----------
PAGE_CSS = """
<style>
* { box-sizing: border-box; }
body { font-family: -apple-system,Segoe UI,Helvetica,Arial,sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; line-height: 1.4; }
h1 { font-size: 1.6em; margin-bottom: 4px;} h2 {font-size:1.2em; margin-top:32px; padding-bottom:4px; border-bottom: 2px solid #ddd;}
h3 { font-size: 1em; margin: 14px 0 6px; } h4 { margin: 4px 0; font-size:.9em; color:#666;}
.target-card { display:flex; gap: 24px; align-items: flex-start; background: #fafbfc; border: 1px solid #e1e4e8; border-radius:8px; padding:16px; }
.target-card img { max-width: 280px; border-radius:4px; background:white; padding:6px; }
table.kv { border-collapse:collapse; }
table.kv th { text-align:left; padding: 2px 12px 2px 0; font-weight:500; color:#586069; vertical-align: top; }
table.kv td { padding: 2px 0; }
table.tbl { border-collapse: collapse; width:100%; margin: 4px 0 12px; font-size: .9em;}
table.tbl th, table.tbl td { padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
table.tbl th { text-align: left; background:#f6f8fa; font-weight: 600;}
table.tbl img { max-width: 480px; }
table.tbl.small td, table.tbl.small th { padding: 2px 6px; font-size: .85em;}
.event { margin: 8px 0 8px 0; padding: 10px 12px; border-left: 3px solid #ddd; background:#fff; }
.event .head { font-size: .85em; color: #586069; }
.event.tool_call { border-left-color: #0366d6; background:#f1f8ff; }
.event.tool_result { border-left-color: #28a745; background:#f6fff8; }
.event.thinking { border-left-color: #d29922; background:#fffbea; font-style: italic; color:#333;}
.event.iteration { border-left-color: #999; background:#fafafa; font-size:.85em; padding: 4px 12px; color:#666;}
.event.complete { border-left-color: #6f42c1; background:#f5efff;}
.event.error { border-left-color: #d73a49; background:#ffeef0; color:#86181d;}
.event.checkpoint_compacted { border-left-color: #ccc; background:#f6f8fa; font-size:.85em; color:#666; padding:4px 12px;}
.kind { font-size:.75em; padding:1px 6px; border-radius:3px; }
.kind.enz { background:#e6f4ea; color:#1e6e3a; }
.kind.org { background:#eaf3ff; color:#114a91; }
.eng { display:inline-block; font-size:.7em; padding:1px 5px; border-radius:3px; background:#eee; }
.eng-rxngraphormer { background:#fde7d2; color:#84490a;}
.eng-chembart { background:#d8e8ff; color:#0a3a84;}
.eng-graph_retrosyn { background:#d8f0d8; color:#0a5a0a;}
.ok { color: #28a745; font-weight:600;} .no { color:#d73a49; font-weight:600;}
.score-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin:8px 0; }
.score-grid > div { padding:8px; border:1px solid #e1e4e8; border-radius:6px; background:#fff; font-size:.85em;}
.score-grid .big { font-size: 1.5em; font-weight:600; color:#0366d6; }
.score-grid .warn { color:#a66800; margin-top:4px; font-size:.8em; }
.score-grid .judge { margin-top:6px; color:#586069; font-size:.8em; }
.notes { font-size:.8em; color: #586069; margin-top:6px; }
.leaf { display:inline-flex; align-items:center; gap:8px; margin:6px 12px 6px 0; padding:4px 8px; background:#fafbfc; border:1px solid #e1e4e8; border-radius:4px; }
.leaf img { background: white; }
ol.steps { padding-left: 24px; }
ol.steps li { margin-bottom: 12px; }
ol.steps img { max-width: 600px; }
pre { background:#f6f8fa; padding:8px; border-radius:4px; font-size: .8em; max-height: 220px; overflow:auto;}
.meta { color: #586069; font-size: .85em; margin: 4px 0;}
.summary-bar { display:flex; gap:24px; margin-top:8px; flex-wrap: wrap;}
.summary-bar .stat { padding:6px 10px; border-radius:6px; background:#fafbfc; border:1px solid #e1e4e8;}
.summary-bar .stat b { display:block; font-size:1.2em; color:#0366d6; }
.toc { font-size: .85em; columns: 2; }
.idx-table th, .idx-table td { padding: 4px 8px; border-bottom:1px solid #eee; text-align:left; font-size:.9em; }
.idx-table { border-collapse:collapse; width:100%; }
</style>"""


def session_dir_for(run: dict) -> Path:
    ts = run["started_at"]  # 2026-05-06 00:02:32
    sess = "_session_" + ts.replace("-","").replace(" ","-").replace(":","")
    return REPO / "runs" / sess


def render_target_page(run: dict) -> str:
    idx = run["idx"]; name = run["name"]; smi = run["smiles"]
    sess = session_dir_for(run)
    tpath = sess / "transcript.json"
    transcript = json.load(open(tpath)) if tpath.exists() else {"events": []}
    target_smi = transcript.get("target_smiles") or smi

    rescored = RESCORED.get(idx, {})
    resc = rescored.get("rescored") or {}
    self_score = rescored.get("agent_self_score")

    head = f"<h1>idx {idx:02d} · {esc(name)} <small style='color:#666;font-size:.6em;'>({esc(run.get('variant',''))})</small></h1>"
    target_img = mol_png(smi, 280, 200)

    summary_html = (
        f"<div class='summary-bar'>"
        f"<div class='stat'><b>{'success' if run.get('success') else 'fail'}</b>agent verdict</div>"
        f"<div class='stat'><b>{run.get('iterations')}</b>iterations</div>"
        f"<div class='stat'><b>{run.get('wall_seconds')} s</b>wall time</div>"
        f"<div class='stat'><b>{fnum(self_score)}</b>self-score (in-run)</div>"
        f"<div class='stat'><b>{fnum(resc.get('total_score'))}</b>rescored total</div>"
        f"<div class='stat'><b>{esc(run.get('ref_step','?'))}</b>ref. steps (paper)</div>"
        f"<div class='stat'><b>{esc(run.get('chembart_step','—'))}</b>ChemBart MCTS step{(' (FAIL)' if run.get('chembart_full_success')=='failed' else '')}</div>"
        f"</div>"
    )

    # rescored sub-pillars
    if isinstance(resc.get("total_score"), (int, float)):
        sub = lambda k: (resc.get(k) or {}).get("score") if isinstance(resc.get(k), dict) else None
        rescored_block = (
            f"<table class='kv'>"
            f"<tr><th>rescored feasibility</th><td>{fnum(sub('feasibility'))}</td></tr>"
            f"<tr><th>rescored cost</th><td>{fnum(sub('cost'))}</td></tr>"
            f"<tr><th>rescored greenness</th><td>{fnum(sub('greenness'))}</td></tr>"
            f"<tr><th>weights</th><td>{esc(resc.get('weights_used'))}</td></tr>"
            f"</table>"
        )
    else:
        rescored_block = f"<p class='meta'>rescore: {esc(resc.get('error') or 'no saved route')}</p>"

    target_img_html = f'<img src="{target_img}"/>' if target_img else ''
    target_card = (f"<div class='target-card'>"
                   f"{target_img_html}"
                   f"<div><div><code>{esc(smi)}</code></div>{summary_html}{rescored_block}</div></div>")

    # final route
    final_section = ""
    rt = run.get("route_text", "") or ""
    if "COMPLETE ROUTE FOUND" in rt:
        # parse Step lines and the Starting materials line
        m = re.search(r"COMPLETE ROUTE FOUND.*?(?=\Z)", rt, flags=re.DOTALL)
        block = m.group(0) if m else rt
        steps = re.findall(r"Step\s*\d+[^\n]*?(?:>>|→|->)[^\n]+", block)
        steps_html = ""
        for s in steps:
            mm = re.match(r"^\s*(?:Step\s*\d+[^:]*:\s*)?(.+?)\s*(?:>>|→|->)\s*(.+?)(?:\s*\(reaction:.*)?$", s)
            if not mm:
                steps_html += f"<li><code>{esc(s)}</code></li>"
                continue
            lhs = [p.strip() for p in re.split(r"\s*\+\s*", mm.group(1)) if p.strip()]
            lhs2 = []
            for p in lhs: lhs2.extend([x for x in p.split('.') if x])
            rhs = re.sub(r"\s*\(.*","", mm.group(2)).strip()
            img = rxn_png(lhs2, [rhs] if rhs else [])
            steps_html += f"<li><code>{esc(s)}</code>"
            if img: steps_html += f"<br><img src='{img}'/>"
            steps_html += "</li>"
        sm_match = re.search(r"Starting materials[^\n:]*:\s*(.+)", block)
        sm_html = ""
        if sm_match:
            sm_line = sm_match.group(1).strip()
            sms = re.findall(r"`([^`]+)`", sm_line) or re.split(r"[,;]", sm_line)
            sms = [x.strip(" `") for x in sms if x.strip(" `")]
            sm_html = "<h4>Starting materials</h4>" + "".join(
                f"<div class='leaf'><img src='{mol_png(s,160,110)}'/><div><code>{esc(s)}</code></div></div>" for s in sms
            )
        final_section = (f"<h2>Final route (agent output)</h2>"
                         f"<ol class='steps'>{steps_html}</ol>{sm_html}"
                         f"<details><summary>raw COMPLETE ROUTE block</summary><pre>{esc(block[:5000])}</pre></details>")
    elif rt:
        final_section = f"<h2>Final output</h2><pre>{esc(rt[:3000])}</pre>"

    # timeline
    events = transcript.get("events", [])
    events_html = []
    iter_n = 0
    for e in events:
        et = e.get("event")
        ts = e.get("ts_rel")
        if et == "iteration":
            iter_n = e.get("n", iter_n + 1)
            events_html.append(f"<div class='event iteration'>iter {iter_n} (t={ts}s)</div>")
        elif et == "thinking":
            content = (e.get("content") or "").strip()
            if content:
                events_html.append(
                    f"<div class='event thinking'><div class='head'>think · t={ts}s</div>{esc(content)[:1500]}</div>"
                )
        elif et == "tool_call":
            args = render_tool_call_args(e.get("arguments"))
            events_html.append(
                f"<div class='event tool_call'><div class='head'>tool_call · {esc(e.get('name'))} · t={ts}s</div>{args}</div>"
            )
        elif et == "tool_result":
            n = e.get("name")
            r = e.get("result")
            ms = e.get("elapsed_ms")
            body = render_tool_result(n, r, target_smi)
            events_html.append(
                f"<div class='event tool_result'><div class='head'>tool_result · {esc(n)} · {ms} ms</div>{body}</div>"
            )
        elif et == "error":
            events_html.append(
                f"<div class='event error'><div class='head'>error · t={ts}s</div>{esc(e.get('message',''))}</div>"
            )
        elif et == "checkpoint_compacted":
            events_html.append(
                f"<div class='event checkpoint_compacted'>microCompact · demoted {e.get('demoted')} old result(s) · t={ts}s</div>"
            )
        elif et == "complete":
            content = (e.get("content") or "")[:5000]
            events_html.append(
                f"<div class='event complete'><div class='head'>complete · t={ts}s</div><pre>{esc(content)}</pre></div>"
            )
        elif et == "prefetch":
            tools = ", ".join(e.get("tools", []))
            events_html.append(f"<div class='event iteration'>prefetch: {esc(tools)} · t={ts}s</div>")

    timeline_html = "<h2>Tool-call timeline</h2>" + "".join(events_html)

    nav = "<p><a href='index.html'>← back to index</a></p>"

    return ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>idx {idx:02d} {esc(name)}</title>{PAGE_CSS}</head><body>"
            f"{nav}{head}{target_card}{final_section}{timeline_html}{nav}</body></html>")


# ---------- index ----------
def render_index(items: list[dict]) -> str:
    rows = []
    for it in items:
        idx = it["run"]["idx"]; name = it["run"]["name"]
        resc = (RESCORED.get(idx, {}).get("rescored") or {})
        total = resc.get("total_score") if "total_score" in resc else None
        self_s = RESCORED.get(idx, {}).get("agent_self_score")
        success = it["run"].get("success")
        rows.append(
            f"<tr><td>{idx}</td>"
            f"<td><a href='{esc(it['file'])}'>{esc(name)}</a></td>"
            f"<td>{'✓' if success else '✗'}</td>"
            f"<td>{it['run'].get('iterations')}</td>"
            f"<td>{it['run'].get('wall_seconds')}</td>"
            f"<td>{fnum(self_s)}</td>"
            f"<td>{fnum(total)}</td>"
            f"<td><code>{esc(it['run']['smiles'])}</code></td></tr>"
        )
    return ("<!DOCTYPE html><html><head><meta charset='utf-8'><title>FDA-30 chemist view</title>"
            f"{PAGE_CSS}</head><body>"
            "<h1>FDA-30 small-molecule benchmark — chemist view</h1>"
            "<p class='meta'>Click a target to open the per-run page with full tool-call timeline, "
            "neural one-step predictions per engine, and the final route as drawn reactions. "
            "Rescored values use the now-healthy ReactionT5 backend (weights 0.40/0.25/0.35 from "
            "<code>agent_config.openrouter.yaml</code>).</p>"
            "<table class='idx-table'>"
            "<thead><tr><th>idx</th><th>name</th><th>ok?</th><th>iters</th><th>wall (s)</th>"
            "<th>self-score</th><th>rescored</th><th>SMILES</th></tr></thead><tbody>"
            f"{''.join(rows)}</tbody></table></body></html>")


def main():
    runs = sorted(glob.glob(str(BENCH / "runs" / "*.json")))
    items = []
    for p in runs:
        run = json.load(open(p))
        idx = run["idx"]; name = run["name"]
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        fname = f"{idx:02d}_{safe_name}.html"
        try:
            html_doc = render_target_page(run)
        except Exception as e:
            print(f"[idx={idx:02d}] render error: {e}")
            html_doc = f"<html><body><h1>idx {idx} {name}</h1><pre>{esc(repr(e))}</pre></body></html>"
        (OUT_DIR / fname).write_text(html_doc, encoding="utf-8")
        items.append({"run": run, "file": fname})
        print(f"[idx={idx:02d}] {name} -> {fname} ({len(html_doc)//1024} KB)")
    (OUT_DIR / "index.html").write_text(render_index(items), encoding="utf-8")
    print(f"\nWrote {len(items)} pages + index.html under {OUT_DIR}")


if __name__ == "__main__":
    main()
