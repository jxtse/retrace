#!/usr/bin/env python3
"""Visualize hard_mol transcripts as standalone HTML reports.

Generates one ``visualization.html`` per case under
``dataset/benchmark/hard_mol/results/<run_id>/<case_dir>/`` for the 11
hard_mol benchmark cases. Each HTML is self-contained (RDKit-rendered
reaction images embedded as base64 data URIs) so chemistry-background
collaborators can double-click open without any local setup.

Reuses ``show_smiles.{reaction_smiles_to_image_base64,smiles_to_image_base64}``
already in the repo root.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# show_smiles imports `DrawingOptions` which the current RDKit removed; provide
# a local replacement instead of relying on it.
import base64
import io
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D


def _data_uri(png_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"


def reaction_smiles_to_image_base64(reactants, reagents, products, size=(640, 200)) -> str:
    try:
        rxn = AllChem.ChemicalReaction()
        for smi in list(reactants) + list(reagents):
            m = Chem.MolFromSmiles(smi)
            if m:
                rxn.AddReactantTemplate(m)
        for smi in products:
            m = Chem.MolFromSmiles(smi)
            if m:
                rxn.AddProductTemplate(m)
        if rxn.GetNumReactantTemplates() == 0 and rxn.GetNumProductTemplates() == 0:
            return ""
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().padding = 0.1
        drawer.DrawReaction(rxn, highlightByReactant=False)
        drawer.FinishDrawing()
        return _data_uri(drawer.GetDrawingText())
    except Exception:
        return ""


def smiles_to_image_base64(smiles, size=(200, 150)) -> str:
    try:
        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return ""
        img = Draw.MolToImage(m, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return _data_uri(buf.getvalue())
    except Exception:
        return ""

RESULTS_ROOT = REPO / "dataset/benchmark/hard_mol/results"
RUNS_ROOT = REPO / "runs"

# (idx → (run_id, optional explicit session_dir name to read transcript.jsonl from
# when transcript.json is absent))
RUN_MAP: dict[int, tuple[str, str | None]] = {
    # 3-case verification run (2026-05-19) after the three fixes landed:
    # stereo_check R/S-signature, per-step LLM judge, RAG prefetch suppression.
    0: ("example-run", None),
    1: ("example-run", None),
    2: ("example-run", None),
}

REACTION_IMG_SIZE = (440, 140)
MOL_IMG_SIZE = (96, 76)


# ---------------------------------------------------------------------------
# Image cache (per-case)
# ---------------------------------------------------------------------------

_rxn_cache: dict[tuple, str] = {}
_mol_cache: dict[tuple, str] = {}


def _reset_caches() -> None:
    _rxn_cache.clear()
    _mol_cache.clear()


def rxn_img(reactants: list[str], products: list[str],
            reagents: list[str] | None = None,
            size: tuple[int, int] = REACTION_IMG_SIZE) -> str:
    key = (tuple(reactants), tuple(products), tuple(reagents or []), size)
    if key in _rxn_cache:
        return _rxn_cache[key]
    url = reaction_smiles_to_image_base64(
        reactants=list(reactants),
        reagents=list(reagents or []),
        products=list(products),
        size=size,
    )
    _rxn_cache[key] = url
    return url


def mol_img(smiles: str, size: tuple[int, int] = MOL_IMG_SIZE) -> str:
    key = (smiles, size)
    if key in _mol_cache:
        return _mol_cache[key]
    url = smiles_to_image_base64(smiles, size=size)
    _mol_cache[key] = url
    return url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def smi_code(smi: str) -> str:
    return f'<code class="smi">{esc(smi)}</code>'


def split_reactants(s: str) -> list[str]:
    if not s:
        return []
    return [t for t in s.split(".") if t]


def img_or_text(url: str, alt: str, max_w: int = 640) -> str:
    if url:
        return f'<img src="{url}" alt="{esc(alt)}" style="max-width:{max_w}px;">'
    return f'<span class="muted">[绘图失败]</span>'


def reaction_block(reactants: list[str], products: list[str],
                   reagents: list[str] | None = None,
                   size: tuple[int, int] = REACTION_IMG_SIZE,
                   caption: str = "") -> str:
    url = rxn_img(reactants, products, reagents, size=size)
    cap = f'<div class="rxn-caption">{esc(caption)}</div>' if caption else ""
    return (
        f'<div class="rxn">'
        f'  {img_or_text(url, "reaction", max_w=size[0])}'
        f'  {cap}'
        f'</div>'
    )


def fmt_score(x: Any, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return esc(x)


def fmt_ms(ms: Any) -> str:
    try:
        ms = float(ms)
        if ms < 1000:
            return f"{ms:.0f} ms"
        return f"{ms/1000:.1f} s"
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Tool-result renderers
# ---------------------------------------------------------------------------

def render_neural_retro(result: dict) -> str:
    target = result.get("target", "")
    pec = result.get("per_engine_counts", {}) or {}
    pem = result.get("per_engine_ms", {}) or {}
    n_unique = result.get("n_unique_after_dedup", "?")
    preds = result.get("predictions") or []

    # Per-engine summary bar
    engines_html = []
    for eng in ("chembart", "rxngraphormer", "graph_retrosyn"):
        if eng in pec:
            engines_html.append(
                f'<span class="engine-pill"><b>{eng}</b>: {pec[eng]} 条 '
                f'<span class="muted">({fmt_ms(pem.get(eng))})</span></span>'
            )
    engines_bar = " ".join(engines_html)

    # Predictions table
    rows = []
    for i, p in enumerate(preds):
        reac = p.get("reactants", "")
        score = fmt_score(p.get("score"), 3)
        support = ", ".join(p.get("support") or [])
        bucket = p.get("bucket")
        cluster = p.get("cluster_size", "")
        rxn_html = reaction_block(
            reactants=split_reactants(reac),
            products=[target],
            size=(580, 170),
        )
        rows.append(
            f'<tr>'
            f'  <td class="rank">#{i+1}</td>'
            f'  <td class="rxn-cell">{rxn_html}'
            f'    <div class="smi-row">前体: {smi_code(reac)}</div></td>'
            f'  <td class="score-cell">'
            f'    <div class="big-score">{score}</div>'
            f'    <div class="muted">support: {esc(support)}</div>'
            f'    <div class="muted">bucket {esc(bucket)} · cluster {esc(cluster)}</div>'
            f'  </td>'
            f'</tr>'
        )

    return (
        f'<div class="result-block neural-retro">'
        f'  <div class="result-header">'
        f'    🧬 <b>neural_retro</b> <span class="en">(神经一步预测集成)</span>'
        f'    <span class="muted"> · 去重后 {esc(n_unique)} 条 · 共展示 {len(preds)} 条</span>'
        f'  </div>'
        f'  <div class="engines-bar">{engines_bar}</div>'
        f'  <table class="pred-table">{"".join(rows)}</table>'
        f'</div>'
    )


def render_suggest_templates(result: dict) -> str:
    product = result.get("product", "")
    sugg = result.get("suggestions") or []
    lib_size = result.get("library_size", "?")
    n_pre = result.get("candidates_after_prefilter", "?")

    rows = []
    for i, s in enumerate(sugg):
        reac = s.get("reactants_smiles") or ".".join(s.get("reactants") or [])
        kind = s.get("template_kind", "")
        src = s.get("template_source", "")
        rs = fmt_score(s.get("rank_score"), 3)
        mwb = fmt_score(s.get("mw_balance"), 3)
        kind_cls = "kind-enz" if kind == "enzymatic" else "kind-org"
        rxn_html = reaction_block(
            reactants=split_reactants(reac),
            products=[product],
            size=(580, 170),
        )
        rows.append(
            f'<tr>'
            f'  <td class="rank">#{i+1}</td>'
            f'  <td class="rxn-cell">{rxn_html}'
            f'    <div class="smi-row">前体: {smi_code(reac)}</div></td>'
            f'  <td class="score-cell">'
            f'    <div class="big-score">{rs}</div>'
            f'    <div class="muted">MW 平衡 {mwb}</div>'
            f'    <div><span class="kind-pill {kind_cls}">{esc(kind)}</span></div>'
            f'    <div class="muted">{esc(src)}</div>'
            f'  </td>'
            f'</tr>'
        )

    return (
        f'<div class="result-block suggest-templates">'
        f'  <div class="result-header">'
        f'    📚 <b>suggest_templates</b> <span class="en">(模板库匹配)</span>'
        f'    <span class="muted"> · 模板库 {esc(lib_size)} 条,预筛 {esc(n_pre)},取前 {len(sugg)}</span>'
        f'  </div>'
        f'  <table class="pred-table">{"".join(rows)}</table>'
        f'</div>'
    )


def render_explore_branches(arguments: dict, result: Any) -> str:
    product = arguments.get("product_smiles", "")
    branches = arguments.get("branches") or []
    # Some agent runs serialize `branches` as a JSON-encoded string; parse
    # if necessary so the renderer doesn't crash on `str.get(...)`.
    if isinstance(branches, str):
        try:
            branches = json.loads(branches)
        except Exception:
            branches = []
    if not isinstance(branches, list):
        branches = []
    rows = []
    for i, b in enumerate(branches):
        if not isinstance(b, dict):
            continue
        # Newer transcripts use `label`, older use `name`. Accept either.
        name = b.get("label") or b.get("name") or f"branch_{i+1}"
        reac = b.get("reactants") or []
        notes = b.get("notes") or ""
        rxn_html = reaction_block(
            reactants=list(reac),
            products=[product],
            size=(560, 170),
        )
        rows.append(
            f'<tr>'
            f'  <td class="rank">#{i+1}</td>'
            f'  <td class="rxn-cell">{rxn_html}'
            f'    <div class="smi-row">前体: {smi_code(".".join(reac))}</div></td>'
            f'  <td class="score-cell">'
            f'    <div class="branch-name">{esc(name)}</div>'
            f'    <div class="muted">{esc(notes)}</div>'
            f'  </td>'
            f'</tr>'
        )

    # If result contains ranked scoring, surface it
    rank_html = ""
    if isinstance(result, dict) and result.get("ranked"):
        rank_html = '<div class="muted">已按 score 重新排序</div>'

    return (
        f'<div class="result-block explore-branches">'
        f'  <div class="result-header">'
        f'    🌳 <b>explore_branches</b> <span class="en">(LLM 自定义分支)</span>'
        f'    <span class="muted"> · {len(branches)} 条候选</span>'
        f'  </div>'
        f'  {rank_html}'
        f'  <table class="pred-table">{"".join(rows)}</table>'
        f'</div>'
    )


def _render_one_buyability_entry(entry: dict, fallback_smi: str = "") -> str:
    """Render a single check_buyability per-mol record as a buy-row card."""
    can = (
        entry.get("canonical_smiles")
        or entry.get("smiles")
        or entry.get("input_smiles")
        or fallback_smi
        or ""
    )
    buyable = bool(entry.get("buyable"))
    stocks = entry.get("found_in_stocks") or []
    badge = (
        '<span class="badge buy-yes">✓ 商业可购</span>'
        if buyable
        else '<span class="badge buy-no">✗ 不在 stock</span>'
    )
    stock_html = (
        f'<div class="muted">stocks: {esc(", ".join(stocks))}</div>'
        if stocks
        else ""
    )
    return (
        f'<div class="result-block buyability {"buy-yes-bg" if buyable else "buy-no-bg"}">'
        f'  <div class="buy-row">'
        f'    {img_or_text(mol_img(can, size=(110, 84)), can, max_w=110)}'
        f'    <div class="buy-text">'
        f'      <div>{badge}</div>'
        f'      <div>{smi_code(can)}</div>'
        f'      {stock_html}'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )


def render_check_buyability(arguments: dict, result: dict) -> str:
    # Two call shapes coexist:
    #   (a) single  : args={smiles: "..."}, result={buyable, canonical_smiles, ...}
    #   (b) batched : args={smiles_list: [...]}, result={results: [{...}, ...]}
    # The original renderer only handled (a); under (b) `result.get("input_smiles")`
    # is empty and we drew an empty card. Branch on result shape.
    entries = result.get("results")
    if isinstance(entries, list) and entries:
        n_buyable = sum(1 for e in entries if e.get("buyable"))
        header = (
            f'<div class="result-header">'
            f'🛒 <b>check_buyability</b> '
            f'<span class="muted">· {n_buyable}/{len(entries)} buyable</span>'
            f'</div>'
        )
        in_smi = arguments.get("smiles_list") or []
        cards = []
        for i, entry in enumerate(entries):
            fallback = in_smi[i] if i < len(in_smi) else ""
            cards.append(_render_one_buyability_entry(entry, fallback_smi=fallback))
        return (
            f'<div class="result-block buyability-batch">'
            f'  {header}'
            f'  <div class="buy-grid">{"".join(cards)}</div>'
            f'</div>'
        )

    fallback = arguments.get("smiles") or ""
    return _render_one_buyability_entry(result or {}, fallback_smi=fallback)


# NOTE: kept for legacy transcripts that still contain `check_reaction`
# tool turns. The tool itself was removed (see commit removing
# tools/builtins/check_reaction.py); this renderer only fires when
# replaying old `.transcript.json` files.
def render_check_reaction(arguments: dict, result: dict) -> str:
    bal = result.get("balanced")
    bal_b = result.get("balanced_with_byproduct")
    diff = result.get("atom_diff") or {}
    bp = result.get("byproduct") or ""
    note = result.get("note") or ""

    if bal:
        badge = '<span class="badge ok">✓ 配平</span>'
    elif bal_b:
        badge = '<span class="badge warn">≈ 副产物配平</span>'
    else:
        badge = '<span class="badge bad">✗ 未配平</span>'

    diff_str = ", ".join(f"{k}:{v:+d}" for k, v in diff.items()) if diff else "无差异"

    return (
        f'<div class="result-block check-rxn">'
        f'  <div class="result-header">⚖️ <b>check_reaction</b> {badge}</div>'
        f'  <div class="muted">原子差: {esc(diff_str)} {("· 副产物 " + esc(bp)) if bp else ""}</div>'
        f'  <div class="rxn-note">{esc(note)}</div>'
        f'</div>'
    )


def render_score_route(arguments: dict, result: dict) -> str:
    raw_steps = arguments.get("route_steps")
    steps = []
    try:
        if isinstance(raw_steps, str):
            steps = json.loads(raw_steps)
        elif isinstance(raw_steps, list):
            steps = raw_steps
    except Exception:
        steps = []

    total = result.get("total_score") or result.get("agent_total_score")
    sub_keys = (
        "molecule_score", "route_score", "validity_score", "sa_score",
        "route_validity_score", "sm_availability_score",
        "step_penalty_score", "convergence_score_adjusted", "balance_score",
        "feasibility",
    )
    sub_html = []
    for k in sub_keys:
        if k in result:
            v = result[k]
            if isinstance(v, dict):
                inner = ", ".join(f"{ik}={fmt_score(iv, 3) if isinstance(iv, (int, float)) else esc(iv)}" for ik, iv in list(v.items())[:5])
                sub_html.append(f'<li><b>{esc(k)}</b>: {inner}</li>')
            else:
                sub_html.append(f'<li><b>{esc(k)}</b>: {fmt_score(v, 3)}</li>')

    step_html = []
    for i, s in enumerate(steps):
        reac = s.get("reactants") or []
        prod = s.get("product") or ""
        if isinstance(reac, str):
            reac = split_reactants(reac)
        rxn = reaction_block(reactants=reac, products=[prod], size=(560, 170))
        step_html.append(
            f'<div class="route-step">'
            f'  <div class="step-label">步 {i+1}</div>'
            f'  {rxn}'
            f'  <div class="smi-row">前体: {smi_code(".".join(reac))}</div>'
            f'  <div class="smi-row">产物: {smi_code(prod)}</div>'
            f'</div>'
        )

    return (
        f'<div class="result-block score-route">'
        f'  <div class="result-header">'
        f'    📊 <b>score_route</b> <span class="en">(综合评分)</span>'
        f'    <span class="big-score">total={fmt_score(total, 3)}</span>'
        f'  </div>'
        f'  <ul class="sub-scores">{"".join(sub_html)}</ul>'
        f'  <div class="route-steps">{"".join(step_html)}</div>'
        f'</div>'
    )


def render_save_best_route(arguments: dict, result: dict) -> str:
    label = arguments.get("label", "")
    steps = arguments.get("steps") or []
    sm = arguments.get("starting_materials") or []
    total = arguments.get("total_score") or (result.get("saved") or {}).get("total_score")
    notes = arguments.get("notes_excerpt") or arguments.get("notes") or ""

    step_html = []
    for i, s in enumerate(steps):
        # Step is a string like "PRODUCT >> REACTANTS (reaction: ...)" — try to parse
        m = re.match(r"^(.*?)\s*>>\s*(.*?)(?:\s*\(reaction:\s*(.+?)\))?$", s.strip())
        if m:
            prod, reac, rname = m.group(1), m.group(2), m.group(3) or ""
            rxn = reaction_block(reactants=split_reactants(reac), products=[prod], size=(560, 170),
                                 caption=rname)
            step_html.append(
                f'<div class="route-step">'
                f'  <div class="step-label">步 {i+1} <span class="muted">{esc(rname)}</span></div>'
                f'  {rxn}'
                f'</div>'
            )
        else:
            step_html.append(f'<div class="route-step muted">{esc(s)}</div>')

    sm_imgs = "".join(
        f'<div class="sm-cell">{img_or_text(mol_img(x), x, max_w=96)}<div class="smi-row">{smi_code(x)}</div></div>'
        for x in sm
    )

    notes_block = (f'<div class="notes muted">{esc(notes)}</div>' if notes else "")
    return (
        f'<div class="result-block save-route">'
        f'  <div class="result-header">'
        f'    💾 <b>save_best_route</b> "{esc(label)}"'
        f'    <span class="big-score">total={fmt_score(total, 3)}</span>'
        f'  </div>'
        f'  <div class="route-steps">{"".join(step_html)}</div>'
        f'  <div class="sm-section"><b>起始物</b><div class="sm-row">{sm_imgs}</div></div>'
        f'  {notes_block}'
        f'</div>'
    )


def render_list_best_routes(arguments: dict, result: dict) -> str:
    routes = (result or {}).get("routes") or []
    rows = []
    for r in routes:
        rows.append(
            f'<tr>'
            f'  <td>{esc(r.get("label"))}</td>'
            f'  <td class="big-score">{fmt_score(r.get("total_score"), 3)}</td>'
            f'  <td>{esc(r.get("num_steps"))} 步</td>'
            f'  <td>{esc(r.get("num_starting_materials"))} 起始物</td>'
            f'</tr>'
        )
    table = (
        '<table class="best-list"><tr><th>label</th><th>total</th><th>steps</th><th>SM</th></tr>'
        + "".join(rows) + '</table>'
    )
    return (
        f'<div class="result-block list-best">'
        f'  <div class="result-header">📋 <b>list_best_routes</b> · {len(routes)} 条</div>'
        f'  {table}'
        f'</div>'
    )


# Generic fallback
def render_generic_tool(name: str, arguments: dict, result: Any) -> str:
    args_s = json.dumps(arguments, ensure_ascii=False, default=str)
    res_s = json.dumps(result, ensure_ascii=False, default=str)
    if len(args_s) > 800:
        args_s = args_s[:800] + "…"
    if len(res_s) > 800:
        res_s = res_s[:800] + "…"
    return (
        f'<div class="result-block generic">'
        f'  <div class="result-header">🛠 <b>{esc(name)}</b></div>'
        f'  <div><b>args:</b> <code class="raw">{esc(args_s)}</code></div>'
        f'  <div><b>result:</b> <code class="raw">{esc(res_s)}</code></div>'
        f'</div>'
    )


def render_recall_similar_routes(arguments: dict, result: Any) -> str:
    """Visualise `recall_similar_routes` results, distinguishing the
    suppressed-prefetch stub from a real hit list.

    The stub shape is emitted by `agent._prefetch_disconnections` when the
    top-1 similarity falls below `precedent_prefetch_min_similarity`:
        {"suppressed": True, "reason": str, "max_similarity": float, "confidence": str}
    Anything else is the full payload from `services/precedent_db.query`
    (keys: query_canonical, confidence, n_examples, max_similarity, examples, ...).
    """
    if not isinstance(result, dict):
        return render_generic_tool("recall_similar_routes", arguments, result)
    q = arguments.get("query_smiles") or arguments.get("smiles") or ""
    q_html = f'<div class="muted">query: <code class="smi">{esc(q)}</code></div>' if q else ""
    if result.get("suppressed") is True:
        max_sim = result.get("max_similarity")
        try:
            max_sim_s = f"{float(max_sim):.2f}"
        except Exception:
            max_sim_s = str(max_sim)
        conf = result.get("confidence", "?")
        reason = result.get("reason", "")
        return (
            f'<div class="result-block recall-suppressed" '
            f'style="background:#fff5cd;border:1px solid #bf8700;border-radius:4px;padding:8px 10px;">'
            f'  <div class="result-header">'
            f'    🚫 <b>recall_similar_routes</b> '
            f'    <span class="badge warn">RAG prefetch suppressed</span>'
            f'    <span class="muted"> · top-1 sim {esc(max_sim_s)} · band={esc(conf)}</span>'
            f'  </div>'
            f'  {q_html}'
            f'  <div class="muted" style="margin-top:4px;">{esc(reason)}</div>'
            f'</div>'
        )
    conf = result.get("confidence", "?")
    max_sim = result.get("max_similarity")
    try:
        max_sim_s = f"{float(max_sim):.2f}"
    except Exception:
        max_sim_s = str(max_sim)
    examples = result.get("examples") or []
    badge_cls = "ok" if conf == "high" else ("warn" if conf == "medium" else "bad")
    rows = []
    for i, ex in enumerate(examples[:5]):
        rid = ex.get("route_id", "?")
        sim = ex.get("similarity")
        try:
            sim_s = f"{float(sim):.2f}"
        except Exception:
            sim_s = str(sim)
        src = ex.get("source_db", "?")
        nstep = ex.get("n_steps", "?")
        enz = " · enzymatic" if ex.get("is_enzymatic") else ""
        src_t = ex.get("source_target") or ""
        rows.append(
            f'<div class="muted">'
            f'  #{i+1} <b>{esc(str(rid))}</b> · sim {esc(sim_s)} · {esc(src)}{esc(enz)} · {esc(nstep)} steps'
            f'  · <code class="smi">{esc(src_t[:80])}</code>'
            f'</div>'
        )
    return (
        f'<div class="result-block recall-similar-routes">'
        f'  <div class="result-header">'
        f'    🧭 <b>recall_similar_routes</b> '
        f'    <span class="badge {badge_cls}">band={esc(conf)}</span>'
        f'    <span class="muted"> · top-1 sim {esc(max_sim_s)} · {len(examples)} hits</span>'
        f'  </div>'
        f'  {q_html}'
        f'  {"".join(rows)}'
        f'</div>'
    )


TOOL_RENDERERS = {
    "neural_retro": lambda a, r: render_neural_retro(r) if isinstance(r, dict) else render_generic_tool("neural_retro", a, r),
    "suggest_templates": lambda a, r: render_suggest_templates(r) if isinstance(r, dict) else render_generic_tool("suggest_templates", a, r),
    "explore_branches": lambda a, r: render_explore_branches(a, r),
    "check_buyability": lambda a, r: render_check_buyability(a, r) if isinstance(r, dict) else render_generic_tool("check_buyability", a, r),
    "check_reaction": lambda a, r: render_check_reaction(a, r) if isinstance(r, dict) else render_generic_tool("check_reaction", a, r),
    "score_route": lambda a, r: render_score_route(a, r) if isinstance(r, dict) else render_generic_tool("score_route", a, r),
    "save_best_route": lambda a, r: render_save_best_route(a, r) if isinstance(r, dict) else render_generic_tool("save_best_route", a, r),
    "list_best_routes": lambda a, r: render_list_best_routes(a, r) if isinstance(r, dict) else render_generic_tool("list_best_routes", a, r),
    "recall_similar_routes": render_recall_similar_routes,
}


def _render_full_args_panel(name: str, args: dict, result: Any) -> str:
    """Collapsed <details> with the full tool-call arguments + result JSON.

    Cherry-picked tool renderers (`render_neural_retro`, `render_score_route`,
    etc.) hide most of the LLM's input. Surfacing the raw payload here lets
    chemists audit *what* the agent actually asked the tool and *exactly*
    what came back, without scraping transcripts.
    """
    try:
        args_s = json.dumps(args, ensure_ascii=False, indent=2, default=str)
    except Exception:
        args_s = repr(args)
    try:
        result_s = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    except Exception:
        result_s = repr(result)
    # Soft cap on raw payloads so a single >1MB neural_retro stub doesn't
    # bloat the HTML; chemists who need more can grep the transcript.
    if len(result_s) > 12000:
        result_s = result_s[:12000] + f"\n… [truncated, total {len(result_s)} chars]"
    return (
        '<details class="raw-payload">'
        f'<summary class="muted">📋 {esc(name)} 完整参数 / 原始结果 '
        f'(args {len(args_s)}b, result {len(result_s)}b)</summary>'
        f'<div class="raw-payload-cols">'
        f'<div><b>arguments</b><pre class="raw">{esc(args_s)}</pre></div>'
        f'<div><b>result</b><pre class="raw">{esc(result_s)}</pre></div>'
        f'</div></details>'
    )


def render_tool_result(ev: dict) -> str:
    name = ev.get("name") or "?"
    args = ev.get("arguments") or {}
    result = ev.get("result")
    elapsed = ev.get("elapsed_ms")
    elapsed_html = f'<span class="muted"> · {fmt_ms(elapsed)}</span>' if elapsed else ""
    body = TOOL_RENDERERS.get(name, lambda a, r: render_generic_tool(name, a, r))(args, result)
    raw_panel = _render_full_args_panel(name, args, result)
    return f'<div class="tool-card">{elapsed_html}{body}{raw_panel}</div>'


# ---------------------------------------------------------------------------
# Final-route parsing
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(
    r"Step\s+(\d+):\s+(.+?)\s+>>\s+(.+?)(?:\s+\(reaction:\s+(.+?)\))?\s*$",
    re.IGNORECASE,
)


def parse_route_text(route_text: str) -> tuple[list[dict], list[str]]:
    """Returns (steps, starting_materials)."""
    if not route_text:
        return [], []
    lines = [ln.strip() for ln in route_text.splitlines() if ln.strip()]
    steps = []
    sms: list[str] = []
    for ln in lines:
        m = _STEP_RE.search(ln)
        if m:
            steps.append({
                "n": int(m.group(1)),
                "reactants_text": m.group(2).strip(),
                "product": m.group(3).strip(),
                "name": (m.group(4) or "").strip(),
            })
            continue
        if ln.lower().startswith("starting materials"):
            tail = ln.split(":", 1)[1] if ":" in ln else ""
            sms = [t.strip() for t in tail.split(",") if t.strip()]
    return steps, sms


def render_final_route(route_text: str) -> str:
    if not route_text:
        return '<div class="muted">本次未输出 COMPLETE 路径</div>'
    steps, sms = parse_route_text(route_text)
    if not steps:
        return f'<pre class="route-raw">{esc(route_text)}</pre>'

    blocks = []
    for s in steps:
        reac_parts = re.split(r"\s*\+\s*", s["reactants_text"])
        rxn = reaction_block(
            reactants=[x.strip() for x in reac_parts if x.strip()],
            products=[s["product"]],
            size=(720, 220),
            caption=s["name"],
        )
        blocks.append(
            f'<div class="route-step final">'
            f'  <div class="step-label">步 {s["n"]} '
            f'    <span class="muted">{esc(s["name"])}</span></div>'
            f'  {rxn}'
            f'</div>'
        )

    sm_imgs = "".join(
        f'<div class="sm-cell">{img_or_text(mol_img(x, size=(110, 90)), x, max_w=110)}<div class="smi-row">{smi_code(x)}</div></div>'
        for x in sms
    )
    sm_block = ""
    if sms:
        sm_block = (
            '<div class="sm-section"><b>起始物</b><div class="sm-row">'
            + sm_imgs
            + '</div></div>'
        )
    return (
        f'<div class="final-route">'
        f'  <div class="route-steps">{"".join(blocks)}</div>'
        f'  {sm_block}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Saved-routes renderer — show EVERY route the agent saved to scratchpad,
# sorted by total_score descending. Use this in preference to
# render_final_route, which only shows the rank-1 result and hides the rest
# of the search.
# ---------------------------------------------------------------------------

_SAVED_STEP_LINE_RE = re.compile(
    r"^\s*(?P<prod>.+?)\s*>>\s*(?P<reac>.+?)"
    r"(?:\s*\(reaction:\s*(?P<rname>.+?)\))?\s*$"
)


def _render_one_saved_route(rank: int, route: dict) -> str:
    """Render a single saved route as a card (rank badge + step images + SM grid)."""
    label = route.get("label") or f"route_{rank}"
    total = route.get("total_score")
    sub_scores = route.get("sub_scores") or {}
    raw_steps = route.get("steps") or []
    sm = route.get("starting_materials") or []
    notes = route.get("notes_excerpt") or route.get("notes") or ""

    step_html = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, str):
            step_html.append(f'<div class="route-step muted">{esc(s)}</div>')
            continue
        m = _SAVED_STEP_LINE_RE.match(s.strip())
        if m:
            prod = m.group("prod").strip()
            reac = m.group("reac").strip()
            rname = (m.group("rname") or "").strip()
            rxn = reaction_block(
                reactants=split_reactants(reac),
                products=[prod],
                size=(720, 220),
                caption=rname,
            )
            step_html.append(
                f'<div class="route-step final">'
                f'  <div class="step-label">步 {i+1}'
                f'    <span class="muted"> · {esc(rname)}</span></div>'
                f"  {rxn}"
                f'  <div class="smi-row">前体: {smi_code(reac)}</div>'
                f'  <div class="smi-row">产物: {smi_code(prod)}</div>'
                f"</div>"
            )
        else:
            step_html.append(f'<div class="route-step muted">{esc(s)}</div>')

    sm_imgs = "".join(
        f'<div class="sm-cell">'
        f'{img_or_text(mol_img(x, size=(110, 90)), x, max_w=110)}'
        f'<div class="smi-row">{smi_code(x)}</div>'
        f'</div>'
        for x in sm
    )
    sm_block = ""
    if sm:
        sm_block = (
            f'<div class="sm-section"><b>起始物</b>'
            f'<div class="sm-row">{sm_imgs}</div></div>'
        )

    sub_html_parts = []
    for k in ("feasibility", "cost", "greenness", "molecule_score",
              "route_score", "validity_score", "sa_score", "balance_score"):
        if k in sub_scores:
            v = sub_scores[k]
            if isinstance(v, dict):
                inner = v.get("score") if "score" in v else None
                if inner is not None:
                    sub_html_parts.append(
                        f'<span class="engine-pill"><b>{esc(k)}</b>: '
                        f'{fmt_score(inner, 3)}</span>'
                    )
            else:
                sub_html_parts.append(
                    f'<span class="engine-pill"><b>{esc(k)}</b>: '
                    f'{fmt_score(v, 3)}</span>'
                )
    sub_bar = (
        f'<div class="engines-bar">{" ".join(sub_html_parts)}</div>'
        if sub_html_parts else ""
    )

    notes_block = (
        f'<div class="notes muted">📝 {esc(notes)}</div>' if notes else ""
    )

    rank_badge = (
        f'<span class="banner ok">#{rank} 最佳</span>'
        if rank == 1
        else f'<span class="banner warn">#{rank}</span>'
    )

    return (
        f'<div class="header-card" style="margin-bottom:12px;">'
        f'  <div style="display:flex; justify-content:space-between; '
        f'              align-items:center; gap:12px; flex-wrap:wrap;">'
        f"    <div>"
        f"      {rank_badge} "
        f'      <b>"{esc(label)}"</b>'
        f'      <span class="muted"> · {len(raw_steps)} 步 · '
        f"{len(sm)} 起始物</span>"
        f"    </div>"
        f'    <div class="big-score">total = {fmt_score(total, 3)}</div>'
        f"  </div>"
        f"  {sub_bar}"
        f'  <div class="route-steps">{"".join(step_html)}</div>'
        f"  {sm_block}"
        f"  {notes_block}"
        f"</div>"
    )


def render_saved_routes(transcript: dict, route_text_fallback: str = "") -> str:
    """Render every saved route in `transcript.scratchpad.routes`, sorted by
    total_score descending.

    If the transcript has no scratchpad (older transcripts), fall back to
    `render_final_route(route_text_fallback)` so we still show the rank-1
    result instead of an empty page.
    """
    sp = transcript.get("scratchpad") or {}
    routes: list[dict] = list(sp.get("routes") or [])
    if not routes:
        if route_text_fallback:
            return (
                '<div class="muted" style="margin-bottom:8px;">'
                "本次未保存到 scratchpad,以下为最终输出的 rank-1 路径:"
                "</div>" + render_final_route(route_text_fallback)
            )
        return (
            '<div class="muted">本次未保存任何完整路径(scratchpad 为空)</div>'
        )

    def _score_key(r: dict) -> float:
        try:
            return float(r.get("total_score") or 0.0)
        except Exception:
            return 0.0

    routes_sorted = sorted(routes, key=_score_key, reverse=True)

    parts = [
        f'<div class="muted" style="margin-bottom:8px;">'
        f"📦 共保存 <b>{len(routes_sorted)}</b> 条完整路径,按 total_score 降序排列"
        f"</div>"
    ]
    for new_rank, r in enumerate(routes_sorted, start=1):
        parts.append(_render_one_saved_route(new_rank, r))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; padding: 12px 24px; background: #f7f8fa; color: #1f2328;
       max-width: 1400px; margin-left: auto; margin-right: auto; }
h1 { font-size: 18px; margin: 0 0 6px 0; }
h2 { font-size: 15px; margin: 12px 0 6px 0; border-bottom: 1px solid #d0d7de; padding-bottom: 3px; }
h3 { font-size: 13px; margin: 8px 0 4px 0; }
.muted { color: #6b7280; font-size: 11px; }
.en { color: #6b7280; font-size: 10px; font-style: italic; }
code.smi { background: #eef0f4; padding: 1px 3px; border-radius: 3px; font-size: 10.5px;
           word-break: break-all; }
code.raw { background: #f3f4f6; padding: 1px 3px; border-radius: 3px; font-size: 10.5px;
           word-break: break-all; display: inline-block; max-width: 100%; }
.header-card { background: white; border: 1px solid #d0d7de; border-radius: 6px;
               padding: 10px 12px; margin-bottom: 10px; }
.target-row { display: flex; gap: 12px; align-items: center; }
.banner { padding: 4px 10px; border-radius: 4px; font-weight: bold; display: inline-block;
          font-size: 12px; }
.banner.ok { background: #d1f4d9; color: #1a7f37; }
.banner.bad { background: #ffe5e5; color: #b91c1c; }
.banner.warn { background: #fff5cd; color: #9a6700; }
.stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; font-size: 12px; }
.stat-row span b { color: #0969da; }
details { background: white; border: 1px solid #d0d7de; border-radius: 5px;
          padding: 4px 10px; margin: 4px 0; }
details > summary { cursor: pointer; font-weight: 600; padding: 2px 0; font-size: 12.5px; }
details[open] { padding-bottom: 8px; }
.tool-card { margin: 6px 0; padding: 6px 8px; border-left: 3px solid #d0d7de;
             background: #fafbfc; border-radius: 0 4px 4px 0; }
.result-block { margin-top: 3px; }
.result-header { font-size: 12.5px; margin-bottom: 4px; }
.engines-bar { margin-bottom: 4px; }
.engine-pill { display: inline-block; background: #eef0f4; padding: 1px 6px;
               border-radius: 10px; margin-right: 4px; font-size: 10px; }
.pred-table { width: 100%; border-collapse: collapse; }
.pred-table tr { border-bottom: 1px dashed #e5e7eb; }
.pred-table td { padding: 3px 4px; vertical-align: top; }
.rank { width: 28px; color: #6b7280; font-weight: bold; font-size: 11px; }
.rxn-cell { width: 440px; }
.score-cell { font-size: 11px; min-width: 110px; }
.big-score { font-size: 14px; font-weight: bold; color: #0969da; }
.smi-row { font-size: 10px; margin-top: 2px; line-height: 1.3; }
.rxn-caption { font-size: 10px; color: #4b5563; margin-top: 1px; }
.kind-pill { padding: 1px 5px; border-radius: 3px; font-size: 9.5px; }
.kind-pill.kind-enz { background: #d1ecf1; color: #084c61; }
.kind-pill.kind-org { background: #fde2e4; color: #6d2727; }
.branch-name { font-weight: bold; color: #0969da; }
.buy-row { display: flex; gap: 8px; align-items: center; }
.buy-grid { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 3px; }
.buy-grid .result-block { margin: 0; flex: 0 0 auto; max-width: 240px; }
.buy-yes-bg { background: #d1f4d9; }
.buy-no-bg { background: #f3f4f6; }
.badge { padding: 1px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }
.badge.buy-yes { background: #1a7f37; color: white; }
.badge.buy-no { background: #6b7280; color: white; }
.badge.ok { background: #1a7f37; color: white; }
.badge.warn { background: #bf8700; color: white; }
.badge.bad { background: #b91c1c; color: white; }
.thinking { background: #fff8dc; border-left: 3px solid #f0c419;
            padding: 4px 8px; margin: 6px 0; font-style: italic; font-size: 11.5px; }
.iter-banner { font-weight: bold; color: #57606a; margin: 6px 0 3px 0;
               border-top: 1px solid #d0d7de; padding-top: 6px; font-size: 12px; }
.complete-block { background: #d1f4d9; border: 2px solid #1a7f37; padding: 8px 10px;
                  border-radius: 5px; margin: 8px 0; font-size: 12.5px; }
.blocked-block { background: #fff5cd; border: 2px solid #bf8700; padding: 6px 10px;
                 border-radius: 5px; margin: 6px 0; font-size: 12px; }
.route-step { margin: 6px 0; padding: 6px; background: white;
              border: 1px solid #e5e7eb; border-radius: 4px; }
.route-step.final { background: #f6fdf8; border-color: #1a7f37; }
.step-label { font-weight: bold; margin-bottom: 3px; font-size: 12px; }
.sm-section { margin-top: 6px; }
.sm-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.sm-cell { background: white; padding: 3px; border: 1px solid #e5e7eb;
           border-radius: 4px; text-align: center; max-width: 118px; }
.sub-scores { font-size: 11px; margin: 3px 0 6px 12px; padding: 0 0 0 6px; }
.best-list { width: 100%; border-collapse: collapse; font-size: 11.5px; }
.best-list th { background: #eef0f4; padding: 3px 4px; text-align: left; }
.best-list td { padding: 3px 4px; border-bottom: 1px dashed #e5e7eb; }
.notes { margin-top: 4px; }
pre.raw { background: #f3f4f6; padding: 6px; border-radius: 4px; font-size: 10.5px;
          white-space: pre-wrap; word-break: break-all; max-height: 340px;
          overflow: auto; margin: 3px 0; line-height: 1.35; }
.raw-payload { margin-top: 6px; border-top: 1px dashed #d0d7de; padding-top: 4px;
               background: transparent; border-left: none; border-right: none;
               border-bottom: none; border-radius: 0; padding-left: 0;
               padding-right: 0; padding-bottom: 0; }
.raw-payload > summary { cursor: pointer; user-select: none; padding: 3px 0;
                         font-weight: normal; font-size: 11px; }
.raw-payload-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
                    margin-top: 4px; }
@media (max-width: 900px) { .raw-payload-cols { grid-template-columns: 1fr; } }
"""


def is_success(transcript: dict) -> bool:
    res = transcript.get("result") or {}
    if res.get("success"):
        return True
    # JSONL transcripts won't have result; check events
    for ev in transcript.get("events", []):
        if ev.get("event") == "complete":
            return True
    return False


def header_html(transcript: dict, target: str, idx: int, label_zh: str,
                run_id: str) -> str:
    elapsed = transcript.get("elapsed_seconds")
    if not elapsed:
        # derive from last event ts_rel
        evs = transcript.get("events", [])
        if evs:
            elapsed = evs[-1].get("ts_rel")
    stats = transcript.get("stats") or {}
    counts = stats.get("tool_call_counts") or {}
    n_iter = stats.get("n_iterations")
    if not n_iter:
        n_iter = sum(1 for ev in transcript.get("events", []) if ev.get("event") == "iteration")

    success = is_success(transcript)
    banner = (
        '<span class="banner ok">✅ 找到完整路径</span>'
        if success
        else '<span class="banner bad">❌ 未输出完整路径</span>'
    )

    counts_html = ", ".join(f"{esc(k)}:{esc(v)}" for k, v in sorted(counts.items())) if counts else ""
    if not counts_html:
        # compute from events
        ev_counts: dict[str, int] = {}
        for ev in transcript.get("events", []):
            if ev.get("event") == "tool_call":
                ev_counts[ev.get("name", "?")] = ev_counts.get(ev.get("name", "?"), 0) + 1
        counts_html = ", ".join(f"{k}:{v}" for k, v in sorted(ev_counts.items()))

    return (
        f'<div class="header-card">'
        f'  <h1>idx {idx} · {esc(label_zh)} <span class="muted">({esc(run_id)})</span></h1>'
        f'  <div class="target-row">'
        f'    {img_or_text(mol_img(target, size=(160, 130)), target, max_w=160)}'
        f'    <div>'
        f'      <div>🎯 <b>目标 SMILES</b>: {smi_code(target)}</div>'
        f'      <div style="margin-top:8px;">{banner}</div>'
        f'      <div class="stat-row">'
        f'        <span>⏱ 耗时 <b>{esc(round(elapsed or 0, 1))} s</b></span>'
        f'        <span>🔁 迭代 <b>{esc(n_iter)}</b></span>'
        f'      </div>'
        f'      <div class="stat-row"><span class="muted">工具调用统计: {counts_html}</span></div>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )


def render_event(ev: dict) -> str:
    et = ev.get("event")
    if et == "thinking":
        return f'<div class="thinking">💭 {esc(ev.get("content"))}</div>'
    if et == "complete":
        return (
            f'<div class="complete-block">'
            f'  <b>🎯 COMPLETE</b>'
            f'  <pre style="white-space:pre-wrap; margin: 6px 0 0 0;">{esc(ev.get("content"))}</pre>'
            f'</div>'
        )
    if et == "complete_blocked_quality":
        return (
            f'<div class="blocked-block">'
            f'  ⚠️ <b>COMPLETE 被质量门拦截</b>'
            f'  <span class="muted"> · target threshold {esc(ev.get("target"))}</span>'
            f'  <div>{esc(ev.get("content"))}</div>'
            f'</div>'
        )
    if et == "checkpoint_compacted":
        return (
            f'<div class="muted">🗜 checkpoint compacted (demoted {esc(ev.get("demoted"))} 旧结果)</div>'
        )
    if et == "prefetch":
        return f'<div class="muted">⚡ prefetch: {esc(ev.get("tools"))}</div>'
    return ""


def render_timeline(events: list[dict]) -> str:
    """Group events by iteration into <details> blocks."""
    if not events:
        return ""

    # Group: events before first iteration go into "iter 0 (prefetch)"
    groups: list[tuple[str, list[dict]]] = []
    current_label = "迭代 0 · 启动 / prefetch"
    current: list[dict] = []
    pending_call: dict[str, dict] = {}  # tool_call_id -> call event

    for ev in events:
        et = ev.get("event")
        if et == "iteration":
            if current:
                groups.append((current_label, current))
            current = []
            current_label = f"迭代 {ev.get('n')}"
            continue
        if et == "session_start":
            continue
        # Pair tool_call with tool_result; only the result actually renders rich
        if et == "tool_call":
            tcid = ev.get("tool_call_id")  # may be None for some events
            if tcid:
                pending_call[tcid] = ev
            current.append(ev)
            continue
        if et == "tool_result":
            current.append(ev)
            continue
        # all other events
        current.append(ev)

    if current:
        groups.append((current_label, current))

    parts = []
    for label, group in groups:
        # Skip empty iteration shells
        non_iter = [e for e in group if e.get("event") not in ("iteration",)]
        if not non_iter:
            continue
        body_parts = []
        for ev in non_iter:
            et = ev.get("event")
            if et == "tool_result":
                body_parts.append(render_tool_result(ev))
            elif et == "tool_call":
                # Skip the call shell — its info is shown by the result
                continue
            else:
                body_parts.append(render_event(ev))
        # Brief inline summary in the summary line
        n_calls = sum(1 for e in non_iter if e.get("event") == "tool_call")
        n_thinking = sum(1 for e in non_iter if e.get("event") == "thinking")
        summary = f'{esc(label)} <span class="muted">({n_calls} 工具调用, {n_thinking} 思考)</span>'
        # Only auto-open the prefetch group
        is_prefetch = "启动" in label
        open_attr = " open" if is_prefetch else ""
        parts.append(
            f'<details{open_attr}><summary>{summary}</summary>{"".join(body_parts)}</details>'
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# Display labels for the case indices of one internal experimental batch. The
# experimental-validation target names are not shipped; supply a mapping
# locally or leave the labels blank.
LABELS_ZH = {}


def find_case_dir(run_id: str, idx: int) -> Path | None:
    root = RESULTS_ROOT / run_id
    prefix = f"{idx:02d}__"
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith(prefix):
            return p
    return None


def load_transcript(case_dir: Path, session_dir_name: str | None) -> dict | None:
    """Prefer transcript.json in case_dir; else read transcript.jsonl from session_dir."""
    tj = case_dir / "transcript.json"
    if tj.exists():
        with open(tj) as f:
            return json.load(f)
    if session_dir_name:
        sd = RUNS_ROOT / session_dir_name
        jsonl = sd / "transcript.jsonl"
        if jsonl.exists():
            target = None
            events = []
            with open(jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    if ev.get("event") == "session_start":
                        target = ev.get("target_smiles")
                        continue
                    events.append(ev)
            if target is None:
                return None
            elapsed = events[-1].get("ts_rel") if events else 0
            return {
                "target_smiles": target,
                "elapsed_seconds": elapsed,
                "result": {},
                "stats": {},
                "events": events,
            }
    return None


def render_case(idx: int, run_id: str, session_dir_name: str | None) -> Path | None:
    _reset_caches()
    case_dir = find_case_dir(run_id, idx)
    if case_dir is None:
        print(f"[skip] idx {idx}: case dir not found in {run_id}")
        return None
    t = load_transcript(case_dir, session_dir_name)
    if t is None:
        print(f"[skip] idx {idx}: no transcript at {case_dir} or {session_dir_name}")
        return None

    target = t.get("target_smiles", "")
    label_zh = LABELS_ZH.get(idx, "")
    head = header_html(t, target, idx, label_zh, run_id)

    route_text = ((t.get("result") or {}).get("route_text")) or ""
    if not route_text:
        # extract from events
        for ev in reversed(t.get("events", [])):
            if ev.get("event") == "complete":
                route_text = ev.get("content") or ""
                break
    # Show ALL saved routes sorted by total_score desc (not just rank-1).
    # Falls back to rank-1 from route_text if scratchpad is missing
    # (older hard_mol transcripts).
    final = render_saved_routes(t, route_text_fallback=route_text)

    timeline = render_timeline(t.get("events") or [])

    body = (
        f'{head}'
        f'<h2>🏆 全部保存路径(按 total_score 降序)</h2>{final}'
        f'<h2>📜 推理时间线</h2>{timeline}'
    )
    page = (
        f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f'<title>idx {idx} · {esc(label_zh)}</title>'
        f'<style>{CSS}</style></head><body>{body}</body></html>'
    )
    out = case_dir / "visualization.html"
    out.write_text(page, encoding="utf-8")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"[ok]  idx {idx:>2}: wrote {out} ({size_mb:.1f} MB)")
    return out


def main():
    out_paths = []
    for idx in sorted(RUN_MAP.keys()):
        run_id, session_dir_name = RUN_MAP[idx]
        p = render_case(idx, run_id, session_dir_name)
        if p:
            out_paths.append((idx, p))

    # index page
    rows = []
    for idx, p in out_paths:
        rel = p.relative_to(RESULTS_ROOT)
        rows.append(
            f'<li><a href="{esc(rel)}">idx {idx} · {esc(LABELS_ZH.get(idx, ""))}</a></li>'
        )
    index_html = (
        f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f'<title>hard_mol 11-case 可视化索引</title>'
        f'<style>{CSS}</style></head><body>'
        f'<h1>hard_mol 11-case 可视化索引</h1>'
        f'<ul>{"".join(rows)}</ul></body></html>'
    )
    idx_out = RESULTS_ROOT / "visualization_index.html"
    idx_out.write_text(index_html, encoding="utf-8")
    print(f"[ok]  index: {idx_out}")


if __name__ == "__main__":
    main()
