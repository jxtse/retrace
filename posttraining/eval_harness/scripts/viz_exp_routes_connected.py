#!/usr/bin/env python
"""Chemist-facing report: each KEPT route as a CONNECTED FORWARD synthesis scheme.

Reads `final_results/exp_validation_opus48/case_{i}/transcript.json` (target + the
10 scratchpad routes) and an optional `review_verdicts.json` (from the finder→
verifier Workflow). For every KEPT route it builds the route tree from the retro
steps and lays it out left→right as a FORWARD synthesis (starting materials →
target), numbered steps, convergent couplings shown merging. Excluded routes go to
a per-case "可能有误的路径" appendix with the offending step + reason.

Pure RDKit inline-SVG (no matplotlib/cairosvg/graphviz). Output: report_routes.html.

  .venv/bin/python scripts/viz_exp_routes_connected.py [--root DIR] [--self-check]
"""
from __future__ import annotations
import argparse, html, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from viz_exp_validation import parse_step, mol_svg, score_badge  # noqa: E402

from rdkit import Chem  # noqa: E402


def canon(s: str) -> str:
    if not s:
        return ""
    m = Chem.MolFromSmiles(s) or Chem.MolFromSmiles(s.replace("[H]", ""))
    return Chem.MolToSmiles(m) if m is not None else s


# ---------------------------------------------------------------------------
# Route tree: map canon(product) -> (reaction, [canon reactants]); root = target;
# leaves = SMILES that are never a product. Full recursive walk into ALL branches.
# ---------------------------------------------------------------------------
class TreeNode:
    __slots__ = ("smiles", "stage", "incoming_rxn", "children", "is_leaf", "is_target")

    def __init__(self, smiles):
        self.smiles = smiles
        self.stage = 0
        self.incoming_rxn = None  # reaction that PRODUCES this node (forward view)
        self.children = []        # precursors (the molecules that react to form this)
        self.is_leaf = False
        self.is_target = False


def build_route_tree(target_canon: str, steps: list[str]):
    """Return (root, ok). root.children are precursors; recursion bottoms at leaves.

    Each step "P >> R1.R2 (reaction: name)" => prod_map[canon(P)] = (name, [canon Ri]).
    Forward synthesis: a node's `children` are its precursors; the reaction that
    converts those precursors into the node is `node.incoming_rxn`.
    """
    prod_map = {}
    for s in steps:
        prod, reacts, note = parse_step(s)
        pc = canon(prod)
        prod_map[pc] = (note or "reaction", [canon(r) for r in reacts])

    root = TreeNode(target_canon)
    root.is_target = True
    seen = set()

    def expand(node: TreeNode, depth: int):
        if depth > 64 or node.smiles in seen:
            return
        seen.add(node.smiles)
        entry = prod_map.get(node.smiles)
        if not entry:
            node.is_leaf = True
            return
        rxn, reactants = entry
        node.incoming_rxn = rxn
        for r in reactants:
            child = TreeNode(r)
            node.children.append(child)
            expand(child, depth + 1)

    expand(root, 0)

    # forward stage = longest path from any leaf up to this node
    def stage_of(node: TreeNode) -> int:
        if not node.children:
            return 0
        node.stage = 1 + max(stage_of(c) for c in node.children)
        return node.stage

    stage_of(root)
    ok = canon(root.smiles) == target_canon and bool(prod_map)
    return root, ok


def linearize(root: TreeNode):
    """Collect the forward reaction steps in synthesis order (leaves first).

    Returns a list of dicts: {step_no, reaction, product, precursors:[smiles],
    stage}. Convergent steps keep all precursors together. Ordered by product
    stage ascending so the chemist reads SM→target top to bottom.
    """
    steps = []

    def visit(node: TreeNode):
        for c in node.children:
            visit(c)
        if node.children:  # an internal node = a forward reaction producing `node`
            steps.append({
                "reaction": node.incoming_rxn or "reaction",
                "product": node.smiles,
                "precursors": [c.smiles for c in node.children],
                "stage": node.stage,
                "is_target": node.is_target,
            })

    visit(root)
    # de-dup (a shared intermediate could be visited twice) by product SMILES,
    # keep first; then order by stage so synthesis flows SM -> target.
    seen, uniq = set(), []
    for s in steps:
        if s["product"] in seen:
            continue
        seen.add(s["product"])
        uniq.append(s)
    uniq.sort(key=lambda s: s["stage"])
    for i, s in enumerate(uniq, 1):
        s["step_no"] = i
    return uniq


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def mol_card(smiles: str, role: str, highlight: bool = False) -> str:
    cls = "mol target" if highlight else "mol"
    w, h = (300, 220) if highlight else (210, 160)
    return (f'<div class="{cls}"><div class="cap">{html.escape(role)}</div>'
            f'{mol_svg(smiles, w, h)}</div>')


# Common reagents/solvents the condition predictor emits as SMILES. Chemists
# read names, not SMILES, so we caption the drawn structure with a familiar name
# where we recognise the canonical SMILES; otherwise the drawing alone carries it.
_REAGENT_NAMES = {
    "O=CO": "甲酸 (formic acid)",
    "O=C(O)O": "碳酸 (carbonic acid)",
    "CC(=O)OC(C)=O": "乙酸酐 (acetic anhydride)",
    "CC(=O)O": "乙酸 (acetic acid)",
    "ClCCl": "二氯甲烷 (DCM)",
    "ClC(Cl)Cl": "氯仿 (CHCl3)",
    "CO": "甲醇 (MeOH)",
    "CCO": "乙醇 (EtOH)",
    "C1CCOC1": "四氢呋喃 (THF)",
    "CC#N": "乙腈 (MeCN)",
    "CN(C)C=O": "DMF",
    "CS(C)=O": "DMSO",
    "O": "水 (H2O)",
    "CCOC(C)=O": "乙酸乙酯 (EtOAc)",
    "c1ccccc1": "苯 (benzene)",
    "Cc1ccccc1": "甲苯 (toluene)",
    "[OH-].[Na+]": "氢氧化钠 (NaOH)",
    "[Na+].[OH-]": "氢氧化钠 (NaOH)",
    "Cl": "盐酸 (HCl)",
    "CCN(CC)CC": "三乙胺 (Et3N)",
}

# Multi-component coupling reagents / salts the predictor emits as one
# `.`-joined SMILES that RDKit often can't canonicalise (hypervalent P/B
# counter-ions). Matched on a characteristic substructure token.
_REAGENT_SUBSTR_NAMES = [
    ("ON1N=NC2=CC=CC=C21", "HBTU/TBTU 型偶联试剂"),
    ("ON1N=NC2=C1N=CC=C2", "HATU (偶联试剂)"),
    ("P(F)(F)(F)(F)F", "六氟磷酸盐偶联试剂 (PF6⁻)"),
    ("[B-](F)(F)F", "四氟硼酸盐 (BF4⁻)"),
    ("C(=NC1CCCCC1)=NC1CCCCC1", "DCC (碳二亚胺)"),
    ("CCN=C=NCCCN(C)C", "EDC (碳二亚胺)"),
    ("c1ccc(P(c2ccccc2)c2ccccc2)cc1", "三苯基膦/磷类试剂"),
]


def _reagent_name_for(s: str, cs: str) -> str | None:
    """Map a (possibly multi-component / non-canonicalisable) reagent SMILES to
    a chemist-readable name, by exact canonical match then substructure token."""
    name = _REAGENT_NAMES.get(cs) or _REAGENT_NAMES.get(s)
    if name:
        return name
    for token, label in _REAGENT_SUBSTR_NAMES:
        if token in s or (cs and token in cs):
            return label
    return None


def safe_mol_svg(smiles: str, w: int, h: int) -> str | None:
    """mol_svg(), but returns None instead of the ``invalid SMILES`` <code>
    fallback — so a reagent SMILES can NEVER leak as text into the report.
    Some tokens pass RDKit parse (canon ok) yet fail draw prep; catch both."""
    if not smiles:
        return None
    try:
        svg = mol_svg(smiles, w, h)
    except Exception:
        return None
    if 'class="bad"' in svg or "invalid SMILES" in svg:
        return None
    return svg


def cond_reagent_html(label: str, val: str) -> str:
    """Render one solvent/reagent/catalyst value as a small molecule drawing
    (chemist-readable) instead of a raw SMILES string. Non-structural values
    such as ``unknown`` / ``none`` fall back to a short text tag."""
    s = str(val).strip()
    low = s.lower()
    if not s or low in ("unknown", "none", "n/a", "na", "-"):
        return f'<div class="condcell"><div class="condlbl">{label}</div>' \
               f'<div class="condna">未指定</div></div>'
    cs = canon(s)
    name = _reagent_name_for(s, cs)
    name_html = f'<div class="condname">{html.escape(name)}</div>' if name else ""
    # Preferred: draw the whole reagent structure.
    whole = safe_mol_svg(cs, 120, 80) if cs else None
    if whole:
        return (f'<div class="condcell"><div class="condlbl">{label}</div>'
                f'{whole}{name_html}</div>')
    # Not drawable as a whole (e.g. hypervalent counter-ion salt). Draw the
    # parseable `.`-components so the chemist still sees structures — and a name
    # if we recognised the reagent. NEVER fall through to a raw SMILES.
    parts = [p for p in s.split(".") if p.strip()]
    drawn = [svg for p in parts if (svg := safe_mol_svg(canon(p), 90, 64))]
    if drawn:
        body = "".join(drawn)
        if not name_html:
            name_html = '<div class="condname">多组分试剂</div>'
        return (f'<div class="condcell"><div class="condlbl">{label}</div>'
                f'<div class="condparts">{body}</div>{name_html}</div>')
    # Last resort: a labelled placeholder, never the SMILES string itself.
    return (f'<div class="condcell"><div class="condlbl">{label}</div>'
            f'<div class="condna">{html.escape(name) if name else "复杂试剂"}</div></div>')


def render_step_prediction(pred: dict | None) -> str:
    if not pred:
        return ""
    cond = pred.get("condition") or {}
    yld = pred.get("yield_prediction") or {}
    # Scalar (non-structural) condition facts render as text chips …
    cond_bits = []
    if cond.get("temperature") is not None:
        try:
            cond_bits.append(f"<b>T</b> {float(cond['temperature']):.0f} &deg;C")
        except (TypeError, ValueError):
            cond_bits.append(f"<b>T</b> {html.escape(str(cond.get('temperature')))}")
    if cond.get("score") is not None:
        try:
            cond_bits.append(f"<b>score</b> {float(cond['score']):.2f}")
        except (TypeError, ValueError):
            cond_bits.append(f"<b>score</b> {html.escape(str(cond.get('score')))}")
    # … while solvent/reagent/catalyst (SMILES from the predictor) render as
    # drawn structures — never raw SMILES (chemist deliverable constraint).
    cond_mols = []
    for label, key in (("溶剂", "solvent"), ("试剂", "reagent"), ("催化剂", "catalyst")):
        val = cond.get(key)
        if val:
            cond_mols.append(cond_reagent_html(label, val))
    if not cond_bits and not cond_mols:
        err = cond.get("error") or pred.get("condition_error")
        cond_html = f"未得到条件预测{': ' + html.escape(str(err)) if err else ''}"
    else:
        chips = f'<div class="condchips">{" · ".join(cond_bits)}</div>' if cond_bits else ""
        mols = f'<div class="condmols">{"".join(cond_mols)}</div>' if cond_mols else ""
        cond_html = chips + mols

    if yld.get("yield_percent") is not None:
        try:
            yield_html = f"{float(yld['yield_percent']):.1f}%"
        except (TypeError, ValueError):
            yield_html = html.escape(str(yld.get("yield_percent")))
    else:
        model = yld.get("model")
        prefix = f"{model}: " if model else ""
        yield_html = html.escape(prefix + (yld.get("note") or "未接入数值产率模型"))

    return (f'<div class="predrow"><div><span>条件预测</span>{cond_html}</div>'
            f'<div><span>产率预测</span>{yield_html}</div></div>')


def mol_thumb(smiles: str, role: str, highlight: bool = False) -> str:
    """A compact molecule thumbnail for the at-a-glance full-path overview strip."""
    cls = "thumb target" if highlight else "thumb"
    w, h = (150, 110) if highlight else (118, 90)
    return (f'<div class="{cls}">{mol_svg(smiles, w, h)}'
            f'<div class="tcap">{html.escape(role)}</div></div>')


def render_path_overview(fsteps: list[dict]) -> str:
    """One horizontal strip showing the WHOLE route at a glance: every starting
    material / intermediate / target as a small thumbnail, joined by reaction
    arrows in synthesis order (SM -> ... -> target). Convergent precursors that
    are themselves made in an earlier step are not repeated; only the *new*
    precursors entering at each step are shown, then the step's product."""
    if not fsteps:
        return ""
    # products made along the route — these are intermediates, not fresh inputs.
    made = {s["product"] for s in fsteps}
    nodes = []  # (smiles, role, highlight)
    seen_inputs = set()
    # Leaf starting materials of step 1 (precursors that are never a product).
    first = fsteps[0]
    for p in first["precursors"]:
        if p not in made and p not in seen_inputs:
            nodes.append((p, "起始原料", False))
            seen_inputs.add(p)
    cells = ["".join(f'<div class="ovnode">{mol_thumb(*n)}</div>' for n in nodes)]
    # Walk each step: arrow (with rxn name), any NEW co-reactants entering here,
    # then the product (target highlighted at the end).
    for i, s in enumerate(fsteps):
        is_last = i == len(fsteps) - 1
        co = [p for p in s["precursors"]
              if p not in made and p not in seen_inputs]
        co_html = ""
        for p in co:
            seen_inputs.add(p)
            co_html += (f'<div class="ovplus">+</div>'
                        f'<div class="ovnode">{mol_thumb(p, "试剂/原料", False)}</div>')
        arrow = (f'<div class="ovarrow"><div class="ovrxn">'
                 f'{html.escape(s["reaction"])}</div><div class="ovline">&#10230;</div>'
                 f'<div class="ovstep">Step {s["step_no"]}</div></div>')
        prod = mol_thumb(s["product"],
                         "目标分子" if (is_last or s.get("is_target")) else "中间体",
                         highlight=bool(is_last or s.get("is_target")))
        cells.append(co_html + arrow + f'<div class="ovnode">{prod}</div>')
    return (f'<div class="overview"><div class="ovtitle">完整路径一览</div>'
            f'<div class="ovstrip">{"".join(cells)}</div></div>')


def render_forward_step(s: dict, pred: dict | None = None) -> str:
    """One forward reaction row: [precursor + precursor ...] --rxn--> product."""
    precs = "".join(
        (('<div class="plus">+</div>' if i else '') +
         mol_card(p, f"原料/中间体" if False else "reactant"))
        for i, p in enumerate(s["precursors"])
    )
    arrow = (f'<div class="fwd"><div class="stepno">Step {s["step_no"]}</div>'
             f'<div class="rxnname">{html.escape(s["reaction"])}</div>'
             f'<div class="arrowline">&#10230;</div></div>')
    prod = mol_card(s["product"], "TARGET（目标分子）" if s.get("is_target") else "product",
                    highlight=s.get("is_target", False))
    return (f'<div class="fstep">{precs}{arrow}{prod}{render_step_prediction(pred)}</div>')


def coalesce_scores(route: dict) -> dict:
    ss = route.get("sub_scores", {}) if isinstance(route.get("sub_scores"), dict) else {}

    def pick(*ks):
        for k in ks:
            if route.get(k) is not None:
                return route[k]
            if ss.get(k) is not None:
                return ss[k]
        return None

    return {
        "total": route.get("total_score", ss.get("total_score")),
        "feasibility": pick("feasibility_score", "feasibility"),
        "cost": pick("cost_score", "cost"),
        "greenness": pick("greenness_score", "greenness"),
    }


def render_route_block(
    route: dict,
    target_canon: str,
    anchor: str,
    route_predictions: dict | None = None,
) -> tuple[str, bool]:
    label = route.get("label", "route")
    rank = route.get("rank", "?")
    sc = coalesce_scores(route)
    badges = (score_badge("total", sc["total"]) + score_badge("feas", sc["feasibility"])
              + score_badge("cost", sc["cost"]) + score_badge("green", sc["greenness"]))
    note = route.get("notes_excerpt") or route.get("notes") or ""
    steps = route.get("steps", [])
    root, ok = build_route_tree(target_canon, steps)
    fsteps = linearize(root) if ok else []
    nstart = len(route.get("starting_materials", []))
    step_preds = (route_predictions or {}).get("steps", {})
    overview = render_path_overview(fsteps)
    body = "".join(render_forward_step(s, step_preds.get(str(s["step_no"]))) for s in fsteps)
    if body:
        # Full path is shown at a glance; per-step detail (conditions, yields,
        # full-size structures) is collapsed by default — expand to inspect.
        detail = (f'<details class="stepdetails"><summary>展开逐步详情'
                  f'（反应条件 · 产率 · 大图）</summary>'
                  f'<div class="forward">{body}</div></details>')
    else:
        overview = ""
        detail = ('<div class="warnrow">⚠ 无法从步骤重建连通路径（步骤可能不构成到目标的'
                  '连通树）；下面按原始步骤列出。</div>'
                  + "".join(f'<div class="rawstep">{html.escape(s)}</div>' for s in steps))
    note_html = f'<div class="rnote">Agent 备注：{html.escape(str(note))}</div>' if note else ""
    blk = (f'<div class="route" id="{anchor}">'
           f'<div class="rhead"><b>#{rank} {html.escape(str(label))}</b>'
           f'<span class="nsteps">{len(fsteps) or len(steps)} 步 · '
           f'{nstart} 个起始原料</span>{badges}</div>'
           f'{note_html}{overview}{detail}</div>')
    return blk, ok


def render_excluded_block(route: dict, verdict: dict) -> str:
    label = route.get("label", "route")
    rank = route.get("rank", "?")
    defect_step = verdict.get("defect_step") or ""
    reason = verdict.get("reason") or ""
    dclass = verdict.get("defect_class") or ""
    conf = verdict.get("confidence") or ""
    step_panel = ""
    if defect_step:
        prod, reacts, note = parse_step(defect_step)
        prec = "".join(('<div class="plus">+</div>' if i else '') + mol_card(r, "reactant")
                       for i, r in enumerate(reacts))
        step_panel = (f'<div class="fstep">{prec}'
                      f'<div class="fwd"><div class="rxnname">{html.escape(note)}</div>'
                      f'<div class="arrowline">&#10230;</div></div>'
                      f'{mol_card(prod, "product")}</div>')
    return (f'<details class="excl"><summary>#{rank} {html.escape(str(label))} '
            f'<span class="xbadge">排除 · {html.escape(dclass)} · {html.escape(conf)}</span>'
            f'</summary><div class="xreason"><b>原因：</b>{html.escape(reason)}</div>'
            f'<div class="xstep"><b>问题步骤：</b></div>{step_panel}</details>')


def render_case(case_id: str, case: dict, verdicts: dict, include_excluded: bool, predictions: dict):
    target = case["target_smiles"]
    target_canon = case.get("target_canon") or canon(target)
    routes = case["routes"]
    cv = verdicts.get(case_id, {})
    kept_html, excl_html = [], []
    kept = 0
    for r in routes:
        v = cv.get(str(r["rank"]), {"keep": True})
        if v.get("keep", True):
            route_predictions = (((predictions.get("cases") or {}).get(case_id) or {})
                                 .get("routes", {}).get(str(r["rank"])))
            blk, _ = render_route_block(
                r, target_canon, f"{case_id}-r{r['rank']}", route_predictions
            )
            kept_html.append(blk)
            kept += 1
        else:
            excl_html.append(render_excluded_block(r, v))
    cnum = case_id.split("_")[1]
    # Chemist deliverable: molecule images only — no raw SMILES strings anywhere
    # in the body (per reviewer request). The target is shown as a drawn structure;
    # the SMILES caption is intentionally omitted.
    head = (f'<section class="case"><h2>Case {cnum}'
            f'<span class="kept">{kept}/{len(routes)} 条路径保留</span></h2>'
            f'<div class="target-hero"><div class="cap">目标分子</div>'
            f'{mol_svg(target_canon, 360, 260)}</div>')
    appendix = ""
    if include_excluded and excl_html:
        appendix = (f'<div class="appendix"><h3>可能有误的路径（已排除，{len(excl_html)} 条）'
                    f'</h3>{"".join(excl_html)}</div>')
    return head + "".join(kept_html) + appendix + "</section>", kept, len(excl_html)


CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',sans-serif;margin:0;background:#eef1f4;color:#1a1a1a}
header{background:#0f2942;color:#fff;padding:20px 30px}
header h1{margin:0;font-size:21px} header .sub{opacity:.85;font-size:13px;margin-top:5px}
.case{background:#fff;margin:20px;padding:20px 26px;border-radius:12px;box-shadow:0 1px 5px rgba(0,0,0,.09)}
.case h2{margin:0 0 12px;font-size:19px;border-bottom:2px solid #e5e7eb;padding-bottom:8px}
.kept{font-size:13px;color:#059669;font-weight:600;margin-left:12px}
.target-hero{text-align:center;border:1px dashed #94a3b8;border-radius:10px;padding:14px;margin-bottom:18px;background:#f8fafc}
.tsmiles{font-size:10px;color:#555;max-width:560px;margin:6px auto 0;word-break:break-all}
.route{border:1px solid #e5e7eb;border-radius:10px;margin:14px 0;padding:14px 16px;background:#fcfcfd}
.rhead{font-size:15px;margin-bottom:8px}
.nsteps{color:#666;font-size:12px;margin:0 10px}
.rnote{font-size:12.5px;color:#374151;margin:6px 0 12px;font-style:italic;background:#f1f5f9;padding:6px 10px;border-radius:6px;border-left:3px solid #94a3b8}
.badge{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;margin-left:6px;color:#fff}
.badge.g{background:#10b981}.badge.m{background:#f59e0b}.badge.r{background:#ef4444}
.overview{margin:8px 0 4px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px}
.ovtitle{font-size:12px;font-weight:700;color:#0f766e;margin-bottom:6px}
.ovstrip{display:flex;align-items:center;flex-wrap:wrap;gap:4px;overflow-x:auto;padding-bottom:4px}
.ovnode{display:inline-flex}
.thumb{text-align:center;background:#fff;border:1px solid #eef0f2;border-radius:6px;padding:3px 4px}
.thumb svg{display:block;margin:0 auto}
.thumb.target{border:2px solid #059669}
.tcap{font-size:9px;color:#94a3b8;margin-top:1px}
.thumb.target .tcap{color:#059669;font-weight:700}
.ovplus{font-size:18px;color:#64748b;font-weight:700;padding:0 1px}
.ovarrow{display:flex;flex-direction:column;align-items:center;min-width:78px;padding:0 3px}
.ovrxn{font-size:10px;color:#1e40af;text-align:center;max-width:110px;line-height:1.15;margin-bottom:1px}
.ovline{font-size:22px;color:#2563eb;line-height:.8}
.ovstep{font-size:9px;font-weight:700;color:#2563eb;background:#dbeafe;border-radius:7px;padding:0 6px;margin-top:1px}
.stepdetails{margin-top:8px}
.stepdetails>summary{cursor:pointer;font-size:12.5px;font-weight:600;color:#2563eb;background:#eff6ff;border:1px solid #bfdbfe;border-radius:7px;padding:6px 12px;list-style:none;user-select:none}
.stepdetails>summary::-webkit-details-marker{display:none}
.stepdetails>summary::before{content:'▸ ';color:#2563eb}
.stepdetails[open]>summary::before{content:'▾ '}
.stepdetails[open]>summary{margin-bottom:10px}
.forward{display:flex;flex-direction:column;gap:14px;margin-top:6px}
.fstep{display:flex;align-items:center;flex-wrap:wrap;gap:8px;background:#fff;border:1px solid #eef0f2;border-radius:8px;padding:10px 12px}
.mol{text-align:center} .mol svg{display:block;margin:0 auto}
.mol.target svg{border:2px solid #059669;border-radius:6px}
.cap{font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.4px;margin-bottom:2px}
.mol.target .cap{color:#059669;font-weight:700}
.plus{font-size:24px;color:#64748b;font-weight:600;padding:0 2px}
.fwd{display:flex;flex-direction:column;align-items:center;min-width:130px;padding:0 6px}
.stepno{font-size:11px;font-weight:700;color:#2563eb;background:#dbeafe;border-radius:8px;padding:1px 8px}
.rxnname{font-size:11.5px;color:#1e40af;text-align:center;margin:3px 0;max-width:160px}
.arrowline{font-size:30px;color:#2563eb;line-height:.9}
.predrow{flex-basis:100%;display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:4px;font-size:12px;color:#334155}
.predrow>div{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:6px 8px}
.predrow span{display:inline-block;font-weight:700;color:#0f766e;margin-right:8px}
.condchips{margin-bottom:4px}
.condmols{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end}
.condcell{text-align:center;background:#fff;border:1px solid #e2e8f0;border-radius:6px;padding:3px 5px}
.condcell svg{display:block;margin:0 auto}
.condlbl{font-size:10px;font-weight:700;color:#0f766e;text-transform:none}
.condname{font-size:10px;color:#475569;max-width:130px;word-break:break-word}
.condtext{font-size:11px;color:#334155;padding:6px 2px}
.condna{font-size:11px;color:#94a3b8;padding:6px 2px}
.condparts{display:flex;flex-wrap:wrap;gap:3px;justify-content:center}
.enrichnote{font-size:12.5px;color:#334155;background:#ecfeff;border:1px solid #a5f3fc;border-radius:8px;padding:8px 10px;margin:0 20px 14px}
.enrichnote b{color:#0f766e}
.warnrow{color:#b45309;font-size:12.5px;background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:6px 10px}
.rawstep{font-family:ui-monospace,monospace;font-size:10.5px;color:#475569;padding:2px 0;word-break:break-all}
.appendix{margin-top:18px;border-top:2px dashed #e5e7eb;padding-top:12px}
.appendix h3{font-size:15px;color:#b91c1c;margin:0 0 10px}
.excl{border:1px solid #fecaca;border-radius:8px;margin:8px 0;padding:6px 12px;background:#fef2f2}
.excl summary{cursor:pointer;font-size:14px;font-weight:600;color:#7f1d1d}
.xbadge{background:#dc2626;color:#fff;font-size:11px;padding:1px 8px;border-radius:8px;margin-left:8px;font-weight:500}
.xreason{font-size:13px;margin:8px 0;color:#374151}
.xstep{font-size:12px;color:#7f1d1d;margin-top:6px}
"""


def load_cases(root: Path) -> dict:
    cases = {}
    for cdir in sorted(root.glob("case_*"), key=lambda p: int(p.name.split("_")[1])):
        tj = cdir / "transcript.json"
        if not tj.exists():
            continue
        t = json.loads(tj.read_text())
        routes = sorted((t.get("scratchpad") or {}).get("routes", []),
                        key=lambda r: r.get("rank", 1e9))
        cases[cdir.name] = {
            "target_smiles": t.get("target_smiles", ""),
            "target_canon": canon(t.get("target_smiles", "")),
            "routes": routes,
        }
    return cases


def load_verdicts(root: Path) -> dict:
    vp = root / "review_verdicts.json"
    if not vp.exists():
        return {}
    raw = json.loads(vp.read_text())
    return {k: v for k, v in raw.items() if k.startswith("case_")}


def load_predictions(root: Path, path: str | None) -> dict:
    pp = Path(path) if path else root / "step_predictions.json"
    if not pp.exists():
        return {}
    raw = json.loads(pp.read_text())
    raw["_path"] = str(pp)
    return raw


def self_check(root: Path, cases: dict, verdicts: dict) -> int:
    problems = []
    # 1. SMILES parse
    nfail = ntot = 0
    for cid, c in cases.items():
        if Chem.MolFromSmiles(c["target_canon"]) is None:
            problems.append(f"{cid}: target unparseable")
        for r in c["routes"]:
            for s in r["steps"]:
                p, rs, _ = parse_step(s)
                for smi in [p] + rs:
                    ntot += 1
                    if Chem.MolFromSmiles(smi) is None and Chem.MolFromSmiles(smi.replace("[H]", "")) is None:
                        nfail += 1
                        problems.append(f"{cid} r{r['rank']}: unparseable {smi[:40]}")
    print(f"[check] SMILES: {ntot} parsed, {nfail} failures")
    # 2. verdict sanity
    if verdicts:
        nverd = sum(len(v) for v in verdicts.values())
        print(f"[check] verdicts: {nverd} entries")
        if nverd != sum(len(c['routes']) for c in cases.values()):
            problems.append(f"verdict count {nverd} != route count")
        nexcl = 0
        for cid, cv in verdicts.items():
            steps_by_rank = {str(r["rank"]): r["steps"] for r in cases[cid]["routes"]}
            for rk, v in cv.items():
                if "keep" not in v:
                    problems.append(f"{cid} r{rk}: no keep field")
                if not v.get("keep", True):
                    nexcl += 1
                    if not v.get("reason"):
                        problems.append(f"{cid} r{rk}: excluded w/o reason")
                    ds = v.get("defect_step")
                    if ds and ds not in steps_by_rank.get(rk, []):
                        problems.append(f"{cid} r{rk}: defect_step not in route steps (hallucinated?)")
        nroutes = sum(len(c['routes']) for c in cases.values())
        print(f"[check] excluded: {nexcl}/{nroutes} ({100*nexcl/nroutes:.0f}%)")
        if nexcl > 0.3 * nroutes:
            print(f"[check] WARN: >30% excluded — review may be too aggressive for a KEEP-biased rubric")
    else:
        print("[check] no verdicts file (all routes will be KEPT)")
    # 3. tree well-formedness per route
    nbad_tree = 0
    for cid, c in cases.items():
        for r in c["routes"]:
            _, ok = build_route_tree(c["target_canon"], r["steps"])
            if not ok:
                nbad_tree += 1
                problems.append(f"{cid} r{r['rank']}: route tree not rooted at target")
    print(f"[check] route trees: {nbad_tree} not-well-formed")
    if problems:
        print(f"[check] {len(problems)} PROBLEM(S):")
        for p in problems[:40]:
            print("   -", p)
        return 1
    print("[check] ALL CHECKS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "final_results/exp_validation_opus48"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--predictions", default=None,
                    help="Optional step enrichment JSON from enrich_exp_validation_predictions.py")
    ap.add_argument("--no-excluded", dest="include_excluded", action="store_false", default=True)
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    cases = load_cases(root)
    verdicts = load_verdicts(root)
    predictions = load_predictions(root, args.predictions)

    if args.self_check:
        return self_check(root, cases, verdicts)

    out = Path(args.out) if args.out else root / "report_routes.html"
    body_parts, tot_kept, tot_excl = [], 0, 0
    for cid, c in cases.items():
        part, kept, excl = render_case(cid, c, verdicts, args.include_excluded, predictions)
        body_parts.append(part)
        tot_kept += kept
        tot_excl += excl
    nroutes = sum(len(c["routes"]) for c in cases.values())
    sub = (f"{len(cases)} 个分子 · 共 {nroutes} 条候选路径 · "
           f"{tot_kept} 条通过化学审查保留 · {tot_excl} 条疑似有误（见各 case 末尾附录） · "
           f"正向合成视图（起始原料 → 目标分子）")
    meta = predictions.get("meta") or {}
    enrich = ""
    if predictions:
        selected_yield = meta.get("yield_selected_model") or "未接入数值模型"
        yield_note = meta.get("yield_note") or "host-d 文档未提供产率 API"
        enrich = (f"<div class='enrichnote'><b>后处理增强：</b>"
                  f"每步已加入反应条件预测（{html.escape(str(meta.get('condition_source', 'condition predictor')))})；"
                  f"产率评分模型已比较并选择 {html.escape(str(selected_yield))}；"
                  f"{html.escape(str(yield_note))}。</div>")
    doc = (f"<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
           f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>实验验证分子 — 合成路径交付</title><style>{CSS}</style></head>"
           f"<body><header><h1>实验验证分子 — 候选合成路径</h1>"
           f"<div class='sub'>{sub}</div></header>{enrich}{''.join(body_parts)}</body></html>")
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(cases)} cases, {tot_kept} kept routes, {tot_excl} excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
