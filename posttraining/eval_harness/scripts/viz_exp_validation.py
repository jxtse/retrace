#!/usr/bin/env python
"""Render the experimental-validation results into a self-contained HTML report.

For each case: target image, success/score badges, and every ranked route drawn
as a sequence of reaction-step panels (product => precursors), with per-route
sub-scores. Pure RDKit SVG embedded inline — no matplotlib / external assets.

Usage:
  .venv/bin/python scripts/viz_exp_validation.py \
      --root final_results/exp_validation_opus48 [--max-routes 3]
"""
from __future__ import annotations
import argparse, html, json, re
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

def parse_step(step: str) -> tuple[str, list[str], str]:
    """Split 'product>>r1.r2 (reaction: note)' -> (product, [reactants], note).

    Reactant SMILES themselves contain '(' (e.g. ring/branch atoms), so we
    must NOT split on the first paren. The '(reaction: ...)' annotation is
    always a trailing parenthesised suffix — peel it off the END only.
    """
    s = step.strip()
    note = ""
    # Peel a trailing "(...)" group only if it looks like the reaction note.
    if s.endswith(")"):
        depth = 0
        cut = -1
        for i in range(len(s) - 1, -1, -1):
            c = s[i]
            if c == ")":
                depth += 1
            elif c == "(":
                depth -= 1
                if depth == 0:
                    cut = i
                    break
        if cut >= 0:
            inner = s[cut + 1:-1]
            if inner.lower().startswith("reaction:") or "reaction" in inner.lower()[:12]:
                note = re.sub(r"^reaction:\s*", "", inner.strip(), flags=re.I)
                s = s[:cut].strip()
    prod, _, reacts = s.partition(">>")
    reactants = [r.strip() for r in reacts.split(".") if r.strip()]
    return prod.strip(), reactants, note


def mol_svg(smiles: str, w: int = 220, h: int = 170) -> str:
    m = Chem.MolFromSmiles(smiles.strip())
    if m is None:
        return f'<div class="bad">invalid SMILES<br><code>{html.escape(smiles[:40])}</code></div>'
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.drawOptions().padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return d.GetDrawingText()


def render_step(step: str) -> str:
    """One retro step string -> an HTML panel: product => [precursor mols]."""
    prod, reacts, note = parse_step(step)
    if not prod or not reacts:
        return f'<div class="step"><pre>{html.escape(step)}</pre></div>'
    prod_html = f'<div class="mol"><div class="cap">product</div>{mol_svg(prod)}</div>'
    react_html = "".join(
        f'<div class="mol"><div class="cap">precursor {i+1}</div>{mol_svg(r)}</div>'
        for i, r in enumerate(reacts)
    )
    note_html = f'<div class="rxnnote">{html.escape(note)}</div>' if note else ""
    return (
        '<div class="step">'
        f'{prod_html}<div class="arrow">&#8658;</div>{react_html}'
        f'{note_html}</div>'
    )


def score_badge(label: str, val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    cls = "g" if v >= 0.7 else ("m" if v >= 0.5 else "r")
    return f'<span class="badge {cls}">{html.escape(label)} {v:.3f}</span>'


def render_route(route: dict, rank: int) -> str:
    label = route.get("label", f"route-{rank}")
    ss = route.get("sub_scores", {}) if isinstance(route.get("sub_scores"), dict) else {}

    def pick(*keys):
        for k in keys:
            if ss.get(k) is not None:
                return ss[k]
        return None

    total = route.get("total_score", ss.get("total_score"))
    badges = score_badge("total", total) \
        + score_badge("feas", pick("feasibility_score", "feasibility")) \
        + score_badge("cost", pick("cost_score", "cost")) \
        + score_badge("green", pick("greenness_score", "greenness"))
    steps = route.get("steps") or []
    sms = route.get("starting_materials") or []
    steps_html = "".join(render_step(s if isinstance(s, str) else json.dumps(s)) for s in steps)
    sm_html = "".join(f'<div class="mol sm"><div class="cap">SM</div>{mol_svg(s)}</div>' for s in sms)
    note = route.get("notes_excerpt") or route.get("notes") or ""
    note_html = f'<div class="rnote">{html.escape(str(note))}</div>' if note else ""
    return (
        f'<div class="route"><div class="rhead"><b>#{rank} {html.escape(str(label))}</b> '
        f'<span class="nsteps">{len(steps)} steps</span> {badges}</div>'
        f'{note_html}'
        f'<div class="steps">{steps_html}</div>'
        f'<div class="sms"><div class="smlabel">Starting materials:</div>{sm_html}</div>'
        '</div>'
    )


def load_full_routes(cdir: Path, res: dict) -> list[dict]:
    """Prefer the transcript scratchpad routes (full multi-step schema with
    `steps`/`sub_scores`); fall back to the slimmer result.json `routes`."""
    tj = cdir / "transcript.json"
    if tj.exists():
        try:
            t = json.loads(tj.read_text())
            rs = (t.get("scratchpad") or {}).get("routes") or []
            full = [r for r in rs if isinstance(r, dict) and r.get("steps")]
            if full:
                full.sort(key=lambda r: r.get("rank", 1e9))
                return full
        except Exception:
            pass
    return [r for r in (res.get("routes") or []) if isinstance(r, dict)]


def render_case(res: dict, max_routes: int, cdir: Path) -> str:
    idx = res.get("idx")
    smiles = res.get("target_canon") or res.get("smiles", "")
    ok = res.get("success")
    has_routes = bool(res.get("n_routes"))
    if ok:
        badge = '<span class="ok">SUCCESS — all leaves buyable</span>'
    elif has_routes:
        badge = '<span class="warn">routes found — some leaves not in catalog</span>'
    else:
        badge = '<span class="fail">no complete route</span>'
    meta = (f'MW {res.get("mw")} &middot; {res.get("natoms")} atoms &middot; '
            f'{res.get("nrings")} rings &middot; {res.get("n_routes")} routes &middot; '
            f'{res.get("time_s")}s &middot; {res.get("iterations")} iters')
    err = res.get("error")
    err_html = f'<div class="err">ERROR: {html.escape(str(err))}</div>' if err else ""
    routes = load_full_routes(cdir, res)[:max_routes]
    routes_html = "".join(render_route(r, i + 1) for i, r in enumerate(routes))
    if not routes_html:
        routes_html = '<div class="norts">No ranked routes recorded.</div>'
    return (
        f'<section class="case"><h2>Case {idx} {badge}</h2>'
        f'<div class="meta">{meta}</div>{err_html}'
        f'<div class="target"><div class="cap">target</div>{mol_svg(smiles, 320, 240)}'
        f'<div class="tsmiles"><code>{html.escape(smiles)}</code></div></div>'
        f'{routes_html}</section>'
    )


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f6f8;color:#1a1a1a}
header{background:#1f2937;color:#fff;padding:18px 28px}
header h1{margin:0;font-size:20px} header .sub{opacity:.8;font-size:13px;margin-top:4px}
.case{background:#fff;margin:18px 22px;padding:18px 22px;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.case h2{margin:0 0 4px;font-size:18px} .meta{color:#666;font-size:12px;margin-bottom:10px}
.ok{background:#10b981;color:#fff;padding:2px 9px;border-radius:10px;font-size:12px;vertical-align:middle}
.fail{background:#9ca3af;color:#fff;padding:2px 9px;border-radius:10px;font-size:12px;vertical-align:middle}
.warn{background:#f59e0b;color:#fff;padding:2px 9px;border-radius:10px;font-size:12px;vertical-align:middle}
.target{border:1px dashed #cbd5e1;border-radius:8px;padding:10px;margin:8px 0 14px;display:inline-block;text-align:center}
.tsmiles{font-size:10px;color:#555;max-width:520px;word-break:break-all;margin-top:4px}
.route{border:1px solid #e5e7eb;border-radius:8px;margin:12px 0;padding:10px 12px;background:#fafbfc}
.rhead{font-size:14px;margin-bottom:6px} .nsteps{color:#666;font-size:12px;margin:0 6px}
.rnote{font-size:12px;color:#444;margin:4px 0 8px;font-style:italic}
.badge{display:inline-block;padding:1px 7px;border-radius:8px;font-size:11px;margin-left:5px;color:#fff}
.badge.g{background:#10b981}.badge.m{background:#f59e0b}.badge.r{background:#ef4444}
.steps{display:flex;flex-direction:column;gap:8px}
.step{display:flex;align-items:center;flex-wrap:wrap;gap:6px;background:#fff;border:1px solid #eee;border-radius:6px;padding:6px}
.mol{text-align:center} .mol svg{display:block}
.cap{font-size:10px;color:#888;text-transform:uppercase;letter-spacing:.5px}
.arrow{font-size:26px;color:#2563eb;margin:0 4px}
.rxnnote{flex-basis:100%;font-size:11px;color:#2563eb;padding-left:4px}
.sms{margin-top:8px;display:flex;align-items:center;flex-wrap:wrap;gap:6px}
.smlabel{font-size:12px;color:#444;font-weight:600;margin-right:6px}
.sm svg{border:1px solid #d1fae5;border-radius:4px}
.bad{color:#ef4444;font-size:11px;padding:20px} .err{color:#b91c1c;font-size:13px;margin:6px 0}
.norts{color:#999;font-size:13px;padding:8px}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="final_results/exp_validation_opus48")
    ap.add_argument("--max-routes", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "report.html"

    cases = []
    for cdir in sorted(root.glob("case_*"), key=lambda p: int(p.name.split("_")[1])):
        rj = cdir / "result.json"
        if rj.exists() and rj.stat().st_size > 0:
            try:
                cases.append((json.loads(rj.read_text()), cdir))
            except Exception as e:
                print(f"skip {rj}: {e}")
    n_ok = sum(1 for c, _ in cases if c.get("success"))
    n_routed = sum(1 for c, _ in cases if c.get("n_routes"))
    body = "".join(render_case(c, args.max_routes, cdir) for c, cdir in cases)
    model = cases[0][0].get("model") if cases else "?"
    sub = (f"model={model} &middot; {n_routed}/{len(cases)} produced ranked routes "
           f"({n_ok}/{len(cases)} fully buyable-terminated) &middot; top {args.max_routes} routes/case")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>Experimental-validation retrosynthesis</title><style>{CSS}</style></head>"
           f"<body><header><h1>Experimental-validation molecules — retrosynthesis routes</h1>"
           f"<div class='sub'>{sub}</div></header>{body}</body></html>")
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(cases)} cases, {n_ok} success)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
