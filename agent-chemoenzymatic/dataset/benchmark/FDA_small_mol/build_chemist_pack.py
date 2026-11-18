#!/usr/bin/env python
"""
Build a chemist-friendly markdown report per successful agent run, then zip
the whole bundle for sharing.

Output layout:
  chemist_pack/
    README.md                          # index + headline numbers
    by_target/{idx:02d}_{Name}/
        {variant}.md                   # human-readable transcript
  chemist_pack.zip
"""
from __future__ import annotations

import json
import re
import shutil
import textwrap
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RUNS = BENCH / "runs"
TR = BENCH / "transcripts"
OUT = BENCH / "chemist_pack"
ZIP_PATH = BENCH / "chemist_pack.zip"

CLEARED_RE = re.compile(r"\[Old tool result cleared by microCompact[^\]]*\]")
PREFETCH_FAIL_RE = re.compile(r'\{"error":\s*"prefetch failed[^"]*"\}')
TOOLCALL_RE = re.compile(r"^\[tool_call\]\s+(\w+)\((.*)\)\s*$", re.S)


def parse_transcript(txt: str) -> list[dict]:
    """Split a transcript .txt into role-tagged blocks."""
    blocks: list[dict] = []
    cur_role = None
    cur_buf: list[str] = []
    for line in txt.splitlines():
        m = re.match(r"^=+\s*(\w+)\s*=+\s*$", line)
        if m:
            if cur_role is not None:
                blocks.append({"role": cur_role, "text": "\n".join(cur_buf).strip()})
            cur_role = m.group(1).lower()
            cur_buf = []
        else:
            cur_buf.append(line)
    if cur_role is not None:
        blocks.append({"role": cur_role, "text": "\n".join(cur_buf).strip()})
    return blocks


def pretty_args(name: str, raw: str, max_len: int = 280) -> str:
    """Format tool arguments — pretty-print JSON, truncate long SMILES lists."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        # Trim very long string values
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str) and len(v) > 200:
                    obj[k] = v[:180] + f"…(+{len(v)-180})"
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = raw
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s


def is_useful_tool_result(text: str) -> bool:
    if not text.strip():
        return False
    if CLEARED_RE.search(text):
        return False
    if PREFETCH_FAIL_RE.search(text):
        return False
    return True


def summarise_tool_result(text: str, max_len: int = 600) -> str:
    """Trim tool result to a chemist-readable snippet (try JSON pretty-print)."""
    # Strip trailing [tool_result for ...] tag
    text = re.sub(r"\n?\[tool_result for [^\]]+\]\s*$", "", text).strip()
    try:
        obj = json.loads(text)
        s = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        s = text
    if len(s) > max_len:
        s = s[:max_len] + f"\n…(+{len(s)-max_len} chars)"
    return s


def render_assistant(text: str) -> list[str]:
    """Convert an assistant block (free text + [tool_call] lines) to markdown lines."""
    out: list[str] = []
    thinking_lines: list[str] = []
    tool_calls: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = TOOLCALL_RE.match(line.strip())
        if m:
            tool_calls.append((m.group(1), m.group(2)))
        else:
            thinking_lines.append(line)
    thinking = "\n".join(thinking_lines).strip()
    if thinking:
        out.append(f"**🧠 Reasoning:**  \n{thinking}")
    for name, args in tool_calls:
        out.append(f"**🔧 Tool call:** `{name}({pretty_args(name, args)})`")
    return out


def build_markdown(record: dict, transcript: str) -> str:
    idx = record["idx"]
    name = record["name"]
    variant = record["variant"]
    smiles = record["smiles"]
    iters = record.get("iterations")
    wall = record.get("wall_seconds")
    below = record.get("below_target")
    route = (record.get("route_text") or "").strip()
    ref_step = record.get("ref_step") or ""
    cb_step = record.get("chembart_step") or ""

    md: list[str] = []
    md.append(f"# {idx:02d} · {name} ({variant})")
    md.append("")
    md.append(f"- **Target SMILES:** `{smiles}`")
    md.append(f"- **Iterations used:** {iters}")
    md.append(f"- **Wall time:** {wall} s")
    md.append(f"- **Score gate:** {'⚠️ below target (score < 0.7)' if below else '✅ score ≥ 0.7'}")
    if ref_step:
        md.append(f"- **Reference (literature) steps:** {ref_step}")
    if cb_step:
        md.append(f"- **ChemBart MCTS steps:** {cb_step}")
    md.append("")
    md.append("## Final route")
    md.append("")
    if route:
        md.append("```")
        md.append(route.strip())
        md.append("```")
    else:
        md.append("_no route text available_")
    md.append("")
    md.append("## Agent execution narrative")
    md.append("")
    md.append("> Below is a cleaned chronological transcript. Repetitive `[Old tool result cleared by microCompact]` stubs (the agent's internal context-compaction garbage-collection) have been suppressed. Tool arguments and results are truncated where long.")
    md.append("")

    blocks = parse_transcript(transcript)
    iter_no = 0
    for i, b in enumerate(blocks):
        role, text = b["role"], b["text"]
        if role == "system":
            continue
        if role == "user" and i <= 2:
            # initial user prompt
            md.append("### User → agent")
            md.append("")
            md.append(f"_{text}_")
            md.append("")
            continue
        if role == "user":
            # Inter-turn nudge from the orchestrator — keep short version
            short = text.split("\n")[0][:240]
            if short.strip():
                md.append(f"<sub>(orchestrator hint: {short})</sub>")
                md.append("")
            continue
        if role == "assistant":
            iter_no += 1
            md.append(f"### Step {iter_no} — Assistant")
            md.append("")
            for line in render_assistant(text):
                md.append(line)
                md.append("")
            continue
        if role == "tool":
            if not is_useful_tool_result(text):
                continue
            snippet = summarise_tool_result(text)
            md.append("**📦 Tool result:**")
            md.append("")
            md.append("```json")
            md.append(snippet)
            md.append("```")
            md.append("")
            continue
    return "\n".join(md) + "\n"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    successes: list[dict] = []
    for jf in sorted(RUNS.glob("*.json")):
        rec = json.loads(jf.read_text(encoding="utf-8"))
        if not rec.get("success"):
            continue
        successes.append(rec)
        idx = rec["idx"]
        name = rec["name"]
        variant = rec["variant"]
        # safe-name dir matches transcript filename
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60]
        tag = f"{idx:02d}_{safe}_{variant}"
        tr_file = TR / f"{tag}.txt"
        transcript = tr_file.read_text(encoding="utf-8") if tr_file.exists() else ""
        target_dir = OUT / "by_target" / f"{idx:02d}_{safe}"
        target_dir.mkdir(parents=True, exist_ok=True)
        md = build_markdown(rec, transcript)
        (target_dir / f"{variant}.md").write_text(md, encoding="utf-8")

    # Headline / index
    index = []
    index.append("# Retrosynthesis Agent — Successful-route bundle for chemist review\n")
    index.append("This bundle contains every **successful** retrosynthesis run from the "
                 "2025 FDA small-molecule benchmark, in human-readable form. Each target "
                 "was run on two SMILES variants:\n")
    index.append("- `stereo` — original SMILES with stereochemistry preserved")
    index.append("- `canonical` — stereochemistry stripped (matches the input ChemBart MCTS used)\n")
    index.append("## Headline")
    index.append("")
    n_targets = len({r["idx"] for r in successes})
    n_stereo = sum(1 for r in successes if r["variant"] == "stereo")
    n_canon = sum(1 for r in successes if r["variant"] == "canonical")
    n_gold_stereo = sum(1 for r in successes if r["variant"] == "stereo" and not r.get("below_target"))
    n_gold_canon = sum(1 for r in successes if r["variant"] == "canonical" and not r.get("below_target"))
    index.append(f"- Targets covered (≥1 successful variant): **{n_targets} / 30**")
    index.append(f"- Successful runs: **{len(successes)} / 60**  (stereo {n_stereo}/30, canonical {n_canon}/30)")
    index.append(f"- Of those, score-gate-passing (`agent_total_score ≥ 0.7`): "
                 f"stereo {n_gold_stereo}/{n_stereo}, canonical {n_gold_canon}/{n_canon}")
    index.append("")
    index.append("## How to read each report")
    index.append("")
    index.append("Open any file under `by_target/{idx}_{Name}/{variant}.md`. Structure:")
    index.append("")
    index.append("1. **Header** — target SMILES, iteration count, wall time, score-gate verdict, reference step count vs ChemBart")
    index.append("2. **Final route** — the proposed multi-step disconnection")
    index.append("3. **Execution narrative** — chronological assistant reasoning, tool calls, and (truncated) tool results")
    index.append("")
    index.append("## Per-target index")
    index.append("")
    index.append("| idx | Name | Variants delivered |")
    index.append("|---:|---|---|")
    by_idx: dict[int, dict] = {}
    for r in successes:
        by_idx.setdefault(r["idx"], {"name": r["name"], "variants": set()})["variants"].add(r["variant"])
    for idx in sorted(by_idx.keys()):
        info = by_idx[idx]
        vs = ", ".join(sorted(info["variants"]))
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in info["name"])[:60]
        index.append(f"| {idx} | [{info['name']}](by_target/{idx:02d}_{safe}/) | {vs} |")
    index.append("")
    index.append("## Definitions")
    index.append("")
    index.append("- *score gate*: `agent_total_score ≥ 0.7`. Sub-scores include synthesizability, atom economy, enzymatic preference, convergence (binding excluded).")
    index.append("- *iterations*: number of LLM ↔ tool turns the agent took.")
    index.append("- *microCompact stubs* (suppressed in the narrative): the agent's internal mechanism for reclaiming context window — old tool results are removed and re-fetched on demand. They have no chemical meaning.")
    index.append("")
    (OUT / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    # Zip it
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=OUT.parent, base_dir=OUT.name)
    print(f"wrote {ZIP_PATH}")
    print(f"  ({sum(1 for _ in OUT.rglob('*.md'))} markdown files, "
          f"{ZIP_PATH.stat().st_size/1024/1024:.2f} MiB)")


if __name__ == "__main__":
    main()
