#!/usr/bin/env python3
"""Frozen audit@k pipeline driver (v1.0). See AUDIT_SPEC.md.

Stage 0: deterministic gate (already annotated in export; enforced here).
Stage 1: deep-read audit via claude-opus-5 @ :8030 (blinded, inline route).
Stage 2: adversarial defense of FAKE verdicts (same judge, fresh context).

Resumable: every verdict is appended to verdicts.jsonl keyed by uid; on
restart, already-judged uids are skipped. Concurrency via thread pool.

Usage:
  python3 audit_driver.py --routes audit_routes_rerun4.json --out verdicts.jsonl \
      [--limit-cases 20] [--conds opus48,gpt56std] [--workers 6]
"""
import argparse, json, os, re, sys, threading, time
import datetime
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "http://localhost:8030/v1/chat/completions"
MODEL = "claude-opus-5"
TEMP = 0.2
MAXTOK = 32000

DEFECTS = ["NONE", "TARGET_NOT_PRODUCED", "WRONG_RING_SIZE",
           "FABRICATED_OR_MISSING_SKELETAL_CARBON", "WRONG_REGIO_OR_CONSTITUTIONAL",
           "IMPOSSIBLE_OR_NONSENSE_REACTION", "WRONG_REAGENT_OR_CONDITIONS",
           "STEREO_ONLY", "NON_BUYABLE_LEAF", "DISCONNECTED_TREE", "OTHER", "ERROR"]

DEEPREAD_PROMPT = """You are a senior synthetic/medicinal chemist auditing ONE retrosynthetic route for scientific validity.

TARGET (SMILES): {target}

ROUTE ({n_steps} steps, written as retro-SMILES "PRODUCT>>PRECURSORS"; deterministic RDKit annotations in brackets):
{steps_block}

RDKit facts (deterministic, TRUST them): every step parsed OK; the target IS produced by some step. Per-step [Cdelta=+N] is product_carbons - sum(precursor_carbons). [ring gained/lost] lists ring sizes appearing/disappearing in the product vs precursors.

Judge each step as a real organic (or enzymatic) reaction converting the PRECURSORS into the PRODUCT. Reason about each step independently as precursors->product.

For EACH step decide VALID / SUSPECT / FAKE:
- FAKE = the transformation cannot give the stated product from the stated precursors. Defect codes:
  * IMPOSSIBLE_OR_NONSENSE_REACTION: no known mechanism does this transformation.
  * FABRICATED_OR_MISSING_SKELETAL_CARBON: skeletal carbons appear/vanish with no carbon-bearing reagent to account for them (use Cdelta - but a nonzero Cdelta is FINE if a simple reagent legitimately supplies/removes those carbons, e.g. formaldehyde, MeI, Boc2O, CO2 loss, succinimide leaving as a unit).
  * WRONG_RING_SIZE: a cycloaddition/annulation yields a ring of the wrong size (check ring gained/lost and Cdelta).
  * WRONG_REGIO_OR_CONSTITUTIONAL: product is a constitutional/regio isomer of what the reaction would actually give.
  * WRONG_REAGENT_OR_CONDITIONS: the implied reagent class cannot effect the change.
- SUSPECT = plausible but a real concern (low-yield, chemoselectivity not obvious, unusual but not impossible). Use STEREO_ONLY for stereochemistry-only doubts.
- VALID = a competent chemist would accept this step.

A carbon-conserved step can still be FAKE (conservation does not prove mechanism). Be skeptical but fair: do not fail a step merely for being enzymatic, protecting-group-heavy, or low-yield.

Route verdict aggregation (mechanical): FAKE if ANY step is FAKE; else SUSPECT if any step is SUSPECT; else VALID. worst_defect_code = most serious step defect (NONE if VALID).

Reply with ONLY a JSON object, no markdown fence, matching:
{{"route_verdict": "VALID|SUSPECT|FAKE", "worst_defect_code": "<code>", "n_fake_steps": <int>, "n_suspect_steps": <int>, "steps": [{{"step_index": <int>, "verdict": "VALID|SUSPECT|FAKE", "defect_code": "<code>", "reason": "<short>"}}], "summary": "<2 sentences>"}}"""

ADVERSARY_PROMPT = """You are a skeptical chemist asked to REFUTE a FAKE verdict - steelman the route. Another auditor called this retrosynthetic route FAKE.

TARGET (SMILES): {target}

ROUTE ({n_steps} steps, retro-SMILES "PRODUCT>>PRECURSORS"):
{steps_block}

STEPS FLAGGED FAKE by the first auditor:
{flagged_block}

Find the strongest case that the flagged step(s) are achievable real chemistry (allow enzymatic/biocatalytic, multi-component, or non-obvious but literature-precedented transformations). Decide honestly: does the FAKE verdict hold (fake_upheld=true) or can the step(s) be defended (fake_upheld=false)? Default to fake_upheld=true ONLY if you genuinely cannot defend the chemistry.

Reply with ONLY a JSON object, no markdown fence:
{{"fake_upheld": true|false, "corrected_verdict": "VALID|SUSPECT|FAKE", "reason": "<short>"}}"""

_lock = threading.Lock()

# --- adaptive concurrency gate -------------------------------------------
# Day (08:00-23:59 local): DAY_LIMIT workers. Night (00:00-07:59): NIGHT_LIMIT.
# On any HTTP 429 / rate-limit signal, clamp to RL_CLAMP for RL_COOLDOWN_S.
DAY_LIMIT = 3
NIGHT_LIMIT = 6
RL_CLAMP = 2
RL_COOLDOWN_S = 600
_gate = threading.Condition()
_active = 0
_rl_until = 0.0

def _allowed():
    now = time.time()
    if now < _rl_until:
        return RL_CLAMP
    h = datetime.datetime.now().hour
    return NIGHT_LIMIT if 0 <= h < 8 else DAY_LIMIT

def _acquire():
    global _active
    with _gate:
        while _active >= _allowed():
            _gate.wait(timeout=30)
        _active += 1

def _release():
    global _active
    with _gate:
        _active -= 1
        _gate.notify_all()

def _note_rate_limited():
    global _rl_until
    with _gate:
        _rl_until = time.time() + RL_COOLDOWN_S
        _gate.notify_all()

def call_llm(prompt, retries=3):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMP,
        "max_tokens": MAXTOK,
    }).encode()
    last = None
    for attempt in range(retries):
        _acquire()
        try:
            req = urllib.request.Request(API, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                d = json.load(resp)
            content = d["choices"][0]["message"]["content"]
            fr = d["choices"][0].get("finish_reason")
            if not content and fr == "length":
                raise RuntimeError("empty content, finish_reason=length")
            return content
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 503, 529):
                _note_rate_limited()
                time.sleep(30 * (attempt + 1))
            else:
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
        finally:
            _release()
    raise RuntimeError(f"LLM call failed after {retries} tries: {last}")

def parse_json_reply(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))

def steps_block(route):
    lines = []
    for s in route["steps"]:
        ann = []
        if s.get("carbon_delta") is not None:
            ann.append(f"Cdelta={s['carbon_delta']:+d}")
        rd = s.get("ring_delta") or {}
        if rd.get("gained"):
            ann.append(f"ring gained {rd['gained']}")
        if rd.get("lost"):
            ann.append(f"ring lost {rd['lost']}")
        tag = ("  [" + ", ".join(ann) + "]") if ann else ""
        lines.append(f"step[{s['i']}]: {s['raw']}{tag}")
    return "\n".join(lines)

def audit_one(route):
    uid = route["uid"]
    flags = route["flags"]
    # Stage 0 gate
    if flags["any_unparseable"] or not flags["target_produced"]:
        code = "ERROR" if flags["any_unparseable"] else "TARGET_NOT_PRODUCED"
        return {"uid": uid, "cond": route["cond"], "idx": route["idx"],
                "rank": route["rank"], "stage": 0, "route_verdict": "FAKE",
                "worst_defect": code, "adv_triggered": False}
    sb = steps_block(route)
    # Stage 1
    reply = call_llm(DEEPREAD_PROMPT.format(target=route["target_smiles"],
                                            n_steps=len(route["steps"]),
                                            steps_block=sb))
    a = parse_json_reply(reply)
    rec = {"uid": uid, "cond": route["cond"], "idx": route["idx"],
           "rank": route["rank"], "stage": 1,
           "route_verdict": a.get("route_verdict"),
           "worst_defect": a.get("worst_defect_code"),
           "n_fake_steps": a.get("n_fake_steps"),
           "n_suspect_steps": a.get("n_suspect_steps"),
           "steps": a.get("steps"), "summary": a.get("summary"),
           "adv_triggered": False}
    # Stage 2
    if rec["route_verdict"] == "FAKE":
        flagged = [s for s in (a.get("steps") or []) if s.get("verdict") == "FAKE"]
        fb = "\n".join(f"  step[{s['step_index']}] {s.get('defect_code')}: {s.get('reason','')}"
                       for s in flagged) or "  (no per-step detail)"
        vreply = call_llm(ADVERSARY_PROMPT.format(target=route["target_smiles"],
                                                  n_steps=len(route["steps"]),
                                                  steps_block=sb, flagged_block=fb))
        v = parse_json_reply(vreply)
        rec["adv_triggered"] = True
        rec["adv_upheld"] = bool(v.get("fake_upheld"))
        rec["adv_reason"] = v.get("reason")
        if not rec["adv_upheld"]:
            rec["route_verdict"] = v.get("corrected_verdict", "SUSPECT")
            rec["stage"] = 2
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--conds", default="")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    routes = json.load(open(args.routes))
    if args.conds:
        keep = set(args.conds.split(","))
        routes = [r for r in routes if r["cond"] in keep]
    if args.limit_cases:
        keep_idx = sorted({r["idx"] for r in routes})[:args.limit_cases]
        routes = [r for r in routes if r["idx"] in set(keep_idx)]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(json.loads(line)["uid"])
            except Exception:
                pass
    todo = [r for r in routes if r["uid"] not in done]
    print(f"routes total {len(routes)}, done {len(done & {r['uid'] for r in routes})}, todo {len(todo)}")

    outf = open(args.out, "a")
    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(audit_one, r): r["uid"] for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            uid = futs[fut]
            try:
                rec = fut.result()
                n_ok += 1
            except Exception as e:
                rec = {"uid": uid, "route_verdict": "ERROR", "error": str(e)[:300]}
                n_err += 1
            with _lock:
                outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                outf.flush()
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] ok={n_ok} err={n_err} "
                      f"elapsed={el/60:.1f}m eta={(el/i*(len(todo)-i))/60:.0f}m", flush=True)
    print(f"DONE ok={n_ok} err={n_err}")

if __name__ == "__main__":
    main()
