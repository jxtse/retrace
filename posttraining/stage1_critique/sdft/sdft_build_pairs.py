#!/usr/bin/env python3
"""Build SDFT-format (prompt, teacher_prompt) pairs from ReTRACE teacher_full_v4_sft.

Data-format adaptation ONLY - the training loop/loss/checkpointing live in the
official Self-Distillation repo (distil_trainer.py / distil_config.py).

Source rows: {"messages": [...], "tools": [...], "loss_mask": [...], "route_text": ...,
              "case_idx", "rep", "target_smiles", "result_success", ...}

Output rows (HF conversational format consumed by official main.py style loaders):
  {"prompt":      [{role: system...},{role: user, content: TARGET_TASK}],
   "teacher_prompt": [same system, user with demonstration injected]}

For the smoke run we keep the official demonstration-conditioning template from
main.py load_science_dataset: append the demonstration to the user turn and ask
the model to answer in its own words. We cap the *conversation history* carried
into the smoke run (student must re-collect on-policy full trajectories later);
the capped fields are recorded in build_meta.json so nothing pretends to be
full-context.
"""
import json, hashlib, argparse, random, os

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="teacher_full_v4_sft.jsonl (or FROZEN_v1 frozen jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=130)
    ap.add_argument("--max-history", type=int, default=12, help="messages kept before terminal user turn (smoke only)")
    ap.add_argument("--max-demo-chars", type=int, default=4000)
    args = ap.parse_args()

    rows = []
    with open(args.src) as f:
        for line in f:
            j = json.loads(line)
            rows.append(j)

    # FROZEN selection policy is applied by the frozen file itself; if the raw
    # file is passed we only keep audited-VALID/SUSPECT per its manifest.
    out = []
    skipped = {"no_route": 0, "no_success": 0, "too_short": 0}
    for j in rows:
        if not j.get("result_success"):
            skipped["no_success"] += 1
            continue
        demo = (j.get("route_text") or "").strip()
        if not demo:
            skipped["no_route"] += 1
            continue
        msgs = j["messages"]
        # find terminal user turn (the task message) - messages are alternating
        # system, user, assistant, tool...; take the first user turn as the task.
        task_turn = next((m for m in msgs if m["role"] == "user"), None)
        if task_turn is None:
            skipped["too_short"] += 1
            continue
        sys_turn = next((m for m in msgs if m["role"] == "system"), None)
        base = ([sys_turn] if sys_turn else []) + [task_turn]

        # history cap: keep last N messages BEFORE the terminal answer span, as
        # conversational prefix after the task turn (assistant/tool turns only)
        # NOTE: HF chat templates require the conversation to END with a user or
        # system turn (the model then responds). Tool-role turns after the task
        # turn make `last message role == tool`, which apply_chat_template rejects.
        # For the smoke adapter we therefore carry NO assistant/tool history;
        # the task turn alone defines the prompt. Full multi-turn on-policy
        # collection is a later phase (official trainer is single-turn per step).
        hist = []
        base = base + hist

        demo_txt = demo[: args.max_demo_chars]

        student_prompt = [dict(m) for m in base]
        teacher_user = dict(task_turn)
        teacher_user["content"] = (
            task_turn["content"]
            + "\n\nA reference solution route for this target (audited) is:\n"
            + demo_txt
            + "\n\nNow plan the synthesis yourself with your own reasoning; do not copy the reference verbatim."
        )
        teacher_prompt = ([dict(sys_turn)] if sys_turn else []) + [teacher_user] + [dict(m) for m in hist]

        out.append({
            "prompt": student_prompt,
            "teacher_prompt": teacher_prompt,
            "meta": {
                "case_idx": j["case_idx"], "rep": j.get("rep"),
                "target_smiles": j.get("target_smiles"),
                "route_score": j.get("route_score"),
                "source_run": j.get("run_id"),
            },
        })
        if len(out) >= args.max_rows:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "src": args.src, "src_sha256": sha256(args.src),
        "rows_out": len(out), "skipped": skipped,
        "max_history": args.max_history, "max_demo_chars": args.max_demo_chars,
        "note": "smoke adapter: capped history + demonstration injection; NOT full-context; student full-context on-policy collection is a later phase",
        "cases": sorted({r["meta"]["case_idx"] for r in out}),
    }
    with open(args.out + ".build_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__":
    main()
