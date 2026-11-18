import { useEffect, useState } from "react";
import type { TraceEvent } from "../lib/eventTypes";
import type { RunStatus } from "../state/runStore";

interface Props {
  status: RunStatus;
  events: TraceEvent[];
  startedAt: number | null;
}

interface Activity {
  iteration: number;
  label: string;
  detail: string | null;
  tone: "thinking" | "tool" | "scoring" | "saving" | "idle" | "done";
}

const TONE_STYLES: Record<Activity["tone"], string> = {
  thinking: "bg-cyan-50 border-cyan-300 text-cyan-900",
  tool: "bg-amber-50 border-amber-300 text-amber-900",
  scoring: "bg-violet-50 border-violet-300 text-violet-900",
  saving: "bg-emerald-50 border-emerald-300 text-emerald-900",
  idle: "bg-slate-50 border-slate-300 text-slate-700",
  done: "bg-emerald-50 border-emerald-400 text-emerald-900",
};

const TOOL_LABELS: Record<string, string> = {
  neural_retro: "Neural one-step retrosynthesis",
  suggest_templates: "Searching template library",
  apply_template: "Applying SMARTS template",
  explore_branches: "Exploring branches",
  analyze_molecule: "Analyzing molecule",
  identify_bonds: "Identifying disconnections",
  retro_decompose: "Recap decomposition",
  compare_molecules: "Comparing molecules",
  validate_smiles: "Validating SMILES",
  check_buyability: "Checking commercial availability",
  score_route: "Scoring route (feasibility · cost · greenness)",
  save_best_route: "Saving route to scratchpad",
  list_best_routes: "Listing scratchpad routes",
  get_best_route: "Retrieving saved route",
  backtrack_and_summarize: "Backtracking & compacting",
  read_doc: "Reading agent doc",
  load_result: "Loading earlier tool result",
  todo_read: "Reading todo plan",
  todo_write: "Updating todo plan",
};

function deriveActivity(status: RunStatus, events: TraceEvent[]): Activity {
  if (status === "done") {
    return { iteration: 0, label: "Run finished", detail: null, tone: "done" };
  }
  if (status === "starting") {
    return { iteration: 0, label: "Spinning up agent…", detail: "loading config + LLM client", tone: "idle" };
  }
  if (status === "idle") {
    return { iteration: 0, label: "Idle", detail: "press Run to start", tone: "idle" };
  }

  let iteration = 0;
  let pending: { tool: string; args: any } | null = null;
  let last: Activity = { iteration: 0, label: "Waiting for first event…", detail: null, tone: "idle" };

  for (const e of events) {
    switch (e.event) {
      case "iteration":
        iteration = Number(e.payload?.n ?? iteration);
        last = { iteration, label: `Starting iteration ${iteration}`, detail: null, tone: "idle" };
        break;
      case "thinking":
        last = {
          iteration,
          label: "Thinking",
          detail: `${Math.round((e.payload?.content ?? "").length / 4)} tokens`,
          tone: "thinking",
        };
        break;
      case "tool_call": {
        const name = e.payload?.name ?? "tool";
        pending = { tool: name, args: e.payload?.args };
        const tone: Activity["tone"] = name === "score_route"
          ? "scoring"
          : name === "save_best_route"
            ? "saving"
            : "tool";
        last = {
          iteration,
          label: TOOL_LABELS[name] ?? `Calling ${name}`,
          detail: previewArgs(e.payload?.args),
          tone,
        };
        break;
      }
      case "tool_result":
        if (pending && pending.tool === e.payload?.tool_name) {
          pending = null;
        }
        last = {
          iteration,
          label: `${TOOL_LABELS[e.payload?.tool_name] ?? e.payload?.tool_name ?? "Tool"} returned`,
          detail: e.payload?.elapsed_ms != null ? `${e.payload.elapsed_ms} ms` : null,
          tone: "tool",
        };
        break;
      case "ranked_table":
        last = { iteration, label: "Ranking routes", detail: null, tone: "scoring" };
        break;
      case "saved_threshold_nudge":
        last = {
          iteration,
          label: "Nudging agent to wrap up",
          detail: `${e.payload?.n_saved}/${e.payload?.threshold} routes saved`,
          tone: "saving",
        };
        break;
      case "soft_threshold_reached":
        last = { iteration, label: "Soft threshold reached", detail: null, tone: "saving" };
        break;
      case "checkpoint_compacted":
      case "backtrack_compacted":
        last = { iteration, label: "Compacting context", detail: null, tone: "idle" };
        break;
    }
  }

  // If a tool_call hasn't returned yet, prefer showing the in-flight tool
  if (pending) {
    const tone: Activity["tone"] = pending.tool === "score_route"
      ? "scoring"
      : pending.tool === "save_best_route"
        ? "saving"
        : "tool";
    return {
      iteration,
      label: TOOL_LABELS[pending.tool] ?? `Running ${pending.tool}`,
      detail: previewArgs(pending.args),
      tone,
    };
  }
  return last;
}

function previewArgs(args: any): string | null {
  if (!args) return null;
  let obj: any = args;
  if (typeof args === "string") {
    try {
      obj = JSON.parse(args);
    } catch {
      return args.length > 80 ? args.slice(0, 80) + "…" : args;
    }
  }
  if (typeof obj !== "object") return String(obj);
  // Preferred keys for chemistry tools
  for (const k of ["smiles", "target", "product", "reaction_smiles", "route", "molecule"]) {
    if (obj[k]) {
      const v = String(obj[k]);
      return `${k}=${v.length > 60 ? v.slice(0, 60) + "…" : v}`;
    }
  }
  const keys = Object.keys(obj).slice(0, 2);
  if (keys.length === 0) return null;
  return keys.map((k) => `${k}=${truncate(String(obj[k]), 30)}`).join(" · ");
}

function truncate(s: string, n: number): string {
  return s.length > n ? s.slice(0, n) + "…" : s;
}

function fmtElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  return `${m}m${rs.toString().padStart(2, "0")}s`;
}

export function StatusBanner({ status, events, startedAt }: Props) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (status !== "running" && status !== "starting") return;
    const id = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [status]);

  if (status === "idle" || status === "error") return null;

  const activity = deriveActivity(status, events);
  const elapsed = startedAt ? now - startedAt : 0;
  const live = status === "running" || status === "starting";

  return (
    <section
      className={`text-sm rounded-md border px-3 py-2 flex items-center gap-3 ${TONE_STYLES[activity.tone]}`}
    >
      {live ? (
        <span className="relative flex h-2.5 w-2.5 flex-shrink-0">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-50" />
          <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-current" />
        </span>
      ) : (
        <span className="inline-flex h-2.5 w-2.5 rounded-full bg-current flex-shrink-0" />
      )}
      <div className="flex-1 min-w-0">
        <div className="font-medium truncate">
          {activity.iteration > 0 && (
            <span className="font-mono text-[11px] mr-2 opacity-70">iter {activity.iteration}</span>
          )}
          {activity.label}
          {live && <DotEllipsis />}
        </div>
        {activity.detail && (
          <div className="text-xs opacity-75 truncate font-mono">{activity.detail}</div>
        )}
      </div>
      {startedAt && (
        <span className="text-xs font-mono opacity-70 flex-shrink-0">{fmtElapsed(elapsed)}</span>
      )}
    </section>
  );
}

function DotEllipsis() {
  return (
    <span className="inline-flex ml-1 align-baseline">
      <span className="animate-pulse" style={{ animationDelay: "0ms" }}>.</span>
      <span className="animate-pulse" style={{ animationDelay: "150ms" }}>.</span>
      <span className="animate-pulse" style={{ animationDelay: "300ms" }}>.</span>
    </span>
  );
}
