import { useEffect, useMemo, useRef } from "react";
import type { TraceEvent } from "../lib/eventTypes";
import { ToolResultCard } from "./ToolResultCard";

interface Props {
  sessionId: string | null;
  events: TraceEvent[];
}

/** Renders the live event stream. Each ReACT iteration becomes a section;
 *  inside, `thinking` / `tool_call` / `tool_result` events render with
 *  the chemistry-friendly visuals their content deserves. */
export function EventStream({ sessionId, events }: Props) {
  const tailRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new events.
  useEffect(() => {
    tailRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length]);

  // Group events by iteration.
  const grouped = useMemo(() => groupByIteration(events), [events]);

  // Tool-call ids that have NOT yet seen a tool_result — used to render a spinner.
  const pendingCallIds = useMemo(() => {
    const pending = new Set<string>();
    for (const e of events) {
      if (e.event === "tool_call" && e.payload?.tool_call_id) {
        pending.add(String(e.payload.tool_call_id));
      } else if (e.event === "tool_result" && e.payload?.tool_call_id) {
        pending.delete(String(e.payload.tool_call_id));
      }
    }
    return pending;
  }, [events]);

  return (
    <div className="flex flex-col gap-4">
      {grouped.length === 0 && (
        <div className="flex flex-col items-center justify-center py-10 text-slate-500">
          <div className="flex gap-1.5 mb-3">
            <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: "0ms" }} />
            <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: "150ms" }} />
            <span className="w-2 h-2 rounded-full bg-slate-400 animate-bounce" style={{ animationDelay: "300ms" }} />
          </div>
          <div className="text-sm italic">Waiting for first event…</div>
          <div className="text-xs text-slate-400 mt-1">
            (cold-start can take a few seconds while LLM client + tools warm up)
          </div>
        </div>
      )}
      {grouped.map((grp) => (
        <IterationBlock key={grp.iteration} group={grp} sessionId={sessionId} pendingCallIds={pendingCallIds} />
      ))}
      <div ref={tailRef} />
    </div>
  );
}

interface Group {
  iteration: number;
  events: TraceEvent[];
}

function groupByIteration(events: TraceEvent[]): Group[] {
  const groups: Group[] = [];
  let current: Group = { iteration: 0, events: [] };
  for (const e of events) {
    if (e.event === "iteration") {
      if (current.events.length > 0) groups.push(current);
      current = { iteration: Number(e.payload?.n ?? 0), events: [] };
      continue;
    }
    current.events.push(e);
  }
  if (current.events.length > 0) groups.push(current);
  return groups;
}

function IterationBlock({ group, sessionId, pendingCallIds }: { group: Group; sessionId: string | null; pendingCallIds: Set<string> }) {
  return (
    <section className="border border-slate-200 rounded-md bg-white shadow-sm overflow-hidden">
      <header className="px-3 py-1.5 bg-slate-100 text-xs font-medium text-slate-600 flex items-center gap-2">
        <span className="inline-block w-5 h-5 rounded-full bg-slate-700 text-white text-[10px] flex items-center justify-center">
          {group.iteration || "·"}
        </span>
        <span>
          {group.iteration ? `Iteration ${group.iteration}` : "Pre-loop"}
        </span>
        <span className="ml-auto text-[10px] text-slate-400">
          {group.events.length} events
        </span>
      </header>
      <div className="p-3 space-y-2 text-xs">
        {group.events.map((e, i) => (
          <EventLine key={i} event={e} sessionId={sessionId} pendingCallIds={pendingCallIds} />
        ))}
      </div>
    </section>
  );
}

function EventLine({ event, sessionId, pendingCallIds }: { event: TraceEvent; sessionId: string | null; pendingCallIds: Set<string> }) {
  const { event: name, payload } = event;
  switch (name) {
    case "thinking":
      return (
        <details className="border-l-4 border-cyan-500 pl-2 py-1 bg-cyan-50/40 rounded-r">
          <summary className="cursor-pointer text-cyan-900 font-medium">
            🧠 Thinking ({Math.round((payload.content ?? "").length / 4)} tok)
          </summary>
          <pre className="whitespace-pre-wrap text-slate-700 mt-1">{payload.content}</pre>
        </details>
      );

    case "tool_call": {
      const inflight = payload?.tool_call_id ? pendingCallIds.has(String(payload.tool_call_id)) : false;
      return (
        <div className="flex items-start gap-2">
          <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-100 text-amber-800 inline-flex items-center gap-1">
            {inflight && (
              <svg className="animate-spin w-3 h-3 text-amber-700" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
                <path d="M22 12a10 10 0 0 1-10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
            )}
            {payload.name}
          </span>
          <code className="text-slate-600 break-all flex-1">{formatArgs(payload.args)}</code>
        </div>
      );
    }

    case "tool_result":
      if (!sessionId) return null;
      return (
        <div>
          {payload.elapsed_ms !== undefined && payload.elapsed_ms !== null && (
            <div className="text-[10px] text-slate-400 mb-0.5">
              {payload.tool_name ?? ""} · {payload.elapsed_ms} ms
            </div>
          )}
          <ToolResultCard
            sessionId={sessionId}
            toolCallId={payload.tool_call_id ?? null}
            toolName={payload.tool_name ?? "tool"}
            inlineContent={typeof payload.content === "string" ? payload.content : JSON.stringify(payload.content)}
          />
        </div>
      );

    case "error":
    case "native_bailout":
    case "wallclock_exceeded":
    case "_failed":
    case "_stream_error":
      return (
        <div className="text-red-700 bg-red-50 border border-red-200 rounded px-2 py-1">
          ❗ <b>{name}</b>: {payload.error ?? payload.message ?? JSON.stringify(payload)}
        </div>
      );

    case "ranked_table":
      return (
        <div className="text-emerald-800 bg-emerald-50 border border-emerald-200 rounded px-2 py-1">
          📊 Routes ranked ({payload?.payload?.routes?.length ?? "?"})
        </div>
      );

    case "_done":
      return (
        <div className="text-emerald-800 bg-emerald-50 border border-emerald-200 rounded px-2 py-1">
          🎉 Done · {payload.iterations ?? "?"} iter · {Number(payload.time ?? 0).toFixed(1)} s ·{" "}
          {payload.n_routes ?? 0} routes
        </div>
      );

    case "mini_status":
      return <div className="text-slate-500 italic">📍 {payload.text}</div>;

    case "saved_threshold_nudge":
      return (
        <div className="text-amber-700 italic">
          💡 nudge: {payload.n_saved}/{payload.threshold} routes saved
        </div>
      );

    case "soft_threshold_reached":
      return <div className="text-emerald-700">✅ soft threshold ({payload.distinct})</div>;

    case "llm_self_complete_honored":
      return <div className="text-emerald-700">✅ LLM self-complete honored</div>;

    case "llm_self_complete_ignored":
      return <div className="text-slate-500 italic">stop marker ignored (pre-threshold)</div>;

    case "checkpoint_compacted":
    case "backtrack_compacted":
      return (
        <div className="text-slate-500 italic">
          🗜 compacted ({payload.demoted ?? payload.dropped ?? "?"})
        </div>
      );

    case "iteration_refunded":
      return <div className="text-slate-500 italic">↩ iter refunded ({payload.reason})</div>;

    case "prefetch":
      return (
        <div className="text-slate-500 italic">
          ⚡ prefetch: {(payload.tools ?? []).join(", ")}
        </div>
      );

    default:
      return (
        <div className="text-slate-500">
          <code className="text-[10px]">{name}</code>: {JSON.stringify(payload)}
        </div>
      );
  }
}

function formatArgs(args: any): string {
  if (typeof args === "string") {
    try {
      const obj = JSON.parse(args);
      return JSON.stringify(obj);
    } catch {
      return args;
    }
  }
  try {
    return JSON.stringify(args);
  } catch {
    return String(args);
  }
}
