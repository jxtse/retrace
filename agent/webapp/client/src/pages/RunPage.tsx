import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { SmilesInput } from "../components/SmilesInput";
import { ModelPicker } from "../components/ModelPicker";
import { EventStream } from "../components/EventStream";
import { RouteGallery } from "../components/RouteGallery";
import { StatusBanner } from "../components/StatusBanner";
import { api } from "../lib/api";
import type { TraceEvent } from "../lib/eventTypes";
import { useRunStore } from "../state/runStore";

export function RunPage() {
  const [smiles, setSmiles] = useState("CC(=O)OC1=CC=CC=C1C(=O)O");
  const [family, setFamily] = useState<"claude" | "gpt5_5">("claude");
  const [preset, setPreset] = useState<"fast" | "standard" | "thorough">("fast");
  const [routeRefreshTick, setRouteRefreshTick] = useState(0);
  const esRef = useRef<EventSource | null>(null);
  const qc = useQueryClient();

  const {
    sessionId,
    status,
    events,
    effective,
    errorMessage,
    startedAt,
    reset,
    setSession,
    setStatus,
    appendEvent,
    setError,
    setFinalResult,
  } = useRunStore();

  const cleanupES = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
  }, []);

  useEffect(() => cleanupES, [cleanupES]);

  const start = useCallback(async () => {
    if (!smiles.trim()) return;
    cleanupES();
    reset();
    setStatus("starting");
    try {
      const created = await api.startRun({ smiles: smiles.trim(), model_family: family, preset });
      setSession(created.session_id, created.effective);
      const es = api.events(created.session_id);
      esRef.current = es;
      es.onmessage = (msg) => {
        let evt: TraceEvent | null = null;
        try {
          evt = JSON.parse(msg.data);
        } catch {
          return;
        }
        if (!evt) return;
        appendEvent(evt);
        if (evt.event === "tool_call" && evt.payload?.name === "save_best_route") {
          setRouteRefreshTick((t) => t + 1);
        }
        if (evt.event === "ranked_table") {
          setRouteRefreshTick((t) => t + 1);
        }
        if (evt.event === "_done") {
          setStatus("done");
          // Fetch final result + refresh route gallery.
          api.result(created.session_id).then((r) => setFinalResult(r.result)).catch(() => {});
          setRouteRefreshTick((t) => t + 1);
          qc.invalidateQueries({ queryKey: ["routes", created.session_id] });
          cleanupES();
        }
        if (evt.event === "_failed" || evt.event === "_stream_error") {
          setError(evt.payload?.error ?? "unknown error");
          cleanupES();
        }
      };
      es.onerror = (e) => {
        console.warn("SSE error", e);
      };
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }, [smiles, family, preset, cleanupES, reset, setSession, setStatus, appendEvent, setError, setFinalResult, qc]);

  const cancel = useCallback(async () => {
    if (!sessionId) return;
    try {
      await api.cancel(sessionId);
    } catch (e) {
      console.warn(e);
    }
  }, [sessionId]);

  const running = status === "starting" || status === "running";

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="max-w-[1400px] mx-auto px-4 py-3 flex items-baseline gap-3">
          <h1 className="text-lg font-semibold text-slate-800">
            Chemoenzymatic Retrosynthesis Agent
          </h1>
          <span className="text-xs text-slate-500">v0.1 · localhost</span>
          {sessionId && (
            <span className="ml-auto text-xs font-mono text-slate-500">
              session={sessionId} · {status}
            </span>
          )}
        </div>
      </header>

      <div className="max-w-[1400px] mx-auto px-4 py-4 w-full flex-1 flex flex-col gap-4">
        <section className="bg-white rounded-lg shadow-sm border border-slate-200 p-4 flex flex-col lg:flex-row gap-6">
          <div className="flex-1">
            <SmilesInput value={smiles} onChange={setSmiles} disabled={running} />
          </div>
          <div className="flex flex-col gap-3 lg:border-l lg:pl-6">
            <ModelPicker family={family} preset={preset} onChange={(f, p) => { setFamily(f); setPreset(p); }} disabled={running} />
            <div className="flex gap-2 mt-1">
              <button
                type="button"
                onClick={start}
                disabled={running || !smiles.trim()}
                className="px-4 py-1.5 rounded bg-blue-600 text-white font-medium text-sm hover:bg-blue-700 disabled:opacity-50"
              >
                {running ? "Running…" : "Run"}
              </button>
              <button
                type="button"
                onClick={cancel}
                disabled={!running || !sessionId}
                className="px-4 py-1.5 rounded border border-slate-300 text-slate-700 font-medium text-sm hover:bg-slate-100 disabled:opacity-40"
              >
                Cancel
              </button>
            </div>
          </div>
        </section>

        {effective && (
          <section className="text-[11px] font-mono text-slate-500 bg-white px-3 py-1.5 rounded border border-slate-200">
            effective: model={String(effective.llm_model)} ·{" "}
            reasoning={String(effective.llm_reasoning_effort)} ·{" "}
            max_iter={String(effective.max_iterations)} ·{" "}
            stop_after_distinct={String(effective.stop_after_distinct_routes)}
          </section>
        )}

        {errorMessage && (
          <section className="text-sm bg-red-50 border border-red-300 text-red-800 rounded p-3">
            ❗ {errorMessage}
          </section>
        )}

        <StatusBanner status={status} events={events} startedAt={startedAt} />

        <section className="grid grid-cols-1 xl:grid-cols-5 gap-4 flex-1 min-h-0">
          <div className="xl:col-span-3 min-w-0">
            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-4 h-full">
              <h2 className="text-base font-semibold text-slate-800 mb-2">
                Agent trace
              </h2>
              <div className="overflow-y-auto pr-2" style={{ maxHeight: "75vh" }}>
                <EventStream sessionId={sessionId} events={events} />
              </div>
            </div>
          </div>
          <div className="xl:col-span-2 min-w-0">
            <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-4 h-full">
              <div className="overflow-y-auto pr-2" style={{ maxHeight: "75vh" }}>
                <RouteGallery sessionId={sessionId} refreshTick={routeRefreshTick} />
              </div>
            </div>
          </div>
        </section>
      </div>

      <footer className="text-center text-xs text-slate-400 py-3">
        Backed by <code>core.agent</code> · ReACT loop · scores from{" "}
        <code>tools/verifier/score_route</code>
      </footer>
    </div>
  );
}
