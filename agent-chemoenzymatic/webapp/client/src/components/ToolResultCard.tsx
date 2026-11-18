import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { MoleculeView } from "./MoleculeView";

interface Props {
  sessionId: string;
  toolCallId: string | null;
  toolName: string;
  inlineContent: string;
}

/** Per-tool detail card. Inline content is what the agent saw (often a
 *  result_store stub). User can expand to lazy-fetch the full payload. */
export function ToolResultCard({ sessionId, toolCallId, toolName, inlineContent }: Props) {
  const [expanded, setExpanded] = useState(false);
  const canRehydrate = !!toolCallId && toolCallId.startsWith("toolu_");

  const { data: full } = useQuery({
    enabled: expanded && canRehydrate,
    queryKey: ["tool-result", sessionId, toolCallId],
    queryFn: () => api.toolResult(sessionId, toolCallId!),
    staleTime: Infinity,
  });

  const inlineParsed = (() => {
    try {
      return JSON.parse(inlineContent);
    } catch {
      return null;
    }
  })();
  const payload = full?.content ?? inlineParsed;

  return (
    <div className="border border-slate-200 rounded bg-white">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full text-left px-3 py-2 text-xs flex items-center gap-2 hover:bg-slate-50"
      >
        <span className="text-slate-500">{expanded ? "▾" : "▸"}</span>
        <span className="font-medium text-slate-700">{toolName}</span>
        {canRehydrate && (
          <span className="ml-auto text-[10px] font-mono text-slate-400">{toolCallId}</span>
        )}
      </button>
      {expanded && (
        <div className="border-t border-slate-100 p-3 text-xs overflow-x-auto">
          {renderByTool(toolName, payload, inlineContent)}
        </div>
      )}
    </div>
  );
}

function renderByTool(name: string, payload: any, raw: string) {
  if (!payload) return <pre className="whitespace-pre-wrap break-words text-slate-600">{raw}</pre>;
  switch (name) {
    case "neural_retro":
      return <NeuralRetroView payload={payload} />;
    case "suggest_templates":
      return <SuggestTemplatesView payload={payload} />;
    case "score_route":
      return <ScoreRouteView payload={payload} />;
    case "explore_branches":
      return <ExploreBranchesView payload={payload} />;
    case "save_best_route":
    case "list_best_routes":
    case "get_best_route":
      return <JsonView payload={payload} />;
    case "check_buyability":
      return <BuyabilityView payload={payload} />;
    case "analyze_molecule":
      return <JsonView payload={payload} />;
    default:
      return <JsonView payload={payload} />;
  }
}

function JsonView({ payload }: { payload: any }) {
  return (
    <pre className="whitespace-pre-wrap break-words text-slate-700 font-mono">
      {JSON.stringify(payload, null, 2)}
    </pre>
  );
}

function NeuralRetroView({ payload }: { payload: any }) {
  const preds: any[] = payload.preds ?? payload.predictions ?? [];
  return (
    <div className="space-y-2">
      <div className="text-slate-500">
        {preds.length} predictions
        {payload.engines && <> · engines: {(payload.engines || []).join(", ")}</>}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {preds.slice(0, 12).map((p, i) => {
          const precursors = p.precursors ?? p.reactants ?? p.reactant_smiles ?? "";
          const arr = typeof precursors === "string" ? precursors.split(".") : Array.isArray(precursors) ? precursors : [];
          return (
            <div key={i} className="border border-slate-200 rounded p-2">
              <div className="text-[10px] text-slate-500 flex justify-between">
                <span>#{i + 1}</span>
                {p.score !== undefined && <span>score {Number(p.score).toFixed(3)}</span>}
                {p.support && <span>support [{(p.support || []).join(", ")}]</span>}
              </div>
              <div className="flex flex-wrap gap-1 mt-1">
                {arr.slice(0, 4).map((smi: string, k: number) => (
                  <MoleculeView key={k} smiles={smi} width={120} height={90} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SuggestTemplatesView({ payload }: { payload: any }) {
  const tpls: any[] = payload.templates ?? payload.results ?? [];
  return (
    <div className="space-y-1">
      <div className="text-slate-500">{tpls.length} templates</div>
      <div className="space-y-1 max-h-72 overflow-y-auto">
        {tpls.slice(0, 25).map((t, i) => (
          <div key={i} className="border border-slate-100 rounded p-2 text-[11px] font-mono">
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>{t.template_id ?? `#${i + 1}`}</span>
              {t.score !== undefined && <span>score {Number(t.score).toFixed(3)}</span>}
              {t.kind && <span>{t.kind}</span>}
            </div>
            <div className="text-slate-700 break-all">{t.smarts ?? t.template ?? "—"}</div>
            {t.reaction_name && <div className="text-slate-500">{t.reaction_name}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

function ScoreRouteView({ payload }: { payload: any }) {
  const f = payload.feasibility_score ?? payload.feasibility?.score ?? 0;
  const c = payload.cost_score ?? payload.cost?.score ?? 0;
  const g = payload.greenness_score ?? payload.greenness?.score ?? 0;
  const total = payload.total_score ?? 0;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-3">
        <div className="text-2xl font-semibold text-slate-800">{Number(total).toFixed(3)}</div>
        <div className="text-xs text-slate-500">total</div>
        {payload.bottleneck && (
          <div className="text-xs text-amber-700 ml-auto">⚠ {payload.bottleneck}</div>
        )}
      </div>
      <Bar label="Feasibility (0.70)" value={f} color="bg-blue-500" />
      <Bar label="Cost (0.15)" value={c} color="bg-amber-500" />
      <Bar label="Greenness (0.15)" value={g} color="bg-emerald-500" />
      {payload.warnings && payload.warnings.length > 0 && (
        <ul className="text-[11px] text-amber-700 list-disc pl-4">
          {payload.warnings.map((w: string, i: number) => (
            <li key={i}>{w}</li>
          ))}
        </ul>
      )}
      <details className="text-[11px] text-slate-600">
        <summary className="cursor-pointer">Full breakdown</summary>
        <JsonView payload={payload} />
      </details>
    </div>
  );
}

function Bar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
  return (
    <div>
      <div className="flex justify-between text-[11px] text-slate-600">
        <span>{label}</span>
        <span>{Number(value).toFixed(3)}</span>
      </div>
      <div className="h-2 bg-slate-100 rounded overflow-hidden">
        <div className={"h-full " + color} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ExploreBranchesView({ payload }: { payload: any }) {
  const branches: any[] = payload.branches ?? payload.candidates ?? [];
  return (
    <div className="space-y-2">
      <div className="text-slate-500">{branches.length} branches</div>
      {branches.slice(0, 10).map((b, i) => (
        <div key={i} className="border-l-2 border-slate-300 pl-2">
          <div className="text-[10px] text-slate-500">
            #{i + 1}
            {b.heuristic_score !== undefined && <> · score {Number(b.heuristic_score).toFixed(3)}</>}
            {b.template_id && <> · {b.template_id}</>}
          </div>
          {b.precursors && (
            <div className="flex flex-wrap gap-1 mt-1">
              {(typeof b.precursors === "string" ? b.precursors.split(".") : b.precursors).slice(0, 4).map((s: string, k: number) => (
                <MoleculeView key={k} smiles={s} width={100} height={75} />
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function BuyabilityView({ payload }: { payload: any }) {
  const items = payload.results ?? payload.items ?? (Array.isArray(payload) ? payload : []);
  if (!Array.isArray(items) || items.length === 0) return <JsonView payload={payload} />;
  return (
    <table className="text-xs w-full">
      <thead className="text-slate-500">
        <tr>
          <th className="text-left">SMILES</th>
          <th>Buyable</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {items.map((r: any, i: number) => (
          <tr key={i} className="border-t border-slate-100">
            <td className="font-mono break-all">{r.smiles}</td>
            <td className="text-center">
              {r.buyable === true ? "✓" : r.buyable === false ? "✗" : "?"}
            </td>
            <td className="text-slate-500 text-[10px]">{r.source ?? r.stock ?? ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
