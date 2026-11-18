import { MoleculeView } from "./MoleculeView";
import type { RouteEntry } from "../lib/eventTypes";

interface ParsedStep {
  product: string;
  reactants: string[];
  reactionName: string | null;
  raw: string;
}

/** Parse "<prod>>>r1.r2 (reaction: Suzuki)" → {product, reactants, name}.
 *  Trailing " (reaction: ...)" annotation is stripped on a literal " ("
 *  boundary to avoid chopping SMILES parens like c1ccccc1B(O)O. */
function parseStep(raw: string): ParsedStep | null {
  if (!raw.includes(">>")) return null;
  const [prod, rest] = raw.split(">>", 2);
  let react = rest;
  let reactionName: string | null = null;
  const annIdx = react.indexOf(" (reaction:");
  if (annIdx >= 0) {
    const tail = react.slice(annIdx + " (reaction:".length);
    const close = tail.indexOf(")");
    if (close >= 0) reactionName = tail.slice(0, close).trim();
    react = react.slice(0, annIdx);
  }
  const reactants = react.split(".").map((s) => s.trim()).filter(Boolean);
  return { product: prod.trim(), reactants, reactionName, raw };
}

interface Props {
  route: RouteEntry;
  rank: number;
}

const PROVENANCE_COLOR: Record<string, string> = {
  neural_consensus: "bg-blue-100 text-blue-800",
  template_suggested: "bg-amber-100 text-amber-800",
  rag_anchored: "bg-violet-100 text-violet-800",
  zero_shot: "bg-slate-100 text-slate-700",
  unknown: "bg-slate-100 text-slate-500",
};

export function RouteCard({ route, rank }: Props) {
  const steps = (route.steps ?? []).map(parseStep).filter(Boolean) as ParsedStep[];
  const provClass =
    PROVENANCE_COLOR[route.provenance ?? "unknown"] ?? PROVENANCE_COLOR.unknown;

  return (
    <article className="border border-slate-200 rounded-lg bg-white shadow-sm overflow-hidden">
      <header className="px-4 py-2 border-b border-slate-100 flex items-baseline gap-3">
        <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-slate-800 text-white text-xs font-medium">
          {rank}
        </span>
        <h3 className="font-medium text-slate-800 flex-1 truncate" title={route.label}>
          {route.label}
        </h3>
        <span className="text-2xl font-semibold text-emerald-700">
          {Number(route.total_score).toFixed(3)}
        </span>
      </header>

      <div className="px-4 py-2 flex flex-wrap items-center gap-2 text-[11px]">
        <span className={"px-1.5 py-0.5 rounded " + provClass}>{route.provenance}</span>
        <span className="text-slate-500">{steps.length} steps</span>
        <span className="ml-auto flex items-center gap-3 text-slate-700">
          {Object.entries(route.sub_scores ?? {}).map(([k, v]) => (
            <span key={k}>
              <span className="text-slate-500">{k}:</span> {Number(v).toFixed(3)}
            </span>
          ))}
        </span>
      </div>

      <div className="divide-y divide-slate-100">
        {steps.map((s, i) => (
          <StepRow key={i} step={s} index={i + 1} />
        ))}
        {steps.length === 0 && (
          <div className="px-4 py-3 text-xs text-slate-500 italic">
            No parsable steps (route may be malformed)
          </div>
        )}
      </div>

      {route.starting_materials && route.starting_materials.length > 0 && (
        <div className="px-4 py-2 bg-slate-50 border-t border-slate-100">
          <div className="text-[11px] text-slate-500 mb-1">Starting materials</div>
          <div className="flex flex-wrap gap-2">
            {route.starting_materials.map((s, i) => (
              <div key={i} className="flex flex-col items-center text-[10px] text-slate-500">
                <MoleculeView smiles={s} width={120} height={90} />
                <span className="mt-0.5 font-mono max-w-[120px] truncate" title={s}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {route.notes && (
        <div className="px-4 py-2 text-[11px] text-slate-600 border-t border-slate-100 bg-slate-50">
          <span className="text-slate-500">notes:</span> {route.notes}
        </div>
      )}
    </article>
  );
}

function StepRow({ step, index }: { step: ParsedStep; index: number }) {
  return (
    <div className="px-4 py-3">
      <div className="text-[11px] text-slate-500 mb-1">
        Step {index}
        {step.reactionName && <span className="ml-2 text-slate-700">· {step.reactionName}</span>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex flex-wrap gap-1">
          {step.reactants.map((r, k) => (
            <MoleculeView key={k} smiles={r} width={130} height={100} />
          ))}
        </div>
        <span className="text-slate-400 text-xl">→</span>
        <MoleculeView smiles={step.product} width={150} height={110} />
      </div>
    </div>
  );
}
