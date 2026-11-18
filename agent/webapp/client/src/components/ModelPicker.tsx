import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import type { PresetInfo } from "../lib/eventTypes";

interface Props {
  family: "claude" | "gpt5_5";
  preset: "fast" | "standard" | "thorough";
  onChange: (family: "claude" | "gpt5_5", preset: "fast" | "standard" | "thorough") => void;
  disabled?: boolean;
}

const FAMILY_LABELS: Record<string, string> = {
  claude: "Claude-Opus-4.7",
  gpt5_5: "GPT-5.5",
};

const PRESET_BLURB: Record<string, string> = {
  fast: "Fast — quick exploration, ~50 iterations.",
  standard: "Standard — balanced reasoning, ~80 iterations.",
  thorough: "Thorough — extensive search, ~150 iterations.",
};

export function ModelPicker({ family, preset, onChange, disabled }: Props) {
  const { data } = useQuery({
    queryKey: ["presets"],
    queryFn: () => api.presets(),
    staleTime: 5 * 60 * 1000,
  });
  const all = data?.presets ?? [];
  const current: PresetInfo | undefined = all.find(
    (p) => p.family === family && p.preset === preset,
  );

  return (
    <div className="flex flex-col gap-2">
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">Model</label>
        <div className="inline-flex rounded border border-slate-300 bg-white overflow-hidden">
          {(["claude", "gpt5_5"] as const).map((f) => (
            <button
              key={f}
              type="button"
              disabled={disabled}
              onClick={() => onChange(f, preset)}
              className={
                "px-3 py-1.5 text-sm transition " +
                (f === family
                  ? "bg-blue-600 text-white"
                  : "text-slate-700 hover:bg-slate-100 disabled:opacity-50")
              }
            >
              {FAMILY_LABELS[f]}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">
          Reasoning budget
        </label>
        <div className="inline-flex rounded border border-slate-300 bg-white overflow-hidden">
          {(["fast", "standard", "thorough"] as const).map((p) => (
            <button
              key={p}
              type="button"
              disabled={disabled}
              onClick={() => onChange(family, p)}
              className={
                "px-3 py-1.5 text-sm capitalize transition " +
                (p === preset
                  ? "bg-emerald-600 text-white"
                  : "text-slate-700 hover:bg-slate-100 disabled:opacity-50")
              }
              title={PRESET_BLURB[p]}
            >
              {p}
            </button>
          ))}
        </div>
        {current && (
          <div className="text-[11px] text-slate-500 mt-1 font-mono">
            → model={current.model} · reasoning_effort={current.reasoning_effort} · max_iter={current.max_iterations}
          </div>
        )}
      </div>
    </div>
  );
}
