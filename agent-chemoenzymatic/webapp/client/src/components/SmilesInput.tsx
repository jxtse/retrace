import { useState } from "react";
import { MoleculeView } from "./MoleculeView";
import { StructureEditor } from "./StructureEditor";

interface Props {
  value: string;
  onChange: (v: string) => void;
  disabled?: boolean;
}

const EXAMPLES: { label: string; smiles: string }[] = [
  { label: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O" },
  { label: "Ibuprofen", smiles: "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" },
  { label: "Paracetamol", smiles: "CC(=O)NC1=CC=C(C=C1)O" },
  { label: "Sildenafil", smiles: "CCCC1=NN(C)C2=C1N=C(NC2=O)C1=CC(=CC=C1OCC)S(=O)(=O)N1CCN(C)CC1" },
];

export function SmilesInput({ value, onChange, disabled }: Props) {
  const [focused, setFocused] = useState(false);
  const [drawing, setDrawing] = useState(false);

  return (
    <div className="flex flex-col md:flex-row gap-4 items-start">
      <div className="flex-1 min-w-0 w-full">
        <div className="flex items-center justify-between mb-1">
          <label className="block text-sm font-medium text-slate-700">
            Target SMILES
          </label>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setDrawing(true)}
            title="Draw the molecule with a structure editor instead of typing SMILES"
            className="text-xs px-2 py-1 rounded border border-slate-300 text-slate-700 hover:bg-slate-100 disabled:opacity-50 inline-flex items-center gap-1"
          >
            <span aria-hidden>✎</span> Draw structure
          </button>
        </div>
        <textarea
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O — or use Draw structure"
          rows={2}
          spellCheck={false}
          className="font-mono text-sm w-full resize-y rounded border border-slate-300 px-3 py-2 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none bg-white disabled:bg-slate-50 disabled:text-slate-400"
        />
        <div className="mt-1 flex flex-wrap gap-1.5 text-xs">
          <span className="text-slate-500">Examples:</span>
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              disabled={disabled}
              onClick={() => onChange(ex.smiles)}
              className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 disabled:opacity-50"
            >
              {ex.label}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-col items-center">
        <div className="text-xs text-slate-500 mb-1">Preview</div>
        <MoleculeView smiles={value} width={220} height={160} />
        {focused && <div className="text-[10px] text-slate-400 mt-1">RDKit-JS</div>}
      </div>

      <StructureEditor
        open={drawing}
        initialSmiles={value}
        onCancel={() => setDrawing(false)}
        onConfirm={(smi) => {
          onChange(smi);
          setDrawing(false);
        }}
      />
    </div>
  );
}
