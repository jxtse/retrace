import { useEffect, useRef, useState } from "react";
import { Editor } from "ketcher-react";
import { StandaloneStructServiceProvider } from "ketcher-standalone";
import type { Ketcher } from "ketcher-core";
import "ketcher-react/dist/index.css";

interface Props {
  open: boolean;
  initialSmiles?: string;
  onCancel: () => void;
  onConfirm: (smiles: string) => void;
}

const structServiceProvider = new StandaloneStructServiceProvider();

export function StructureEditor({ open, initialSmiles, onCancel, onConfirm }: Props) {
  const ketcherRef = useRef<Ketcher | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open || !ketcherRef.current || !initialSmiles?.trim()) return;
    const k = ketcherRef.current;
    k.setMolecule(initialSmiles.trim()).catch((e: any) => {
      setError(`Could not load initial SMILES: ${e?.message ?? e}`);
    });
  }, [open, initialSmiles]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);

  if (!open) return null;

  const handleConfirm = async () => {
    const k = ketcherRef.current;
    if (!k) return;
    setBusy(true);
    setError(null);
    try {
      const smi = await k.getSmiles();
      const trimmed = (smi ?? "").trim();
      if (!trimmed) {
        setError("Canvas is empty — draw a structure or press Cancel.");
        return;
      }
      onConfirm(trimmed);
    } catch (e: any) {
      setError(`Failed to export SMILES: ${e?.message ?? e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onClick={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div className="bg-white rounded-lg shadow-xl w-full max-w-[1200px] h-[85vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between px-4 py-2 border-b border-slate-200">
          <h2 className="text-base font-semibold text-slate-800">
            Draw structure
          </h2>
          <div className="text-xs text-slate-500">
            Esc to cancel · Click outside to dismiss
          </div>
        </div>

        <div className="flex-1 min-h-0 relative">
          <Editor
            staticResourcesUrl=""
            structServiceProvider={structServiceProvider as any}
            errorHandler={(msg: string) => setError(msg)}
            onInit={(ketcher: Ketcher) => {
              ketcherRef.current = ketcher;
              if (initialSmiles?.trim()) {
                ketcher.setMolecule(initialSmiles.trim()).catch(() => {});
              }
            }}
          />
        </div>

        <div className="flex items-center justify-between gap-3 px-4 py-3 border-t border-slate-200 bg-slate-50">
          <div className="text-sm text-red-700 flex-1 truncate">
            {error ? `❗ ${error}` : ""}
          </div>
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="px-4 py-1.5 rounded border border-slate-300 text-slate-700 font-medium text-sm hover:bg-slate-100 disabled:opacity-40"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleConfirm}
            disabled={busy}
            className="px-4 py-1.5 rounded bg-blue-600 text-white font-medium text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {busy ? "Reading…" : "Use this structure"}
          </button>
        </div>
      </div>
    </div>
  );
}
