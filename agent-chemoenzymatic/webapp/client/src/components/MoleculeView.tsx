import { useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { smilesToSvg } from "../lib/rdkit";

interface Props {
  smiles: string;
  width?: number;
  height?: number;
  className?: string;
}

/** Render a SMILES via RDKit-JS; on failure fall back to the server's
 *  /api/mol/png endpoint (RDKit Python). Both are cached per-mount via
 *  useState so we don't redraw on every parent rerender. */
export function MoleculeView({ smiles, width = 200, height = 150, className }: Props) {
  const [svg, setSvg] = useState<string | null>(null);
  const [fallbackPng, setFallbackPng] = useState<string | null>(null);
  const lastSmiles = useRef<string>("");

  useEffect(() => {
    let cancelled = false;
    const target = smiles.trim();
    if (!target || target === lastSmiles.current) return;
    lastSmiles.current = target;
    setSvg(null);
    setFallbackPng(null);
    (async () => {
      const out = await smilesToSvg(target, width, height);
      if (cancelled) return;
      if (out) {
        setSvg(out);
      } else {
        setFallbackPng(api.molPng(target, width, height));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [smiles, width, height]);

  if (svg) {
    return (
      <div
        className={"inline-block bg-white rounded border border-slate-200 " + (className ?? "")}
        style={{ width, height }}
        dangerouslySetInnerHTML={{ __html: svg }}
      />
    );
  }
  if (fallbackPng) {
    return (
      <img
        src={fallbackPng}
        alt={smiles}
        width={width}
        height={height}
        className={"bg-white rounded border border-slate-200 " + (className ?? "")}
        onError={() => setFallbackPng(null)}
      />
    );
  }
  return (
    <div
      className={
        "inline-flex items-center justify-center bg-slate-100 text-slate-400 text-xs border border-dashed border-slate-300 rounded " +
        (className ?? "")
      }
      style={{ width, height }}
    >
      {smiles ? "…" : "—"}
    </div>
  );
}
