/**
 * Lazy RDKit-JS singleton. RDKit init is ~2 MB of WASM + ~200 ms parse,
 * so we load on first call and reuse across the app.
 *
 * If init fails (network blocked, WASM not delivered), `getRDKit` resolves
 * to `null` and the caller can fall back to SmilesDrawer or the
 * server-side `/api/mol/png` renderer.
 */

let pending: Promise<any | null> | null = null;

declare global {
  interface Window {
    initRDKitModule?: () => Promise<any>;
    RDKit?: any;
  }
}

export function getRDKit(): Promise<any | null> {
  if (pending) return pending;
  pending = (async () => {
    if (window.RDKit) return window.RDKit;
    if (!window.initRDKitModule) {
      // Inject the official UMD bundle from a CDN. Pinned major to avoid
      // surprise upgrades.
      await new Promise<void>((res, rej) => {
        const s = document.createElement("script");
        s.src = "https://unpkg.com/@rdkit/rdkit/dist/RDKit_minimal.js";
        s.async = true;
        s.onload = () => res();
        s.onerror = () => rej(new Error("failed to load RDKit-JS"));
        document.head.appendChild(s);
      });
    }
    if (!window.initRDKitModule) return null;
    try {
      const mod = await window.initRDKitModule();
      window.RDKit = mod;
      return mod;
    } catch (e) {
      console.error("RDKit init failed", e);
      return null;
    }
  })();
  return pending;
}

/** Render a SMILES to SVG via RDKit-JS, or return null on parse failure. */
export async function smilesToSvg(smiles: string, w = 200, h = 150): Promise<string | null> {
  const rdkit = await getRDKit();
  if (!rdkit) return null;
  let mol: any = null;
  try {
    mol = rdkit.get_mol(smiles);
    if (!mol || !mol.is_valid()) return null;
    const svg = mol.get_svg(w, h);
    return svg;
  } catch (e) {
    console.warn("smilesToSvg failed", e);
    return null;
  } finally {
    mol?.delete?.();
  }
}
