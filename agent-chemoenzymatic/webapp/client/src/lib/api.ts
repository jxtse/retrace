import type { PresetInfo, RouteEntry, RunCreated } from "./eventTypes";

/** Same-origin in production (`uvicorn` mounts the built client); proxied via
 *  Vite to `http://127.0.0.1:8000` in dev. */
const API_BASE = "";

async function jget<T>(url: string): Promise<T> {
  const r = await fetch(API_BASE + url);
  if (!r.ok) throw new Error(`${url} → ${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

async function jpost<T>(url: string, body: any): Promise<T> {
  const r = await fetch(API_BASE + url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} → ${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

export const api = {
  presets: () => jget<{ presets: PresetInfo[] }>("/api/presets"),
  health: () => jget<{ ok: boolean }>("/api/health"),
  startRun: (body: {
    smiles: string;
    model_family: "claude" | "gpt5_5";
    preset: "fast" | "standard" | "thorough";
    overrides?: Record<string, number | null>;
  }) => jpost<RunCreated>("/api/runs", body),
  cancel: (sid: string) =>
    jpost<{ cancelled: boolean; status: string }>(`/api/runs/${sid}/cancel`, {}),
  result: (sid: string, timeout = 0) =>
    jget<any>(`/api/runs/${sid}/result${timeout ? `?timeout=${timeout}` : ""}`),
  routes: (sid: string) =>
    jget<{ session_id: string; status: string; routes: RouteEntry[] }>(`/api/runs/${sid}/routes`),
  toolResult: (sid: string, tcid: string) =>
    jget<{ tool: string; tool_call_id: string; ts: string; content: any }>(
      `/api/runs/${sid}/results/${tcid}`,
    ),
  events: (sid: string) => new EventSource(API_BASE + `/api/runs/${sid}/events`),
  molPng: (smiles: string, w = 240, h = 180) =>
    API_BASE + `/api/mol/png?smiles=${encodeURIComponent(smiles)}&w=${w}&h=${h}`,
  rxnPng: (reactants: string, products: string, reagents = "", w = 480, h = 220) =>
    API_BASE +
    `/api/rxn/png?reactants=${encodeURIComponent(reactants)}` +
    `&products=${encodeURIComponent(products)}` +
    (reagents ? `&reagents=${encodeURIComponent(reagents)}` : "") +
    `&w=${w}&h=${h}`,
};
