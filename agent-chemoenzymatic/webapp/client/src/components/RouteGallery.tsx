import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { RouteCard } from "./RouteCard";
import type { RouteEntry } from "../lib/eventTypes";

interface Props {
  sessionId: string | null;
  refreshTick: number;
}

/** Top-K best routes panel. Polls `/api/runs/{sid}/routes` and re-fetches
 *  whenever `refreshTick` changes (the parent bumps it on `save_best_route`
 *  events). */
export function RouteGallery({ sessionId, refreshTick }: Props) {
  const { data, isLoading } = useQuery({
    enabled: !!sessionId,
    queryKey: ["routes", sessionId, refreshTick],
    queryFn: () => api.routes(sessionId!),
    refetchInterval: false,
  });

  const routes: RouteEntry[] = useMemo(() => {
    const arr = data?.routes ?? [];
    return [...arr].sort((a, b) => (b.total_score ?? 0) - (a.total_score ?? 0));
  }, [data]);

  return (
    <div className="flex flex-col gap-3">
      <header className="flex items-baseline justify-between">
        <h2 className="text-base font-semibold text-slate-800">Best routes</h2>
        <span className="text-xs text-slate-500">
          {routes.length} saved
          {data?.status && <> · {data.status}</>}
        </span>
      </header>
      {!sessionId && (
        <div className="text-sm text-slate-500 italic">
          Start a run to see candidate routes appear here.
        </div>
      )}
      {sessionId && isLoading && (
        <div className="text-sm text-slate-500">Loading…</div>
      )}
      {sessionId && routes.length === 0 && !isLoading && (
        <div className="text-sm text-slate-500 italic">
          No routes yet. The first one will appear as soon as the agent calls
          <code className="font-mono mx-1">save_best_route</code>.
        </div>
      )}
      <div className="space-y-3">
        {routes.map((r, i) => (
          <RouteCard key={(r.label ?? "r") + i} route={r} rank={i + 1} />
        ))}
      </div>
    </div>
  );
}
