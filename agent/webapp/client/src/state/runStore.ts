import { create } from "zustand";
import type { RouteEntry, TraceEvent } from "../lib/eventTypes";

export type RunStatus = "idle" | "starting" | "running" | "done" | "error" | "cancelled";

interface RunState {
  sessionId: string | null;
  status: RunStatus;
  events: TraceEvent[];
  routes: RouteEntry[];
  effective: Record<string, any> | null;
  errorMessage: string | null;
  finalResult: any | null;
  startedAt: number | null;

  reset: () => void;
  setSession: (sid: string, effective: Record<string, any>) => void;
  setStatus: (s: RunStatus) => void;
  appendEvent: (e: TraceEvent) => void;
  setRoutes: (rs: RouteEntry[]) => void;
  setError: (msg: string) => void;
  setFinalResult: (r: any) => void;
}

export const useRunStore = create<RunState>((set) => ({
  sessionId: null,
  status: "idle",
  events: [],
  routes: [],
  effective: null,
  errorMessage: null,
  finalResult: null,
  startedAt: null,

  reset: () =>
    set({
      sessionId: null,
      status: "idle",
      events: [],
      routes: [],
      effective: null,
      errorMessage: null,
      finalResult: null,
      startedAt: null,
    }),
  setSession: (sid, effective) =>
    set({ sessionId: sid, effective, status: "running", events: [], routes: [], errorMessage: null, finalResult: null, startedAt: Date.now() }),
  setStatus: (status) => set({ status }),
  appendEvent: (e) =>
    set((s) => {
      if (s.events.length > 10000) return { events: [...s.events.slice(-9000), e] };
      return { events: [...s.events, e] };
    }),
  setRoutes: (routes) => set({ routes }),
  setError: (msg) => set({ status: "error", errorMessage: msg }),
  setFinalResult: (finalResult) => set({ finalResult }),
}));
