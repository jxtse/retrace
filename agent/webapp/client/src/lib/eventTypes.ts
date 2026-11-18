/**
 * Trace event types - must stay in sync with `core.agent` `self.trace(...)`
 * dispatch sites and `webapp/server/runner.py` (SENTINEL_DONE / SENTINEL_FAIL).
 */

export type EventName =
  | "iteration"
  | "thinking"
  | "tool_call"
  | "tool_result"
  | "error"
  | "ranked_table"
  | "mini_status"
  | "saved_threshold_nudge"
  | "iteration_refunded"
  | "wallclock_exceeded"
  | "soft_threshold_reached"
  | "llm_self_complete_honored"
  | "llm_self_complete_ignored"
  | "checkpoint_compacted"
  | "backtrack_compacted"
  | "native_bailout"
  | "prefetch"
  | "_done"
  | "_failed"
  | "_stream_error"
  | string;     // forward-compat

export interface TraceEvent {
  event: EventName;
  payload: Record<string, any>;
  ts: number;
}

export interface PresetInfo {
  family: "claude" | "gpt5_5";
  preset: "fast" | "standard" | "thorough";
  model: string;
  reasoning_effort: string;
  max_iterations: number;
  stop_after_distinct_routes: number;
}

export interface RouteEntry {
  label: string;
  total_score: number;
  steps: string[];
  sub_scores: Record<string, number>;
  starting_materials: string[];
  notes: string;
  provenance: string;
  rag_source_ids?: string[] | null;
  created_at?: number;
}

export interface RunCreated {
  session_id: string;
  effective: Record<string, any>;
}
