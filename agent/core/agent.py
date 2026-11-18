"""
Chemoenzymatic Retrosynthesis Agent
====================================
A ReACT-style agent that uses an LLM to plan multi-step chemoenzymatic
retrosynthesis routes. The LLM replaces tree search (MCTS*), autonomously
deciding which molecule to decompose and which precursors to pick at each step.

The LLM uses its own chemistry knowledge to propose retrosynthetic disconnections,
and uses tools to validate SMILES and check if molecules are commercially available.
"""

import json
import logging
import os
import time
from copy import deepcopy
from typing import Any

from core.iteration import (
    DEFAULT_ITERATION_BILLABLE_TOOLS,
    _is_iteration_billable,
    _refund_tool_names,
)
from core.llm_client import LLMClient
from core.prompt_loader import PromptLoader
from tools.base import ToolRegistry
from services.concurrent_tools import execute_tools_concurrent
from services.autocompact import CompactionConfig, maybe_compact, compact_to_checkpoint
from services.result_store import ResultStore
from services.transcript import TranscriptRecorder
from services.reaction_ordering_review import parse_reviewer_decision

logger = logging.getLogger(__name__)

# Default retroplanner HTTP endpoint must be configured via
# built-in config (`neural_retro.retroplanner_http_url` /
# `pistachio_ringbreaker_url`) or the matching env vars
# (RETROPLANNER_HTTP_URL / RINGBREAKER_API_URL). Hardcoded IPs were
# removed in May 2026 so the repo can be cloned without leaking the
# production deploy's intranet address. If neither config nor env is set
# the affected engine silently disables itself in NeuralRetroTool — the
# rest of the ensemble still works.


class RetrosynthesisAgent:
    """ReACT-loop retrosynthesis agent."""

    def __init__(self, llm_client: LLMClient, tool_registry: ToolRegistry,
                 max_depth: int = 6, max_iterations: int = 50,
                 trace_callback=None,
                 compaction_config: CompactionConfig | None = None,
                 todo_store=None,
                 scratchpad_store=None,
                 # SOFT exit threshold: once the scratchpad holds this many
                 # DISTINCT saved routes (dedup enforced by
                 # tools/scratchpad.py), the loop emits a one-shot reminder
                 # telling the LLM it may end the loop by writing
                 # `COMPLETE ROUTE FOUND:`. After the reminder, the LLM's
                 # stop marker is honoured (instead of being logged-and-
                 # ignored as it is pre-threshold). max_iterations and
                 # wallclock_s remain the absolute upper bounds. Raised
                 # 5→10 (2026-05-19): the hard-exit-at-5 floor was leaving
                 # 75 % of the 50-iter retro-search budget unused on cases
                 # where the model had more useful exploration to do.
                 stop_after_distinct_routes: int = 10,
                 result_store: ResultStore | None = None,
                 system_prompt: str | None = None,
                 iteration_billable_tools: list[str] | None = None,
                 prompts: PromptLoader | None = None,
                 auto_finish_after_saved: int | None = 5,
                 precedent_prefetch_enabled: bool = False,
                 precedent_prefetch_k: int = 3,
                 precedent_prefetch_source: str = "mixed",
                 # Minimum top-1 Tanimoto similarity for the prefetch to
                 # actually inject the recall_similar_routes payload. When
                 # the top hit falls below this floor (i.e. confidence band
                 # is "low" or "none"), the synthetic tool_call envelope is
                 # kept (audit trail) but the content is replaced with a
                 # one-line suppression stub. Defaults to thresholds.medium
                 # (0.4) when read from yaml; raw hits remain reachable via
                 # load_result(<prefetch_call_id>).
                 precedent_prefetch_min_similarity: float = 0.4,
                 reaction_ordering_block: str | None = None,
                 reviewer_client: LLMClient | None = None,
                 postprocess_config: dict | None = None,
                 effective_config: dict | None = None,
                 prompt_artifacts: dict | None = None,
                 # Per-tool wallclock budget (seconds). Bounds any single
                 # tool call inside execute_tools_concurrent so a wedged
                 # HTTP engine cannot freeze the loop. Default mirrors
                 # services/concurrent_tools.DEFAULT_TOOL_TIMEOUT_S.
                 tool_timeout_s: float | None = None,
                 # Per-run wallclock budget (seconds). Hard upper bound on
                 # `.run()` regardless of iteration count. None disables.
                 # Default = 90 minutes — long enough for the worst hard
                 # target with full ensemble, short enough that a single
                 # stuck run can't hold up an eval sweep overnight.
                 wallclock_s: float | None = 5400.0):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_depth = max_depth
        # Stash the user-provided trace; `run()` wraps it in a
        # TranscriptRecorder so every event also lands on disk for replay.
        self._user_trace = trace_callback
        self.trace = trace_callback or (lambda event, **kw: None)
        self.max_iterations = max_iterations
        self.compaction_config = compaction_config or CompactionConfig()
        self.todo_store = todo_store
        self.scratchpad_store = scratchpad_store
        # SOFT exit threshold: once the scratchpad has saved at least this
        # many DISTINCT routes, the loop emits a one-shot reminder telling
        # the LLM it may emit `COMPLETE ROUTE FOUND:` to stop. After that
        # reminder fires the LLM's stop marker is honoured; before it, the
        # marker is logged-and-ignored. max_iterations / wallclock_s are
        # still the absolute upper bounds.
        self.stop_after_distinct_routes = int(stop_after_distinct_routes)
        self.result_store = result_store
        if system_prompt is None:
            raise ValueError(
                "RetrosynthesisAgent requires a system_prompt. "
                "Set prompts.core in the built-in config or config override."
            )
        self.system_prompt = system_prompt
        self.iteration_billable_tools = list(
            iteration_billable_tools or DEFAULT_ITERATION_BILLABLE_TOOLS
        )
        self.prompts = prompts
        # Auto-finish soft-converge knob. When the scratchpad has at least
        # this many distinct routes (dedup is enforced in tools/scratchpad.py),
        # the loop pushes a single nudge telling the LLM to wrap up. None
        # disables the nudge entirely.
        self.auto_finish_after_saved = auto_finish_after_saved
        # Per-tool wallclock + per-run wallclock. Imported here (lazy) to
        # avoid a circular import on services.concurrent_tools.
        from services.concurrent_tools import DEFAULT_TOOL_TIMEOUT_S
        self.tool_timeout_s = float(
            tool_timeout_s if tool_timeout_s is not None else DEFAULT_TOOL_TIMEOUT_S
        )
        self.wallclock_s = (
            float(wallclock_s) if wallclock_s is not None else None
        )
        # Track buyability checks: canonical SMILES -> bool (buyable=true/false).
        # We need the bool, not just "was it checked" — otherwise a leaf that
        # came back buyable=false would still pass the gate.
        self._buyability_checked: dict[str, bool] = {}
        # Whether the most recent COMPLETE-attempt's buyability gate confirmed
        # all leaves buyable. Initialised True so first-time completion attempts
        # aren't spuriously blocked before the gate has run.
        self._all_leaves_buyable_flag: bool = True
        # Tracks whether suggest_templates and neural_retro have been called
        # on the target. Populated by the prefetch (which always runs both).
        self._target_tools_called_set: set[str] = set()
        # Target SMILES for the current run; set at the top of
        # `_run_with_native_tools` so helpers can reach it.
        self._target_smiles: str | None = None
        # Precedent recall prefetch knobs (used by _prefetch_disconnections).
        self.precedent_prefetch_enabled = bool(precedent_prefetch_enabled)
        self.precedent_prefetch_k = int(precedent_prefetch_k)
        self.precedent_prefetch_source = str(precedent_prefetch_source or "mixed")
        self.precedent_prefetch_min_similarity = float(precedent_prefetch_min_similarity)
        self._reaction_ordering_block = reaction_ordering_block
        if self._reaction_ordering_block is None:
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _section_path = os.path.join(
                _repo_root,
                "prompts", "sections", "core_reaction_ordering.md",
            )
            try:
                with open(_section_path) as _f:
                    self._reaction_ordering_block = _f.read().rstrip("\n")
            except FileNotFoundError:
                logger.warning(
                    "reaction-ordering section missing at %s; system prompt "
                    "will omit ordering guidance.",
                    _section_path,
                )
                self._reaction_ordering_block = ""
        self.reviewer_client = reviewer_client
        self.postprocess_config = dict(postprocess_config or {})
        self.effective_config = effective_config or {}
        self.prompt_artifacts = prompt_artifacts or {}
        # System-prompt block for precedent recall guidance. Rendered into
        # the `{precedent_recall_block}` placeholder in prompts/core.md so
        # the LLM is only told about `recall_similar_routes` when the tool
        # is actually registered. When precedent_recall is disabled in
        # config the tool is not registered and the block becomes empty,
        # keeping the prompt/tool surface consistent. The block text lives
        # in prompts/sections/core_precedent_recall.md — loaded the same
        # way build_agent loads core.md (file read, no inline LLM strings).
        self._precedent_recall_block = ""
        if self.precedent_prefetch_enabled:
            # prompts/ lives at the repo root, NOT under core/. __file__ is
            # core/agent.py, so dirname(__file__) is core/ — go up one level.
            _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _section_path = os.path.join(
                _repo_root,
                "prompts", "sections", "core_precedent_recall.md",
            )
            try:
                with open(_section_path) as _f:
                    self._precedent_recall_block = _f.read().rstrip("\n")
            except FileNotFoundError:
                logger.warning(
                    "precedent_recall enabled but section template missing at %s; "
                    "system prompt will omit recall guidance.",
                    _section_path,
                )
        # Sliding window of tool names from the previous K assistant turns,
        # used to attribute provenance to save_best_route calls. Each entry
        # is a list of tool names called in one turn (the order they
        # appeared). The window holds the most recent 3 turns.
        self._recent_tool_turns: list[list[str]] = []
        # Most recent RAG payload (parsed dict from recall_similar_routes)
        # — used to pull route_ids when a save_best_route is attributed to
        # ``rag_anchored``. Stays until the next ``recall_similar_routes``
        # call overrides it (or until ``run()`` resets state). Because the
        # provenance check ALSO requires ``recall_similar_routes`` to be
        # present in the recent_tool_turns window (3 turns), a stale
        # payload here can't accidentally tag an unrelated save —
        # the window will have rotated past it.
        self._last_recall_payload: dict | None = None


    def _active_recorder(self) -> TranscriptRecorder | None:
        return self.trace if isinstance(self.trace, TranscriptRecorder) else None

    @staticmethod
    def _client_settings(client: Any) -> dict:
        if client is None:
            return {}
        return {
            "base_url": getattr(client, "base_url", None),
            "model": getattr(client, "model", None),
            "temperature": getattr(client, "temperature", None),
            "max_tokens": getattr(client, "max_tokens", None),
            "reasoning_effort": getattr(client, "reasoning_effort", None),
            "responses_state": getattr(client, "responses_state", None),
            "max_retries": getattr(client, "max_retries", None),
            "read_timeout": getattr(client, "read_timeout", None),
        }

    def _model_settings(self) -> dict:
        return {
            "main": self._client_settings(self.llm),
            "reviewer": self._client_settings(self.reviewer_client),
        }

    def _write_manifest(self, target_smiles: str) -> None:
        recorder = self._active_recorder()
        if recorder is None:
            return
        recorder.update_metadata(
            config=self.effective_config,
            model_settings=self._model_settings(),
            prompt_artifacts=self.prompt_artifacts,
            resume_state={
                "session_dir": recorder.session_dir,
                "latest_messages_path": os.path.join(
                    recorder.session_dir, "conversation", "latest_messages.json"
                ),
                "llm_requests_path": os.path.join(
                    recorder.session_dir, "conversation", "llm_requests.jsonl"
                ),
            },
        )
        recorder.write_session_manifest(
            config=self.effective_config,
            model_settings=self._model_settings(),
            prompt_artifacts=self.prompt_artifacts,
            postprocess=recorder.metadata.get("postprocess"),
            tool_names=list(self.tools.tool_names),
            paths={
                "transcript_json": os.path.join(recorder.session_dir, "transcript.json"),
                "transcript_jsonl": os.path.join(recorder.session_dir, "transcript.jsonl"),
                "latest_messages": os.path.join(
                    recorder.session_dir, "conversation", "latest_messages.json"
                ),
                "llm_requests": os.path.join(
                    recorder.session_dir, "conversation", "llm_requests.jsonl"
                ),
            },
            target_smiles=target_smiles,
        )

    def _record_llm_request(
        self,
        *,
        phase: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        client: Any | None = None,
        extra: dict | None = None,
    ) -> None:
        recorder = self._active_recorder()
        if recorder is None:
            return
        recorder.record_llm_request(
            phase=phase,
            messages=deepcopy(messages),
            model_settings=self._client_settings(client or self.llm),
            tools=deepcopy(tools) if tools is not None else None,
            extra=extra,
        )

    def _write_latest_messages(
        self,
        messages: list[dict],
        *,
        stage: str,
        iteration: int | None = None,
    ) -> None:
        recorder = self._active_recorder()
        if recorder is not None:
            recorder.write_latest_messages(
                deepcopy(messages),
                stage=stage,
                iteration=iteration,
            )

    @staticmethod
    def _compact_json_for_prompt(value: Any, max_chars: int = 40_000) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str, indent=2)
        except Exception:
            text = str(value)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"

    def _reviewer_message_text(self, target_smiles: str, result: dict) -> str:
        return self.prompts.render(
            "reaction_ordering_review",
            reaction_ordering_rules=self._reaction_ordering_block,
            target_smiles=target_smiles,
            route_text=result.get("route_text", "") or "",
            ranked_json=self._compact_json_for_prompt(result.get("ranked")),
            all_routes_json=self._compact_json_for_prompt(result.get("all_routes")),
        )

    def _extract_response_text(self, response: dict) -> str:
        choices = response.get("choices") or []
        for choice in choices:
            msg = choice.get("message") or {}
            if msg.get("content"):
                return str(msg["content"])
        return ""

    def _write_review_artifact(
        self,
        *,
        round_index: int,
        messages: list[dict],
        raw_output: str,
        decision: dict,
    ) -> str | None:
        recorder = self._active_recorder()
        if recorder is None:
            return None
        path = os.path.join(
            recorder.session_dir,
            "conversation",
            f"reaction_ordering_review_round_{round_index}.json",
        )
        payload = {
            "schema_version": 1,
            "round": round_index,
            "model_settings": self._client_settings(self.reviewer_client),
            "messages": messages,
            "raw_output": raw_output,
            "decision": decision,
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.debug("review artifact write failed: %s", exc)
            return None
        return path

    def _run_reaction_ordering_postprocess(
        self,
        target_smiles: str,
        result: dict,
    ) -> dict:
        cfg = self.postprocess_config or {}
        if not cfg.get("enabled", False):
            payload = {"enabled": False, "status": "skipped"}
            result["postprocess"] = payload
            self._update_postprocess_metadata(payload)
            return result
        if self.reviewer_client is None:
            payload = {
                "enabled": True,
                "status": "failed",
                "error": "reviewer client is not configured",
            }
            result["postprocess"] = payload
            self._update_postprocess_metadata(payload)
            return result
        if int(cfg.get("max_review_rounds", 1) or 0) <= 0:
            payload = {
                "enabled": True,
                "status": "skipped",
                "reason": "max_review_rounds <= 0",
            }
            result["postprocess"] = payload
            self._update_postprocess_metadata(payload)
            return result

        review_messages = [
            {"role": "user", "content": self._reviewer_message_text(target_smiles, result)}
        ]
        raw_output = ""
        review_path = None
        try:
            self._record_llm_request(
                phase="reaction_ordering_review",
                messages=review_messages,
                client=self.reviewer_client,
                extra={"round": 1},
            )
            response = self.reviewer_client.chat_with_override(
                review_messages,
                tools=None,
                max_tokens=int(cfg.get("max_tokens", getattr(self.reviewer_client, "max_tokens", 32000))),
                temperature=float(cfg.get("temperature", 0.0)),
                reasoning_effort=str(cfg.get("reasoning_effort", "xhigh")),
                use_responses_state=False,
            )
            raw_output = self._extract_response_text(response)
            decision = parse_reviewer_decision(raw_output)
        except Exception as exc:
            decision = {
                "status": "failed",
                "error": str(exc),
                "raw_output": raw_output,
            }

        review_path = self._write_review_artifact(
            round_index=1,
            messages=review_messages,
            raw_output=raw_output,
            decision=decision,
        )
        postprocess_payload = {
            "enabled": True,
            "status": decision.get("status"),
            "action": decision.get("action"),
            "summary": decision.get("summary"),
            "issues": decision.get("issues") or [],
            "review_path": review_path,
            "error": decision.get("error"),
        }

        if decision.get("status") != "ok":
            result["postprocess"] = postprocess_payload
            self._update_postprocess_metadata(postprocess_payload)
            return result

        action = decision.get("action")
        if action == "accept":
            postprocess_payload["status"] = "accepted"
            result["postprocess"] = postprocess_payload
            self._update_postprocess_metadata(postprocess_payload)
            return result

        if action == "reorder":
            # The reviewer may only *permute* the verified steps — never add,
            # drop, or mutate them. We prove that structurally: parse the
            # reviewer's rewrite back into steps and require its order-
            # insensitive route_fingerprint to equal the selected route's.
            # Any divergence means the rewrite changed the chemistry, so we
            # REJECT it and keep the original structure (user-chosen behaviour).
            from core.route_text import parse_route_from_text
            from tools._route_fp import route_fingerprint

            original = result.get("route_text", "") or ""
            new_text = decision.get("optimized_route_text", "") or ""

            # Original structure: the steps the selected route_text was
            # synthesised from (best_route is resolved to the selected ranked
            # entry). Fall back to parsing the original text if absent.
            best_route = result.get("best_route") or {}
            orig_steps_struct = list(best_route.get("steps") or [])
            if not orig_steps_struct:
                orig_steps_struct = [
                    f"{s['product']}>>{s['reactants']}"
                    for s in parse_route_from_text(original)
                    if s.get("product") and s.get("reactants")
                ]

            new_parsed = parse_route_from_text(new_text)
            new_steps_struct = [
                f"{s['product']}>>{s['reactants']}"
                for s in new_parsed
                if s.get("product") and s.get("reactants")
            ]

            if not new_steps_struct:
                # Could not recover any reaction step from the rewrite.
                postprocess_payload["status"] = "reorder_rejected_unparseable"
                postprocess_payload["reason"] = (
                    "reviewer's optimized_route_text yielded zero parseable steps"
                )
                postprocess_payload["original_route_text"] = original
                postprocess_payload["rejected_route_text"] = new_text
                result["postprocess"] = postprocess_payload
                self._update_postprocess_metadata(postprocess_payload)
                return result

            orig_fp = route_fingerprint(orig_steps_struct)
            new_fp = route_fingerprint(new_steps_struct)

            if orig_fp == new_fp:
                # True permutation — accept the reordered presentation.
                result["route_text"] = new_text
                postprocess_payload["status"] = "reordered"
                postprocess_payload["original_route_text"] = original
                postprocess_payload["route_fingerprint"] = new_fp
            else:
                # Add / drop / mutate disguised as a reorder — reject and leave
                # route_text + all structures untouched.
                postprocess_payload["status"] = "reorder_rejected_non_permutation"
                postprocess_payload["reason"] = (
                    "rewrite is not a pure permutation of the verified steps "
                    "(step multiset changed)"
                )
                postprocess_payload["original_fingerprint"] = orig_fp
                postprocess_payload["rewrite_fingerprint"] = new_fp
                postprocess_payload["n_orig_steps"] = len(orig_steps_struct)
                postprocess_payload["n_rewrite_steps"] = len(new_steps_struct)
                postprocess_payload["rejected_route_text"] = new_text
            result["postprocess"] = postprocess_payload
            self._update_postprocess_metadata(postprocess_payload)
            return result

        if action == "revise":
            remaining = max(0, self.max_iterations - int(result.get("iterations") or 0))
            postprocess_payload["feedback"] = decision.get("feedback")
            if remaining <= 0:
                postprocess_payload["status"] = "blocked"
                postprocess_payload["error"] = "no remaining iterations for revision"
                result["postprocess"] = postprocess_payload
                self._update_postprocess_metadata(postprocess_payload)
                return result
            feedback_msg = {
                "role": "user",
                "content": self.prompts.render(
                    "nudge_reaction_ordering_feedback",
                    summary=decision.get("summary") or "",
                    feedback=decision.get("feedback") or "",
                    issues_json=json.dumps(
                        decision.get("issues") or [],
                        ensure_ascii=False,
                        default=str,
                    ),
                ),
            }
            messages = deepcopy(result.get("messages") or [])
            messages.append(feedback_msg)
            self._write_latest_messages(
                messages,
                stage="reaction_ordering_feedback",
                iteration=int(result.get("iterations") or 0),
            )
            continued = self._run_with_native_tools(
                target_smiles,
                initial_messages=messages,
                max_iterations=remaining,
                iteration_offset=int(result.get("iterations") or 0),
            )
            continued["time"] = float(result.get("time") or 0.0) + float(
                continued.get("time") or 0.0
            )
            postprocess_payload["status"] = "continued"
            postprocess_payload["continuation_iterations"] = max(
                0,
                int(continued.get("iterations") or 0) - int(result.get("iterations") or 0),
            )
            continued["postprocess"] = postprocess_payload
            self._update_postprocess_metadata(postprocess_payload)
            return continued

        result["postprocess"] = postprocess_payload
        self._update_postprocess_metadata(postprocess_payload)
        return result

    def _update_postprocess_metadata(self, payload: dict) -> None:
        recorder = self._active_recorder()
        if recorder is None:
            return
        recorder.update_metadata(postprocess=payload)
        self._write_manifest(self._target_smiles or "")

    def _compute_provenance(self) -> tuple[str, list[str] | None]:
        """Infer the generation path that produced any save_best_route
        about to fire this turn, based on the last 3 turns of tool calls
        (sliding window in ``self._recent_tool_turns``) plus the prefetch
        bookkeeping.

        Priority (most specific wins):
          1. rag_anchored — recall_similar_routes appeared in the window
             AND the cached recall payload has examples.
          2. neural_consensus — neural_retro appeared in the window.
          3. template_suggested — suggest_templates appeared in the window.
          4. zero_shot — none of the above.

        Returns ``(provenance_label, rag_source_ids_or_None)``.
        """
        names = {n for turn in self._recent_tool_turns for n in turn}
        # The prefetch always seeds the target's suggest_templates and
        # neural_retro before the loop even starts; treat them as in-
        # window for the FIRST save_best_route attempt so we don't tag
        # everything zero_shot on case-1 turn-2 saves.
        if not self._recent_tool_turns:
            names |= self._target_tools_called_set
        if "recall_similar_routes" in names and self._last_recall_payload:
            examples = self._last_recall_payload.get("examples") or []
            rag_ids = [
                e.get("route_id") for e in examples
                if isinstance(e, dict) and e.get("route_id")
            ]
            return "rag_anchored", rag_ids or None
        if "neural_retro" in names:
            return "neural_consensus", None
        if "suggest_templates" in names:
            return "template_suggested", None
        return "zero_shot", None


    def _canonicalize_smiles(self, smiles: str) -> str:
        """Canonicalize SMILES for consistent tracking."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles
            return Chem.MolToSmiles(mol)
        except Exception:
            return smiles

    @staticmethod
    def _is_error_payload(content: str) -> bool:
        """True if a tool-result string is an error / timeout payload.

        Concurrent execution wraps tool failures and timeouts as JSON with a
        top-level `"error"` key (see services/concurrent_tools.py). Such
        payloads must not be memoized by the short-circuit cache.
        """
        if not content:
            return True
        try:
            parsed = json.loads(content)
        except Exception:
            return False
        return isinstance(parsed, dict) and "error" in parsed

    def _build_mini_status(self, iteration: int) -> str:
        """One-line status snapshot pushed every 5 iters. The iteration
        counter shows retro-search calls used (suggest_templates +
        neural_retro), not turn count — see _is_iteration_billable.
        """
        best_score = None
        try:
            if self.scratchpad_store is not None:
                best = self.scratchpad_store.best()
                if best is not None:
                    best_score = round(best.total_score, 3)
        except Exception:
            best_score = None
        best_str = f"{best_score:.3f}" if best_score is not None else "—"
        return self.prompts.render(
            "status_mini",
            iter=iteration,
            max_iter=self.max_iterations,
            best_str=best_str,
        )

    def _maybe_refund_iteration(self, iteration: int, tool_calls) -> int:
        """If `tool_calls` for this turn were all non-billable, refund.

        Returns the (possibly decremented) iteration count.
        """
        if not _is_iteration_billable(tool_calls, self.iteration_billable_tools):
            iteration -= 1
            logger.info(
                "Iteration %d refunded (non-billable turn: %s)",
                iteration + 1, _refund_tool_names(tool_calls),
            )
            try:
                self.trace(
                    "iteration_refunded",
                    reason="non-billable turn",
                    tools=_refund_tool_names(tool_calls),
                )
            except Exception as exc:
                logger.debug("iteration_refunded trace failed: %s", exc)
        return iteration

    def _fail_native_with_best_so_far(
        self,
        target_smiles: str,
        iteration: int,
        t0: float,
        error: str,
        messages: list[dict],
    ) -> dict:
        """Bail out of the native loop after unrecoverable LLM errors.

        Historically this site fell back to a text-ReACT mode that bypassed
        every structural completion gate (buyability / single-step cheat /
        ≥3 distinct routes) and silently rubber-stamped the LLM's next
        "COMPLETE ROUTE FOUND" string. That fallback has been removed.
        Returning the best route saved on the scratchpad is honest: if the
        agent did real work before the crash, the user gets it back;
        otherwise success=False is reported.
        """
        best = None
        best_dict = None
        n_saved = 0
        if self.scratchpad_store is not None:
            try:
                best = self.scratchpad_store.best()
                n_saved = len(self.scratchpad_store.list())
                if best is not None:
                    best_dict = best.to_dict()
            except Exception as exc:
                logger.warning("scratchpad inspection during fail-out failed: %s", exc)

        # Synthesise a human-facing route_text from the saved entry's structure
        # (RouteEntry.to_dict has no route_text key, so the old
        # `best_dict.get("route_text","")` was always empty) and apply the same
        # hard buyability gate the main return uses. success must mean the same
        # thing on the bail path as on the happy path: a non-empty route whose
        # leaves are all buyable — never a bare "I saved something" flag.
        route_text = ""
        leaves_buyable = False
        if best_dict is not None:
            lines = [f"Step {i}: {s}" for i, s in enumerate(best_dict.get("steps") or [], start=1)]
            sm = best_dict.get("starting_materials") or []
            if sm:
                lines.append("Starting materials: " + ", ".join(sm))
            steps_text = "\n".join(lines)
            header = (
                f"COMPLETE ROUTE FOUND: (best-so-far at bailout, "
                f"label={best_dict.get('label')}, "
                f"total_score={best_dict.get('total_score')})"
            )
            route_text = header + ("\n" + steps_text if steps_text else "")
            if best is not None:
                try:
                    leaves_buyable = self._verify_route_leaves_buyable(best)
                except Exception as exc:
                    logger.warning("buyability gate during fail-out failed (%s); fail-closed", exc)
                    leaves_buyable = False

        success = bool(best_dict) and bool(route_text) and leaves_buyable
        logger.error(
            "native loop bailed out at iter=%d (saved=%d, leaves_buyable=%s, success=%s); error=%s",
            iteration, n_saved, leaves_buyable, success, error,
        )
        try:
            self.trace(
                "native_bailout",
                iteration=iteration,
                n_saved=n_saved,
                error=error,
            )
        except Exception:
            pass
        return {
            "success": success,
            "below_target": True,
            "route_text": route_text,
            "best_route": best_dict,
            "leaves_buyable": leaves_buyable,
            "iterations": iteration,
            "time": time.time() - t0,
            "messages": messages,
            "error": error,
            "bailed_out": True,
        }

    # ------------------------------------------------------------------
    # Completion-gate helpers
    # ------------------------------------------------------------------

    def _all_leaves_buyable(self) -> bool:
        """Return True if the buyability soft gate has confirmed all leaves
        of the most recent COMPLETE-attempt are buyable, OR if it has not
        yet had reason to fire (in which case the soft gate downstream will
        still enforce the check).
        """
        return bool(self._all_leaves_buyable_flag)

    def _target_tools_called(self) -> bool:
        """Return True if both `suggest_templates` and `neural_retro` have
        been called on the target. Set by the prefetch (which always runs
        both)."""
        return {"suggest_templates", "neural_retro"}.issubset(
            self._target_tools_called_set
        )

    def _distinct_routes_count(self) -> int:
        """Count of saved candidate routes (approximation of distinct)."""
        return len(self.scratchpad_store.list()) if self.scratchpad_store else 0

    def _auto_capture_scored_route(
        self,
        *,
        tool_call_id: str,
        arguments: dict,
        result: dict,
    ) -> bool:
        """Durably capture a valid ``score_route`` candidate without waiting
        for the model to repeat the exact route through ``save_best_route``.

        The score tool has already checked route grounding and stamped a
        structural fingerprint.  Requiring a second lossy LLM transcription
        created a large closure gap: the model frequently altered a step or
        cited the wrong result id between score and save.  This deterministic
        bridge uses the *exact scored payload*, then leaves the normal final
        buyability gate in charge.  It never invents or repairs chemistry.
        """
        if not isinstance(result, dict) or result.get("error"):
            return False
        grounding = result.get("route_grounding") or {}
        if grounding and grounding.get("checked", True) and not grounding.get("ok"):
            return False
        score = result.get("total_score")
        # Accept the same argument aliases score_route itself tolerates
        # (steps / route / route_json) — the LLM has been observed emitting
        # them, and RouteVerifierTool.execute normalizes identically, so a
        # successfully scored route must not silently skip auto-capture just
        # because it arrived under an alias. (Codex review 2026-07-18.)
        raw_steps = arguments.get("route_steps")
        if raw_steps is None:
            raw_steps = (arguments.get("steps") or arguments.get("route")
                         or arguments.get("route_json"))
        if isinstance(raw_steps, str):
            try:
                raw_steps = json.loads(raw_steps)
            except Exception:
                return False
        if not isinstance(raw_steps, list) or not raw_steps:
            return False
        steps: list[str] = []
        for step in raw_steps:
            if not isinstance(step, dict) or not step.get("product"):
                return False
            reactants = step.get("reactants") or []
            if not isinstance(reactants, list) or not reactants:
                return False
            product = str(step["product"]).strip()
            rhs = ".".join(str(r).strip() for r in reactants if str(r).strip())
            if not rhs:
                return False
            steps.append(f"{product}>>{rhs}")
        save_tool = self.tools.get("save_best_route") if self.tools else None
        if save_tool is None:
            return False
        try:
            raw = save_tool.execute(
                label=f"auto-scored-{str(tool_call_id)[-8:]}",
                steps=steps,
                total_score=float(score or 0.0),
                sub_scores={
                    k: float(v) for k, v in result.items()
                    if k in {"feasibility_score", "cost_score", "greenness_score"}
                    and isinstance(v, (int, float))
                },
                starting_materials=[],
                notes=(
                    "Automatically captured from the exact score_route payload; "
                    "final structural-leaf buyability is still checked at run exit."
                ),
                score_route_result_id=str(tool_call_id),
                _provenance="deterministic_closure",
            )
            payload = json.loads(raw) if isinstance(raw, str) else raw
            ok = isinstance(payload, dict) and bool(payload.get("ok") or payload.get("saved"))
            if ok:
                self.trace(
                    "auto_capture_scored_route",
                    source_tool_call_id=tool_call_id,
                    total_score=float(score or 0.0),
                    route_fp=result.get("route_fp"),
                )
            return ok
        except Exception as exc:
            logger.warning("auto-capture of scored route failed: %s", exc)
            return False

    def _score_unscored_routes(self) -> None:
        """Run score_route on any saved entry whose total_score is None or
        whose sub_scores dict is empty. Used by the finalisation phase to
        ensure the rank_routes table is complete."""
        if self.scratchpad_store is None:
            return
        score_tool = self.tools.get("score_route")
        if score_tool is None:
            return
        for entry in self.scratchpad_store.all():
            if entry.sub_scores and entry.total_score:
                continue
            try:
                steps_payload = []
                for s in entry.steps:
                    if ">>" not in s:
                        continue
                    product, _, reactants = s.partition(">>")
                    reactant_list = [r.strip() for r in reactants.split(".") if r.strip()]
                    steps_payload.append({
                        "reactants": reactant_list,
                        "product": product.strip(),
                    })
                if not steps_payload:
                    continue
                raw = score_tool.execute(
                    smiles=self._target_smiles,
                    route_steps=steps_payload,
                )
                # Best-effort: if score_tool didn't write back, attach
                # the parsed result onto the entry.
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and "total_score" in parsed:
                    entry.total_score = float(parsed.get("total_score") or 0.0)
                    sub = {}
                    # score_route emits flat pillar keys (`feasibility_score`,
                    # `cost_score`, `greenness_score`); fall back to the legacy
                    # nested `{pillar: {"score": ...}}` shape for robustness.
                    for key in ("feasibility", "cost", "greenness"):
                        flat = parsed.get(f"{key}_score")
                        if isinstance(flat, (int, float)):
                            sub[key] = flat
                            continue
                        v = parsed.get(key)
                        if isinstance(v, dict) and "score" in v:
                            sub[key] = v["score"]
                        elif isinstance(v, (int, float)):
                            sub[key] = v
                    if sub:
                        entry.sub_scores = {**(entry.sub_scores or {}), **sub}
            except Exception as e:
                logger.warning("finalisation: failed to score %s: %s",
                               entry.label, e)

    def _finalise_phase(self) -> dict | None:
        """Score any unscored saved routes and return the ranked payload.

        Best-effort: any failure is swallowed so the caller can still
        synthesise a result from whatever routes are present.
        """
        try:
            self._score_unscored_routes()
        except Exception as e:
            logger.warning("finalisation: _score_unscored_routes failed: %s", e)
        ranked: dict | None = None
        rank_tool = self.tools.get("rank_routes")
        if rank_tool is not None:
            try:
                raw = rank_tool.execute()
                ranked = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                logger.warning("finalisation: rank_routes failed: %s", e)
                ranked = None
        try:
            self.trace("ranked_table", payload=ranked)
        except Exception:
            pass
        return ranked

    def _verify_route_leaves_buyable(self, entry) -> bool:
        """Hard buyability gate for a finalised route.

        The prompt and the cost pillar are *soft* buyability signals (the LLM
        is asked to check, low-buyability routes score lower) but nothing
        previously *enforced* that a route declared successful actually
        terminates in purchasable leaves. This closes that gap: every leaf of
        `entry` must be buyable. Leaves already confirmed via the per-turn
        `_buyability_checked` tracker are reused; any not yet checked are
        verified now with a single batched `check_buyability` call.

        Returns True only if every leaf is confirmed buyable. Fail-closed:
        an unresolvable / unparseable leaf counts as not-buyable.

        The leaves are derived STRUCTURALLY from the route's steps (reactants
        never produced by any step), NOT taken from the LLM-supplied
        `starting_materials` field. That field is unvalidated: a route can
        declare a couple of trivially-buyable blocks while its steps actually
        decompose to entirely different, non-buyable precursors, sailing
        through the only hard completion gate. We check the true structural
        leaves; the declared field is used only as a fallback when the steps
        can't be parsed, and any disagreement is logged.
        """
        from tools.scratchpad import derive_leaf_smiles

        declared = [s for s in (getattr(entry, "starting_materials", None) or [])
                    if isinstance(s, str)]
        structural = derive_leaf_smiles(getattr(entry, "steps", None) or [])

        if structural is not None:
            canon_leaves = structural
            # Surface a mismatch between the declared field and the true leaves
            # — a divergence is exactly the mis-declared-starting-materials
            # evasion this gate now defends against.
            declared_canon = set()
            for s in declared:
                c = self._canonicalize_smiles(s)
                if c:
                    declared_canon.add(c)
            if declared_canon != set(canon_leaves):
                logger.info(
                    "buyability gate: declared starting_materials %s != structural "
                    "leaves %s — gating on structural leaves",
                    sorted(declared_canon), canon_leaves,
                )
            if not canon_leaves:
                # Steps parsed but yielded no leaves (e.g. every reactant is also
                # a product — a cyclic/degenerate route). Cannot confirm.
                return False
        else:
            # Steps unparseable / RDKit unavailable → fall back to the declared
            # field rather than crash, but keep the same fail-closed posture.
            if not declared:
                return False
            canon_leaves = []
            for leaf in declared:
                c = self._canonicalize_smiles(leaf)
                if not c:
                    return False
                canon_leaves.append(c)

        unknown = [c for c in canon_leaves if c not in self._buyability_checked]
        if unknown:
            tool = self.tools.get("check_buyability")
            if tool is None:
                return False
            try:
                raw = tool.execute(smiles_list=unknown)
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                logger.warning("buyability gate: check failed (%s); fail-closed", e)
                return False
            for row in (parsed or {}).get("results", []):
                c = self._canonicalize_smiles(row.get("input_smiles", "")) \
                    if isinstance(row.get("input_smiles"), str) else ""
                if c:
                    self._buyability_checked[c] = bool(row.get("buyable"))

        return all(self._buyability_checked.get(c, False) for c in canon_leaves)

    def _select_first_buyable_route(self, ranked: dict | None) -> int:
        """Walk the ranked routes (best score first) and return the index of the
        first whose leaves are all buyable, setting `_all_leaves_buyable_flag`
        accordingly. Returns 0 when no gate can run (no store / empty payload) so
        callers still surface rank-1; returns -1 when routes exist but none are
        fully buyable. The highest-scoring route often carries a single unbuyable
        leaf while a slightly lower-scoring one terminates in commercial blocks,
        so checking rank-1 alone produced false-negative FAILEDs."""
        if self.scratchpad_store is None or not isinstance(ranked, dict):
            return 0
        ranked_routes = ranked.get("routes") or []
        if not ranked_routes:
            return 0
        self._all_leaves_buyable_flag = False
        unbuyable_labels: list[str] = []
        try:
            entries = self.scratchpad_store.all()
        except Exception as e:
            logger.warning("buyability gate: could not resolve ranked entries: %s", e)
            entries = []
        for idx, route in enumerate(ranked_routes):
            label = route.get("label") if isinstance(route, dict) else None
            entry = next((e for e in entries if e.label == label), None)
            if entry is None:
                continue
            if self._verify_route_leaves_buyable(entry):
                self._all_leaves_buyable_flag = True
                if idx > 0:
                    logger.info(
                        "buyability gate: rank-1 had non-buyable leaves; "
                        "selected rank-%d route %s (all leaves buyable).",
                        idx + 1,
                        label,
                    )
                return idx
            unbuyable_labels.append(label or "?")
        logger.warning(
            "buyability gate: all %d ranked route(s) have non-buyable leaves; "
            "marking run unsuccessful. routes=%s",
            len(ranked_routes),
            unbuyable_labels,
        )
        return -1

    def _resolve_best_route(
        self, routes_sorted: list, ranked: dict | None, selected_index: int
    ) -> tuple[dict | None, dict | None]:
        """Return ``(best_route, score_best_route)`` dicts for the result.

        ``best_route`` must agree with the route the gate actually selected (and
        whose text was synthesised). When rank-1 has a non-buyable leaf the gate
        falls back to a lower-ranked all-buyable route; returning the
        top-by-score route here would ship ``success=True`` with rank-2 text
        while ``best_route`` points at the unbuyable rank-1. We therefore resolve
        ``best_route`` by the *selected* ranked label, falling back to the
        score-best entry only when the gate could not select (``selected_index <
        0``, or the label has no matching entry). ``score_best_route`` always
        carries the unconditional top-by-score entry so callers that want the
        raw best score still have it.
        """
        if not routes_sorted:
            return None, None
        score_best_dict = routes_sorted[0].to_dict()
        best_dict = score_best_dict
        selected_label = None
        if isinstance(ranked, dict) and selected_index >= 0:
            ranked_routes = ranked.get("routes") or []
            if 0 <= selected_index < len(ranked_routes):
                sel = ranked_routes[selected_index]
                selected_label = sel.get("label") if isinstance(sel, dict) else None
        if selected_label is not None:
            sel_entry = next(
                (r for r in routes_sorted if r.label == selected_label), None
            )
            if sel_entry is not None:
                best_dict = sel_entry.to_dict()
        return best_dict, score_best_dict

    def _synthesise_route_text_from_ranked(
        self, ranked: dict | None, *, route_index: int = 0
    ) -> str:
        """Emit a COMPLETE-ROUTE-FOUND-style text block for the ranked entry at
        `route_index` (0 = rank-1). The index lets the buyability gate select a
        lower-ranked route when the best-scoring one has non-buyable leaves."""
        if not ranked or not isinstance(ranked, dict):
            return ""
        routes = ranked.get("routes") or []
        if not routes:
            return ""
        if route_index < 0 or route_index >= len(routes):
            route_index = 0
        top = routes[route_index]
        # Prefer pulling steps/starting_materials directly from the saved entry.
        label = top.get("label")
        steps_text = ""
        if self.scratchpad_store is not None and label:
            entry = next((e for e in self.scratchpad_store.all() if e.label == label), None)
            if entry is not None:
                lines = []
                for i, s in enumerate(entry.steps, start=1):
                    lines.append(f"Step {i}: {s}")
                if entry.starting_materials:
                    lines.append("Starting materials: " + ", ".join(entry.starting_materials))
                steps_text = "\n".join(lines)
        header = (
            f"COMPLETE ROUTE FOUND: (rank-{route_index + 1} from finalisation, "
            f"label={label}, total_score={top.get('total_score')})"
        )
        return header + ("\n" + steps_text if steps_text else "")


    def run(
        self,
        target_smiles: str,
        *,
        resume_from: dict | None = None,
        feedback: str | None = None,
        extra_rounds: int | None = None,
    ) -> dict:
        """
        Run the agent to find a retrosynthesis route for target_smiles.
        Returns a dict with route info.

        Continuation (human-in-the-loop): when ``resume_from`` is given, the
        stores are NOT cleared — instead the prior scratchpad / todos are
        rehydrated and the agent continues from the prior message transcript
        with a chemist-``feedback`` message appended. ``resume_from`` is a dict
        carrying (any of) ``messages`` (the prior conversation), ``all_routes``
        (prior scratchpad ``to_dict`` payloads), ``todos``, and ``iterations``
        (the prior iteration count, used as the continuation offset).
        ``extra_rounds`` caps the continuation's iteration budget.
        """
        resuming = resume_from is not None
        # Fresh plan state for this run — prevents stale todos from a prior
        # target leaking if the agent instance is reused. On a continuation we
        # REHYDRATE the prior stores instead of clearing them, so the chemist's
        # feedback lands on top of the routes they were reviewing.
        if resuming:
            if self.scratchpad_store is not None:
                try:
                    self.scratchpad_store.restore(resume_from.get("all_routes") or [])
                except Exception as exc:
                    logger.warning("scratchpad restore failed: %s", exc)
            if self.todo_store is not None:
                try:
                    self.todo_store.restore(resume_from.get("todos") or [])
                except Exception as exc:
                    logger.warning("todo restore failed: %s", exc)
        else:
            if self.todo_store is not None:
                self.todo_store.clear()
            if self.scratchpad_store is not None:
                self.scratchpad_store.clear()
        # Reset per-run telemetry counters.
        self._todo_read_calls = 0
        self._todo_write_calls = 0
        # Reset buyability tracking for new run
        self._buyability_checked.clear()
        self._all_leaves_buyable_flag = True
        self._target_tools_called_set = set()
        self._recent_tool_turns = []
        self._last_recall_payload = None
        self._target_smiles = target_smiles
        # Hand the target to the save_best_route tool so its structural
        # route-grounding gate (root==target + no dangling intermediates) is
        # active for this run. Set per-run because a re-used agent instance
        # (core.evaluation loops) changes target between runs.
        try:
            _save_tool = self.tools.get("save_best_route") if self.tools else None
            if _save_tool is not None and hasattr(_save_tool, "set_target"):
                _save_tool.set_target(target_smiles)
        except Exception:
            pass
        # Reset sticky per-run state. Without these, a re-used agent
        # instance (e.g. core.evaluation looping over targets) carried over
        # the previous target's repeated-call fingerprints, short-circuit
        # cache, and per-tool call counts — surfacing as ghost
        # "you've called X 4 times, try a different tool" nudges on the
        # very first iteration of a fresh target. The cache key is
        # (tool, canonical_smiles), so cross-target collisions are rare
        # in production but trivial under randomised eval ordering.
        self._call_fingerprints = {}
        self._short_circuit_cache = {}
        self._tool_call_counts = {}

        # Wire the live transcript recorder. It hooks into self.trace so
        # every event (iteration / thinking / tool_call / tool_result /
        # complete / error / prefetch / *_compacted) is streamed to
        # transcript.jsonl AND forwarded to the user's trace callback (e.g.
        # `main.py run --trace` colored stdout).
        recorder: TranscriptRecorder | None = None
        if self.result_store is not None:
            try:
                recorder = TranscriptRecorder(
                    session_dir=self.result_store.session_dir,
                    target_smiles=target_smiles,
                    user_trace=self._user_trace,
                    scratchpad_store=self.scratchpad_store,
                    todo_store=self.todo_store,
                )
                self.trace = recorder
                self._write_manifest(target_smiles)
            except Exception as exc:
                logger.warning("TranscriptRecorder init failed: %s", exc)
                recorder = None

        result: dict | None = None
        # Reset per-run flags so a re-used agent instance behaves correctly
        # across targets.
        self._saved_threshold_nudged = False
        self._soft_threshold_nudged = False
        # Guards mini-status against double-injection on the same iteration
        # number: a refunded (non-billable) turn re-enters the loop at the
        # same iteration, re-tripping the `iteration % 5 == 0` gate.
        self._last_mini_status_iter = -1
        try:
            if resuming:
                # Continuation: rebuild the prior conversation, append the
                # chemist's feedback, and run on with an offset iteration
                # counter so anti-loop budgets stay coherent. The stores were
                # rehydrated (not cleared) above.
                prior_iters = int(resume_from.get("iterations") or 0)
                messages = deepcopy(resume_from.get("messages") or [])
                if feedback:
                    messages.append({
                        "role": "user",
                        "content": self.prompts.render(
                            "human_feedback", feedback=feedback
                        ),
                    })
                result = self._run_with_native_tools(
                    target_smiles,
                    initial_messages=messages,
                    max_iterations=extra_rounds or self.max_iterations,
                    iteration_offset=prior_iters,
                )
            else:
                result = self._run_with_native_tools(target_smiles)
            if result is not None:
                result = self._run_reaction_ordering_postprocess(
                    target_smiles, result
                )
        finally:
            # Always persist a transcript with the full ranked scratchpad +
            # full ReACT timeline so a human reviewer can audit every route
            # the agent considered, not just the final pick. Failure here
            # must never mask the primary result.
            if recorder is not None:
                try:
                    transcript_payload = recorder.finalize(
                        target_smiles=target_smiles,
                        result=result,
                        scratchpad_store=self.scratchpad_store,
                        todo_store=self.todo_store,
                    )
                    if result is not None:
                        result["session_dir"] = recorder.session_dir
                        result["transcript_json"] = os.path.join(
                            recorder.session_dir, "transcript.json"
                        )
                        result["transcript_jsonl"] = os.path.join(
                            recorder.session_dir, "transcript.jsonl"
                        )
                        result["transcript_stats"] = transcript_payload.get("stats", {})
                except Exception as exc:
                    logger.warning("transcript finalize failed: %s", exc)
            # Restore the user trace so a reused agent instance doesn't keep
            # writing to the previous run's session_dir.
            self.trace = self._user_trace or (lambda event, **kw: None)

        # Audit trail: attach final todo snapshot to the result so callers
            # (core/repl.py, core.evaluation) can persist it in session metadata.
        if self.todo_store is not None:
            result["todos"] = [it.to_dict() for it in self.todo_store.read()]
            result["todos_summary"] = self.todo_store.summary()
        return result

    def _run_with_native_tools(
        self,
        target_smiles: str,
        *,
        initial_messages: list[dict] | None = None,
        max_iterations: int | None = None,
        iteration_offset: int = 0,
    ) -> dict:
        """Run using OpenAI function-calling API."""
        self._target_smiles = target_smiles
        run_budget = int(max_iterations if max_iterations is not None else self.max_iterations)
        if initial_messages is None:
            system_msg = self.system_prompt.format(
                target_smiles=target_smiles,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                precedent_recall_block=self._precedent_recall_block,
                reaction_ordering_block=self._reaction_ordering_block,
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": self.prompts.render(
                    "user_kickoff", target_smiles=target_smiles
                )},
            ]
        else:
            messages = deepcopy(initial_messages)
        openai_tools = self.tools.get_openai_tools()
        self._write_latest_messages(
            messages,
            stage="continuation_start" if initial_messages is not None else "main_start",
            iteration=iteration_offset,
        )

        # ---- Prefetch (approach B) -----------------------------------------
        # The LLM systematically skips `suggest_templates` + `neural_retro`
        # on the target when it thinks it can write the route from memory.
        # That bypasses the whole point of the multi-engine ensemble +
        # Morgan/Butina clustering we just merged. Force-feed both tools'
        # output for the TARGET as a synthetic assistant tool-call turn so
        # the LLM sees the candidate sets in its context window for free.
        # Cheap (templates ~1s, ensemble ~30s) and parallel.
        disable_target_prefetch = (
            os.environ.get("RETRO_AGENT_DISABLE_TARGET_PREFETCH", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if initial_messages is None and not disable_target_prefetch:
            try:
                self._prefetch_disconnections(messages, target_smiles)
                self._write_latest_messages(
                    messages, stage="after_prefetch", iteration=iteration_offset
                )
            except Exception as e:
                logger.warning("Prefetch disconnections failed (non-fatal): %s", e)
        elif initial_messages is None and disable_target_prefetch:
            logger.info("Target prefetch disabled by RETRO_AGENT_DISABLE_TARGET_PREFETCH")

        iteration = 0
        t0 = time.time()
        # Per-run wallclock deadline. Hard upper bound on .run() so a
        # combination of slow LLM calls + slow tool calls cannot extend
        # a single run indefinitely. None disables the bound.
        wallclock_deadline = (
            t0 + self.wallclock_s if self.wallclock_s is not None else None
        )
        wallclock_bailed = False
        exit_reason: str | None = None
        consecutive_no_action = 0
        # M7 backstop: `iteration` is a whitelist meter (only suggest_templates
        # / neural_retro bill; everything else is refunded), so a model that
        # loops forever on free tools — check_buyability, score_route,
        # list_best_routes — never advances `iteration` and would only be
        # bounded by wallclock (90 min). Count *every* LLM turn separately and
        # hard-stop at a generous multiple so a pathological free-tool loop
        # terminates in minutes, not at the wallclock ceiling.
        total_turns = 0
        max_total_turns = max(run_budget * 4, run_budget + 50)
        # M5 plateau detector: once at least one route is saved, track how many
        # consecutive turns pass without a NEW distinct route. If the model
        # stops finding new routes well before the (possibly unreachable)
        # stop_after_distinct_routes floor, trip the soft threshold early so an
        # easy target can stop instead of burning the whole budget.
        last_distinct_count = 0
        plateau_turns = 0
        plateau_limit = 12

        while iteration < run_budget:
            # Wallclock guard. Checked at the top of every iteration so a
            # ballooning LLM call or stuck tool can extend at most one
            # iteration past the deadline.
            if wallclock_deadline is not None and time.time() > wallclock_deadline:
                logger.warning(
                    "Run wallclock budget %.0fs exceeded after %d iterations; "
                    "bailing out with best-saved route",
                    self.wallclock_s, iteration_offset + iteration,
                )
                try:
                    self.trace(
                        "wallclock_exceeded",
                        budget_s=self.wallclock_s,
                        iterations=iteration_offset + iteration,
                    )
                except Exception:
                    pass
                wallclock_bailed = True
                exit_reason = "wallclock_exceeded"
                break
            # M7 backstop: hard-stop a runaway free-tool loop that never
            # advances the billable `iteration` meter.
            if total_turns >= max_total_turns:
                logger.warning(
                    "Total-turn backstop %d reached (billable iteration=%d); "
                    "the model is looping on non-billable tools. Bailing out.",
                    max_total_turns, iteration_offset + iteration,
                )
                try:
                    self.trace(
                        "total_turns_exceeded",
                        total_turns=total_turns,
                        max_total_turns=max_total_turns,
                        iteration=iteration_offset + iteration,
                    )
                except Exception:
                    pass
                exit_reason = "total_turns_exceeded"
                break
            total_turns += 1
            iteration += 1
            global_iteration = iteration_offset + iteration
            logger.info("=== Iteration %d ===", global_iteration)
            self.trace("iteration", n=global_iteration)

            # Periodic transcript.json rollup so the summary view survives an
            # external SIGKILL (e.g. agent hangs in an LLM retry storm and the
            # benchmark harness kills it before run()'s finally writes the
            # authoritative copy). Best-effort; never raises.
            if global_iteration % 5 == 0 and isinstance(self.trace, TranscriptRecorder):
                self.trace.checkpoint()

            # Auto-finish nudge: once the scratchpad holds the configured
            # number of distinct routes, push a one-shot user message
            # asking the LLM to wrap up. Does NOT halt the loop — the LLM
            # still has to call list_best_routes / emit COMPLETE / pass
            # the structural gate.
            if (
                self.auto_finish_after_saved
                and self.scratchpad_store is not None
                and not getattr(self, "_saved_threshold_nudged", False)
            ):
                n_saved = len(self.scratchpad_store.list())
                if n_saved >= self.auto_finish_after_saved:
                    messages.append({"role": "user", "content": self.prompts.render(
                        "nudge_saved_threshold_reached",
                        n_saved=n_saved,
                        threshold=self.auto_finish_after_saved,
                    )})
                    self._saved_threshold_nudged = True
                    self.trace("saved_threshold_nudge",
                               n_saved=n_saved,
                               threshold=self.auto_finish_after_saved)

            # Reset the inline LLM-call retry budget every iteration. The
            # flag gates the "retry once before bailing out" path below; we
            # want it to apply *within* an iteration (catch a single transient
            # blip) but NOT decay to a once-per-run hard limit, which would
            # convert any second LLM-call failure across the entire 150-iter
            # session into an immediate bail-out.
            self._native_retry_used = False

            # Mini-status injection: every 5 iterations push a single
            # user-message snapshot of retro-search calls used / max and
            # current best route score (template lives in
            # `prompts/messages/status_mini.md`). Helps the LLM keep
            # self-awareness of where it is without polluting tool outputs
            # with internal state.
            try:
                if (global_iteration > 1 and global_iteration % 5 == 0
                        and global_iteration != self._last_mini_status_iter):
                    status_line = self._build_mini_status(global_iteration)
                    if status_line:
                        messages.append({"role": "user", "content": status_line})
                        self.trace("mini_status", text=status_line)
                    self._last_mini_status_iter = global_iteration
            except Exception as exc:
                logger.debug("mini-status injection failed: %s", exc, exc_info=True)

            # Dual-layer compaction: microCompact superseded tool-results every
            # iteration once usage crosses ~60% of budget; autoCompact at ~85%.
            messages = maybe_compact(
                messages,
                cfg=self.compaction_config,
                llm_client=self.llm,
                trace=self.trace,
            )
            self._write_latest_messages(
                messages, stage="before_llm", iteration=global_iteration
            )

            try:
                llm_t0 = time.monotonic()
                try:
                    self._record_llm_request(
                        phase="main",
                        messages=messages,
                        tools=openai_tools,
                        client=self.llm,
                        extra={"iteration": global_iteration},
                    )
                    content, tool_calls, finish_reason = self.llm.chat_with_tools(
                        messages, openai_tools
                    )
                finally:
                    llm_elapsed_ms = int((time.monotonic() - llm_t0) * 1000)
                self.trace(
                    "llm_call",
                    phase="primary",
                    elapsed_ms=llm_elapsed_ms,
                    finish_reason=finish_reason,
                    n_tool_calls=len(tool_calls or []),
                    content_chars=len(content or ""),
                    n_messages=len(messages),
                )
                logger.info(
                    "LLM call (primary): %dms finish=%s tool_calls=%d content_chars=%d",
                    llm_elapsed_ms, finish_reason, len(tool_calls or []), len(content or ""),
                )
            except Exception as e:
                try:
                    self.trace(
                        "llm_call",
                        phase="primary",
                        elapsed_ms=llm_elapsed_ms if "llm_elapsed_ms" in locals() else None,
                        error=str(e),
                        n_messages=len(messages),
                    )
                except Exception:
                    pass
                logger.error(
                    "LLM call failed after %s: %s",
                    f"{llm_elapsed_ms}ms" if "llm_elapsed_ms" in locals() else "?",
                    e,
                )
                self.trace("error", message=str(e))
                # Inline retry once before giving up. We deliberately do
                # NOT swap to a different client mode here: any fallback
                # that bypasses the structural gates (buyability / cheat /
                # ≥3 distinct routes) and accepts the LLM's next "COMPLETE
                # ROUTE FOUND" string would produce fake successes that
                # look identical to real ones in result.json/transcript.
                # Better to fail loudly with the work done so far than to
                # lie about it.
                if not getattr(self, "_native_retry_used", False):
                    self._native_retry_used = True
                    logger.info("Retrying native LLM call once")
                    try:
                        time.sleep(2.0)
                        llm_t0 = time.monotonic()
                        try:
                            self._record_llm_request(
                                phase="main_inline_retry",
                                messages=messages,
                                tools=openai_tools,
                                client=self.llm,
                                extra={"iteration": global_iteration},
                            )
                            content, tool_calls, finish_reason = self.llm.chat_with_tools(
                                messages, openai_tools
                            )
                        finally:
                            llm_elapsed_ms = int((time.monotonic() - llm_t0) * 1000)
                        self.trace(
                            "llm_call",
                            phase="inline_retry",
                            elapsed_ms=llm_elapsed_ms,
                            finish_reason=finish_reason,
                            n_tool_calls=len(tool_calls or []),
                            content_chars=len(content or ""),
                            n_messages=len(messages),
                        )
                        logger.info(
                            "LLM call (inline_retry): %dms finish=%s tool_calls=%d content_chars=%d",
                            llm_elapsed_ms, finish_reason, len(tool_calls or []), len(content or ""),
                        )
                    except Exception as e2:
                        try:
                            self.trace(
                                "llm_call",
                                phase="inline_retry",
                                elapsed_ms=llm_elapsed_ms if "llm_elapsed_ms" in locals() else None,
                                error=str(e2),
                                n_messages=len(messages),
                            )
                        except Exception:
                            pass
                        logger.error("Native retry also failed: %s", e2)
                        self.trace("error", message=str(e2))
                        return self._fail_native_with_best_so_far(
                            target_smiles, global_iteration, t0,
                            error=f"native LLM call failed twice: {e2}",
                            messages=messages,
                        )
                else:
                    return self._fail_native_with_best_so_far(
                        target_smiles, global_iteration, t0,
                        error=f"native LLM call failed: {e}",
                        messages=messages,
                    )

            # Trace assistant thinking
            if content:
                self.trace("thinking", content=content)
            # Trace model reasoning/thinking trace (separate from the final
            # answer text in `content`). Logging only — never re-sent to the
            # model. Stashed on the client by chat_with_tools.
            _reasoning = getattr(self.llm, "last_reasoning", None)
            if _reasoning:
                self.trace("reasoning", content=_reasoning)

            # Check for complete route in content
            if content and "COMPLETE ROUTE FOUND" in content:
                distinct = self._distinct_routes_count()
                if getattr(self, "_soft_threshold_nudged", False):
                    # SOFT-EXIT honour: the system reminder has already
                    # told the LLM that emitting `COMPLETE ROUTE FOUND:`
                    # ends the loop. Record the assistant turn so the
                    # transcript reflects the actual stop signal, then
                    # break out to the finalisation phase. The final
                    # ranking is still produced by the system verifier;
                    # the LLM's role is only to decide *when* to stop.
                    logger.info(
                        "LLM self-declared COMPLETE after soft threshold "
                        "(distinct=%d, threshold=%d); honouring as stop "
                        "signal.",
                        distinct, self.stop_after_distinct_routes,
                    )
                    try:
                        self.trace(
                            "llm_self_complete_honored",
                            distinct=distinct,
                            target_distinct=self.stop_after_distinct_routes,
                            iteration=global_iteration,
                        )
                    except Exception:
                        pass
                    messages.append({"role": "assistant", "content": content})
                    exit_reason = "llm_complete"
                    break
                # Pre-threshold: the LLM cannot self-declare completion.
                # Log + trace, push a "continue planning" reminder, refund
                # this turn, and continue. Once the soft threshold fires
                # (distinct >= stop_after_distinct_routes) this branch
                # falls through to the honour-the-marker path above.
                logger.info(
                    "LLM self-declared COMPLETE before soft threshold "
                    "(distinct=%d, target=%d); treating as informational "
                    "only — loop continues until threshold fires.",
                    distinct, self.stop_after_distinct_routes,
                )
                self.trace(
                    "llm_self_complete_ignored",
                    distinct=distinct,
                    target_distinct=self.stop_after_distinct_routes,
                    iteration=global_iteration,
                )
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": self.prompts.render(
                        "nudge_self_complete_ignored",
                        distinct=distinct,
                        target_distinct=self.stop_after_distinct_routes,
                    ),
                })
                iteration = self._maybe_refund_iteration(iteration, tool_calls)
                continue

            # Handle tool calls
            if tool_calls:
                consecutive_no_action = 0
                assistant_msg = {"role": "assistant", "tool_calls": tool_calls}
                if content:
                    assistant_msg["content"] = content
                messages.append(assistant_msg)

                # Trace each tool call
                for tc in tool_calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                    tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    name = fn["name"] if isinstance(fn, dict) else fn.name
                    args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                    self.trace("tool_call", tool_call_id=tc_id, name=name, args=args)

                # --- Repeated-call detector (infinite-loop guard) -----------
                # Fingerprint (name, canonical_args) for every tool call this
                # turn; maintain a counter across iterations. If any fingerprint
                # repeats >= N times, nudge the LLM to change tactics.
                if not hasattr(self, "_call_fingerprints"):
                    self._call_fingerprints = {}
                for tc in tool_calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                    _name = fn["name"] if isinstance(fn, dict) else fn.name
                    _args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                    try:
                        _canon = json.dumps(json.loads(_args), sort_keys=True)
                    except Exception:
                        _canon = str(_args)
                    key = f"{_name}::{_canon}"
                    self._call_fingerprints[key] = self._call_fingerprints.get(key, 0) + 1
                repeated = [(k, n) for k, n in self._call_fingerprints.items() if n >= 4]

                # F5: short-circuit cache for expensive idempotent tools.
                # ``neural_retro`` / ``suggest_templates`` / ``score_route``
                # are deterministic given (args), and re-executing them on
                # the same input is the dominant cost in the timeout cases
                # (idx 7/8/9 fired the chembart HTTP path 40+ times on the
                # same SMILES). Serve repeats from cache without calling.
                if not hasattr(self, "_short_circuit_cache"):
                    self._short_circuit_cache: dict[str, str] = {}
                _SHORTCIRCUIT_TOOLS = {"neural_retro", "suggest_templates", "score_route"}

                def _canon_args(raw_args) -> str:
                    """Canonicalize tool args for cache key.

                    Parses JSON, then RDKit-canonicalizes any value whose key
                    looks SMILES-ish (smiles, target_smiles, product, reactants,
                    smiles_list). This prevents the cache from missing on
                    trivial Kekulé / atom-order rewrites of the same input.
                    """
                    smiles_keys = {
                        "smiles", "target_smiles", "product", "reactant",
                        "smiles_list", "reactants",
                    }
                    try:
                        obj = json.loads(raw_args)
                    except Exception:
                        return str(raw_args)
                    if not isinstance(obj, dict):
                        return json.dumps(obj, sort_keys=True, default=str)
                    try:
                        from rdkit import Chem
                        def _c(s):
                            if not isinstance(s, str):
                                return s
                            m = Chem.MolFromSmiles(s)
                            return Chem.MolToSmiles(m) if m is not None else s
                    except Exception:
                        def _c(s):  # type: ignore[no-redef]
                            return s
                    norm: dict = {}
                    for k, v in obj.items():
                        if k in smiles_keys:
                            if isinstance(v, list):
                                norm[k] = sorted(_c(x) for x in v)
                            else:
                                norm[k] = _c(v)
                        else:
                            norm[k] = v
                    return json.dumps(norm, sort_keys=True, default=str)

                cached_results: list[dict] = []
                fresh_calls: list = []
                for tc in tool_calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                    _name = fn["name"] if isinstance(fn, dict) else fn.name
                    _args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                    _canon = _canon_args(_args)
                    key = f"{_name}::{_canon}"
                    if _name in _SHORTCIRCUIT_TOOLS and key in self._short_circuit_cache:
                        cached_content = self._short_circuit_cache[key]
                        # Annotate the cached payload so the LLM knows to vary
                        # its strategy rather than retry yet again.
                        try:
                            cobj = json.loads(cached_content)
                            if isinstance(cobj, dict):
                                cobj["_short_circuit"] = (
                                    "Identical call already executed in this session — "
                                    "result served from cache. Change arguments or try "
                                    "a different tool to make progress."
                                )
                                cached_content = json.dumps(cobj, ensure_ascii=False, default=str)
                        except Exception:
                            pass
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        cached_results.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": cached_content,
                            "tool_name": _name,
                            "elapsed_ms": 0,
                        })
                        logger.info("Short-circuit cache HIT: %s", key[:120])
                    else:
                        fresh_calls.append(tc)
                # Provenance attribution: for any save_best_route call in
                # this turn, decide which generation path the route came
                # from by looking at the last 3 turns' tool-call names.
                # We inject _provenance / _rag_source_ids into the args
                # JSON (the tool schema does NOT expose these — internal).
                provenance, rag_ids = self._compute_provenance()
                for tc in fresh_calls:
                    try:
                        fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                        fn_name_ = fn["name"] if isinstance(fn, dict) else fn.name
                        if fn_name_ != "save_best_route":
                            continue
                        args_raw = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                        try:
                            args_dict = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                        except Exception:
                            args_dict = {}
                        args_dict["_provenance"] = provenance
                        if rag_ids:
                            args_dict["_rag_source_ids"] = list(rag_ids)
                        new_args = json.dumps(args_dict, default=str)
                        if isinstance(fn, dict):
                            fn["arguments"] = new_args
                        else:
                            fn.arguments = new_args
                    except Exception as e:
                        logger.debug("provenance injection skipped: %s", e)
                # Execute the un-cached tool calls concurrently
                tool_results = (
                    execute_tools_concurrent(
                        self.tools,
                        fresh_calls,
                        tool_timeout_s=self.tool_timeout_s,
                    )
                    if fresh_calls else []
                )
                # Populate the short-circuit cache from fresh results
                for tr in tool_results:
                    tn = tr.get("tool_name")
                    if tn not in _SHORTCIRCUIT_TOOLS:
                        continue
                    # Re-derive the canonical key for this result by matching tool_call_id
                    for tc in fresh_calls:
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        if tc_id != tr.get("tool_call_id"):
                            continue
                        fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                        _name = fn["name"] if isinstance(fn, dict) else fn.name
                        _args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                        _canon = _canon_args(_args)
                        # Do NOT memoize error / timeout payloads. A single
                        # transient failure (e.g. a ChemBART HTTP wedge — the
                        # exact case this cache exists to amortise) would
                        # otherwise be served forever for identical args, so
                        # the model could never recover that disconnection by
                        # retrying. Only cache genuinely successful results.
                        _content = tr.get("content") or ""
                        if not self._is_error_payload(_content):
                            self._short_circuit_cache[f"{_name}::{_canon}"] = _content
                        break
                # Merge cached + fresh, preserving original tool_call order
                if cached_results:
                    by_id = {r["tool_call_id"]: r for r in tool_results + cached_results}
                    tool_results = []
                    for tc in tool_calls:
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        if tc_id in by_id:
                            tool_results.append(by_id[tc_id])
                backtrack_triggered = False
                save_best_route_seen = False
                this_turn_tool_names: list[str] = []
                for tr in tool_results:
                    elapsed_ms = tr.pop("elapsed_ms", None)
                    tool_name = tr.pop("tool_name", None)
                    # Track that the model actually exercised the plan-state tools
                    # (universally violated in the last batch — the COMPLETE gate
                    # only checks "no pending todos" which trivially passes when
                    # the model never called todo_write/todo_read at all).
                    if tool_name == "todo_read":
                        self._todo_read_calls = getattr(self, "_todo_read_calls", 0) + 1
                    elif tool_name == "todo_write":
                        self._todo_write_calls = getattr(self, "_todo_write_calls", 0) + 1
                    # Per-tool counter for the mini-status budget display.
                    if tool_name:
                        if not hasattr(self, "_tool_call_counts"):
                            self._tool_call_counts = {}
                        self._tool_call_counts[tool_name] = (
                            self._tool_call_counts.get(tool_name, 0) + 1
                        )
                    logger.info(
                        "Tool result [%s] %s (%sms): %s",
                        tr["tool_call_id"], tool_name or "?",
                        elapsed_ms if elapsed_ms is not None else "?",
                        tr["content"][:300],
                    )
                    self.trace(
                        "tool_result",
                        tool_call_id=tr["tool_call_id"],
                        content=tr["content"],
                        elapsed_ms=elapsed_ms,
                        tool_name=tool_name,
                    )
                    # Filesystem-offload heavy tool results (B/D/E). The full
                    # JSON goes to disk; the message stream gets a compact
                    # projection stub once the result is older than the
                    # per-tool recency window.
                    # NOTE: capture the raw content for downstream parsing
                    # (buyability tracking, save_best_route detection) BEFORE
                    # we replace tr["content"] with the stub.
                    raw_content = tr.get("content")
                    if self.result_store is not None and tool_name:
                        try:
                            tr["content"] = self.result_store.persist(
                                tool_name, tr["tool_call_id"], tr["content"]
                            )
                        except Exception as e:
                            logger.warning("result_store.persist failed: %s", e)
                    if tool_name == "score_route":
                        # Deterministic closure bridge: preserve the exact
                        # grounded route that was just scored.  This avoids a
                        # second LLM transcription changing steps/result ids.
                        try:
                            score_args = {}
                            for _tc in tool_calls:
                                _id = _tc.get("id") if isinstance(_tc, dict) else getattr(_tc, "id", None)
                                if _id != tr["tool_call_id"]:
                                    continue
                                _fn = _tc.get("function", _tc) if isinstance(_tc, dict) else _tc.function
                                _raw_args = _fn["arguments"] if isinstance(_fn, dict) else _fn.arguments
                                score_args = json.loads(_raw_args) if isinstance(_raw_args, str) else dict(_raw_args)
                                break
                            score_payload = json.loads(raw_content or "{}")
                            if self._auto_capture_scored_route(
                                tool_call_id=tr["tool_call_id"],
                                arguments=score_args,
                                result=score_payload,
                            ):
                                save_best_route_seen = True
                        except Exception as exc:
                            logger.warning("score_route auto-capture dispatch failed: %s", exc)
                    if tool_name == "save_best_route":
                        try:
                            payload = json.loads(raw_content or "{}")
                            if payload.get("ok") or payload.get("saved"):
                                save_best_route_seen = True
                        except Exception as e:
                            # Parse failure here silently leaves
                            # save_best_route_seen False, which skews the
                            # auto-finish / threshold logic. Surface it.
                            logger.warning(
                                "could not parse save_best_route result "
                                "(%s); auto-finish tracking may be off", e
                            )
                    # Stash latest recall payload for provenance attribution
                    # on subsequent save_best_route calls. We parse from the
                    # raw_content (pre-stub) so route_ids are present.
                    if tool_name == "recall_similar_routes":
                        try:
                            recall_payload = json.loads(raw_content or "{}")
                            if isinstance(recall_payload, dict):
                                self._last_recall_payload = recall_payload
                        except Exception as e:
                            logger.warning(
                                "could not parse recall_similar_routes result "
                                "(%s); provenance attribution may be incomplete", e
                            )
                    if tool_name:
                        this_turn_tool_names.append(tool_name)
                    # Track buyability checks by canonical SMILES → buyable bool.
                    # The hard gate above keys off both "was it checked" and
                    # "did it return true", so we MUST capture the result, not
                    # just the fact that the call happened.
                    if tool_name == "check_buyability":
                        try:
                            # check_buyability supports two shapes:
                            #   (1) single: args={"smiles": "..."},
                            #       result={"buyable": bool, ...}
                            #   (2) batch:  args={"smiles_list": [...]},
                            #       result={"n": N, "results": [{...}, ...]}
                            # The earlier tracker only handled (1) — when the
                            # LLM took the (encouraged) batch path, nothing was
                            # recorded and the completion gate kept reporting
                            # `missing=N` even after every leaf had been checked.
                            def _coerce_buyable(v):
                                if isinstance(v, bool):
                                    return v
                                if isinstance(v, str):
                                    return v.strip().lower() == "true"
                                return None

                            payload = None
                            try:
                                payload = json.loads(raw_content or "{}")
                            except Exception:
                                payload = None

                            # Build {input_smiles_or_canonical → buyable bool}
                            # from the result payload first; we'll then assign
                            # to the canonical forms of whatever args were sent.
                            result_by_smiles: dict[str, bool] = {}
                            top_buyable: bool | None = None
                            if isinstance(payload, dict):
                                top_buyable = _coerce_buyable(payload.get("buyable"))
                                rows = payload.get("results")
                                if isinstance(rows, list):
                                    for row in rows:
                                        if not isinstance(row, dict):
                                            continue
                                        bv = _coerce_buyable(row.get("buyable"))
                                        if bv is None:
                                            continue
                                        for key in (
                                            "canonical_smiles",
                                            "smiles",
                                            "input_smiles",
                                        ):
                                            s = row.get(key)
                                            if isinstance(s, str) and s:
                                                c = self._canonicalize_smiles(s)
                                                if c:
                                                    result_by_smiles[c] = bv

                            for tc in tool_calls:
                                fn = tc.get("function", tc) if isinstance(tc, dict) else tc.function
                                name = fn["name"] if isinstance(fn, dict) else fn.name
                                if name != "check_buyability" or tc.get("id") != tr.get("tool_call_id"):
                                    continue
                                args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
                                args_dict = json.loads(args) if isinstance(args, str) else args

                                arg_smiles_list: list[str] = []
                                s_single = args_dict.get("smiles")
                                if isinstance(s_single, str) and s_single:
                                    arg_smiles_list.append(s_single)
                                s_list = args_dict.get("smiles_list")
                                if isinstance(s_list, list):
                                    arg_smiles_list.extend(
                                        x for x in s_list if isinstance(x, str) and x
                                    )

                                for s in arg_smiles_list:
                                    canonical = self._canonicalize_smiles(s)
                                    if not canonical:
                                        continue
                                    # Prefer per-row result; else single-form
                                    # top-level buyable; else False (safer than
                                    # silently recording True on garbage).
                                    bv = result_by_smiles.get(canonical)
                                    if bv is None and len(arg_smiles_list) == 1:
                                        bv = top_buyable
                                    self._buyability_checked[canonical] = (
                                        bool(bv) if bv is not None else False
                                    )

                                # Also record any extra rows that appeared in
                                # results but weren't in args (defensive — some
                                # tool versions echo canonical forms only).
                                for c, bv in result_by_smiles.items():
                                    self._buyability_checked.setdefault(c, bv)
                                break
                        except Exception as e:
                            logger.warning(
                                "Failed to track buyability check (%s); the "
                                "leaf-buyability completion gate may be "
                                "inaccurate for this run", e
                            )
                    messages.append(tr)
                    # Watch for the backtrack marker — the scratchpad tool
                    # asks us to aggressively compact after a
                    # backtrack_and_summarize call.
                    try:
                        from tools.scratchpad import BACKTRACK_MARKER
                        if BACKTRACK_MARKER in (tr.get("content") or ""):
                            backtrack_triggered = True
                    except Exception:
                        pass

                # Demote any tool results that fell out of the result_store's
                # per-tool recency window (their full content is on disk;
                # rewrite the message in-place to the projection stub).
                if self.result_store is not None:
                    for demoted_id in self.result_store.pop_demoted():
                        stub = self.result_store.stub_for(demoted_id)
                        if not stub:
                            continue
                        for m in messages:
                            if m.get("role") == "tool" and m.get("tool_call_id") == demoted_id:
                                m["content"] = stub
                                break

                # Update sliding tool-name window for provenance attribution
                # on FUTURE save_best_route calls. Keep last 8 turns — empirically
                # a recall_similar_routes -> save_best_route sequence on hard
                # cases spans ~6 turns (score_route, check_buyability, …),
                # so the previous 3-turn cap rotated past the recall and
                # mis-tagged all saves zero_shot.
                self._recent_tool_turns.append(this_turn_tool_names)
                while len(self._recent_tool_turns) > 8:
                    self._recent_tool_turns.pop(0)

                # Milestone compaction (C): after a successful save_best_route
                # the in-flight branch's verbose tool results are dead weight —
                # the route is now durably on the scratchpad. Demote earlier
                # `tool` messages to placeholders, but keep the system prompt,
                # the original user message, and the recent tail intact.
                if save_best_route_seen:
                    cutoff = max(2, len(messages) - self.compaction_config.keep_recent_tail)
                    new_msgs, n_demoted = compact_to_checkpoint(
                        messages, cutoff, self.compaction_config
                    )
                    if n_demoted:
                        messages = new_msgs
                        logger.info(
                            "save_best_route checkpoint: demoted %d tool results",
                            n_demoted,
                        )
                        self.trace("checkpoint_compacted", demoted=n_demoted)

                # If any tool-call fingerprint has been repeated a lot, append
                # a user steer so the LLM breaks out of the loop.
                if repeated:
                    worst = max(repeated, key=lambda kv: kv[1])
                    logger.warning("Repeated tool call detected: %s (x%d)", worst[0], worst[1])
                    # Only suggest `recall_similar_routes` when it is actually
                    # registered (precedent_recall enabled). Otherwise the
                    # nudge steers the model toward an unregistered tool and
                    # provokes an "Unknown tool" hallucination — mirrors the
                    # conditional `{precedent_recall_block}` in the system prompt.
                    recall_hint = ""
                    if self.tools.get("recall_similar_routes") is not None:
                        recall_hint = (
                            ", or `recall_similar_routes` to pull a strategic "
                            "anchor from a known analog's route"
                        )
                    messages.append({"role": "user", "content": self.prompts.render(
                        "nudge_repeated_call",
                        tool_name=worst[0].split('::', 1)[0],
                        n=worst[1],
                        recall_hint=recall_hint,
                    )})
                    # reset so we don't nag every iteration
                    self._call_fingerprints = {}

                if backtrack_triggered and self.scratchpad_store is not None:
                    # Rewrite history: keep only system prompt + original
                    # user message + the session-start prefetch envelope
                    # (so post-mortem can still see what precedents seeded
                    # the run) + a fresh synthetic snapshot.
                    snapshot = self.scratchpad_store.snapshot(keep_best_n=3)
                    todo_snap = self.todo_store.read() if self.todo_store else []
                    todo_dump = [t.to_dict() for t in todo_snap]
                    system_msg = next(
                        (m for m in messages if m.get("role") == "system"), None
                    )
                    user_msg = next(
                        (m for m in messages if m.get("role") == "user"), None
                    )
                    # Locate the prefetch assistant message + its tool results.
                    # Prefetch assistant tool_calls all carry ids of the form
                    # `call_prefetch_*` (see _prefetch_disconnections).
                    prefetch_asst = None
                    prefetch_tool_ids: set[str] = set()
                    for m in messages:
                        if (m.get("role") == "assistant"
                                and m.get("tool_calls")):
                            ids = []
                            for tc in m["tool_calls"]:
                                tcid = (
                                    tc.get("id") if isinstance(tc, dict)
                                    else getattr(tc, "id", None)
                                )
                                if isinstance(tcid, str) and tcid.startswith("call_prefetch_"):
                                    ids.append(tcid)
                            if ids and len(ids) == len(m["tool_calls"]):
                                prefetch_asst = m
                                prefetch_tool_ids = set(ids)
                                break
                    prefetch_tool_msgs = [
                        m for m in messages
                        if m.get("role") == "tool"
                        and m.get("tool_call_id") in prefetch_tool_ids
                    ] if prefetch_tool_ids else []

                    compact_note = {
                        "role": "user",
                        "content": self.prompts.render(
                            "nudge_below_target",
                            todos_json=json.dumps(todo_dump),
                            scratchpad_json=json.dumps(snapshot),
                        ),
                    }
                    new_messages = []
                    if system_msg:
                        new_messages.append(system_msg)
                    if user_msg and user_msg is not system_msg:
                        new_messages.append(user_msg)
                    if prefetch_asst is not None:
                        new_messages.append(prefetch_asst)
                        new_messages.extend(prefetch_tool_msgs)
                    new_messages.append(compact_note)
                    dropped = len(messages) - len(new_messages)
                    messages = new_messages
                    logger.info(
                        "backtrack_and_summarize: dropped %d messages "
                        "(kept prefetch=%s)",
                        dropped, prefetch_asst is not None,
                    )
                    self.trace("backtrack_compacted", dropped=dropped,
                               snapshot=snapshot,
                               kept_prefetch=prefetch_asst is not None)
            elif content:
                messages.append({"role": "assistant", "content": content})
                consecutive_no_action += 1

                if consecutive_no_action >= 3:
                    messages.append({"role": "user", "content": self.prompts.render(
                        "nudge_no_action"
                    )})
                    consecutive_no_action = 0
                elif finish_reason == "stop":
                    messages.append({"role": "user", "content": self.prompts.render(
                        "nudge_continue_planning"
                    )})
            else:
                # Neither content nor tool calls
                consecutive_no_action += 1
                messages.append({"role": "user", "content": "Please continue with the retrosynthesis planning."})

            # End-of-turn refund: if this turn dispatched ONLY exempt tools
            # (e.g. a batch of `check_buyability` calls), don't bill it
            # against the search budget.
            iteration = self._maybe_refund_iteration(iteration, tool_calls)

            # SOFT exit (system reminder + LLM-controlled stop): once the
            # scratchpad has saved at least `stop_after_distinct_routes`
            # distinct routes, push a one-shot reminder telling the LLM it
            # may now emit `COMPLETE ROUTE FOUND:` to end the loop. The
            # loop itself keeps running until either the LLM emits that
            # marker (handled at the top of the next iteration, gated on
            # `_soft_threshold_nudged`) or `max_iterations` /
            # `wallclock_s` is hit. Rationale: fixed-threshold hard exits
            # leave most of the iteration budget on the floor on cases
            # where the model has more useful exploration to do; let the
            # model decide when "enough is enough".
            if (
                self.scratchpad_store is not None
                and not getattr(self, "_soft_threshold_nudged", False)
            ):
                _distinct = self._distinct_routes_count()
                # M5 plateau tracking: count consecutive turns with no new
                # distinct route once we have at least one saved.
                if _distinct > last_distinct_count:
                    last_distinct_count = _distinct
                    plateau_turns = 0
                elif _distinct > 0:
                    plateau_turns += 1
                # Two ways to trip the soft threshold:
                #   (1) the configured floor is reached (original behaviour);
                #   (2) the search has plateaued — at least 3 distinct routes
                #       saved AND no new distinct route for `plateau_limit`
                #       turns. (2) lets an easy target whose route space is
                #       smaller than stop_after_distinct_routes stop instead of
                #       grinding to the iteration/wallclock ceiling.
                plateaued = _distinct >= 3 and plateau_turns >= plateau_limit
                if _distinct >= self.stop_after_distinct_routes or plateaued:
                    iters_remaining = max(0, self.max_iterations - global_iteration)
                    logger.info(
                        "Soft threshold reached: distinct_routes=%d (floor=%d, "
                        "plateau_turns=%d/%d, via=%s); emitting reminder and "
                        "honouring future COMPLETE ROUTE FOUND.",
                        _distinct, self.stop_after_distinct_routes,
                        plateau_turns, plateau_limit,
                        "plateau" if plateaued and _distinct < self.stop_after_distinct_routes else "floor",
                    )
                    try:
                        self.trace(
                            "soft_threshold_reached",
                            distinct=_distinct,
                            stop_after=self.stop_after_distinct_routes,
                            iters_remaining=iters_remaining,
                            plateau_turns=plateau_turns,
                        )
                    except Exception:
                        pass
                    messages.append({
                        "role": "user",
                        "content": self.prompts.render(
                            "nudge_soft_threshold_reached",
                            n_saved=_distinct,
                            threshold=self.stop_after_distinct_routes,
                            iters_remaining=iters_remaining,
                        ),
                    })
                    self._soft_threshold_nudged = True
        # ----------------------------------------------------------------
        # Finalisation phase (loop exited via budget exhaustion or early
        # exit). Score every saved route still missing a score, then call
        # `rank_routes` so the trace surfaces the full ranked breakdown
        # — not just the score-gated top-1. Synthesise a result from
        # rank-1 so callers always get a usable route.
        # ----------------------------------------------------------------
        ranked_payload = self._finalise_phase()
        # Hard buyability gate: a route is only a *success* if its leaves are
        # actually purchasable. Walk the ranked list (best score first) and pick
        # the first route whose leaves are all buyable, rather than gambling on
        # rank-1 alone — the highest-scoring route often carries one unbuyable
        # leaf (a complex intermediate, or a reagent absent from the building-
        # block catalogues) while a slightly lower-scoring route terminates
        # cleanly in commercial blocks. Only when *every* ranked route fails is
        # the run unsuccessful. (Previously only routes[0] was checked, so a
        # solvable target with a non-buyable rank-1 leaf was reported FAILED
        # even though rank-2+ were fully buyable.)
        selected_index = self._select_first_buyable_route(ranked_payload)
        synth_route_text = self._synthesise_route_text_from_ranked(
            ranked_payload, route_index=max(selected_index, 0)
        )
        bail_reason = (
            "wallclock_exceeded"
            if wallclock_bailed
            else (None if exit_reason == "llm_complete" else "max_iterations")
        )
        best_dict = None
        score_best_dict = None
        all_routes_dicts: list[dict] = []
        if self.scratchpad_store is not None:
            try:
                routes_sorted = self.scratchpad_store.list()
                all_routes_dicts = [r.to_dict() for r in routes_sorted]
                best_dict, score_best_dict = self._resolve_best_route(
                    routes_sorted, ranked_payload, selected_index
                )
            except Exception as e:
                logger.warning("finalisation: scratchpad serialisation failed: %s", e)
        return {
            "success": bool(synth_route_text) and self._all_leaves_buyable_flag,
            "route_text": (
                synth_route_text
                if (synth_route_text and self._all_leaves_buyable_flag)
                else (
                    synth_route_text
                    + "\n[WARNING: no ranked route terminates in buyable leaves "
                    "— not a complete synthesis to commercial building blocks]"
                    if synth_route_text
                    else (
                        "Wallclock budget exceeded"
                        if wallclock_bailed
                        else (
                            "No saved route after completion"
                            if exit_reason == "llm_complete"
                            else "Max iterations reached"
                        )
                    )
                )
            ),
            "iterations": iteration_offset + iteration,
            "time": time.time() - t0,
            "messages": messages,
            "ranked": ranked_payload,
            "best_route": best_dict,
            "score_best_route": score_best_dict,
            "all_routes": all_routes_dicts,
            "bail_reason": bail_reason,
        }

    # ------------------------------------------------------------------
    # Text-ReACT fallback
    # ------------------------------------------------------------------

    def _prefetch_disconnections(self, messages: list, target_smiles: str) -> None:
        """Inject `suggest_templates` + `neural_retro` results for the target
        as a synthetic assistant tool-call turn (approach B).

        Run both tools in parallel, then append: (1) one assistant message
        with two tool_calls, (2) two tool messages carrying the results.
        Net effect: when the LLM's first real turn fires, it already sees
        ~15 USPTO/RetroBioCat templates + ~12 ensemble cluster reps in
        context — no instruction-following required to use them.
        """
        from concurrent.futures import ThreadPoolExecutor
        import uuid

        sugg = self.tools.get("suggest_templates")
        neural = self.tools.get("neural_retro")
        recall = (
            self.tools.get("recall_similar_routes")
            if self.precedent_prefetch_enabled else None
        )
        if sugg is None and neural is None and recall is None:
            return  # nothing to prefetch

        jobs = {}
        ex = ThreadPoolExecutor(max_workers=3)
        try:
            if sugg is not None:
                jobs["suggest_templates"] = (
                    ex.submit(
                        sugg.execute,
                        product_smiles=target_smiles, top_k=15,
                    ),
                    time.monotonic(),
                )
            if neural is not None:
                jobs["neural_retro"] = (
                    ex.submit(
                        neural.execute,
                        target_smiles=target_smiles,
                    ),
                    time.monotonic(),
                )
            if recall is not None:
                jobs["recall_similar_routes"] = (
                    ex.submit(
                        recall.execute,
                        query_smiles=target_smiles,
                        k=self.precedent_prefetch_k,
                        source=self.precedent_prefetch_source,
                    ),
                    time.monotonic(),
                )
            results = {}
            elapsed_by_name: dict[str, int] = {}
            prefetch_deadline = time.monotonic() + 600
            for name, (fut, started_at) in jobs.items():
                try:
                    remaining_s = max(0.0, prefetch_deadline - time.monotonic())
                    out = fut.result(timeout=remaining_s)
                    elapsed_by_name[name] = int((time.monotonic() - started_at) * 1000)
                    results[name] = out if isinstance(out, str) else json.dumps(out, default=str)
                except Exception as e:
                    elapsed_by_name[name] = int((time.monotonic() - started_at) * 1000)
                    # Use %r so empty str(e) (e.g. silently-swallowed deadlock
                    # signals) still print the type; exc_info=True attaches a
                    # full traceback so prefetch failures are not opaque.
                    logger.warning(
                        "Prefetch %s failed: %r", name, e, exc_info=True
                    )
                    results[name] = json.dumps({"error": f"prefetch failed: {e}"})
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

        if not results:
            return

        # Mark target tools as called for the completion gate (prefetch
        # always runs both suggest_templates and neural_retro on the target).
        for tn in results.keys():
            self._target_tools_called_set.add(tn)

        # Build OpenAI tool-call envelope with synthetic IDs.
        # Persist each result through self.result_store so (a) the LLM-visible
        # `content` carries the injected `_result_id` for audit/save_best_route
        # cross-checks and (b) load_result(result_id="call_prefetch_…") works
        # as a rescue path if microCompact ever drops the envelope.
        def _persist(name: str, tcid: str) -> str:
            raw = results[name]
            if self.result_store is not None:
                try:
                    return self.result_store.persist(name, tcid, raw)
                except Exception as e:
                    logger.warning(
                        "Prefetch persist failed for %s: %r", name, e,
                    )
            return raw

        tool_calls = []
        tool_msgs = []
        if "suggest_templates" in results:
            tcid = f"call_prefetch_{uuid.uuid4().hex[:8]}"
            tool_calls.append({
                "id": tcid, "type": "function",
                "function": {
                    "name": "suggest_templates",
                    "arguments": json.dumps({
                        "product_smiles": target_smiles, "top_k": 15,
                    }),
                },
            })
            tool_msgs.append({
                "role": "tool", "tool_call_id": tcid,
                "name": "suggest_templates",
                "content": _persist("suggest_templates", tcid),
            })
        if "neural_retro" in results:
            tcid = f"call_prefetch_{uuid.uuid4().hex[:8]}"
            tool_calls.append({
                "id": tcid, "type": "function",
                "function": {
                    "name": "neural_retro",
                    "arguments": json.dumps({"target_smiles": target_smiles}),
                },
            })
            tool_msgs.append({
                "role": "tool", "tool_call_id": tcid,
                "name": "neural_retro",
                "content": _persist("neural_retro", tcid),
            })
        if "recall_similar_routes" in results:
            raw_recall = results["recall_similar_routes"]
            try:
                recall_payload = json.loads(raw_recall)
            except Exception:
                recall_payload = {}
            confidence = (recall_payload.get("confidence") or "none") if isinstance(recall_payload, dict) else "none"
            try:
                max_sim = float(recall_payload.get("max_similarity") or 0.0) if isinstance(recall_payload, dict) else 0.0
            except (TypeError, ValueError):
                max_sim = 0.0
            # Suppress when top-1 similarity is below the medium threshold —
            # injecting low/none-band hits encourages the LLM to anchor on
            # non-analogous routes. Errors are NOT suppressed so the LLM still
            # sees that prefetch broke.
            suppress = (
                isinstance(recall_payload, dict)
                and "error" not in recall_payload
                and max_sim < self.precedent_prefetch_min_similarity
            )
            tcid = f"call_prefetch_{uuid.uuid4().hex[:8]}"
            tool_calls.append({
                "id": tcid, "type": "function",
                "function": {
                    "name": "recall_similar_routes",
                    "arguments": json.dumps({
                        "query_smiles": target_smiles,
                        "k": self.precedent_prefetch_k,
                        "source": self.precedent_prefetch_source,
                    }),
                },
            })
            if suppress:
                stub = json.dumps({
                    "suppressed": True,
                    "reason": (
                        f"top-1 similarity {max_sim:.2f} below medium threshold "
                        f"{self.precedent_prefetch_min_similarity:.2f} (band={confidence}); "
                        f"no useful analog in the precedent DB. Call "
                        f"recall_similar_routes(query_smiles=<intermediate>) "
                        f"explicitly mid-search if a sub-target looks more familiar."
                    ),
                    "max_similarity": max_sim,
                    "confidence": confidence,
                })
                # Persist the RAW payload under this tcid so load_result(tcid)
                # surfaces the real hits if the LLM asks — audit trail intact.
                if self.result_store is not None:
                    try:
                        self.result_store.persist("recall_similar_routes", tcid, raw_recall)
                    except Exception as e:
                        logger.warning(
                            "Prefetch persist (suppressed) failed: %r", e,
                        )
                tool_msgs.append({
                    "role": "tool", "tool_call_id": tcid,
                    "name": "recall_similar_routes",
                    "content": stub,
                })
                logger.info(
                    "Prefetch RAG suppressed: confidence=%s max_sim=%.3f floor=%.2f",
                    confidence, max_sim, self.precedent_prefetch_min_similarity,
                )
            else:
                tool_msgs.append({
                    "role": "tool", "tool_call_id": tcid,
                    "name": "recall_similar_routes",
                    "content": _persist("recall_similar_routes", tcid),
                })
            # NOTE: deliberately NOT stashing into self._last_recall_payload
            # here. The prefetch fires on the TARGET; if the agent then
            # saves a route for a sub-intermediate without calling recall
            # itself, the fallback in _compute_provenance would tag that
            # save rag_anchored with the *target's* exemplar ids — wrong
            # attribution. Only the explicit in-loop recall_similar_routes
            # path (core.agent: tool dispatch in _run_with_native_tools)
            # writes _last_recall_payload.

        messages.append({
            "role": "assistant",
            "content": (
                "Prefetching the curated template library and the multi-engine "
                "neural ensemble for the target so we can compare cross-source "
                "candidates before committing to any disconnection."
            ),
            "tool_calls": tool_calls,
        })
        messages.extend(tool_msgs)
        # Mirror the synthetic prefetch tool_call / tool_result pairs into
        # the transcript so a reviewer can see the full per-engine raw
        # predictions, Morgan/Butina cluster buckets, and cross-engine
        # support — not just a "prefetch happened" marker.
        for tc, tm in zip(tool_calls, tool_msgs):
            try:
                self.trace(
                    "tool_call",
                    tool_call_id=tc["id"],
                    name=tc["function"]["name"],
                    args=tc["function"]["arguments"],
                )
                self.trace(
                    "tool_result",
                    tool_call_id=tm["tool_call_id"],
                    tool_name=tm["name"],
                    content=tm["content"],
                    elapsed_ms=elapsed_by_name.get(tm["name"]),
                )
            except Exception:
                pass
        # Nudge the LLM to actually use what we just handed it
        messages.append({
            "role": "user",
            "content": (
                "Templates and the neural ensemble (Morgan/Butina-clustered) are above. "
                "Compare cross-source candidates — entries with `support: [≥2 engines]` "
                "or that span DIFFERENT `bucket` ids are the high-confidence picks. "
                "If there are 3+ promising candidates, evaluate them in a single "
                "`explore_branches` call before committing to one disconnection."
            ),
        })
        try:
            self.trace("prefetch", tools=list(results.keys()),
                       n_messages_added=1 + len(tool_msgs) + 1)
        except Exception:
            pass
