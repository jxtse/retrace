"""
Interactive REPL interface for the retrosynthesis agent.
Provides a CLI similar to Claude Code's interactive mode with
streaming output, session management, and conversation history.
"""

import json
import logging
import os
import readline
import signal
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session import Session, SessionManager
from services.compaction import compact_messages, estimate_tokens

logger = logging.getLogger(__name__)

# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"


class RetroSynthesisREPL:
    """Interactive REPL for the retrosynthesis agent."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.agent = None
        self.session_manager = SessionManager()
        self.current_session: Session | None = None
        self.running = True

        # Context management
        self.max_context_tokens = self.config.get("max_context_tokens", 100000)
        self.auto_compact = self.config.get("auto_compact", True)

    def _build_agent(self):
        """Lazy-build the agent."""
        if self.agent is None:
            from core import build_agent
            self.agent = build_agent(self.config)

    def _print_banner(self):
        print(f"""
{BOLD}{CYAN}========================================{RESET}
{BOLD}  Chemoenzymatic Retrosynthesis Agent{RESET}
{BOLD}{CYAN}========================================{RESET}
{DIM}ReACT-style agent for multi-step retrosynthesis planning{RESET}
{DIM}LLM: {self.config.get('llm_model')} @ {self.config.get('llm_base_url')}{RESET}

{BOLD}Commands:{RESET}
  {GREEN}/run <SMILES>{RESET}     - Find retrosynthetic route for a molecule
  {GREEN}/continue <text>{RESET}  - Continue the current run with chemist feedback
  {GREEN}/sessions{RESET}         - List saved sessions
  {GREEN}/resume <id>{RESET}      - Resume a previous session (inspect-only)
  {GREEN}/status{RESET}           - Show current session status
  {GREEN}/history{RESET}          - Show conversation history
  {GREEN}/compact{RESET}          - Compact conversation context
  {GREEN}/tools{RESET}            - List available tools
  {GREEN}/help{RESET}             - Show this help
  {GREEN}/quit{RESET}             - Exit
""")

    def _print_status(self):
        if not self.current_session:
            print(f"{YELLOW}No active session.{RESET}")
            return
        s = self.current_session
        tokens = estimate_tokens(s.messages) if s.messages else 0
        print(f"""
{BOLD}Session:{RESET} {s.session_id}
{BOLD}Target:{RESET}  {s.target_smiles or 'None'}
{BOLD}Status:{RESET}  {'Route found' if s.route_found else 'In progress' if s.iterations > 0 else 'Ready'}
{BOLD}Iterations:{RESET} {s.iterations}
{BOLD}Messages:{RESET} {len(s.messages)} (~{tokens} tokens)
{BOLD}Tools used:{RESET} {len(s.tool_results)}
""")

    def _print_history(self, last_n: int = 20):
        if not self.current_session or not self.current_session.messages:
            print(f"{YELLOW}No messages in current session.{RESET}")
            return
        msgs = self.current_session.messages[-last_n:]
        for msg in msgs:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                print(f"{DIM}[system] {content[:100]}...{RESET}")
            elif role == "user":
                print(f"{GREEN}[user] {content[:200]}{RESET}")
            elif role == "assistant":
                text = content[:300] if content else ""
                if "tool_calls" in msg:
                    tc_names = [tc["function"]["name"] for tc in msg["tool_calls"]]
                    text += f" [calls: {', '.join(tc_names)}]"
                print(f"{BLUE}[assistant] {text}{RESET}")
            elif role == "tool":
                print(f"{CYAN}[tool] {content[:150]}{RESET}")

    def _handle_run(self, smiles: str):
        """Run retrosynthesis for a target SMILES via the canonical gated loop."""
        if not smiles.strip():
            print(f"{RED}Please provide a SMILES string.{RESET}")
            return

        self._build_agent()
        self.current_session = self.session_manager.create_session(smiles.strip())

        print(f"\n{BOLD}Target:{RESET} {smiles.strip()}")
        print(f"{DIM}Starting retrosynthesis planning...{RESET}\n")

        t0 = time.time()
        try:
            # Drive the canonical, fully-gated agent.run(); streaming is a pure
            # presentation layer via the trace callback (mirrors webapp runner).
            self.agent._user_trace = self._trace_print
            self.agent.trace = self._trace_print
            result = self.agent.run(smiles.strip())
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted by user.{RESET}")
            result = {"success": False, "route_text": "Interrupted",
                      "iterations": self.current_session.iterations, "time": time.time() - t0}

        self._apply_result_to_session(result, t0)
        self._print_run_result(result, t0)

    def _apply_result_to_session(self, result: dict, t0: float):
        """Fold an agent.run() result dict into the current session + persist."""
        s = self.current_session
        s.route_found = bool(result.get("success"))
        s.route_text = result.get("route_text", "")
        s.iterations = result.get("iterations", 0)
        # agent.run() does NOT return a top-level `messages` list; it persists
        # the full transcript to disk. Record the pointer so /continue can
        # rehydrate prior messages, and mirror the convenience keys.
        if result.get("transcript_json"):
            s.metadata["transcript_json"] = result["transcript_json"]
        if result.get("session_dir"):
            s.metadata["session_dir"] = result["session_dir"]
        if "all_routes" in result:
            s.metadata["all_routes"] = result["all_routes"]
        # Persist plan-state snapshot for audit / later inspection. We do NOT
        # auto-restore on /resume — the LLM re-registers todos via todo_write
        # from the message transcript, which avoids stale state.
        if "todos" in result:
            s.metadata["todos"] = result["todos"]
        if "todos_summary" in result:
            s.metadata["todos_summary"] = result["todos_summary"]
        self.session_manager.save(s)

    def _print_run_result(self, result: dict, t0: float):
        """Render the gated result. agent.run() is always gate-checked, so a
        success here is buyability-verified — no UNVERIFIED banner."""
        elapsed = result.get("time", time.time() - t0)
        print(f"\n{'=' * 60}")
        if result.get("success"):
            print(f"{GREEN}{BOLD}Route found!{RESET}")
            text = result.get("route_text", "") or ""
            idx = text.find("COMPLETE ROUTE FOUND")
            if idx >= 0:
                print(text[idx:])
            else:
                print(text[-1000:])
        else:
            print(f"{RED}Route not found within iteration limit.{RESET}")
            text = result.get("route_text", "") or ""
            if text and text != "Interrupted":
                print(text[-1000:])
        print(f"{DIM}Iterations: {result.get('iterations', 0)} | Time: {elapsed:.1f}s{RESET}")
        print(f"{'=' * 60}\n")

    def _trace_print(self, event: str, **kw):
        """Render canonical agent trace events to colored stdout.

        Completion is learned from agent.run()'s return dict, not a trace
        event (the native loop emits no `complete` event).
        """
        try:
            if event == "iteration":
                n = kw.get("n")
                print(f"{DIM}--- Iteration {n}/{self.agent.max_iterations} ---{RESET}")
            elif event in ("thinking", "reasoning"):
                content = (kw.get("content") or "").strip()
                if not content:
                    return
                for line in content.split("\n"):
                    if line.strip().startswith("Step") or ">>" in line:
                        print(f"{GREEN}{line}{RESET}")
                    elif line.strip().startswith("##") or line.strip().startswith("Thought"):
                        print(f"{BOLD}{line}{RESET}")
                    else:
                        print(line)
            elif event == "tool_call":
                name = kw.get("name", "?")
                args = kw.get("args", "")
                if not isinstance(args, str):
                    try:
                        args = json.dumps(args)
                    except Exception:
                        args = str(args)
                print(f"  {CYAN}> {name}({args[:80]}){RESET}")
            elif event == "tool_result":
                name = kw.get("tool_name", "")
                content = kw.get("content", "") or ""
                print(f"    {DIM}-> {name}: {content[:120]}{RESET}")
            elif event == "error":
                print(f"{RED}error: {kw.get('message', '')}{RESET}")
        except Exception:
            # Tracing must never break the run.
            pass

    def _load_resume_state(self) -> dict | None:
        """Assemble a ``resume_from`` dict for the current session.

        Sources prior ``messages`` from the session transcript's
        ``latest_messages.json`` (written by agent.run()), the prior
        scratchpad ``all_routes`` + ``todos`` + ``iterations`` from session
        metadata (set in _apply_result_to_session). Returns None if there's no
        runnable prior state.
        """
        s = self.current_session
        if s is None:
            return None

        messages: list[dict] = []
        tj = s.metadata.get("transcript_json")
        if tj and os.path.exists(tj):
            try:
                with open(tj, "r", encoding="utf-8") as fh:
                    transcript = json.load(fh)
                lm_path = (transcript.get("resume_state") or {}).get("latest_messages_path")
                if lm_path and os.path.exists(lm_path):
                    with open(lm_path, "r", encoding="utf-8") as fh:
                        messages = (json.load(fh) or {}).get("messages") or []
            except Exception as e:
                logger.debug("resume: transcript/messages load failed: %s", e)

        if not messages:
            return None

        return {
            "messages": messages,
            "all_routes": s.metadata.get("all_routes") or [],
            "todos": s.metadata.get("todos") or [],
            "iterations": s.iterations or 0,
        }

    def _handle_continue(self, feedback: str):
        """Continue the current session's run with a chemist's feedback.

        Rehydrates the prior scratchpad + transcript and runs on for a bounded
        number of extra rounds (agent.continue_extra_rounds). This is the
        human-in-the-loop seam — distinct from /resume, which is inspect-only.
        """
        if not feedback.strip():
            print(f"{RED}Usage: /continue <feedback for the chemist refinement>{RESET}")
            return
        if self.current_session is None:
            print(f"{RED}No active session. Run /run <SMILES> or /resume <id> first.{RESET}")
            return

        target = self.current_session.target_smiles
        if not target:
            print(f"{RED}Current session has no target SMILES.{RESET}")
            return

        resume_from = self._load_resume_state()
        if resume_from is None:
            print(f"{RED}No prior transcript for this session — cannot continue. "
                  f"Run /run first so a transcript is written.{RESET}")
            return

        self._build_agent()
        extra_rounds = int(
            self.config.get("agent", {}).get("continue_extra_rounds", 15)
        )

        print(f"\n{BOLD}Continuing:{RESET} {target}")
        print(f"{DIM}Feedback: {feedback.strip()}{RESET}")
        print(f"{DIM}Rehydrated {len(resume_from['all_routes'])} prior route(s); "
              f"+{extra_rounds} rounds from iter {resume_from['iterations']}...{RESET}\n")

        t0 = time.time()
        try:
            self.agent._user_trace = self._trace_print
            self.agent.trace = self._trace_print
            result = self.agent.run(
                target,
                resume_from=resume_from,
                feedback=feedback.strip(),
                extra_rounds=extra_rounds,
            )
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Interrupted by user.{RESET}")
            result = {"success": False, "route_text": "Interrupted",
                      "iterations": self.current_session.iterations, "time": time.time() - t0}

        self._apply_result_to_session(result, t0)
        self._print_run_result(result, t0)

    def _handle_command(self, line: str):
        """Process a REPL command."""
        parts = line.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            self.running = False
        elif cmd == "/run":
            self._handle_run(arg)
        elif cmd == "/continue":
            self._handle_continue(arg)
        elif cmd == "/sessions":
            sessions = self.session_manager.list_sessions()
            if not sessions:
                print(f"{YELLOW}No saved sessions.{RESET}")
            else:
                for s in sessions[-10:]:
                    status = f"{GREEN}found{RESET}" if s["route_found"] else f"{YELLOW}in progress{RESET}"
                    print(f"  {s['session_id']}  {s['target_smiles'][:40]}  [{status}]  {s['iterations']} iters")
        elif cmd == "/resume":
            if not arg:
                print(f"{RED}Usage: /resume <session_id>{RESET}")
                return
            session = self.session_manager.load(arg.strip())
            if session:
                self.current_session = session
                print(f"{GREEN}Resumed session: {session.session_id}{RESET}")
                self._print_status()
            else:
                print(f"{RED}Session not found: {arg}{RESET}")
        elif cmd == "/status":
            self._print_status()
        elif cmd == "/history":
            n = int(arg) if arg.isdigit() else 20
            self._print_history(n)
        elif cmd == "/compact":
            if self.current_session and self.current_session.messages:
                before = estimate_tokens(self.current_session.messages)
                self.current_session.messages = compact_messages(
                    self.current_session.messages, self.max_context_tokens
                )
                after = estimate_tokens(self.current_session.messages)
                self.session_manager.save(self.current_session)
                print(f"{GREEN}Compacted: {before} -> {after} tokens{RESET}")
            else:
                print(f"{YELLOW}No messages to compact.{RESET}")
        elif cmd == "/tools":
            self._build_agent()
            for name in self.agent.tools.tool_names:
                tool = self.agent.tools.get(name)
                print(f"  {CYAN}{name}{RESET}: {tool.description[:80]}")
        elif cmd == "/help":
            self._print_banner()
        else:
            print(f"{RED}Unknown command: {cmd}{RESET}")
            print(f"{DIM}Type /help for available commands.{RESET}")

    def run(self):
        """Main REPL loop."""
        self._print_banner()

        # Handle Ctrl+C gracefully
        def sigint_handler(sig, frame):
            print(f"\n{YELLOW}Use /quit to exit.{RESET}")
        signal.signal(signal.SIGINT, sigint_handler)

        while self.running:
            try:
                line = input(f"{BOLD}retro> {RESET}").strip()
                if not line:
                    continue
                if line.startswith("/"):
                    self._handle_command(line)
                else:
                    # Treat bare input as SMILES to run
                    self._handle_run(line)
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                continue

        print(f"\n{DIM}Goodbye!{RESET}")


def main():
    """Entry point for the REPL."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrosynthesis Agent REPL")
    parser.add_argument("--thinking", choices=("fast", "standard", "thorough"), default="fast")
    parser.add_argument("--config", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from core.config import load_config
    config = load_config(args.config, thinking=args.thinking)

    repl = RetroSynthesisREPL(config)
    repl.run()


if __name__ == "__main__":
    main()
