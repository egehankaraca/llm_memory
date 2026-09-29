#!/usr/bin/env python3
"""Interactive Memory Service demo with no reply LLM, STT, or TTS."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib import parse
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_http import JsonHttpClient, MemoryClientError  # noqa: E402


HELP = """Commands:
  /context [query]  Show the bounded, query-aware context package
  /profile         Show active long-term profile facts
  /memories        Alias for /profile
  /episodes        Show append-only episodic memories
  /session         Show active short-term/session memories
  /temporary       Alias for /session
  /status EVENT_ID Show an async outbox job and its eventual decisions
  /debug           Toggle technical output
  /help            Show commands
  /exit            Exit

Any other text is sent only to the Memory Service for classification/storage.
No reply model, STT, or TTS is called.
"""

SIMPLE_HELP = """Commands:
  /profile   Show long-term memories
  /session   Show short-term memories
  /episodes  Show episodic memories
  /debug     Toggle technical details
  /help      Show commands
  /exit      Exit

Type naturally. Relevant memory is retrieved automatically before the input is
processed for storage.
"""


DESTINATIONS = {
    "profile": "profile (PostgreSQL + pgvector)",
    "episode": "episodic memory (PostgreSQL)",
    "session": "session memory (TTL)",
    "discard": "none",
}


@dataclass(frozen=True)
class DemoSettings:
    memory_url: str
    timeout_seconds: int
    async_memory: bool = True
    debug: bool = False
    simple: bool = False
    auto_wait_seconds: int = 0
    query_before_store: bool = False


class MemoryTerminalDemo:
    def __init__(
        self,
        settings: DemoSettings,
        user_id: str,
        session_id: str,
        *,
        client: JsonHttpClient | None = None,
        output: Callable[[str], None] = print,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_id.strip() or len(user_id) > 128:
            raise ValueError("user_id must contain 1-128 characters")
        if not session_id.strip() or len(session_id) > 128:
            raise ValueError("session_id must contain 1-128 characters")
        self.settings = settings
        self.user_id = user_id
        self.session_id = session_id
        self.client = client or JsonHttpClient(
            settings.memory_url, settings.timeout_seconds
        )
        self.output = output
        self.monotonic = monotonic
        self.sleeper = sleeper
        self.debug = settings.debug

    def print_json(self, value: object) -> None:
        self.output(json.dumps(value, ensure_ascii=False, indent=2))

    def check_services(self) -> dict[str, Any]:
        health = self.client.request("GET", "/healthz")
        if health.get("status") != "ok":
            raise MemoryClientError("Memory Service healthcheck failed")
        analyzer = self.client.request("GET", "/v1/analyzer/status")
        return analyzer

    @staticmethod
    def compact_decision(decision: dict[str, Any], *, debug: bool) -> dict[str, Any]:
        memory_type = decision.get("memory_type")
        keys = (
            "candidate_id",
            "event_id",
            "memory_type",
            "scope",
            "category",
            "key",
            "value",
            "sensitivity",
            "confidence",
            "analyzer_source",
            "status",
            "consolidation_action",
            "consolidates_fact_id",
            "expires_at",
            "applied_ref",
        )
        result = {key: decision.get(key) for key in keys}
        scope = decision.get("scope")
        if scope is None:
            scope = (
                "profile" if memory_type == "long_term"
                else "session" if memory_type == "short_term"
                else "discard" if memory_type == "discard"
                else None
            )
        destination = DESTINATIONS.get(str(scope), "unknown")
        result["destination"] = destination
        if debug:
            result["reason"] = decision.get("reason")
            result["analysis"] = decision.get("analysis")
        return result

    def show_simple_decisions(self, response: dict[str, Any]) -> None:
        decisions = response.get("decisions", [])
        if not isinstance(decisions, list):
            raise MemoryClientError("Memory Service decisions format is invalid")
        if not decisions:
            self.output("Memory processing completed; no decision was returned.")
            return
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            scope = decision.get("scope") or decision.get("memory_type")
            status = decision.get("status")
            if scope == "discard" or decision.get("memory_type") == "discard":
                self.output("• Not stored: this input is not a reusable memory.")
                continue
            category = decision.get("category") or "memory"
            key = decision.get("key") or "value"
            action = decision.get("consolidation_action") or "stored"
            self.output(f"✓ {scope}: {category}.{key} ({action}, {status})")
            if self.debug:
                self.print_json(self.compact_decision(decision, debug=True))

    def wait_for_event(self, event_id: str) -> dict[str, Any]:
        deadline = self.monotonic() + self.settings.auto_wait_seconds
        while True:
            response = self.status(event_id, render=False)
            job = response.get("job")
            if not isinstance(job, dict):
                raise MemoryClientError("Memory Service job status is invalid")
            job_status = job.get("status")
            if job_status == "completed":
                return response
            if job_status in {"failed", "canceled"}:
                raise MemoryClientError(
                    f"Memory processing ended with status {job_status}: "
                    f"{job.get('last_error') or 'unknown error'}"
                )
            if self.monotonic() >= deadline:
                raise MemoryClientError(
                    f"Memory processing did not finish within "
                    f"{self.settings.auto_wait_seconds} seconds"
                )
            self.sleeper(0.5)

    def wait_for_embeddings(self) -> dict[str, Any]:
        deadline = self.monotonic() + self.settings.auto_wait_seconds
        path = "/v1/embeddings/status?" + parse.urlencode({"user_id": self.user_id})
        while True:
            response = self.client.request("GET", path)
            if response.get("missing_count") == 0:
                return response
            failed = response.get("job_counts", {}).get("failed", 0)
            if failed:
                raise MemoryClientError(f"{failed} embedding job(s) failed")
            if self.monotonic() >= deadline:
                raise MemoryClientError(
                    f"Embeddings did not finish within "
                    f"{self.settings.auto_wait_seconds} seconds"
                )
            self.sleeper(0.5)

    def show_decisions(
        self,
        response: dict[str, Any],
        *,
        elapsed_seconds: float | None = None,
    ) -> None:
        decisions = response.get("decisions")
        if not isinstance(decisions, list):
            raise MemoryClientError("Memory Service decisions format is invalid")
        heading = f"Memory result: {response.get('status', 'unknown')}"
        self.output(f"\n{heading}")
        service_timing = response.get("timing")
        timing: dict[str, Any] = {
            "client_round_trip_ms": (
                round(elapsed_seconds * 1000, 3)
                if elapsed_seconds is not None
                else None
            ),
        }
        if isinstance(service_timing, dict):
            timing.update(service_timing)
        self.output("Timing:")
        self.print_json(timing)
        if not decisions:
            self.output("No decision is available yet.")
            return
        for index, decision in enumerate(decisions, start=1):
            if not isinstance(decision, dict):
                raise MemoryClientError("Memory Service returned an invalid decision")
            self.output(f"\nDecision {index}:")
            self.print_json(self.compact_decision(decision, debug=self.settings.debug))

    def process_text(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise MemoryClientError("Message cannot be empty")
        if len(text) > 10_000:
            raise MemoryClientError("Message cannot exceed 10,000 characters")
        event_id = str(uuid.uuid4())
        path = (
            "/v1/interactions:enqueue"
            if self.settings.async_memory
            else "/v1/interactions:process"
        )
        started = self.monotonic()
        response = self.client.request("POST", path, {
            "event_id": event_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "text": text,
        })
        elapsed = self.monotonic() - started
        if self.settings.async_memory:
            job = response.get("job")
            if not isinstance(job, dict):
                raise MemoryClientError("Memory Service outbox response is invalid")
            if self.settings.simple:
                self.output(f"✓ Accepted in {elapsed * 1000:.1f} ms; processing memory…")
            else:
                self.output(f"\nMemory enqueue: {response.get('status')} ({elapsed:.3f}s)")
                self.print_json({
                    "event_id": response.get("event_id"),
                    "job_status": job.get("status"),
                    "attempt_count": job.get("attempt_count"),
                })
                self.output(f"Check later with: /status {event_id}")
            if self.settings.auto_wait_seconds > 0:
                completed = self.wait_for_event(event_id)
                if self.settings.simple:
                    self.show_simple_decisions(completed)
                else:
                    self.show_decisions(completed)
                decisions = completed.get("decisions", [])
                wrote_profile = any(
                    isinstance(decision, dict)
                    and decision.get("status") == "auto_applied"
                    and (
                        decision.get("scope") == "profile"
                        or decision.get("memory_type") == "long_term"
                    )
                    for decision in decisions
                )
                if wrote_profile:
                    embedding_status = self.wait_for_embeddings()
                    if self.settings.simple:
                        self.output(
                            "✓ Embedding ready "
                            f"({embedding_status.get('indexed_count', 0)} indexed)"
                        )
        else:
            self.show_decisions(response, elapsed_seconds=elapsed)
        return response

    def context(
        self,
        query: str | None = None,
        *,
        quiet_if_empty: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
        }
        if query:
            payload["query"] = query
        started = self.monotonic()
        response = self.client.request("POST", "/v1/context:build", payload)
        elapsed = self.monotonic() - started
        service_timing = response.get("timing")
        timing: dict[str, Any] = {
            "client_round_trip_ms": round(elapsed * 1000, 3),
        }
        if isinstance(service_timing, dict):
            timing.update(service_timing)
        else:
            semantic = response.get("profile_retrieval", {}).get("semantic", {})
            if isinstance(semantic, dict):
                timing.update({
                    "query_embedding_ms": semantic.get("query_embedding_ms"),
                    "vector_search_ms": semantic.get("vector_search_ms"),
                })
        profile_facts = response.get("profile_facts", [])
        episodes = response.get("episodes", [])
        temporary = response.get("temporary_memories", [])
        if self.settings.simple and not self.debug:
            memories = [
                *(profile_facts if isinstance(profile_facts, list) else []),
                *(episodes if isinstance(episodes, list) else []),
                *(temporary if isinstance(temporary, list) else []),
            ]
            if memories:
                self.output("\nRelevant stored memory:")
                for item in memories:
                    if not isinstance(item, dict):
                        continue
                    value = item.get("value")
                    if value is None and isinstance(item.get("value_json"), dict):
                        value = item["value_json"].get("value")
                    label = ".".join(
                        str(part) for part in (item.get("category"), item.get("key"))
                        if part
                    )
                    self.output(f"  • {label or 'memory'}: {value}")
                self.output(
                    "  Retrieval: "
                    f"{timing.get('client_round_trip_ms')} ms total, "
                    f"{timing.get('vector_search_ms')} ms vector search"
                )
            elif not quiet_if_empty:
                self.output("\nRelevant stored memory: none")
            return response

        self.output("\nRetrieval timing:")
        self.print_json(timing)
        compact = {
            "as_of": response.get("as_of"),
            "user_id": response.get("user_id"),
            "profile_facts": response.get("profile_facts", []),
            "episodes": response.get("episodes", []),
            "episode_memory_budget": response.get("episode_memory_budget", {}),
            "temporary_memories": response.get("temporary_memories", []),
            "session": response.get("session", {}),
            "temporary_observations": response.get("temporary_observations", []),
            "recent_messages": response.get("recent_messages", []),
            "conversation_window": response.get("conversation_window", {}),
            "profile_retrieval": response.get("profile_retrieval", {}),
            "timing": response.get("timing", {}),
            "memory_refs": response.get("memory_refs", []),
        }
        self.print_json(compact)
        return response

    def memories(self) -> dict[str, Any]:
        path = f"/v1/users/{parse.quote(self.user_id, safe='')}/memories"
        response = self.client.request("GET", path)
        self.print_json(response)
        return response

    def episodes(self) -> dict[str, Any]:
        path = f"/v1/users/{parse.quote(self.user_id, safe='')}/episodes"
        response = self.client.request("GET", path)
        self.print_json(response)
        return response

    def temporary(self) -> dict[str, Any]:
        path = (
            f"/v1/sessions/{parse.quote(self.session_id, safe='')}/temporary-memories?"
            + parse.urlencode({"user_id": self.user_id})
        )
        response = self.client.request("GET", path)
        self.print_json(response)
        return response

    def status(self, event_id: str, *, render: bool = True) -> dict[str, Any]:
        if not event_id:
            raise MemoryClientError("/status requires an event ID")
        path = (
            f"/v1/interactions/{parse.quote(event_id, safe='')}/status?"
            + parse.urlencode({"user_id": self.user_id})
        )
        response = self.client.request("GET", path)
        if render:
            self.print_json(response)
        return response

    def handle_command(self, text: str) -> bool:
        command, _, argument = text.partition(" ")
        argument = argument.strip()
        if command == "/exit":
            return False
        if command == "/help":
            self.output(SIMPLE_HELP if self.settings.simple else HELP)
        elif command == "/context":
            self.context(argument or None)
        elif command in {"/profile", "/memories"}:
            self.memories()
        elif command == "/episodes":
            self.episodes()
        elif command in {"/session", "/temporary"}:
            self.temporary()
        elif command == "/status":
            self.status(argument)
        elif command == "/debug":
            self.debug = not self.debug
            self.output(f"Debug output: {'on' if self.debug else 'off'}")
        else:
            self.output("Unknown command. Use /help.")
        return True


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def environment_boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.casefold().strip()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", help="Reuse this ID to keep long-term memory")
    parser.add_argument("--session-id", help="Reuse this ID to keep session memory")
    parser.add_argument(
        "--memory-url",
        default=os.getenv("MEMORY_SERVICE_URL", "http://127.0.0.1:8001"),
    )
    parser.add_argument("--timeout-seconds", type=positive_integer, default=120)
    memory_mode = parser.add_mutually_exclusive_group()
    memory_mode.add_argument(
        "--async-memory",
        dest="async_memory",
        action="store_true",
        help="Enqueue memory analysis (default; requires the worker)",
    )
    memory_mode.add_argument(
        "--sync-memory",
        dest="async_memory",
        action="store_false",
        help="Wait for extraction; use only for analyzer diagnostics",
    )
    parser.set_defaults(async_memory=None)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Also show analyzer reason and audit metadata",
    )
    parser.add_argument("--simple", action="store_true", help="Use concise demo output")
    parser.add_argument(
        "--auto-wait-seconds",
        type=int,
        default=0,
        help="Automatically wait for extraction and embeddings",
    )
    parser.add_argument(
        "--query-before-store",
        action="store_true",
        help="Retrieve relevant context for each natural-language input",
    )
    parser.add_argument(
        "--text",
        action="append",
        help="Process one message and exit; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suffix = uuid.uuid4().hex[:8]
    settings = DemoSettings(
        memory_url=args.memory_url,
        timeout_seconds=args.timeout_seconds,
        async_memory=(
            args.async_memory
            if args.async_memory is not None
            else environment_boolean("MEMORY_ASYNC_INGESTION", True)
        ),
        debug=args.debug,
        simple=args.simple,
        auto_wait_seconds=max(0, args.auto_wait_seconds),
        query_before_store=args.query_before_store,
    )
    try:
        demo = MemoryTerminalDemo(
            settings,
            args.user_id or f"memory-demo-{suffix}",
            args.session_id or f"memory-session-{suffix}",
        )
        analyzer = demo.check_services()
        print(f"User: {demo.user_id}")
        print(f"Session: {demo.session_id}")
        if not settings.simple:
            print("Reply model: disabled")
            print("STT/TTS: disabled")
            print(
                "Memory ingestion: "
                + ("async outbox/worker" if settings.async_memory else "synchronous")
            )
            print(
                f"Memory analyzer: {analyzer.get('provider')} / {analyzer.get('model')} "
                f"(available={analyzer.get('available')})"
            )
        if args.text:
            for text in args.text:
                print(f"\nInput: {text}")
                demo.process_text(text)
            return 0
        print(SIMPLE_HELP if settings.simple else HELP)
        while True:
            text = input("\nMemory> ").strip()
            if not text:
                continue
            try:
                if text.startswith("/"):
                    if not demo.handle_command(text):
                        break
                else:
                    if settings.query_before_store:
                        demo.context(text, quiet_if_empty=True)
                    demo.process_text(text)
            except MemoryClientError as exc:
                print(f"[Error] {exc}", file=sys.stderr)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return 0
    except (MemoryClientError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
