#!/usr/bin/env python3
"""HTTP-only Memory Service checks; no reply model or direct database access."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib import error, parse, request
import uuid


SCENARIO_TEXTS = (
    "Bana bundan sonra Ahmet Bey diye hitap et.",
    "Her sabah sade Türk kahvesi içerim.",
    "Her cumartesi sabah saat 7'de tenis oynarım.",
    "Her sabah sade Türk kahvesi içerim.",
    "Artık her cumartesi sabah saat 8'de tenis oynarım.",
    "Bugün dışarı çıkmak istiyorum.",
    "Bu akşam haberleri izlemek istiyorum.",
    "Şimdi bir bardak su içeceğim.",
    "Biraz dinlenmek istiyorum.",
    "Tansiyon ilacımı sabah kahvaltıdan sonra alıyorum.",
    "Başım ağrıyor.",
    "Her sabah ne içerim?",
    "Tenisi hangi saatte oynarım?",
    "Merhaba.",
    "Şimdi odama geçiyorum.",
)
EXPECTED_TYPES = (
    "long_term", "long_term", "long_term", "long_term", "long_term",
    "short_term", "short_term", "short_term", "short_term", "sensitive",
    "sensitive", "discard", "discard", "discard", "short_term",
)


class ScenarioError(RuntimeError):
    def __init__(self, message: str, detail: Any = None):
        self.detail = detail
        super().__init__(message)


class HttpError(ScenarioError):
    def __init__(self, method: str, path: str, status_code: int):
        self.status_code = status_code
        super().__init__(f"{method} {path}: HTTP {status_code}")


class JsonHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120):
        parsed = parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment
        ):
            raise ValueError("--memory-url must be an HTTP(S) URL without credentials/query.")
        if timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive.")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self, method: str, path: str, payload: dict | None = None,
        *, timeout_seconds: float | None = None,
    ) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = request.Request(
            self.base_url + path, data=body, method=method,
            headers={"Content-Type": "application/json"},
        )
        effective_timeout = self.timeout_seconds
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise ScenarioError("HTTP timeout must be positive")
            effective_timeout = min(effective_timeout, timeout_seconds)
        try:
            with request.urlopen(req, timeout=effective_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            exc.close()
            # Server bodies can contain sensitive user text: do not echo them.
            raise HttpError(method, path, exc.code) from exc
        except (error.URLError, OSError, TimeoutError) as exc:
            raise ScenarioError(f"{method} {path}: service unreachable or timed out") from exc
        except (UnicodeError, ValueError) as exc:
            raise ScenarioError(f"{method} {path}: invalid JSON response") from exc
        if not isinstance(result, dict):
            raise ScenarioError(f"{method} {path}: expected a JSON object")
        return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioError(message)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def aware_datetime(value: Any) -> datetime:
    require(isinstance(value, str), "Expected a serialized expiration timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ScenarioError("Invalid expiration timestamp") from exc
    require(result.tzinfo is not None and result.utcoffset() is not None, "Expiration must include a timezone")
    return result


class ScenarioRunner:
    def __init__(
        self, client: JsonHttpClient, ttl_seconds: float = 3,
        *, clock: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        output: Callable[[str], None] = print,
    ):
        if not 1 <= ttl_seconds <= 8:
            raise ValueError("--ttl-seconds must be between 1 and 8 (polling is bounded to 10 seconds).")
        self.client, self.ttl_seconds = client, ttl_seconds
        self.clock, self.monotonic, self.sleep, self.output = clock, monotonic, sleep, output
        run_id = uuid.uuid4().hex
        self.user_id = f"memory-scenario-{run_id}"
        self.session_id = f"memory-session-{run_id}"
        self.new_session_id = f"memory-new-session-{run_id}"
        self.other_user_id = f"memory-other-{run_id}"
        self.message_payloads: list[dict] = []
        self.results: list[dict] = []
        self.analyzer_ids: dict[str, str] | None = None

    def check(self, name: str, action: Callable[[], Any], *, mode: str = "storage") -> None:
        started = self.monotonic()
        try:
            detail = action()
            result = {"name": name, "mode": mode, "passed": True, "detail": detail, "error": None}
        except Exception as exc:
            result = {"name": name, "mode": mode, "passed": False, "detail": getattr(exc, "detail", None), "error": f"{type(exc).__name__}: {exc}"}
        result["elapsed_seconds"] = round(self.monotonic() - started, 3)
        self.results.append(result)
        self.output(f"{'PASS' if result['passed'] else 'FAIL'} [{mode}] {name}" + (f": {result['error']}" if result["error"] else ""))

    def context(self, user_id: str | None = None, session_id: str | None = None, **extra: Any) -> dict:
        return self.client.request("POST", "/v1/context:build", {
            "user_id": user_id or self.user_id, "session_id": session_id or self.session_id, **extra,
        })

    def temporary(self, *, timeout_seconds: float | None = None) -> list[dict]:
        result = self.client.request(
            "GET",
            f"/v1/sessions/{self.session_id}/temporary-memories?" + parse.urlencode({"user_id": self.user_id}),
            timeout_seconds=timeout_seconds,
        )
        items = result.get("temporary_memories")
        require(isinstance(items, list), "Missing temporary_memories array; migrate/restart the API")
        require(result.get("memory_count") == len(items), "Incorrect temporary memory_count")
        require(isinstance(result.get("max_items"), int) and isinstance(result.get("max_tokens"), int), "Missing temporary memory limits")
        return items

    def create_temporary(self, key: str, value: str, expires_at: datetime) -> dict:
        return self.client.request("POST", "/v1/temporary-memories", {
            "user_id": self.user_id, "session_id": self.session_id,
            "category": "session", "key": key, "value": value,
            "expires_at": expires_at.isoformat(), "occurred_at": self.clock().isoformat(),
        })

    def profile_lifecycle(self) -> dict:
        payload = {"user_id": self.user_id, "category": "routine", "key": "weekly_tennis", "value": SCENARIO_TEXTS[2]}
        first = self.client.request("POST", "/v1/memories", payload)
        repeat = self.client.request("POST", "/v1/memories", payload)
        require(repeat.get("status") == "unchanged" and repeat.get("fact_id") == first.get("fact_id"), "Identical fact was duplicated")
        changed = self.client.request("POST", "/v1/memories", {**payload, "value": SCENARIO_TEXTS[4]})
        require(changed.get("supersedes_id") == first.get("fact_id"), "Changed fact did not supersede the previous value")
        active = self.client.request("GET", f"/v1/users/{self.user_id}/memories").get("items", [])
        require(len(active) == 1 and active[0].get("value") == SCENARIO_TEXTS[4], "Expected only the updated active profile fact")
        return {"active_fact_id": changed["fact_id"], "superseded_fact_id": first["fact_id"]}

    def multiple_temporary_items(self) -> dict:
        expiry = self.clock() + timedelta(minutes=20)
        self.create_temporary("evening_news", SCENARIO_TEXTS[6], expiry)
        self.create_temporary("rest", SCENARIO_TEXTS[8], expiry)
        items = self.temporary()
        require({item.get("key") for item in items} == {"evening_news", "rest"}, "Different goals overwrote one another")
        updated_value = "Bu akşam önce haberleri izlemek istiyorum."
        self.create_temporary("evening_news", updated_value, expiry)
        items = self.temporary()
        require(len(items) == 2, "Same-slot update duplicated an item or deleted another goal")
        by_key = {item["key"]: item for item in items}
        require(by_key["evening_news"].get("value") == updated_value and by_key["rest"].get("value") == SCENARIO_TEXTS[8], "Incorrect same-slot temporary update")
        context = self.context()
        require({item.get("key") for item in context.get("temporary_memories", [])} == set(by_key), "Context omitted a live temporary goal")
        return {"keys": sorted(by_key), "updated_key": "evening_news"}

    def independent_expiry(self) -> dict:
        now = self.clock()
        self.create_temporary("ttl_short", "Kısa süreli test niyeti", now + timedelta(seconds=self.ttl_seconds))
        self.create_temporary("ttl_live", "Daha uzun süreli test niyeti", now + timedelta(minutes=5))
        require({"ttl_short", "ttl_live"}.issubset({item.get("key") for item in self.temporary()}), "Temporary items were not initially live")
        deadline = self.monotonic() + 10
        polls = 0
        while True:
            remaining = deadline - self.monotonic()
            require(remaining > 0, "Short TTL item stayed visible after the bounded 10-second poll")
            items = self.temporary(timeout_seconds=remaining)
            polls += 1
            keys = {item.get("key") for item in items}
            require("ttl_live" in keys and "rest" in keys, "Expiring one temporary item removed another live item")
            if "ttl_short" not in keys:
                break
            require(self.monotonic() < deadline, "Short TTL item stayed visible after the bounded 10-second poll")
            self.sleep(min(0.25, max(0, deadline - self.monotonic())))
        context_keys = {item.get("key") for item in self.context().get("temporary_memories", [])}
        require("ttl_short" not in context_keys and "ttl_live" in context_keys, "Context did not apply independent item TTL")
        return {"ttl_seconds": self.ttl_seconds, "polls": polls, "expired_key": "ttl_short", "live_key": "ttl_live"}

    def conversation_suffix(self) -> dict:
        started = self.clock()
        for index, text in enumerate(SCENARIO_TEXTS):
            payload = {"message_id": str(uuid.uuid4()), "user_id": self.user_id, "role": "user", "content": text, "occurred_at": (started + timedelta(milliseconds=index)).isoformat()}
            self.client.request("POST", f"/v1/sessions/{self.session_id}/messages", payload)
            self.message_payloads.append(payload)
        result = self.client.request("GET", f"/v1/sessions/{self.session_id}/messages?" + parse.urlencode({"user_id": self.user_id}))
        require(result.get("max_messages") == 10, "This scenario requires MEMORY_WINDOW_MAX_MESSAGES=10")
        messages = result.get("messages", [])
        require([item.get("content") for item in messages] == list(SCENARIO_TEXTS[-10:]), "Conversation window was not the newest 10-message suffix of 15 inputs")
        context = self.context()
        require([item.get("content") for item in context.get("recent_messages", [])] == list(SCENARIO_TEXTS[-10:]), "Context conversation suffix mismatch")
        return {"input_count": 15, "window_count": len(messages), "first_visible_input": 6}

    def replay_message(self) -> dict:
        require(bool(self.message_payloads), "Conversation seed did not complete")
        payload = self.message_payloads[-1]
        result = self.client.request("POST", f"/v1/sessions/{self.session_id}/messages", payload)
        require(result.get("status") == "duplicate", "Replayed message_id was not idempotent")
        context = self.context()
        require(context.get("message_refs", []).count(payload["message_id"]) == 1, "Replayed message appeared more than once")
        require(len(context.get("recent_messages", [])) == 10, "Replay changed the bounded window")
        return {"message_id": payload["message_id"], "status": result["status"]}

    def session_isolation(self) -> dict:
        context = self.context(session_id=self.new_session_id)
        require(context.get("temporary_memories") == [] and context.get("recent_messages") == [] and not context.get("session"), "A new session inherited old temporary state/history")
        require(any(item.get("value") == SCENARIO_TEXTS[4] for item in context.get("profile_facts", [])), "Long-term profile was not retained across sessions")
        other_context = self.context(user_id=self.other_user_id, session_id=f"other-{uuid.uuid4().hex}")
        require(other_context.get("profile_facts") == [], "Another user received the test profile")
        return {"new_session_id": self.new_session_id, "profile_retained": True, "temporary_count": 0, "history_count": 0}

    def ownership_conflict(self) -> dict:
        paths = (
            f"/v1/sessions/{self.session_id}/temporary-memories?" + parse.urlencode({"user_id": self.other_user_id}),
            f"/v1/sessions/{self.session_id}/messages?" + parse.urlencode({"user_id": self.other_user_id}),
        )
        for path in paths:
            self.expect_status("GET", path, None, 403)
        self.expect_status("POST", "/v1/context:build", {"user_id": self.other_user_id, "session_id": self.session_id}, 403)
        self.expect_status("POST", "/v1/temporary-memories", {"user_id": self.other_user_id, "session_id": self.session_id, "key": "foreign", "value": "Must not be stored", "expires_at": (self.clock() + timedelta(minutes=5)).isoformat()}, 403)
        return {"expected_http_status": 403, "checks": 4}

    def expect_status(self, method: str, path: str, payload: dict | None, expected: int) -> None:
        try:
            self.client.request(method, path, payload)
        except HttpError as exc:
            require(exc.status_code == expected, f"Expected HTTP {expected}, got HTTP {exc.status_code}")
        else:
            raise ScenarioError(f"Expected HTTP {expected}, request unexpectedly succeeded")

    def run_storage(self) -> None:
        self.check("profile duplicate and supersession", self.profile_lifecycle)
        self.check("multiple temporary goals and same-slot update", self.multiple_temporary_items)
        self.check("independent temporary-item expiry", self.independent_expiry)
        self.check("newest 10 of 15 conversation messages", self.conversation_suffix)
        self.check("message_id replay", self.replay_message)
        self.check("new-session and user isolation", self.session_isolation)
        self.check("session ownership conflicts", self.ownership_conflict)

    def run_analyzer(self) -> None:
        run_id = uuid.uuid4().hex
        user_id, session_id = f"memory-analyzer-{run_id}", f"analyzer-session-{run_id}"
        self.analyzer_ids = {"user_id": user_id, "session_id": session_id}
        ingests: list[dict] = []

        def ingest(index: int) -> dict:
            payload = {"event_id": str(uuid.uuid4()), "user_id": user_id, "session_id": session_id, "text": SCENARIO_TEXTS[index], "occurred_at": self.clock().isoformat()}
            response = self.client.request("POST", "/v1/interactions:process", payload)
            ingests.append({"index": index, "payload": payload, "response": response})
            decisions = response.get("decisions", [])
            require(bool(decisions), "Analyzer returned no decisions")
            sources = {item.get("analyzer_source") for item in decisions}
            model_backed = all(
                item.get("analyzer_source") == "ollama"
                or (
                    item.get("analyzer_source") == "rules_guard"
                    and isinstance(item.get("analysis"), dict)
                    and item["analysis"].get("upstream_analyzer_source") == "ollama"
                )
                for item in decisions
            )
            require(
                model_backed,
                "Real analyzer required; observed "
                f"{sorted(str(source) for source in sources)} "
                "(rules/rules_fallback is not model-backed)",
            )
            expected = EXPECTED_TYPES[index]
            require({item.get("memory_type") for item in decisions} == {expected}, f"Expected {expected}, observed {[item.get('memory_type') for item in decisions]}")
            status = "pending" if expected == "sensitive" else "ignored" if expected == "discard" else "auto_applied"
            require(all(item.get("status") == status for item in decisions), f"Expected candidate status {status}")
            if expected == "sensitive":
                require(all(item.get("sensitivity") == "health" and item.get("requires_confirmation") for item in decisions), "Health information must await explicit confirmation")
                if index == 9:
                    require(all(item.get("expires_at") is None for item in decisions), "Persistent medication routine received temporary expiry")
                else:
                    require(all(aware_datetime(item.get("expires_at")) > aware_datetime(payload["occurred_at"]) for item in decisions), "Current symptom needs finite future expiry")
            if expected == "short_term":
                require(all(aware_datetime(item.get("expires_at")) > aware_datetime(payload["occurred_at"]) for item in decisions), "Short-term intent needs finite future expiry")
            return {"input_number": index + 1, "text": payload["text"], "expected_type": expected, "decisions": decisions}

        def ingest_with_report(index: int) -> dict:
            try:
                return ingest(index)
            except Exception as exc:
                attempt = next((item for item in reversed(ingests) if item["index"] == index), None)
                detail = {
                    "input_number": index + 1, "text": SCENARIO_TEXTS[index],
                    "expected_type": EXPECTED_TYPES[index],
                    "decisions": attempt["response"].get("decisions", []) if attempt else [],
                }
                raise ScenarioError(str(exc), detail) from exc

        for index in range(len(SCENARIO_TEXTS)):
            self.check(f"input {index + 1}: {EXPECTED_TYPES[index]}", lambda index=index: ingest_with_report(index), mode="analyzer")

        def profile_check() -> dict:
            facts = self.client.request("GET", f"/v1/users/{user_id}/memories").get("items", [])
            coffee = [fact for fact in facts if "kahve" in json.dumps(fact.get("value"), ensure_ascii=False).casefold()]
            tennis = [fact for fact in facts if "tenis" in json.dumps(fact.get("value"), ensure_ascii=False).casefold()]
            require(len(coffee) == 1, "Coffee repeat did not consolidate to one active fact")
            require(len(tennis) == 1 and "8" in json.dumps(tennis[0].get("value")), "Tennis correction did not produce one active 8 o'clock fact")
            require(not any(fact.get("sensitivity") == "health" for fact in facts), "Unconfirmed health information leaked into active profile")
            return {"active_coffee_count": len(coffee), "active_tennis_count": len(tennis), "tennis_value": tennis[0]["value"]}

        def temporary_collision_check() -> dict:
            temporary = self.client.request(
                "GET",
                f"/v1/sessions/{session_id}/temporary-memories?"
                + parse.urlencode({"user_id": user_id}),
            ).get("temporary_memories", [])
            serialized = json.dumps(temporary, ensure_ascii=False).casefold()
            require("dışarı çıkmak" in serialized, "Outdoor intent was lost from temporary memory")
            require("haberleri izlemek" in serialized, "News intent was lost from temporary memory")
            keys = [item.get("key") for item in temporary]
            require(len(keys) == len(set(keys)), "Active temporary memories contain duplicate slots")
            return {
                "temporary_count": len(temporary),
                "outdoor_and_news_coexist": True,
                "active_keys_are_unique": True,
            }

        def privacy_check() -> dict:
            raw = self.client.request("GET", f"/v1/sessions/{session_id}/messages?" + parse.urlencode({"user_id": user_id})).get("messages", [])
            raw_texts = [item.get("content") for item in raw]
            require(raw_texts == list(SCENARIO_TEXTS[-10:]), "Analyzer ingestion did not retain the newest raw 10-message suffix")
            context = self.context(user_id=user_id, session_id=session_id)
            hidden = set(SCENARIO_TEXTS[9:11])
            require(not hidden.intersection(item.get("content") for item in context.get("recent_messages", [])), "Pending sensitive user text leaked into default context")
            require(context.get("history_policy", {}).get("include_unconfirmed_sensitive_history") is False, "Default sensitive-history policy was not explicit")
            trusted = self.context(user_id=user_id, session_id=session_id, include_unconfirmed_sensitive_history=True)
            require(hidden.issubset({item.get("content") for item in trusted.get("recent_messages", [])}), "Explicit trusted-backend raw-history opt-in did not restore pending text")
            temporary = self.client.request(
                "GET", f"/v1/sessions/{session_id}/temporary-memories?" + parse.urlencode({"user_id": user_id}),
            ).get("temporary_memories", [])
            serialized_temporary = json.dumps(temporary, ensure_ascii=False)
            require(not any(text in serialized_temporary for text in hidden), "Unconfirmed health data leaked into temporary memory")
            return {"raw_window_count": len(raw), "default_pending_text_hidden": True,
                    "explicit_opt_in_restored": True, "pending_health_in_temporary": False}

        def replay_check() -> dict:
            last = next((item for item in reversed(ingests) if item["index"] == 14), None)
            require(last is not None, "Final interaction did not complete")
            response = self.client.request("POST", "/v1/interactions:process", last["payload"])
            require(response.get("status") == "duplicate", "Interaction replay was not idempotent")
            require([item.get("candidate_id") for item in response.get("decisions", [])] == [item.get("candidate_id") for item in last["response"].get("decisions", [])], "Interaction replay created new candidates")
            return {"event_id": last["payload"]["event_id"], "status": response["status"]}

        self.check("active coffee deduplication and tennis correction", profile_check, mode="analyzer")
        self.check("unrelated temporary intents survive key collisions", temporary_collision_check, mode="analyzer")
        self.check("pending sensitive raw-history policy", privacy_check, mode="analyzer")
        self.check("interaction event_id replay", replay_check, mode="analyzer")

    def report(self) -> dict:
        analyzer_decisions = [
            decision
            for result in self.results
            if result.get("mode") == "analyzer" and isinstance(result.get("detail"), dict)
            for decision in result["detail"].get("decisions", [])
        ]
        guard_interventions = sum(
            decision.get("analyzer_source") == "rules_guard"
            for decision in analyzer_decisions
        )
        return {
            "runner": "memory_scenarios_v1", "created_at": self.clock().isoformat(),
            "memory_url": self.client.base_url, "user_id": self.user_id,
            "session_id": self.session_id, "new_session_id": self.new_session_id,
            "other_user_id": self.other_user_id, "analyzer_ids": self.analyzer_ids,
            "ttl_seconds": self.ttl_seconds, "passed": all(item["passed"] for item in self.results),
            "analyzer_summary": {
                "decision_count": len(analyzer_decisions),
                "pure_ollama_count": sum(
                    decision.get("analyzer_source") == "ollama"
                    for decision in analyzer_decisions
                ),
                "guard_intervention_count": guard_interventions,
                "fallback_count": sum(
                    decision.get("analyzer_source") in {"rules", "rules_fallback"}
                    for decision in analyzer_decisions
                ),
            },
            "checks": self.results,
            "notes": [
                "Storage checks bypass extraction and use no reply model. They are not semantic model-quality claims.",
                "Optional analyzer checks are finite behavioral cases. A model-backed rules_guard is a passing safety intervention and is counted separately; rules/rules_fallback is a failure.",
                "Fresh synthetic records remain in the database selected by the running Memory API; no automatic cleanup/deletion.",
            ],
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-url", default=os.getenv("MEMORY_SERVICE_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--ttl-seconds", type=float, default=3, help="Temporary expiry probe, 1–8 seconds (default: 3)")
    parser.add_argument("--with-analyzer", action="store_true", help="Also ingest the original 15 Turkish inputs through the real Ollama extractor")
    parser.add_argument("--json-report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runner = ScenarioRunner(JsonHttpClient(args.memory_url, args.timeout_seconds), args.ttl_seconds)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"Memory API: {runner.client.base_url}\nFresh user: {runner.user_id}\nFresh session: {runner.session_id}")
    print("Writes synthetic isolated records to the API's database; records are left in place.")
    runner.run_storage()
    if args.with_analyzer:
        print("Running real extraction checks sequentially; this can take several minutes. No reply model is called.")
        runner.run_analyzer()
    report = runner.report()
    if args.with_analyzer:
        analyzer_summary = report["analyzer_summary"]
        print(
            "Analyzer provenance: "
            f"pure_ollama={analyzer_summary['pure_ollama_count']} "
            f"rules_guard={analyzer_summary['guard_intervention_count']} "
            f"fallback={analyzer_summary['fallback_count']}"
        )
    if args.json_report:
        try:
            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            args.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Could not save report: {exc}", file=sys.stderr)
            return 2
        print(f"Report: {args.json_report.resolve()}")
    print("All requested checks passed." if report["passed"] else "Some checks failed; inspect the report/FAIL lines.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
