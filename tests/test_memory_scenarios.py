"""Runner tests use HTTP fakes only: no main/database imports or configured DB."""

from datetime import datetime, timedelta, timezone
import io
import json
import unittest
from unittest.mock import patch
from urllib import error, parse

from scripts import run_memory_scenarios as scenarios


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0
        self.started = datetime(2026, 9, 14, 10, tzinfo=timezone.utc)
        self.sleeps = []

    def now(self):
        return self.started + timedelta(seconds=self.elapsed)

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.elapsed += seconds


class FakeMemoryHttp:
    """Small HTTP contract fake; deliberately does not import application code."""

    base_url = "http://fake-memory:8001"

    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.facts = {}
        self.temporary_items = {}
        self.messages = {}
        self.owners = {}
        self.events = {}
        self.source = "ollama"
        self.sticky_ttl = False

    def owner(self, method, path, user_id, session_id):
        if session_id in self.owners and self.owners[session_id] != user_id:
            raise scenarios.HttpError(method, path, 403)

    def visible_temporary(self, user_id, session_id):
        return [item for item in self.temporary_items.values() if item["owner_user_id"] == user_id and item["session_id"] == session_id and (self.sticky_ttl or scenarios.aware_datetime(item["expires_at"]) > self.clock.now())]

    def window(self, user_id, session_id):
        return sorted([item for item in self.messages.values() if item["user_id"] == user_id and item["session_id"] == session_id], key=lambda item: item["occurred_at"])[-10:]

    def request(self, method, path, payload=None, *, timeout_seconds=None):
        self.calls.append((method, path, payload))
        url = parse.urlsplit(path)
        parts = url.path.split("/")
        query = parse.parse_qs(url.query)
        if path == "/v1/memories":
            slot = (payload["user_id"], payload["category"], payload["key"])
            current = self.facts.get(slot)
            if current and current["value"] == payload["value"]:
                return {"status": "unchanged", "fact_id": current["id"]}
            fact = {**payload, "id": f"fact-{len(self.calls)}", "sensitivity": payload.get("sensitivity", "normal")}
            self.facts[slot] = fact
            return {"status": "created", "fact_id": fact["id"], "supersedes_id": current["id"] if current else None}
        if method == "GET" and len(parts) == 5 and parts[2] == "users" and parts[4] == "memories":
            return {"items": [fact for (owner, *_), fact in self.facts.items() if owner == parts[3]]}
        if path == "/v1/temporary-memories":
            user_id, session_id = payload["user_id"], payload["session_id"]
            self.owner(method, path, user_id, session_id)
            self.owners[session_id] = user_id
            slot = (user_id, session_id, payload.get("category", "session"), payload["key"])
            item = {
                "memory_id": f"temporary-{len(self.calls)}",
                "owner_user_id": user_id,
                "session_id": session_id,
                "category": payload.get("category", "session"),
                "key": payload["key"],
                "value": payload["value"],
                "occurred_at": payload.get("occurred_at", self.clock.now().isoformat()),
                "expires_at": payload["expires_at"],
                "sensitivity": payload.get("sensitivity", "normal"),
                "provenance": {
                    "source_event_id": payload.get("source_event_id"),
                    "verification_status": payload.get("verification_status", "unverified"),
                    "confidence": payload.get("confidence", 1.0),
                },
            }
            self.temporary_items[slot] = item
            return {"status": "success", "temporary_memory": item}
        if len(parts) == 5 and parts[2] == "sessions":
            session_id = parts[3]
            user_id = payload["user_id"] if payload else query["user_id"][0]
            self.owner(method, path, user_id, session_id)
            if parts[4] == "temporary-memories":
                items = self.visible_temporary(user_id, session_id)
                return {"temporary_memories": items, "memory_count": len(items), "max_items": 20, "max_tokens": 1500}
            if parts[4] == "messages" and method == "POST":
                self.owners[session_id] = user_id
                duplicate = payload["message_id"] in self.messages
                item = {**payload, "session_id": session_id}
                self.messages.setdefault(payload["message_id"], item)
                return {"status": "duplicate" if duplicate else "created", "message": item}
            if parts[4] == "messages":
                return {"messages": self.window(user_id, session_id), "max_messages": 10, "max_tokens": 2000}
        if path == "/v1/context:build":
            user_id, session_id = payload["user_id"], payload["session_id"]
            self.owner(method, path, user_id, session_id)
            items = self.visible_temporary(user_id, session_id)
            window = self.window(user_id, session_id)
            newest = items[0]["value"] if items else None
            return {"profile_facts": [fact for (owner, *_), fact in self.facts.items() if owner == user_id], "temporary_memories": items, "session": ({"description": newest} if newest is not None and not isinstance(newest, dict) else newest or {}), "recent_messages": [{"role": item["role"], "content": item["content"]} for item in window], "message_refs": [item["message_id"] for item in window]}
        if path == "/v1/interactions:process":
            event_id = payload["event_id"]
            if event_id in self.events:
                return {**self.events[event_id], "status": "duplicate"}
            text, user_id, session_id = payload["text"], payload["user_id"], payload["session_id"]
            # The repeated coffee input intentionally has the same expected type.
            index = scenarios.SCENARIO_TEXTS.index(text)
            memory_type = scenarios.EXPECTED_TYPES[index]
            status = "ignored" if memory_type == "discard" else "auto_applied"
            expiry = None if index == 9 or memory_type in {"long_term", "discard"} else (self.clock.now() + timedelta(hours=24)).isoformat()
            decision = {"candidate_id": f"candidate-{len(self.calls)}", "memory_type": memory_type, "status": status, "analyzer_source": self.source, "sensitivity": "health" if memory_type == "sensitive" else "normal", "expires_at": expiry}
            if self.source == "rules_guard":
                decision["analysis"] = {"upstream_analyzer_source": "ollama"}
            elif memory_type == "sensitive":
                decision["analysis"] = {"verification_status": "user_asserted"}
            if memory_type == "long_term":
                key = "coffee" if "kahve" in text else "tennis" if "tenis" in text else "address_preference"
                self.request("POST", "/v1/memories", {"user_id": user_id, "category": "routine", "key": key, "value": text})
            if memory_type == "sensitive" and status == "auto_applied" and expiry is None:
                self.request("POST", "/v1/memories", {
                    "user_id": user_id, "category": "medication", "key": f"health-{index}",
                    "value": text, "sensitivity": "health",
                    "verification_status": "user_asserted",
                })
            if memory_type == "sensitive" and status == "auto_applied" and expiry is not None:
                self.request("POST", "/v1/temporary-memories", {
                    "user_id": user_id, "session_id": session_id,
                    "category": "symptom", "key": f"health-{index}", "value": text,
                    "sensitivity": "health", "verification_status": "user_asserted",
                    "expires_at": expiry, "occurred_at": payload["occurred_at"],
                })
            if memory_type == "short_term":
                self.request("POST", "/v1/temporary-memories", {
                    "user_id": user_id, "session_id": session_id,
                    "category": "session", "key": f"intent-{index}", "value": text,
                    "expires_at": expiry, "occurred_at": payload["occurred_at"],
                })
            self.request("POST", f"/v1/sessions/{session_id}/messages", {"message_id": event_id, "user_id": user_id, "role": "user", "content": text, "occurred_at": payload["occurred_at"]})
            result = {"status": "processed", "event_id": event_id, "decisions": [decision]}
            self.events[event_id] = result
            return result
        raise AssertionError(f"Unexpected fake HTTP request {method} {path}")


class MemoryScenarioRunnerTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.client = FakeMemoryHttp(self.clock)
        self.lines = []
        self.runner = scenarios.ScenarioRunner(self.client, clock=self.clock.now, monotonic=self.clock.monotonic, sleep=self.clock.sleep, output=self.lines.append)

    def test_default_storage_checks_pass_without_extraction_or_reply_model(self):
        self.runner.run_storage()
        report = self.runner.report()
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(len(report["checks"]), 7)
        self.assertIsNone(report["analyzer_ids"])
        self.assertFalse(any(path == "/v1/interactions:process" or "/api/chat" in path for _, path, _ in self.client.calls))
        self.assertEqual(len(self.client.messages), 15)
        self.assertEqual(len(self.runner.message_payloads), 15)
        self.assertTrue(all(duration <= 0.25 for duration in self.clock.sleeps))
        self.assertGreaterEqual(self.clock.elapsed, 3)

    def test_run_ids_are_fresh_and_not_user_configurable(self):
        other = scenarios.ScenarioRunner(self.client)
        self.assertNotEqual(self.runner.user_id, other.user_id)
        self.assertNotEqual(self.runner.session_id, other.session_id)
        args = scenarios.build_parser().parse_args([])
        self.assertFalse(args.with_analyzer)
        self.assertEqual(args.ttl_seconds, 3)
        self.assertFalse(hasattr(args, "user_id"))

    def test_expiry_failure_is_bounded_and_does_not_abort_following_checks(self):
        self.client.sticky_ttl = True
        self.runner.run_storage()
        failures = [item for item in self.runner.results if not item["passed"]]
        self.assertEqual([item["name"] for item in failures], ["independent temporary-item expiry"])
        self.assertIn("10-second", failures[0]["error"])
        self.assertLessEqual(self.clock.elapsed, 10.25)
        self.assertEqual(len(self.runner.results), 7)

    def test_optional_analyzer_runs_exact_original_inputs_separately(self):
        self.runner.run_analyzer()
        self.assertTrue(self.runner.report()["passed"], self.runner.results)
        ingests = [payload for _, path, payload in self.client.calls if path == "/v1/interactions:process"]
        self.assertEqual([payload["text"] for payload in ingests[:15]], list(scenarios.SCENARIO_TEXTS))
        self.assertEqual(len(ingests), 16)  # Last event is replayed, not re-created.
        self.assertEqual(ingests[-1]["event_id"], ingests[14]["event_id"])
        self.assertNotEqual(ingests[0]["user_id"], self.runner.user_id)
        self.assertEqual(len({payload["event_id"] for payload in ingests[:15]}), 15)

    def test_rules_fallback_is_an_explicit_analyzer_failure(self):
        self.client.source = "rules_fallback"
        self.runner.run_analyzer()
        self.assertFalse(self.runner.report()["passed"])
        routing_results = self.runner.results[:15]
        self.assertTrue(all(not result["passed"] for result in routing_results))
        self.assertTrue(all("not model-backed" in result["error"] for result in routing_results))
        self.assertTrue(all(result["detail"]["decisions"][0]["analyzer_source"] == "rules_fallback" for result in routing_results))

    def test_model_backed_rules_guard_passes_and_is_reported_separately(self):
        self.client.source = "rules_guard"
        self.runner.run_analyzer()
        report = self.runner.report()
        self.assertTrue(report["passed"], report["checks"])
        self.assertEqual(report["analyzer_summary"]["guard_intervention_count"], 15)
        self.assertEqual(report["analyzer_summary"]["pure_ollama_count"], 0)
        self.assertEqual(report["analyzer_summary"]["fallback_count"], 0)

    def test_expected_http_status_must_be_exact(self):
        with patch.object(self.client, "request", side_effect=scenarios.HttpError("GET", "/test", 409)):
            with self.assertRaisesRegex(scenarios.ScenarioError, "Expected HTTP 403, got HTTP 409"):
                self.runner.expect_status("GET", "/test", None, 403)
        with patch.object(self.client, "request", return_value={}):
            with self.assertRaisesRegex(scenarios.ScenarioError, "unexpectedly succeeded"):
                self.runner.expect_status("GET", "/test", None, 403)

    def test_cli_saves_json_report_without_direct_database_access(self):
        with patch.object(scenarios, "JsonHttpClient", return_value=self.client), patch.object(scenarios, "ScenarioRunner", return_value=self.runner), patch.object(scenarios.Path, "mkdir") as mkdir, patch.object(scenarios.Path, "write_text") as write_text, patch("sys.stdout", new=io.StringIO()):
            exit_code = scenarios.main(["--json-report", "evals/mock-memory-report.json"])
        self.assertEqual(exit_code, 0)
        mkdir.assert_called_once_with(parents=True, exist_ok=True)
        report = json.loads(write_text.call_args.args[0])
        self.assertTrue(report["passed"])
        self.assertEqual(report["user_id"], self.runner.user_id)
        self.assertEqual(len(report["checks"]), 7)
        self.assertEqual(write_text.call_args.kwargs["encoding"], "utf-8")

    def test_small_and_invalid_ttl_options(self):
        for ttl in (0, -1, 0.2, 0.9, 8.1, float("nan")):
            with self.assertRaises(ValueError):
                scenarios.ScenarioRunner(self.client, ttl)
        runner = scenarios.ScenarioRunner(self.client, 1, clock=self.clock.now, monotonic=self.clock.monotonic, sleep=self.clock.sleep, output=lambda _: None)
        runner.multiple_temporary_items()
        runner.independent_expiry()
        self.assertLessEqual(self.clock.elapsed, 1.25)


class JsonHttpClientTest(unittest.TestCase):
    def test_client_sends_utf8_json_over_http(self):
        response = io.BytesIO(b'{"status":"ok"}')
        with patch.object(scenarios.request, "urlopen", return_value=response) as urlopen:
            result = scenarios.JsonHttpClient("http://127.0.0.1:8001/", 12).request("POST", "/v1/memories", {"value": "Türk kahvesi"})
        self.assertEqual(result, {"status": "ok"})
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:8001/v1/memories")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(json.loads(req.data), {"value": "Türk kahvesi"})
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 12)

    def test_per_request_timeout_cannot_exceed_global_timeout(self):
        for requested, expected in ((3, 3), (30, 12)):
            with self.subTest(requested=requested), patch.object(
                scenarios.request, "urlopen", return_value=io.BytesIO(b'{"status":"ok"}')
            ) as urlopen:
                scenarios.JsonHttpClient("http://127.0.0.1:8001", 12).request(
                    "GET", "/healthz", timeout_seconds=requested,
                )
            self.assertEqual(urlopen.call_args.kwargs["timeout"], expected)

    def test_http_error_does_not_echo_server_sensitive_body(self):
        secret = "private-health-payload"
        exc = error.HTTPError("http://fake", 403, "Forbidden", {}, io.BytesIO(secret.encode()))
        with patch.object(scenarios.request, "urlopen", side_effect=exc):
            with self.assertRaises(scenarios.HttpError) as caught:
                scenarios.JsonHttpClient("http://fake").request("GET", "/v1/private")
        self.assertEqual(caught.exception.status_code, 403)
        self.assertNotIn(secret, str(caught.exception))

    def test_invalid_json_and_non_object_fail(self):
        for body in (b"not json", b"[]", b"null"):
            with self.subTest(body=body), patch.object(scenarios.request, "urlopen", return_value=io.BytesIO(body)):
                with self.assertRaises(scenarios.ScenarioError):
                    scenarios.JsonHttpClient("http://fake").request("GET", "/test")

    def test_bad_configuration_fails_before_http_access(self):
        for url in ("file:///tmp/db", "http://name:pass@localhost", "http://localhost?secret=x", "http://localhost#x"):
            with self.assertRaises(ValueError):
                scenarios.JsonHttpClient(url)
        with patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(scenarios.main(["--ttl-seconds", "0"]), 2)


if __name__ == "__main__":
    unittest.main()
