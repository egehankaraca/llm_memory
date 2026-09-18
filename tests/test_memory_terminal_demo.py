import unittest

from scripts.run_memory_terminal_demo import DemoSettings, MemoryTerminalDemo


class FakeMemoryClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        self.calls.append((method, path, payload))
        if path == "/healthz":
            return {"status": "ok"}
        if path == "/v1/analyzer/status":
            return {"provider": "ollama", "model": "qwen3:8b", "available": True}
        if path == "/v1/interactions:process":
            assert payload is not None
            return {
                "status": "processed",
                "event_id": payload["event_id"],
                "timing": {
                    "total_ms": 125.0,
                    "preparation_ms": 5.0,
                    "analyzer_ms": 110.0,
                    "memory_write_ms": 10.0,
                },
                "decisions": [{
                    "candidate_id": "candidate-1",
                    "event_id": payload["event_id"],
                    "memory_type": "long_term",
                    "category": "routine",
                    "key": "morning_coffee",
                    "value": payload["text"],
                    "sensitivity": "normal",
                    "confidence": 0.95,
                    "analyzer_source": "ollama",
                    "status": "auto_applied",
                    "consolidation_action": "create",
                    "consolidates_fact_id": None,
                    "expires_at": None,
                    "applied_ref": "fact-1",
                }],
            }
        if path == "/v1/interactions:enqueue":
            assert payload is not None
            return {
                "status": "queued",
                "event_id": payload["event_id"],
                "job": {"status": "pending", "attempt_count": 0},
            }
        if path == "/v1/context:build":
            return {
                "user_id": "demo-user",
                "profile_facts": [],
                "temporary_memories": [],
                "recent_messages": [],
                "profile_retrieval": {},
                "timing": {
                    "total_ms": 42.0,
                    "query_embedding_ms": 35.0,
                    "vector_search_ms": 1.5,
                },
            }
        if path.startswith("/v1/users/"):
            return {"items": []}
        if path.startswith("/v1/sessions/"):
            return {"temporary_memories": [], "memory_count": 0}
        raise AssertionError(f"Unexpected request: {method} {path}")


class MemoryTerminalDemoTest(unittest.TestCase):
    def test_async_memory_is_the_default_setting(self) -> None:
        settings = DemoSettings(
            memory_url="http://memory.test",
            timeout_seconds=30,
        )
        self.assertTrue(settings.async_memory)

    def make_demo(self, *, async_memory: bool = False) -> tuple[MemoryTerminalDemo, FakeMemoryClient, list[str]]:
        client = FakeMemoryClient()
        output: list[str] = []
        demo = MemoryTerminalDemo(
            DemoSettings(
                memory_url="http://memory.test",
                timeout_seconds=30,
                async_memory=async_memory,
            ),
            "demo-user",
            "demo-session",
            client=client,
            output=output.append,
        )
        return demo, client, output

    def test_sync_text_calls_only_memory_process_and_prints_decision(self) -> None:
        demo, client, output = self.make_demo()

        result = demo.process_text("Her sabah kahve içerim")

        self.assertEqual(result["decisions"][0]["memory_type"], "long_term")
        self.assertEqual(client.calls[0][0:2], ("POST", "/v1/interactions:process"))
        self.assertFalse(any("/api/chat" in path for _, path, _ in client.calls))
        rendered = "\n".join(output)
        self.assertIn('"destination": "profile (PostgreSQL + pgvector)"', rendered)
        self.assertIn('"analyzer_ms": 110.0', rendered)
        self.assertIn('"memory_write_ms": 10.0', rendered)
        self.assertNotIn("Asistan", rendered)

    def test_async_text_only_enqueues_and_exposes_event_status_command(self) -> None:
        demo, client, output = self.make_demo(async_memory=True)

        response = demo.process_text("Bugün dışarı çıkmak istiyorum")

        self.assertEqual(response["job"]["status"], "pending")
        self.assertEqual(client.calls[0][0:2], ("POST", "/v1/interactions:enqueue"))
        self.assertIn("Check later with: /status", "\n".join(output))

    def test_context_command_passes_query_without_any_reply_model(self) -> None:
        demo, client, output = self.make_demo()

        self.assertTrue(demo.handle_command("/context Sabah ne içerim?"))

        method, path, payload = client.calls[-1]
        self.assertEqual((method, path), ("POST", "/v1/context:build"))
        self.assertEqual(payload, {
            "user_id": "demo-user",
            "session_id": "demo-session",
            "query": "Sabah ne içerim?",
        })
        rendered = "\n".join(output)
        self.assertIn("Retrieval timing", rendered)
        self.assertIn('"query_embedding_ms": 35.0', rendered)
        self.assertIn('"vector_search_ms": 1.5', rendered)

    def test_confirmation_commands_are_removed(self) -> None:
        demo, client, output = self.make_demo()

        demo.handle_command("/confirm candidate-1")

        self.assertEqual(client.calls, [])
        self.assertIn("Unknown command", "\n".join(output))

    def test_episodes_command_lists_episodic_memory_without_reply_model(self) -> None:
        demo, client, _ = self.make_demo()

        demo.handle_command("/episodes")

        self.assertEqual(
            client.calls[-1][0:2],
            ("GET", "/v1/users/demo-user/episodes"),
        )

    def test_service_check_uses_only_memory_health_and_analyzer_status(self) -> None:
        demo, client, _ = self.make_demo()

        analyzer = demo.check_services()

        self.assertTrue(analyzer["available"])
        self.assertEqual([path for _, path, _ in client.calls], [
            "/healthz",
            "/v1/analyzer/status",
        ])


if __name__ == "__main__":
    unittest.main()
