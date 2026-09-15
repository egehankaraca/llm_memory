import json
import unittest

from conversation_orchestrator import OrchestratorError, OrchestratorSettings, build_chat_prompt


def payload(count=2, value_size=20):
    return {
        "user_id": "owner", "session_id": "session", "as_of": "2026-09-14T09:00:00+00:00",
        "profile_facts": [], "recent_messages": [], "session": {"description": "Legacy duplicate"},
        "temporary_memories": [{
            "memory_id": str(index), "owner_user_id": "owner", "session_id": "session",
            "category": "session", "key": f"goal_{index}", "value": "x" * value_size,
            "occurred_at": "2026-09-14T08:50:00+00:00", "expires_at": "2026-09-14T10:00:00+00:00",
            "sensitivity": "normal", "provenance": {
                "source_event_id": f"event-{index}", "confidence": 1,
                "verification_status": "unverified", "source_quote": "Do not expand this",
            }, "debug": "Do not forward this",
        } for index in range(count)],
    }


class TemporaryContextConsumerTest(unittest.TestCase):
    def test_all_items_reach_consumer_with_attribution_without_legacy_duplicates(self):
        prompt = build_chat_prompt(payload(), "Ne planladım?", OrchestratorSettings(),
                                   user_id="owner", session_id="session")
        data = json.loads(prompt.messages[0]["content"].split("\nMEMORY_CONTEXT_JSON:\n", 1)[1])
        self.assertEqual(len(data["temporary_memories"]), 2)
        self.assertEqual(data["session"], {})
        self.assertNotIn("source_quote", data["temporary_memories"][0]["provenance"])
        self.assertNotIn("debug", data["temporary_memories"][0])
        self.assertEqual(data["temporary_memories"][0]["owner_user_id"], "owner")

    def test_foreign_owner_foreign_session_and_invalid_expiry_fail_closed(self):
        for key, value in [("owner_user_id", "other"), ("session_id", "other-session"),
                           ("expires_at", "2026-09-14T08:00:00+00:00"),
                           ("expires_at", "2026-09-14T10:00:00"), ("expires_at", "nonsense")]:
            with self.subTest(key=key, value=value):
                context = payload()
                context["temporary_memories"][0][key] = value
                with self.assertRaises(OrchestratorError):
                    build_chat_prompt(context, "Merhaba", OrchestratorSettings(),
                                      user_id="owner", session_id="session")

    def test_malformed_temporary_provenance_and_time_fail_closed(self):
        cases = [
            ("provenance", {}),
            ("provenance", {"source_event_id": None, "verification_status": "invented", "confidence": 1}),
            ("provenance", {"source_event_id": None, "verification_status": "unverified", "confidence": "1"}),
            ("sensitivity", "invented"), ("occurred_at", "2026-09-14T08:50:00"),
            ("occurred_at", "2026-09-14T10:30:00+00:00"),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                context = payload()
                context["temporary_memories"][0][key] = value
                with self.assertRaises(OrchestratorError):
                    build_chat_prompt(context, "Merhaba", OrchestratorSettings(),
                                      user_id="owner", session_id="session")

    def test_input_budget_drops_whole_items_without_truncating_values(self):
        context = payload(count=5, value_size=800)
        prompt = build_chat_prompt(context, "Merhaba", OrchestratorSettings(num_ctx=2048, num_predict=256),
                                   user_id="owner", session_id="session")
        data = json.loads(prompt.messages[0]["content"].split("\nMEMORY_CONTEXT_JSON:\n", 1)[1])
        self.assertTrue(prompt.trimmed)
        self.assertLess(len(data["temporary_memories"]), 5)
        self.assertLessEqual(prompt.estimated_tokens, prompt.input_budget)
        self.assertTrue(all(row["value"] == "x" * 800 for row in data["temporary_memories"]))
        self.assertEqual(len(context["temporary_memories"]), 5)


if __name__ == "__main__":
    unittest.main()
