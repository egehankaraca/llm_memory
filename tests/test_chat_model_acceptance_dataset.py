"""Validate review fixtures, not the semantic quality of any model's answers."""

from pathlib import Path
import unittest

from conversation_orchestrator import OrchestratorSettings, build_chat_prompt
from scripts.run_conversation_baseline import load_scenarios


DATASET = Path(__file__).resolve().parents[1] / "evals" / "chat_model_acceptance.jsonl"


class ChatModelAcceptanceDatasetTest(unittest.TestCase):
    def test_bounded_independent_scenarios_have_human_review_expectations(self):
        scenarios = load_scenarios(DATASET)
        self.assertEqual(len(scenarios), 10)
        self.assertEqual(sum(len(scenario["turns"]) for scenario in scenarios), 14)
        self.assertEqual(len({scenario["id"] for scenario in scenarios}), 10)
        for scenario in scenarios:
            with self.subTest(scenario=scenario["id"]):
                self.assertNotIn("recent_messages", scenario["context"])
                for turn in scenario["turns"]:
                    self.assertTrue(turn["expectation"].startswith("İnsan değerlendirmesi:"))

    def test_owned_attributed_fixtures_fit_without_losing_facts(self):
        settings = OrchestratorSettings()
        for scenario in load_scenarios(DATASET):
            context = scenario["context"]
            with self.subTest(scenario=scenario["id"]):
                for fact in context["profile_facts"]:
                    self.assertEqual(fact["owner_user_id"], context["user_id"])
                    self.assertEqual(set(fact["provenance"]), {
                        "source_event_id", "verification_status", "confidence",
                    })
                for turn in scenario["turns"]:
                    prompt = build_chat_prompt(context, turn["text"], settings,
                                               user_id=context["user_id"])
                    self.assertFalse(prompt.trimmed)
                    self.assertEqual(prompt.profile_fact_count, len(context["profile_facts"]))
                    self.assertEqual(prompt.messages[-1], {"role": "user", "content": turn["text"]})
                    self.assertEqual(prompt.history_message_count, 0)


if __name__ == "__main__":
    unittest.main()
