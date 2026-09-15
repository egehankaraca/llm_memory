import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from conversation_orchestrator import OrchestratorSettings
from scripts import run_chat_regressions


class ChatRegressionRunnerTest(unittest.TestCase):
    def coordinators(self):
        seed, chat = Mock(), Mock()
        seed.check_services.return_value = {"provider": "ollama", "available": True}
        seed.chat.return_value.decisions = [{"status": "auto_applied", "memory_type": "long_term", "analyzer_source": "ollama"}]
        seed.memories.return_value = {"items": [
            {"id": "fact-1", "value": "weekly tennis"}, {"id": "fact-2", "value": "Ahmet Bey"},
        ]}
        chat.memories.return_value = seed.memories.return_value
        chat.context.return_value = {"session": {}}
        chat.chat.side_effect = [
            SimpleNamespace(
                reply=reply, decisions=[{"memory_type": "discard", "analyzer_source": "ollama"}],
                prompt=SimpleNamespace(history_message_count=0 if index == 0 else index * 2),
                model_stats={},
            )
            for index, reply in enumerate([
                "Cumartesi saat 7 bilgisi var. Her gün mü?",
                "Tenis oynamak konusunda neyi değerlendirmek istersiniz?",
                "Tenis hakkında neyi öğrenmek istersiniz?",
                "Cumartesi saat 7'de tenis oynarsınız.",
                "İki artı iki 4 eder.",
            ])
        ]
        return seed, chat

    def test_runner_checks_five_turns_in_new_session_without_claiming_semantic_score(self):
        seed, chat = self.coordinators()
        with patch.object(run_chat_regressions, "ConversationOrchestrator", side_effect=[seed, chat]) as factory, patch("sys.stdout", new=io.StringIO()):
            results = run_chat_regressions.run(OrchestratorSettings())
        self.assertEqual(len(results), 5)
        self.assertEqual(chat.memories.call_count, 5)
        self.assertTrue(all(result["needs_human_review"] for result in results))
        self.assertTrue(all("reply" in result for result in results))
        self.assertTrue(all("raw_model_answer" not in result for result in results))
        self.assertEqual(factory.call_args_list[0].args[1], factory.call_args_list[1].args[1])
        self.assertNotEqual(factory.call_args_list[0].args[2], factory.call_args_list[1].args[2])

    def test_runner_fails_when_question_creates_an_extra_fact(self):
        seed, chat = self.coordinators()
        chat.memories.return_value = {"items": seed.memories.return_value["items"] + [{"id": "bad-fact", "value": "daily tennis"}]}
        with patch.object(run_chat_regressions, "ConversationOrchestrator", side_effect=[seed, chat]), patch("sys.stdout", new=io.StringIO()), self.assertRaisesRegex(AssertionError, "profile changed"):
            run_chat_regressions.run(OrchestratorSettings())

    def test_runner_cannot_claim_model_pass_when_analyzer_falls_back(self):
        seed, chat = self.coordinators()
        seed.chat.return_value.decisions[0]["analyzer_source"] = "rules_fallback"
        with patch.object(run_chat_regressions, "ConversationOrchestrator", side_effect=[seed, chat]), patch("sys.stdout", new=io.StringIO()), self.assertRaisesRegex(AssertionError, "fallback"):
            run_chat_regressions.run(OrchestratorSettings())


if __name__ == "__main__":
    unittest.main()
