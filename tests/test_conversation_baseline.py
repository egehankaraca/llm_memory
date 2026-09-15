import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from conversation_orchestrator import JsonHttpClient, OrchestratorError, OrchestratorSettings
from scripts.run_conversation_baseline import DEFAULT_DATASET, load_scenarios, main, run_baseline


def completion(answer="Size nasıl yardımcı olabilirim?"):
    return {"message": {"role": "assistant", "content": answer},
            "done": True, "prompt_eval_count": 200, "eval_count": 15, "done_reason": "stop"}


def scenario():
    return {"id": "review", "context": {"profile": {}, "recent_messages": []},
            "turns": [{"text": "Tenis", "expectation": "Ask what the user wants."},
                      {"text": "Kuralları nedir?", "expectation": "Explain tennis rules."}]}


class ConversationBaselineTest(unittest.TestCase):
    def setUp(self):
        self.http = Mock(spec=JsonHttpClient)
        self.http.request.return_value = completion()
        self.settings = OrchestratorSettings(ollama_timeout_seconds=60)

    def test_only_ollama_requests_and_history_is_preserved(self):
        initial = scenario()
        self.http.request.side_effect = [completion("Tenisin hangi yönünü soruyorsunuz?"), completion("Kısaca kuralları...")]
        report = run_baseline([initial], self.settings, ollama_http=self.http)
        self.assertEqual([(call.args[0], call.args[1]) for call in self.http.request.call_args_list],
                         [("POST", "/api/chat"), ("POST", "/api/chat")])
        second_messages = self.http.request.call_args_list[1].args[2]["messages"]
        self.assertNotIn("format", self.http.request.call_args.args[2])
        self.assertEqual(second_messages[1:], [
            {"role": "user", "content": "Tenis"},
            {"role": "assistant", "content": "Tenisin hangi yönünü soruyorsunuz?"},
            {"role": "user", "content": "Kuralları nedir?"},
        ])
        self.assertEqual(initial["context"]["recent_messages"], [])
        self.assertEqual(report["memory_requests"], 0)
        self.assertEqual(report["database_writes"], 0)
        self.assertIsNone(report["semantic_score"])
        self.assertEqual(report["summary"]["needs_human_review"], 2)
        self.assertEqual(report["summary"]["valid"], 2)
        self.assertEqual(report["results"][1]["history_before"], second_messages[1:-1])
        self.assertEqual(report["results"][1]["history_sent"], second_messages[1:-1])
        self.assertEqual(report["results"][0]["history_sent"], [])
        self.assertEqual(report["results"][1]["prompt_stats"]["history_message_count"], 2)

    def test_current_preset_uses_demo_generation_options_and_only_real_history(self):
        report = run_baseline([scenario()], self.settings, ollama_http=self.http)
        self.assertEqual(report["effective_options"], self.settings.generation_options)
        self.assertEqual(self.http.request.call_args.args[2]["options"], self.settings.generation_options)
        self.assertEqual(report["results"][0]["prompt_stats"]["history_message_count"], 0)
        self.assertEqual(report["results"][1]["prompt_stats"]["history_message_count"], 2)
        for result in report["results"]:
            self.assertNotIn("style_message_count", result["prompt_stats"])
            for message in result["history_before"] + result["history_sent"]:
                self.assertNotIn("ÜSLUP ÖRNEKLERİ", message["content"])

    def test_report_has_one_answer_field_and_model_stats(self):
        self.http.request.return_value = completion("  Ahmet Bey, tenis hakkında ne soracaksınız?  ")
        report = run_baseline([scenario()], self.settings, ollama_http=self.http)
        entry = report["results"][0]
        self.assertEqual(entry["answer"], "Ahmet Bey, tenis hakkında ne soracaksınız?")
        self.assertEqual(entry["model_stats"]["eval_count"], 15)
        for removed in ("raw_model_content", "raw_model_answer", "displayed_answer", "answer_rewritten"):
            self.assertNotIn(removed, entry)
        self.assertNotIn("rewritten", report["summary"])

    def test_native_text_is_not_a_semantic_pass(self):
        self.http.request.return_value = completion("Alakasız cevap")
        report = run_baseline([scenario()], self.settings, ollama_http=self.http)
        entry = report["results"][0]
        self.assertEqual(entry["answer"], "Alakasız cevap")
        self.assertEqual(entry["outcome"], "valid")
        self.assertTrue(entry["needs_human_review"])
        self.assertIsNone(report["semantic_score"])

    def test_invalid_native_text_is_visible_and_not_used_as_history(self):
        for content in ["", "  ", "x" * 10_001]:
            with self.subTest(content=content[:30]):
                self.http.request.reset_mock()
                self.http.request.return_value = completion(content)
                report = run_baseline([scenario()], self.settings, ollama_http=self.http)
                entry = report["results"][0]
                self.assertIsNone(entry["answer"])
                self.assertEqual(entry["outcome"], "error")
                self.assertIsNotNone(entry["error"])
                self.assertEqual(report["results"][1]["outcome"], "skipped_previous_error")
                self.assertIsNone(report["results"][1]["answer"])
                self.assertEqual(self.http.request.call_count, 1)

    def test_json_looking_native_text_is_not_decoded_or_rewritten(self):
        text = '{"answer":"Cevap","extra":"native text"}'
        self.http.request.return_value = completion(text)
        report = run_baseline([scenario()], self.settings, ollama_http=self.http)
        entry = report["results"][0]
        self.assertEqual(entry["answer"], text)
        self.assertEqual(entry["outcome"], "valid")

    def test_transport_error_does_not_become_memory_not_found(self):
        self.http.request.side_effect = OrchestratorError("POST /api/chat: timeout")
        report = run_baseline([scenario()], self.settings, ollama_http=self.http)
        self.assertIsNone(report["results"][0]["answer"])
        self.assertEqual(report["summary"]["errors"], 1)
        self.assertEqual(report["summary"]["requests"], 1)

    def test_diagnostic_preset_options_are_explicit_without_extra_requests(self):
        report = run_baseline([scenario()], self.settings, ollama_http=self.http, preset="qwen3-non-thinking")
        expected = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0,
                    "num_ctx": self.settings.num_ctx, "num_predict": self.settings.num_predict}
        self.assertEqual(report["preset"], "qwen3-non-thinking")
        self.assertEqual(report["effective_options"], expected)
        self.assertEqual(self.http.request.call_count, 2)
        self.assertEqual(self.http.request.call_args.args[2]["options"], expected)
        with self.assertRaises(OrchestratorError):
            run_baseline([scenario()], self.settings, ollama_http=self.http, preset="unknown")
        self.assertEqual(self.http.request.call_count, 2)

    def test_non_string_content_is_reported_as_error(self):
        for content in [None, [], {"answer": "Cevap"}, 7]:
            with self.subTest(content=content):
                self.http.request.return_value = completion(content)
                report = run_baseline([scenario()], self.settings, ollama_http=self.http)
                self.assertEqual(report["summary"]["errors"], 1)
                self.assertIsNone(report["results"][0]["answer"])

    def test_dataset_filters_and_unknown_ids(self):
        all_scenarios = load_scenarios(DEFAULT_DATASET)
        self.assertEqual(len(all_scenarios), 10)
        self.assertEqual(sum(len(item["turns"]) for item in all_scenarios), 21)
        filtered = load_scenarios(DEFAULT_DATASET, ["question_presupposition", "tennis_advice_recall"])
        self.assertEqual([item["id"] for item in filtered], ["tennis_advice_recall", "question_presupposition"])
        with self.assertRaises(OrchestratorError):
            load_scenarios(DEFAULT_DATASET, ["unknown"])

    def test_cli_uses_only_ollama_and_writes_review_report(self):
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.run_conversation_baseline.JsonHttpClient", return_value=self.http) as constructor, patch("builtins.print"):
            path = Path(temporary) / "report.json"
            result = main(["--scenario", "topic_followup", "--json-report", str(path), "--model", "qwen3:8b"])
            report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result, 0)
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(constructor.call_args.args, (self.settings.ollama_url, 60))
        self.assertEqual(report["summary"]["planned_turns"], 2)
        self.assertIsNone(report["semantic_score"])
        self.assertTrue(all(entry["needs_human_review"] for entry in report["results"]))


if __name__ == "__main__":
    unittest.main()
