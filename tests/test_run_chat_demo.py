import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_chat_demo import print_turn


class ChatDemoOutputTest(unittest.TestCase):
    def turn(self, *, done_reason="stop"):
        return SimpleNamespace(
            reply="Anladım, Ahmet Bey.",
            event_id="event-1",
            assistant_message_id="assistant-1",
            prompt=SimpleNamespace(
                profile_fact_count=1,
                history_message_count=6,
                estimated_tokens=651,
                input_budget=3328,
                trimmed=False,
            ),
            model_stats={"done_reason": done_reason, "response_mode": "native_ollama_chat"},
            decisions=[],
            pending_candidates=[],
        )

    def test_debug_prints_answer_once_without_redundant_answer_fields(self):
        turn = self.turn()
        with patch("sys.stdout", new=io.StringIO()) as output:
            print_turn(turn, debug=True)

        rendered = output.getvalue()
        self.assertEqual(rendered.count(turn.reply), 1)
        debug = json.loads(rendered[rendered.index("{"):])
        self.assertNotIn("raw_model_answer", debug)
        self.assertNotIn("displayed_answer", debug)
        self.assertEqual(debug["event_id"], turn.event_id)
        self.assertEqual(debug["history_message_count"], 6)
        self.assertEqual(debug["model_stats"], turn.model_stats)
        self.assertEqual(debug["decisions"], [])

    def test_normal_output_only_prints_answer(self):
        with patch("sys.stdout", new=io.StringIO()) as output:
            print_turn(self.turn(), debug=False)

        self.assertEqual(output.getvalue(), "\nAsistan: Anladım, Ahmet Bey.\n")

    def test_token_limit_warning_is_preserved(self):
        with patch("sys.stdout", new=io.StringIO()) as output:
            print_turn(self.turn(done_reason="length"), debug=False)

        self.assertIn("Cevap token limitinde durdu", output.getvalue())


if __name__ == "__main__":
    unittest.main()
