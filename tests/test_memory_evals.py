import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import models
from memory_analyzer import MemoryDecision
from scripts import run_memory_evals


class MemoryEvalRunnerTest(unittest.TestCase):
    def test_regression_dataset_is_valid_and_kept_separate(self) -> None:
        cases = run_memory_evals.load_dataset(Path(__file__).parents[1] / "evals" / "memory_regressions.jsonl")
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case.id for case in cases}), 12)
        self.assertTrue(all("regression" in case.tags for case in cases))

    def test_curated_dataset_is_valid_and_has_fifty_unique_cases(self) -> None:
        dataset = Path(__file__).parents[1] / "evals" / "memory_cases.jsonl"
        cases = run_memory_evals.load_dataset(dataset)

        self.assertEqual(len(cases), 50)
        self.assertEqual(len({case.id for case in cases}), 50)
        expected_types = Counter(
            decision.memory_type
            for case in cases
            for decision in case.expected.decisions
        )
        self.assertGreaterEqual(expected_types[models.CandidateMemoryType.SENSITIVE], 13)
        self.assertGreaterEqual(expected_types[models.CandidateMemoryType.DISCARD], 10)

    def test_comparison_ignores_decision_order(self) -> None:
        case = run_memory_evals.EvalCase.model_validate(
            {
                "id": "multi",
                "input": {"text": "test", "recent_messages": []},
                "expected": {
                    "decisions": [
                        {
                            "memory_type": "long_term",
                            "sensitivity": "normal",

                            "destination": "profile",
                        },
                        {
                            "memory_type": "sensitive",
                            "sensitivity": "health",

                            "destination": "profile",
                        },
                    ]
                },
                "tags": ["multiple_decisions"],
            }
        )
        actual = [
            MemoryDecision(
                memory_type=models.CandidateMemoryType.SENSITIVE,
                category="health",
                key="symptom",
                value="test",
                sensitivity=models.Sensitivity.HEALTH,
                confidence=0.9,

                expires_at=None,
                reason="test",
                analyzer_source="ollama",
            ),
            MemoryDecision(
                memory_type=models.CandidateMemoryType.LONG_TERM,
                category="preferences",
                key="music",
                value="test",
                sensitivity=models.Sensitivity.NORMAL,
                confidence=0.9,

                expires_at=None,
                reason="test",
                analyzer_source="ollama",
            ),
        ]

        passed, _, _ = run_memory_evals.compare_decisions(
            case.expected.decisions,
            actual,
        )
        self.assertTrue(passed)

    def test_ollama_fallback_never_counts_as_model_pass(self) -> None:
        case = run_memory_evals.EvalCase.model_validate(
            {
                "id": "fallback",
                "input": {"text": "Bugün dışarı çıkacağım"},
                "expected": {
                    "decisions": [
                        {
                            "memory_type": "short_term",
                            "sensitivity": "normal",

                            "destination": "session",
                        }
                    ]
                },
                "tags": [],
            }
        )
        fallback = MemoryDecision(
            memory_type=models.CandidateMemoryType.SHORT_TERM,
            category="session",
            key="active_goal",
            value={"type": "outing"},
            sensitivity=models.Sensitivity.NORMAL,
            confidence=0.8,

            expires_at=None,
            reason="fallback",
            analyzer_source="rules_fallback",
        )
        with (
            patch.dict("os.environ", {"MEMORY_ANALYZER_PROVIDER": "ollama"}),
            patch.object(run_memory_evals, "analyze_message", return_value=[fallback]),
        ):
            result = run_memory_evals.evaluate_case(
                case,
                datetime.now(timezone.utc),
            )

        self.assertFalse(result["passed"])
        self.assertTrue(result["used_fallback"])


if __name__ == "__main__":
    unittest.main()
