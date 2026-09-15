import unittest
from datetime import datetime, timedelta, timezone

from memory_retriever import (
    ProfileMemory,
    estimate_tokens,
    select_profile_memories,
)


class MemoryRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
        self.memories = [
            ProfileMemory(
                id="wake",
                category="routine",
                key="morning_wake_time",
                value="Sabah saat 7'de kalkarım",
                confidence=0.95,
                verification_status="unverified",
                updated_at=self.now - timedelta(days=2),
            ),
            ProfileMemory(
                id="tea",
                category="preference",
                key="tea_style",
                value="Çayımı açık içerim",
                confidence=0.95,
                verification_status="unverified",
                updated_at=self.now - timedelta(days=1),
            ),
            ProfileMemory(
                id="address",
                category="communication",
                key="form_of_address",
                value="Ahmet Bey",
                confidence=0.99,
                verification_status="user_confirmed",
                updated_at=self.now - timedelta(days=30),
            ),
        ]

    def test_query_selects_relevant_fact_and_excludes_unrelated_fact(self) -> None:
        result = select_profile_memories(
            self.memories,
            query="Bu sabah kaçta kalkıyorum?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories=set(),
            now=self.now,
        )

        self.assertEqual(
            [selected.memory.id for selected in result.selected],
            ["wake"],
        )
        self.assertTrue(result.query_used)
        self.assertEqual(result.candidate_fact_count, 1)

    def test_pinned_fact_survives_unrelated_query(self) -> None:
        result = select_profile_memories(
            self.memories,
            query="Bugün tenis oynayacak mıyım?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories={"communication"},
            now=self.now,
        )

        self.assertEqual(
            [selected.memory.id for selected in result.selected],
            ["address"],
        )
        self.assertIn("pinned_category", result.selected[0].reasons)

    def test_fact_count_limit_is_enforced(self) -> None:
        result = select_profile_memories(
            self.memories,
            query=None,
            max_facts=2,
            max_tokens=1_500,
            pinned_categories=set(),
            now=self.now,
        )

        self.assertEqual(len(result.selected), 2)
        self.assertEqual(result.omitted_fact_count, 1)

    def test_token_budget_skips_oversized_fact(self) -> None:
        oversized = ProfileMemory(
            id="oversized",
            category="notes",
            key="large_note",
            value="x" * 2_000,
            confidence=1.0,
            verification_status="user_confirmed",
            updated_at=self.now,
        )
        token_budget = estimate_tokens({"small": {"fact": "ok"}})
        result = select_profile_memories(
            [oversized, self.memories[1]],
            query=None,
            max_facts=20,
            max_tokens=token_budget,
            pinned_categories=set(),
            now=self.now,
        )

        self.assertNotIn(
            "oversized",
            [selected.memory.id for selected in result.selected],
        )
        self.assertLessEqual(result.estimated_tokens, token_budget)


if __name__ == "__main__":
    unittest.main()
