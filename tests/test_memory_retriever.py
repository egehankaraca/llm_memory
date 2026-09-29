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
                value="Çayımı açık ve şekersiz içerim",
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

    def test_turkish_alias_resolves_form_of_address_without_pinning(self) -> None:
        result = select_profile_memories(
            self.memories,
            query="Bana nasıl seslenmelisin?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories=set(),
            now=self.now,
        )

        self.assertEqual(
            [selected.memory.id for selected in result.selected],
            ["address"],
        )
        self.assertTrue(result.alias_used)
        self.assertIn("concept_alias", result.selected[0].reasons)

    def test_turkish_alias_bridges_sweetener_and_sugarless_value(self) -> None:
        result = select_profile_memories(
            self.memories,
            query="Tatlandırıcı kullanır mıydım?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories=set(),
            now=self.now,
        )

        self.assertEqual(
            [selected.memory.id for selected in result.selected],
            ["tea"],
        )
        self.assertIn("concept_alias", result.selected[0].reasons)

    def test_food_suggestion_intent_retrieves_stored_food_preference(self) -> None:
        food = ProfileMemory(
            id="pasta",
            category="food_preference",
            key="preferred_food",
            value="Ben makarna yemeyi çok severim.",
            confidence=0.95,
            verification_status="unverified",
            updated_at=self.now,
        )
        tennis = ProfileMemory(
            id="tennis",
            category="routine",
            key="tennis_schedule",
            value="Her cumartesi sabah 7'de tenis oynarım.",
            confidence=0.95,
            verification_status="unverified",
            updated_at=self.now,
        )

        for query in (
            "Bugün ne yesem?",
            "Akşam ne yiyebilirim?",
            "Acıktım, bana bir şey önerir misin?",
        ):
            with self.subTest(query=query):
                result = select_profile_memories(
                    [food, tennis],
                    query=query,
                    max_facts=20,
                    max_tokens=1_500,
                    pinned_categories=set(),
                    semantic_scores={"pasta": 0.30, "tennis": 0.20},
                    semantic_min_similarity=0.40,
                    now=self.now,
                )

                self.assertEqual(
                    [selected.memory.id for selected in result.selected],
                    ["pasta"],
                )
                self.assertIn("concept_alias", result.selected[0].reasons)

    def test_semantic_score_can_recover_a_lexically_unrelated_fact(self) -> None:
        result = select_profile_memories(
            self.memories,
            query="Bunu sıcak mı tüketiyordum?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories=set(),
            semantic_scores={"tea": 0.91, "wake": 0.2, "address": 0.1},
            semantic_min_similarity=0.55,
            now=self.now,
        )

        self.assertEqual(
            [selected.memory.id for selected in result.selected],
            ["tea"],
        )
        self.assertTrue(result.semantic_used)
        self.assertEqual(result.semantic_candidate_count, 1)
        self.assertIn("semantic_similarity", result.selected[0].reasons)

    def test_unrelated_query_excludes_protected_domains_and_weak_time_overlap(self) -> None:
        memories = [
            *self.memories,
            ProfileMemory(
                id="medication",
                category="medication",
                key="medication_time",
                value="Tansiyon ilacımı her sabah saat 8'de alırım.",
                confidence=1.0,
                verification_status="user_asserted",
                updated_at=self.now,
            ),
            ProfileMemory(
                id="emergency",
                category="emergency_contact",
                key="primary_emergency_contact",
                value="Acil durumda kızım Ayşe'yi ara.",
                confidence=1.0,
                verification_status="user_asserted",
                updated_at=self.now,
            ),
        ]

        result = select_profile_memories(
            memories,
            query="Sabah içecek alışkanlığım nedir?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories={"communication", "emergency_contact"},
            semantic_scores={"tea": 0.82, "medication": 0.60, "emergency": 0.55},
            now=self.now,
        )

        self.assertEqual(
            [selected.memory.id for selected in result.selected],
            ["tea", "address"],
        )

    def test_protected_domains_are_retrieved_for_matching_queries(self) -> None:
        medication = ProfileMemory(
            id="medication",
            category="medication",
            key="medication_time",
            value="Tansiyon ilacımı her sabah saat 8'de alırım.",
            confidence=1.0,
            verification_status="user_asserted",
            updated_at=self.now,
        )
        emergency = ProfileMemory(
            id="emergency",
            category="emergency_contact",
            key="primary_emergency_contact",
            value="Acil durumda kızım Ayşe'yi ara.",
            confidence=1.0,
            verification_status="user_asserted",
            updated_at=self.now,
        )

        health = select_profile_memories(
            [medication, emergency],
            query="Tansiyon ilacımı kaçta alıyorum?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories=set(),
            now=self.now,
        )
        crisis = select_profile_memories(
            [*self.memories, medication, emergency],
            query="Düştüm, her yer kan içinde!",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories=set(),
            now=self.now,
        )
        head_impact = select_profile_memories(
            [*self.memories, medication, emergency],
            query="Banyoda kaydım, kafamı vurdum, ne yapacağım?",
            max_facts=20,
            max_tokens=1_500,
            pinned_categories=set(),
            now=self.now,
        )

        self.assertEqual([item.memory.id for item in health.selected], ["medication"])
        self.assertEqual([item.memory.id for item in crisis.selected], ["emergency"])
        self.assertEqual([item.memory.id for item in head_impact.selected], ["emergency"])

    def test_episode_mode_recalls_health_event_from_temporal_query(self) -> None:
        episode = ProfileMemory(
            id="fall",
            category="health_event",
            key="fall_incident",
            value="Dün banyoda düştüm.",
            confidence=1.0,
            verification_status="user_asserted",
            updated_at=self.now,
        )

        result = select_profile_memories(
            [episode],
            query="Dün ne olmuştu?",
            max_facts=10,
            max_tokens=1_000,
            pinned_categories=set(),
            enforce_protected_query_gate=False,
            now=self.now,
        )

        self.assertEqual(
            [selected.memory.id for selected in result.selected],
            ["fall"],
        )

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
