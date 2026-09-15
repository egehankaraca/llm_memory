import unittest

from memory_consolidator import (
    ActiveMemory,
    ConsolidationAction,
    resolve_consolidation,
)


class MemoryConsolidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wake_memory = ActiveMemory(
            id="wake-fact-001",
            category="routine",
            key="morning_wake_up_time",
            value="Sabah saat 7'de kalkarım",
        )

    def test_new_slot_is_canonicalized_and_created(self) -> None:
        plan = resolve_consolidation(
            category="preferences",
            key="tea_style",
            value="Çayımı açık içerim",
            source_text="Çayımı açık içerim",
            active_memories=[],
        )

        self.assertEqual(plan.action, ConsolidationAction.CREATE)
        self.assertEqual(plan.category, "preference")
        self.assertEqual(plan.key, "tea_style")

    def test_alias_slot_with_same_value_is_unchanged(self) -> None:
        plan = resolve_consolidation(
            category="habit",
            key="wake_up_hour",
            value="Sabah saat 7'de kalkarım",
            source_text="Sabah saat 7'de kalkarım",
            active_memories=[self.wake_memory],
        )

        self.assertEqual(plan.action, ConsolidationAction.UNCHANGED)
        self.assertEqual(plan.matched_fact_id, "wake-fact-001")
        self.assertEqual(plan.key, "morning_wake_up_time")

    def test_terminal_sentence_punctuation_does_not_create_conflict(self) -> None:
        coffee = ActiveMemory(
            id="coffee-fact-001",
            category="routine",
            key="morning_drink",
            value="Her sabah sade T\u00fcrk kahvesi i\u00e7erim",
        )
        plan = resolve_consolidation(
            category="routine",
            key="morning_drink",
            value="Her sabah sade T\u00fcrk kahvesi i\u00e7erim.",
            source_text="Her sabah sade T\u00fcrk kahvesi i\u00e7erim.",
            active_memories=[coffee],
            ignore_terminal_sentence_punctuation=True,
        )

        self.assertEqual(plan.action, ConsolidationAction.UNCHANGED)
        self.assertEqual(plan.matched_fact_id, coffee.id)

    def test_terminal_punctuation_can_remain_significant_for_literal_values(self) -> None:
        secret = ActiveMemory(
            id="credential-fact-001",
            category="credential",
            key="door_code",
            value="secret!",
        )
        plan = resolve_consolidation(
            category="credential",
            key="door_code",
            value="secret",
            source_text="secret",
            active_memories=[secret],
        )

        self.assertEqual(
            plan.action,
            ConsolidationAction.CONFLICT_REQUIRES_CONFIRMATION,
        )

    def test_exclamation_is_not_treated_as_optional_sentence_punctuation(self) -> None:
        routine = ActiveMemory(
            id="routine-fact-001",
            category="routine",
            key="reminder_phrase",
            value="Dikkat!",
        )
        plan = resolve_consolidation(
            category="routine",
            key="reminder_phrase",
            value="Dikkat",
            source_text="Dikkat",
            active_memories=[routine],
            ignore_terminal_sentence_punctuation=True,
        )

        self.assertEqual(
            plan.action,
            ConsolidationAction.CONFLICT_REQUIRES_CONFIRMATION,
        )

    def test_explicit_correction_supersedes_existing_slot(self) -> None:
        plan = resolve_consolidation(
            category="behavior",
            key="wake_up_hour",
            value="Sabah saat 8'de kalkıyorum",
            source_text="Artık sabah saat 8'de kalkıyorum",
            active_memories=[self.wake_memory],
        )

        self.assertEqual(plan.action, ConsolidationAction.SUPERSEDE)
        self.assertEqual(plan.matched_fact_id, "wake-fact-001")

    def test_different_value_without_correction_requires_confirmation(self) -> None:
        plan = resolve_consolidation(
            category="routine",
            key="morning_wake_time",
            value="Sabah saat 8'de kalkıyorum",
            source_text="Sabah saat 8'de kalkıyorum",
            active_memories=[self.wake_memory],
        )

        self.assertEqual(
            plan.action,
            ConsolidationAction.CONFLICT_REQUIRES_CONFIRMATION,
        )

    def test_extractor_update_hint_cannot_bypass_explicit_change_policy(self) -> None:
        plan = resolve_consolidation(
            category="routine",
            key="morning_wake_time",
            value="Sabah saat 8'de kalkıyorum",
            source_text="Sabah saat 8'de kalkıyorum",
            active_memories=[self.wake_memory],
            matched_memory_id="wake-fact-001",
            relation_to_existing="update",
        )

        self.assertEqual(
            plan.action,
            ConsolidationAction.CONFLICT_REQUIRES_CONFIRMATION,
        )

    def test_extractor_same_hint_cannot_suppress_explicit_correction(self) -> None:
        plan = resolve_consolidation(
            category="routine",
            key="morning_wake_time",
            value="Sabah saat 8'de kalkarım",
            source_text="Artık sabah saat 8'de kalkarım",
            active_memories=[self.wake_memory],
            matched_memory_id="wake-fact-001",
            relation_to_existing="same",
        )

        self.assertEqual(plan.action, ConsolidationAction.SUPERSEDE)

    def test_extractor_same_hint_cannot_hide_different_unmarked_value(self) -> None:
        plan = resolve_consolidation(
            category="routine",
            key="morning_wake_time",
            value="Sabah saat 8'de kalkarım",
            source_text="Sabah saat 8'de kalkarım",
            active_memories=[self.wake_memory],
            matched_memory_id="wake-fact-001",
            relation_to_existing="same",
        )

        self.assertEqual(
            plan.action,
            ConsolidationAction.CONFLICT_REQUIRES_CONFIRMATION,
        )

    def test_valid_extractor_reference_can_link_nonidentical_key(self) -> None:
        tea_memory = ActiveMemory(
            id="tea-fact-001",
            category="preference",
            key="tea_drinking_routine",
            value="Çayımı açık içerim",
        )
        plan = resolve_consolidation(
            category="preferences",
            key="tea_preparation_style",
            value="Çayımı koyu içerim",
            source_text="Artık çayımı koyu içerim",
            active_memories=[tea_memory],
            matched_memory_id="tea-fact-001",
            relation_to_existing="update",
        )

        self.assertEqual(plan.action, ConsolidationAction.SUPERSEDE)
        self.assertEqual(plan.matched_by, "extractor_reference")

    def test_unknown_extractor_reference_is_not_trusted(self) -> None:
        plan = resolve_consolidation(
            category="preference",
            key="music_style",
            value="Türk sanat müziği",
            source_text="Türk sanat müziği severim",
            active_memories=[self.wake_memory],
            matched_memory_id="invented-id",
            relation_to_existing="update",
        )

        self.assertEqual(plan.action, ConsolidationAction.CREATE)
        self.assertIsNone(plan.matched_fact_id)


if __name__ == "__main__":
    unittest.main()
