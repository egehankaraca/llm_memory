import unittest

from memory_safety import evidence_clause, has_medication_dose_amount, has_mixed_question_clauses, has_reported_speech, is_question_like, numeric_atoms, supported_quote, temporal_atoms


class MemorySafetyTest(unittest.TestCase):
    def test_final_punctuation_does_not_attach_next_question(self):
        text = "Çayımı açık içerim. Yarın hava nasıl?"
        self.assertEqual(evidence_clause("Çayımı açık içerim.", text), "Çayımı açık içerim.")
        self.assertFalse(is_question_like(evidence_clause("açık", text)))
        self.assertTrue(is_question_like(evidence_clause("Yarın hava", text)))

    def test_time_decimal_cannot_hide_question_from_clause_guard(self):
        text = "Kaçta saat 7.00 tenis oynarım"
        self.assertTrue(is_question_like(evidence_clause("tenis oynarım", text)))

    def test_turkish_casefold_offsets_cannot_jump_out_of_question_clause(self):
        text = "İ" * 20 + ". Kaçta tenis oynarım. Çayımı açık içerim."
        self.assertTrue(is_question_like(evidence_clause("TENİS OYNARIM", text)))
        self.assertFalse(is_question_like(evidence_clause("Çayımı açık içerim.", text)))

    def test_quote_can_normalize_whitespace_but_not_invent_words(self):
        self.assertTrue(supported_quote("Çayımı açık", "Çayımı   açık içerim"))
        self.assertFalse(supported_quote("Çayımı koyu", "Çayımı açık içerim"))

    def test_numeric_zero_and_minutes_are_not_silently_removed(self):
        self.assertEqual(numeric_atoms("7, 07:00, 7.00"), {7})
        self.assertEqual(numeric_atoms("00:00, 7:30"), {0, 7, 30})

    def test_medication_dose_amount_does_not_confuse_time_with_dose(self):
        self.assertFalse(has_medication_dose_amount("İlacımı her sabah saat 8'de alırım"))
        self.assertFalse(has_medication_dose_amount("İlacımı 08:30'da alırım"))
        self.assertTrue(has_medication_dose_amount("İlacımın 5 mg olanını alırım"))
        self.assertTrue(has_medication_dose_amount("Akşam hapından iki tane alırım"))

    def test_weekday_does_not_match_prefix_of_other_day(self):
        self.assertEqual(temporal_atoms("Cumartesi"), {"weekly", "day:saturday"})
        self.assertEqual(temporal_atoms("Her sabah"), {"daily"})

    def test_current_statement_and_actual_question_are_distinct(self):
        self.assertFalse(is_question_like("Biraz dinlenmek istiyorum."))
        self.assertTrue(is_question_like("Biraz dinlenmeli miyim?"))
        self.assertTrue(is_question_like("Her sabah kaçta tenis oynarım"))

    def test_question_guard_covers_inflected_turkish_and_english_forms(self):
        for text in (
            "Biraz dinlenmek ister miydim",
            "Dinlenmek istiyor muyduk",
            "Yarın tenis oynasam mıydı",
            "Should I rest",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_question_like(text))

    def test_reported_speech_requires_both_quote_and_reporting_verb(self):
        self.assertTrue(has_reported_speech("Kızım 'başım ağrıyor' dedi."))
        self.assertFalse(has_reported_speech("Başım ağrıyor."))
        self.assertFalse(has_reported_speech("Doktor bana dinlenmemi söyledi."))

    def test_mixed_clause_review_does_not_treat_question_fragment_as_assertion(self):
        self.assertTrue(has_mixed_question_clauses("Çayımı açık içerim. Yarın hava nasıl?"))
        self.assertFalse(has_mixed_question_clauses("Her sabah kaçta tenis oynarım"))
        self.assertFalse(has_mixed_question_clauses("tenis. Her sabah kaçta tenis oynarım?"))


if __name__ == "__main__":
    unittest.main()
