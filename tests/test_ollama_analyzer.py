import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.error import URLError

import memory_analyzer
import models


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def extracted_item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "should_store": True,
        "claim_kind": "assertion",
        "subject": "user",
        "speech_act": "preference",
        "temporal_scope": "persistent",
        "sensitivity_domain": "none",
        "category": "preferences",
        "key": "tea_style",
        "value": "açık",
        "confidence": 0.93,
        "reason": "Kullanıcının kalıcı tercihi.",
    }
    item.update(overrides)
    if "claim_kind" not in overrides:
        if item["speech_act"] == "question":
            item["claim_kind"] = "question"
        elif item["speech_act"] == "courtesy":
            item["claim_kind"] = "contextual_reply"
    return item


class OllamaAnalyzerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.occurred_at = datetime(2026, 9, 11, 9, 0, tzinfo=timezone.utc)
        self.environment = patch.dict(
            os.environ,
            {
                "MEMORY_ANALYZER_PROVIDER": "ollama",
                "OLLAMA_MODEL": "test-model",
                "OLLAMA_TIMEOUT_SECONDS": "1",
                "MEMORY_TEMPORARY_TTL_MINUTES": "60",
                "MEMORY_TIMEZONE": "Europe/Istanbul",
                # Most tests below exercise the legacy opt-in repair path
                # explicitly. Production now defaults this expensive second
                # model call to off; dedicated tests cover that default.
                "MEMORY_SEMANTIC_RETRY": "true",
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()

    def analyze_mocked(
        self,
        item: dict[str, object],
        text: str | None = None,
    ) -> list[memory_analyzer.MemoryDecision]:
        return self.analyze_mocked_items([item], text)

    def analyze_mocked_items(
        self,
        items: list[dict[str, object]],
        text: str | None = None,
        recent_messages: list[dict[str, str]] | None = None,
    ) -> list[memory_analyzer.MemoryDecision]:
        if text is None:
            text = ". ".join(f"Tercihim {item['value']}" for item in items)
        items = [{**item, "evidence_text": item.get("evidence_text", text)} for item in items]
        ollama_response = FakeResponse(
            {
                "message": {
                    "content": json.dumps({"items": items}, ensure_ascii=False)
                }
            }
        )
        with patch.object(
            memory_analyzer.urllib_request,
            "urlopen",
            return_value=ollama_response,
        ):
            return memory_analyzer.analyze_message(
                text,
                self.occurred_at,
                recent_messages,
            )

    def test_policy_routes_semantic_preference_to_long_term(self) -> None:
        decisions = self.analyze_mocked(extracted_item())

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.LONG_TERM)
        self.assertEqual(decisions[0].key, "tea_style")
        self.assertEqual(decisions[0].analyzer_source, "ollama")
        self.assertEqual(decisions[0].analysis_metadata["speech_act"], "preference")

    def test_coffee_slot_requires_coffee_evidence(self) -> None:
        pasta_text = "Makarna yemeyi çok severim"
        pasta = self.analyze_mocked(extracted_item(
            category="dietary_preference",
            key="coffee_preference",
            value=pasta_text,
            evidence_text=pasta_text,
        ), text=pasta_text)[0]
        coffee_text = "Her sabah sütlü Türk kahvesi içerim"
        coffee = self.analyze_mocked(extracted_item(
            speech_act="habit",
            category="dietary_preference",
            key="coffee_preference",
            value=coffee_text,
            evidence_text=coffee_text,
        ), text=coffee_text)[0]

        self.assertNotEqual(pasta.key, "coffee_preference")
        self.assertTrue(pasta.key.startswith("preference_"))
        self.assertEqual(
            pasta.analysis_metadata["slot_evidence_guard"],
            "coffee_slot_without_coffee_evidence",
        )
        self.assertEqual(coffee.key, "coffee_preference")
        self.assertNotIn("slot_evidence_guard", coffee.analysis_metadata)

    def test_existing_memories_are_sent_and_relation_is_auditable(self) -> None:
        captured_payload: dict[str, object] = {}
        response_item = extracted_item(
            category="routine",
            key="morning_wake_time",
            value="Sabah 8'de kalkarım",
            matched_memory_id="12345678-1234-1234-1234-123456789012",
            relation_to_existing="update",
            evidence_text="Artık sabah 8'de kalkarım",
        )

        def urlopen(request: object, timeout: float) -> FakeResponse:
            del timeout
            payload = json.loads(request.data.decode("utf-8"))
            captured_payload.update(
                json.loads(payload["messages"][1]["content"])
            )
            return FakeResponse(
                {
                    "message": {
                        "content": json.dumps(
                            {"items": [response_item]},
                            ensure_ascii=False,
                        )
                    }
                }
            )

        existing = [
            {
                "id": "12345678-1234-1234-1234-123456789012",
                "category": "routine",
                "key": "morning_wake_time",
                "value": "Sabah 7'de kalkarım",
            }
        ]
        with patch.object(memory_analyzer.urllib_request, "urlopen", urlopen):
            decisions = memory_analyzer.analyze_message(
                "Artık sabah 8'de kalkarım",
                self.occurred_at,
                existing_memories=existing,
            )

        self.assertEqual(captured_payload["existing_memories"], existing)
        self.assertEqual(
            decisions[0].analysis_metadata["matched_memory_id"],
            existing[0]["id"],
        )
        self.assertEqual(
            decisions[0].analysis_metadata["relation_to_existing"],
            "update",
        )

    def test_unavailable_ollama_uses_rules_fallback(self) -> None:
        with patch.object(
            memory_analyzer.urllib_request,
            "urlopen",
            side_effect=URLError("offline"),
        ):
            decisions = memory_analyzer.analyze_message(
                "Bugün dışarı çıkmak istiyorum",
                self.occurred_at,
            )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SHORT_TERM)
        self.assertEqual(decisions[0].analyzer_source, "rules_fallback")

    def test_rules_fallback_keeps_emergency_contact_name_and_phone(self) -> None:
        text = "Acil durumda kızım Ayşe'yi 0555 123 45 67 numarasından ara."
        with patch.object(
            memory_analyzer.urllib_request,
            "urlopen",
            side_effect=URLError("offline"),
        ):
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]

        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.category, "emergency_contact")
        self.assertEqual(decision.key, "primary_emergency_contact")
        self.assertEqual(decision.value, text)
        self.assertEqual(decision.sensitivity, models.Sensitivity.EMERGENCY_CONTACT)
        self.assertEqual(decision.analyzer_source, "rules_fallback")

    def test_emergency_contact_uses_full_evidence_when_model_returns_only_phone(self) -> None:
        text = "Acil durumda kızım Ayşe'yi 0555 123 45 67 numarasından ara."
        decision = self.analyze_mocked(extracted_item(
            speech_act="preference",
            sensitivity_domain="personal",
            category="emergency_contact",
            key="emergency_contact_number",
            value="0555 123 45 67",
            evidence_text=text,
        ), text=text)[0]

        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.category, "emergency_contact")
        self.assertEqual(decision.key, "primary_emergency_contact")
        self.assertEqual(decision.value, text)
        self.assertEqual(decision.sensitivity, models.Sensitivity.EMERGENCY_CONTACT)

    def test_direct_health_assertion_is_sensitive_but_user_asserted(self) -> None:
        decisions = self.analyze_mocked(
            extracted_item(
                speech_act="current_state",
                temporal_scope="unknown",
                sensitivity_domain="health",
                category="health",
                key="heart_condition",
                value="ritim bozukluğu",
            )
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decisions[0].sensitivity, models.Sensitivity.HEALTH)
        self.assertEqual(decisions[0].expires_at, self.occurred_at + timedelta(hours=1))
        self.assertEqual(decisions[0].analysis_metadata["temporal_scope"], "current_session")
        self.assertEqual(decisions[0].analysis_metadata["extracted_temporal_scope"], "unknown")
        self.assertEqual(
            decisions[0].analysis_metadata["verification_status"],
            "user_asserted",
        )

    def test_explicit_current_symptom_is_user_asserted_temporary_memory(self) -> None:
        for text in ["başım ağrıyor", "Mideme kramp giriyor", "Kendimi halsiz hissediyorum"]:
            with self.subTest(text=text):
                decisions = self.analyze_mocked(extracted_item(
                    speech_act="current_state", temporal_scope="current_session",
                    sensitivity_domain="health", category="symptom", key="current_symptom",
                    value=text, evidence_text=text,
                ), text=text)
                self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SENSITIVE)
                self.assertEqual(decisions[0].sensitivity, models.Sensitivity.HEALTH)
                self.assertEqual(decisions[0].expires_at, self.occurred_at + timedelta(hours=1))

    def test_semantically_unasserted_symptom_is_not_resurrected_by_sensitive_terms(self) -> None:
        for claim_kind in ["question", "inferred", "ambiguous"]:
            with self.subTest(claim_kind=claim_kind):
                decisions = self.analyze_mocked(extracted_item(
                    claim_kind=claim_kind, speech_act="current_state", temporal_scope="current_session",
                    sensitivity_domain="health", category="symptom", key="current_symptom",
                    value="başım ağrıyor", evidence_text="başım ağrıyor",
                ), text="başım ağrıyor")
                self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)
                self.assertEqual(decisions[0].analysis_metadata["evidence_guard"], "not_asserted")


    def test_credential_domain_is_discarded(self) -> None:
        decisions = self.analyze_mocked(
            extracted_item(
                speech_act="profile_fact",
                sensitivity_domain="credential",
                category="security",
                key="wifi_password",
                value="MaviEv-2026",
            )
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decisions[0].sensitivity, models.Sensitivity.NORMAL)
        self.assertEqual(decisions[0].analysis_metadata["evidence_guard"], "protected_secret_discarded")

    def test_category_policy_corrects_missed_medication_domain(self) -> None:
        decisions = self.analyze_mocked(
            extracted_item(
                speech_act="habit",
                sensitivity_domain="none",
                category="medication",
                key="medication_schedule",
                value="İlacımı kahvaltıdan sonra alıyorum",
            )
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decisions[0].sensitivity, models.Sensitivity.HEALTH)
        self.assertEqual(decisions[0].analyzer_source, "ollama")
        self.assertIsNone(decisions[0].expires_at)
        self.assertEqual(
            decisions[0].analysis_metadata["verification_status"],
            "user_asserted",
        )
        self.assertTrue(
            decisions[0].analysis_metadata["sensitivity_overridden_by_policy"]
        )

    def test_medication_clock_time_is_user_asserted_not_dosage(self) -> None:
        text = "Tansiyon ilacımı her sabah saat 8'de alırım"
        decision = self.analyze_mocked(extracted_item(
            speech_act="habit", temporal_scope="persistent",
            sensitivity_domain="health", category="medication",
            key="morning_medication_time", value=text, evidence_text=text,
        ), text=text)[0]

        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.analysis_metadata["verification_status"], "user_asserted")
        self.assertNotIn("health_review_reason", decision.analysis_metadata)

    def test_explicit_medication_dose_is_user_asserted(self) -> None:
        text = "Akşam hapından iki tane alırım"
        decision = self.analyze_mocked(extracted_item(
            speech_act="habit", temporal_scope="persistent",
            sensitivity_domain="health", category="medication",
            key="evening_medication_dose", value=text, evidence_text=text,
        ), text=text)[0]

        self.assertEqual(decision.analysis_metadata["verification_status"], "user_asserted")

    def test_uncertain_medication_claim_is_discarded(self) -> None:
        text = "Galiba ilacımı sabah alıyorum"
        for claim_kind in ["inferred", "ambiguous"]:
            with self.subTest(claim_kind=claim_kind):
                decision = self.analyze_mocked(extracted_item(
                    should_store=True, claim_kind=claim_kind,
                    speech_act="habit", temporal_scope="persistent",
                    sensitivity_domain="health", category="medication",
                    key="medication_schedule", value=text, evidence_text=text,
                ), text=text)[0]
                self.assertEqual(
                    decision.memory_type,
                    models.CandidateMemoryType.DISCARD,
                )

    def test_explicit_medication_schedule_correction_is_user_asserted(self) -> None:
        text = "Artık tansiyon ilacımı her sabah saat 9'da alıyorum"
        decision = self.analyze_mocked(extracted_item(
            speech_act="habit", temporal_scope="persistent",
            sensitivity_domain="health", category="medication",
            key="morning_medication_time", value=text, evidence_text=text,
            matched_memory_id="existing-medication",
            relation_to_existing="update",
        ), text=text)[0]

        self.assertEqual(decision.analysis_metadata["verification_status"], "user_asserted")

    def test_communication_category_is_personal_but_not_confirmation_gated(self) -> None:
        decisions = self.analyze_mocked(
            extracted_item(
                speech_act="preference",
                sensitivity_domain="none",
                category="communication",
                key="title_of_address",
                value="Ahmet Bey",
            ),
            text="Bana bundan sonra Ahmet Bey diye hitap et",
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.LONG_TERM)
        self.assertEqual(decisions[0].sensitivity, models.Sensitivity.PERSONAL)

    def test_protected_category_cannot_be_downgraded_to_personal(self) -> None:
        decisions = self.analyze_mocked(extracted_item(
            speech_act="profile_fact", sensitivity_domain="personal", category="diagnosis",
            key="asthma", value="Ben astımlıyım",
        ), text="Ben astımlıyım")
        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decisions[0].sensitivity, models.Sensitivity.HEALTH)
        self.assertEqual(
            decisions[0].analysis_metadata["verification_status"],
            "user_asserted",
        )

    def test_protected_literal_data_overrides_personal_even_in_generic_category(self) -> None:
        for text, sensitivity in [
            ("Acil durumda kızım Ayşe’yi 0555 123 45 67 numarasından ara", models.Sensitivity.EMERGENCY_CONTACT),
            ("Adresim Bahar Sokak 12 numara, Kadıköy", models.Sensitivity.LOCATION),
            ("İnternet şifrem MaviEv-2026", models.Sensitivity.CREDENTIAL),
            ("IBANım TR12 0006 2000 0000 0000 1234 56", models.Sensitivity.FINANCIAL),
        ]:
            with self.subTest(sensitivity=sensitivity):
                decisions = self.analyze_mocked(extracted_item(
                    speech_act="profile_fact", category="profile", key="private_data",
                    sensitivity_domain="personal", value=text,
                ), text=text)
                if sensitivity in {models.Sensitivity.CREDENTIAL, models.Sensitivity.FINANCIAL}:
                    self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)
                    self.assertEqual(decisions[0].sensitivity, models.Sensitivity.NORMAL)
                else:
                    self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SENSITIVE)
                    self.assertEqual(decisions[0].sensitivity, sensitivity)
                    self.assertTrue(decisions[0].analysis_metadata["sensitivity_overridden_by_policy"])

    def test_protection_checks_only_the_validated_evidence_clause(self) -> None:
        text = "Çayımı açık içerim. Adresim nedir?"
        decisions = self.analyze_mocked(extracted_item(
            value="Çayımı açık içerim", evidence_text="Çayımı açık içerim.",
        ), text=text)
        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.LONG_TERM)

    def test_semantic_preference_overrides_conflicting_should_store_hint(self) -> None:
        decisions = self.analyze_mocked(
            extracted_item(
                should_store=False,
                speech_act="preference",
                temporal_scope="persistent",
                sensitivity_domain="personal",
                category="communication",
                key="speech_style",
                value="Kısa cümleler ve yavaş konuşma",
            )
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.LONG_TERM)
        self.assertEqual(decisions[0].sensitivity, models.Sensitivity.PERSONAL)

    def test_discard_residue_is_removed_when_message_has_a_memory(self) -> None:
        decisions = self.analyze_mocked_items(
            [
                extracted_item(
                    speech_act="intent",
                    temporal_scope="today",
                    category="session",
                    key="walk",
                    value="Bugün yürüyüşe çıkmak",
                ),
                extracted_item(
                    should_store=False,
                    speech_act="courtesy",
                    temporal_scope="current_session",
                    category="none",
                    key="none",
                    value="Evet",
                ),
            ]
        )

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SHORT_TERM)

    def test_short_term_intent_preserves_its_concept_key(self) -> None:
        decisions = self.analyze_mocked(
            extracted_item(
                speech_act="intent",
                temporal_scope="today",
                category="session",
                key="park_walk",
                value="parkta yürümek",
            )
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SHORT_TERM)
        self.assertIsInstance(decisions[0].value, dict)
        self.assertEqual(decisions[0].value["description"], "parkta yürümek")
        self.assertEqual(decisions[0].key, "park_walk")
        self.assertEqual(
            decisions[0].expires_at,
            datetime(2026, 9, 11, 21, tzinfo=timezone.utc),
        )

    def test_acknowledgement_only_message_is_not_temporary_memory(self) -> None:
        for text in ("Evet doğru", "Tamam", "Aynen"):
            with self.subTest(text=text):
                decision = self.analyze_mocked(
                    extracted_item(
                        speech_act="intent",
                        temporal_scope="today",
                        category="session",
                        key="confirmation",
                        value=text,
                        evidence_text=text,
                    ),
                    text=text,
                )[0]
                self.assertEqual(
                    decision.memory_type,
                    models.CandidateMemoryType.DISCARD,
                )
                self.assertEqual(
                    decision.analysis_metadata["evidence_guard"],
                    "acknowledgement_only",
                )

    def test_independent_temporary_items_have_distinct_safe_keys(self) -> None:
        statements = ["Bugün parkta yürümek istiyorum", "Bu akşam haberleri izleyeceğim"]
        decisions = self.analyze_mocked_items([
            extracted_item(
                speech_act="intent", temporal_scope="today", category="session",
                key="Park Walk!", value=statements[0], evidence_text=statements[0],
            ),
            extracted_item(
                speech_act="intent", temporal_scope="today", category="session",
                key="evening_news", value=statements[1], evidence_text=statements[1],
            ),
        ], text=". ".join(statements))
        self.assertEqual([decision.key for decision in decisions], ["park_walk", "evening_news"])
        self.assertTrue(all(decision.memory_type == models.CandidateMemoryType.SHORT_TERM for decision in decisions))

    def test_unusable_temporary_keys_get_distinct_deterministic_fallbacks(self) -> None:
        statements = ["Şimdi kitap okumak istiyorum", "Biraz müzik dinlemek istiyorum"]
        for unusable in ["none", "unknown", "!!!"]:
            with self.subTest(key=unusable):
                items = [extracted_item(
                    speech_act="intent", temporal_scope="current_session", category="session",
                    key=unusable, value=statement, evidence_text=statement,
                ) for statement in statements]
                decisions = self.analyze_mocked_items(items, text=". ".join(statements))
                repeated = self.analyze_mocked_items(items, text=". ".join(statements))
                self.assertNotEqual(decisions[0].key, decisions[1].key)
                self.assertEqual([decision.key for decision in decisions], [decision.key for decision in repeated])
                self.assertTrue(all(decision.analysis_metadata["key_fallback_used"] for decision in decisions))

    def test_session_and_unknown_intents_use_configured_bounded_ttl(self) -> None:
        for configured, expected in [("15", 15), ("0", 1), ("-100", 1), ("5000", 1440), ("bad", 60)]:
            for scope in ["current_session", "unknown", "persistent"]:
                with self.subTest(configured=configured, scope=scope), patch.dict(
                    os.environ, {"MEMORY_TEMPORARY_TTL_MINUTES": configured},
                ):
                    text = "Biraz dinlenmek istiyorum."
                    decision = self.analyze_mocked(extracted_item(
                        speech_act="intent", temporal_scope=scope, category="session",
                        key="rest", value=text, evidence_text=text,
                    ), text=text)[0]
                    self.assertEqual(decision.expires_at, self.occurred_at + timedelta(minutes=expected))
                    self.assertEqual(decision.value["target"], "current_session")

    def test_sensitivity_does_not_remove_explicit_day_scope(self) -> None:
        text = "Adresim bugün Bahar Sokak 12 numara"
        for scope, expected_day in [("today", 11), ("tomorrow", 12)]:
            with self.subTest(scope=scope):
                decision = self.analyze_mocked(extracted_item(
                    speech_act="profile_fact", temporal_scope=scope, category="location",
                    key="temporary_address", sensitivity_domain="location", value=text,
                ), text=text)[0]
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
                self.assertEqual(decision.expires_at, datetime(2026, 9, expected_day, 21, tzinfo=timezone.utc))

    def test_symptom_expiry_and_stable_health_lifetime_are_separate(self) -> None:
        for scope in ["unknown", "current_session"]:
            with self.subTest(scope=scope):
                text = "Başım dönüyor"
                decision = self.analyze_mocked(extracted_item(
                    speech_act="current_state", temporal_scope=scope, sensitivity_domain="health",
                    category="symptom", key="dizziness", value=text,
                ), text=text)[0]
                self.assertEqual(decision.expires_at, self.occurred_at + timedelta(hours=1))
                self.assertEqual(decision.scope, models.MemoryScope.SESSION)
        persistent_text = "Uzun süredir başım dönüyor"
        persistent = self.analyze_mocked(extracted_item(
            speech_act="current_state", temporal_scope="persistent",
            sensitivity_domain="health", category="symptom",
            key="persistent_dizziness", value=persistent_text,
        ), text=persistent_text)[0]
        self.assertIsNone(persistent.expires_at)
        self.assertEqual(persistent.scope, models.MemoryScope.PROFILE)
        for act, text in [
            ("habit", "Tansiyon ilacımı kahvaltıdan sonra alıyorum"),
            ("profile_fact", "Ben astımlıyım"),
        ]:
            with self.subTest(act=act):
                decision = self.analyze_mocked(extracted_item(
                    speech_act=act, temporal_scope="persistent", sensitivity_domain="health",
                    category="health", key="stable_health", value=text,
                ), text=text)[0]
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
                self.assertIsNone(decision.expires_at)

    def test_health_safety_escalation_preserves_transient_expiry(self) -> None:
        text = "Başım ağrıyor"
        decision = self.analyze_mocked(extracted_item(
            speech_act="current_state", temporal_scope="unknown", sensitivity_domain="none",
            category="state", key="headache", value=text,
        ), text=text)[0]
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.sensitivity, models.Sensitivity.HEALTH)
        self.assertEqual(decision.key, "headache")
        self.assertEqual(decision.expires_at, self.occurred_at + timedelta(hours=1))
        self.assertEqual(decision.analysis_metadata["temporal_scope"], "current_session")
        self.assertEqual(decision.value["type"], "user_state")
        self.assertEqual(decision.analyzer_source, "rules_guard")
        self.assertEqual(decision.analysis_metadata["upstream_analyzer_source"], "ollama")
        self.assertEqual(decision.analysis_metadata["policy_guard"], "sensitive_rules_guard")
        self.assertFalse(decision.analysis_metadata["guard_generated_candidate"])

    def test_guard_generated_sensitive_candidate_keeps_model_provenance(self) -> None:
        text = "Ba\u015f\u0131m a\u011fr\u0131yor"
        decision = self.analyze_mocked(extracted_item(
            should_store=False,
            claim_kind="assertion",
            subject="user",
            speech_act="other",
            temporal_scope="unknown",
            sensitivity_domain="none",
            category="none",
            key="none",
            value=text,
        ), text=text)[0]

        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.analyzer_source, "rules_guard")
        self.assertEqual(decision.analysis_metadata["upstream_analyzer_source"], "ollama")
        self.assertEqual(decision.analysis_metadata["upstream_memory_types"], ["discard"])
        self.assertTrue(decision.analysis_metadata["guard_generated_candidate"])

    def test_rule_fallback_symptom_is_temporary_but_medication_habit_is_persistent(self) -> None:
        with patch.dict(os.environ, {"MEMORY_ANALYZER_PROVIDER": "rules"}):
            symptom = memory_analyzer.analyze_message("Başım ağrıyor", self.occurred_at)[0]
            habit = memory_analyzer.analyze_message("Tansiyon ilacımı kahvaltıdan sonra alıyorum", self.occurred_at)[0]
        self.assertEqual(symptom.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(symptom.expires_at, self.occurred_at + timedelta(hours=1))
        self.assertIsNone(habit.expires_at)

    def test_rule_fallback_does_not_attribute_quoted_health_state_to_user(self) -> None:
        with patch.dict(os.environ, {"MEMORY_ANALYZER_PROVIDER": "rules"}):
            decision = memory_analyzer.analyze_message(
                "Kızım 'başım ağrıyor' dedi.", self.occurred_at,
            )[0]
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "reported_speech")

    def test_semantic_review_recovers_mislabeled_statement_without_phrase_rule(self) -> None:
        for text, key in [
            ("Biraz dinlenmek istiyorum.", "rest"),
            ("Müzik dinlemek istiyorum.", "listen_to_music"),
        ]:
            with self.subTest(text=text):
                initial = extracted_item(
                    claim_kind="question", should_store=False, speech_act="intent",
                    temporal_scope="current_session", category="session", key=key,
                    value=text, evidence_text=text,
                )
                repaired = {**initial, "claim_kind": "assertion", "should_store": True}
                responses = [FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                             for item in [initial, repaired]]
                with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=responses) as mocked:
                    decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
                self.assertEqual(mocked.call_count, 2)
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.SHORT_TERM)
                self.assertEqual(decision.key, key)
                self.assertEqual(decision.analysis_metadata["extraction_retry_count"], 1)
                self.assertEqual(decision.analysis_metadata["initial_claim_kinds"], ["question"])
                self.assertIn("inconsistent_semantic_labels", decision.analysis_metadata["extraction_retry_reasons"])

    def test_semantic_review_recovers_all_question_tuple_for_declarative_state(self) -> None:
        text = "Başım ağrıyor."
        initial = extracted_item(
            should_store=False, claim_kind="question", subject="unknown",
            speech_act="question", temporal_scope="unknown",
            sensitivity_domain="none", category="none", key="none",
            value=text, evidence_text=text,
        )
        repaired = extracted_item(
            should_store=True, claim_kind="assertion", subject="user",
            speech_act="current_state", temporal_scope="current_session",
            sensitivity_domain="health", category="symptom", key="headache",
            value=text, evidence_text=text,
        )
        responses = [FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                     for item in (initial, repaired)]
        with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=responses) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.sensitivity, models.Sensitivity.HEALTH)
        self.assertEqual(decision.expires_at, self.occurred_at + timedelta(hours=1))
        self.assertEqual(
            decision.analysis_metadata["extraction_retry_reasons"],
            ["declarative_labeled_as_question"],
        )

    def test_actual_question_all_question_tuple_is_not_reviewed(self) -> None:
        text = "Başım ağrıyor mu?"
        item = extracted_item(
            should_store=False, claim_kind="question", subject="unknown",
            speech_act="question", temporal_scope="unknown", category="none",
            key="none", value=text, evidence_text=text,
        )
        response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
        with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)

    def test_inflected_or_english_questions_cannot_be_repaired_into_intents(self) -> None:
        for text in (
            "Biraz dinlenmek ister miydim",
            "Dinlenmek istiyor muyduk",
            "Yarın tenis oynasam mıydı",
            "Should I rest",
        ):
            with self.subTest(text=text):
                initial = extracted_item(
                    should_store=False, claim_kind="question", subject="unknown",
                    speech_act="intent", temporal_scope="current_session",
                    category="session", key="rest", value=text, evidence_text=text,
                )
                response = FakeResponse({"message": {"content": json.dumps({"items": [initial]})}})
                with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response) as mocked:
                    decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)

    def test_related_person_transient_state_is_not_user_session_memory(self) -> None:
        text = "Kızım bugün dinlenmek istiyor."
        decision = self.analyze_mocked(extracted_item(
            subject="related_person", speech_act="intent", temporal_scope="today",
            category="session", key="rest", value=text, evidence_text=text,
        ), text=text)[0]
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "related_person_transient")

    def test_string_value_must_be_grounded_in_literal_evidence(self) -> None:
        text = "Biraz dinlenmek istiyorum."
        decision = self.analyze_mocked(extracted_item(
            speech_act="intent", temporal_scope="current_session",
            category="session", key="rest", value="Kitap okumak istiyorum",
            evidence_text=text,
        ), text=text)[0]
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "unsupported_value")

    def test_model_extraction_schema_rejects_structured_values(self) -> None:
        item = extracted_item(
            speech_act="intent",
            temporal_scope="current_session",
            category="session",
            key="drink_water",
            value={"description": "Bugün dışarı çıkmak istiyorum."},
            evidence_text="Şimdi bir bardak su içeceğim.",
        )
        with self.assertRaises(memory_analyzer.ValidationError):
            memory_analyzer.MemoryExtraction.model_validate({"items": [item]})

    def test_semantic_review_repairs_value_copied_from_unrelated_memory(self) -> None:
        existing = [{
            "id": "temporary-outdoor",
            "category": "session",
            "key": "outdoor_intent",
            "value": "Bugün dışarı çıkmak istiyorum.",
        }]
        cases = [
            (
                "Bu akşam haberleri izlemek istiyorum.",
                "today",
                "watch_news",
            ),
            (
                "Şimdi bir bardak su içeceğim.",
                "current_session",
                "drink_water",
            ),
        ]
        for text, scope, repaired_key in cases:
            with self.subTest(text=text):
                initial = extracted_item(
                    speech_act="intent", temporal_scope=scope,
                    category="session", key="outdoor_intent",
                    value="Bugün dışarı çıkmak istiyorum.", evidence_text=text,
                )
                repaired = {
                    **initial,
                    "key": repaired_key,
                    "value": text,
                    "matched_memory_id": None,
                    "relation_to_existing": "none",
                }
                responses = [
                    FakeResponse({
                        "message": {"content": json.dumps({"items": [item]})}
                    })
                    for item in (initial, repaired)
                ]
                with patch.object(
                    memory_analyzer.urllib_request,
                    "urlopen",
                    side_effect=responses,
                ) as mocked:
                    decision = memory_analyzer.analyze_message(
                        text,
                        self.occurred_at,
                        existing_memories=existing,
                    )[0]

                self.assertEqual(mocked.call_count, 2)
                self.assertEqual(
                    decision.memory_type,
                    models.CandidateMemoryType.SHORT_TERM,
                )
                self.assertEqual(decision.key, repaired_key)
                self.assertEqual(decision.value["description"], text)
                self.assertEqual(
                    decision.analysis_metadata["extraction_retry_reasons"],
                    ["unsupported_value"],
                )
                review_request = mocked.call_args_list[1].args[0]
                review_payload = json.loads(review_request.data.decode("utf-8"))
                review_data = json.loads(
                    review_payload["messages"][1]["content"]
                )
                self.assertEqual(
                    review_data["previous_extraction"]["items"][0]["value"],
                    "Bugün dışarı çıkmak istiyorum.",
                )
                self.assertEqual(
                    review_data["contract_violations"],
                    ["unsupported_value"],
                )

    def test_multi_item_grounding_review_keeps_valid_and_repairs_other_clause(self) -> None:
        preference = "Çayımı açık içerim."
        intent = "Bu akşam haberleri izlemek istiyorum."
        text = f"{preference} {intent}"
        valid = extracted_item(
            speech_act="preference",
            temporal_scope="persistent",
            category="preference",
            key="tea_style",
            value="açık",
            evidence_text=preference,
        )
        copied = extracted_item(
            speech_act="intent",
            temporal_scope="today",
            category="session",
            key="outdoor_intent",
            value="Bugün dışarı çıkmak istiyorum.",
            evidence_text=intent,
        )
        repaired = {
            **copied,
            "key": "watch_news",
            "value": intent,
        }
        contradictory_rewrite = {
            **valid,
            "key": "tea_style_rewritten",
            "value": preference,
        }
        responses = [
            FakeResponse({
                "message": {
                    "content": json.dumps({"items": [valid, copied]})
                }
            }),
            # A repair may repeat an already-valid clause differently. The
            # original validated decision wins; only the rejected clause is new.
            FakeResponse({
                "message": {
                    "content": json.dumps({
                        "items": [contradictory_rewrite, repaired]
                    })
                }
            }),
        ]
        with patch.object(
            memory_analyzer.urllib_request,
            "urlopen",
            side_effect=responses,
        ) as mocked:
            decisions = memory_analyzer.analyze_message(text, self.occurred_at)

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(
            {decision.memory_type for decision in decisions},
            {
                models.CandidateMemoryType.LONG_TERM,
                models.CandidateMemoryType.SHORT_TERM,
            },
        )
        self.assertEqual({decision.key for decision in decisions}, {"tea_style", "watch_news"})
        tea = next(decision for decision in decisions if decision.key == "tea_style")
        self.assertEqual(tea.value, "açık")
        self.assertTrue(all(
            decision.analysis_metadata["extraction_retry_reasons"]
            == ["unsupported_value"]
            for decision in decisions
        ))

    def test_grounding_review_is_bounded_and_rechecks_repaired_value(self) -> None:
        text = "Şimdi bir bardak su içeceğim."
        ungrounded = extracted_item(
            speech_act="intent", temporal_scope="current_session",
            category="session", key="outdoor_intent",
            value="Bugün dışarı çıkmak istiyorum.", evidence_text=text,
        )
        response = FakeResponse({
            "message": {"content": json.dumps({"items": [ungrounded]})}
        })
        with patch.object(
            memory_analyzer.urllib_request,
            "urlopen",
            return_value=response,
        ) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "unsupported_value")
        self.assertEqual(decision.analysis_metadata["extraction_retry_count"], 1)

    def test_grounding_review_excludes_non_memory_utterances(self) -> None:
        cases = [
            ("Bugün dışarı çıkmalı mıyım?", "user", "intent"),
            ("Televizyonu aç.", "user", "device_command"),
            ("Kızım bugün dinlenmek istiyor.", "third_party", "intent"),
            ("Ankara Türkiye'nin başkentidir.", "unknown", "profile_fact"),
        ]
        for text, subject, speech_act in cases:
            with self.subTest(text=text):
                item = extracted_item(
                    subject=subject,
                    speech_act=speech_act,
                    temporal_scope="today",
                    category="session",
                    key="wrong_slot",
                    value="Bugün dışarı çıkmak istiyorum.",
                    evidence_text=text,
                )
                response = FakeResponse({
                    "message": {"content": json.dumps({"items": [item]})}
                })
                with patch.object(
                    memory_analyzer.urllib_request,
                    "urlopen",
                    return_value=response,
                ) as mocked:
                    decision = memory_analyzer.analyze_message(
                        text,
                        self.occurred_at,
                    )[0]

                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(
                    decision.memory_type,
                    models.CandidateMemoryType.DISCARD,
                )

    def test_semantic_review_is_bounded_and_never_relabels_uncertain_claims(self) -> None:
        for claim in ["question", "inferred", "ambiguous"]:
            with self.subTest(claim=claim):
                text = "Biraz dinlenmek istiyorum."
                item = extracted_item(
                    claim_kind=claim, speech_act="intent", temporal_scope="current_session",
                    category="session", key="rest", value=text, evidence_text=text,
                )
                response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response) as mocked:
                    decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
                self.assertEqual(mocked.call_count, 2)
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
                self.assertEqual(decision.analysis_metadata["claim_kind"], claim)
                self.assertEqual(decision.analysis_metadata["evidence_guard"], "not_asserted")

    def test_semantic_review_reextracts_literal_unknown_subject_intent(self) -> None:
        text = "Biraz dinlenmek istiyorum."
        initial = extracted_item(
            should_store=False, claim_kind="question", subject="unknown", speech_act="intent",
            temporal_scope="unknown", category="session", key="rest", value=text, evidence_text=text,
        )
        repaired = {**initial, "should_store": True, "claim_kind": "assertion",
                    "subject": "user", "temporal_scope": "current_session"}
        responses = [FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                     for item in (initial, repaired)]
        with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=responses) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SHORT_TERM)
        self.assertEqual(decision.key, "rest")
        self.assertEqual(decision.expires_at, self.occurred_at + timedelta(hours=1))
        self.assertEqual(decision.analysis_metadata["claim_kind"], "assertion")
        self.assertEqual(decision.analysis_metadata["subject"], "user")
        self.assertEqual(decision.analysis_metadata["temporal_scope"], "current_session")
        self.assertEqual(decision.analysis_metadata["extraction_retry_count"], 1)
        self.assertEqual(decision.analysis_metadata["initial_claim_kinds"], ["question"])
        self.assertNotIn("semantic_normalizations", decision.analysis_metadata)

    def test_uncertain_unknown_intent_is_not_contract_normalized(self) -> None:
        text = "Biraz dinlenmek istiyorum."
        for claim in ["ambiguous", "inferred"]:
            with self.subTest(claim=claim):
                item = extracted_item(
                    should_store=False, claim_kind=claim, subject="unknown", speech_act="intent",
                    temporal_scope="unknown", category="session", key="rest", value=text, evidence_text=text,
                )
                response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response) as mocked:
                    decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
                self.assertEqual(mocked.call_count, 2)
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
                self.assertEqual(decision.analysis_metadata["subject"], "unknown")
                self.assertEqual(decision.analysis_metadata["extraction_retry_count"], 1)
                self.assertNotIn("semantic_normalizations", decision.analysis_metadata)

    def test_unknown_subject_gate_does_not_review_third_party_or_actual_question(self) -> None:
        for subject, text in [
            ("third_party", "Kızım biraz dinlenmek istiyor."),
            ("unknown", "Biraz dinlenmek istiyor muyum?"),
            ("unknown", "dinlenmek"),
        ]:
            with self.subTest(subject=subject):
                item = extracted_item(
                    should_store=False, claim_kind="question", subject=subject, speech_act="intent",
                    temporal_scope="unknown", category="session", key="rest", value=text, evidence_text=text,
                )
                response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response) as mocked:
                    decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)

    def test_contract_normalization_requires_literal_supported_evidence(self) -> None:
        text = "Biraz dinlenmek istiyorum."
        item = extracted_item(
            should_store=False, claim_kind="question", subject="unknown", speech_act="intent",
            temporal_scope="unknown", category="session", key="rest", value=text,
            evidence_text="Ben biraz dinlenmek istiyorum",
        )
        response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
        with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "not_asserted")
        self.assertNotIn("semantic_normalizations", decision.analysis_metadata)

    def test_semantic_review_preserves_explicit_day_scope(self) -> None:
        text = "Yarın biraz dinlenmek istiyorum."
        item = extracted_item(
            should_store=False, claim_kind="question", subject="unknown", speech_act="intent",
            temporal_scope="tomorrow", category="session", key="rest", value=text,
            evidence_text=text,
        )
        repaired = {**item, "should_store": True, "claim_kind": "assertion", "subject": "user"}
        responses = [FakeResponse({"message": {"content": json.dumps({"items": [value]})}})
                     for value in (item, repaired)]
        with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=responses) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SHORT_TERM)
        self.assertEqual(decision.value["target"], "tomorrow")
        self.assertEqual(decision.expires_at, datetime(2026, 9, 12, 21, tzinfo=timezone.utc))
        self.assertEqual(decision.analysis_metadata["temporal_scope"], "tomorrow")
        self.assertEqual(decision.analysis_metadata["extraction_retry_count"], 1)
        self.assertNotIn("semantic_normalizations", decision.analysis_metadata)

    def test_contract_normalization_is_exactly_bounded_to_question_unknown_intent(self) -> None:
        text = "Biraz dinlenmek istiyorum."
        variants = [
            {"subject": "user", "speech_act": "intent"},
            {"subject": "related_person", "speech_act": "intent"},
            {"subject": "third_party", "speech_act": "intent"},
            {"subject": "unknown", "speech_act": "current_state"},
            {"subject": "unknown", "speech_act": "preference"},
            {"subject": "unknown", "speech_act": "device_command"},
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                item = extracted_item(
                    should_store=False, claim_kind="question", temporal_scope="unknown",
                    category="session", key="rest", value=text, evidence_text=text, **variant,
                )
                response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response):
                    decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
                self.assertNotIn("semantic_normalizations", decision.analysis_metadata)

    def test_actual_question_or_fragment_is_not_retried_into_an_assertion(self) -> None:
        for text in ["Biraz dinlenmeli miyim?", "Bugün dışarı çıkabilir miyim", "dinlenmek"]:
            with self.subTest(text=text):
                item = extracted_item(
                    claim_kind="question", speech_act="intent", temporal_scope="current_session",
                    category="session", key="rest", value=text, evidence_text=text,
                )
                response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                with patch.object(memory_analyzer.urllib_request, "urlopen", return_value=response) as mocked:
                    decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
                self.assertEqual(mocked.call_count, 1)
                self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)

    def test_failed_semantic_review_keeps_initial_discard_without_rules_resurrection(self) -> None:
        text = "Başım ağrıyor"
        item = extracted_item(
            claim_kind="inferred", speech_act="current_state", temporal_scope="current_session",
            category="symptom", key="headache", sensitivity_domain="health", value=text, evidence_text=text,
        )
        response = FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
        with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=[response, URLError("offline")]) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
        self.assertTrue(decision.analysis_metadata["extraction_retry_failed"])
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "not_asserted")

    def test_semantic_review_cannot_bypass_evidence_validation(self) -> None:
        text = "Biraz dinlenmek istiyorum."
        initial = extracted_item(
            claim_kind="question", speech_act="intent", temporal_scope="current_session",
            category="session", key="rest", value=text, evidence_text=text,
        )
        repaired = {**initial, "claim_kind": "assertion", "value": "Saat 8'de dinlenmek istiyorum"}
        responses = [FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                     for item in [initial, repaired]]
        with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=responses):
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "unsupported_number")

    def test_mixed_clause_review_keeps_original_schema_and_matching_context(self) -> None:
        text = "Çayımı açık içerim. Yarın hava nasıl?"
        initial = extracted_item(
            should_store=False, claim_kind="question", subject="unknown", speech_act="question",
            temporal_scope="tomorrow", category="none", key="none", value="Yarın hava nasıl?",
            evidence_text="Yarın hava nasıl?",
        )
        repaired = extracted_item(value="açık", evidence_text="Çayımı açık içerim.")
        responses = [FakeResponse({"message": {"content": json.dumps({"items": [item]})}})
                     for item in [initial, repaired]]
        existing = [{"id": "temporary-1", "category": "session", "key": "walk", "value": "Bugün yürümek"}]
        recent = [{"role": "assistant", "content": "Nasıl yardımcı olabilirim?"}]
        with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=responses) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at, recent, existing)[0]
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.LONG_TERM)
        self.assertEqual(decision.analysis_metadata["extraction_retry_reasons"], ["mixed_question_clauses"])
        review_request = mocked.call_args_list[1].args[0]
        payload = json.loads(review_request.data.decode("utf-8"))
        self.assertEqual(payload["format"], memory_analyzer.MemoryExtraction.model_json_schema())
        self.assertIn("REVIEW:", payload["messages"][0]["content"])
        review_data = json.loads(payload["messages"][1]["content"])
        self.assertEqual(review_data["message_to_analyze"], text)
        self.assertEqual(review_data["existing_memories"], existing)
        self.assertEqual(review_data["recent_messages"], recent)
        self.assertEqual(review_data["contract_violations"], ["mixed_question_clauses"])
        self.assertEqual(
            review_data["previous_extraction"]["items"][0]["claim_kind"],
            "question",
        )

    def test_personal_semantic_categories_are_policy_mapped(self) -> None:
        for category in ("preferred_name", "communication_preference"):
            with self.subTest(category=category):
                text = "Bana Ahmet Bey diye hitap et"
                decision = self.analyze_mocked(extracted_item(
                    category=category,
                    key="form_of_address",
                    value="Ahmet Bey",
                    evidence_text=text,
                    sensitivity_domain="none",
                ), text=text)[0]
                self.assertEqual(decision.sensitivity, models.Sensitivity.PERSONAL)

    def test_location_category_requires_protected_location_evidence(self) -> None:
        ordinary = "Gözlüğümü komodinin üstünde tutarım"
        ordinary_decision = self.analyze_mocked(extracted_item(
            speech_act="habit",
            category="location",
            key="glasses_location",
            value=ordinary,
            evidence_text=ordinary,
            sensitivity_domain="none",
        ), text=ordinary)[0]
        self.assertEqual(ordinary_decision.sensitivity, models.Sensitivity.NORMAL)

        address = "Adresim Bahar Sokak 12"
        address_decision = self.analyze_mocked(extracted_item(
            speech_act="profile_fact",
            category="location",
            key="home_address",
            value=address,
            evidence_text=address,
            sensitivity_domain="none",
        ), text=address)[0]
        self.assertEqual(address_decision.sensitivity, models.Sensitivity.LOCATION)

    def test_context_guard_recovers_time_bound_reply_discarded_by_model(self) -> None:
        decisions = self.analyze_mocked_items(
            [
                extracted_item(
                    should_store=False,
                    speech_act="courtesy",
                    temporal_scope="today",
                    category="none",
                    key="none",
                    value="Evet, bugün yapalım",
                )
            ],
            text="Evet, bugün yapalım",
            recent_messages=[
                {
                    "role": "assistant",
                    "content": "Yürüyüşe ne zaman çıkmak istersiniz?",
                }
            ],
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SHORT_TERM)
        self.assertTrue(
            decisions[0].analysis_metadata[
                "contextual_intent_recovered_by_policy"
            ]
        )
        self.assertIn("Yürüyüşe", decisions[0].value["in_reply_to"])

    def test_context_guard_does_not_turn_device_command_into_memory(self) -> None:
        decisions = self.analyze_mocked_items(
            [
                extracted_item(
                    should_store=False,
                    speech_act="device_command",
                    temporal_scope="current_session",
                    category="none",
                    key="none",
                    value="Şimdi televizyonu aç",
                )
            ],
            text="Şimdi televizyonu aç",
            recent_messages=[
                {"role": "assistant", "content": "Ne yapmak istersiniz?"}
            ],
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)

    def test_context_reply_with_contradictory_question_label_is_temporary(self) -> None:
        text = "Evet bugün yapalım"
        decisions = self.analyze_mocked_items([
            extracted_item(
                should_store=False, claim_kind="contextual_reply", subject="unknown",
                speech_act="question", temporal_scope="today", category="none", key="none",
                value=text, evidence_text=text,
            )
        ], text=text, recent_messages=[{
            "role": "assistant", "content": "Bugün yürüyüşe çıkmak ister misiniz?",
        }])
        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.SHORT_TERM)
        self.assertTrue(decisions[0].analysis_metadata["contextual_intent_recovered_by_policy"])

    def test_valid_contextual_reply_keeps_verified_prompt_in_temporary_value(self) -> None:
        prompt = "Yürüyüşe ne zaman çıkmak istersiniz?"
        text = "Evet, bugün yapalım"
        decision = self.analyze_mocked_items([
            extracted_item(
                should_store=True,
                claim_kind="contextual_reply",
                speech_act="intent",
                temporal_scope="today",
                category="session",
                key="walk",
                value=text,
                evidence_text=text,
            )
        ], text=text, recent_messages=[{
            "role": "assistant", "content": prompt,
        }])[0]

        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SHORT_TERM)
        self.assertEqual(decision.value["type"], "contextual_user_intent")
        self.assertEqual(decision.value["in_reply_to"], prompt)

    def test_protected_contextual_prompt_auto_applies(self) -> None:
        prompt = "Tansiyon ilacınızı bugün aldınız mı?"
        text = "Evet, bugün aldım"
        decision = self.analyze_mocked_items([
            extracted_item(
                should_store=True,
                claim_kind="contextual_reply",
                speech_act="current_state",
                temporal_scope="today",
                sensitivity_domain="none",
                category="session",
                key="medication_status",
                value=text,
                evidence_text=text,
            )
        ], text=text, recent_messages=[{
            "role": "assistant", "content": prompt,
        }])[0]

        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.sensitivity, models.Sensitivity.HEALTH)
        self.assertEqual(decision.value["in_reply_to"], prompt)
        self.assertIsNotNone(decision.expires_at)
        self.assertTrue(decision.analysis_metadata["contextual_prompt_verified"])
        self.assertEqual(
            decision.analysis_metadata["protected_context_hint"],
            "health",
        )

    def test_contextual_reply_requires_direct_preceding_assistant_question(self) -> None:
        text = "Evet bugün yapalım"
        item = extracted_item(
            claim_kind="contextual_reply",
            speech_act="intent",
            temporal_scope="today",
            category="session",
            key="outing",
            value=text,
            evidence_text=text,
        )
        contexts = [
            [],
            [{"role": "assistant", "content": "Elbette."}],
            [
                {
                    "role": "assistant",
                    "content": "Bugün yürüyüşe çıkmak ister misiniz?",
                },
                {"role": "user", "content": "Önce düşüneyim."},
            ],
        ]
        for recent_messages in contexts:
            with self.subTest(recent_messages=recent_messages):
                decision = self.analyze_mocked_items(
                    [item],
                    text=text,
                    recent_messages=recent_messages,
                )[0]
                self.assertEqual(
                    decision.memory_type,
                    models.CandidateMemoryType.DISCARD,
                )
                self.assertEqual(
                    decision.analysis_metadata["evidence_guard"],
                    "contextual_reply_without_prompt",
                )

    def test_semantic_review_recovers_self_contained_statement_from_contextual_label(self) -> None:
        text = "Şimdi odama geçiyorum."
        initial = extracted_item(
            should_store=False,
            claim_kind="contextual_reply",
            speech_act="intent",
            temporal_scope="current_session",
            category="session",
            key="move_to_room",
            value=text,
            evidence_text=text,
        )
        repaired = {
            **initial,
            "should_store": True,
            "claim_kind": "assertion",
        }
        responses = [
            FakeResponse({
                "message": {"content": json.dumps({"items": [item]})}
            })
            for item in (initial, repaired)
        ]
        with patch.object(
            memory_analyzer.urllib_request,
            "urlopen",
            side_effect=responses,
        ) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]

        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(
            decision.memory_type,
            models.CandidateMemoryType.SHORT_TERM,
        )
        self.assertEqual(decision.value["description"], text)
        self.assertEqual(
            decision.analysis_metadata["extraction_retry_reasons"],
            ["contextual_reply_without_prompt"],
        )

    def test_context_guard_does_not_store_explicit_memory_instruction(self) -> None:
        decisions = self.analyze_mocked_items(
            [
                extracted_item(
                    should_store=False,
                    speech_act="memory_instruction",
                    temporal_scope="today",
                    category="none",
                    key="none",
                    value="Bugün bunu hafızaya kaydet",
                )
            ],
            text="Bugün bunu hafızaya kaydet",
            recent_messages=[
                {"role": "assistant", "content": "Ne yapmak istersiniz?"}
            ],
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)

    def test_question_and_device_command_are_discarded(self) -> None:
        for speech_act in ("question", "device_command"):
            with self.subTest(speech_act=speech_act):
                decisions = self.analyze_mocked(
                    extracted_item(
                        should_store=False,
                        subject="unknown",
                        speech_act=speech_act,
                        temporal_scope="today",
                        category="none",
                        key="none",
                        value="not reusable",
                    )
                )
                self.assertEqual(
                    decisions[0].memory_type,
                    models.CandidateMemoryType.DISCARD,
                )

    def test_third_party_transient_health_statement_is_discarded(self) -> None:
        decisions = self.analyze_mocked(
            extracted_item(
                should_store=False,
                subject="third_party",
                speech_act="current_state",
                temporal_scope="today",
                sensitivity_domain="health",
                category="none",
                key="none",
                value="Kızımın başı ağrıyor",
            ),
            text="Kızım bugün ‘başım ağrıyor’ dedi",
        )

        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)

    def test_misclassified_questions_and_fragment_never_become_facts(self) -> None:
        for text in [
            "her sabah kaçta tenis oynarım", "Her gün kaçta tenis oynarım?",
            "Tansiyon ilacımı ne zaman alıyorum", "Bugün dışarı çıkabilir miyim",
            "tenis", "Her sabah tenis oynuyor muyum",
        ]:
            with self.subTest(text=text):
                decisions = self.analyze_mocked(extracted_item(
                    claim_kind="assertion", speech_act="profile_fact", category="routine",
                    key="tennis_wake_time", value="Her sabah tenis oynarım", evidence_text=text,
                ), text=text)
                self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)
                self.assertIn("evidence_guard", decisions[0].analysis_metadata)

    def test_invented_quote_time_and_frequency_are_rejected(self) -> None:
        text = "Her cumartesi sabah saat 7'de tenis oynarım"
        for overrides in [
            {"evidence_text": "Her sabah tenis oynarım"},
            {"value": "Her sabah saat 7'de tenis oynarım"},
            {"value": "Her cumartesi saat 8'de tenis oynarım"},
            {"value": "Her pazar saat 7'de tenis oynarım"},
            {"claim_kind": "inferred"},
            {"claim_kind": "ambiguous"},
        ]:
            with self.subTest(overrides=overrides):
                item = extracted_item(speech_act="habit", category="routine", value=text, evidence_text=text)
                item.update(overrides)
                decisions = self.analyze_mocked(item, text=text)
                self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)

    def test_explicit_assertion_survives_question_in_separate_clause(self) -> None:
        assertion = "Çayımı açık içerim"
        text = f"{assertion}. Yarın hava nasıl?"
        for evidence in [assertion, assertion + "."]:
            with self.subTest(evidence=evidence):
                decisions = self.analyze_mocked(extracted_item(value="açık", evidence_text=evidence), text=text)
                self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.LONG_TERM)

    def test_fragment_of_question_cannot_be_used_as_assertion_evidence(self) -> None:
        decisions = self.analyze_mocked(extracted_item(
            value="Her sabah tenis oynarım", evidence_text="her sabah",
        ), text="her sabah kaçta tenis oynarım")
        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)

    def test_fragment_misclassified_as_contextual_intent_cannot_be_stored(self) -> None:
        decisions = self.analyze_mocked_items([
            extracted_item(
                claim_kind="contextual_reply", speech_act="intent", temporal_scope="today",
                value="Bugün tenis oynamak istiyor", evidence_text="tenis",
            )
        ], text="tenis", recent_messages=[{
            "role": "assistant", "content": "Bunu her gün yaptığınızı mı söylüyorsunuz?",
        }])
        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)
        self.assertEqual(decisions[0].analysis_metadata["evidence_guard"], "fragment")

    def test_rule_fallback_cannot_resurrect_a_question(self) -> None:
        with patch.object(memory_analyzer.urllib_request, "urlopen", side_effect=URLError("offline")):
            decisions = memory_analyzer.analyze_message("Her gün kaçta ilaç alıyorum", self.occurred_at)
        self.assertEqual(decisions[0].memory_type, models.CandidateMemoryType.DISCARD)

    def test_default_single_pass_uses_verified_evidence_when_value_is_ungrounded(self) -> None:
        text = "Artık sabah 7'de değil, 8'de kalkıyorum"
        item = extracted_item(
            speech_act="habit",
            category="routine",
            key="morning_wake_time",
            value="Sabah saat 8'de kalkarım",
            evidence_text=text,
        )
        response = FakeResponse({
            "message": {"content": json.dumps({"items": [item]})}
        })
        with patch.dict(os.environ, {"MEMORY_SEMANTIC_RETRY": "false"}), patch.object(
            memory_analyzer.urllib_request,
            "urlopen",
            return_value=response,
        ) as mocked:
            decision = memory_analyzer.analyze_message(text, self.occurred_at)[0]

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(decision.scope, models.MemoryScope.PROFILE)
        self.assertEqual(decision.value, text)
        self.assertEqual(
            decision.analysis_metadata["value_grounding_fallback"],
            "unsupported_value",
        )

    def test_default_single_pass_normalizes_complete_transient_assertion(self) -> None:
        for claim_kind, text in [
            ("contextual_reply", "Şimdi biraz dinleneceğim"),
            ("question", "Birazdan mutfağa gideceğim"),
        ]:
            with self.subTest(claim_kind=claim_kind), patch.dict(
                os.environ, {"MEMORY_SEMANTIC_RETRY": "false"}
            ):
                decision = self.analyze_mocked(extracted_item(
                    should_store=False,
                    claim_kind=claim_kind,
                    subject="user",
                    speech_act="intent",
                    temporal_scope="current_session",
                    category="none",
                    key="none",
                    value=text,
                    evidence_text=text,
                ), text=text)[0]
                self.assertEqual(
                    decision.memory_type,
                    models.CandidateMemoryType.SHORT_TERM,
                )
                self.assertEqual(decision.scope, models.MemoryScope.SESSION)
                self.assertEqual(
                    decision.analysis_metadata["contract_normalization"]["reason"],
                    "complete_literal_transient_assertion",
                )

    def test_storage_scope_is_independent_from_health_sensitivity(self) -> None:
        cases = [
            (
                "Dün banyoda düştüm",
                extracted_item(
                    speech_act="episode", temporal_scope="unknown",
                    sensitivity_domain="health", category="incident",
                    key="bathroom_fall", value="Dün banyoda düştüm",
                ),
                models.MemoryScope.EPISODE,
            ),
            (
                "Başım ağrıyor",
                extracted_item(
                    speech_act="current_state", temporal_scope="current_session",
                    sensitivity_domain="health", category="symptom",
                    key="headache", value="Başım ağrıyor",
                ),
                models.MemoryScope.SESSION,
            ),
            (
                "Astım hastasıyım",
                extracted_item(
                    speech_act="profile_fact", temporal_scope="persistent",
                    sensitivity_domain="health", category="condition",
                    key="asthma", value="Astım hastasıyım",
                ),
                models.MemoryScope.PROFILE,
            ),
        ]
        for text, item, expected_scope in cases:
            with self.subTest(scope=expected_scope):
                decision = self.analyze_mocked(item, text=text)[0]
                self.assertEqual(decision.sensitivity, models.Sensitivity.HEALTH)
                self.assertEqual(decision.scope, expected_scope)

    def test_durable_health_accessibility_state_is_profile_scoped(self) -> None:
        text = "İşitme cihazım olmadan konuşmaları anlamakta zorlanıyorum"
        decision = self.analyze_mocked(extracted_item(
            speech_act="current_state",
            temporal_scope="unknown",
            sensitivity_domain="health",
            category="communication",
            key="hearing_accessibility",
            value=text,
        ), text=text)[0]
        self.assertEqual(decision.memory_type, models.CandidateMemoryType.SENSITIVE)
        self.assertEqual(decision.scope, models.MemoryScope.PROFILE)
        self.assertIsNone(decision.expires_at)

    def test_high_similarity_health_quote_uses_literal_source_only(self) -> None:
        text = "Geçen yıl kalça ameliyatı oldum"
        item = extracted_item(
            speech_act="profile_fact",
            temporal_scope="unknown",
            sensitivity_domain="health",
            category="episode",
            key="past_surgery",
            value="Geçen yıl kalça ameliżyati oldum",
            evidence_text="Geçen yıl kalça ameliżyati oldum",
        )
        with patch.dict(os.environ, {"MEMORY_SEMANTIC_RETRY": "false"}):
            decision = self.analyze_mocked(item, text=text)[0]
        self.assertEqual(decision.scope, models.MemoryScope.EPISODE)
        self.assertEqual(decision.value, text)
        self.assertEqual(
            decision.analysis_metadata["evidence_quote_fallback"],
            "high_similarity_health_quote",
        )

    def test_fuzzy_quote_fallback_never_applies_to_credentials(self) -> None:
        text = "İnternet şifrem MaviEv-2026"
        decision = self.analyze_mocked(extracted_item(
            speech_act="profile_fact",
            temporal_scope="persistent",
            sensitivity_domain="credential",
            category="credential",
            key="wifi_password",
            value="İnternet şifrem MaviEx-2026",
            evidence_text="İnternet şifrem MaviEx-2026",
        ), text=text)[0]
        self.assertEqual(decision.scope, models.MemoryScope.DISCARD)
        self.assertEqual(decision.analysis_metadata["evidence_guard"], "unsupported_quote")


if __name__ == "__main__":
    unittest.main()
