import unittest

from emergency_orchestrator import (
    actionable_emergency_contact,
    canonical_phone,
    emergency_reason,
    evaluate_emergency,
)


def context_with_contact(*, verification_status: str = "user_asserted") -> dict:
    return {
        "profile_facts": [{
            "fact_id": "emergency-1",
            "owner_user_id": "user-1",
            "category": "emergency_contact",
            "key": "primary_emergency_contact",
            "value": "Acil durumda kızım Ayşe'yi 0555 123 45 67 numarasından ara",
            "provenance": {
                "verification_status": verification_status,
                "confidence": 1.0,
                "source_event_id": "event-1",
            },
        }],
        "recent_messages": [],
    }


class EmergencyOrchestratorTest(unittest.TestCase):
    def test_high_signal_current_emergencies_are_detected(self) -> None:
        self.assertEqual(
            emergency_reason("Banyoda düştüm, her yer kan oldu"),
            "severe_bleeding",
        )
        self.assertEqual(
            emergency_reason("Nefes alamıyorum"),
            "breathing_difficulty",
        )
        self.assertEqual(
            emergency_reason("Banyoda kaydım, kafamı vurdum, ne yapacağım?"),
            "fall_with_head_impact",
        )
        self.assertEqual(
            emergency_reason("Başımı dolaba çarptım"),
            "head_impact",
        )
        self.assertEqual(
            emergency_reason("Bir kolumu kaldıramıyorum"),
            "stroke_warning_sign",
        )
        self.assertEqual(
            emergency_reason("Nöbet geçiriyorum"),
            "seizure",
        )
        self.assertEqual(
            emergency_reason("Lütfen ambulans çağır"),
            "explicit_emergency_help_request",
        )

    def test_past_fall_and_non_emergency_health_text_do_not_trigger(self) -> None:
        self.assertIsNone(emergency_reason("Dün banyoda düştüm"))
        self.assertIsNone(emergency_reason("Başım ağrıyor"))
        self.assertIsNone(emergency_reason("Düşmedim ve kanama yok"))
        self.assertIsNone(emergency_reason("Dün kafamı vurdum"))
        self.assertIsNone(emergency_reason("Kafamı vurmadım"))
        self.assertEqual(
            emergency_reason("Kanama yok ama nefes alamıyorum"),
            "breathing_difficulty",
        )

    def test_bleeding_follow_up_uses_recent_fall_context(self) -> None:
        reason = emergency_reason(
            "Kolum kanıyor",
            [{"role": "user", "content": "Banyoda düştüm"}],
        )
        self.assertEqual(reason, "fall_with_bleeding")

    def test_post_fall_warning_and_inability_to_stand_use_recent_context(self) -> None:
        recent_fall = [{"role": "user", "content": "Banyoda düştüm"}]
        self.assertEqual(
            emergency_reason("Başım dönüyor", recent_fall),
            "post_fall_neurological_warning",
        )
        self.assertEqual(
            emergency_reason("Yerden kalkamıyorum", recent_fall),
            "fall_unable_to_stand",
        )

    def test_user_asserted_contact_is_callable_and_phone_is_normalized(self) -> None:
        self.assertEqual(canonical_phone("0555 123 45 67"), "+905551234567")
        contact = actionable_emergency_contact(context_with_contact())
        self.assertIsNotNone(contact)
        self.assertEqual(contact.name, "Ayşe")
        self.assertEqual(contact.masked_phone, "+9055 *** ** 67")
        self.assertIsNone(actionable_emergency_contact(
            context_with_contact(verification_status="unverified")
        ))

    def test_action_is_explicitly_simulated_and_not_repeated(self) -> None:
        first = evaluate_emergency("Her yer kan oldu", context_with_contact())
        self.assertEqual(first.action, "call_simulated")
        self.assertTrue(first.simulated)
        self.assertIn("gerçek arama yapmaz", first.reply)

        repeated = evaluate_emergency(
            "Kan durmuyor",
            context_with_contact(),
            action_already_started=True,
        )
        self.assertEqual(repeated.action, "already_active")

    def test_missing_actionable_contact_never_claims_a_call(self) -> None:
        action = evaluate_emergency(
            "Her yer kan oldu",
            context_with_contact(verification_status="unverified"),
        )
        self.assertEqual(action.action, "no_emergency_contact")
        self.assertNotIn("başlatıldı", action.reply)


if __name__ == "__main__":
    unittest.main()
