import os
import json
import unittest
from datetime import timedelta
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
import models


class MemoryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = patch.dict(
            os.environ,
            {"MEMORY_ANALYZER_PROVIDER": "rules"},
        )
        self.environment.start()
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        models.Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db = self.Session()

    def tearDown(self) -> None:
        self.db.close()
        models.Base.metadata.drop_all(self.engine)
        self.engine.dispose()
        self.environment.stop()

    def test_fact_update_supersedes_old_value(self) -> None:
        first = main.create_memory_fact(
            main.FactCreate(
                user_id="ahmet-001",
                category="communication",
                key="form_of_address",
                value="Ahmet",
            ),
            self.db,
        )
        second = main.create_memory_fact(
            main.FactCreate(
                user_id="ahmet-001",
                category="communication",
                key="form_of_address",
                value="Ahmet Bey",
                verification_status=models.VerificationStatus.USER_CONFIRMED,
            ),
            self.db,
        )

        facts = self.db.query(models.MemoryFact).all()
        statuses = {fact.id: fact.status for fact in facts}
        self.assertEqual(statuses[first["fact_id"]], models.FactStatus.SUPERSEDED)
        self.assertEqual(statuses[second["fact_id"]], models.FactStatus.ACTIVE)
        self.assertEqual(second["supersedes_id"], first["fact_id"])

        context = main.build_context(
            main.ContextRequest(user_id="ahmet-001", session_id="missing"),
            self.db,
        )
        self.assertEqual(
            context["profile"]["communication"]["form_of_address"],
            "Ahmet Bey",
        )

    def test_context_profile_is_query_aware_and_keeps_pinned_fact(self) -> None:
        for category, key, value in [
            ("routine", "morning_wake_time", "Sabah saat 7'de kalkarım"),
            ("preference", "tea_style", "Çayımı açık içerim"),
            ("communication", "form_of_address", "Ahmet Bey"),
        ]:
            main.create_memory_fact(
                main.FactCreate(
                    user_id="retrieval-user",
                    category=category,
                    key=key,
                    value=value,
                ),
                self.db,
            )

        context = main.build_context(
            main.ContextRequest(
                user_id="retrieval-user",
                session_id="retrieval-session",
                query="Bu sabah kaçta kalkıyorum?",
            ),
            self.db,
        )

        self.assertIn("routine", context["profile"])
        self.assertIn("communication", context["profile"])
        self.assertNotIn("preference", context["profile"])
        self.assertEqual(context["profile_retrieval"]["eligible_fact_count"], 3)
        self.assertEqual(context["profile_retrieval"]["selected_fact_count"], 2)
        self.assertEqual(context["profile_retrieval"]["omitted_fact_count"], 1)
        self.assertEqual(len(context["memory_refs"]), 2)

    def test_context_attributes_first_person_fact_without_rewriting_value(self) -> None:
        value = "Her sabah şekersiz kahve içerim."
        main.create_memory_event(
            main.EventCreate(
                event_id="coffee-event",
                user_id="coffee-user",
                event_type="user_message",
                payload={"text": value},
            ),
            self.db,
        )
        created = main.create_memory_fact(
            main.FactCreate(
                user_id="coffee-user",
                category="habit",
                key="coffee_routine",
                value=value,
                source_event_id="coffee-event",
                confidence=0.95,
            ),
            self.db,
        )
        context = main.build_context(
            main.ContextRequest(user_id="coffee-user", session_id="new-coffee-session"),
            self.db,
        )
        self.assertEqual(context["user_id"], "coffee-user")
        self.assertEqual(context["recent_messages"], [])
        self.assertEqual(context["profile"], {"habit": {"coffee_routine": value}})
        self.assertEqual(context["profile_facts"], [{
            "fact_id": created["fact_id"],
            "owner_user_id": "coffee-user",
            "category": "habit",
            "key": "coffee_routine",
            "value": value,
            "provenance": {
                "source_event_id": "coffee-event",
                "verification_status": "unverified",
                "confidence": 0.95,
            },
        }])
        self.assertEqual(self.db.get(models.MemoryFact, created["fact_id"]).value_json,
                         {"value": value})
        self.assertEqual(context["memory_refs"], [created["fact_id"]])

    def test_attributed_provenance_does_not_expose_another_users_source(self) -> None:
        main.create_memory_event(
            main.EventCreate(
                event_id="other-owner-event",
                user_id="other-owner",
                event_type="user_message",
                payload={"text": "Başka kullanıcının gizli bilgisi"},
            ),
            self.db,
        )
        # Historical rows can have a mismatched source reference. Do not expand
        # that reference or rewrite the historical fact while building context.
        created = main.create_memory_fact(
            main.FactCreate(
                user_id="owner-user",
                category="habit",
                key="coffee",
                value={"drink": "coffee", "sugar": False},
                source_event_id="other-owner-event",
            ),
            self.db,
        )
        main.create_memory_fact(
            main.FactCreate(
                user_id="other-owner",
                category="habit",
                key="coffee",
                value="Other user's private preference",
            ),
            self.db,
        )
        context = main.build_context(
            main.ContextRequest(user_id="owner-user", session_id="owner-fresh-session"),
            self.db,
        )
        self.assertEqual(len(context["profile_facts"]), 1)
        fact = context["profile_facts"][0]
        self.assertEqual(fact["owner_user_id"], "owner-user")
        self.assertEqual(fact["value"], {"drink": "coffee", "sugar": False})
        self.assertIsNone(fact["provenance"]["source_event_id"])
        self.assertNotIn("other-owner-event", json.dumps(context))
        self.assertNotIn("Other user's private preference", json.dumps(context))
        self.assertEqual(self.db.get(models.MemoryFact, created["fact_id"]).source_event_id,
                         "other-owner-event")

    def test_attributed_context_does_not_expand_pending_or_raw_source_information(self) -> None:
        raw_text = "Her sabah kahve içerim. Tansiyon ilacım değişti."
        main.create_memory_event(
            main.EventCreate(
                event_id="mixed-source",
                user_id="mixed-user",
                event_type="user_message",
                payload={"text": raw_text},
            ),
            self.db,
        )
        main.create_memory_fact(
            main.FactCreate(
                user_id="mixed-user",
                category="habit",
                key="coffee",
                value="Her sabah kahve içerim.",
                source_event_id="mixed-source",
            ),
            self.db,
        )
        pending = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.SENSITIVE,
            category="health", key="medication_change", value="Tansiyon ilacım değişti.",
            sensitivity=models.Sensitivity.HEALTH, confidence=0.95,
            requires_confirmation=True, expires_at=None,
            reason="Hassas bilgi onay beklemeli.", analyzer_source="ollama",
        )
        with patch.object(main, "analyze_message", return_value=[pending]):
            result = main.process_interaction(
                main.InteractionProcessRequest(
                    event_id="mixed-source", user_id="mixed-user",
                    session_id="mixed-original-session", text=raw_text,
                ),
                self.db,
            )
        self.assertEqual(result["decisions"][0]["status"], "pending")
        context = main.build_context(
            main.ContextRequest(user_id="mixed-user", session_id="mixed-fresh-session"),
            self.db,
        )
        self.assertEqual(context["recent_messages"], [])
        self.assertEqual(len(context["profile_facts"]), 1)
        self.assertEqual(context["profile_facts"][0]["provenance"]["source_event_id"],
                         "mixed-source")
        self.assertNotIn("Tansiyon", json.dumps(context["profile_facts"], ensure_ascii=False))
        self.assertNotIn("source_quote", context["profile_facts"][0]["provenance"])
        self.assertEqual(context["profile"], {"habit": {"coffee": "Her sabah kahve içerim."}})

    def test_attributed_profile_budget_includes_wrappers_and_keeps_views_aligned(self) -> None:
        created = []
        for category, key, value in [
            ("communication", "form_of_address", "Ahmet Bey"),
            ("routine", "large", "x" * 180),
            ("preference", "drink", "tea"),
        ]:
            created.append(main.create_memory_fact(
                main.FactCreate(user_id="budget-user", category=category, key=key, value=value),
                self.db,
            ))
        with patch.dict(os.environ, {"MEMORY_PROFILE_MAX_TOKENS": "128"}):
            context = main.build_context(
                main.ContextRequest(user_id="budget-user", session_id="budget-session"),
                self.db,
            )
        self.assertEqual(context["profile"], {"communication": {"form_of_address": "Ahmet Bey"}})
        self.assertEqual(len(context["profile_facts"]), 1)
        self.assertEqual(context["profile_facts"][0]["fact_id"], created[0]["fact_id"])
        self.assertEqual(context["profile_retrieval"]["eligible_fact_count"], 3)
        self.assertEqual(context["profile_retrieval"]["selected_fact_count"], 1)
        self.assertEqual(context["profile_retrieval"]["omitted_fact_count"], 2)
        self.assertEqual(context["memory_refs"], [fact["fact_id"] for fact in context["profile_facts"]])
        self.assertEqual(context["memory_refs"], [item["fact_id"] for item in context["profile_retrieval"]["selected"]])
        cost = main.estimate_profile_tokens(context["profile_facts"])
        self.assertEqual(context["profile_retrieval"]["estimated_tokens"], cost)
        self.assertEqual(
            context["profile_retrieval"]["selected"][0]["estimated_tokens"],
            main.estimate_profile_tokens(context["profile_facts"][0]),
        )
        self.assertLessEqual(cost, 128)
        self.assertEqual(self.db.get(models.MemoryFact, created[1]["fact_id"]).value_json,
                         {"value": "x" * 180})

    def test_session_owner_cannot_be_changed(self) -> None:
        future = main.utc_now() + timedelta(hours=1)
        main.update_session_state(
            "session-001",
            main.SessionUpdate(
                user_id="ahmet-001",
                state=main.SessionStatePayload(
                    active_goal={"type": "outing"},
                ),
                expires_at=future,
            ),
            self.db,
        )

        with self.assertRaises(HTTPException) as captured:
            main.update_session_state(
                "session-001",
                main.SessionUpdate(
                    user_id="ayse-001",
                    state=main.SessionStatePayload(
                        active_goal={"type": "other"},
                    ),
                    expires_at=future,
                ),
                self.db,
            )
        self.assertEqual(captured.exception.status_code, 403)

        with self.assertRaises(HTTPException) as captured:
            main.build_context(
                main.ContextRequest(user_id="ayse-001", session_id="session-001"),
                self.db,
            )
        self.assertEqual(captured.exception.status_code, 403)

    def test_expired_observations_are_not_returned(self) -> None:
        now = main.utc_now()
        state = models.SessionState(
            session_id="session-001",
            user_id="ahmet-001",
            expires_at=now + timedelta(hours=1),
            state_json={
                "active_goal": {"type": "outing"},
                "temporary_observations": [
                    {
                        "type": "weather",
                        "value": {"temperature_c": 15},
                        "expires_at": (now - timedelta(minutes=1)).isoformat(),
                    },
                    {
                        "type": "weather",
                        "value": {"temperature_c": 22},
                        "expires_at": (now + timedelta(minutes=10)).isoformat(),
                    },
                ],
            },
        )
        self.db.add(state)
        self.db.commit()

        context = main.build_context(
            main.ContextRequest(user_id="ahmet-001", session_id="session-001"),
            self.db,
        )
        self.assertEqual(len(context["temporary_observations"]), 1)
        self.assertEqual(
            context["temporary_observations"][0]["value"]["temperature_c"],
            22,
        )

    def test_events_are_idempotent_and_session_scoped(self) -> None:
        future = main.utc_now() + timedelta(hours=1)
        main.update_session_state(
            "session-001",
            main.SessionUpdate(
                user_id="ahmet-001",
                state=main.SessionStatePayload(),
                expires_at=future,
            ),
            self.db,
        )
        event = main.EventCreate(
            event_id="event-001",
            user_id="ahmet-001",
            session_id="session-001",
            event_type="user_message",
            payload={"text": "Bugün dışarı çıkmak istiyorum"},
        )
        created = main.create_memory_event(event, self.db)
        duplicate = main.create_memory_event(event, self.db)

        self.assertEqual(created["status"], "created")
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(self.db.query(models.MemoryEvent).count(), 1)

        with self.assertRaises(HTTPException) as captured:
            main.create_memory_event(
                main.EventCreate(
                    event_id="event-002",
                    user_id="ayse-001",
                    session_id="session-001",
                    event_type="user_message",
                    payload={"text": "test"},
                ),
                self.db,
            )
        self.assertEqual(captured.exception.status_code, 403)

    def test_conversation_window_is_bounded_and_chronological(self) -> None:
        now = main.utc_now()
        with patch.dict(
            os.environ,
            {
                "MEMORY_WINDOW_MAX_MESSAGES": "2",
                "MEMORY_WINDOW_MAX_TOKENS": "2000",
            },
        ):
            for index, (role, content) in enumerate(
                [
                    (models.ConversationRole.USER, "Birinci mesaj"),
                    (models.ConversationRole.ASSISTANT, "İkinci mesaj"),
                    (models.ConversationRole.USER, "Üçüncü mesaj"),
                ]
            ):
                main.create_conversation_message(
                    "window-session",
                    main.ConversationMessageCreate(
                        message_id=f"window-{index}",
                        user_id="ahmet-001",
                        role=role,
                        content=content,
                        occurred_at=now - timedelta(minutes=3 - index),
                    ),
                    self.db,
                )

            context = main.build_context(
                main.ContextRequest(
                    user_id="ahmet-001",
                    session_id="window-session",
                ),
                self.db,
            )

        self.assertEqual(
            [message["content"] for message in context["recent_messages"]],
            ["İkinci mesaj", "Üçüncü mesaj"],
        )
        self.assertEqual(context["conversation_window"]["message_count"], 2)
        self.assertEqual(context["conversation_window"]["max_messages"], 2)
        self.assertEqual(context["message_refs"], ["window-1", "window-2"])

    def test_conversation_window_respects_token_budget(self) -> None:
        now = main.utc_now()
        for index in range(2):
            main.create_conversation_message(
                "token-session",
                main.ConversationMessageCreate(
                    message_id=f"token-{index}",
                    user_id="ahmet-001",
                    role=models.ConversationRole.USER,
                    content=str(index) * 300,
                    occurred_at=now - timedelta(minutes=2 - index),
                ),
                self.db,
            )

        with patch.dict(
            os.environ,
            {
                "MEMORY_WINDOW_MAX_MESSAGES": "10",
                "MEMORY_WINDOW_MAX_TOKENS": "128",
            },
        ):
            window = main.list_conversation_messages(
                "token-session",
                "ahmet-001",
                self.db,
            )

        self.assertEqual(window["message_count"], 1)
        self.assertLessEqual(window["estimated_tokens"], 128)
        self.assertEqual(window["messages"][0]["message_id"], "token-1")

    def test_expired_conversation_messages_are_not_returned(self) -> None:
        now = main.utc_now()
        self.db.add_all(
            [
                models.ConversationMessage(
                    id="expired-message",
                    user_id="ahmet-001",
                    session_id="expiry-session",
                    role=models.ConversationRole.USER,
                    content="Artık görünmemeli",
                    estimated_tokens=10,
                    occurred_at=now - timedelta(days=2),
                    expires_at=now - timedelta(days=1),
                ),
                models.ConversationMessage(
                    id="live-message",
                    user_id="ahmet-001",
                    session_id="expiry-session",
                    role=models.ConversationRole.ASSISTANT,
                    content="Hâlâ geçerli",
                    estimated_tokens=10,
                    occurred_at=now - timedelta(minutes=1),
                    expires_at=now + timedelta(hours=1),
                ),
            ]
        )
        self.db.commit()

        context = main.build_context(
            main.ContextRequest(
                user_id="ahmet-001",
                session_id="expiry-session",
            ),
            self.db,
        )

        self.assertEqual(
            context["recent_messages"],
            [{"role": "assistant", "content": "Hâlâ geçerli"}],
        )

    def test_conversation_session_owner_is_enforced(self) -> None:
        main.create_conversation_message(
            "owned-session",
            main.ConversationMessageCreate(
                message_id="owned-1",
                user_id="ahmet-001",
                role=models.ConversationRole.USER,
                content="Ahmet'in mesajı",
            ),
            self.db,
        )

        with self.assertRaises(HTTPException) as captured:
            main.create_conversation_message(
                "owned-session",
                main.ConversationMessageCreate(
                    message_id="owned-2",
                    user_id="ayse-001",
                    role=models.ConversationRole.ASSISTANT,
                    content="Başka kullanıcı mesajı",
                ),
                self.db,
            )

        self.assertEqual(captured.exception.status_code, 403)

    def test_interaction_uses_stored_window_and_saves_user_message_once(self) -> None:
        main.create_conversation_message(
            "context-session",
            main.ConversationMessageCreate(
                message_id="assistant-context-1",
                user_id="ahmet-001",
                role=models.ConversationRole.ASSISTANT,
                content="Yürüyüşe ne zaman çıkmak istersiniz?",
            ),
            self.db,
        )
        discard = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.DISCARD,
            category=None,
            key=None,
            value=None,
            sensitivity=models.Sensitivity.NORMAL,
            confidence=0.9,
            requires_confirmation=False,
            expires_at=None,
            reason="test",
        )
        request = main.InteractionProcessRequest(
            event_id="stored-context-interaction",
            user_id="ahmet-001",
            session_id="context-session",
            text="Evet, bugün yapalım",
        )

        with patch.object(
            main,
            "analyze_message",
            return_value=[discard],
        ) as analyzer:
            main.process_interaction(request, self.db)
            main.process_interaction(request, self.db)

        self.assertEqual(analyzer.call_count, 1)
        self.assertEqual(
            analyzer.call_args.args[2],
            [
                {
                    "role": "assistant",
                    "content": "Yürüyüşe ne zaman çıkmak istersiniz?",
                }
            ],
        )
        self.assertEqual(
            self.db.query(models.ConversationMessage)
            .filter(models.ConversationMessage.session_id == "context-session")
            .count(),
            2,
        )

    def test_interaction_routes_outing_to_short_term_and_is_idempotent(self) -> None:
        request = main.InteractionProcessRequest(
            event_id="interaction-short-001",
            user_id="ahmet-001",
            session_id="session-short-001",
            text="Bugün dışarı çıkmak istiyorum",
            occurred_at=main.utc_now(),
        )
        first = main.process_interaction(request, self.db)
        second = main.process_interaction(request, self.db)

        self.assertEqual(first["decisions"][0]["memory_type"], "short_term")
        self.assertEqual(first["decisions"][0]["status"], "auto_applied")
        self.assertEqual(first["decisions"][0]["analyzer_source"], "rules")
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(self.db.query(models.MemoryEvent).count(), 1)
        self.assertEqual(self.db.query(models.MemoryCandidate).count(), 1)

        context = main.build_context(
            main.ContextRequest(
                user_id="ahmet-001",
                session_id="session-short-001",
            ),
            self.db,
        )
        self.assertEqual(context["session"]["type"], "outing")

    def test_interaction_auto_applies_explicit_long_term_preference(self) -> None:
        result = main.process_interaction(
            main.InteractionProcessRequest(
                event_id="interaction-long-001",
                user_id="ahmet-001",
                session_id="session-long-001",
                text="Bana bundan sonra Ahmet Bey diye hitap et",
            ),
            self.db,
        )

        self.assertEqual(result["decisions"][0]["memory_type"], "long_term")
        self.assertEqual(result["decisions"][0]["status"], "auto_applied")
        stored_fact = self.db.query(models.MemoryFact).one()
        self.assertEqual(
            stored_fact.verification_status,
            models.VerificationStatus.UNVERIFIED,
        )
        context = main.build_context(
            main.ContextRequest(
                user_id="ahmet-001",
                session_id="session-long-001",
            ),
            self.db,
        )
        self.assertEqual(
            context["profile"]["communication"]["form_of_address"],
            "Ahmet Bey",
        )

    def test_explicit_memory_correction_supersedes_alias_slot(self) -> None:
        existing = main.create_memory_fact(
            main.FactCreate(
                user_id="ahmet-001",
                category="routine",
                key="morning_wake_up_time",
                value="Sabah saat 7'de kalkarım",
            ),
            self.db,
        )
        corrected = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.LONG_TERM,
            category="behavior",
            key="wake_up_hour",
            value="Sabah saat 8'de kalkıyorum",
            sensitivity=models.Sensitivity.NORMAL,
            confidence=0.95,
            requires_confirmation=False,
            expires_at=None,
            reason="Kullanıcı rutinini güncelledi.",
            analyzer_source="ollama",
            analysis_metadata={"relation_to_existing": "update"},
        )

        with patch.object(main, "analyze_message", return_value=[corrected]):
            result = main.process_interaction(
                main.InteractionProcessRequest(
                    event_id="correction-001",
                    user_id="ahmet-001",
                    session_id="correction-session",
                    text="Artık sabah saat 8'de kalkıyorum",
                ),
                self.db,
            )

        candidate = result["decisions"][0]
        self.assertEqual(candidate["consolidation_action"], "supersede")
        self.assertEqual(candidate["consolidates_fact_id"], existing["fact_id"])
        self.assertEqual(candidate["status"], "auto_applied")
        old_fact = self.db.get(models.MemoryFact, existing["fact_id"])
        new_fact = self.db.get(models.MemoryFact, candidate["applied_ref"])
        self.assertEqual(old_fact.status, models.FactStatus.SUPERSEDED)
        self.assertEqual(new_fact.status, models.FactStatus.ACTIVE)
        self.assertEqual(new_fact.key, "morning_wake_up_time")

    def test_ambiguous_memory_conflict_waits_for_confirmation(self) -> None:
        existing = main.create_memory_fact(
            main.FactCreate(
                user_id="ahmet-001",
                category="routine",
                key="morning_wake_up_time",
                value="Sabah saat 7'de kalkarım",
            ),
            self.db,
        )
        conflicting = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.LONG_TERM,
            category="routine",
            key="morning_wake_time",
            value="Sabah saat 8'de kalkıyorum",
            sensitivity=models.Sensitivity.NORMAL,
            confidence=0.9,
            requires_confirmation=False,
            expires_at=None,
            reason="Aynı rutin için farklı saat çıkarıldı.",
            analyzer_source="ollama",
        )

        with patch.object(main, "analyze_message", return_value=[conflicting]):
            result = main.process_interaction(
                main.InteractionProcessRequest(
                    event_id="conflict-001",
                    user_id="ahmet-001",
                    session_id="conflict-session",
                    text="Sabah saat 8'de kalkıyorum",
                ),
                self.db,
            )

        candidate = result["decisions"][0]
        self.assertEqual(
            candidate["consolidation_action"],
            "conflict_requires_confirmation",
        )
        self.assertEqual(candidate["status"], "pending")
        self.assertTrue(candidate["requires_confirmation"])
        self.assertEqual(self.db.query(models.MemoryFact).count(), 1)

        confirmed = main.confirm_memory_candidate(
            candidate["candidate_id"],
            main.CandidateDecisionRequest(user_id="ahmet-001"),
            self.db,
        )
        self.assertEqual(confirmed["status"], "confirmed")
        old_fact = self.db.get(models.MemoryFact, existing["fact_id"])
        self.assertEqual(old_fact.status, models.FactStatus.SUPERSEDED)
        self.assertEqual(self.db.query(models.MemoryFact).count(), 2)

    def test_sensitive_interaction_waits_for_confirmation(self) -> None:
        processed = main.process_interaction(
            main.InteractionProcessRequest(
                event_id="interaction-sensitive-001",
                user_id="ahmet-001",
                session_id="session-sensitive-001",
                text="Doktor tansiyon ilacımı değiştirdi",
            ),
            self.db,
        )
        decision = processed["decisions"][0]

        self.assertEqual(decision["memory_type"], "sensitive")
        self.assertEqual(decision["status"], "pending")
        self.assertEqual(self.db.query(models.MemoryFact).count(), 0)

        confirmed = main.confirm_memory_candidate(
            decision["candidate_id"],
            main.CandidateDecisionRequest(user_id="ahmet-001"),
            self.db,
        )
        self.assertEqual(confirmed["status"], "confirmed")
        self.assertIsNotNone(confirmed["applied_ref"])
        self.assertEqual(self.db.query(models.MemoryFact).count(), 1)

    def test_service_policy_blocks_unconfirmed_sensitive_long_term(self) -> None:
        unsafe_decision = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.LONG_TERM,
            category="health",
            key="condition",
            value="kalp rahatsızlığı",
            sensitivity=models.Sensitivity.HEALTH,
            confidence=0.95,
            requires_confirmation=False,
            expires_at=None,
            reason="Model hassas veriyi yanlışlıkla otomatik seçti.",
            analyzer_source="ollama",
        )
        with patch.object(main, "analyze_message", return_value=[unsafe_decision]):
            result = main.process_interaction(
                main.InteractionProcessRequest(
                    event_id="interaction-policy-001",
                    user_id="ahmet-001",
                    session_id="session-policy-001",
                    text="Kalbimle ilgili bir rahatsızlığım var",
                ),
                self.db,
            )

        self.assertEqual(result["decisions"][0]["status"], "pending")
        self.assertTrue(result["decisions"][0]["requires_confirmation"])
        self.assertEqual(self.db.query(models.MemoryFact).count(), 0)

    def test_memories_can_be_listed_and_deleted(self) -> None:
        created = main.create_memory_fact(
            main.FactCreate(
                user_id="ahmet-001",
                category="preferences",
                key="news",
                value="evening",
            ),
            self.db,
        )
        listed = main.list_user_memories(
            "ahmet-001",
            include_inactive=False,
            limit=100,
            offset=0,
            db=self.db,
        )
        self.assertEqual(len(listed["items"]), 1)

        response = main.delete_memory(
            created["fact_id"],
            user_id="ahmet-001",
            db=self.db,
        )
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.db.query(models.MemoryFact).count(), 0)


if __name__ == "__main__":
    unittest.main()
