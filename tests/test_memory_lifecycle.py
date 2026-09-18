"""Memory-only integration checks; clocks and extraction are controlled here."""

import os
import unittest
from dataclasses import replace
from datetime import timedelta
from unittest.mock import patch

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import main
import memory_analyzer
import models


class MemoryLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"MEMORY_ANALYZER_PROVIDER": "rules"})
        self.environment.start()
        self.engine = create_engine("sqlite://")
        models.Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.start = main.utc_now()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()
        self.environment.stop()

    def temporary(self, key, value, *, minutes=60, user="owner", session="session"):
        return main.create_temporary_memory(main.TemporaryMemoryCreate(
            user_id=user, session_id=session, key=key, value=value,
            occurred_at=self.start, expires_at=self.start + timedelta(minutes=minutes),
        ), self.db)["temporary_memory"]

    def context(self, *, now=None, session="session"):
        with patch.object(main, "utc_now", return_value=now or self.start + timedelta(seconds=1)):
            return main.build_context(main.ContextRequest(
                user_id="owner", session_id=session,
            ), self.db)

    def decision(self, kind, key, value, *, sensitive=False, expiry=None):
        return main.MemoryDecision(
            memory_type=kind, category="health" if sensitive else "session", key=key,
            value=value, sensitivity=models.Sensitivity.HEALTH if sensitive else models.Sensitivity.NORMAL,
            confidence=0.95,expires_at=expiry,
            reason="Controlled extraction", analyzer_source="ollama",
        )

    def ingest(self, decisions, *, event="event", text="Kullanıcı beyanı."):
        with patch.object(main, "analyze_message", return_value=decisions):
            return main.process_interaction(main.InteractionProcessRequest(
                event_id=event, user_id="owner", session_id="session", text=text,
                occurred_at=self.start,
            ), self.db)

    def test_independent_slots_replace_only_same_concept(self):
        first = self.temporary("outing", {"description": "Dışarı çıkmak"})
        news = self.temporary("news", {"description": "Haber izlemek"})
        replacement = self.temporary("outing", {"description": "Yarın dışarı çıkmak"}, minutes=120)
        rows = self.context()["temporary_memories"]
        self.assertEqual({row["memory_id"] for row in rows}, {news["memory_id"], replacement["memory_id"]})
        self.assertEqual(self.db.get(models.TemporaryMemory, first["memory_id"]).status,
                         models.FactStatus.SUPERSEDED)
        self.assertEqual(self.db.query(models.TemporaryMemory).count(), 3)

    def test_each_item_expires_without_resurrecting_legacy_goal(self):
        main.update_session_state("session", main.SessionUpdate(
            user_id="owner", state=main.SessionStatePayload(active_goal={"description": "Legacy goal"}),
            expires_at=self.start + timedelta(days=1),
        ), self.db)
        self.temporary("short", "one", minutes=1)
        live = self.temporary("long", "two", minutes=2)
        context = self.context(now=self.start + timedelta(minutes=1))
        self.assertEqual([row["memory_id"] for row in context["temporary_memories"]], [live["memory_id"]])
        context = self.context(now=self.start + timedelta(minutes=2))
        self.assertEqual(context["temporary_memories"], [])
        self.assertEqual(context["session"], {})

    def test_unrelated_analyzer_intents_cannot_overwrite_on_model_key_collision(self):
        expiry = self.start + timedelta(hours=1)
        outing = self.decision(
            models.CandidateMemoryType.SHORT_TERM,
            "outdoor_activity_intent",
            {"description": "Bug\u00fcn d\u0131\u015far\u0131 \u00e7\u0131kmak istiyorum"},
            expiry=expiry,
        )
        news = self.decision(
            models.CandidateMemoryType.SHORT_TERM,
            "outdoor_activity_intent",
            {"description": "Bu ak\u015fam haberleri izlemek istiyorum"},
            expiry=expiry,
        )

        first = self.ingest([outing], event="collision-outing", text="Bug\u00fcn d\u0131\u015far\u0131 \u00e7\u0131kmak istiyorum.")
        news = replace(
            news,
            analysis_metadata={
                "matched_memory_id": first["decisions"][0]["applied_ref"],
                "relation_to_existing": "same",
                "evidence_text": "Bu ak\u015fam haberleri izlemek istiyorum.",
            },
        )
        second = self.ingest([news], event="collision-news", text="Bu ak\u015fam haberleri izlemek istiyorum.")

        active = self.db.query(models.TemporaryMemory).filter(
            models.TemporaryMemory.status == models.FactStatus.ACTIVE,
        ).all()
        self.assertEqual(len(active), 2)
        self.assertEqual(len({row.key for row in active}), 2)
        self.assertEqual(first["decisions"][0]["key"], "outdoor_activity_intent")
        self.assertNotEqual(second["decisions"][0]["key"], "outdoor_activity_intent")
        self.assertEqual(
            second["decisions"][0]["analysis"]["temporary_slot"]["action"],
            "collision_disambiguated",
        )

    def test_ordinary_repeat_with_terminal_punctuation_is_unchanged(self):
        base = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.LONG_TERM,
            category="routine",
            key="morning_drink",
            value="Her sabah sade Türk kahvesi içerim",
            sensitivity=models.Sensitivity.NORMAL,
            confidence=0.95,

            expires_at=None,
            reason="Controlled extraction",
            analyzer_source="ollama",
        )
        first = self.ingest(
            [base],
            event="coffee-without-period",
            text="Her sabah sade Türk kahvesi içerim.",
        )
        repeated = self.ingest(
            [replace(base, value="Her sabah sade Türk kahvesi içerim.")],
            event="coffee-with-period",
            text="Her sabah sade Türk kahvesi içerim.",
        )

        self.assertEqual(first["decisions"][0]["consolidation_action"], "create")
        self.assertEqual(repeated["decisions"][0]["consolidation_action"], "unchanged")
        self.assertEqual(repeated["decisions"][0]["status"], "auto_applied")
        self.assertEqual(
            first["decisions"][0]["applied_ref"],
            repeated["decisions"][0]["applied_ref"],
        )
        self.assertEqual(self.db.query(models.MemoryFact).count(), 1)

    def test_model_match_and_correction_cannot_overwrite_when_key_drifts(self):
        expiry = self.start + timedelta(hours=1)
        original = self.decision(
            models.CandidateMemoryType.SHORT_TERM,
            "outing",
            {"description": "Bugün dışarı çıkmak istiyorum"},
            expiry=expiry,
        )
        first = self.ingest(
            [original],
            event="temporary-update-base",
            text="Bugün dışarı çıkmak istiyorum.",
        )
        corrected = replace(
            original,
            key="model_key_drifted",
            value={"description": "Artık bugün parkta yürümek istiyorum"},
            analysis_metadata={
                "matched_memory_id": first["decisions"][0]["applied_ref"],
                "relation_to_existing": "update",
                "evidence_text": "Artık bugün parkta yürümek istiyorum.",
            },
        )
        second = self.ingest(
            [corrected],
            event="temporary-update-correction",
            text="Artık bugün parkta yürümek istiyorum.",
        )

        active = self.db.query(models.TemporaryMemory).filter(
            models.TemporaryMemory.status == models.FactStatus.ACTIVE,
        ).all()
        self.assertEqual(len(active), 2)
        self.assertEqual({row.key for row in active}, {"outing", "model_key_drifted"})
        self.assertEqual(second["decisions"][0]["key"], "model_key_drifted")
        self.assertEqual(
            second["decisions"][0]["analysis"]["temporary_slot"]["action"],
            "model_key",
        )

    def test_late_exact_repeat_reuses_newer_temporary_row_without_rollback(self):
        decision = self.decision(
            models.CandidateMemoryType.SHORT_TERM,
            "rest",
            {"description": "Biraz dinlenmek istiyorum"},
            expiry=self.start + timedelta(hours=2),
        )
        newer = main.apply_short_term_decision(
            decision,
            "owner",
            "session",
            self.db,
            occurred_at=self.start + timedelta(minutes=2),
        )
        late = main.apply_short_term_decision(
            decision,
            "owner",
            "session",
            self.db,
            occurred_at=self.start,
        )

        self.assertEqual(late["memory_id"], newer["memory_id"])
        self.assertEqual(late["_slot_resolution"]["action"], "stale_reused_newer")
        self.assertEqual(self.db.query(models.TemporaryMemory).count(), 1)
        row = self.db.get(models.TemporaryMemory, newer["memory_id"])
        self.assertEqual(main.stored_utc_datetime(row.occurred_at), self.start + timedelta(minutes=2))

    def test_sensitive_current_state_auto_applies_and_stays_temporary(self):
        decision = self.decision(models.CandidateMemoryType.SENSITIVE, "headache", "Başım ağrıyor.",
                                 sensitive=True, expiry=self.start + timedelta(minutes=60))
        result = self.ingest([decision], text=decision.value)
        candidate = result["decisions"][0]
        self.assertEqual(candidate["status"], "auto_applied")
        self.assertEqual(candidate["analysis"]["storage_destination"], "session")
        self.assertEqual(self.db.query(models.MemoryFact).count(), 0)
        self.assertEqual(len(self.context()["temporary_memories"]), 1)
        self.assertEqual(self.context(now=self.start + timedelta(minutes=60))["temporary_memories"], [])
        self.assertEqual(len(self.context()["recent_messages"]), 1)
        self.assertEqual(self.context(session="new-session")["temporary_memories"], [])

    def test_direct_health_symptom_auto_applies_with_ttl_as_user_asserted(self):
        text = "Başım ağrıyor."
        decision = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.SENSITIVE,
            category="symptom",
            key="headache",
            value=text,
            sensitivity=models.Sensitivity.HEALTH,
            confidence=0.95,

            expires_at=self.start + timedelta(hours=1),
            reason="Direct health assertion",
            analyzer_source="ollama",
            analysis_metadata={
                "health_persistence_policy": memory_analyzer.HEALTH_ASSERTION_POLICY,
                "verification_status": models.VerificationStatus.USER_ASSERTED.value,
                "claim_kind": "assertion",
                "subject": "user",
                "should_store": True,
                "speech_act": "current_state",
                "evidence_text": text,
            },
        )

        result = self.ingest(
            [decision],
            event="direct-health-symptom",
            text=text,
        )

        candidate = result["decisions"][0]
        self.assertEqual(candidate["status"], "auto_applied")
        row = self.db.get(models.TemporaryMemory, candidate["applied_ref"])
        self.assertEqual(row.sensitivity, models.Sensitivity.HEALTH)
        self.assertEqual(
            row.verification_status,
            models.VerificationStatus.USER_ASSERTED,
        )
        self.assertEqual(
            main.stored_utc_datetime(row.expires_at),
            self.start + timedelta(hours=1),
        )

    def test_bounded_health_event_is_append_only_episode_and_query_retrievable(self):
        text = "Dün banyoda dengemi kaybedip düştüm."
        decision = main.MemoryDecision(
            memory_type=models.CandidateMemoryType.SENSITIVE,
            category="incident",
            key="bathroom_fall",
            value=text,
            sensitivity=models.Sensitivity.HEALTH,
            confidence=0.95,

            expires_at=None,
            reason="Bounded past incident",
            analyzer_source="ollama",
            analysis_metadata={
                "health_persistence_policy": memory_analyzer.HEALTH_ASSERTION_POLICY,
                "verification_status": models.VerificationStatus.USER_ASSERTED.value,
                "claim_kind": "assertion",
                "subject": "user",
                "should_store": True,
                "speech_act": "episode",
                "evidence_text": text,
            },
            scope=models.MemoryScope.EPISODE,
        )

        first = self.ingest([decision], event="fall-one", text=text)
        second = self.ingest(
            [replace(decision, value="Geçen ay salonda düştüm.")],
            event="fall-two",
            text="Geçen ay salonda düştüm.",
        )

        self.assertEqual(first["decisions"][0]["scope"], "episode")
        self.assertEqual(second["decisions"][0]["scope"], "episode")
        self.assertEqual(first["decisions"][0]["status"], "auto_applied")
        self.assertEqual(self.db.query(models.MemoryEpisode).count(), 2)
        self.assertEqual(self.db.query(models.MemoryFact).count(), 0)
        self.assertEqual(self.db.query(models.TemporaryMemory).count(), 0)

        context = main.build_context(
            main.ContextRequest(
                user_id="owner",
                session_id="session",
                query="Nerede düştüm?",
            ),
            self.db,
        )
        self.assertEqual(len(context["episodes"]), 2)
        self.assertEqual(
            set(context["episode_refs"]),
            {first["decisions"][0]["applied_ref"], second["decisions"][0]["applied_ref"]},
        )
        self.assertEqual(context["episode_memory_budget"]["selected_item_count"], 2)

    def test_expired_episode_is_not_returned(self):
        created = main._write_memory_episode(
            main.EpisodeCreate(
                user_id="owner",
                session_id="session",
                category="incident",
                key="old_fall",
                value="Uzun zaman önce düştüm.",
                occurred_at=self.start - timedelta(days=10),
                retention_until=self.start - timedelta(seconds=1),
            ),
            self.db,
        )
        self.assertIsNotNone(self.db.get(models.MemoryEpisode, created["episode_id"]))
        self.assertEqual(self.context()["episodes"], [])

    def test_sensitive_temporary_memory_expires_without_confirmation(self):
        result = self.ingest([self.decision(models.CandidateMemoryType.SENSITIVE, "symptom", "Belirti",
                                           sensitive=True, expiry=self.start + timedelta(minutes=1))])
        self.assertEqual(result["decisions"][0]["status"], "auto_applied")
        self.assertEqual(len(self.context()["temporary_memories"]), 1)
        self.assertEqual(
            self.context(now=self.start + timedelta(minutes=1))["temporary_memories"],
            [],
        )
        self.assertEqual(self.db.query(models.MemoryFact).count(), 0)
        self.assertEqual(self.db.query(models.TemporaryMemory).count(), 1)

    def test_sensitive_history_is_available_without_confirmation(self):
        result = self.ingest([self.decision(models.CandidateMemoryType.SENSITIVE, "health", "Sağlık bilgisi",
                                           sensitive=True)])
        self.assertEqual(result["decisions"][0]["status"], "auto_applied")
        self.assertEqual(len(self.context()["recent_messages"]), 1)
        self.assertEqual(main.list_conversation_messages("session", "owner", self.db)["message_count"], 1)

    def test_sensitive_turn_linked_assistant_reply_remains_in_window(self):
        result = self.ingest([self.decision(
            models.CandidateMemoryType.SENSITIVE, "health", "Başım ağrıyor.",
            sensitive=True,
        )], event="health-turn", text="Başım ağrıyor.")
        self.assertEqual(result["decisions"][0]["status"], "auto_applied")
        main.create_conversation_message("session", main.ConversationMessageCreate(
            message_id="health-reply", user_id="owner",
            role=models.ConversationRole.ASSISTANT,
            content="Başınızın ağrıdığını anlıyorum.", parent_message_id="health-turn",
            occurred_at=self.start + timedelta(seconds=1),
        ), self.db)
        self.assertEqual(
            [message["role"] for message in self.context()["recent_messages"]],
            ["user", "assistant"],
        )

    def test_discard_cannot_store_even_if_malformed_expiry_is_present(self):
        result = self.ingest([self.decision(models.CandidateMemoryType.DISCARD, "bad", "Bad",
                                           expiry=self.start + timedelta(minutes=60))])
        self.assertEqual(result["decisions"][0]["status"], "ignored")
        self.assertEqual(result["decisions"][0]["analysis"]["storage_destination"], "discard")
        self.assertEqual(self.db.query(models.TemporaryMemory).count(), 0)

    def test_duplicate_event_with_changed_payload_is_rejected(self):
        self.ingest([self.decision(models.CandidateMemoryType.DISCARD, "none", None)])
        with self.assertRaises(HTTPException) as error:
            main.process_interaction(main.InteractionProcessRequest(
                event_id="event", user_id="owner", session_id="session", text="Değişik beyan"), self.db)
        self.assertEqual(error.exception.status_code, 409)

    def test_sessionless_event_candidate_cannot_replay_across_sessions(self):
        main.create_memory_event(main.EventCreate(
            event_id="generic-event", user_id="owner", session_id=None,
            event_type="user_message", payload={"text": "Kullanıcı beyanı."},
            occurred_at=self.start,
        ), self.db)
        with patch.object(main, "analyze_message", return_value=[
            self.decision(models.CandidateMemoryType.DISCARD, "none", None)
        ]) as analyzer:
            first = main.process_interaction(main.InteractionProcessRequest(
                event_id="generic-event", user_id="owner", session_id="session-a",
                text="Kullanıcı beyanı.", occurred_at=self.start,
            ), self.db)
            with self.assertRaises(HTTPException) as error:
                main.process_interaction(main.InteractionProcessRequest(
                    event_id="generic-event", user_id="owner", session_id="session-b",
                    text="Kullanıcı beyanı.", occurred_at=self.start,
                ), self.db)
        self.assertEqual(first["status"], "processed")
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(analyzer.call_count, 1)
        self.assertEqual(self.db.query(models.MemoryCandidate).one().session_id, "session-a")

    def test_partial_ingest_retry_preserves_original_time_and_ttl(self):
        main.create_memory_event(main.EventCreate(event_id="partial", user_id="owner", session_id="session",
                                                 event_type="user_message", payload={"text": "Geçici niyet"},
                                                 occurred_at=self.start), self.db)
        main.create_conversation_message("session", main.ConversationMessageCreate(
            message_id="partial", user_id="owner", role=models.ConversationRole.USER,
            content="Geçici niyet", occurred_at=self.start), self.db)
        decision = self.decision(models.CandidateMemoryType.SHORT_TERM, "goal", "Geçici niyet",
                                 expiry=self.start + timedelta(minutes=60))
        with patch.object(main, "analyze_message", return_value=[decision]) as analyzer:
            main.process_interaction(main.InteractionProcessRequest(
                event_id="partial", user_id="owner", session_id="session", text="Geçici niyet",
                occurred_at=self.start + timedelta(minutes=5)), self.db)
        self.assertEqual(analyzer.call_args.args[1], self.start)
        row = self.db.query(models.TemporaryMemory).one()
        self.assertEqual(main.stored_utc_datetime(row.occurred_at), self.start)
        self.assertEqual(self.db.query(models.ConversationMessage).count(), 1)

    def test_mixed_promotion_failure_does_not_commit_partial_candidates_or_items(self):
        short = self.decision(models.CandidateMemoryType.SHORT_TERM, "goal", "Goal",
                              expiry=self.start + timedelta(minutes=60))
        long = self.decision(models.CandidateMemoryType.LONG_TERM, "routine", "Routine")
        bad = self.decision(models.CandidateMemoryType.SHORT_TERM, "x" * 101, "Bad",
                            expiry=self.start + timedelta(minutes=60))
        with self.assertRaises(ValueError):
            self.ingest([short, long, bad])
        self.db.rollback()
        self.assertEqual(self.db.query(models.MemoryCandidate).count(), 0)
        self.assertEqual(self.db.query(models.TemporaryMemory).count(), 0)
        self.assertEqual(self.db.query(models.MemoryFact).count(), 0)
        # The raw event/message survive so the caller can retry extraction.
        self.assertEqual(self.db.query(models.MemoryEvent).count(), 1)
        self.assertEqual(self.db.query(models.ConversationMessage).count(), 1)

    def test_temp_only_session_ownership_and_deletion(self):
        self.temporary("goal", "Goal")
        with self.assertRaises(HTTPException) as error:
            self.temporary("goal", "Other", user="other")
        self.assertEqual(error.exception.status_code, 403)
        response = main.delete_session("session", "owner", self.db)
        self.assertEqual(response.status_code, 204)
        self.assertEqual(self.db.query(models.TemporaryMemory).count(), 0)

    def test_oversized_temporary_value_is_omitted_not_truncated(self):
        value = "x" * 1000
        row = self.temporary("large", value)
        with patch.dict(os.environ, {"MEMORY_TEMPORARY_MAX_TOKENS": "128"}):
            context = self.context()
        self.assertEqual(context["temporary_memories"], [])
        self.assertEqual(self.db.get(models.TemporaryMemory, row["memory_id"]).value_json["value"], value)

    def test_direct_temporary_endpoint_accepts_user_asserted_sensitive_data(self):
        result = main.create_temporary_memory(main.TemporaryMemoryCreate(
            user_id="owner", session_id="session", key="health", value="Belirti",
            sensitivity=models.Sensitivity.HEALTH,
            verification_status=models.VerificationStatus.USER_ASSERTED,
            occurred_at=self.start,
            expires_at=self.start + timedelta(minutes=60)), self.db)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["temporary_memory"]["sensitivity"], "health")

    def test_blank_external_identifiers_are_rejected(self):
        constructors = (
            lambda: main.EventCreate(event_id="", user_id="owner", event_type="user_message", payload={}),
            lambda: main.ConversationMessageCreate(
                message_id="", user_id="owner", role=models.ConversationRole.USER, content="x"),
            lambda: main.InteractionProcessRequest(
                event_id="", user_id="owner", session_id="session", text="x"),
            lambda: main.TemporaryMemoryCreate(
                user_id="owner", session_id="session", key="x", value="x",
                source_event_id="", expires_at=self.start + timedelta(minutes=1)),
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor), self.assertRaises(ValueError):
                constructor()
        with self.assertRaises(ValueError):
            main.upsert_temporary_memory(
                self.db, user_id="owner", session_id="session", category="session",
                key="x", value="x", sensitivity=models.Sensitivity.NORMAL,
                verification_status=models.VerificationStatus.UNVERIFIED,
                confidence=1.0, source_event_id="", occurred_at=self.start,
                expires_at=self.start + timedelta(minutes=1),
            )

    def test_legacy_goal_own_expiry_is_respected(self):
        main.update_session_state("session", main.SessionUpdate(
            user_id="owner", state=main.SessionStatePayload(active_goal={
                "description": "Legacy", "expires_at": (self.start - timedelta(minutes=1)).isoformat()}),
            expires_at=self.start + timedelta(days=1)), self.db)
        self.assertEqual(self.context()["session"], {})

    def test_user_only_history_is_not_sent_to_memory_extractor(self):
        for index, text in enumerate(("Mutfağa geçiyorum.", "Su içiyorum."), start=1):
            main.create_conversation_message("session", main.ConversationMessageCreate(
                message_id=f"user-history-{index}", user_id="owner",
                role=models.ConversationRole.USER, content=text,
                occurred_at=self.start - timedelta(minutes=3 - index),
            ), self.db)
        discard = self.decision(models.CandidateMemoryType.DISCARD, "none", None)
        with patch.object(main, "analyze_message", return_value=[discard]) as analyzer:
            result = main.process_interaction(main.InteractionProcessRequest(
                event_id="standalone-assertion", user_id="owner", session_id="session",
                text="Biraz dinlenmek istiyorum.", occurred_at=self.start,
            ), self.db)
        self.assertEqual(analyzer.call_args.args[2], [])
        self.assertEqual(result["decisions"][0]["analysis"]["analysis_context"], {
            "strategy": "last_assistant_suffix_v1", "message_count": 0,
        })

    def test_memory_extractor_uses_only_last_assistant_suffix(self):
        messages = (
            ("old-user", models.ConversationRole.USER, "Eski ve bağımsız bilgi.", 3),
            ("last-assistant", models.ConversationRole.ASSISTANT,
             "Yürüyüşü bugün yapmak ister misiniz?", 2),
            ("reply-fragment", models.ConversationRole.USER, "Bir düşüneyim.", 1),
        )
        for message_id, role, content, minutes in messages:
            main.create_conversation_message("session", main.ConversationMessageCreate(
                message_id=message_id, user_id="owner", role=role, content=content,
                occurred_at=self.start - timedelta(minutes=minutes),
            ), self.db)
        discard = self.decision(models.CandidateMemoryType.DISCARD, "none", None)
        with patch.object(main, "analyze_message", return_value=[discard]) as analyzer:
            main.process_interaction(main.InteractionProcessRequest(
                event_id="contextual-reply", user_id="owner", session_id="session",
                text="Evet, bugün yapalım.", occurred_at=self.start,
            ), self.db)
        self.assertEqual(analyzer.call_args.args[2], [
            {"role": "assistant", "content": "Yürüyüşü bugün yapmak ister misiniz?"},
            {"role": "user", "content": "Bir düşüneyim."},
        ])


if __name__ == "__main__":
    unittest.main()
