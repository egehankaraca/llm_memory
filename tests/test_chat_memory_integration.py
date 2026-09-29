"""Coordinator + real memory routing with SQLite; only LLM/HTTP are replaced."""

from datetime import timedelta
import json
import os
import unittest
from unittest.mock import Mock, patch
from urllib.parse import unquote, urlsplit

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
import models
from conversation_orchestrator import ConversationOrchestrator, OrchestratorError, OrchestratorSettings
from memory_analyzer import MemoryDecision
import memory_analyzer


class InProcessMemory:
    def __init__(self, db):
        self.db = db

    def request(self, method, path, payload=None):
        try:
            if path == "/v1/context:build":
                return main.build_context(main.ContextRequest(**payload), self.db)
            if path == "/v1/interactions:process":
                return main.process_interaction(main.InteractionProcessRequest(**payload), self.db)
            route = urlsplit(path).path
            if route.startswith("/v1/sessions/") and route.endswith("/messages"):
                session = unquote(route[len("/v1/sessions/"):-len("/messages")])
                return main.create_conversation_message(session, main.ConversationMessageCreate(**payload), self.db)
        except HTTPException as exc:
            raise OrchestratorError(f"HTTP {exc.status_code}") from exc
        raise AssertionError(f"Unexpected request: {method} {path}")


def extracted(memory_type, category, key, value, *, sensitive=False, expires_at=None):
    return MemoryDecision(
        memory_type=memory_type, category=category, key=key, value=value,
        sensitivity=models.Sensitivity.HEALTH if sensitive else models.Sensitivity.NORMAL,
        confidence=0.95,expires_at=expires_at,
        reason="Test extraction", analyzer_source="ollama",
    )


class ChatMemoryIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"MEMORY_ANALYZER_PROVIDER": "rules"})
        self.environment.start()
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        models.Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.memory = InProcessMemory(self.db)
        self.ollama = Mock()
        self.ollama.request.return_value = {"message": {"role": "assistant", "content": "Anladım."}, "done": True}
        self.orchestrator = self.coordinator("chat-user", "chat-session")

    def coordinator(self, user, session):
        return ConversationOrchestrator(
            OrchestratorSettings(async_memory_ingestion=False),
            user,
            session,
            memory_http=self.memory,
            ollama_http=self.ollama,
        )

    def tearDown(self):
        self.db.close()
        models.Base.metadata.drop_all(self.engine)
        self.engine.dispose()
        self.environment.stop()

    def test_new_session_recalls_profile_without_conversation_history(self):
        routine = extracted(models.CandidateMemoryType.LONG_TERM, "routine", "tennis_time", "Cumartesi sabah saat 7'de tenis oynarım")
        with patch("main.analyze_message", return_value=[routine]):
            self.orchestrator.chat(routine.value)
        self.assertEqual(self.db.query(models.ConversationMessage).count(), 2)
        new_session = self.coordinator("chat-user", "new-session")
        with patch("main.analyze_message", return_value=[extracted(models.CandidateMemoryType.DISCARD, None, None, None)]):
            turn = new_session.chat("Cumartesi tenis saatim kaçtı?")
        self.assertEqual(turn.context["profile"]["routine"]["tennis_time"], routine.value)
        self.assertEqual(turn.prompt.history_message_count, 0)
        self.assertIn(routine.value, turn.prompt.messages[0]["content"])
        self.assertEqual(self.coordinator("other-user", "other-session").context("tenis")["profile"], {})
        self.assertEqual(self.db.query(models.ConversationMessage).count(), 4)

    def test_database_fact_reaches_new_session_and_model_writes_final_sentence(self):
        # SQLite exercises storage/retrieval; the reply model is mocked here.
        # This proves the boundary, not native model grammar or factual quality.
        coffee = extracted(
            models.CandidateMemoryType.LONG_TERM, "routine", "morning_coffee",
            "Her sabah bol köpüklü Türk kahvesi içerim",
        )
        with patch("main.analyze_message", return_value=[coffee]):
            saved = self.orchestrator.chat(coffee.value)
        fact = self.db.query(models.MemoryFact).one()
        before = (fact.id, dict(fact.value_json), fact.status)
        native_answer = "Sabahları bol köpüklü Türk kahvesi içiyorsunuz."
        self.ollama.request.return_value = {
            "message": {"role": "assistant", "content": native_answer},
            "done": True,
        }
        new_session = self.coordinator("chat-user", "new-coffee-session")
        question = "Sabahları ne içerdim unuttum"
        with patch("main.analyze_message", return_value=[
            extracted(models.CandidateMemoryType.DISCARD, None, None, None),
        ]):
            turn = new_session.chat(question)
        data = json.loads(
            turn.prompt.messages[0]["content"]
            .split("\nMEMORY_CONTEXT_JSON:\n", 1)[1]
            .split("\nEND_MEMORY_CONTEXT_JSON\n", 1)[0]
        )
        self.assertEqual(turn.prompt.history_message_count, 0)
        self.assertNotIn("profile", data)
        self.assertEqual(len(data["profile_facts"]), 1)
        retrieved = data["profile_facts"][0]
        self.assertEqual(retrieved["owner_user_id"], "chat-user")
        self.assertEqual(retrieved["fact_id"], fact.id)
        self.assertEqual(retrieved["value"], coffee.value)
        self.assertEqual(retrieved["provenance"]["source_event_id"], saved.event_id)
        self.assertEqual(turn.prompt.messages[-1], {"role": "user", "content": question})
        self.assertEqual(turn.reply, native_answer)
        self.assertNotEqual(turn.reply, retrieved["value"])
        self.assertEqual(self.db.get(models.ConversationMessage, turn.assistant_message_id).content,
                         native_answer)
        self.assertEqual(self.db.query(models.MemoryFact).count(), 1)
        self.assertEqual((fact.id, fact.value_json, fact.status), before)

    def test_short_term_goal_and_previous_turn_reach_reply_model(self):
        short = extracted(
            models.CandidateMemoryType.SHORT_TERM, "intent", "active_goal",
            {"type": "outing", "description": "Bugün parka gitmek istiyorum"},
            expires_at=main.utc_now() + timedelta(hours=1),
        )
        with patch("main.analyze_message", return_value=[short]):
            self.orchestrator.chat("Bugün parka gitmek istiyorum")
        with patch("main.analyze_message", return_value=[extracted(models.CandidateMemoryType.DISCARD, None, None, None)]):
            turn = self.orchestrator.chat("Bugün ne yapmak istiyorum?")
        self.assertEqual(turn.context["session"]["type"], "outing")
        self.assertEqual(turn.prompt.history_message_count, 2)
        self.assertIn("parka gitmek", turn.prompt.messages[0]["content"])
        self.assertEqual(self.coordinator("chat-user", "another-session").context()["session"], {})

    def test_explicit_sensitive_statement_enters_profile_without_confirmation(self):
        health = extracted(models.CandidateMemoryType.SENSITIVE, "health", "hypertension", "Tansiyon hastasıyım", sensitive=True)
        with patch("main.analyze_message", return_value=[health]):
            turn = self.orchestrator.chat(health.value)
        self.assertEqual(turn.decisions[0]["status"], "auto_applied")
        self.assertEqual(self.orchestrator.context("tansiyon")["profile"]["health"]["hypertension"], health.value)
        self.assertEqual(self.db.query(models.ConversationMessage).count(), 2)

    def test_new_explicit_value_supersedes_previous_active_fact(self):
        for hour in (7, 8):
            routine = extracted(models.CandidateMemoryType.LONG_TERM, "routine", "tennis_time", f"Cumartesi saat {hour}'de tenis oynarım")
            with patch("main.analyze_message", return_value=[routine]):
                turn = self.orchestrator.chat(routine.value)
        self.assertEqual(turn.decisions[0]["consolidation_action"], "supersede")
        self.assertEqual(turn.decisions[0]["status"], "auto_applied")
        facts = self.db.query(models.MemoryFact).filter(models.MemoryFact.status == models.FactStatus.ACTIVE).all()
        self.assertEqual(len(facts), 1)
        self.assertIn("8'de", facts[0].value_json["value"])
        self.assertEqual(self.db.query(models.MemoryFact).count(), 2)

    def test_actual_policy_prevents_misclassified_question_from_changing_database(self):
        routine = extracted(models.CandidateMemoryType.LONG_TERM, "routine", "tennis_time", "Her cumartesi saat 7'de tenis oynarım")
        with patch("main.analyze_message", return_value=[routine]):
            self.orchestrator.chat(routine.value)
        for text, claim_kind, speech_act, temporal_scope in [
            ("her sabah kaçta tenis oynarım", "assertion", "profile_fact", "persistent"),
            ("tenis", "assertion", "profile_fact", "persistent"),
            ("tenis", "contextual_reply", "intent", "today"),
        ]:
            bad_extraction = memory_analyzer.MemoryExtraction.model_validate({"items": [{
                "should_store": True, "claim_kind": claim_kind, "evidence_text": text,
                "subject": "user", "speech_act": speech_act, "temporal_scope": temporal_scope,
                "sensitivity_domain": "none", "category": "routine", "key": "tennis_wake_time",
                "value": "Her sabah tenis oynarım", "confidence": 1.0, "reason": "Deliberately wrong model output",
            }]})
            with patch.dict(os.environ, {"MEMORY_ANALYZER_PROVIDER": "ollama"}), patch.object(memory_analyzer, "_ollama_request", return_value=bad_extraction):
                turn = self.orchestrator.chat(text)
            self.assertEqual(turn.decisions[0]["memory_type"], "discard")
            self.assertEqual(self.db.query(models.MemoryFact).count(), 1)
            self.assertEqual(self.db.query(models.MemoryFact).one().value_json["value"], routine.value)
            self.assertEqual(self.db.query(models.SessionState).count(), 0)

    def test_advice_then_topic_keeps_answers_and_does_not_change_profile(self):
        routine = extracted(models.CandidateMemoryType.LONG_TERM, "routine", "tennis_time", "Her cumartesi saat 7'de tenis oynarım")
        with patch("main.analyze_message", return_value=[routine]):
            self.orchestrator.chat(routine.value)
        fact = self.db.query(models.MemoryFact).one()
        before = (fact.id, dict(fact.value_json), fact.status)
        for text, answer in [
            ("Tenis oynayayım mı?", "Bugünkü planınızla ilgili neyi değerlendirmek istersiniz?"),
            ("tenis", "Tenis oynamak konusunda nasıl yardımcı olabilirim?"),
        ]:
            self.ollama.request.return_value = {"message": {"role": "assistant", "content": answer}, "done": True}
            # Real routing + conservative rules; only the answer model is mocked.
            turn = self.orchestrator.chat(text)
            self.assertEqual(turn.reply, answer)
            self.assertEqual(turn.decisions[0]["memory_type"], "discard")
            self.assertEqual(self.db.query(models.MemoryFact).count(), 1)
            self.assertEqual((fact.id, fact.value_json, fact.status), before)
            self.assertEqual(self.db.query(models.SessionState).count(), 0)
            last_answer = self.db.get(models.ConversationMessage, turn.assistant_message_id)
            self.assertIsNotNone(last_answer)
            self.assertEqual(last_answer.content, answer)

    def test_actual_policy_auto_applies_mislabeled_address_and_emergency_contact(self):
        for text in [
            "Adresim Bahar Sokak 12 numara, Kadıköy",
            "Acil durumda kızım Ayşe’yi 0555 123 45 67 numarasından ara",
        ]:
            extraction = memory_analyzer.MemoryExtraction.model_validate({"items": [{
                "should_store": True, "claim_kind": "assertion", "evidence_text": text,
                "subject": "user", "speech_act": "profile_fact", "temporal_scope": "persistent",
                "sensitivity_domain": "personal", "category": "profile", "key": "private_data",
                "value": text, "confidence": 1.0, "reason": "Deliberately downgraded privacy label",
            }]})
            with patch.dict(os.environ, {"MEMORY_ANALYZER_PROVIDER": "ollama"}), patch.object(memory_analyzer, "_ollama_request", return_value=extraction):
                turn = self.orchestrator.chat(text)
            self.assertEqual(turn.decisions[0]["memory_type"], "sensitive")
            self.assertEqual(turn.decisions[0]["status"], "auto_applied")
            self.assertTrue(turn.decisions[0]["analysis"]["sensitivity_overridden_by_policy"])
        self.assertEqual(self.db.query(models.MemoryFact).count(), 2)
