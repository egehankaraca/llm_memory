import os
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
import models
from conversation_orchestrator import (
    ConversationOrchestrator,
    JsonHttpClient,
    OrchestratorSettings,
)
from memory_analyzer import MemoryDecision
from memory_outbox import claim_next_outbox_job, complete_outbox_job
from memory_worker import process_one_outbox_job


def long_term_decision(text: str) -> MemoryDecision:
    return MemoryDecision(
        memory_type=models.CandidateMemoryType.LONG_TERM,
        category="routine",
        key="morning_coffee",
        value=text,
        sensitivity=models.Sensitivity.NORMAL,
        confidence=0.95,

        expires_at=None,
        reason="Test routine",
        analyzer_source="ollama",
        analysis_metadata={
            "claim_kind": "assertion",
            "subject": "user",
            "speech_act": "habit",
            "temporal_scope": "persistent",
            "evidence_text": text,
        },
    )


class MemoryOutboxTest(unittest.TestCase):
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

    def enqueue(self, event_id: str, text: str, session_id: str = "session-1") -> dict:
        return main.enqueue_interaction(
            main.InteractionProcessRequest(
                event_id=event_id,
                user_id="user-1",
                session_id=session_id,
                text=text,
            ),
            self.db,
        )

    def test_enqueue_is_fast_path_without_analyzer_and_is_idempotent(self) -> None:
        with patch.object(main, "analyze_message", side_effect=AssertionError("must not run")):
            first = self.enqueue("event-1", "Her sabah kahve içerim")
            second = self.enqueue("event-1", "Her sabah kahve içerim")

        self.assertEqual(first["status"], "queued")
        self.assertEqual(first["job"]["status"], "pending")
        self.assertEqual(second["status"], "duplicate")
        self.assertEqual(self.db.query(models.MemoryEvent).count(), 1)
        self.assertEqual(self.db.query(models.ConversationMessage).count(), 1)
        self.assertEqual(self.db.query(models.MemoryOutbox).count(), 1)
        self.assertEqual(self.db.query(models.MemoryCandidate).count(), 0)

    def test_enqueue_snapshots_only_preceding_dialogue(self) -> None:
        main.create_conversation_message(
            "session-1",
            main.ConversationMessageCreate(
                message_id="assistant-before",
                user_id="user-1",
                role=models.ConversationRole.ASSISTANT,
                content="Bugün dışarı çıkmak ister misiniz?",
            ),
            self.db,
        )
        self.enqueue("event-context", "Evet, bugün çıkalım")
        job = self.db.get(models.MemoryOutbox, "event-context")
        self.assertEqual(job.analysis_context_json, [{
            "role": "assistant",
            "content": "Bugün dışarı çıkmak ister misiniz?",
        }])

    def test_worker_processes_job_and_marks_it_completed(self) -> None:
        text = "Her sabah kahve içerim"
        self.enqueue("event-worker", text)
        with patch.object(main, "analyze_message", return_value=[long_term_decision(text)]):
            result = process_one_outbox_job(
                session_factory=self.Session,
                worker_id="worker-1",
            )

        self.assertIsNotNone(result)
        self.assertNotIn("error", result)
        self.assertEqual(result["job"]["status"], "completed")
        self.assertEqual(result["job"]["attempt_count"], 1)
        verify = self.Session()
        try:
            fact = verify.query(models.MemoryFact).one()
            self.assertEqual(fact.value_json, {"value": text})
            self.assertEqual(verify.get(models.MemoryOutbox, "event-worker").status,
                             models.OutboxStatus.COMPLETED)
        finally:
            verify.close()

    def test_worker_failure_is_retried_without_losing_the_job(self) -> None:
        self.enqueue("event-retry", "Her sabah kahve içerim")
        job = self.db.get(models.MemoryOutbox, "event-retry")
        job.max_attempts = 2
        self.db.commit()
        with patch.object(main, "analyze_message", side_effect=RuntimeError("model offline")):
            first = process_one_outbox_job(
                session_factory=self.Session,
                worker_id="worker-retry",
                base_retry_seconds=0,
            )
            second = process_one_outbox_job(
                session_factory=self.Session,
                worker_id="worker-retry",
                base_retry_seconds=0,
            )

        self.assertEqual(first["job"]["status"], "retry")
        self.assertEqual(second["job"]["status"], "failed")
        self.assertEqual(second["job"]["attempt_count"], 2)
        self.assertIn("model offline", second["job"]["last_error"])

    def test_same_session_jobs_are_claimed_in_order(self) -> None:
        self.enqueue("event-a", "Şimdi su içeceğim")
        self.enqueue("event-b", "Bu akşam haber izleyeceğim")
        worker_one = self.Session()
        worker_two = self.Session()
        try:
            first = claim_next_outbox_job(worker_one, worker_id="worker-a")
            self.assertEqual(first.event_id, "event-a")
            self.assertIsNone(claim_next_outbox_job(worker_two, worker_id="worker-b"))
            complete_outbox_job(
                worker_one,
                event_id="event-a",
                worker_id="worker-a",
            )
            second = claim_next_outbox_job(worker_two, worker_id="worker-b")
            self.assertEqual(second.event_id, "event-b")
        finally:
            worker_one.close()
            worker_two.close()

    def test_context_hides_unreviewed_async_message(self) -> None:
        self.enqueue("event-hidden", "Tansiyon ilacım değişti")
        context = main.build_context(
            main.ContextRequest(user_id="user-1", session_id="session-1"),
            self.db,
        )
        self.assertEqual(context["recent_messages"], [])

    def test_status_returns_job_and_completed_decisions(self) -> None:
        text = "Her sabah kahve içerim"
        self.enqueue("event-status", text)
        with patch.object(main, "analyze_message", return_value=[long_term_decision(text)]):
            process_one_outbox_job(
                session_factory=self.Session,
                worker_id="worker-status",
            )
        status = main.interaction_status("event-status", "user-1", self.db)
        self.assertEqual(status["job"]["status"], "completed")
        self.assertIsNotNone(
            status["job"]["timing"]["enqueue_to_completion_ms"]
        )
        self.assertGreaterEqual(
            status["job"]["timing"]["enqueue_to_completion_ms"],
            0,
        )
        self.assertEqual(status["decisions"][0]["memory_type"], "long_term")


class AsyncOrchestratorTest(unittest.TestCase):
    def test_async_mode_enqueues_instead_of_waiting_for_extraction(self) -> None:
        memory = Mock(spec=JsonHttpClient)
        ollama = Mock(spec=JsonHttpClient)
        context = {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "profile": {},
            "session": {},
            "temporary_observations": [],
            "recent_messages": [],
            "memory_refs": [],
            "profile_retrieval": {"pinned_categories": []},
        }
        ollama.request.return_value = {
            "message": {"role": "assistant", "content": "Merhaba."},
            "done": True,
        }
        orchestrator = ConversationOrchestrator(
            OrchestratorSettings(async_memory_ingestion=True),
            "async-user",
            "async-session",
            memory_http=memory,
            ollama_http=ollama,
        )

        # Make the mocked outbox echo the generated id, as the real API does.
        def request(method, path, payload=None):
            if path == "/v1/context:build":
                return context
            if path == "/v1/interactions:enqueue":
                return {
                    "status": "queued",
                    "event_id": payload["event_id"],
                    "job": {"event_id": payload["event_id"], "status": "pending"},
                }
            return {"status": "created"}

        memory.request.side_effect = request
        turn = orchestrator.chat("Merhaba")
        paths = [call.args[1] for call in memory.request.call_args_list]
        self.assertEqual(paths, [
            "/v1/context:build",
            "/v1/interactions:enqueue",
            "/v1/sessions/async-session/messages",
        ])
        self.assertEqual(turn.interaction["job"]["status"], "pending")
        self.assertEqual(turn.decisions, [])


if __name__ == "__main__":
    unittest.main()
