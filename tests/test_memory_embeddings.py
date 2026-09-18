import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import main
import memory_embeddings
import models
from profile_semantic import SemanticSettings


class MemoryEmbeddingsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        models.Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.settings = SemanticSettings(
            provider="ollama",
            model="embeddinggemma:latest",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=1.0,
            max_candidates=100,
            min_similarity=0.4,
            cache_size=0,
        )

    def tearDown(self) -> None:
        models.Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def create_fact_and_job(self) -> str:
        with self.Session() as db:
            with patch("main.semantic_settings", return_value=self.settings), patch(
                "memory_embeddings.semantic_settings", return_value=self.settings
            ):
                created = main.create_memory_fact(
                    main.FactCreate(
                        user_id="vector-user",
                        category="preference",
                        key="favorite_food",
                        value="Makarna yemeyi severim",
                    ),
                    db,
                )
            job = db.get(
                models.MemoryEmbeddingJob,
                {
                    "memory_id": created["fact_id"],
                    "model": self.settings.model,
                },
            )
            self.assertIsNotNone(job)
            self.assertEqual(job.status, models.OutboxStatus.PENDING)
            return created["fact_id"]

    def test_fact_write_queues_and_worker_persists_vector(self) -> None:
        fact_id = self.create_fact_and_job()
        vector = tuple([0.25] * memory_embeddings.EMBEDDING_DIMENSIONS)

        with patch("memory_embeddings.embed_texts", return_value=[vector]) as embed:
            result = memory_embeddings.process_one_embedding_job(
                session_factory=self.Session,
                worker_id="test-worker",
                settings=self.settings,
            )

        self.assertTrue(result["indexed"])
        self.assertIsNotNone(
            result["job"]["timing"]["enqueue_to_completion_ms"]
        )
        self.assertGreaterEqual(
            result["job"]["timing"]["enqueue_to_completion_ms"],
            0,
        )
        embed.assert_called_once()
        with self.Session() as db:
            stored = db.get(
                models.MemoryEmbedding,
                {"memory_id": fact_id, "model": self.settings.model},
            )
            job = db.get(
                models.MemoryEmbeddingJob,
                {"memory_id": fact_id, "model": self.settings.model},
            )
            self.assertIsNotNone(stored)
            self.assertEqual(stored.dimensions, 768)
            self.assertEqual(job.status, models.OutboxStatus.COMPLETED)

    def test_worker_failure_is_durable_and_retryable(self) -> None:
        fact_id = self.create_fact_and_job()
        with patch(
            "memory_embeddings.embed_texts",
            side_effect=RuntimeError("Ollama offline"),
        ):
            result = memory_embeddings.process_one_embedding_job(
                session_factory=self.Session,
                worker_id="test-worker",
                base_retry_seconds=0,
                settings=self.settings,
            )

        self.assertIn("Ollama offline", result["error"])
        with self.Session() as db:
            job = db.get(
                models.MemoryEmbeddingJob,
                {"memory_id": fact_id, "model": self.settings.model},
            )
            self.assertEqual(job.status, models.OutboxStatus.RETRY)
            self.assertEqual(job.attempt_count, 1)

    def test_backfill_is_idempotent(self) -> None:
        with self.Session() as db:
            fact = models.MemoryFact(
                id="legacy-fact",
                user_id="legacy-user",
                category="habit",
                key="tea",
                value_json={"value": "Çayı şekersiz içerim"},
                sensitivity=models.Sensitivity.NORMAL,
                verification_status=models.VerificationStatus.USER_ASSERTED,
                confidence=0.9,
                status=models.FactStatus.ACTIVE,
            )
            db.add(fact)
            db.commit()
            first = memory_embeddings.enqueue_missing_embeddings(
                db, settings=self.settings
            )
            second = memory_embeddings.enqueue_missing_embeddings(
                db, settings=self.settings
            )

        self.assertEqual(first, 1)
        self.assertEqual(second, 0)
        # The composite primary key also guarantees one durable job.
        with self.Session() as db:
            self.assertEqual(db.query(models.MemoryEmbeddingJob).count(), 1)


if __name__ == "__main__":
    unittest.main()
