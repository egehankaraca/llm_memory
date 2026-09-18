import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from memory_retriever import ProfileMemory
from profile_semantic import (
    SemanticSettings,
    clear_embedding_cache,
    score_profile_memories_semantically,
)


class ProfileSemanticTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_embedding_cache()
        now = datetime(2026, 9, 16, tzinfo=timezone.utc)
        self.memories = [
            ProfileMemory(
                id="tea",
                category="preference",
                key="tea_style",
                value="Çayımı şekersiz içerim",
                confidence=0.95,
                verification_status="unverified",
                updated_at=now,
            ),
            ProfileMemory(
                id="wake",
                category="routine",
                key="wake_time",
                value="Sabah 7'de kalkarım",
                confidence=0.95,
                verification_status="unverified",
                updated_at=now,
            ),
        ]

    def settings(self, provider: str = "ollama") -> SemanticSettings:
        return SemanticSettings(
            provider=provider,
            model="test-embedding",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=1.0,
            max_candidates=10,
            min_similarity=0.55,
            cache_size=20,
        )

    def test_disabled_provider_returns_empty_fallback(self) -> None:
        result = score_profile_memories_semantically(
            self.memories,
            "Çayımı nasıl içerim?",
            settings=self.settings("none"),
        )

        self.assertFalse(result.enabled)
        self.assertFalse(result.available)
        self.assertEqual(result.scores, {})
        self.assertEqual(result.reason, "disabled")

    @patch("profile_semantic._request_ollama_embeddings")
    def test_ollama_embeddings_are_converted_to_cosine_scores(self, embed) -> None:
        embed.return_value = [
            (1.0, 0.0),
            (0.9, 0.1),
            (0.0, 1.0),
        ]

        result = score_profile_memories_semantically(
            self.memories,
            "Sabah içeceğimi nasıl hazırlardım?",
            settings=self.settings(),
        )

        self.assertTrue(result.available)
        self.assertGreater(result.scores["tea"], 0.9)
        self.assertEqual(result.scores["wake"], 0.0)
        embed.assert_called_once()

    @patch("profile_semantic._request_ollama_embeddings")
    def test_embedding_failure_falls_back_without_raising(self, embed) -> None:
        embed.side_effect = RuntimeError("offline")

        result = score_profile_memories_semantically(
            self.memories,
            "Çayımı nasıl içerim?",
            settings=self.settings(),
        )

        self.assertFalse(result.available)
        self.assertEqual(result.scores, {})
        self.assertIn("offline", result.reason or "")


if __name__ == "__main__":
    unittest.main()
