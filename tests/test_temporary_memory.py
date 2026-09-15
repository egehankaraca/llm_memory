import unittest
from datetime import datetime, timedelta, timezone

from alembic.migration import MigrationContext
from alembic.operations import Operations
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from memory_retriever import estimate_tokens
from models import (
    Base,
    FactStatus,
    MemoryEvent,
    Sensitivity,
    SessionState,
    TemporaryMemory,
    VerificationStatus,
)
from temporary_memory import (
    StaleTemporaryMemoryError,
    load_temporary_memories,
    serialize_temporary_memory,
    upsert_temporary_memory,
)


class TemporaryMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine("sqlite://")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.now = datetime(2026, 9, 14, 12, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def store(self, **overrides) -> TemporaryMemory:
        arguments = {
            "user_id": "user-a",
            "session_id": "session-a",
            "category": "current_intent",
            "key": "rest",
            "value": "Biraz dinlenmek istiyorum",
            "sensitivity": Sensitivity.NORMAL,
            "verification_status": VerificationStatus.UNVERIFIED,
            "confidence": 0.95,
            "source_event_id": None,
            "occurred_at": self.now,
            "expires_at": self.now + timedelta(hours=1),
        }
        arguments.update(overrides)
        return upsert_temporary_memory(self.db, **arguments)

    def load(self, **overrides) -> list[TemporaryMemory]:
        arguments = {"user_id": "user-a", "session_id": "session-a", "now": self.now}
        arguments.update(overrides)
        return load_temporary_memories(self.db, **arguments)

    def test_multiple_slots_coexist_and_only_same_slot_is_superseded(self) -> None:
        original = self.store()
        news = self.store(key="evening_news", value="Bu akşam haberleri izlemek istiyorum")
        replacement = self.store(value="Şimdi yürümek istiyorum", occurred_at=self.now + timedelta(minutes=1))
        self.assertEqual(original.status, FactStatus.SUPERSEDED)
        self.assertEqual(news.status, FactStatus.ACTIVE)
        self.assertEqual(replacement.status, FactStatus.ACTIVE)
        self.assertEqual(self.db.query(TemporaryMemory).count(), 3)
        self.assertEqual(
            [row.id for row in self.load(now=self.now + timedelta(minutes=2))],
            [replacement.id, news.id],
        )

    def test_expiry_is_independent_and_exact_deadline_is_excluded(self) -> None:
        self.store(expires_at=self.now + timedelta(seconds=10))
        survivor = self.store(key="evening_news", expires_at=self.now + timedelta(hours=2))
        self.db.commit()
        self.db.expire_all()  # Exercise SQLite's timezone-naive loaded DateTime.
        self.assertEqual(
            [row.id for row in self.load(now=self.now + timedelta(seconds=10))],
            [survivor.id],
        )
        self.assertEqual(self.db.query(TemporaryMemory).count(), 2)

    def test_expired_slot_is_preserved_when_replaced(self) -> None:
        expired = self.store(occurred_at=self.now - timedelta(hours=2), expires_at=self.now)
        active = self.store()
        self.assertEqual(expired.status, FactStatus.EXPIRED)
        self.assertEqual([row.id for row in self.load()], [active.id])
        self.assertEqual(self.db.query(TemporaryMemory).count(), 2)

    def test_late_older_event_cannot_supersede_newer_active_slot(self) -> None:
        newer = self.store(
            occurred_at=self.now + timedelta(minutes=2),
            expires_at=self.now + timedelta(hours=2),
        )
        with self.assertRaises(StaleTemporaryMemoryError) as captured:
            self.store(
                value="Geç gelen eski değer",
                occurred_at=self.now,
                expires_at=self.now + timedelta(hours=1),
            )

        self.assertEqual(captured.exception.newer_memory_id, newer.id)
        self.assertEqual(newer.status, FactStatus.ACTIVE)
        self.assertEqual(self.db.query(TemporaryMemory).count(), 1)

    def test_session_and_user_scope_and_category_are_distinct_slots(self) -> None:
        original = self.store()
        category = self.store(category="current_state", value="Yorgunum")
        other_session = self.store(session_id="session-b")
        other_user = self.store(user_id="user-b")
        self.assertEqual({row.id for row in self.load()}, {original.id, category.id})
        self.assertEqual([row.id for row in self.load(session_id="session-b")], [other_session.id])
        self.assertEqual([row.id for row in self.load(user_id="user-b")], [other_user.id])
        self.assertEqual(self.load(session_id="new-session"), [])

    def test_unexpired_future_occurrence_and_revoked_items_are_excluded(self) -> None:
        self.store(occurred_at=self.now + timedelta(minutes=1))
        revoked = self.store(key="revoked")
        revoked.status = FactStatus.REVOKED
        self.db.flush()
        self.assertEqual(self.load(), [])

    def test_positive_finite_ttl_and_confidence_are_required(self) -> None:
        for invalid in (None, "forever", self.now, self.now - timedelta(seconds=1)):
            with self.subTest(expiry=invalid), self.assertRaises(ValueError):
                self.store(expires_at=invalid)
        for invalid in (-0.1, 1.1, float("inf"), float("nan")):
            with self.subTest(confidence=invalid), self.assertRaises(ValueError):
                self.store(confidence=invalid)
        self.assertEqual(self.db.query(TemporaryMemory).count(), 0)

    def test_storage_constraint_rejects_no_ttl_even_outside_helper(self) -> None:
        row = self.store()
        row.expires_at = row.occurred_at
        with self.assertRaises(IntegrityError):
            self.db.flush()
        self.db.rollback()

    def test_timezone_offsets_are_normalized_for_sqlite_expiry_comparison(self) -> None:
        istanbul = timezone(timedelta(hours=3))
        row = self.store(
            occurred_at=self.now.astimezone(istanbul),
            expires_at=(self.now + timedelta(minutes=10)).astimezone(istanbul),
        )
        self.db.commit()
        self.db.expire_all()
        self.assertEqual([memory.id for memory in self.load()], [row.id])
        self.assertEqual(self.load(now=self.now + timedelta(minutes=10)), [])
        payload = serialize_temporary_memory(row)
        self.assertEqual(payload["occurred_at"], self.now.isoformat())

    def test_source_owner_session_validation_and_idempotent_event_retry(self) -> None:
        event = MemoryEvent(
            id="event-a", user_id="user-a", session_id="session-a",
            event_type="user_message", payload_json={"text": "Dinlenmek istiyorum"},
            occurred_at=self.now,
        )
        self.db.add(event)
        self.db.flush()
        original = self.store(source_event_id=event.id)
        self.assertIs(self.store(source_event_id=event.id), original)
        self.assertEqual(self.db.query(TemporaryMemory).count(), 1)
        for overrides in (
            {"source_event_id": "missing"},
            {"source_event_id": event.id, "user_id": "other"},
            {"source_event_id": event.id, "session_id": "other"},
            {"source_event_id": event.id, "value": "Changed retry"},
            {"source_event_id": event.id, "expires_at": self.now + timedelta(hours=2)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.store(**overrides)
        payload = serialize_temporary_memory(original)
        self.assertEqual(payload["memory_id"], original.id)
        self.assertEqual(payload["owner_user_id"], "user-a")
        self.assertEqual(payload["provenance"]["source_event_id"], event.id)
        self.assertEqual(payload["provenance"]["verification_status"], "unverified")
        self.assertEqual(payload["provenance"]["confidence"], 0.95)

    def test_sensitive_item_expiry_remains_finite_and_can_be_independent(self) -> None:
        row = self.store(
            category="current_state", sensitivity=Sensitivity.HEALTH,
            verification_status=VerificationStatus.USER_CONFIRMED,
            expires_at=self.now + timedelta(minutes=5),
        )
        self.assertEqual(serialize_temporary_memory(row)["sensitivity"], "health")
        self.assertEqual(self.load(now=self.now + timedelta(minutes=5)), [])

    def test_helpers_do_not_commit_callers_transaction(self) -> None:
        self.store()
        self.db.rollback()
        self.assertEqual(self.db.query(TemporaryMemory).count(), 0)

    def test_item_budget_and_whole_transport_token_budget(self) -> None:
        older = self.store(key="older", value={"message": "do not truncate", "nested": [1, 2]})
        newest = self.store(key="newest", value="x" * 3000, occurred_at=self.now + timedelta(seconds=1))
        retrieval = self.now + timedelta(seconds=2)
        self.assertEqual([row.id for row in self.load(now=retrieval, max_items=1, max_tokens=5000)], [newest.id])
        older_cost = estimate_tokens({"temporary_memories": [serialize_temporary_memory(older)]})
        selected = self.load(now=retrieval, max_tokens=older_cost)
        self.assertEqual([row.id for row in selected], [older.id])
        self.assertEqual(serialize_temporary_memory(selected[0])["value"], older.value_json["value"])
        self.assertLessEqual(
            estimate_tokens({"temporary_memories": [serialize_temporary_memory(row) for row in selected]}),
            older_cost,
        )
        self.assertEqual(self.load(now=retrieval, max_tokens=older_cost - 1), [])
        self.assertEqual(self.load(max_tokens=0), [])
        self.assertEqual(self.load(max_items=0), [])
        for budget in ({"max_items": -1}, {"max_tokens": -1}, {"max_items": 1.5}):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.load(**budget)

    def test_additive_migration_preserves_legacy_session_json(self) -> None:
        legacy_json = {"active_goal": "legacy goal", "unrelated": {"keep": True}}
        state = SessionState(
            session_id="legacy", user_id="user-a", state_json=legacy_json,
            expires_at=self.now + timedelta(hours=1),
        )
        self.db.add(state)
        self.db.commit()
        # Run only the new additive revision against a disposable SQLite DB.
        TemporaryMemory.__table__.drop(self.engine)
        migration_path = Path(__file__).resolve().parents[1] / "alembic/versions/20260914_07_temporary_memories.py"
        spec = spec_from_file_location("temporary_memory_migration", migration_path)
        migration = module_from_spec(spec)
        spec.loader.exec_module(migration)
        with self.engine.begin() as connection:
            operations = Operations(MigrationContext.configure(connection))
            with patch.object(migration, "op", operations):
                migration.upgrade()
            self.assertIn("temporary_memories", inspect(connection).get_table_names())
            stored_json = connection.execute(select(SessionState.state_json)).scalar_one()
            self.assertEqual(stored_json, legacy_json)
            checks = {constraint["name"] for constraint in inspect(connection).get_check_constraints("temporary_memories")}
            self.assertIn("ck_temporary_memories_positive_ttl", checks)
            with patch.object(migration, "op", operations):
                migration.downgrade()
            self.assertNotIn("temporary_memories", inspect(connection).get_table_names())
            self.assertEqual(connection.execute(select(SessionState.state_json)).scalar_one(), legacy_json)

    def test_oversized_values_cannot_cause_unbounded_sql_history_fetch(self) -> None:
        older = self.store(key="oldest_small", value="Fits")
        for index in range(7):
            self.store(
                key=f"oversized_{index}", value="x" * 3000,
                occurred_at=self.now + timedelta(seconds=index + 1),
            )
        statements = []

        def collect_sql(connection, cursor, statement, parameters, context, executemany):
            if "FROM temporary_memories" in statement:
                statements.append((statement, parameters))

        event.listen(self.engine, "before_cursor_execute", collect_sql)
        try:
            # The first five candidates do not fit, and SQL never fetches older
            # history looking for the small eighth item when max_items is one.
            self.assertEqual(self.load(now=self.now + timedelta(minutes=1), max_items=1, max_tokens=500), [])
            self.assertEqual(len(statements), 1)
            self.assertIn("LIMIT", statements[0][0].upper())
            self.assertEqual(statements[0][1][-2], 5)
            # A larger, still bounded pool can include the whole older value.
            selected = self.load(now=self.now + timedelta(minutes=1), max_items=2, max_tokens=500)
            self.assertEqual([row.id for row in selected], [older.id])
            self.assertEqual(statements[1][1][-2], 10)
            self.load(now=self.now + timedelta(minutes=1), max_items=10_000, max_tokens=500)
            self.assertEqual(statements[2][1][-2], 500)
        finally:
            event.remove(self.engine, "before_cursor_execute", collect_sql)


if __name__ == "__main__":
    unittest.main()
