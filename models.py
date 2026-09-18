import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from pgvector.sqlalchemy import Vector

from database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def enum_values(enum_class: type[enum.Enum]) -> list[str]:
    return [member.value for member in enum_class]


class FactStatus(str, enum.Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    REVOKED = "revoked"


class Sensitivity(str, enum.Enum):
    NORMAL = "normal"
    PERSONAL = "personal"
    HEALTH = "health"
    EMERGENCY_CONTACT = "emergency_contact"
    LOCATION = "location"
    FINANCIAL = "financial"
    CREDENTIAL = "credential"


PROTECTED_SENSITIVITIES = frozenset(
    {
        Sensitivity.HEALTH,
        Sensitivity.EMERGENCY_CONTACT,
        Sensitivity.LOCATION,
        Sensitivity.FINANCIAL,
        Sensitivity.CREDENTIAL,
    }
)

class VerificationStatus(str, enum.Enum):
    UNVERIFIED = "unverified"
    USER_ASSERTED = "user_asserted"
    USER_CONFIRMED = "user_confirmed"
    CAREGIVER_CONFIRMED = "caregiver_confirmed"
    SYSTEM_VERIFIED = "system_verified"


class CandidateMemoryType(str, enum.Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SENSITIVE = "sensitive"
    DISCARD = "discard"


class MemoryScope(str, enum.Enum):
    """Lifetime/storage axis, independent from sensitivity."""

    PROFILE = "profile"
    EPISODE = "episode"
    SESSION = "session"
    DISCARD = "discard"


class CandidateStatus(str, enum.Enum):
    AUTO_APPLIED = "auto_applied"
    IGNORED = "ignored"


class OutboxStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    RETRY = "retry"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ConversationRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"


class MemoryFact(Base):
    __tablename__ = "memory_facts"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_facts_confidence_range",
        ),
        Index(
            "uq_memory_facts_active_user_category_key",
            "user_id",
            "category",
            "key",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
        Index("ix_memory_facts_user_status", "user_id", "status"),
    )

    id = Column(String(36), primary_key=True, default=new_id)
    user_id = Column(String(128), nullable=False, index=True)
    category = Column(String(100), nullable=False)
    key = Column(String(100), nullable=False)
    value_json = Column(JSON, nullable=False)
    sensitivity = Column(
        SqlEnum(
            Sensitivity,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_sensitivity",
        ),
        nullable=False,
        default=Sensitivity.NORMAL,
    )
    verification_status = Column(
        SqlEnum(
            VerificationStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_verification_status",
        ),
        nullable=False,
        default=VerificationStatus.UNVERIFIED,
    )
    confidence = Column(Numeric(4, 3), nullable=False, default=1)
    status = Column(
        SqlEnum(
            FactStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_fact_status",
        ),
        nullable=False,
        default=FactStatus.ACTIVE,
    )
    source_event_id = Column(
        String(36),
        ForeignKey("memory_events.id", ondelete="SET NULL"),
        nullable=True,
    )
    supersedes_id = Column(
        String(36),
        ForeignKey("memory_facts.id", ondelete="SET NULL"),
        nullable=True,
    )
    valid_from = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    valid_to = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class MemoryEpisode(Base):
    """An immutable, bounded past event that may be retrieved later."""

    __tablename__ = "memory_episodes"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_episodes_confidence_range",
        ),
        Index(
            "ix_memory_episodes_user_occurred",
            "user_id",
            "occurred_at",
        ),
        Index(
            "ix_memory_episodes_user_retention",
            "user_id",
            "retention_until",
        ),
    )

    id = Column(String(36), primary_key=True, default=new_id)
    user_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=True, index=True)
    category = Column(String(100), nullable=False)
    key = Column(String(100), nullable=False)
    value_json = Column(JSON, nullable=False)
    sensitivity = Column(
        SqlEnum(
            Sensitivity,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_episode_sensitivity",
        ),
        nullable=False,
        default=Sensitivity.NORMAL,
    )
    verification_status = Column(
        SqlEnum(
            VerificationStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_episode_verification_status",
        ),
        nullable=False,
        default=VerificationStatus.UNVERIFIED,
    )
    confidence = Column(Numeric(4, 3), nullable=False, default=1)
    source_event_id = Column(
        String(36),
        ForeignKey("memory_events.id", ondelete="SET NULL"),
        nullable=True,
    )
    occurred_at = Column(DateTime(timezone=True), nullable=False)
    retention_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class MemoryEmbedding(Base):
    """Persistent retrieval vector for one immutable memory-fact version."""

    __tablename__ = "memory_embeddings"
    __table_args__ = (
        CheckConstraint("dimensions = 768", name="ck_memory_embeddings_dimensions"),
        Index("ix_memory_embeddings_user_model", "user_id", "model"),
    )

    memory_id = Column(
        String(36),
        ForeignKey("memory_facts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    model = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False)
    content_hash = Column(String(64), nullable=False)
    dimensions = Column(Integer, nullable=False, default=768)
    embedding = Column(Vector(768), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class MemoryEmbeddingJob(Base):
    """Durable request to embed one committed long-term memory fact."""

    __tablename__ = "memory_embedding_jobs"
    __table_args__ = (
        CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0",
            name="ck_memory_embedding_jobs_attempts",
        ),
        Index(
            "ix_memory_embedding_jobs_ready",
            "status",
            "available_at",
            "created_at",
        ),
        Index("ix_memory_embedding_jobs_user", "user_id", "status"),
    )

    memory_id = Column(
        String(36),
        ForeignKey("memory_facts.id", ondelete="CASCADE"),
        primary_key=True,
    )
    model = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False)
    content_hash = Column(String(64), nullable=False)
    status = Column(
        SqlEnum(
            OutboxStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_embedding_job_status",
        ),
        nullable=False,
        default=OutboxStatus.PENDING,
    )
    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=5)
    available_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    locked_at = Column(DateTime(timezone=True), nullable=True)
    locked_by = Column(String(128), nullable=True)
    last_error = Column(Text, nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class SessionState(Base):
    __tablename__ = "session_states"
    __table_args__ = (
        CheckConstraint("version > 0", name="ck_session_states_positive_version"),
        Index("ix_session_states_user_expires", "user_id", "expires_at"),
    )

    session_id = Column(String(128), primary_key=True)
    user_id = Column(String(128), nullable=False, index=True)
    state_json = Column(JSON, nullable=False, default=dict)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class TemporaryMemory(Base):
    """One independently expiring, session-scoped memory item."""

    __tablename__ = "temporary_memories"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_temporary_memories_confidence_range",
        ),
        CheckConstraint(
            "expires_at > occurred_at",
            name="ck_temporary_memories_positive_ttl",
        ),
        Index(
            "uq_temporary_memories_active_owner_slot",
            "user_id",
            "session_id",
            "category",
            "key",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
        Index(
            "ix_temporary_memories_owner_expiry",
            "user_id",
            "session_id",
            "status",
            "expires_at",
        ),
        Index("ix_temporary_memories_expires", "expires_at"),
    )

    id = Column(String(36), primary_key=True, default=new_id)
    user_id = Column(String(128), nullable=False)
    session_id = Column(String(128), nullable=False)
    category = Column(String(100), nullable=False)
    key = Column(String(100), nullable=False)
    value_json = Column(JSON, nullable=False)
    sensitivity = Column(
        SqlEnum(
            Sensitivity,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="temporary_memory_sensitivity",
        ),
        nullable=False,
        default=Sensitivity.NORMAL,
    )
    verification_status = Column(
        SqlEnum(
            VerificationStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="temporary_memory_verification_status",
        ),
        nullable=False,
        default=VerificationStatus.UNVERIFIED,
    )
    confidence = Column(Numeric(4, 3), nullable=False, default=1)
    status = Column(
        SqlEnum(
            FactStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="temporary_memory_status",
        ),
        nullable=False,
        default=FactStatus.ACTIVE,
    )
    source_event_id = Column(
        String(36),
        ForeignKey("memory_events.id", ondelete="SET NULL"),
        nullable=True,
    )
    occurred_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        CheckConstraint(
            "estimated_tokens > 0",
            name="ck_conversation_messages_positive_tokens",
        ),
        Index(
            "ix_conversation_messages_window",
            "user_id",
            "session_id",
            "occurred_at",
        ),
        Index("ix_conversation_messages_expires", "expires_at"),
    )

    id = Column(String(36), primary_key=True, default=new_id)
    user_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    role = Column(
        SqlEnum(
            ConversationRole,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="conversation_message_role",
        ),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    parent_message_id = Column(
        String(36),
        ForeignKey("conversation_messages.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    estimated_tokens = Column(Integer, nullable=False)
    occurred_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class MemoryEvent(Base):
    __tablename__ = "memory_events"
    __table_args__ = (
        Index("ix_memory_events_user_occurred", "user_id", "occurred_at"),
    )

    id = Column(String(36), primary_key=True)
    user_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=True, index=True)
    event_type = Column(String(50), nullable=False)
    payload_json = Column(JSON, nullable=False)
    occurred_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    retention_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)


class MemoryOutbox(Base):
    """Durable request to analyze one already-persisted user event."""

    __tablename__ = "memory_outbox"
    __table_args__ = (
        CheckConstraint(
            "attempt_count >= 0 AND max_attempts > 0",
            name="ck_memory_outbox_attempts",
        ),
        Index(
            "ix_memory_outbox_ready",
            "status",
            "available_at",
            "created_at",
        ),
        Index(
            "ix_memory_outbox_session_order",
            "user_id",
            "session_id",
            "created_at",
        ),
    )

    # One event can produce at most one asynchronous extraction job. Keeping
    # event_id as the primary key makes API retries idempotent by construction.
    event_id = Column(
        String(36),
        ForeignKey("memory_events.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    status = Column(
        SqlEnum(
            OutboxStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_outbox_status",
        ),
        nullable=False,
        default=OutboxStatus.PENDING,
    )
    # Snapshot only the bounded dialogue needed by the extractor. Without it,
    # a delayed worker could accidentally inspect messages that happened after
    # this event.
    analysis_context_json = Column(JSON, nullable=False, default=list)
    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=5)
    available_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    locked_at = Column(DateTime(timezone=True), nullable=True)
    locked_by = Column(String(128), nullable=True)
    last_error = Column(Text, nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )


class MemoryCandidate(Base):
    __tablename__ = "memory_candidates"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_candidates_confidence_range",
        ),
        UniqueConstraint(
            "source_event_id",
            "decision_index",
            name="uq_memory_candidates_event_decision",
        ),
        Index(
            "ix_memory_candidates_user_status",
            "user_id",
            "status",
            "created_at",
        ),
    )

    id = Column(String(36), primary_key=True, default=new_id)
    user_id = Column(String(128), nullable=False, index=True)
    session_id = Column(String(128), nullable=False, index=True)
    source_event_id = Column(
        String(36),
        ForeignKey("memory_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    decision_index = Column(Integer, nullable=False)
    memory_type = Column(
        SqlEnum(
            CandidateMemoryType,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="candidate_memory_type",
        ),
        nullable=False,
    )
    scope = Column(
        SqlEnum(
            MemoryScope,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="candidate_memory_scope",
        ),
        nullable=False,
        default=MemoryScope.DISCARD,
    )
    category = Column(String(100), nullable=True)
    key = Column(String(100), nullable=True)
    value_json = Column(JSON, nullable=True)
    sensitivity = Column(
        SqlEnum(
            Sensitivity,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="candidate_sensitivity",
        ),
        nullable=False,
        default=Sensitivity.NORMAL,
    )
    confidence = Column(Numeric(4, 3), nullable=False)
    analyzer_source = Column(String(50), nullable=False, default="rules")
    analysis_json = Column(JSON, nullable=True)
    consolidation_action = Column(
        String(50),
        nullable=False,
        default="not_applicable",
    )
    consolidates_fact_id = Column(
        String(36),
        ForeignKey("memory_facts.id", ondelete="SET NULL"),
        nullable=True,
    )
    status = Column(
        SqlEnum(
            CandidateStatus,
            values_callable=enum_values,
            native_enum=False,
            create_constraint=True,
            name="memory_candidate_status",
        ),
        nullable=False,
        default=CandidateStatus.IGNORED,
    )
    reason = Column(String(500), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    applied_ref = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utc_now)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
