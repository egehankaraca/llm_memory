#!/usr/bin/env python3
"""Benchmark pgvector retrieval independently from the final reply LLM.

The runner creates isolated synthetic long-term facts with deterministic
768-dimensional vectors, measures HNSW/planner and forced-exact cosine search,
and removes its rows unless --keep-data is selected. Optional Ollama timing
measures query embedding separately; it never invokes the chat/reply model.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable
import uuid

from sqlalchemy import delete, insert, select, text
from sqlalchemy.orm import Session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import SessionLocal, engine  # noqa: E402
from memory_embeddings import EMBEDDING_DIMENSIONS  # noqa: E402
import models  # noqa: E402
from profile_semantic import (  # noqa: E402
    clear_embedding_cache,
    embed_texts,
    semantic_settings,
)


RUNNER_NAME = "pgvector_retrieval_benchmark_v1"
BENCHMARK_MODEL = "benchmark-deterministic-v1"
DEFAULT_SIZES = (1_000, 10_000)
EMBEDDING_QUERIES = (
    "Acıktım, ne yemek önerirsin?",
    "Bugün hangi fiziksel aktiviteyi yapabilirim?",
    "Sabahları ne içmeyi severim?",
    "Bana nasıl hitap etmelisin?",
    "Her zamanki uyanma saatim nedir?",
    "Hafta sonu hangi etkinliği tercih ederim?",
    "Çayımı nasıl hazırlamalıyım?",
    "Akşam için sevdiğim bir yemek öner.",
    "Hangi müzik türünden hoşlanırım?",
    "Günlük rutinimde önce ne yaparım?",
)


class BenchmarkError(RuntimeError):
    pass


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary_ms(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": round(min(values, default=0.0), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
        "max_ms": round(max(values, default=0.0), 3),
        "mean_ms": round(sum(values) / len(values), 3) if values else 0.0,
    }


def deterministic_vector(index: int, seed: int) -> list[float]:
    """Create a stable sparse unit vector without an embedding-model call."""
    if index < 0:
        raise ValueError("index must be non-negative")
    values = [0.0] * EMBEDDING_DIMENSIONS
    state = ((index + 1) * 2_654_435_761 + seed) & 0xFFFFFFFF
    for offset in range(32):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        position = state % EMBEDDING_DIMENSIONS
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        magnitude = 0.25 + (state & 0xFFFF) / 65_535
        sign = -1.0 if state & 0x10000 else 1.0
        values[position] += sign * magnitude / (1.0 + offset / 16.0)
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise RuntimeError("deterministic vector unexpectedly has zero norm")
    return [value / norm for value in values]


def fact_id(run_id: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{RUNNER_NAME}:{run_id}:{index}"))


def vector_literal(vector: Iterable[float]) -> str:
    return "[" + ",".join(f"{value:.9g}" for value in vector) + "]"


def walk_plan_nodes(plan: dict[str, Any]) -> list[str]:
    result = [str(plan.get("Node Type", "unknown"))]
    for child in plan.get("Plans", []):
        if isinstance(child, dict):
            result.extend(walk_plan_nodes(child))
    return result


def walk_plan_indexes(plan: dict[str, Any]) -> list[str]:
    result = []
    index_name = plan.get("Index Name")
    if index_name is not None:
        result.append(str(index_name))
    for child in plan.get("Plans", []):
        if isinstance(child, dict):
            result.extend(walk_plan_indexes(child))
    return result


class RetrievalBenchmark:
    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        batch_size: int,
        output: Callable[[str], None] = print,
    ) -> None:
        self.run_id = run_id
        self.seed = seed
        self.batch_size = batch_size
        self.user_id = f"retrieval-bench-{run_id}"
        self.output = output

    def validate_database(self) -> dict[str, Any]:
        if engine.dialect.name != "postgresql":
            raise BenchmarkError("This benchmark requires PostgreSQL + pgvector")
        with engine.connect() as connection:
            extension = connection.execute(text(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )).scalar_one_or_none()
            hnsw_index = connection.execute(text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'memory_embeddings' "
                "AND indexname = 'ix_memory_embeddings_hnsw_cosine'"
            )).scalar_one_or_none()
        if extension is None:
            raise BenchmarkError("pgvector extension is not installed in this database")
        if hnsw_index is None:
            raise BenchmarkError("HNSW cosine index is missing; run alembic upgrade head")
        return {"pgvector_version": extension, "hnsw_index": hnsw_index}

    def seed_until(self, start: int, target: int) -> float:
        if target < start:
            raise ValueError("target size cannot be lower than current size")
        started = time.perf_counter()
        now = datetime.now(timezone.utc)
        for batch_start in range(start, target, self.batch_size):
            batch_end = min(target, batch_start + self.batch_size)
            facts: list[dict[str, Any]] = []
            embeddings: list[dict[str, Any]] = []
            for index in range(batch_start, batch_end):
                memory_id = fact_id(self.run_id, index)
                value = f"synthetic-memory-{index:06d}"
                facts.append({
                    "id": memory_id,
                    "user_id": self.user_id,
                    "category": "benchmark",
                    "key": f"slot_{index:06d}",
                    "value_json": {"value": value},
                    "sensitivity": models.Sensitivity.NORMAL.value,
                    "verification_status": models.VerificationStatus.SYSTEM_VERIFIED.value,
                    "confidence": 1.0,
                    "status": models.FactStatus.ACTIVE.value,
                    "source_event_id": None,
                    "supersedes_id": None,
                    "valid_from": now,
                    "valid_to": None,
                    "expires_at": None,
                    "created_at": now,
                    "updated_at": now,
                })
                embeddings.append({
                    "memory_id": memory_id,
                    "model": BENCHMARK_MODEL,
                    "user_id": self.user_id,
                    "content_hash": hashlib.sha256(value.encode()).hexdigest(),
                    "dimensions": EMBEDDING_DIMENSIONS,
                    "embedding": deterministic_vector(index, self.seed),
                    "created_at": now,
                    "updated_at": now,
                })
            with engine.begin() as connection:
                connection.execute(insert(models.MemoryFact.__table__), facts)
                connection.execute(insert(models.MemoryEmbedding.__table__), embeddings)
            if target >= 5_000 and (
                batch_end == target or batch_end % max(1_000, self.batch_size) == 0
            ):
                self.output(f"  seeded {batch_end}/{target} vectors")
        return time.perf_counter() - started

    def cleanup(self) -> int:
        with engine.begin() as connection:
            result = connection.execute(
                delete(models.MemoryFact).where(
                    models.MemoryFact.user_id == self.user_id
                )
            )
        return int(result.rowcount or 0)

    def analyze(self) -> float:
        started = time.perf_counter()
        with engine.begin() as connection:
            connection.execute(text("ANALYZE memory_embeddings"))
        return time.perf_counter() - started

    def vacuum_after_cleanup(self) -> float:
        started = time.perf_counter()
        with engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            connection.execute(text("VACUUM (ANALYZE) memory_embeddings"))
        return time.perf_counter() - started

    @staticmethod
    def _configure_search_mode(db: Session, mode: str) -> None:
        if mode == "planner_default":
            return
        if mode == "forced_exact":
            db.execute(text("SET LOCAL enable_indexscan = off"))
            db.execute(text("SET LOCAL enable_bitmapscan = off"))
            return
        if mode == "forced_hnsw":
            # Removing sequential/bitmap/sort paths forces the order-capable
            # vector index instead of a btree-filter-plus-sort alternative.
            db.execute(text("SET LOCAL enable_seqscan = off"))
            db.execute(text("SET LOCAL enable_bitmapscan = off"))
            db.execute(text("SET LOCAL enable_sort = off"))
            return
        raise ValueError(f"Unknown search mode: {mode}")

    def _search_once(
        self,
        db: Session,
        *,
        index: int,
        top_k: int,
    ) -> tuple[float, list[str]]:
        vector = deterministic_vector(index, self.seed)
        distance = models.MemoryEmbedding.embedding.cosine_distance(vector)
        statement = (
            select(models.MemoryEmbedding.memory_id)
            .where(
                models.MemoryEmbedding.user_id == self.user_id,
                models.MemoryEmbedding.model == BENCHMARK_MODEL,
            )
            .order_by(distance.asc())
            .limit(top_k)
        )
        started = time.perf_counter()
        rows = list(db.scalars(statement).all())
        elapsed_ms = (time.perf_counter() - started) * 1000
        return elapsed_ms, rows

    def search_batch(
        self,
        *,
        record_count: int,
        query_count: int,
        top_k: int,
        mode: str,
    ) -> dict[str, Any]:
        indices = [
            ((query_index + 1) * 7_919 + self.seed) % record_count
            for query_index in range(query_count + 5)
        ]
        latencies: list[float] = []
        hits = 0
        with SessionLocal() as db:
            self._configure_search_mode(db, mode)
            for position, index in enumerate(indices):
                elapsed_ms, rows = self._search_once(db, index=index, top_k=top_k)
                if position < 5:
                    continue
                latencies.append(elapsed_ms)
                if fact_id(self.run_id, index) in rows:
                    hits += 1
            db.rollback()
        return {
            "mode": mode,
            "query_count": query_count,
            "top_k": top_k,
            "recall_at_k": round(hits / query_count if query_count else 0.0, 4),
            "hits": hits,
            "latency": latency_summary_ms(latencies),
        }

    def explain(self, *, index: int, top_k: int, mode: str) -> dict[str, Any]:
        query = (
            "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
            "SELECT memory_id FROM memory_embeddings "
            "WHERE user_id = :user_id AND model = :model "
            "ORDER BY embedding <=> CAST(:query_vector AS vector) LIMIT :top_k"
        )
        with SessionLocal() as db:
            self._configure_search_mode(db, mode)
            payload = db.execute(text(query), {
                "user_id": self.user_id,
                "model": BENCHMARK_MODEL,
                "query_vector": vector_literal(deterministic_vector(index, self.seed)),
                "top_k": top_k,
            }).scalar_one()
            db.rollback()
        root = payload[0]
        plan = root["Plan"]
        return {
            "mode": mode,
            "node_types": walk_plan_nodes(plan),
            "index_names": walk_plan_indexes(plan),
            "planning_ms": round(float(root.get("Planning Time", 0.0)), 3),
            "execution_ms": round(float(root.get("Execution Time", 0.0)), 3),
            "shared_hit_blocks": int(plan.get("Shared Hit Blocks", 0)),
            "shared_read_blocks": int(plan.get("Shared Read Blocks", 0)),
        }

    def benchmark_size(
        self,
        *,
        size: int,
        query_count: int,
        top_k: int,
        compare_exact: bool,
        seed_seconds: float,
    ) -> dict[str, Any]:
        self.output(f"Benchmarking {size:,} vectors...")
        analyze_seconds = self.analyze()
        planner = self.search_batch(
            record_count=size,
            query_count=query_count,
            top_k=top_k,
            mode="planner_default",
        )
        hnsw = self.search_batch(
            record_count=size,
            query_count=query_count,
            top_k=top_k,
            mode="forced_hnsw",
        )
        exact = self.search_batch(
            record_count=size,
            query_count=query_count,
            top_k=top_k,
            mode="forced_exact",
        ) if compare_exact else None
        sample_index = (self.seed + 7_919) % size
        result = {
            "vector_count": size,
            "seed_seconds_incremental": round(seed_seconds, 3),
            "analyze_seconds": round(analyze_seconds, 3),
            "planner_default": planner,
            "planner_default_explain": self.explain(
                index=sample_index, top_k=top_k, mode="planner_default"
            ),
            "forced_hnsw": hnsw,
            "forced_hnsw_explain": self.explain(
                index=sample_index, top_k=top_k, mode="forced_hnsw"
            ),
        }
        if exact is not None:
            result["forced_exact"] = exact
            result["forced_exact_explain"] = self.explain(
                index=sample_index, top_k=top_k, mode="forced_exact"
            )
        self.output(
            f"  planner p50={planner['latency']['p50_ms']} ms "
            f"p95={planner['latency']['p95_ms']} ms "
            f"p99={planner['latency']['p99_ms']} ms "
            f"recall@{top_k}={planner['recall_at_k']:.1%}"
        )
        self.output(
            f"  hnsw   p50={hnsw['latency']['p50_ms']} ms "
            f"p95={hnsw['latency']['p95_ms']} ms "
            f"p99={hnsw['latency']['p99_ms']} ms "
            f"recall@{top_k}={hnsw['recall_at_k']:.1%}"
        )
        if exact is not None:
            self.output(
                f"  exact   p50={exact['latency']['p50_ms']} ms "
                f"p95={exact['latency']['p95_ms']} ms "
                f"p99={exact['latency']['p99_ms']} ms"
            )
        return result


def benchmark_query_embeddings(query_count: int) -> dict[str, Any]:
    settings = semantic_settings()
    if not settings.enabled:
        raise BenchmarkError(
            "Set MEMORY_PROFILE_SEMANTIC_PROVIDER=ollama for --with-ollama"
        )
    uncached = replace(settings, cache_size=0)
    clear_embedding_cache()
    latencies: list[float] = []
    dimensions: int | None = None
    for index in range(query_count):
        base = EMBEDDING_QUERIES[index % len(EMBEDDING_QUERIES)]
        query = f"{base} [benchmark-{index + 1}]"
        started = time.perf_counter()
        vector = embed_texts([query], settings=uncached)[0]
        latencies.append((time.perf_counter() - started) * 1000)
        dimensions = len(vector)
        if dimensions != EMBEDDING_DIMENSIONS:
            raise BenchmarkError(
                f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {dimensions}"
            )
    return {
        "provider": settings.provider,
        "model": settings.model,
        "query_count": query_count,
        "dimensions": dimensions,
        "first_request_ms": round(latencies[0], 3) if latencies else 0.0,
        "remaining_requests": latency_summary_ms(latencies[1:]),
        "all_requests": latency_summary_ms(latencies),
        "cache_disabled": True,
    }


def parse_sizes(values: list[int]) -> list[int]:
    if not values or any(value < 100 for value in values):
        raise ValueError("sizes must contain values >= 100")
    return sorted(set(values))


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20_260_917)
    parser.add_argument("--with-ollama", action="store_true")
    parser.add_argument("--embedding-queries", type=int, default=20)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument(
        "--json-report",
        type=Path,
        default=Path("evals/retrieval_benchmark.json"),
    )
    args = parser.parse_args(argv)
    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.queries < 1
        or args.top_k < 1
        or args.top_k > 100
        or args.batch_size < 1
        or args.embedding_queries < 1
    ):
        parser.error("queries/batch/embedding counts must be positive; top-k <= 100")

    run_id = uuid.uuid4().hex[:12]
    benchmark = RetrievalBenchmark(
        run_id=run_id,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    report: dict[str, Any] = {
        "runner": RUNNER_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "database": str(engine.url).replace(engine.url.password or "", "***")
        if engine.url.password else str(engine.url),
        "configuration": {
            "sizes": sizes,
            "queries_per_size": args.queries,
            "top_k": args.top_k,
            "dimensions": EMBEDDING_DIMENSIONS,
            "synthetic_model": BENCHMARK_MODEL,
            "layout": "one_synthetic_user_with_many_facts",
            "seed": args.seed,
            "compare_exact": not args.skip_exact,
        },
        "database_capabilities": benchmark.validate_database(),
        "query_embedding": None,
        "datasets": [],
        "cleanup": None,
    }
    current_size = 0
    try:
        if args.with_ollama:
            print("Measuring Ollama query embedding independently...")
            report["query_embedding"] = benchmark_query_embeddings(
                args.embedding_queries
            )
            timing = report["query_embedding"]
            print(
                f"  first={timing['first_request_ms']} ms "
                f"warm-p50={timing['remaining_requests']['p50_ms']} ms "
                f"warm-p95={timing['remaining_requests']['p95_ms']} ms"
            )
        for size in sizes:
            print(f"Seeding {size - current_size:,} rows to reach {size:,}...")
            seed_seconds = benchmark.seed_until(current_size, size)
            current_size = size
            report["datasets"].append(benchmark.benchmark_size(
                size=size,
                query_count=args.queries,
                top_k=args.top_k,
                compare_exact=not args.skip_exact,
                seed_seconds=seed_seconds,
            ))
    finally:
        if args.keep_data:
            report["cleanup"] = {
                "performed": False,
                "user_id": benchmark.user_id,
                "row_count": current_size,
            }
        else:
            deleted = benchmark.cleanup()
            vacuum_seconds = benchmark.vacuum_after_cleanup()
            report["cleanup"] = {
                "performed": True,
                "deleted_facts": deleted,
                "vacuum_analyze_seconds": round(vacuum_seconds, 3),
            }
            print(f"Cleaned {deleted:,} synthetic facts (embeddings cascaded).")
        write_report(args.json_report, report)
        print(f"Report: {args.json_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
