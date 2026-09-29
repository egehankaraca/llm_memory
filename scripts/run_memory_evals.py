#!/usr/bin/env python3
"""Run the curated memory-classification evaluation against the analyzer."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import models  # noqa: E402
from memory_analyzer import (  # noqa: E402
    MemoryDecision,
    analyze_message,
    get_analyzer_status,
    resolved_memory_scope,
)


DEFAULT_DATASET = PROJECT_ROOT / "evals" / "memory_cases.jsonl"
Destination = Literal["profile", "episode", "session", "none"]


class EvalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=10_000)
    recent_messages: list[dict[str, str]] = Field(default_factory=list, max_length=10)


class ExpectedDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_type: models.CandidateMemoryType
    sensitivity: models.Sensitivity
    destination: Destination


class ExpectedResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[ExpectedDecision] = Field(min_length=1, max_length=8)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    input: EvalInput
    expected: ExpectedResult
    tags: list[str] = Field(default_factory=list)


def load_dataset(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Dataset okunamadı: {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            case = EvalCase.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(
                f"Geçersiz dataset satırı {line_number}: {exc}"
            ) from exc
        if case.id in seen_ids:
            raise ValueError(f"Tekrarlanan case ID: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise ValueError(f"Dataset boş: {path}")
    return cases


def destination_for(decision: MemoryDecision) -> Destination:
    scope = resolved_memory_scope(decision)
    if scope == models.MemoryScope.PROFILE:
        return "profile"
    if scope == models.MemoryScope.EPISODE:
        return "episode"
    if scope == models.MemoryScope.SESSION:
        return "session"
    return "none"


def expected_signature(decision: ExpectedDecision) -> tuple[str, str, str]:
    return (
        decision.memory_type.value,
        decision.sensitivity.value,
        decision.destination,
    )


def actual_signature(decision: MemoryDecision) -> tuple[str, str, str]:
    return (
        decision.memory_type.value,
        decision.sensitivity.value,
        destination_for(decision),
    )


def signature_dict(signature: tuple[str, str, str]) -> dict[str, object]:
    return {
        "memory_type": signature[0],
        "sensitivity": signature[1],
        "destination": signature[2],
    }


def compare_decisions(
    expected: list[ExpectedDecision],
    actual: list[MemoryDecision],
) -> tuple[bool, list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    expected_signatures = [expected_signature(item) for item in expected]
    actual_signatures = [actual_signature(item) for item in actual]
    return (
        Counter(expected_signatures) == Counter(actual_signatures),
        expected_signatures,
        actual_signatures,
    )


def compare_dimensions(
    expected: list[tuple[str, str, str]],
    actual: list[tuple[str, str, str]],
) -> dict[str, bool]:
    dimensions = {
        "memory_type": 0,
        "sensitivity": 1,
        "destination": 2,
    }
    return {
        name: Counter(item[index] for item in expected)
        == Counter(item[index] for item in actual)
        for name, index in dimensions.items()
    }


def evaluate_case(case: EvalCase, occurred_at: datetime) -> dict[str, object]:
    started_at = perf_counter()
    try:
        decisions = analyze_message(
            case.input.text,
            occurred_at,
            case.input.recent_messages,
        )
        elapsed_seconds = perf_counter() - started_at
        passed, expected, actual = compare_decisions(
            case.expected.decisions,
            decisions,
        )
        sources = sorted({decision.analyzer_source for decision in decisions})
        ollama_requested = (
            os.getenv("MEMORY_ANALYZER_PROVIDER", "ollama") == "ollama"
        )
        used_fallback = ollama_requested and any(
            source != "ollama" for source in sources
        )
        if ollama_requested:
            passed = passed and not used_fallback
        policy_guards = set()
        for decision in decisions:
            analysis = decision.analysis_metadata or {}
            if analysis.get("evidence_guard"):
                policy_guards.add(analysis["evidence_guard"])
            if analysis.get("sensitivity_overridden_by_policy"):
                policy_guards.add("sensitivity_override")
            if analysis.get("contextual_intent_recovered_by_policy"):
                policy_guards.add("contextual_intent_recovery")
            if analysis.get("extraction_retry_count"):
                policy_guards.add("extraction_retry")
            if analysis.get("contract_normalization"):
                policy_guards.add("contract_normalization")
            if analysis.get("evidence_quote_fallback"):
                policy_guards.add("evidence_quote_fallback")
            if analysis.get("value_grounding_fallback"):
                policy_guards.add("value_grounding_fallback")
        return {
            "id": case.id,
            "passed": passed,
            "text": case.input.text,
            "tags": case.tags,
            "expected": [signature_dict(item) for item in expected],
            "actual": [signature_dict(item) for item in actual],
            "decision_details": [
                {"value": decision.value, "analysis": decision.analysis_metadata, "reason": decision.reason}
                for decision in decisions
            ],
            "dimension_matches": compare_dimensions(expected, actual),
            "sources": sources,
            "policy_guards": sorted(policy_guards),
            "used_fallback": used_fallback,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "error": None,
        }
    except Exception as exc:  # Keep the remaining evaluation cases running.
        return {
            "id": case.id,
            "passed": False,
            "text": case.input.text,
            "tags": case.tags,
            "expected": [
                signature_dict(expected_signature(item))
                for item in case.expected.decisions
            ],
            "actual": [],
            "dimension_matches": {
                "memory_type": False,
                "sensitivity": False,
                "destination": False,
            },
            "sources": [],
            "used_fallback": False,
            "elapsed_seconds": round(perf_counter() - started_at, 3),
            "error": f"{type(exc).__name__}: {exc}",
        }


def select_cases(
    cases: list[EvalCase],
    case_ids: list[str],
    tags: list[str],
    limit: int | None,
) -> list[EvalCase]:
    selected = cases
    if case_ids:
        requested = set(case_ids)
        available = {case.id for case in cases}
        missing = requested - available
        if missing:
            raise ValueError(f"Bilinmeyen case ID: {', '.join(sorted(missing))}")
        selected = [case for case in selected if case.id in requested]
    if tags:
        selected = [case for case in selected if all(tag in case.tags for tag in tags)]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Filtrelerden sonra çalıştırılacak case kalmadı")
    return selected


def print_case_result(result: dict[str, object], show_passed: bool) -> None:
    passed = bool(result["passed"])
    status = "PASS" if passed else "FAIL"
    elapsed = float(result["elapsed_seconds"])
    sources = ",".join(result["sources"]) or "none"
    print(f"[{status}] {result['id']} ({elapsed:.2f}s, source={sources})")
    if result.get("policy_guards"):
        print(f"  policy guards: {', '.join(result['policy_guards'])}")
    if passed and not show_passed:
        return
    print(f"  text:     {result['text']}")
    print(
        "  expected: "
        + json.dumps(result["expected"], ensure_ascii=False, sort_keys=True)
    )
    print(
        "  actual:   "
        + json.dumps(result["actual"], ensure_ascii=False, sort_keys=True)
    )
    if result["error"]:
        print(f"  error:    {result['error']}")


def build_summary(results: list[dict[str, object]]) -> dict[str, object]:
    passed_count = sum(bool(result["passed"]) for result in results)
    fallbacks = sum(bool(result["used_fallback"]) for result in results)
    errors = sum(result["error"] is not None for result in results)
    total_seconds = sum(float(result["elapsed_seconds"]) for result in results)

    expected_by_type: Counter[str] = Counter()
    matched_by_type: Counter[str] = Counter()
    dimension_matches = {
        "memory_type": 0,
        "sensitivity": 0,
        "destination": 0,
    }
    for result in results:
        expected_types = Counter(item["memory_type"] for item in result["expected"])
        actual_types = Counter(item["memory_type"] for item in result["actual"])
        for memory_type, count in expected_types.items():
            expected_by_type[memory_type] += count
            matched_by_type[memory_type] += min(count, actual_types[memory_type])

        for dimension, matched in result["dimension_matches"].items():
            dimension_matches[dimension] += int(matched)

    type_metrics = {
        memory_type: {
            "matched": matched_by_type[memory_type],
            "expected": expected_count,
            "rate": round(matched_by_type[memory_type] / expected_count, 4),
        }
        for memory_type, expected_count in sorted(expected_by_type.items())
    }
    return {
        "cases": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "pass_rate": round(passed_count / len(results), 4),
        "fallbacks": fallbacks,
        "guarded_cases": sum(bool(result.get("policy_guards")) for result in results),
        "errors": errors,
        "elapsed_seconds": round(total_seconds, 3),
        "average_seconds": round(total_seconds / len(results), 3),
        "case_metrics": {
            dimension: {
                "matched": count,
                "cases": len(results),
                "rate": round(count / len(results), 4),
            }
            for dimension, count in dimension_matches.items()
        },
        "decision_metrics_by_type": type_metrics,
    }


def print_summary(summary: dict[str, object]) -> None:
    print("\nSummary")
    print(
        f"  Cases:      {summary['passed']}/{summary['cases']} passed "
        f"({float(summary['pass_rate']):.1%})"
    )
    print(f"  Failed:     {summary['failed']}")
    print(f"  Fallbacks:  {summary['fallbacks']}")
    print(f"  Guarded:    {summary['guarded_cases']} (system-policy passes, not pure model correctness)")
    print(f"  Errors:     {summary['errors']}")
    print(f"  Duration:   {float(summary['elapsed_seconds']):.2f}s")
    print(f"  Avg/case:   {float(summary['average_seconds']):.2f}s")
    print("  Case metrics:")
    for dimension, metric in summary["case_metrics"].items():
        print(
            f"    {dimension:<12} "
            f"{metric['matched']}/{metric['cases']} ({metric['rate']:.1%})"
        )
    print("  Decisions:")
    for memory_type, metric in summary["decision_metrics_by_type"].items():
        print(
            f"    {memory_type:<10} "
            f"{metric['matched']}/{metric['expected']} ({metric['rate']:.1%})"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Ollama memory classification without writing to PostgreSQL."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--provider", choices=["ollama", "rules"], default="ollama")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma4:12b"))
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        help="Run one case ID; repeat this option for multiple cases.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Run cases containing this tag; repeated tags use AND matching.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--show-passed", action="store_true")
    parser.add_argument("--json-report", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.limit is not None and args.limit < 1:
        print("--limit pozitif olmalıdır", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout pozitif olmalıdır", file=sys.stderr)
        return 2

    os.environ["MEMORY_ANALYZER_PROVIDER"] = args.provider
    os.environ["OLLAMA_MODEL"] = args.model
    os.environ["OLLAMA_TIMEOUT_SECONDS"] = str(args.timeout)

    try:
        cases = select_cases(
            load_dataset(args.dataset),
            args.case_ids,
            args.tag,
            args.limit,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    analyzer_status = get_analyzer_status()
    print(
        "Analyzer: "
        f"provider={analyzer_status['provider']} "
        f"model={analyzer_status.get('model')} "
        f"available={analyzer_status['available']}"
    )
    if args.provider == "ollama" and not analyzer_status["available"]:
        print(
            f"Ollama hazır değil: {analyzer_status.get('error')}",
            file=sys.stderr,
        )
        return 2
    print(f"Dataset:  {args.dataset} ({len(cases)} cases)\n")

    occurred_at = datetime.now(timezone.utc)
    results: list[dict[str, object]] = []
    for case in cases:
        result = evaluate_case(case, occurred_at)
        results.append(result)
        print_case_result(result, args.show_passed)

    summary = build_summary(results)
    print_summary(summary)

    if args.json_report is not None:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analyzer": analyzer_status,
            "dataset": str(args.dataset),
            "summary": summary,
            "results": results,
        }
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nJSON report: {args.json_report}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
