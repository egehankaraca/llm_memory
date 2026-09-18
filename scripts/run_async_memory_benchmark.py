#!/usr/bin/env python3
"""Measure async enqueue latency and verify durable worker recovery over HTTP."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable
from urllib import error, parse, request
import uuid


RUNNER_NAME = "async_outbox_benchmark_v1"
TERMINAL_JOB_STATUSES = {"completed", "failed", "canceled"}
WORKLOAD = (
    ("long_term_preference", "Her sabah bol köpüklü Türk kahvesi içerim."),
    ("short_term_intent", "Bugün öğleden sonra yürüyüşe çıkmak istiyorum."),
    ("discard_greeting", "Merhaba, nasılsın?"),
    ("long_term_routine", "Her cumartesi sabah saat 7'de tenis oynarım."),
    ("discard_question", "Yarın hava nasıl olacak?"),
)


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: dict[str, Any]


class JsonHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30):
        parsed = parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Memory URL must be an HTTP(S) URL without credentials/query")
        if timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> HttpResponse:
        encoded = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        req = request.Request(
            self.base_url + path,
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                body = json.loads(raw)
                status_code = response.status
        except error.HTTPError as exc:
            status_code = exc.code
            exc.close()
            raise BenchmarkError(f"{method} {path}: HTTP {status_code}") from exc
        except (error.URLError, OSError, TimeoutError) as exc:
            raise BenchmarkError(f"{method} {path}: service unreachable or timed out") from exc
        except (UnicodeError, ValueError) as exc:
            raise BenchmarkError(f"{method} {path}: invalid JSON response") from exc
        if not isinstance(body, dict):
            raise BenchmarkError(f"{method} {path}: expected a JSON object")
        return HttpResponse(status_code=status_code, body=body)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


def latency_summary(latencies: list[float]) -> dict[str, float]:
    return {
        "min_ms": round(min(latencies, default=0) * 1000, 2),
        "p50_ms": round(percentile(latencies, 0.50) * 1000, 2),
        "p95_ms": round(percentile(latencies, 0.95) * 1000, 2),
        "p99_ms": round(percentile(latencies, 0.99) * 1000, 2),
        "max_ms": round(max(latencies, default=0) * 1000, 2),
    }


def safe_error(exc: Exception) -> str:
    # Never put response bodies or user content in benchmark reports.
    return f"{type(exc).__name__}: {exc}"[:500]


class BenchmarkRunner:
    def __init__(
        self,
        client: JsonHttpClient,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        output: Callable[[str], None] = print,
    ):
        self.client = client
        self.monotonic = monotonic
        self.sleep = sleep
        self.output = output

    def healthcheck(self) -> None:
        response = self.client.request("GET", "/healthz")
        if response.status_code != 200 or response.body.get("status") != "ok":
            raise BenchmarkError("Memory Service healthcheck failed")

    def enqueue(
        self,
        *,
        request_count: int,
        user_count: int,
        concurrency: int,
    ) -> dict[str, Any]:
        self.healthcheck()
        run_id = uuid.uuid4().hex[:12]
        payloads: list[tuple[dict[str, Any], str]] = []
        for index in range(request_count):
            user_index = index % user_count
            label, text = WORKLOAD[index % len(WORKLOAD)]
            payloads.append(({
                "event_id": str(uuid.uuid4()),
                "user_id": f"bench-{run_id}-u{user_index}",
                "session_id": f"bench-{run_id}-s{user_index}",
                "text": text,
            }, label))

        wall_started = self.monotonic()

        def send(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
            payload, label = item
            started = self.monotonic()
            try:
                response = self.client.request(
                    "POST", "/v1/interactions:enqueue", payload
                )
                elapsed = self.monotonic() - started
                job = response.body.get("job")
                if response.status_code != 202 or not isinstance(job, dict):
                    raise BenchmarkError("enqueue did not return HTTP 202 with a job")
                return {
                    "event_id": payload["event_id"],
                    "user_id": payload["user_id"],
                    "session_id": payload["session_id"],
                    "workload_label": label,
                    "ok": True,
                    "http_status": response.status_code,
                    "enqueue_status": response.body.get("status"),
                    "immediate_job_status": job.get("status"),
                    "latency_ms": round(elapsed * 1000, 2),
                    "error": None,
                }
            except Exception as exc:
                return {
                    "event_id": payload["event_id"],
                    "user_id": payload["user_id"],
                    "session_id": payload["session_id"],
                    "workload_label": label,
                    "ok": False,
                    "http_status": None,
                    "enqueue_status": None,
                    "immediate_job_status": None,
                    "latency_ms": round((self.monotonic() - started) * 1000, 2),
                    "error": safe_error(exc),
                }

        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(send, item) for item in payloads]
            for future in as_completed(futures):
                results.append(future.result())
        wall_seconds = self.monotonic() - wall_started
        results.sort(key=lambda item: item["event_id"])
        successful = [item for item in results if item["ok"]]
        latencies = [item["latency_ms"] / 1000 for item in successful]
        immediate_counts: dict[str, int] = {}
        for item in successful:
            status = str(item["immediate_job_status"])
            immediate_counts[status] = immediate_counts.get(status, 0) + 1

        report = {
            "runner": RUNNER_NAME,
            "created_at": utc_now().isoformat(),
            "memory_url": self.client.base_url,
            "run_id": run_id,
            "configuration": {
                "request_count": request_count,
                "user_count": user_count,
                "concurrency": concurrency,
            },
            "enqueue": {
                "passed": len(successful) == request_count,
                "successful": len(successful),
                "failed": request_count - len(successful),
                "wall_seconds": round(wall_seconds, 3),
                "throughput_requests_per_second": round(
                    len(successful) / wall_seconds if wall_seconds > 0 else 0, 2
                ),
                "latency": latency_summary(latencies),
                "immediate_job_status_counts": immediate_counts,
            },
            "events": results,
        }
        self.output(
            f"Enqueue: {len(successful)}/{request_count} successful; "
            f"{report['enqueue']['throughput_requests_per_second']} req/s"
        )
        latency = report["enqueue"]["latency"]
        self.output(
            "Latency: "
            f"p50={latency['p50_ms']} ms p95={latency['p95_ms']} ms "
            f"p99={latency['p99_ms']} ms max={latency['max_ms']} ms"
        )
        self.output(f"Immediate jobs: {immediate_counts}")
        return report

    def fetch_statuses(
        self,
        events: list[dict[str, Any]],
        *,
        concurrency: int,
    ) -> list[dict[str, Any]]:
        def fetch(event: dict[str, Any]) -> dict[str, Any]:
            event_id = event.get("event_id")
            user_id = event.get("user_id")
            if not isinstance(event_id, str) or not isinstance(user_id, str):
                return {"event_id": event_id, "ok": False, "error": "Invalid report event"}
            path = (
                f"/v1/interactions/{parse.quote(event_id, safe='')}/status?"
                + parse.urlencode({"user_id": user_id})
            )
            try:
                response = self.client.request("GET", path)
                job = response.body.get("job")
                decisions = response.body.get("decisions")
                if response.status_code != 200 or not isinstance(job, dict):
                    raise BenchmarkError("status endpoint returned an invalid job")
                return {
                    "event_id": event_id,
                    "ok": True,
                    "job": job,
                    "decisions": decisions if isinstance(decisions, list) else [],
                    "error": None,
                }
            except Exception as exc:
                return {
                    "event_id": event_id,
                    "ok": False,
                    "job": None,
                    "decisions": [],
                    "error": safe_error(exc),
                }

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            return list(executor.map(fetch, events))

    def inspect(
        self,
        report: dict[str, Any],
        *,
        concurrency: int,
    ) -> dict[str, Any]:
        events = successful_events(report)
        statuses = self.fetch_statuses(events, concurrency=concurrency)
        counts: dict[str, int] = {}
        errors = 0
        for item in statuses:
            if not item["ok"]:
                errors += 1
                continue
            status = str(item["job"].get("status"))
            counts[status] = counts.get(status, 0) + 1
        result = {
            "checked_at": utc_now().isoformat(),
            "status_counts": counts,
            "status_errors": errors,
            "all_unfinished": bool(statuses) and errors == 0 and not any(
                status in TERMINAL_JOB_STATUSES for status in counts
            ),
        }
        self.output(f"Jobs: {counts}; status_errors={errors}")
        return result

    def drain(
        self,
        report: dict[str, Any],
        *,
        concurrency: int,
        poll_seconds: float,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        events = successful_events(report)
        deadline = self.monotonic() + timeout_seconds
        started = self.monotonic()
        latest: list[dict[str, Any]] = []
        polls = 0
        while True:
            latest = self.fetch_statuses(events, concurrency=concurrency)
            polls += 1
            unfinished = [
                item for item in latest
                if not item["ok"]
                or item["job"].get("status") not in TERMINAL_JOB_STATUSES
            ]
            if not unfinished or self.monotonic() >= deadline:
                break
            self.sleep(min(poll_seconds, max(0, deadline - self.monotonic())))

        elapsed = self.monotonic() - started
        counts: dict[str, int] = {}
        status_errors = 0
        attempt_counts: list[int] = []
        decision_types: dict[str, int] = {}
        analyzer_sources: dict[str, int] = {}
        for item in latest:
            if not item["ok"]:
                status_errors += 1
                continue
            job = item["job"]
            status = str(job.get("status"))
            counts[status] = counts.get(status, 0) + 1
            attempts = job.get("attempt_count")
            if isinstance(attempts, int):
                attempt_counts.append(attempts)
            for decision in item["decisions"]:
                if not isinstance(decision, dict):
                    continue
                memory_type = str(decision.get("memory_type"))
                source = str(decision.get("analyzer_source"))
                decision_types[memory_type] = decision_types.get(memory_type, 0) + 1
                analyzer_sources[source] = analyzer_sources.get(source, 0) + 1

        completed = counts.get("completed", 0)
        passed = completed == len(events) and status_errors == 0
        result = {
            "checked_at": utc_now().isoformat(),
            "passed": passed,
            "expected_jobs": len(events),
            "status_counts": counts,
            "status_errors": status_errors,
            "poll_count": polls,
            "drain_wait_seconds": round(elapsed, 3),
            "completed_jobs_per_second": round(
                completed / elapsed if elapsed > 0 else 0, 2
            ),
            "attempts": {
                "mean": round(
                    sum(attempt_counts) / len(attempt_counts), 2
                ) if attempt_counts else 0,
                "max": max(attempt_counts, default=0),
                "retried_jobs": sum(value > 1 for value in attempt_counts),
            },
            "decision_type_counts": decision_types,
            "analyzer_source_counts": analyzer_sources,
        }
        self.output(
            f"Drain: completed={completed}/{len(events)} statuses={counts} "
            f"errors={status_errors} elapsed={result['drain_wait_seconds']}s"
        )
        self.output(
            f"Attempts: {result['attempts']}; decisions={decision_types}; "
            f"sources={analyzer_sources}"
        )
        return result


def successful_events(report: dict[str, Any]) -> list[dict[str, Any]]:
    events = report.get("events")
    if not isinstance(events, list):
        raise BenchmarkError("Benchmark report has no events array")
    return [item for item in events if isinstance(item, dict) and item.get("ok") is True]


def load_report(path: Path) -> dict[str, Any]:
    try:
        if path.stat().st_size > 10_000_000:
            raise BenchmarkError("Benchmark report is unexpectedly large")
        result = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"Could not read report: {path}") from exc
    except ValueError as exc:
        raise BenchmarkError(f"Report is not valid JSON: {path}") from exc
    if not isinstance(result, dict) or result.get("runner") != RUNNER_NAME:
        raise BenchmarkError("File is not an async outbox benchmark report")
    successful_events(result)
    return result


def save_report(path: Path, report: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise BenchmarkError(f"Could not write report: {path}") from exc


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def add_http_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-url",
        default=None,
        help="Memory Service base URL; defaults to report URL or MEMORY_SERVICE_URL",
    )
    parser.add_argument("--http-timeout-seconds", type=positive_float, default=30)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    enqueue = subparsers.add_parser("enqueue", help="Submit concurrent jobs and measure API latency")
    add_http_arguments(enqueue)
    enqueue.add_argument("--requests", type=positive_int, default=20)
    enqueue.add_argument("--users", type=positive_int, default=5)
    enqueue.add_argument("--concurrency", type=positive_int, default=10)
    enqueue.add_argument(
        "--report",
        type=Path,
        default=Path("evals/async_outbox_benchmark.json"),
    )

    inspect = subparsers.add_parser("inspect", help="Read current states without waiting")
    add_http_arguments(inspect)
    inspect.add_argument("--report", type=Path, required=True)
    inspect.add_argument("--concurrency", type=positive_int, default=10)
    inspect.add_argument(
        "--expect-unfinished",
        action="store_true",
        help="Fail unless every job is still unfinished (use while worker is stopped)",
    )

    drain = subparsers.add_parser("drain", help="Wait until worker jobs finish")
    add_http_arguments(drain)
    drain.add_argument("--report", type=Path, required=True)
    drain.add_argument("--concurrency", type=positive_int, default=10)
    drain.add_argument("--poll-seconds", type=positive_float, default=1)
    drain.add_argument("--timeout-seconds", type=positive_float, default=600)
    return parser


def resolve_url(explicit: str | None, report: dict[str, Any] | None = None) -> str:
    if explicit:
        return explicit
    if report is not None and isinstance(report.get("memory_url"), str):
        return report["memory_url"]
    return os.getenv("MEMORY_SERVICE_URL", "http://127.0.0.1:8001")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "enqueue":
            if args.users > args.requests:
                raise BenchmarkError("--users cannot exceed --requests")
            client = JsonHttpClient(
                resolve_url(args.memory_url), args.http_timeout_seconds
            )
            report = BenchmarkRunner(client).enqueue(
                request_count=args.requests,
                user_count=args.users,
                concurrency=min(args.concurrency, args.requests),
            )
            save_report(args.report, report)
            print(f"Report: {args.report.resolve()}")
            return 0 if report["enqueue"]["passed"] else 1

        report = load_report(args.report)
        client = JsonHttpClient(
            resolve_url(args.memory_url, report), args.http_timeout_seconds
        )
        runner = BenchmarkRunner(client)
        runner.healthcheck()
        if args.command == "inspect":
            inspection = runner.inspect(report, concurrency=args.concurrency)
            report["last_inspection"] = inspection
            save_report(args.report, report)
            if inspection["status_errors"]:
                return 1
            if args.expect_unfinished and not inspection["all_unfinished"]:
                print("Expected every job to remain unfinished, but terminal jobs were found.", file=sys.stderr)
                return 1
            return 0

        drain_result = runner.drain(
            report,
            concurrency=args.concurrency,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        report["drain"] = drain_result
        save_report(args.report, report)
        return 0 if drain_result["passed"] else 1
    except (BenchmarkError, ValueError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
