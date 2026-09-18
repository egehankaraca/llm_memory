import threading
import unittest

from scripts.run_async_memory_benchmark import (
    BenchmarkRunner,
    HttpResponse,
    latency_summary,
)


class FakeMemoryClient:
    base_url = "http://memory.test"

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.complete_on_status = False
        self.lock = threading.Lock()

    def request(self, method: str, path: str, payload: dict | None = None) -> HttpResponse:
        if method == "GET" and path == "/healthz":
            return HttpResponse(200, {"status": "ok"})
        if method == "POST" and path == "/v1/interactions:enqueue":
            assert payload is not None
            with self.lock:
                self.jobs[payload["event_id"]] = {
                    "event_id": payload["event_id"],
                    "status": "pending",
                    "attempt_count": 0,
                }
            return HttpResponse(202, {
                "status": "queued",
                "event_id": payload["event_id"],
                "job": self.jobs[payload["event_id"]].copy(),
            })
        if method == "GET" and path.startswith("/v1/interactions/"):
            event_id = path.split("/", 4)[3]
            with self.lock:
                job = self.jobs[event_id]
                if self.complete_on_status:
                    job["status"] = "completed"
                    job["attempt_count"] = 1
                decisions = []
                if job["status"] == "completed":
                    decisions = [{
                        "memory_type": "long_term",
                        "analyzer_source": "ollama",
                    }]
                return HttpResponse(200, {
                    "job": job.copy(),
                    "decisions": decisions,
                })
        raise AssertionError(f"Unexpected request: {method} {path}")


class AsyncMemoryBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeMemoryClient()
        self.output: list[str] = []
        self.runner = BenchmarkRunner(self.client, output=self.output.append)

    def test_enqueue_measures_concurrent_fast_path_without_storing_text_in_report(self) -> None:
        report = self.runner.enqueue(request_count=8, user_count=2, concurrency=4)

        self.assertTrue(report["enqueue"]["passed"])
        self.assertEqual(report["enqueue"]["successful"], 8)
        self.assertEqual(report["enqueue"]["failed"], 0)
        self.assertEqual(report["enqueue"]["immediate_job_status_counts"], {"pending": 8})
        self.assertEqual(len(report["events"]), 8)
        self.assertNotIn("text", report["events"][0])
        self.assertGreaterEqual(report["enqueue"]["latency"]["p95_ms"], 0)

    def test_inspect_proves_jobs_are_unfinished_while_worker_is_off(self) -> None:
        report = self.runner.enqueue(request_count=3, user_count=1, concurrency=2)

        inspection = self.runner.inspect(report, concurrency=2)

        self.assertEqual(inspection["status_counts"], {"pending": 3})
        self.assertEqual(inspection["status_errors"], 0)
        self.assertTrue(inspection["all_unfinished"])

    def test_drain_reports_completed_jobs_attempts_and_decisions(self) -> None:
        report = self.runner.enqueue(request_count=4, user_count=2, concurrency=2)
        self.client.complete_on_status = True

        result = self.runner.drain(
            report,
            concurrency=2,
            poll_seconds=0.01,
            timeout_seconds=1,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["status_counts"], {"completed": 4})
        self.assertEqual(result["attempts"], {"mean": 1.0, "max": 1, "retried_jobs": 0})
        self.assertEqual(result["decision_type_counts"], {"long_term": 4})
        self.assertEqual(result["analyzer_source_counts"], {"ollama": 4})

    def test_latency_summary_handles_one_value(self) -> None:
        self.assertEqual(latency_summary([0.125]), {
            "min_ms": 125.0,
            "p50_ms": 125.0,
            "p95_ms": 125.0,
            "p99_ms": 125.0,
            "max_ms": 125.0,
        })


if __name__ == "__main__":
    unittest.main()
