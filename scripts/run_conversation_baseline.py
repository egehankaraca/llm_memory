"""Review conversational behavior without calling Memory Service or a database."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversation_orchestrator import (
    JsonHttpClient,
    OrchestratorError,
    OrchestratorSettings,
    build_chat_prompt,
    parse_reply,
)

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "evals" / "conversation_scenarios.jsonl"
PRESETS = {
    "current": {},
    # Diagnostic non-thinking sampling values from the official Qwen3-8B card.
    "qwen3-non-thinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0},
}


def load_scenarios(path: Path, selected: list[str] | None = None) -> list[dict[str, Any]]:
    scenarios = []
    identifiers = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                scenario = json.loads(line)
            except ValueError as exc:
                raise OrchestratorError(f"Dataset satır {line_number}: geçersiz JSON.") from exc
            if (
                not isinstance(scenario, dict)
                or not isinstance(scenario.get("id"), str)
                or not scenario["id"].strip()
                or scenario["id"] in identifiers
                or not isinstance(scenario.get("context"), dict)
                or not isinstance(scenario.get("turns"), list)
                or not 1 <= len(scenario["turns"]) <= 8
            ):
                raise OrchestratorError(f"Dataset satır {line_number}: id/context/turns geçersiz.")
            for turn in scenario["turns"]:
                if not isinstance(turn, dict) or any(
                    not isinstance(turn.get(key), str) or not turn[key].strip()
                    for key in ("text", "expectation")
                ):
                    raise OrchestratorError(f"Dataset satır {line_number}: text/expectation gerekli.")
            identifiers.add(scenario["id"])
            scenarios.append(scenario)
    if not scenarios:
        raise OrchestratorError("Dataset boş.")
    if selected:
        missing = set(selected) - identifiers
        if missing:
            raise OrchestratorError(f"Bilinmeyen scenario: {', '.join(sorted(missing))}")
        scenarios = [scenario for scenario in scenarios if scenario["id"] in set(selected)]
    if sum(len(scenario["turns"]) for scenario in scenarios) > 50:
        raise OrchestratorError("Tek çalıştırmada en fazla 50 tur; --scenario ile alt küme seçin.")
    return scenarios


def run_baseline(
    scenarios: list[dict[str, Any]],
    settings: OrchestratorSettings,
    *,
    ollama_http: JsonHttpClient | None = None,
    preset: str = "current",
) -> dict[str, Any]:
    """One completion per turn; no extraction, memory writes, retries, or fallback answers."""
    if preset not in PRESETS:
        raise OrchestratorError(f"Bilinmeyen preset: {preset}")
    options = {**settings.generation_options, **PRESETS[preset]}
    ollama = ollama_http or JsonHttpClient(settings.ollama_url, settings.ollama_timeout_seconds)
    results = []
    for scenario in scenarios:
        context = deepcopy(scenario["context"])
        history = list(context.get("recent_messages") or [])
        interrupted = False
        for index, turn in enumerate(scenario["turns"], 1):
            entry: dict[str, Any] = {
                "scenario_id": scenario["id"],
                "turn": index,
                "text": turn["text"],
                "expectation": turn["expectation"],
                "history_before": deepcopy(history),
                "answer": None,
                "model_stats": {},
                "needs_human_review": True,
                "error": None,
            }
            if interrupted:
                entry["outcome"] = "skipped_previous_error"
                entry["needs_human_review"] = False
                results.append(entry)
                continue
            started = time.monotonic()
            entry["request_sent"] = False
            try:
                context["recent_messages"] = deepcopy(history)
                prompt = build_chat_prompt(context, turn["text"], settings)
                entry["history_sent"] = deepcopy(prompt.messages[1:-1])
                entry["prompt_stats"] = {
                    "estimated_tokens": prompt.estimated_tokens,
                    "input_budget": prompt.input_budget,
                    "profile_fact_count": prompt.profile_fact_count,
                    "history_message_count": prompt.history_message_count,
                    "trimmed": prompt.trimmed,
                }
                entry["request_sent"] = True
                response = ollama.request("POST", "/api/chat", {
                    "model": settings.model,
                    "messages": prompt.messages,
                    "stream": False,
                    "think": False,
                    "keep_alive": settings.keep_alive,
                    "options": options,
                })
                entry["model_stats"] = {
                    key: response.get(key)
                    for key in ("prompt_eval_count", "eval_count", "done_reason")
                }
                message = response.get("message")
                if not isinstance(message, dict) or message.get("role", "assistant") != "assistant":
                    raise OrchestratorError("Ollama assistant mesajı döndürmedi.")
                content = message.get("content")
                if response.get("done") is False:
                    raise OrchestratorError("Ollama cevabı tamamlanmadı.")
                if not isinstance(content, str):
                    raise OrchestratorError("Ollama cevabı metin değil.")
                answer = parse_reply(content)
                entry["answer"] = answer
                entry["outcome"] = "valid"
                history.extend([
                    {"role": "user", "content": turn["text"]},
                    {"role": "assistant", "content": answer},
                ])
            except OrchestratorError as exc:
                entry["outcome"] = "error"
                entry["error"] = str(exc)
                interrupted = True
            entry["duration_seconds"] = round(time.monotonic() - started, 3)
            results.append(entry)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.model,
        "ollama_url": settings.ollama_url,
        "preset": preset,
        "effective_options": options,
        "mode": "synthetic_context_no_memory_service",
        "memory_requests": 0,
        "database_writes": 0,
        "semantic_score": None,
        "review_note": "Valid means text/transport only. Read expectations and answers; no semantic pass percentage is calculated.",
        "summary": {
            "scenarios": len(scenarios),
            "planned_turns": len(results),
            "requests": sum(entry.get("request_sent", False) for entry in results),
            "valid": sum(entry["outcome"] == "valid" for entry in results),
            "errors": sum(entry["outcome"] == "error" for entry in results),
            "skipped": sum(entry["outcome"] == "skipped_previous_error" for entry in results),
            "needs_human_review": sum(entry["needs_human_review"] for entry in results),
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenario", action="append", help="Repeat to select scenarios.")
    parser.add_argument("--json-report", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--ollama-url")
    parser.add_argument("--preset", choices=PRESETS, default="current")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request seconds (1–60).")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.timeout <= 60:
        parser.error("--timeout must be between 1 and 60 seconds")
    try:
        settings = replace(
            OrchestratorSettings.from_environment(),
            ollama_timeout_seconds=arguments.timeout,
            **({"model": arguments.model} if arguments.model else {}),
            **({"ollama_url": arguments.ollama_url} if arguments.ollama_url else {}),
        )
        scenarios = load_scenarios(arguments.dataset, arguments.scenario)
        print(f"Model: {settings.model}; synthetic context only, no Memory API/DB writes.", flush=True)
        report = run_baseline(scenarios, settings, preset=arguments.preset)
        for entry in report["results"]:
            print(f"\n{entry['scenario_id']} / {entry['turn']} [{entry['outcome']}]")
            print(f"User: {entry['text']}")
            print(f"Answer: {entry['answer'] or entry['error'] or '(skipped)'}")
            print(f"Review: {entry['expectation']}")
        print("\nMechanical summary:", json.dumps(report["summary"], ensure_ascii=False))
        print("No semantic accuracy score: inspect the answers and expectations.")
        if arguments.json_report:
            arguments.json_report.parent.mkdir(parents=True, exist_ok=True)
            arguments.json_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"JSON report: {arguments.json_report}")
        return 1 if report["summary"]["errors"] else 0
    except (OrchestratorError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
