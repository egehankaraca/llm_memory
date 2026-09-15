#!/usr/bin/env python3
"""Real-model regression: assert persisted state; expose answers for human review."""

from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conversation_orchestrator import ConversationOrchestrator, OrchestratorError, OrchestratorSettings


def profile_snapshot(coordinator):
    return sorted((item["id"], json.dumps(item["value"], ensure_ascii=False, sort_keys=True)) for item in coordinator.memories()["items"])


def run(settings):
    suffix = uuid.uuid4().hex[:8]
    user_id = f"chat-regression-{suffix}"
    seed = ConversationOrchestrator(settings, user_id, f"seed-{suffix}")
    analyzer = seed.check_services()
    assert analyzer.get("provider") == "ollama" and analyzer.get("available"), "Memory API must use an available Ollama analyzer"
    print(f"User: {user_id} (isolated demo data, API-selected database)", flush=True)
    address = seed.chat("Bana bundan sonra Ahmet Bey diye hitap et")
    assert any(item["status"] == "auto_applied" and item["memory_type"] == "long_term" for item in address.decisions), "Explicit address preference was not stored"
    initial = seed.chat("Her cumartesi sabah saat 7’de tenis oynarım")
    assert all(item.get("analyzer_source") == "ollama" for item in address.decisions + initial.decisions), "Seed used analyzer fallback"
    assert any(item["status"] == "auto_applied" and item["memory_type"] == "long_term" for item in initial.decisions), "Explicit routine was not stored"
    snapshot = profile_snapshot(seed)
    assert len(snapshot) == 2, "Seed must create exactly two facts (address and routine)"
    coordinator = ConversationOrchestrator(settings, user_id, f"new-session-{suffix}")
    cases = [
        ("daily_question", "her sabah kaçta tenis oynarım", "Do not turn the weekly Saturday7 routine into a daily one."),
        ("advice_question", "tenis oynayayım mı?", "Address advice, not just schedule recall; do not infer medical fitness."),
        ("topic_fragment", "tenis", "Interpret this follow-up in context; clarify when needed."),
        ("routine_recall", "Cumartesi tenis saatim kaçtı?", "Recall the stored Saturday7 routine."),
        ("general_question", "İki artı iki kaç eder?", "Answer four; no profile fact is needed."),
    ]
    results = []
    for index, (case_id, text, expectation) in enumerate(cases):
        turn = coordinator.chat(text)
        assert turn.decisions and all(item.get("analyzer_source") == "ollama" for item in turn.decisions), f"{case_id}: analyzer fallback"
        assert profile_snapshot(coordinator) == snapshot, f"{case_id}: profile changed after a question/fragment"
        assert all(item["memory_type"] == "discard" for item in turn.decisions), f"{case_id}: unwanted memory candidate"
        assert coordinator.context()["session"] == {}, f"{case_id}: question/fragment created an active goal"
        if index == 0:
            assert turn.prompt.history_message_count == 0, "New session had history"
        result = {"case": case_id, "reply": turn.reply, "profile_unchanged": True, "session_unchanged": True, "model_stats": turn.model_stats, "expectation": expectation, "needs_human_review": True}
        results.append(result)
        print("STATE PASS / ANSWER REVIEW " + json.dumps(result, ensure_ascii=False), flush=True)
    print(f"STATE PASS: {len(results)}/{len(cases)} cases; no automatic semantic score. Demo records retained for inspection.", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-url", default=None)
    args = parser.parse_args()
    try:
        settings = OrchestratorSettings.from_environment()
        if args.memory_url:
            settings = replace(settings, memory_url=args.memory_url)
        run(settings)
        return 0
    except (OrchestratorError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
