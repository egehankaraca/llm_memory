#!/usr/bin/env python3
"""Chat through the Memory Service and Ollama, without STT/TTS or direct DB access."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conversation_orchestrator import (  # noqa: E402
    ChatTurn,
    ConversationOrchestrator,
    OrchestratorError,
    OrchestratorSettings,
    PersistenceError,
)


HELP = """Komutlar:
  /context [soru]  Memory Service context paketini göster
  /memories       Aktif uzun süreli hafızaları göster
  /pending        Bekleyen son 200 adayı göster
  /status ID      Async memory işinin durumunu ve kararlarını göster
  /confirm ID     Gösterilen adayı kalıcı hafızaya açıkça onayla
  /reject ID      Gösterilen adayı reddet
  /retry          Eksik kaydı aynı ID'lerle tamamla; cevabı yeniden üretme
  /help           Komutları göster
  /exit           Çık
Komutlar LLM'e gönderilmez. Sıradan bir 'evet' otomatik hafıza onayı değildir.
"""


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def print_candidates(candidates: list[dict[str, object]]) -> None:
    if not candidates:
        print("Bekleyen hafıza adayı yok.")
        return
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        print("\nKalıcı hafıza için açık onay gerekiyor:")
        print_json({key: candidate.get(key) for key in (
            "candidate_id", "category", "key", "value", "sensitivity", "consolidation_action",
        )})
        print(f"Kaydedilsin mi? /confirm {candidate_id} veya /reject {candidate_id}")


def print_turn(turn: ChatTurn, debug: bool) -> None:
    print(f"\nAsistan: {turn.reply}")
    if turn.model_stats.get("done_reason") == "length":
        print("[Uyarı] Cevap token limitinde durdu; gerekirse CHAT_OLLAMA_NUM_PREDICT artırın.")
    if debug:
        interaction = getattr(turn, "interaction", None) or {}
        print_json({
            "event_id": turn.event_id,
            "assistant_message_id": turn.assistant_message_id,
            "profile_fact_count": turn.prompt.profile_fact_count,
            "history_message_count": turn.prompt.history_message_count,
            "estimated_prompt_tokens": turn.prompt.estimated_tokens,
            "input_budget": turn.prompt.input_budget,
            "prompt_trimmed": turn.prompt.trimmed,
            "model_stats": turn.model_stats,
            "memory_ingestion": (
                interaction.get("job")
                if isinstance(interaction.get("job"), dict)
                else {"mode": "synchronous"}
            ),
            "decisions": [{key: decision.get(key) for key in (
                "candidate_id", "memory_type", "status", "analyzer_source", "consolidation_action",
            )} | {"evidence_guard": (decision.get("analysis") or {}).get("evidence_guard")} for decision in turn.decisions],
        })
    fallback_sources = {
        decision.get("analyzer_source") for decision in turn.decisions
        if decision.get("analyzer_source") in {"rules_fallback", "rules_guard"}
    }
    if fallback_sources:
        print(f"[Uyarı] Hafıza analizinde kural katmanı kullanıldı: {', '.join(sorted(fallback_sources))}")
    if turn.pending_candidates:
        print_candidates(turn.pending_candidates)


def handle_command(orchestrator: ConversationOrchestrator, text: str, debug: bool) -> bool:
    """Return False to leave the REPL. Consent commands bypass LLM extraction."""
    command, _, argument = text.partition(" ")
    argument = argument.strip()
    if command == "/exit":
        return False
    if command == "/help":
        print(HELP)
    elif command == "/context":
        print_json(orchestrator.context(argument or None))
    elif command == "/memories":
        print_json(orchestrator.memories())
    elif command == "/pending":
        print_candidates(orchestrator.pending_candidates())
    elif command == "/status":
        print_json(orchestrator.interaction_status(argument))
    elif command in {"/confirm", "/reject"}:
        result = orchestrator.resolve_candidate(argument, confirm=command == "/confirm")
        if result.get("status") == "confirmed":
            print("Hafıza adayı kullanıcı onayıyla kalıcı olarak kaydedildi.")
        elif result.get("status") == "rejected":
            print("Hafıza adayı reddedildi; uzun süreli profile eklenmedi.")
        else:
            raise OrchestratorError("Beklenmeyen onay/red yanıtı.")
    elif command == "/retry":
        print_turn(orchestrator.retry_pending(), debug)
    else:
        print("Bilinmeyen komut. /help kullanın.")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", help="Aynı ID ile uzun süreli hafızayı tekrar kullanın")
    parser.add_argument("--session-id", help="Aynı ID ile kısa süreli sohbeti sürdürün")
    parser.add_argument("--memory-url", help="Memory Service URL (varsayılan port 8001)")
    parser.add_argument("--ollama-url", help="Cevap Ollama URL")
    parser.add_argument("--model", help="Cevap modeli; extraction modelini değiştirmez")
    parser.add_argument(
        "--text", action="append",
        help="Bir mesaj gönderip çık; birden fazla kez kullanılabilir",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Seçim, token bütçesi ve kayıt bilgilerini göster",
    )
    parser.add_argument(
        "--async-memory",
        action="store_true",
        help="Memory analizini outbox worker'a bırak; cevap yolunu bekletme",
    )
    args = parser.parse_args(argv)
    orchestrator: ConversationOrchestrator | None = None
    try:
        settings = OrchestratorSettings.from_environment()
        settings = replace(settings, **{
            key: value for key, value in {
                "memory_url": args.memory_url, "ollama_url": args.ollama_url, "model": args.model,
            }.items() if value is not None
        })
        if args.async_memory:
            settings = replace(settings, async_memory_ingestion=True)
        suffix = uuid.uuid4().hex[:8]
        orchestrator = ConversationOrchestrator(
            settings,
            args.user_id or f"chat-demo-{suffix}",
            args.session_id or f"chat-session-{suffix}",
        )
        analyzer = orchestrator.check_services()
        print(f"User: {orchestrator.user_id}\nSession: {orchestrator.session_id}")
        print(f"Cevap modeli: {settings.model}")
        print(f"Hafıza analizi: {analyzer.get('provider')} / {analyzer.get('model')}")
        print(
            "Memory ingestion: "
            + ("async outbox/worker" if settings.async_memory_ingestion else "synchronous")
        )
        if analyzer.get("provider") == "ollama" and not analyzer.get("available"):
            print("[Uyarı] Extractor hazır değil; Memory Service kural fallback'i kullanabilir.")
        if args.text:
            for text in args.text:
                print(f"\nSen: {text}")
                print_turn(orchestrator.chat(text), args.debug)
            return 0
        print(HELP)
        print("Bu bir yerel geliştirme demosudur; canlı hava durumu/cihaz/acil arama araçları yoktur.")
        while True:
            text = input("\nSen: ").strip()
            if not text:
                continue
            try:
                if text.startswith("/"):
                    if not handle_command(orchestrator, text, args.debug):
                        break
                else:
                    print_turn(orchestrator.chat(text), args.debug)
            except PersistenceError as exc:
                print(f"\n[Hata] {exc}", file=sys.stderr)
                if orchestrator.pending_turn:
                    print(f"Üretilen fakat tam kaydedilemeyen cevap: {orchestrator.pending_turn.reply}")
            except OrchestratorError as exc:
                print(f"\n[Hata] {exc}", file=sys.stderr)
        return 1 if orchestrator.pending_turn else 0
    except (EOFError, KeyboardInterrupt):
        print("\nÇıkılıyor.")
        return 1 if orchestrator and orchestrator.pending_turn else 0
    except PersistenceError as exc:
        print(f"[Hata] {exc}", file=sys.stderr)
        if orchestrator and orchestrator.pending_turn:
            print(f"Üretilen fakat tam kaydedilemeyen cevap: {orchestrator.pending_turn.reply}")
        return 1
    except OrchestratorError as exc:
        print(f"[Hata] {exc}", file=sys.stderr)
        return 1
    finally:
        if orchestrator and orchestrator.pending_turn:
            print(
                "[Uyarı] Eksik kayıt var. /retry kuyruğu yalnızca bu process'teydi; "
                "kapanınca kalıcı değildir.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    raise SystemExit(main())
