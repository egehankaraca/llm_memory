#!/usr/bin/env python3
"""Start the local Memory Service demo with one command and one terminal."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from typing import IO
from urllib import error, request
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_chat_demo import main as run_chat  # noqa: E402
from scripts.run_memory_terminal_demo import main as run_terminal  # noqa: E402


def http_json(url: str, *, timeout: float = 1.0) -> dict | None:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def available_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])


def tail(path: Path, lines: int = 20) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def start_process(
    command: list[str],
    *,
    environment: dict[str, str],
    log: IO[str],
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_api(
    process: subprocess.Popen[str],
    health_url: str,
    log_path: Path,
    timeout_seconds: int = 20,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Memory API stopped during startup.\n" + tail(log_path)
            )
        health = http_json(health_url)
        if health and health.get("status") == "ok":
            return
        time.sleep(0.2)
    raise RuntimeError("Memory API did not become healthy.\n" + tail(log_path))


def check_ollama(
    environment: dict[str, str],
    *,
    extra_models: set[str] | None = None,
) -> None:
    base_url = environment["OLLAMA_BASE_URL"].rstrip("/")
    payload = http_json(f"{base_url}/api/tags", timeout=2.0)
    if payload is None:
        raise RuntimeError(
            f"Ollama is not reachable at {base_url}. Start Ollama and try again."
        )
    installed = {
        str(model.get("name"))
        for model in payload.get("models", [])
        if isinstance(model, dict)
    }
    required = {
        environment["OLLAMA_MODEL"],
        environment["MEMORY_PROFILE_EMBEDDING_MODEL"],
    } | (extra_models or set())
    missing = sorted(model for model in required if model not in installed)
    if missing:
        commands = "\n".join(f"  ollama pull {model}" for model in missing)
        raise RuntimeError("Required Ollama model(s) are missing:\n" + commands)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", default="demo-user-001")
    parser.add_argument(
        "--session-id",
        default=f"demo-session-{uuid.uuid4().hex[:8]}",
    )
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--wait-seconds", type=int, default=120)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Open the Ollama conversation demo instead of the memory-only UI",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CHAT_OLLAMA_MODEL", "gemma4:12b"),
        help="Shared chat and memory-extraction model (default: gemma4:12b)",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "postgresql+psycopg2:///memory_db"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.wait_seconds < 1:
        raise SystemExit("--wait-seconds must be positive")

    environment = os.environ.copy()
    environment["DATABASE_URL"] = args.database_url
    environment.setdefault("MEMORY_ANALYZER_PROVIDER", "ollama")
    environment.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    # The one-command demo intentionally uses one generative model for both
    # chat and asynchronous extraction. The embedding model remains separate.
    environment["OLLAMA_MODEL"] = args.model
    environment["CHAT_OLLAMA_MODEL"] = args.model
    environment.setdefault("MEMORY_PROFILE_SEMANTIC_PROVIDER", "ollama")
    environment.setdefault(
        "MEMORY_PROFILE_EMBEDDING_MODEL", "embeddinggemma:latest"
    )
    environment["MEMORY_ASYNC_INGESTION"] = "true"

    api_process: subprocess.Popen[str] | None = None
    worker_process: subprocess.Popen[str] | None = None
    try:
        print("Checking Ollama…", flush=True)
        check_ollama(
            environment,
            extra_models={args.model} if args.chat else None,
        )
        print("Applying database migrations…", flush=True)
        migration = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        if migration.returncode != 0:
            detail = migration.stderr.strip() or migration.stdout.strip()
            raise RuntimeError("Database migration failed:\n" + detail)

        port = available_port(args.port)
        memory_url = f"http://127.0.0.1:{port}"
        environment["MEMORY_SERVICE_URL"] = memory_url
        if port != args.port:
            print(f"Port {args.port} is busy; using {port} for this demo.", flush=True)

        with tempfile.TemporaryDirectory(prefix="memory-demo-") as runtime_dir:
            runtime_path = Path(runtime_dir)
            api_log_path = runtime_path / "api.log"
            worker_log_path = runtime_path / "worker.log"
            with api_log_path.open("w") as api_log, worker_log_path.open("w") as worker_log:
                api_process = start_process(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    environment=environment,
                    log=api_log,
                )
                wait_for_api(
                    api_process,
                    f"{memory_url}/healthz",
                    api_log_path,
                )
                worker_process = start_process(
                    [
                        sys.executable,
                        "scripts/run_memory_worker.py",
                        "--poll-seconds",
                        "0.25",
                    ],
                    environment=environment,
                    log=worker_log,
                )
                time.sleep(0.2)
                if worker_process.poll() is not None:
                    raise RuntimeError(
                        "Memory worker stopped during startup.\n"
                        + tail(worker_log_path)
                    )

                print(
                    "\nChat demo ready ✓" if args.chat else "\nMemory demo ready ✓"
                )
                if args.chat:
                    chat_args = [
                        "--user-id",
                        args.user_id,
                        "--session-id",
                        args.session_id,
                        "--memory-url",
                        memory_url,
                        "--ollama-url",
                        environment["OLLAMA_BASE_URL"],
                        "--model",
                        args.model,
                        "--async-memory",
                    ]
                    if args.debug:
                        chat_args.append("--debug")
                    return run_chat(chat_args)
                terminal_args = [
                    "--user-id",
                    args.user_id,
                    "--session-id",
                    args.session_id,
                    "--memory-url",
                    memory_url,
                    "--async-memory",
                    "--simple",
                    "--query-before-store",
                    "--auto-wait-seconds",
                    str(args.wait_seconds),
                ]
                if args.debug:
                    terminal_args.append("--debug")
                return run_terminal(terminal_args)
    except (RuntimeError, OSError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1
    finally:
        stop_process(worker_process)
        stop_process(api_process)


if __name__ == "__main__":
    raise SystemExit(main())
