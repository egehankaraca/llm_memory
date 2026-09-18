"""Small JSON HTTP client shared by Memory Service command-line tools."""

from __future__ import annotations

import json
from typing import Any
from urllib import error, parse, request


class MemoryClientError(RuntimeError):
    """Safe, actionable Memory Service transport or response failure."""


class JsonHttpClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        parsed = parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Memory URL must be HTTP(S) without credentials, query, or fragment"
            )
        if timeout_seconds <= 0:
            raise ValueError("HTTP timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        encoded = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        call = request.Request(
            self.base_url + path,
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(call, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail: str | None = None
            try:
                body = exc.read(4096)
                payload = json.loads(body.decode("utf-8"))
                candidate = payload.get("detail") if isinstance(payload, dict) else None
                if isinstance(candidate, str) and candidate.strip():
                    detail = candidate.strip()[:500]
            except (OSError, UnicodeError, ValueError):
                detail = None
            finally:
                exc.close()
            suffix = f": {detail}" if detail is not None else ""
            raise MemoryClientError(
                f"{method} {path}: HTTP {exc.code}{suffix}"
            ) from exc
        except (error.URLError, OSError, TimeoutError) as exc:
            raise MemoryClientError(
                f"{method} {path}: Memory Service unavailable or timed out"
            ) from exc
        except (UnicodeError, ValueError) as exc:
            raise MemoryClientError(f"{method} {path}: invalid JSON response") from exc
        if not isinstance(result, dict):
            raise MemoryClientError(f"{method} {path}: expected a JSON object")
        return result
