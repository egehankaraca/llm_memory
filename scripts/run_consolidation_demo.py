#!/usr/bin/env python3
"""Exercise create and automatic supersession through the live API."""

from __future__ import annotations

import argparse
import json
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import uuid


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    request = urllib_request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def process(
    base_url: str,
    user_id: str,
    session_id: str,
    text: str,
) -> dict[str, object]:
    response = request_json(
        base_url,
        "POST",
        "/v1/interactions:process",
        {
            "event_id": str(uuid.uuid4()),
            "user_id": user_id,
            "session_id": session_id,
            "text": text,
        },
    )
    return response["decisions"][0]


def print_step(title: str, decision: dict[str, object]) -> None:
    print(f"\n{title}")
    print(
        json.dumps(
            {
                "value": decision["value"],
                "consolidation_action": decision["consolidation_action"],
                "consolidates_fact_id": decision["consolidates_fact_id"],
                "status": decision["status"],
                "applied_ref": decision["applied_ref"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--user-id", default=None)
    args = parser.parse_args()

    suffix = uuid.uuid4().hex[:8]
    user_id = args.user_id or f"consolidation-demo-{suffix}"
    session_id = f"consolidation-session-{suffix}"
    print(f"User: {user_id}")
    print(f"Session: {session_id}")

    created = process(
        args.base_url,
        user_id,
        session_id,
        "Sabahları saat 7’de kalkarım",
    )
    print_step("1. CREATE", created)
    assert created["consolidation_action"] == "create"
    assert created["status"] == "auto_applied"

    updated = process(
        args.base_url,
        user_id,
        session_id,
        "Artık sabahları saat 8’de kalkıyorum",
    )
    print_step("2. EXPLICIT UPDATE", updated)
    assert updated["consolidation_action"] == "supersede"
    assert updated["status"] == "auto_applied"

    latest = process(
        args.base_url,
        user_id,
        session_id,
        "Sabahları saat 9’da kalkıyorum",
    )
    print_step("3. LATEST EXPLICIT VALUE", latest)
    assert latest["consolidation_action"] == "supersede"
    assert latest["status"] == "auto_applied"

    encoded_user = urllib_parse.quote(user_id, safe="")
    history = request_json(
        args.base_url,
        "GET",
        f"/v1/users/{encoded_user}/memories?include_inactive=true",
    )
    facts = history["items"]
    active = [fact for fact in facts if fact["status"] == "active"]
    superseded = [fact for fact in facts if fact["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 2

    print("\nPASS")
    print(f"Active value: {active[0]['value']}")
    print(f"Superseded facts: {len(superseded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
