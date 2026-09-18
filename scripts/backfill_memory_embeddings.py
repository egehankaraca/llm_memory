#!/usr/bin/env python3
"""Queue durable pgvector embedding jobs for active long-term memories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import SessionLocal  # noqa: E402
from memory_embeddings import embedding_status, enqueue_missing_embeddings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", help="Only queue one user's active facts")
    args = parser.parse_args(argv)

    with SessionLocal() as db:
        queued = enqueue_missing_embeddings(db, user_id=args.user_id)
        status = embedding_status(db, user_id=args.user_id)
    print(json.dumps({"queued": queued, **status}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
