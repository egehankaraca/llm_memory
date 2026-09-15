#!/usr/bin/env python3
"""Process durable memory-outbox jobs without blocking the conversation API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_worker import default_worker_id, process_one_outbox_job  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Process at most one job and exit")
    parser.add_argument("--worker-id", default=default_worker_id())
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--lease-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    if args.poll_seconds < 0.05 or args.lease_seconds < 1:
        parser.error("poll-seconds >= 0.05 and lease-seconds >= 1 are required")

    print(f"Memory worker started: {args.worker_id}", flush=True)
    while True:
        result = process_one_outbox_job(
            worker_id=args.worker_id,
            lease_seconds=args.lease_seconds,
        )
        if result is not None:
            print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.once:
            return 1 if result and result.get("error") else 0
        if result is None:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
