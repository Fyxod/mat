"""Detect Phase 4 landmarks on data/face_*/instruct_512.png."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase4.src.phase4_runner import detect_phase4_landmarks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MAT repository root")
    parser.add_argument("--force", action="store_true", help="Regenerate landmark outputs")
    parser.add_argument("--require-real-landmarks", action="store_true", help="Fail instead of falling back to the 24-point template")
    parser.add_argument("--dry-run", action="store_true", help="Try detection without writing outputs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of face folders for a quick local test")
    args = parser.parse_args()
    detect_phase4_landmarks(
        Path(args.root),
        force=args.force,
        dry_run=args.dry_run,
        limit=args.limit,
        require_real_landmarks=args.require_real_landmarks,
    )


if __name__ == "__main__":
    main()
