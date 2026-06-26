"""Run Phase 3A clean baseline discovery on the A6000."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3.src.phase3_runner import run_phase3_prompt_discovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MAT repository root")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if DONE.json exists")
    args = parser.parse_args()
    run_phase3_prompt_discovery(Path(args.root), force=args.force)


if __name__ == "__main__":
    main()

