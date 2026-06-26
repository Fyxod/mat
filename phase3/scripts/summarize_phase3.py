"""Summarize Phase 3 clean discovery and breadth-probe outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3.src.phase3_runner import summarize_phase3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MAT repository root")
    args = parser.parse_args()
    summarize_phase3(Path(args.root))


if __name__ == "__main__":
    main()

