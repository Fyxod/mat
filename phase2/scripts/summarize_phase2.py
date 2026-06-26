"""Summarize Phase 2 final-edit search results."""
from __future__ import annotations

import argparse

from phase2.src.phase2_runner import summarize_phase2


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 final-edit CEM outputs.")
    parser.add_argument("--root", required=True, help="MAT project root")
    args = parser.parse_args()
    summarize_phase2(args.root)


if __name__ == "__main__":
    main()

