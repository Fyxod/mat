"""Summarize Phase 2C targeted headphone failure probe results."""
from __future__ import annotations

import argparse

from phase2.src.phase2c_runner import summarize_phase2c


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 2C targeted headphone failure probe outputs.")
    parser.add_argument("--root", required=True, help="MAT project root")
    args = parser.parse_args()
    summarize_phase2c(args.root)


if __name__ == "__main__":
    main()
