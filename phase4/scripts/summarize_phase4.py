"""Summarize Phase 4 outputs after A6000 runs."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase4.src.phase4_runner import summarize_phase4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MAT repository root")
    args = parser.parse_args()
    summarize_phase4(Path(args.root))


if __name__ == "__main__":
    main()

