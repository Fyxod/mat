"""Summarize Phase 1C/1D semantic attack results."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase1.src.phase1c_runner import summarize_phase1c
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Phase 1C final report.")
    parser.add_argument("--root", required=True, help="MAT project root")
    args = parser.parse_args()
    root = project_root(Path(args.root))
    payload = summarize_phase1c(root)
    print(payload)
    print("Wrote phase1/outputs/summaries/phase1c_final_report.md")


if __name__ == "__main__":
    main()
