"""Run Phase 1C final-edit-aligned screening on the A6000."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase1.src.phase1c_runner import run_phase1c_screening
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1C focused screening.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Recompute completed starts")
    args = parser.parse_args()
    root = project_root(Path(args.root))
    rows = run_phase1c_screening(root, force=args.force)
    print(f"Phase 1C screening rows: {len(rows)}")
    print("Inspect phase1/outputs/phase1c_screening/phase1c_semantic_top_sheet.jpg")


if __name__ == "__main__":
    main()
