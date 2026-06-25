"""Run Phase 1D deepening only when Phase 1C has semantic candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase1.src.phase1c_runner import run_phase1d_deepen
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1D semantic deepening.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Recompute completed starts")
    args = parser.parse_args()
    root = project_root(Path(args.root))
    rows = run_phase1d_deepen(root, force=args.force)
    print(f"Phase 1D deepening rows: {len(rows)}")
    print("Inspect phase1/outputs/phase1d_deepen/phase1d_decision_report.md")


if __name__ == "__main__":
    main()
