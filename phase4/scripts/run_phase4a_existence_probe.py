"""Run Phase 4A landmark semantic geometry existence probe."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase4.src.phase4_runner import run_phase4a_existence_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MAT repository root")
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if DONE.json exists")
    args = parser.parse_args()
    run_phase4a_existence_probe(Path(args.root), force=args.force)


if __name__ == "__main__":
    main()

