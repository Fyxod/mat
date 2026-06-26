"""Run focused Phase 2B CEM if Phase 2A found a promising candidate."""
from __future__ import annotations

import argparse

from phase2.src.phase2_runner import run_phase2b_cem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2B focused final-edit geometric CEM.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Recompute completed Phase 2B combos")
    args = parser.parse_args()
    run_phase2b_cem(args.root, force=args.force)


if __name__ == "__main__":
    main()

