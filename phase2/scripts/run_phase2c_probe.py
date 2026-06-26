"""Run the Phase 2C targeted headphone failure probe on the A6000."""
from __future__ import annotations

import argparse

from phase2.src.phase2c_runner import run_phase2c_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2C targeted headphone failure probe.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Recompute completed Phase 2C outputs")
    args = parser.parse_args()
    run_phase2c_probe(args.root, force=args.force)


if __name__ == "__main__":
    main()
