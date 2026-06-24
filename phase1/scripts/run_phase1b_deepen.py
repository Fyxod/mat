from __future__ import annotations

import argparse
import json

from phase1.src.runners import run_phase1b
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1B deepening on top Phase 1A combinations.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Recompute completed deepening starts")
    args = parser.parse_args()
    rows = run_phase1b(project_root(args.root), force=args.force)
    print(json.dumps({"completed_starts": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
