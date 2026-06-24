from __future__ import annotations

import argparse
import json

from phase1.src.runners import run_final_validation
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final image and optional identity validation.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Rebuild completed validation candidates")
    args = parser.parse_args()
    rows = run_final_validation(project_root(args.root), force=args.force)
    print(json.dumps({"validated_candidates": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
