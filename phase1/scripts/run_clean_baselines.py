from __future__ import annotations

import argparse
import json

from phase1.src.runners import run_clean_baselines
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clean baselines for selected prompt/settings.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Regenerate completed baselines")
    parser.add_argument("--identity", action="store_true", help="Try optional DeepFace SFace for baselines")
    args = parser.parse_args()
    rows = run_clean_baselines(project_root(args.root), force=args.force, identity_enabled=args.identity)
    print(json.dumps({"baseline_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
