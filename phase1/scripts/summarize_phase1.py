from __future__ import annotations

import argparse
import json

from phase1.src.runners import run_summary
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 1 final summary and report.")
    parser.add_argument("--root", required=True, help="MAT project root")
    args = parser.parse_args()
    print(json.dumps(run_summary(project_root(args.root)), indent=2))


if __name__ == "__main__":
    main()
