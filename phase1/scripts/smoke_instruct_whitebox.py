from __future__ import annotations

import argparse
import json

from phase1.src.runners import run_smoke
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 3-step true InstructPix2Pix white-box smoke test.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Recompute even if smoke/DONE.json exists")
    args = parser.parse_args()
    print(json.dumps(run_smoke(project_root(args.root), force=args.force), indent=2))


if __name__ == "__main__":
    main()
