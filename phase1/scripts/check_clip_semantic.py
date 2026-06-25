"""Preflight CLIP semantic scoring before expensive Phase 1C runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase1.src.phase1c_runner import check_clip_semantic_preflight
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CLIP semantic scorer availability.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit non-zero if CLIP cannot load. Use this before Phase 1C screening.",
    )
    args = parser.parse_args()
    root = project_root(Path(args.root))
    payload = check_clip_semantic_preflight(root, require_available=args.require)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    if payload.get("available"):
        print("CLIP semantic scoring: available")
    else:
        print("CLIP semantic scoring: unavailable")
        print("Diagnostics written to phase1/outputs/summaries/clip_semantic_diagnostics.json")


if __name__ == "__main__":
    main()
