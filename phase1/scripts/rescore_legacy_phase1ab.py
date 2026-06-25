"""Rescore legacy Phase 1A/1B artifacts with final-edit-aware semantics."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase1.src.phase1c_runner import rescore_legacy_phase1ab
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic rescore for legacy Phase 1A/1B candidates.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing rescore outputs")
    args = parser.parse_args()
    root = project_root(Path(args.root))
    rescore_legacy_phase1ab(root, force=args.force)
    print("Wrote phase1/outputs/semantic_rescore")


if __name__ == "__main__":
    main()
