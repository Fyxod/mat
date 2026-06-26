"""Validate the Phase 3 image bundle without running GPU work."""
from __future__ import annotations

import argparse
from pathlib import Path

from phase3.src.phase3_runner import REQUIRED_PHASE3_FACE_IDS, inspect_phase3_images
from phase3.src.utils import project_root, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MAT repository root")
    parser.add_argument("--check-only", action="store_true", help="Only print validation information")
    args = parser.parse_args()

    root = project_root(Path(args.root))
    payload = inspect_phase3_images(root)
    if not args.check_only:
        write_json(root / "phase3" / "configs" / "phase3_detected_images.json", payload)
    print(f"Detected faces: {payload['detected_face_ids']}")
    print(f"Detected face count: {payload['detected_face_count']}")
    print(f"Prompt-restricted faces: {payload['prompt_restricted_faces']}")
    missing_ids = sorted(REQUIRED_PHASE3_FACE_IDS - set(payload["detected_face_ids"]))
    if missing_ids:
        print("Missing required Phase 3 face IDs:")
        for face_id in missing_ids:
            print(f"- {face_id}")
        raise SystemExit(1)
    if payload["missing_files"]:
        print("Missing files:")
        for item in payload["missing_files"]:
            print(f"- {item}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
