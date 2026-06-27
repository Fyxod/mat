"""Check which MediaPipe Face Mesh backend Phase 4 landmark detection will use."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from phase4.src.landmarks import detect_landmarks_for_image, is_real_landmark_record, mediapipe_backend_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MAT repository root")
    parser.add_argument("--require-real", action="store_true", help="Exit nonzero if real dense MediaPipe landmarks are unavailable")
    parser.add_argument("--sample-face", default=None, help="Optional face_id to test, e.g. face_002")
    args = parser.parse_args()

    root = Path(args.root)
    report = mediapipe_backend_report()
    print(json.dumps(report, indent=2))
    backend = report.get("selected_backend") or "template"
    print(f"Phase 4 landmark backend that will be tried first: {backend}")

    face_id = args.sample_face
    if face_id is None:
        faces = sorted(path.name for path in (root / "data").glob("face_*") if (path / "instruct_512.png").exists())
        face_id = faces[0] if faces else None

    sample_real = False
    if face_id:
        image_path = root / "data" / face_id / "instruct_512.png"
        try:
            record, warning = detect_landmarks_for_image(image_path, prefer_mediapipe=True, require_real_landmarks=False)
            sample_real = is_real_landmark_record({"success": True, **record})
            print(
                "Sample detection: "
                f"face_id={face_id} detector={record.get('detector')} "
                f"landmark_count={record.get('landmark_count')} real={sample_real} warning={warning}"
            )
        except Exception as error:
            print(f"Sample detection failed for {face_id}: {type(error).__name__}: {error}")
    else:
        print("No data/face_*/instruct_512.png image found for sample detection.")

    backend_ok = backend != "template"
    if args.require_real and not (backend_ok and sample_real):
        print("Real dense MediaPipe landmarks are not available. Run scripts/fix_mediapipe_legacy_a6000.sh and retry.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
