# Phase 4 real landmark detection report

- Require real landmarks: True
- Total faces checked: 8
- Real MediaPipe landmark faces: 8
- Template fallback faces: 0
- Failed faces: 0

## Backend check

```json
{
  "python_executable": "/home/interns/Desktop/mat/.micromamba/envs/mat-a6000/bin/python",
  "python_version": "3.10.20 | packaged by conda-forge | (main, Jun 11 2026, 03:31:56) [GCC 14.3.0]",
  "mediapipe_imported": false,
  "mediapipe_version": null,
  "mediapipe_file": null,
  "has_solutions": false,
  "has_solutions_face_mesh": false,
  "mp_solutions_face_mesh_ok": false,
  "direct_python_solutions_face_mesh_ok": true,
  "tasks_face_landmarker_import_ok": false,
  "tasks_model_asset_path": null,
  "tasks_model_asset_exists": false,
  "selected_backend": "mediapipe_python_solutions_face_mesh",
  "errors": [
    "mediapipe_import_failed:ImportError:cannot import name 'runtime_version' from 'google.protobuf' (/home/interns/Desktop/mat/.micromamba/envs/mat-a6000/lib/python3.10/site-packages/google/protobuf/__init__.py)",
    "tasks_face_landmarker_import_failed:NameError:name 'framework' is not defined"
  ]
}
```

## Per-face status

- face_001: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
- face_002: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
- face_003: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
- face_004: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
- face_005: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
- face_006: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
- face_007: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
- face_008: success=True detector=mediapipe_python_solutions_face_mesh count=478 real=True failure=None
