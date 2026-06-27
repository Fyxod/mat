#!/usr/bin/env bash
set -euo pipefail

# Repair the A6000 environment only if the installed MediaPipe package cannot
# expose the legacy Face Mesh API used by Phase 4C. This avoids rebuilding the
# whole environment and leaves the rest of MAT untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
MAT_ENV_PREFIX="${MAT_ENV_PREFIX:-${REPO_ROOT}/.micromamba/envs/mat-a6000}"

run_python() {
  "${MICROMAMBA_BIN}" run -p "${MAT_ENV_PREFIX}" python "$@"
}

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

test_facemesh() {
  run_python - <<'PY'
import importlib
import json
import sys

payload = {
    "python": sys.version.replace("\n", " "),
    "ok": False,
    "mediapipe_version": None,
    "mediapipe_file": None,
    "mp_solutions_ok": False,
    "direct_import_ok": False,
    "errors": [],
}
try:
    import mediapipe as mp
    payload["mediapipe_version"] = getattr(mp, "__version__", None)
    payload["mediapipe_file"] = getattr(mp, "__file__", None)
    try:
        getattr(mp.solutions.face_mesh, "FaceMesh")
        payload["mp_solutions_ok"] = True
    except Exception as exc:
        payload["errors"].append(f"mp.solutions failed: {type(exc).__name__}: {exc}")
except Exception as exc:
    payload["errors"].append(f"import mediapipe failed: {type(exc).__name__}: {exc}")

try:
    module = importlib.import_module("mediapipe.python.solutions.face_mesh")
    getattr(module, "FaceMesh")
    payload["direct_import_ok"] = True
except Exception as exc:
    payload["errors"].append(f"direct import failed: {type(exc).__name__}: {exc}")

payload["ok"] = bool(payload["mp_solutions_ok"] or payload["direct_import_ok"])
print(json.dumps(payload, indent=2))
raise SystemExit(0 if payload["ok"] else 1)
PY
}

main() {
  cd "${REPO_ROOT}"
  if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
    printf 'ERROR: micromamba not found at %s\n' "${MICROMAMBA_BIN}" >&2
    exit 2
  fi
  if [[ ! -d "${MAT_ENV_PREFIX}" ]]; then
    printf 'ERROR: MAT env not found at %s\n' "${MAT_ENV_PREFIX}" >&2
    exit 2
  fi

  log "Current Python / MediaPipe Face Mesh import test"
  if test_facemesh; then
    log "MediaPipe Face Mesh is already usable; no reinstall needed."
  else
    log "MediaPipe Face Mesh legacy API is unavailable. Pinning mediapipe==0.10.14."
    "${MICROMAMBA_BIN}" run -p "${MAT_ENV_PREFIX}" python -m pip install --no-cache-dir --force-reinstall \
      "protobuf>=4.25.3,<5" \
      "mediapipe==0.10.14"
    log "Re-testing MediaPipe Face Mesh after pin"
    test_facemesh
  fi

  log "Running MAT Phase 4 MediaPipe checker"
  run_python -m phase4.scripts.check_mediapipe_landmarks --root "${REPO_ROOT}" --require-real
}

main "$@"
