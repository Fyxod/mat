#!/usr/bin/env bash
set -euo pipefail

# No sudo is required. Everything defaults to /workspace/mat/.micromamba.
# Override MAT_ENV_PREFIX, MICROMAMBA_BIN, MAMBA_ROOT_PREFIX, or PYTORCH_CUDA
# only when the workstation has a site-specific preferred location.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${REPO_ROOT}/.micromamba/root}"
MAT_ENV_PREFIX="${MAT_ENV_PREFIX:-${REPO_ROOT}/.micromamba/envs/mat-a6000}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu121}"
INSTALL_DEEPFACE="${INSTALL_DEEPFACE:-0}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 2
}

ensure_micromamba() {
  if [[ -x "${MICROMAMBA_BIN}" ]]; then
    return
  fi
  command -v curl >/dev/null 2>&1 || die "curl is required to install micromamba."
  command -v tar >/dev/null 2>&1 || die "tar is required to unpack micromamba."
  local tmp
  tmp="$(mktemp -d)"
  mkdir -p "$(dirname "${MICROMAMBA_BIN}")"
  log "Downloading micromamba to ${MICROMAMBA_BIN}"
  curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "${tmp}/micromamba.tar.bz2"
  tar -xjf "${tmp}/micromamba.tar.bz2" -C "${tmp}"
  mv "${tmp}/bin/micromamba" "${MICROMAMBA_BIN}"
  chmod +x "${MICROMAMBA_BIN}"
  rm -rf "${tmp}"
}

main() {
  cd "${REPO_ROOT}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || log "nvidia-smi is not on PATH; CUDA verification will decide whether this host is ready."
  ensure_micromamba
  export MAMBA_ROOT_PREFIX

  if [[ ! -d "${MAT_ENV_PREFIX}" ]]; then
    log "Creating environment at ${MAT_ENV_PREFIX}"
    "${MICROMAMBA_BIN}" create -y -p "${MAT_ENV_PREFIX}" -c conda-forge python=3.10 pip setuptools wheel
  else
    log "Reusing environment at ${MAT_ENV_PREFIX}"
  fi

  log "Installing CUDA PyTorch (${PYTORCH_CUDA})"
  "${MICROMAMBA_BIN}" run -p "${MAT_ENV_PREFIX}" python -m pip install --upgrade pip setuptools wheel
  "${MICROMAMBA_BIN}" run -p "${MAT_ENV_PREFIX}" python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"

  log "Installing Phase 1 requirements"
  "${MICROMAMBA_BIN}" run -p "${MAT_ENV_PREFIX}" python -m pip install -r requirements.txt

  if [[ "${INSTALL_DEEPFACE}" == "1" ]]; then
    log "Installing optional DeepFace identity package"
    if ! "${MICROMAMBA_BIN}" run -p "${MAT_ENV_PREFIX}" python -m pip install "deepface>=0.0.93"; then
      log "DeepFace did not install cleanly. Phase 1 will still run; final validation records identity as unavailable."
    fi
  else
    log "DeepFace is optional and skipped. Use INSTALL_DEEPFACE=1 bash scripts/install_linux_a6000.sh to enable its final identity panel."
  fi

  log "Checking the installed Python stack (does not download the model)"
  "${MICROMAMBA_BIN}" run -p "${MAT_ENV_PREFIX}" python scripts/check_env.py

  cat <<EOF

Installation complete.

Use this exact runner prefix on the A6000:
  ${MICROMAMBA_BIN} run -p ${MAT_ENV_PREFIX} python

Example:
  ${MICROMAMBA_BIN} run -p ${MAT_ENV_PREFIX} python -m phase1.scripts.a6000_run --root ${REPO_ROOT} --mode smoke

EOF
}

main "$@"
