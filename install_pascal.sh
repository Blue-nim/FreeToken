#!/usr/bin/env bash
#
# FreeToken — Pascal (sm_61, GTX 10-series) installer.
#
# CUDA 12.6 (cu126) path: the last PyTorch line that still ships Pascal SASS.
# Installs the BASE package (the cu13-only accel extras — flashinfer / sglang-kernel
# — cannot install against cu126). Builds the kernel-cache with an sm_61 cubin.
#
# Linux:        ./install_pascal.sh
# Windows:      run from an x64 VS2022 developer prompt with CUDA_HOME set to your
#               CUDA 12.6 toolkit, then: uv pip install --index-strategy unsafe-best-match -r requirements.txt
#               (this script prints that instruction and bails if cl.exe is missing).
set -euo pipefail

PY_VERSION="${FREETOKEN_PY_VERSION:-3.12}"
VENV_DIR="${FREETOKEN_VENV:-.venv}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# --- Windows: require the MSVC + CUDA build environment -------------------
case "$(uname -s 2>/dev/null)" in
  MINGW*|MSYS*|CYGWIN*)
    if ! command -v cl.exe >/dev/null 2>&1; then
      echo "[install_pascal] cl.exe not found — open an 'x64 Native Tools Command Prompt"
      echo "for VS 2022', set CUDA_HOME to your CUDA 12.6 toolkit, then run:"
      echo
      echo "    uv pip install --index-strategy unsafe-best-match -r \"$REPO_ROOT/requirements.txt\""
      echo
      echo "(A CUDA toolkit + nvcc are needed because FreeToken JIT-compiles its kernels.)"
      exit 1
    fi
    ;;
esac

echo "==> Creating venv at $VENV_DIR (python $PY_VERSION)"
if command -v uv >/dev/null 2>&1; then
  uv venv "$VENV_DIR" --python "$PY_VERSION"
else
  python -m venv "$VENV_DIR"
fi

# Activate (bin on Linux/macOS, Scripts on Windows).
# shellcheck disable=SC1091
if [ -f "$VENV_DIR/bin/activate" ]; then
  source "$VENV_DIR/bin/activate"
else
  source "$VENV_DIR/Scripts/activate"
fi

# Be explicit so a stray FREETOKEN_KERNEL_CACHE_ARCHES cannot drop sm_61.
export FREETOKEN_KERNEL_CACHE_ARCHES="${FREETOKEN_KERNEL_CACHE_ARCHES:-6.1 8.0 8.6 8.9 9.0 10.0 12.0}"

echo "==> Installing FreeToken (Pascal / cu126) from requirements.txt"
if command -v uv >/dev/null 2>&1; then
  uv pip install --index-strategy unsafe-best-match -r "$REPO_ROOT/requirements.txt"
else
  pip install --index-strategy unsafe-best-match -r "$REPO_ROOT/requirements.txt"
fi

echo
echo "==> FreeToken Pascal install complete."
echo "    Activate:  source $VENV_DIR/bin/activate   (or Scripts/activate on Windows)"
echo "    Verify:    ft --version"
