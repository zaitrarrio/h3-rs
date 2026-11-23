#!/usr/bin/env bash
# Toolchain preflight for the tasks in Taskfile.yml. Missing optional GPU
# pieces are reported and do not fail the run: the CPU reference builds
# without them.
set -uo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

failures=0
ok() { printf 'ok   %s\n' "$*"; }
fail() { printf 'FAIL %s\n' "$*"; failures=$((failures + 1)); }
note() { printf 'note %s\n' "$*"; }

require_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "$1 ($("$1" --version 2>&1 | head -n 1))"
  else
    fail "$1 missing"
  fi
}

require_cmd rustc
require_cmd cargo
require_cmd python3

if [[ -f Cargo.lock ]]; then
  ok "Cargo.lock (tasks pass --locked)"
else
  fail "Cargo.lock missing"
fi

lvp="/usr/share/vulkan/icd.d/lvp_icd.json"
if [[ -f "$lvp" ]]; then
  ok "lavapipe ICD at $lvp (task test:lavapipe)"
elif command -v vulkaninfo >/dev/null 2>&1; then
  note "Vulkan is installed; task test uses whatever adapter wgpu finds"
else
  note "no Vulkan ICD found; WebGPU tests skip until a driver or lavapipe is installed"
fi

nvrtc_found=0
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  IFS=':' read -r -a nvrtc_paths <<< "$LD_LIBRARY_PATH"
  for dir in "${nvrtc_paths[@]}"; do
    if compgen -G "$dir/libnvrtc.so*" >/dev/null || compgen -G "$dir/libnvrtc.dylib" >/dev/null; then
      nvrtc_found=1
      ok "libnvrtc on LD_LIBRARY_PATH ($dir)"
      break
    fi
  done
fi
if [[ "$nvrtc_found" -eq 0 ]]; then
  shopt -s nullglob
  candidates=(
    "$root"/.nvrtc/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib
    "$root"/.venv/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib
  )
  shopt -u nullglob
  if ((${#candidates[@]})); then
    for dir in "${candidates[@]}"; do
      if compgen -G "$dir/libnvrtc.so*" >/dev/null || compgen -G "$dir/libnvrtc.dylib" >/dev/null; then
        nvrtc_found=1
        ok "libnvrtc in $dir (task test:cuda exports it)"
        break
      fi
    done
  fi
fi
if [[ "$nvrtc_found" -eq 0 ]]; then
  note "libnvrtc not found; task test:cuda needs the pip wheel nvidia-cuda-nvrtc-cu12==12.8.*"
  note "  python3 -m venv .nvrtc && .nvrtc/bin/pip install 'nvidia-cuda-nvrtc-cu12==12.8.*'"
fi

exit "$failures"
