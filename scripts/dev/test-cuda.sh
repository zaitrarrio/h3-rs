#!/usr/bin/env bash
# cargo test -p h3-cuda, with libnvrtc on LD_LIBRARY_PATH.
# The kernels compile through NVRTC; a missing library makes the test skip.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

has_nvrtc() {
  local dir="$1"
  compgen -G "$dir/libnvrtc.so*" >/dev/null || compgen -G "$dir/libnvrtc.dylib" >/dev/null
}

found=""
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  IFS=':' read -r -a nvrtc_paths <<< "$LD_LIBRARY_PATH"
  for dir in "${nvrtc_paths[@]}"; do
    if has_nvrtc "$dir"; then
      found="$dir"
      break
    fi
  done
fi

if [[ -z "$found" ]]; then
  shopt -s nullglob
  candidates=(
    "$root"/.nvrtc/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib
    "$root"/.venv/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib
  )
  shopt -u nullglob
  if ((${#candidates[@]})); then
    for dir in "${candidates[@]}"; do
      if has_nvrtc "$dir"; then
        found="$dir"
        export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        break
      fi
    done
  fi
fi

if [[ -z "$found" ]]; then
  for py in "$root/.nvrtc/bin/python3" python3; do
    if [[ -x "$py" ]] || command -v "$py" >/dev/null 2>&1; then
      lib="$("$py" -c 'import pathlib, nvidia.cuda_nvrtc as m; print(pathlib.Path(m.__file__).resolve().parent / "lib")' 2>/dev/null || true)"
      if [[ -n "$lib" ]] && has_nvrtc "$lib"; then
        found="$lib"
        export LD_LIBRARY_PATH="$lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        break
      fi
    fi
  done
fi

if [[ -z "$found" ]]; then
  echo "libnvrtc not loadable. Install the wheel CI uses, then re-run:" >&2
  echo "  python3 -m venv .nvrtc && .nvrtc/bin/pip install 'nvidia-cuda-nvrtc-cu12==12.8.*'" >&2
  exit 1
fi

echo "libnvrtc: $found"
exec cargo test -p h3-cuda --locked -- --nocapture "$@"
