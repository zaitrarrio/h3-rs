#!/usr/bin/env bash
# Workspace tests on lavapipe, the software Vulkan adapter CI uses.
# Forces that ICD so a machine with a real GPU still runs the CI path.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

icd="/usr/share/vulkan/icd.d/lvp_icd.json"
if [[ ! -f "$icd" ]]; then
  echo "lavapipe ICD not found at $icd" >&2
  echo "Debian/Ubuntu: sudo apt-get install mesa-vulkan-drivers vulkan-tools" >&2
  exit 1
fi

export VK_ICD_FILENAMES="$icd"
export H3_ALLOW_SOFTWARE=1
export CI=1

exec cargo test --workspace --locked --no-fail-fast -- "$@"
