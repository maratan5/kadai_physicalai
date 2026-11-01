#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [ -f "$ROOT/config.env" ]; then
  set -a
  source "$ROOT/config.env"
  set +a
fi
# Remove the obsolete router left behind when updating an older extracted ZIP.
# It belongs to this project and is no longer referenced by any module.
if [ -f "$ROOT/kadai_physicalai/ollama_router.py" ]; then
  rm -f -- "$ROOT/kadai_physicalai/ollama_router.py"
fi
cd "$ROOT"
exec python3 -m kadai_physicalai gui
