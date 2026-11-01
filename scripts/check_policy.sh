#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ ! -f "$ROOT/config.env" ] || source "$ROOT/config.env"
HOST="${KADAI_POLICY_HOST:-127.0.0.1}"
PORT="${KADAI_POLICY_PORT:-8000}"
curl --fail --show-error --max-time 10 "http://$HOST:$PORT/healthz"
echo
