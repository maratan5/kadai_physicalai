#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ ! -f "$ROOT/config.env" ] || source "$ROOT/config.env"
if ! command -v docker >/dev/null 2>&1; then echo 'ERROR: Dockerが必要です' >&2; exit 1; fi
if ! docker info >/dev/null 2>&1; then echo 'ERROR: Docker daemon に接続できません。Dockerを起動してください' >&2; exit 1; fi
if ! command -v nvidia-smi >/dev/null 2>&1; then echo 'WARNING: nvidia-smiが見つかりません。NVIDIA GPUを確認してください。' >&2; fi
PORT="${KADAI_POLICY_PORT:-8000}"
BIND="${KADAI_POLICY_BIND:-127.0.0.1}"
IMAGE='ghcr.io/vivekchandra1324/b1k-pi05-policy:135199'
if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then echo 'ERROR: Invalid port' >&2; exit 2; fi
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$VRAM" =~ ^[0-9]+$ ]] && (( VRAM < 24000 )); then
        echo "WARNING: GPU VRAM ${VRAM} MiB。公開モデルは24GB GPU用です。" >&2
        if [ "${KADAI_ALLOW_UNTESTED_16GB:-0}" != 1 ]; then
            echo 'このGPUで実行するには KADAI_ALLOW_UNTESTED_16GB=1 が必要です（動作保証なし）。' >&2
            echo '推奨: 24GB GPUを搭載した別PCでサーバーを動かして接続。' >&2
            exit 3
        fi
    fi
fi
printf 'Upstream image: %s\nURL: http://%s:%s/healthz\n' "$IMAGE" "$BIND" "$PORT"
echo '初回は大容量Dockerイメージをダウンロードします。'
exec docker run --rm --gpus all --name "kadai-pi05-$$" -p "$BIND:$PORT:8000" "$IMAGE"
