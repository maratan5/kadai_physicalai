#!/usr/bin/env bash
# Execute the official π0.5 BEHAVIOR evaluator. Each invocation gets a fresh output directory.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ ! -f "$ROOT/config.env" ] || source "$ROOT/config.env"
TASK="${1:-installing_a_modem}"
INSTANCE="${2:-0}"
HEADLESS="${3:-0}"
HOST="${KADAI_POLICY_HOST:-127.0.0.1}"
PORT="${KADAI_POLICY_PORT:-8000}"
DIR="${KADAI_BEHAVIOR_DIR:-$ROOT/external/BEHAVIOR-1K}"
CONDA_ENV="${KADAI_CONDA_ENV:-behavior}"
# Existing shell/CLI callers retain config.env defaults. The GUI explicitly
# supplies positional args 4 and 5 to override these settings per run.
MAX_STEPS="${KADAI_MAX_STEPS:-}"
ACTION_CHUNK="${KADAI_ACTION_CHUNK_SIZE:-20}"
if (( $# >= 4 )); then
    if [[ "$4" == auto ]]; then MAX_STEPS=""; else MAX_STEPS="$4"; fi
fi
if (( $# >= 5 )); then ACTION_CHUNK="$5"; fi
CAMERA_MODE="${KADAI_CAMERA_MODE:-follow}"
CAMERA_HEIGHT="${KADAI_CAMERA_HEIGHT:-2.0}"
if (( $# >= 6 )); then CAMERA_MODE="$6"; fi
if (( $# >= 7 )); then CAMERA_HEIGHT="$7"; fi
if [[ ! "$TASK" =~ ^[A-Za-z0-9_]+$ ]] || [[ ! "$INSTANCE" =~ ^([0-9]|1[0-9])$ ]] || [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
    echo 'ERROR: 不正なタスク名・インスタンス・ポートです' >&2
    exit 2
fi
if [[ "$CAMERA_MODE" != fixed_overhead && "$CAMERA_MODE" != follow ]]; then
    echo 'ERROR: 録画カメラはfixed_overhead / follow' >&2
    exit 2
fi
if [[ ! "$CAMERA_HEIGHT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo 'ERROR: 録画カメラの高さは数値で指定してください' >&2
    exit 2
fi
if [[ "$HEADLESS" != 0 && "$HEADLESS" != 1 ]]; then
    echo 'ERROR: headless flagは0/1' >&2
    exit 2
fi
if [[ -n "$MAX_STEPS" && ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ERROR: KADAI_MAX_STEPSは正の整数にしてください' >&2
    exit 2
fi
if [[ ! "$ACTION_CHUNK" =~ ^[1-9][0-9]*$ ]] || (( 10#$ACTION_CHUNK < 1 || 10#$ACTION_CHUNK > 100 )); then
    echo 'ERROR: アクションチャンクは1〜100の整数にしてください' >&2
    exit 2
fi
if [[ -n "$MAX_STEPS" ]] && (( 10#$MAX_STEPS > 100000 )); then
    echo 'ERROR: 最大ステップ数は100000以下にしてください' >&2
    exit 2
fi
if [ ! -f "$DIR/OmniGibson/omnigibson/eval/eval.py" ]; then
    echo "ERROR: BEHAVIOR-1Kの評価器が見つかりません: $DIR" >&2
    echo '先に bash scripts/setup_behavior.sh を実行してください' >&2
    exit 2
fi
if ! command -v conda >/dev/null 2>&1; then echo 'ERROR: condaを見つけられません' >&2; exit 2; fi
if ! command -v curl >/dev/null 2>&1; then echo 'ERROR: curlがありません' >&2; exit 2; fi
printf 'π0.5 policy server health: http://%s:%s/healthz\n' "$HOST" "$PORT"
if ! curl -fsS --max-time 5 "http://$HOST:$PORT/healthz" >/dev/null; then
    echo 'ERROR: モデルサーバーに接続できません。先にrun_policy.sh、またはリモートサーバーを起動してください。' >&2
    exit 2
fi
# Name every run task_name_YYYYMMDD_HHMMSS (with instance and a collision-safe suffix).
STAMP=$(TZ="${KADAI_OUTPUT_TZ:-Asia/Tokyo}" date +%Y%m%d_%H%M%S)
mkdir -p "$ROOT/results"
DIR_RESULTS=$(mktemp -d "$ROOT/results/${TASK}_${STAMP}_instance${INSTANCE}_XXXXXX")
printf '\nTask: %s\nInstance: %s\nOutput: %s\n' "$TASK" "$INSTANCE" "$DIR_RESULTS"
printf 'Videos: %s/videos/\nJSON: %s/json/\nLog: %s/run.log\n\n' "$DIR_RESULTS" "$DIR_RESULTS" "$DIR_RESULTS"
printf 'Max steps: %s | Action chunk: %s | Camera: %s, height %sm\n' "${MAX_STEPS:-公式のタスク別既定値}" "$ACTION_CHUNK" "$CAMERA_MODE" "$CAMERA_HEIGHT"
# Use an extra spectator camera for a presentation-friendly MP4, alongside the
# official head/wrist MP4. Disable via KADAI_OVERVIEW_VIDEO=0 if GPU is tight.
OVERVIEW="${KADAI_OVERVIEW_VIDEO:-1}"
if [[ "$OVERVIEW" != 0 && "$OVERVIEW" != 1 ]]; then
    echo 'ERROR: KADAI_OVERVIEW_VIDEOは0/1にしてください' >&2
    exit 2
fi
# The evaluator wrapper reports completed simulation steps, even with
# KADAI_OVERVIEW_VIDEO=0. It does not alter policy actions or episode length.
EVAL_MODULE=kadai_physicalai.record_eval
export KADAI_OVERVIEW_VIDEO="$OVERVIEW"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ "$OVERVIEW" = 1 ]; then
    export OMNIGIBSON_KEEP_VIEWER_CAMERA=1
    export KADAI_CAMERA_DISTANCE KADAI_CAMERA_SIDE KADAI_CAMERA_OVERHEAD_X KADAI_CAMERA_OVERHEAD_Y
fi
export KADAI_CAMERA_MODE="$CAMERA_MODE" KADAI_CAMERA_HEIGHT="$CAMERA_HEIGHT"
# These flags match the working evaluation invocation used with the published B1K model.
FLAGS=(
    --task-name "$TASK" --mode public_test
    --instance-indices "$INSTANCE" --num-envs 1 --num-rollouts 1
    --host "$HOST" --port "$PORT"
    --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper
    --replay-action-chunk-size "$ACTION_CHUNK"
    --output-dir "$DIR_RESULTS" --write-video
)
# Upstream BooleanOptionalAction defaults to headless=True; pass both modes
# explicitly so the GUI checkbox reflects the actual rendering mode.
if [ "$HEADLESS" = 1 ]; then FLAGS+=(--headless); else FLAGS+=(--no-headless); fi
if [ -n "$MAX_STEPS" ]; then FLAGS+=(--max-steps "$MAX_STEPS"); fi
cd "$DIR"
set +e
conda run --no-capture-output -n "$CONDA_ENV" python -m "$EVAL_MODULE" "${FLAGS[@]}" \
    2>&1 | tee "$DIR_RESULTS/run.log"
RC=${PIPESTATUS[0]}
set -e
# The pinned upstream evaluator may terminate with exit=0 even after it
# printed a traceback during the context-manager shutdown. An evaluation
# without any JSON result is therefore NOT a successful run.
if [[ "$RC" -eq 0 ]] && ! compgen -G "$DIR_RESULTS/json/*.json" >/dev/null; then
    echo 'ERROR: 評価結果JSONがありません。初期化エラーや実行中断の可能性があります。run.logをご確認ください。' | tee -a "$DIR_RESULTS/run.log" >&2
    RC=1
fi
printf '\n実行終了 (exit=%s)。今回の結果: %s\n' "$RC" "$DIR_RESULTS"
exit "$RC"
