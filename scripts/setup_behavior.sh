#!/usr/bin/env bash
# Install BEHAVIOR / OmniGibson into a separately managed conda env.
# The project checkout stays under external/; never delete existing checkouts.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$ROOT/config.env" ]]; then
    # Optional local project-specific settings; user-maintained file.
    source "$ROOT/config.env"
fi
DIR="${KADAI_BEHAVIOR_DIR:-$ROOT/external/BEHAVIOR-1K}"
# The v3.9.3 tag mentioned in official docs was not present on GitHub on
# 2026-11-01. v3.9.3-post2 is a published release tag for the same version line.
REF="${KADAI_BEHAVIOR_REF:-v3.9.3-post2}"
REPO='https://github.com/StanfordVL/BEHAVIOR-1K.git'
mkdir -p "$(dirname "$DIR")"
for name in git conda; do
    if ! command -v "$name" >/dev/null 2>&1; then
        echo "ERROR: $name がありません。OmniGibsonの公式セットアップに必要です。" >&2
        exit 1
    fi
done

if [ -d "$DIR/.git" ]; then
    echo "既存のBEHAVIOR checkoutを再利用: $DIR"
    INSTALLED_REF=$(git -C "$DIR" describe --tags --exact-match 2>/dev/null || true)
    if [ "$INSTALLED_REF" != "$REF" ]; then
        echo "WARNING: 現在のtag=${INSTALLED_REF:-unknown}、推奨=$REF。既存checkoutは変更しません。" >&2
    fi
else
    if [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ]; then
        echo "ERROR: $DIR にgit管理されていないファイルがあります。勝手に削除しません。" >&2
        echo 'KADAI_BEHAVIOR_DIRで別の新規ディレクトリを指定してください。' >&2
        exit 1
    fi
    echo "BEHAVIOR-1K ($REF) を $DIR にcloneします"
    if ! git clone --branch "$REF" --depth 1 "$REPO" "$DIR"; then
        echo "ERROR: リリースタグ '$REF' をcloneできませんでした。" >&2
        echo "確認: git ls-remote --tags $REPO" >&2
        echo '別タグ: KADAI_BEHAVIOR_REF=v3.9.3-post1 bash scripts/setup_behavior.sh' >&2
        exit 1
    fi
fi
if [ ! -f "$DIR/setup.sh" ]; then
    echo "ERROR: BEHAVIOR setup.sh が見つかりません: $DIR" >&2
    exit 1
fi
# For the pretrained-policy evaluator, the Python 3.12-only LeRobot
# *dataset* package is not needed. The upstream OmniGibson package also
# imports this package eagerly from envs/__init__.py, so patch both that
# import and the setup.py dependency, preserving a backup of each file.
if [ "${KADAI_SKIP_LEROBOT:-1}" = 1 ]; then
    python3 "$ROOT/scripts/patch_omnigibson_lerobot.py" "$DIR"
else
    echo 'WARNING: LeRobotを含める指定です。Python3.11との衝突が再発する可能性があります。' >&2
fi

# Install in a dedicated Python 3.11 conda environment, never touch system Python.
# Avoid the upstream --new-env mode on retries: it errors if behavior already exists.
CONDA_ENV="${KADAI_CONDA_ENV:-behavior}"
if [[ ! "$CONDA_ENV" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo 'ERROR: 不正なConda環境名です' >&2
    exit 2
fi
ENV_EXISTS=$(conda env list --json | python3 -c 'import json,sys,os; name=sys.argv[1]; data=json.load(sys.stdin); print("yes" if any(os.path.basename(os.path.normpath(x))==name for x in data["envs"]) else "no")' "$CONDA_ENV")
if [ "$ENV_EXISTS" = yes ]; then
    echo "既存のConda環境 '$CONDA_ENV' を再利用します（削除しません）。"
else
    echo "Conda環境 '$CONDA_ENV' をPython 3.11で新規作成します。"
    conda create -n "$CONDA_ENV" python=3.11 pip 'setuptools>=71,<81' wheel -c conda-forge -y
fi
# Activate only inside this process.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
if [ "$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != '3.11' ]; then
    echo "ERROR: $CONDA_ENV のPythonは3.11ではありません。自動更新はしません。" >&2
    exit 2
fi

# OmniGibson imports these two modules during its dataset import check, but
# BEHAVIOR v3.9.3-post2 does not install them with --omnigibson --eval.
# Installing them *before* invoking upstream setup fixes both new environments
# and partially installed ones; --no-deps avoids altering PyTorch/Isaac Sim.
# Only install missing/broken imports, so retries don't re-download wheels.
if ! python -c 'import warp' >/dev/null 2>&1; then
    echo 'OmniGibson依存を補修: NVIDIA Warp (warp-lang==1.12.0)'
    python -m pip install --only-binary=:all: --no-deps 'warp-lang==1.12.0'
fi
if ! python -c 'import av' >/dev/null 2>&1; then
    echo 'OmniGibson依存を補修: PyAV (av==15.1.0)'
    python -m pip install --only-binary=:all: --no-deps 'av==15.1.0'
fi
python -c 'import warp, av; print("OmniGibson import dependencies OK: warp", warp.__version__, "av", av.__version__)'

# Upstream setup handles Isaac Sim / datasets and the actual license prompts.
# Do not create the conda env again; retain the installed packages and assets.
cd "$DIR"
bash ./setup.sh --omnigibson --bddl --joylo --dataset --eval
if [ -z "${KADAI_BEHAVIOR_DIR:-}" ]; then
    echo "BEHAVIOR checkout: $DIR"
fi
echo '公式タスク一覧を取得します（取得失敗時も内蔵の100タスクは利用可能）。'
(cd "$ROOT" && python3 -m kadai_physicalai fetch-catalog) || true
printf '\n完了。次: bash scripts/run_policy.sh（24GB以上のGPUを推奨）\n'
