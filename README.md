# 課題用

**学習済みπ₀.₅ × BEHAVIOR 2026 × OmniGibson × Galaxea R1 Pro**。

> **RTX 5080 / 16GBで確認済み。（推奨は24GB以上）**

## 使用した学習済みモデル

- [pi05-behavior-2026-dexterity（Hugging Face）](https://huggingface.co/VivekChandra/pi05-behavior-2026-dexterity)

## セットアップ（初回のみ）

Ubuntu上でDocker（NVIDIA GPU対応）、Git、Condaが使える状態で実行します。OpenAI APIキーは `~/.bashrc` に `OPENAI_API_KEY` として設定してください。

```bash
cd kadai_physicalai
export PATH="$PWD/.miniforge3/bin:$PATH"
bash scripts/setup_behavior.sh
```

初回は公式シミュレータ・データなどのダウンロードが必要です。

## 起動（毎回2つのターミナル）

**ターミナル1：π₀.₅モデルサーバー**

```bash
KADAI_ALLOW_UNTESTED_16GB=1 bash scripts/run_policy.sh
```

**ターミナル2：GUI**

```bash
export PATH="$PWD/.miniforge3/bin:$PATH"
bash scripts/run_gui.sh
```

GUIの「OmniGibson画面を表示」は初期状態でOFFです。録画と結果は `results/` に実行ごとに保存されます。
