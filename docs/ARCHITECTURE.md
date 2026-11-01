# アーキテクチャ

```text
ユーザー（日本語の依頼）
       ↓
GUI / CLI ── OpenAI Responses API → 公式100タスクから選択（確認）
       ↓
OmniGibson 2026 evaluator v3.9.3 ─── RGB/head+wrist, proprioception
       │                                     ↓
       │                           WebSocket policy port 8000
       │                                     ↓
       │                           π0.5 学習済みDocker
       │                                     ↓
       └──────────────── 23次元ロボット動作 ─────────┘
       ↓
R1 Pro: 台車・胴体・腕・グリッパを制御
       ↓
評価動画とタスク達成率のJSON
```

日本語の解釈にはOpenAI APIを使います。Physical AIの本体は公開済みπ₀.₅のモデルで、タスクの姿勢・関節指令を独自実装していません。直接的な汎用VLA自然言語プロンプトとは異なり、モデルはsnake_caseのタスクIDによって条件付けされます。

リモートモデル: モデル用GPU PCで `KADAI_POLICY_BIND=0.0.0.0 bash scripts/run_policy.sh` を実行。シミュレータPCで `KADAI_POLICY_HOST=192.168.x.x` の設定に変更（IPは例）。安全なLAN環境に限定します。

このプロジェクトは、行動結果を偽装するシミュレーション、ルールベースの把持やテレポートなどは実装しません。行動の成否は公式OmniGibson評価器でのみ判定します。
