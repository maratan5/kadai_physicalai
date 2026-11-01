"""Read-only environment validation."""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def gpu_info() -> tuple[str, int | None]:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            text=True, timeout=7, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f'取得不可 ({exc})', None
    first = out.splitlines()[0]
    name, mem = first.rsplit(',', 1)
    try:
        return name.strip(), int(mem.strip())
    except ValueError:
        return first, None


def get_health(host='127.0.0.1', port=8000, timeout=3) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f'http://{host}:{int(port)}/healthz', timeout=timeout) as r:
            body = r.read(200).decode('utf-8', errors='replace')
            return 200 <= r.status < 300, body
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    print('=== kadai_physicalai 診断（読み取り専用）===')
    for name in ('git', 'docker', 'conda', 'nvidia-smi', 'python3'):
        print(f'{name:13}: {shutil.which(name) or "なし"}')
    gpu, mem = gpu_info()
    print(f'GPU          : {gpu} VRAM={mem} MiB')
    if mem is not None and mem < 24000:
        print('注意: 学習済みπ0.5の公開Dockerは24GB GPU実行例です。このGPUでの動作は未検証。')
    behavior = Path(os.path.expanduser(os.environ.get('KADAI_BEHAVIOR_DIR', str(ROOT / 'external' / 'BEHAVIOR-1K'))))
    print(f'BEHAVIOR     : {behavior} / {"存在" if (behavior/"OmniGibson"/"omnigibson"/"eval"/"eval.py").exists() else "未セットアップ"}')
    host = os.environ.get('KADAI_POLICY_HOST', '127.0.0.1')
    port = int(os.environ.get('KADAI_POLICY_PORT', '8000'))
    ok, response = get_health(host, port)
    print(f'policy health: http://{host}:{port}/healthz / {"OK" if ok else "未接続"} ({response[:120]})')
    print('実行前にモデルサーバーとOmniGibson環境をセットアップしてください。')
    return 0
