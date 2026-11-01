"""Run exactly one official OmniGibson evaluation without synthetic robot actions."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def make_command(task: str, *, instance: int = 0, headless: bool = False,
                 max_steps: int | None = None, action_chunk: int | None = None,
                 camera_mode: str | None = None, camera_height: float | None = None) -> list[str]:
    from .catalog import TASK_NAMES
    if task not in TASK_NAMES:
        raise ValueError(f'未登録の公式タスク: {task}')
    if not 0 <= instance <= 19:
        raise ValueError('公開テストのインスタンス番号は0〜19です')
    if max_steps is not None and (type(max_steps) is not int or not 1 <= max_steps <= 100000):
        raise ValueError('最大ステップ数は1〜100000の整数、または空欄（公式既定値）にしてください')
    if action_chunk is not None and (type(action_chunk) is not int or not 1 <= action_chunk <= 100):
        raise ValueError('アクションチャンクは1〜100の整数にしてください')
    if camera_mode is not None and camera_mode not in ('fixed_overhead', 'follow'):
        raise ValueError('録画カメラは固定俯瞰または追従のいずれかにしてください')
    if camera_height is not None and (not isinstance(camera_height, (int, float)) or
                                      isinstance(camera_height, bool) or
                                      not 1.0 <= camera_height <= 5.0):
        raise ValueError('カメラの高さは1.0〜5.0 mにしてください')
    cmd = ['bash', str(ROOT / 'scripts' / 'run_task.sh'), task, str(instance), '1' if headless else '0']
    # Keep the original CLI invocation unchanged unless explicitly overridden.
    # 'auto' deliberately ignores KADAI_MAX_STEPS in config.env for GUI's blank field.
    if max_steps is not None or action_chunk is not None:
        cmd.extend([str(max_steps) if max_steps is not None else 'auto',
                    str(20 if action_chunk is None else action_chunk)])
    if camera_mode is not None or camera_height is not None:
        # Positional arguments 6 and 7 override optional config.env values.
        if max_steps is None and action_chunk is None:
            cmd.extend(['auto', '20'])
        cmd.extend([camera_mode or 'follow', str(camera_height if camera_height is not None else 2.2)])
    return cmd


def run(task: str, *, instance: int = 0, headless: bool = False,
        max_steps: int | None = None, action_chunk: int | None = None, dry_run: bool = False) -> int:
    command = make_command(task, instance=instance, headless=headless,
                           max_steps=max_steps, action_chunk=action_chunk)
    if dry_run:
        print(' '.join(command))
        print('実際の起動なし。設定・GPU・OmniGibsonは実環境で確認してください。')
        return 0
    return subprocess.call(command, cwd=ROOT, env=os.environ.copy())
