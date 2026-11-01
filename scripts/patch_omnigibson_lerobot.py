#!/usr/bin/env python3
"""Disable only LeRobot *dataset* imports in a local OmniGibson checkout.

OmniGibson v3.9.3-post2 / BEHAVIOR Challenge evaluation runs under Python 3.11.
The upstream unpinned LeRobot dataset dependency now requires >=3.12.

Removing its setup.py requirement alone is insufficient: omnigibson.envs
unconditionally imports lerobot_data_wrapper, which imports lerobot.configs.
The BEHAVIOR websocket evaluator needs neither LeRobot dataset class, so
remove its eager import and two __all__ exports as well. Back up each file.

This does *not* patch or install LeRobot itself, nor make LeRobot replay work.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys

DEPENDENCY = '"lerobot[dataset] @ git+https://github.com/wensi-ai/lerobot@release/b1k",'
MARKER = '# kadai_physicalai: LeRobot dataset dependency omitted for Python 3.11 evaluator'
IMPORT = 'from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper, LeRobotPlaybackWrapper'
EXPORTS = ('"LeRobotDataWrapper",', '"LeRobotPlaybackWrapper",')
ENVS_MARKER = '# kadai_physicalai: LeRobot dataset wrappers disabled (not needed for websocket evaluation)'


def _backup_once(path: Path) -> Path:
    original = path.with_name(path.name + '.kadai-original')
    if not original.exists():
        shutil.copy2(path, original)
    return original


def patch(setup_path: Path) -> str:
    """Keep the existing API for setup.py patch compatibility."""
    if not setup_path.is_file():
        raise FileNotFoundError(f'OmniGibson/setup.py が見つかりません: {setup_path}')
    source = setup_path.read_text(encoding='utf-8')
    if MARKER in source:
        return 'already_patched'
    if source.count(DEPENDENCY) != 1:
        raise ValueError('想定していたLeRobot依存が見つかりません。上流コードを確認してください。')
    _backup_once(setup_path)
    setup_path.write_text(source.replace(DEPENDENCY, MARKER), encoding='utf-8')
    return 'patched'


def patch_envs_imports(envs_init_path: Path) -> str:
    """Remove ONLY the eager LeRobot dataset wrappers from envs/__init__.py."""
    if not envs_init_path.is_file():
        raise FileNotFoundError(f'OmniGibson envs/__init__.py が見つかりません: {envs_init_path}')
    source = envs_init_path.read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    import_lines = [line for line in lines if line.strip() == IMPORT]
    export_lines = [line for line in lines if line.strip() in EXPORTS]
    if not import_lines and not export_lines:
        # Either previously patched by this script, or manually disabled using
        # the commands supplied to the user. Do not modify or overwrite.
        return 'already_disabled'
    if len(import_lines) != 1 or len(export_lines) != 2 or len({line.strip() for line in export_lines}) != 2:
        raise ValueError('envs/__init__.py のLeRobot参照が予期しない構造です。上流コードを確認してください。')
    _backup_once(envs_init_path)
    filtered = [line for line in lines if line.strip() != IMPORT and line.strip() not in EXPORTS]
    marker_line = ENVS_MARKER + '\n'
    # Keep a single explicit annotation so it's easy to understand and undo.
    if not any(ENVS_MARKER in line for line in filtered):
        filtered.insert(0, marker_line)
    envs_init_path.write_text(''.join(filtered), encoding='utf-8')
    return 'patched'


def patch_behavior(behavior_dir: Path) -> tuple[str, str]:
    og = behavior_dir / 'OmniGibson'
    return (patch(og / 'setup.py'), patch_envs_imports(og / 'omnigibson' / 'envs' / '__init__.py'))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('behavior_dir', type=Path)
    args = parser.parse_args()
    try:
        setup_status, envs_status = patch_behavior(args.behavior_dir)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    print(f'LeRobot依存パッチ: setup.py={setup_status}, envs/__init__.py={envs_status}')
    print('Python 3.11を保持。LeRobotのデータ記録・再生用wrapperは無効、WebSocket評価は維持。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
