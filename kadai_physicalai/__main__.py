import argparse
import sys
from .catalog import TASK_NAMES, TaskSelectionError, download_catalog, load_catalog, select_task
from .runner import run


def main(argv=None):
    p = argparse.ArgumentParser(prog='python3 -m kadai_physicalai', description='BEHAVIOR π0.5 R1 Pro')
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('gui', help='日本語入力GUIを起動')
    sub.add_parser('list', help='公式タスク一覧')
    sub.add_parser('fetch-catalog', help='公式100タスクの説明を取得')
    sub.add_parser('doctor', help='環境を診断（システム設定変更なし）')
    r = sub.add_parser('run', help='タスクを指定して実行')
    group = r.add_mutually_exclusive_group(required=True)
    group.add_argument('--task', help='公式タスクID')
    group.add_argument('--text', help='日本語指示 (公式タスクに対応するもののみ)')
    r.add_argument('--instance', type=int, default=0)
    r.add_argument('--headless', action='store_true')
    r.add_argument('--max-steps', type=int, default=None, help='最大実行ステップ数（省略時はconfig/公式既定値）')
    r.add_argument('--action-chunk', type=int, default=None, help='アクションチャンク（省略時はconfig/20）')
    r.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.action == 'gui':
        from .gui import main as gui_main
        gui_main()
        return 0
    if args.action == 'list':
        for i, (task, desc) in enumerate(load_catalog().items()):
            print(f'{i:02d} {task}: {desc}')
        return 0
    if args.action == 'fetch-catalog':
        print('公式タスクを取得しました:', download_catalog(), '件')
        return 0
    if args.action == 'doctor':
        from .doctor import main as doctor
        return doctor()
    if args.action == 'run':
        try:
            if args.text:
                task, why = select_task(args.text)
                print(f'指示: {args.text}\n対応する公式タスク: {task}\n理由: {why}', flush=True)
            else:
                task = args.task
            return run(task, instance=args.instance, headless=args.headless,
                       max_steps=args.max_steps, action_chunk=args.action_chunk, dry_run=args.dry_run)
        except (ValueError, OSError) as exc:
            print(f'エラー: {exc}', file=sys.stderr)
            return 2
    return 2


if __name__ == '__main__':
    sys.exit(main())
