"""Japanese GUI: OpenAI routes requests; the official π0.5 evaluator runs them."""
from __future__ import annotations

import os
import queue
import re
import signal
import subprocess
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from .catalog import TASK_NAMES, TaskSelectionError, load_catalog, select_task, suggestions
from .openai_router import DEFAULT_MODEL
from .runner import ROOT, make_command


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('kadai_physicalai  |  OpenAI × π0.5 × BEHAVIOR-1K')
        self.root.geometry('1000x960')
        self.process: subprocess.Popen | None = None
        self.events = queue.Queue()
        self.catalog = load_catalog()
        self.japanese = tk.StringVar(value='モデムを設置して')
        self.task = tk.StringVar(value='installing_a_modem')
        self.instance = tk.StringVar(value='0')
        # This controls only whether the interactive OmniGibson window appears.
        # Rendering for the robot / recording cameras remains enabled in headless mode.
        self.show_omnigibson = tk.BooleanVar(value=False)
        # Empty max_steps uses official task-specific human-demo-based timeout.
        self.max_steps = tk.StringVar(value='')
        self.action_chunk = tk.StringVar(value='20')
        self.camera_mode = tk.StringVar(value='ロボット追従（斜め後方）')
        self.camera_height = tk.StringVar(value='2.0')
        self.step_text = tk.StringVar(value='現在ステップ: 未開始')
        self.last_step = 0
        self.last_total = 0
        self.status = tk.StringVar(value='日本語の指示を入力して「シミュレーション実行」を押すと、OpenAIが自動的にタスクを選択します。')
        self.resolving = False
        self.pending_start = False
        self.selected_source = None  # None / 'api' / 'manual'
        self.resolved_instruction = None
        self.selection_epoch = 0  # Discard responses for changed input / manual overrides
        self.build()
        self.japanese.trace_add('write', self.on_instruction_change)
        self.root.after(100, self.poll)
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    def build(self):
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='π0.5  |  Physical AI  |  Galaxea R1 Pro', font=('DejaVu Sans', 19, 'bold')).pack(anchor='w')
        ttk.Label(frame, text='自然言語の理解: OpenAI API  |  ロボット制御: 学習済みπ0.5  |  シミュレータ: OmniGibson', foreground='#666').pack(anchor='w', pady=(0, 10))

        box = ttk.LabelFrame(frame, text='日本語の指示（OpenAI APIで公式100タスクに対応付け）', padding=12)
        box.pack(fill='x', pady=(0, 10))
        entry = ttk.Entry(box, textvariable=self.japanese, font=('', 12))
        entry.pack(fill='x', pady=5)
        entry.bind('<Return>', lambda _e: self.resolve())
        row = ttk.Frame(box)
        row.pack(fill='x')
        self.resolve_btn = ttk.Button(row, text='OpenAIでタスクを選択', command=self.resolve)
        self.resolve_btn.pack(side='left', padx=(0, 8))
        model = os.environ.get('KADAI_OPENAI_MODEL', DEFAULT_MODEL)
        ttk.Label(row, text=f'使用モデル: {model}　/　APIキー: ~/.bashrc から自動読込', foreground='#555').pack(side='left')
        ttk.Label(box, text='※ 実行ボタンからも自動判定します。100タスクに該当しない指示は実行しません。APIの利用料金が発生します。', foreground='#955').pack(anchor='w', pady=(8, 0))

        box2 = ttk.LabelFrame(frame, text='実行するBEHAVIOR公式タスク（手動選択も可能）', padding=12)
        box2.pack(fill='x', pady=(0, 10))
        self.combo = ttk.Combobox(box2, textvariable=self.task, values=TASK_NAMES, width=64)
        self.combo.pack(fill='x')
        self.combo.bind('<<ComboboxSelected>>', self.on_change)
        self.combo.bind('<KeyRelease>', self.filter_combo)
        self.description = ttk.Label(box2, text='', wraplength=900, foreground='#555')
        self.description.pack(anchor='w', pady=5)
        row2 = ttk.Frame(box2)
        row2.pack(fill='x', pady=(4, 0))
        ttk.Label(row2, text='試行インスタンス (0–19):').pack(side='left')
        ttk.Spinbox(row2, from_=0, to=19, textvariable=self.instance, width=6).pack(side='left', padx=5)
        ttk.Checkbutton(row2, text='OmniGibson画面を表示', variable=self.show_omnigibson).pack(side='left', padx=14)
        self.start_btn = ttk.Button(row2, text='シミュレーション実行', command=self.start)
        self.start_btn.pack(side='right', padx=(6, 0))
        ttk.Button(row2, text='停止', command=self.stop).pack(side='right')

        ttk.Label(box2, text='チェックOFF（既定）: OmniGibsonの画面を開かずに実行します。動画録画は継続します。',
                  foreground='#666').pack(anchor='w', pady=(6, 0))

        row3 = ttk.Frame(box2)
        row3.pack(fill='x', pady=(10, 0))
        ttk.Label(row3, text='最大ステップ数:').pack(side='left')
        ttk.Entry(row3, textvariable=self.max_steps, width=11).pack(side='left', padx=(5, 6))
        ttk.Label(row3, text='空欄 = タスク別の公式既定値', foreground='#666').pack(side='left', padx=(0, 24))
        ttk.Label(row3, text='アクションチャンク:').pack(side='left')
        ttk.Spinbox(row3, from_=1, to=100, textvariable=self.action_chunk, width=6).pack(side='left', padx=5)
        ttk.Label(box2, text='目安: 1ステップ ≈ シミュレーション0.033秒（30Hz）。実際の処理速度とは異なります。',
                  foreground='#666').pack(anchor='w', pady=(7, 0))

        cam_row = ttk.Frame(box2)
        cam_row.pack(fill='x', pady=(8, 0))
        ttk.Label(cam_row, text='第三者視点の録画カメラ:').pack(side='left')
        ttk.Combobox(cam_row, textvariable=self.camera_mode, state='readonly',
                     values=('ロボット追従（斜め後方）', '固定俯瞰（天井風）'), width=25).pack(side='left', padx=6)
        ttk.Label(cam_row, text='高さ (m):').pack(side='left', padx=(15, 0))
        ttk.Entry(cam_row, textvariable=self.camera_height, width=7).pack(side='left', padx=6)
        ttk.Label(box2, text='追従はロボットの斜め後ろ右側から旋回にも追従します。固定俯瞰は初期位置から動きません。',
                  foreground='#666').pack(anchor='w', pady=(5, 0))

        progress_row = ttk.Frame(frame)
        progress_row.pack(fill='x', pady=(0, 7))
        ttk.Label(progress_row, textvariable=self.step_text).pack(side='left', padx=(0, 12))
        self.step_bar = ttk.Progressbar(progress_row, orient='horizontal', mode='determinate', maximum=1)
        self.step_bar.pack(side='left', fill='x', expand=True)

        ttk.Label(frame, textvariable=self.status, foreground='#3375a5', wraplength=910).pack(anchor='w', pady=(0, 10))
        self.log = scrolledtext.ScrolledText(frame, height=18, state='disabled', font=('DejaVu Sans Mono', 10), wrap='word')
        self.log.pack(fill='both', expand=True)
        self.show_description()

    def on_instruction_change(self, *_):
        self.selection_epoch += 1
        self.selected_source = None
        self.resolved_instruction = None

    def on_change(self, *_):
        self.selection_epoch += 1
        self.selected_source = 'manual'
        self.show_description()
        self.status.set('公式タスクを手動で選択しました。実行時はOpenAI APIを呼びません。')

    def filter_combo(self, *_):
        self.selection_epoch += 1
        self.selected_source = 'manual'
        current = self.task.get()
        self.combo['values'] = suggestions(current, 30) if current else TASK_NAMES
        self.show_description()

    def show_description(self):
        self.description.configure(text=self.catalog.get(self.task.get(), '公式タスクIDを選択してください'))

    def resolve(self):
        self.resolve_async(run_after=False)

    def resolve_async(self, *, run_after: bool):
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo('実行中', '先に現在のシミュレーションを停止してください。')
            return
        if self.resolving:
            if run_after:
                self.pending_start = True
                self.status.set('OpenAIが判定中です。終了後にタスク確認画面を表示します。')
            return
        instruction = self.japanese.get().strip()
        if not instruction:
            messagebox.showwarning('入力エラー', '日本語の指示を入力してください。')
            return
        epoch = self.selection_epoch
        self.resolving = True
        self.pending_start = run_after
        self.resolve_btn.configure(state='disabled')
        self.status.set('OpenAI APIで公式100タスクを照合しています…')
        threading.Thread(target=self.resolve_worker, args=(instruction, epoch), daemon=True).start()

    def resolve_worker(self, instruction: str, epoch: int):
        try:
            task, method = select_task(instruction)
            self.events.put(('resolved', (instruction, epoch, task, method)))
        except (TaskSelectionError, ValueError, OSError) as exc:
            self.events.put(('resolve_error', (epoch, str(exc))))
        except Exception as exc:
            # Do not expose internal request headers or API key through logs.
            self.events.put(('resolve_error', (epoch, f'タスク選択に失敗しました（{type(exc).__name__}）'))) 

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        self.log.see('end')
        self.log.configure(state='disabled')

    def start(self):
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo('実行中', '先に現在のシミュレーションを停止してください')
            return
        if self.selected_source == 'manual':
            self.start_selected()
        elif self.selected_source == 'api' and self.resolved_instruction == self.japanese.get().strip():
            self.start_selected()
        else:
            # One click: resolve via API, then ask user to confirm execution.
            self.resolve_async(run_after=True)

    def start_selected(self):
        try:
            max_steps_text = self.max_steps.get().strip()
            max_steps = int(max_steps_text) if max_steps_text else None
            action_chunk = int(self.action_chunk.get().strip())
            camera_mode = {'固定俯瞰（天井風）': 'fixed_overhead', 'ロボット追従（斜め後方）': 'follow'}[self.camera_mode.get()]
            camera_height = float(self.camera_height.get().strip())
            cmd = make_command(self.task.get(), instance=int(self.instance.get()),
                               headless=not self.show_omnigibson.get(), max_steps=max_steps,
                               action_chunk=action_chunk, camera_mode=camera_mode,
                               camera_height=camera_height)
        except (ValueError, TypeError, KeyError) as exc:
            messagebox.showerror('選択エラー', str(exc))
            return
        display_text = '表示' if self.show_omnigibson.get() else '非表示（ヘッドレス）'
        if not messagebox.askyesno('実行確認', f'公式タスク: {self.task.get()}\n\n{self.catalog[self.task.get()]}\n\n最大ステップ: {max_steps if max_steps is not None else "公式既定値"}\nアクションチャンク: {action_chunk}\n録画カメラ: {self.camera_mode.get()} / 高さ: {camera_height}m\nOmniGibson画面: {display_text}\n\nこの内容でOmniGibsonを開始しますか？'):
            self.status.set('ユーザーが実行をキャンセルしました。')
            return
        self.last_step = 0
        self.last_total = max_steps or 0
        self.step_text.set('現在ステップ: 初期化中（0ステップ）')
        self.step_bar.configure(maximum=max_steps or 1, value=0)
        self.append('\n=== RUN ' + self.task.get() + ' ===\n')
        self.append(f'最大ステップ: {max_steps if max_steps is not None else "公式既定値"} / '
                    f'アクションチャンク: {action_chunk} / 録画カメラ: {self.camera_mode.get()} / 高さ: {camera_height}m / OmniGibson画面: {display_text}\n')
        try:
            self.process = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                             text=True, bufsize=1, start_new_session=True)
        except OSError as exc:
            messagebox.showerror('起動エラー', str(exc))
            return
        self.start_btn.configure(state='disabled')
        self.status.set('実行中: OmniGibsonとπ0.5は別プロセスです。ESCではなく「停止」を押してください。')
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()

    def read_output(self, proc):
        try:
            for line in proc.stdout:
                match = re.search(r'\[KADAI_PROGRESS\] step=(\d+) total=(\d+)', line)
                if match:
                    self.events.put(('step', (int(match.group(1)), int(match.group(2)))))
                else:
                    self.events.put(('line', line))
            self.events.put(('exit', proc.wait()))
        except Exception as exc:
            self.events.put(('line', f'出力読み取りエラー: {exc}\n'))
            self.events.put(('exit', 1))

    def poll(self):
        try:
            for _ in range(150):
                kind, value = self.events.get_nowait()
                if kind == 'line':
                    self.append(value)
                elif kind == 'step':
                    step, total = value
                    self.last_step, self.last_total = step, total
                    self.step_text.set(f'現在ステップ: {step:,} / {total:,}')
                    self.step_bar.configure(maximum=max(1, total), value=min(step, total))
                    # The dedicated counter updates every step, without flooding
                    # the scrollback with thousands of near-identical lines.
                    if step == 0 or step == 1 or step % 10 == 0 or step == total:
                        self.append(f'[進捗] {step} / {total} ステップ\n')
                elif kind == 'exit':
                    self.start_btn.configure(state='normal')
                    self.status.set(f'シミュレーション終了: exit={value}。結果はresults/以下を確認してください。')
                    if self.last_total:
                        self.step_text.set(f'終了: {self.last_step:,} / {self.last_total:,} ステップ')
                    else:
                        self.step_text.set('終了（ステップ取得前）')
                    self.process = None
                elif kind == 'resolved':
                    instruction, epoch, task, method = value
                    self.resolving = False
                    self.resolve_btn.configure(state='normal')
                    should_start = self.pending_start
                    self.pending_start = False
                    if epoch != self.selection_epoch or instruction != self.japanese.get().strip():
                        self.status.set('入力が変更されたため古いOpenAIの判定結果を破棄しました。もう一度実行してください。')
                        continue
                    self.task.set(task)
                    self.selected_source = 'api'
                    self.resolved_instruction = instruction
                    self.show_description()
                    self.status.set(f'選択: {task}（{method}）。内容を確認して実行してください。')
                    if should_start:
                        self.start_selected()
                elif kind == 'resolve_error':
                    epoch, error_text = value
                    self.resolving = False
                    self.pending_start = False
                    self.resolve_btn.configure(state='normal')
                    if epoch == self.selection_epoch:
                        self.status.set(error_text)
                        messagebox.showwarning('タスクを選択できません', error_text)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def stop(self):
        proc = self.process
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                self.append('ユーザー操作: 実行プロセスへSIGTERMを送信\n')
            except ProcessLookupError:
                pass

    def close(self):
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
