"""Regression checks for non-overwriting evaluation recordings."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts' / 'run_task.sh'
README = ROOT / 'README.md'


class RecordingTests(unittest.TestCase):
    def test_each_run_gets_unique_directory(self):
        text = RUNNER.read_text()
        self.assertIn('mktemp -d', text)
        self.assertIn('date +%Y%m%d_%H%M%S', text)
        self.assertIn('DIR_RESULTS=', text)
        self.assertIn('${TASK}_${STAMP}_instance${INSTANCE}_XXXXXX', text)
        self.assertIn('Asia/Tokyo', text)

    def test_evaluator_writes_video_json_to_run_directory(self):
        text = RUNNER.read_text()
        self.assertIn('"$DIR_RESULTS"', text)
        self.assertIn('--write-video', text)
        self.assertIn('--replay-action-chunk-size "$ACTION_CHUNK"', text)
        self.assertIn('run.log', text)
        self.assertIn('kadai_physicalai.record_eval', text)
        self.assertIn('OMNIGIBSON_KEEP_VIEWER_CAMERA=1', text)
        self.assertIn('KADAI_OVERVIEW_VIDEO', text)

    def test_gui_uses_same_launcher(self):
        from kadai_physicalai.runner import make_command
        cmd = make_command('turning_on_radio')
        self.assertEqual(Path(cmd[1]), RUNNER)

    def test_documented(self):
        self.assertIn('録画と結果は `results/` に実行ごとに保存されます。', README.read_text())


if __name__ == '__main__':
    unittest.main()
