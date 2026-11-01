"""Check GUI execution settings and live step reporting without OmniGibson."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from kadai_physicalai.runner import make_command

ROOT = Path(__file__).resolve().parents[1]

class GuiProgressTests(unittest.TestCase):
    def test_gui_passes_steps_and_chunk_through_runner(self):
        cmd = make_command('turning_on_radio', instance=3, headless=True, max_steps=1200, action_chunk=12)
        self.assertEqual(cmd[-5:], ['turning_on_radio', '3', '1', '1200', '12'])

    def test_auto_ignores_old_config_env_max_steps(self):
        cmd = make_command('turning_on_radio', max_steps=None, action_chunk=20)
        self.assertEqual(cmd[-2:], ['auto', '20'])

    def test_gui_overhead_options_are_passed_as_cli_overrides(self):
        cmd = make_command('turning_on_radio', max_steps=500, action_chunk=20,
                           camera_mode='fixed_overhead', camera_height=2.2)
        self.assertEqual(cmd[-4:], ['500', '20', 'fixed_overhead', '2.2'])
        cmd = make_command('turning_on_radio', camera_mode='follow', camera_height=2.0)
        self.assertEqual(cmd[-4:], ['auto', '20', 'follow', '2.0'])
        with self.assertRaises(ValueError):
            make_command('turning_on_radio', camera_mode='outside_wall')
        with self.assertRaises(ValueError):
            make_command('turning_on_radio', camera_height=0.0)

    def test_bad_numbers(self):
        for steps in (0, -1, 100001, 1.5, True, '30'):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                make_command('turning_on_radio', max_steps=steps)
        for chunk in (0, 101, 1.5, False, '20'):
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                make_command('turning_on_radio', action_chunk=chunk)

    def test_progress_callback_is_on_every_official_advanced_step(self):
        text = (ROOT / 'kadai_physicalai/record_eval.py').read_text()
        self.assertIn('super()._write_video(instance_eval_state)', text)
        self.assertIn('self._kadai_steps[env_idx] = self._kadai_steps.get(env_idx, 0) + 1', text)
        self.assertIn('[KADAI_PROGRESS] step=', text)
        gui_text = (ROOT / 'kadai_physicalai/gui.py').read_text()
        self.assertIn("self.events.put(('step',", gui_text)
        self.assertIn("elif kind == 'step':", gui_text)
        self.assertIn("step % 10 == 0", gui_text)

    def test_scripts_are_syntax_valid(self):
        script = ROOT / 'scripts/run_task.sh'
        subprocess.run(['bash','-n',str(script)], check=True)
        self.assertIn('else FLAGS+=(--no-headless)', script.read_text())

if __name__ == '__main__':
    unittest.main()
