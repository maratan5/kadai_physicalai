"""Default UI settings and the exact headless flags passed to OmniGibson."""
from pathlib import Path
import unittest
from kadai_physicalai.catalog import ALIASES, TASK_NAMES, load_catalog
from kadai_physicalai.runner import make_command

ROOT = Path(__file__).resolve().parents[1]
GUI = (ROOT / 'kadai_physicalai/gui.py').read_text(encoding='utf-8')
LAUNCH = (ROOT / 'scripts/run_task.sh').read_text(encoding='utf-8')

class GuiInitialSettingsTests(unittest.TestCase):
    def test_default_japanese_matches_task_id(self):
        self.assertIn("self.japanese = tk.StringVar(value='モデムを設置して')", GUI)
        self.assertIn("self.task = tk.StringVar(value='installing_a_modem')", GUI)
        self.assertIn('installing_a_modem', TASK_NAMES)
        self.assertEqual(ALIASES['モデムを設置して'], 'installing_a_modem')
        self.assertIn('モデム', load_catalog()['installing_a_modem'])

    def test_omnigibson_display_default_off(self):
        self.assertIn('self.show_omnigibson = tk.BooleanVar(value=False)', GUI)
        self.assertIn("text='OmniGibson画面を表示', variable=self.show_omnigibson", GUI)
        self.assertIn('headless=not self.show_omnigibson.get()', GUI)
        self.assertEqual(make_command('installing_a_modem', headless=True)[-1], '1')
        self.assertEqual(make_command('installing_a_modem', headless=False)[-1], '0')
        self.assertIn('if [ "$HEADLESS" = 1 ]; then FLAGS+=(--headless); else FLAGS+=(--no-headless); fi', LAUNCH)

    def test_default_rendering_does_not_turn_off_recording(self):
        self.assertIn('--write-video', LAUNCH)
        self.assertIn('KADAI_OVERVIEW_VIDEO', LAUNCH)

if __name__ == '__main__':
    unittest.main()
