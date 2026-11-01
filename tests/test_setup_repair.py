"""Guard against regressions that repeatedly broke Python 3.11 setup."""
from pathlib import Path
import unittest

SCRIPT = (Path(__file__).resolve().parents[1] / 'scripts' / 'setup_behavior.sh').read_text(encoding='utf-8')


class SetupRepairTests(unittest.TestCase):
    def test_repair_before_upstream_dataset_import(self):
        warp = SCRIPT.index("'warp-lang==1.12.0'")
        pyav = SCRIPT.index("'av==15.1.0'")
        upstream = SCRIPT.index('bash ./setup.sh --omnigibson --bddl --joylo --dataset --eval')
        self.assertLess(warp, upstream)
        self.assertLess(pyav, upstream)

    def test_python311_and_existing_env_reused(self):
        self.assertIn('if [ "$ENV_EXISTS" = yes ]; then', SCRIPT)
        self.assertIn('python=3.11', SCRIPT)
        self.assertNotIn('bash ./setup.sh --new-env', SCRIPT)
        self.assertNotIn('conda env remove', SCRIPT)

    def test_lerobot_import_is_patched_before_upstream_install(self):
        self.assertIn('patch_omnigibson_lerobot.py', SCRIPT)
        self.assertLess(SCRIPT.index('python3 "$ROOT/scripts/patch_omnigibson_lerobot.py"'), SCRIPT.index('bash ./setup.sh --omnigibson'))

    def test_repair_preserves_torch_deps(self):
        self.assertEqual(SCRIPT.count('python -m pip install --only-binary=:all: --no-deps'), 2)
        self.assertIn("if ! python -c 'import warp'", SCRIPT)
        self.assertIn("if ! python -c 'import av'", SCRIPT)


if __name__ == '__main__':
    unittest.main()
