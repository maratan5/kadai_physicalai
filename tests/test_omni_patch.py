from pathlib import Path
import tempfile
import unittest

from scripts.patch_omnigibson_lerobot import (
    patch, patch_envs_imports, patch_behavior,
    DEPENDENCY, MARKER, IMPORT, EXPORTS, ENVS_MARKER,
)


UPSTREAM_ENVS_INIT = '''from omnigibson.envs.data_wrapper import DataWrapper, DataPlaybackWrapper
from omnigibson.envs.hdf5_data_wrapper import HDF5CollectionWrapper, HDF5PlaybackWrapper
from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper, LeRobotPlaybackWrapper
from omnigibson.envs.metrics_wrapper import MetricsWrapper
from omnigibson.metrics.metric_base import MetricBase
from omnigibson.envs.env_base import Environment
from omnigibson.envs.env_wrapper import REGISTERED_ENV_WRAPPERS, EnvironmentWrapper, create_wrapper
__all__ = [
    "create_wrapper",
    "DataWrapper",
    "DataPlaybackWrapper",
    "HDF5CollectionWrapper",
    "HDF5PlaybackWrapper",
    "LeRobotDataWrapper",
    "LeRobotPlaybackWrapper",
    "MetricsWrapper",
    "MetricBase",
    "Environment",
    "EnvironmentWrapper",
    "REGISTERED_ENV_WRAPPERS",
]
'''


class PatchTests(unittest.TestCase):
    def test_patch_only_dependency_and_backup(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'setup.py'
            path.write_text('before\n    ' + DEPENDENCY + '\nafter\n', encoding='utf-8')
            self.assertEqual(patch(path), 'patched')
            self.assertNotIn(DEPENDENCY, path.read_text())
            self.assertIn(MARKER, path.read_text())
            self.assertEqual(patch(path), 'already_patched')
            self.assertIn(DEPENDENCY, (Path(d) / 'setup.py.kadai-original').read_text())

    def test_unknown_setup_upstream_fails_without_modification(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'setup.py'
            path.write_text('some_other_dependency', encoding='utf-8')
            with self.assertRaises(ValueError):
                patch(path)
            self.assertEqual(path.read_text(), 'some_other_dependency')
            self.assertFalse((Path(d) / 'setup.py.kadai-original').exists())

    def test_patch_eager_import_exact_upstream(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '__init__.py'
            path.write_text(UPSTREAM_ENVS_INIT)
            self.assertEqual(patch_envs_imports(path), 'patched')
            updated = path.read_text()
            self.assertNotIn(IMPORT, updated)
            for name in EXPORTS:
                self.assertNotIn(name, updated)
            self.assertIn(ENVS_MARKER, updated)
            self.assertIn('from omnigibson.envs.env_base import Environment', updated)
            self.assertIn('"HDF5CollectionWrapper",', updated)
            self.assertIn('"MetricsWrapper",', updated)
            self.assertIn('"Environment",', updated)
            self.assertIn('"REGISTERED_ENV_WRAPPERS",', updated)
            self.assertEqual(patch_envs_imports(path), 'already_disabled')
            self.assertEqual((Path(d) / '__init__.py.kadai-original').read_text(), UPSTREAM_ENVS_INIT)

    def test_manually_applied_fix_is_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '__init__.py'
            lines = [line for line in UPSTREAM_ENVS_INIT.splitlines(True) if line.strip() != IMPORT and line.strip() not in EXPORTS]
            path.write_text(''.join(lines))
            self.assertEqual(patch_envs_imports(path), 'already_disabled')
            self.assertFalse((Path(d) / '__init__.py.kadai-original').exists())

    def test_unexpected_envs_layout_fails_safe(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '__init__.py'
            path.write_text('from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper, LeRobotPlaybackWrapper\n')
            with self.assertRaises(ValueError):
                patch_envs_imports(path)
            self.assertFalse((Path(d) / '__init__.py.kadai-original').exists())

    def test_entire_behavior_tree_patched_idempotently(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            setup = root / 'OmniGibson' / 'setup.py'
            init = root / 'OmniGibson' / 'omnigibson' / 'envs' / '__init__.py'
            setup.parent.mkdir(parents=True)
            init.parent.mkdir(parents=True)
            setup.write_text(DEPENDENCY)
            init.write_text(UPSTREAM_ENVS_INIT)
            self.assertEqual(patch_behavior(root), ('patched','patched'))
            self.assertEqual(patch_behavior(root), ('already_patched','already_disabled'))


if __name__ == '__main__':
    unittest.main()
