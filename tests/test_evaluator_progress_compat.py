"""Regression test for BEHAVIOR-1K v3.9.3-post2 (no BatchedEvaluator.max_steps).

This uses stand-in upstream modules so the regression is tested without Isaac Sim.
"""
import contextlib
import io
import os
import sys
import types
import unittest
from unittest.mock import patch

from kadai_physicalai import record_eval


class ProgressCompatTests(unittest.TestCase):
    def run_fake_evaluation(self, configured_limit):
        captured = io.StringIO()

        class OldUpstreamEvaluator:
            def __init__(self, cfg):
                self.cfg = cfg
                self.human_stats = {'length': 240}
                # Deliberately do not create self.max_steps, as in v3.9.3-post2.

            def load_batch(self, *args, **kwargs):
                pass

            def _write_video(self, state):
                pass

        # Create fake upstream modules, preserving the fully-qualified import tree.
        og = types.ModuleType('omnigibson')
        og.__path__ = []
        ev = types.ModuleType('omnigibson.eval')
        ev.__path__ = []
        cli = types.ModuleType('omnigibson.eval.eval')
        utils = types.ModuleType('omnigibson.eval.utils')
        utils.__path__ = []
        obs = types.ModuleType('omnigibson.eval.utils.obs_utils')
        obs.create_video_writer = lambda **kw: None
        obs.write_video = lambda *a, **kw: None
        eval_utils = types.ModuleType('omnigibson.eval.utils.eval_utils')
        eval_utils.EVAL_TIMEOUT_MULTIPLIER = 1.5
        cli.BatchedEvaluator = OldUpstreamEvaluator
        og.eval = ev
        ev.eval = cli
        ev.utils = utils
        utils.obs_utils = obs
        utils.eval_utils = eval_utils

        def fake_upstream_main():
            cfg = types.SimpleNamespace(max_steps=configured_limit)
            evaluator = cli.BatchedEvaluator(cfg)
            evaluator.load_batch({0: 3}, write_video=False)
            evaluator._write_video(types.SimpleNamespace(env_idx=0))
            evaluator._write_video(types.SimpleNamespace(env_idx=0))
            self.assertFalse(hasattr(evaluator, 'max_steps'))

        cli.main = fake_upstream_main
        modules = {
            m.__name__: m
            for m in [og, ev, cli, utils, obs, eval_utils]
        }
        with patch.dict(sys.modules, modules), patch.dict(os.environ, {'KADAI_OVERVIEW_VIDEO': '0'}), contextlib.redirect_stdout(captured):
            record_eval.main()
        return captured.getvalue()

    def test_default_timeout_uses_human_demo_factor(self):
        out = self.run_fake_evaluation(None)
        self.assertIn('[KADAI_PROGRESS] step=0 total=360', out)
        self.assertIn('[KADAI_PROGRESS] step=1 total=360', out)
        self.assertIn('[KADAI_PROGRESS] step=2 total=360', out)

    def test_user_supplied_timeout_is_reported_without_attribute(self):
        out = self.run_fake_evaluation(500)
        self.assertIn('[KADAI_PROGRESS] step=0 total=500', out)
        self.assertIn('[KADAI_PROGRESS] step=2 total=500', out)


if __name__ == '__main__':
    unittest.main()
