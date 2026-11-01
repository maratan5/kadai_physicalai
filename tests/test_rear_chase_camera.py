"""CPU-only regression tests: rear chase camera yaw, translation, elevation."""
from pathlib import Path
import unittest
import numpy as np

from kadai_physicalai.record_eval import _rear_follow_eye_target
from kadai_physicalai.runner import make_command

ROOT = Path(__file__).resolve().parents[1]


class RearChasePoseTests(unittest.TestCase):
    def test_rear_right_when_facing_positive_x(self):
        eye, target = _rear_follow_eye_target([3., 4., 0.], np.eye(3),
                                               distance=1.8, side=0.85, height=2.0)
        np.testing.assert_allclose(eye, [1.2, 3.15, 2.0], atol=1e-6)
        np.testing.assert_allclose(target, [3.3, 4., 1.0], atol=1e-6)

    def test_default_horizontal_offset_is_halved(self):
        # Before: 1.8 m behind and 0.85 m to the right.
        eye, target = _rear_follow_eye_target([3., 4., 0.], np.eye(3))
        np.testing.assert_allclose(eye, [2.1, 3.575, 2.0], atol=1e-6)
        np.testing.assert_allclose(target, [3.3, 4., 1.0], atol=1e-6)
        self.assertAlmostEqual(np.hypot(0.9, 0.425), np.hypot(1.8, 0.85) / 2)

    def test_default_camera_config_matches_recording(self):
        import inspect
        from kadai_physicalai import record_eval
        evaluator_source = inspect.getsource(record_eval.main)
        self.assertIn("KADAI_CAMERA_DISTANCE', '0.9'", evaluator_source)
        self.assertIn("KADAI_CAMERA_SIDE', '0.425'", evaluator_source)
        cfg = (ROOT / 'config.env.example').read_text(encoding='utf-8')
        self.assertIn('KADAI_CAMERA_DISTANCE=0.9', cfg)
        self.assertIn('KADAI_CAMERA_SIDE=0.425', cfg)

    def test_yaw_rotates_camera_with_robot(self):
        # Turn robot counter-clockwise 90 degrees: local +X now world +Y.
        yaw90 = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        eye, target = _rear_follow_eye_target([3., 4., 0.], yaw90,
                                               distance=1.8, side=0.85, height=2.0)
        np.testing.assert_allclose(eye, [3.85, 2.2, 2.0], atol=1e-6)
        np.testing.assert_allclose(target, [3., 4.3, 1.0], atol=1e-6)

    def test_translating_robot_moves_camera_without_changing_offset(self):
        eye1, target1 = _rear_follow_eye_target([0., 0., 0.], np.eye(3))
        eye2, target2 = _rear_follow_eye_target([7., -2., 0.], np.eye(3))
        np.testing.assert_allclose(eye2 - eye1, [7., -2., 0.])
        np.testing.assert_allclose(target2 - target1, [7., -2., 0.])

    def test_height_stays_world_vertical_even_if_robot_tilts(self):
        tilt30 = np.array([[1., 0., 0.],
                           [0., np.cos(np.pi / 6), -np.sin(np.pi / 6)],
                           [0., np.sin(np.pi / 6), np.cos(np.pi / 6)]])
        eye, target = _rear_follow_eye_target([0., 0., 0.4], tilt30)
        self.assertAlmostEqual(eye[2], 2.4)
        self.assertAlmostEqual(target[2], 1.4)

    def test_invalid_heading(self):
        with self.assertRaises(ValueError):
            _rear_follow_eye_target([0., 0., 0.], np.zeros((3, 3)))


class ChaseGuiDefaultsTests(unittest.TestCase):
    def test_initial_english_japanese_modem_and_chase(self):
        gui = (ROOT / 'kadai_physicalai' / 'gui.py').read_text(encoding='utf-8')
        self.assertIn("self.japanese = tk.StringVar(value='モデムを設置して')", gui)
        self.assertIn("self.task = tk.StringVar(value='installing_a_modem')", gui)
        self.assertIn("self.camera_mode = tk.StringVar(value='ロボット追従（斜め後方）')", gui)
        self.assertIn("'ロボット追従（斜め後方）': 'follow'", gui)
        self.assertIn('固定俯瞰（天井風）', gui)
        self.assertIn("self.show_omnigibson = tk.BooleanVar(value=False)", gui)

    def test_shell_default_task_and_camera(self):
        sh = (ROOT / 'scripts' / 'run_task.sh').read_text(encoding='utf-8')
        self.assertIn('TASK="${1:-installing_a_modem}"', sh)
        self.assertIn('CAMERA_MODE="${KADAI_CAMERA_MODE:-follow}"', sh)
        cmd = make_command('installing_a_modem', camera_mode='follow', camera_height=2.0)
        self.assertEqual(cmd[-4:], ['auto', '20', 'follow', '2.0'])


if __name__ == '__main__':
    unittest.main()
