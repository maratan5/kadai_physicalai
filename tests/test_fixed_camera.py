"""CPU-only tests for ceiling-view camera math; no OmniGibson installation needed."""
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

from kadai_physicalai.record_eval import fixed_overhead_pose

class FakeRobot:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)
    def get_position_orientation(self):
        return self.position, [0., 0., 0., 1.]


class CeilingCameraTests(unittest.TestCase):
    def test_world_fixed_eye_near_robot_and_no_orientation_singularity(self):
        def mat2quat(rotation):
            rotation = np.asarray(rotation)
            # These should form a valid non-degenerate right-handed camera frame.
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            self.assertGreater(np.linalg.det(rotation), 0.99)
            return np.array([0., 0., 0., 1.])
        torch = types.ModuleType('torch')
        torch.float32 = np.float32
        torch.as_tensor = lambda value, dtype=None: np.asarray(value, dtype=dtype)
        transform = types.ModuleType('omnigibson.utils.transform_utils')
        transform.mat2quat = mat2quat
        utils = types.ModuleType('omnigibson.utils')
        utils.transform_utils = transform
        og = types.ModuleType('omnigibson')
        og.utils = utils
        with patch.dict(sys.modules, {'torch': torch, 'omnigibson': og,
                                      'omnigibson.utils': utils,
                                      'omnigibson.utils.transform_utils': transform}):
            eye, orientation = fixed_overhead_pose(FakeRobot((3., 4., 0.)), height=2.2)
        np.testing.assert_allclose(eye, [3.40, 3.65, 2.2], atol=1e-5)
        self.assertLess(np.linalg.norm(eye[:2] - [3., 4.]), 0.6)
        self.assertEqual(len(orientation), 4)

if __name__ == '__main__':
    unittest.main()
