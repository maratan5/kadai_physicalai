"""CPU-only tests: no Isaac Sim, no policy, no extra GPU are required."""
import unittest

import numpy as np

from kadai_physicalai.record_eval import compose_overview


class OverviewVideoTests(unittest.TestCase):
    def test_720p_layout_and_unmodified_inputs(self):
        external = np.full((720, 1280, 3), (40, 80, 120), dtype=np.uint8)
        head = np.full((720, 720, 4), (120, 100, 80, 255), dtype=np.uint8)
        left = np.full((480, 480, 3), (170, 20, 0), dtype=np.uint8)
        right = np.full((480, 480, 3), (5, 200, 11), dtype=np.uint8)
        before = external.copy()
        result = compose_overview(external, head, left, right)
        self.assertEqual(result.shape, (720, 1280, 3))
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(external, before)
        self.assertEqual(tuple(result[360, 425]), (40, 80, 120))
        self.assertEqual(tuple(result[200, 1060]), (120, 100, 80))
        self.assertEqual(tuple(result[550, 960]), (170, 20, 0))
        self.assertEqual(tuple(result[550, 1175]), (5, 200, 11))

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            compose_overview(np.zeros((3, 4), dtype=np.uint8),
                             np.zeros((5, 5, 3), dtype=np.uint8),
                             np.zeros((5, 5, 3), dtype=np.uint8),
                             np.zeros((5, 5, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
