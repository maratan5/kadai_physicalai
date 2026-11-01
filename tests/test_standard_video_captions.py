"""CPU-only regression tests for standard (non-overview) camera-name captions."""
import unittest

import cv2
import numpy as np

from kadai_physicalai.record_eval import compose_standard


class StandardVideoCaptionTests(unittest.TestCase):
    def test_layout_uses_upstream_camera_order_and_size(self):
        head = np.full((480, 640, 3), (20, 40, 60), dtype=np.uint8)
        left = np.full((512, 512, 3), (75, 105, 135), dtype=np.uint8)
        right = np.full((400, 300, 3), (145, 175, 205), dtype=np.uint8)
        copies = [v.copy() for v in (head, left, right)]
        frame = compose_standard(head, left, right)
        self.assertEqual(frame.shape, (448, 672, 3))
        self.assertEqual(frame.dtype, np.uint8)
        np.testing.assert_array_equal(frame[100, 120], left[100, 120])
        np.testing.assert_array_equal(frame[300, 120], right[200, 120])
        np.testing.assert_array_equal(frame[300, 400], head[300, 400])
        for value, before in zip((head, left, right), copies):
            np.testing.assert_array_equal(value, before)

    def test_all_three_captions_are_drawn(self):
        neutral = np.full((240, 240, 3), 127, dtype=np.uint8)
        result = compose_standard(neutral, neutral, neutral)
        # Each caption occupies just the area near the appropriate camera's top-left.
        for x, y in ((10, 5), (10, 229), (237, 5)):
            region = result[y:y + 35, x:x + 140]
            self.assertTrue(np.any(region != 127), (x, y))
        # Other image regions should remain unmodified.
        self.assertTrue(np.all(result[95:140, 100:170] == 127))
        self.assertTrue(np.all(result[310:340, 100:170] == 127))
        self.assertTrue(np.all(result[200:225, 400:500] == 127))

    def test_torch_like_camera_objects_and_rgba(self):
        class TensorLike:
            def __init__(self, arr):
                self.arr = arr
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.arr
        data = np.zeros((4, 6, 4), dtype=np.uint8)
        data[:, :, 0] = 41
        data[:, :, 3] = 255
        frame = compose_standard(TensorLike(data), data, data)
        self.assertEqual(frame.shape, (448, 672, 3))
        self.assertEqual(tuple(frame[120, 300]), (41, 0, 0))

    def test_invalid_camera_shape(self):
        with self.assertRaises(ValueError):
            compose_standard(np.zeros((5, 5), dtype=np.uint8),
                             np.zeros((5, 5, 3), dtype=np.uint8),
                             np.zeros((5, 5, 3), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
