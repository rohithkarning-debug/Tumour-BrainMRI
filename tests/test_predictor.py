import unittest

import numpy as np
import torch

from src.predictor import SegmentationPredictor


class PredictorPreprocessingTest(unittest.TestCase):
    def test_prepare_input_matches_requested_spatial_size(self) -> None:
        predictor = SegmentationPredictor.__new__(SegmentationPredictor)
        image = np.zeros((32, 40, 24), dtype=np.float32)

        tensor = predictor._prepare_input(image, spatial_size=(16, 20, 12))

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.ndim, 5)
        self.assertEqual(tuple(tensor.shape[-3:]), (16, 20, 12))
        self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
