import unittest
import numpy as np
from main import broadcast_tensors

class TestCalculations(unittest.TestCase):

    def setUp(self):
        self.tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
        self.tensor2 = np.array([1, 1, 1])
        self.broadcasted_tensor2_shape, self.broadcasted_tensor2 = (
            broadcast_tensors(self.tensor1, self.tensor2)
        )

    def test_broadcasted_shape(self):
        self.assertEqual(
            self.broadcasted_tensor2_shape, (2, 3), "Shape mismatch"
        )

if __name__ == "__main__":
    unittest.main()