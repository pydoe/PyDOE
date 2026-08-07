import unittest

import numpy as np

from pydoe import hammersley_sequence


class TestHammersleySequence(unittest.TestCase):
    def test_hammersley_sequence(self):
        seq = hammersley_sequence(num_points=4, dimension=2)

        expected = np.array([
            [0.0, 0.0],
            [0.25, 0.5],
            [0.5, 0.25],
            [0.75, 0.75],
        ])

        np.testing.assert_allclose(seq, expected, atol=1e-8)

    def test_shape(self):
        seq = hammersley_sequence(num_points=10, dimension=3)
        self.assertEqual(seq.shape, (10, 3))

    def test_first_dimension_is_equispaced(self):
        n = 8
        seq = hammersley_sequence(num_points=n, dimension=4)
        np.testing.assert_allclose(seq[:, 0], np.arange(n) / n)

    def test_one_dimensional(self):
        n = 5
        seq = hammersley_sequence(num_points=n, dimension=1)
        np.testing.assert_allclose(seq[:, 0], np.arange(n) / n)

    def test_points_within_unit_cube(self):
        seq = hammersley_sequence(num_points=20, dimension=3)
        self.assertTrue(bool(np.all(seq >= 0)))
        self.assertTrue(bool(np.all(seq < 1)))

    def test_num_points_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            hammersley_sequence(num_points=0, dimension=2)

    def test_dimension_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            hammersley_sequence(num_points=5, dimension=0)
