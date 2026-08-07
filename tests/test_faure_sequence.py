import unittest

import numpy as np

from pydoe import faure_sequence


class TestFaureSequence(unittest.TestCase):
    def test_faure_sequence(self):
        seq = faure_sequence(num_points=4, dimension=2)

        expected = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [0.25, 0.75],
            [0.75, 0.25],
        ])

        np.testing.assert_allclose(seq, expected, atol=1e-8)

    def test_shape(self):
        seq = faure_sequence(num_points=10, dimension=3)
        self.assertEqual(seq.shape, (10, 3))

    def test_shape_various(self):
        for num_points, dimension in [(1, 1), (5, 1), (7, 4), (16, 5)]:
            seq = faure_sequence(num_points=num_points, dimension=dimension)
            self.assertEqual(seq.shape, (num_points, dimension))

    def test_points_within_unit_cube(self):
        seq = faure_sequence(num_points=20, dimension=3)
        self.assertTrue(bool(np.all(seq >= 0)))
        self.assertTrue(bool(np.all(seq < 1)))

    def test_deterministic(self):
        seq1 = faure_sequence(num_points=10, dimension=4)
        seq2 = faure_sequence(num_points=10, dimension=4)
        np.testing.assert_array_equal(seq1, seq2)

    def test_skip_shifts_sequence(self):
        n = 5
        skip = 3
        dimension = 3

        full = faure_sequence(num_points=n + skip, dimension=dimension)
        shifted = faure_sequence(num_points=n, dimension=dimension, skip=skip)

        np.testing.assert_allclose(shifted, full[skip:], atol=1e-8)

    def test_one_dimensional_is_van_der_corput_base_2(self):
        seq = faure_sequence(num_points=5, dimension=1)

        expected = np.array([[0.0], [0.5], [0.25], [0.75], [0.125]])

        np.testing.assert_allclose(seq, expected, atol=1e-8)

    def test_num_points_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            faure_sequence(num_points=0, dimension=2)

    def test_dimension_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            faure_sequence(num_points=5, dimension=0)

    def test_negative_skip_raises(self):
        with self.assertRaises(ValueError):
            faure_sequence(num_points=5, dimension=2, skip=-1)
