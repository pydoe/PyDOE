import unittest

import numpy as np

from pydoe import niederreiter_sequence


class TestNiederreiterSequence(unittest.TestCase):
    def test_shape(self):
        for num_points, dimension in [(1, 1), (4, 2), (10, 5), (32, 20)]:
            seq = niederreiter_sequence(num_points, dimension)
            self.assertEqual(seq.shape, (num_points, dimension))

    def test_dtype(self):
        seq = niederreiter_sequence(4, 2)
        self.assertEqual(seq.dtype, np.float64)

    def test_values_within_unit_interval(self):
        seq = niederreiter_sequence(64, 10)
        self.assertTrue(bool(np.all(seq >= 0)))
        self.assertTrue(bool(np.all(seq < 1)))

    def test_first_point_is_origin(self):
        seq = niederreiter_sequence(1, 5)
        np.testing.assert_allclose(seq, np.zeros((1, 5)))

    def test_determinism(self):
        seq1 = niederreiter_sequence(20, 4)
        seq2 = niederreiter_sequence(20, 4)
        np.testing.assert_array_equal(seq1, seq2)

    def test_n_bits_changes_resolution(self):
        seq_default = niederreiter_sequence(4, 2)
        seq_small = niederreiter_sequence(4, 2, n_bits=4)
        self.assertEqual(seq_default.shape, seq_small.shape)
        self.assertFalse(np.allclose(seq_default, seq_small))

    def test_num_points_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            niederreiter_sequence(0, 2)

    def test_dimension_less_than_one_raises(self):
        with self.assertRaises(ValueError):
            niederreiter_sequence(5, 0)

    def test_dimension_greater_than_twenty_raises(self):
        with self.assertRaises(ValueError):
            niederreiter_sequence(5, 21)

    def test_n_bits_less_than_two_raises(self):
        with self.assertRaises(ValueError):
            niederreiter_sequence(5, 2, n_bits=1)

    def test_max_dimension_is_supported(self):
        seq = niederreiter_sequence(8, 20)
        self.assertEqual(seq.shape, (8, 20))
        self.assertTrue(bool(np.all(seq >= 0)))
        self.assertTrue(bool(np.all(seq < 1)))

    def test_equidistribution_sanity_check(self):
        num_points = 256
        seq = niederreiter_sequence(num_points, 2, n_bits=8)

        bin_indices = np.minimum((seq * 4).astype(int), 3)
        counts = np.zeros((4, 4), dtype=int)
        for x_bin, y_bin in bin_indices:
            counts[x_bin, y_bin] += 1

        expected = num_points / 16
        self.assertTrue(bool(np.all(counts >= expected * 0.4)))
        self.assertTrue(bool(np.all(counts <= expected * 2.5)))
