import unittest

import numpy as np

from pydoe import sliced_lhs


class TestSlicedLhs(unittest.TestCase):
    def test_shape(self):
        design, slices = sliced_lhs(2, 3, 2, seed=0)
        self.assertEqual(design.shape, (6, 2))
        self.assertEqual(slices.shape, (6,))

    def test_unit_hypercube(self):
        design, _slices = sliced_lhs(3, 4, 5, seed=1)
        self.assertTrue(bool(np.all(design >= 0.0)))
        self.assertTrue(bool(np.all(design < 1.0)))

    def test_slice_labels(self):
        _design, slices = sliced_lhs(2, 3, 4, seed=0)
        np.testing.assert_array_equal(slices, np.repeat(np.arange(4), 3))

    def test_full_design_is_latin_hypercube(self):
        design, _slices = sliced_lhs(2, 3, 2, seed=3)
        n_runs = 6
        for j in range(design.shape[1]):
            cells = np.floor(design[:, j] * n_runs).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n_runs))

    def test_each_slice_is_latin_hypercube(self):
        m, t = 3, 2
        design, slices = sliced_lhs(2, m, t, seed=4)
        for s in range(t):
            rescaled = design * t - slices[:, None]
            sub = rescaled[slices == s]
            for j in range(sub.shape[1]):
                cells = np.floor(sub[:, j] * m).astype(int)
                np.testing.assert_array_equal(np.sort(cells), np.arange(m))

    def test_reproducible_with_seed(self):
        design1, slices1 = sliced_lhs(2, 3, 2, seed=42)
        design2, slices2 = sliced_lhs(2, 3, 2, seed=42)
        np.testing.assert_array_equal(design1, design2)
        np.testing.assert_array_equal(slices1, slices2)

    def test_invalid_n_factors_raises(self):
        with self.assertRaises(ValueError):
            sliced_lhs(0, 3, 2)

    def test_invalid_m_raises(self):
        with self.assertRaises(ValueError):
            sliced_lhs(2, 0, 2)

    def test_invalid_t_raises(self):
        with self.assertRaises(ValueError):
            sliced_lhs(2, 3, 0)


if __name__ == "__main__":
    unittest.main()
