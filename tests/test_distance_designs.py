import unittest

import numpy as np
from scipy.spatial.distance import pdist

from pydoe import maximin_design, minimax_design


class TestMaximinDesign(unittest.TestCase):
    def test_shape(self):
        design = maximin_design(6, 3, iterations=20, seed=0)
        self.assertEqual(design.shape, (6, 3))

    def test_unit_hypercube(self):
        design = maximin_design(6, 3, iterations=20, seed=1)
        self.assertTrue(bool(np.all(design >= 0.0)))
        self.assertTrue(bool(np.all(design < 1.0)))

    def test_columns_are_permutations(self):
        n = 6
        design = maximin_design(n, 3, iterations=20, seed=2)
        for j in range(design.shape[1]):
            cells = np.floor(design[:, j] * n).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n))

    def test_local_search_improves_min_distance(self):
        random_design = maximin_design(8, 2, iterations=0, seed=0)
        optimized_design = maximin_design(8, 2, iterations=100, seed=0)
        random_min_dist = pdist(random_design).min()
        optimized_min_dist = pdist(optimized_design).min()
        self.assertGreaterEqual(optimized_min_dist, random_min_dist)

    def test_reproducible_with_seed(self):
        design1 = maximin_design(5, 2, iterations=20, seed=42)
        design2 = maximin_design(5, 2, iterations=20, seed=42)
        np.testing.assert_array_equal(design1, design2)

    def test_invalid_n_points_raises(self):
        with self.assertRaises(ValueError):
            maximin_design(1, 2)

    def test_invalid_n_factors_raises(self):
        with self.assertRaises(ValueError):
            maximin_design(5, 0)

    def test_invalid_iterations_raises(self):
        with self.assertRaises(ValueError):
            maximin_design(5, 2, iterations=-1)


class TestMinimaxDesign(unittest.TestCase):
    def test_shape(self):
        design = minimax_design(5, 2, n_candidates=200, seed=0)
        self.assertEqual(design.shape, (5, 2))

    def test_unit_hypercube(self):
        design = minimax_design(5, 2, n_candidates=200, seed=1)
        self.assertTrue(bool(np.all(design >= 0.0)))
        self.assertTrue(bool(np.all(design < 1.0)))

    def test_distinct_points(self):
        design = minimax_design(5, 2, n_candidates=200, seed=2)
        self.assertEqual(np.unique(design, axis=0).shape[0], 5)

    def test_reproducible_with_seed(self):
        design1 = minimax_design(5, 2, n_candidates=100, seed=42)
        design2 = minimax_design(5, 2, n_candidates=100, seed=42)
        np.testing.assert_array_equal(design1, design2)

    def test_invalid_n_points_raises(self):
        with self.assertRaises(ValueError):
            minimax_design(0, 2)

    def test_invalid_n_factors_raises(self):
        with self.assertRaises(ValueError):
            minimax_design(5, 0)

    def test_invalid_n_candidates_raises(self):
        with self.assertRaises(ValueError):
            minimax_design(5, 2, n_candidates=4)

    def test_invalid_iterations_raises(self):
        with self.assertRaises(ValueError):
            minimax_design(5, 2, iterations=-1)


if __name__ == "__main__":
    unittest.main()
