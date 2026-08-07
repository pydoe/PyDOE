import unittest

import numpy as np

from pydoe import nearly_orthogonal_lhs


def _max_abs_corr(design):
    corr = np.corrcoef(design, rowvar=False)
    n_factors = corr.shape[0]
    mask = ~np.eye(n_factors, dtype=bool)
    return float(np.max(np.abs(corr[mask])))


class TestNearlyOrthogonalLhs(unittest.TestCase):
    def test_shape(self):
        design = nearly_orthogonal_lhs(8, 3, iterations=50, seed=0)
        self.assertEqual(design.shape, (8, 3))

    def test_unit_hypercube(self):
        design = nearly_orthogonal_lhs(10, 4, iterations=50, seed=1)
        self.assertTrue(bool(np.all(design >= 0.0)))
        self.assertTrue(bool(np.all(design < 1.0)))

    def test_is_latin_hypercube(self):
        n_points = 12
        design = nearly_orthogonal_lhs(n_points, 4, iterations=50, seed=2)
        for j in range(design.shape[1]):
            cells = np.floor(design[:, j] * n_points).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n_points))

    def test_correlation_improves(self):
        n_points, n_factors, seed = 12, 4, 0
        design_initial = nearly_orthogonal_lhs(
            n_points, n_factors, iterations=0, seed=seed
        )
        design_optimized = nearly_orthogonal_lhs(
            n_points, n_factors, iterations=300, seed=seed
        )
        self.assertLessEqual(
            _max_abs_corr(design_optimized), _max_abs_corr(design_initial)
        )

    def test_single_factor(self):
        n_points = 6
        design = nearly_orthogonal_lhs(n_points, 1, seed=3)
        self.assertEqual(design.shape, (n_points, 1))
        cells = np.floor(design[:, 0] * n_points).astype(int)
        np.testing.assert_array_equal(np.sort(cells), np.arange(n_points))

    def test_reproducible_with_seed(self):
        design1 = nearly_orthogonal_lhs(8, 3, iterations=50, seed=42)
        design2 = nearly_orthogonal_lhs(8, 3, iterations=50, seed=42)
        np.testing.assert_array_equal(design1, design2)

    def test_invalid_n_points_raises(self):
        with self.assertRaises(ValueError):
            nearly_orthogonal_lhs(1, 3)

    def test_invalid_n_factors_raises(self):
        with self.assertRaises(ValueError):
            nearly_orthogonal_lhs(5, 0)

    def test_invalid_iterations_raises(self):
        with self.assertRaises(ValueError):
            nearly_orthogonal_lhs(5, 2, iterations=-1)


if __name__ == "__main__":
    unittest.main()
