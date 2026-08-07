import unittest

import numpy as np

from pydoe import maxpro_design


def _psi(design):
    diff = design[:, None, :] - design[None, :, :]
    sq = diff**2
    prod = np.prod(sq, axis=2)
    n_points = design.shape[0]
    iu = np.triu_indices(n_points, k=1)
    return float(np.sum(1.0 / prod[iu]))


class TestMaxproDesign(unittest.TestCase):
    def test_shape(self):
        design = maxpro_design(5, 3, iterations=10, seed=0)
        self.assertEqual(design.shape, (5, 3))

    def test_unit_hypercube(self):
        design = maxpro_design(6, 2, iterations=10, seed=1)
        self.assertTrue(bool(np.all(design >= 0.0)))
        self.assertTrue(bool(np.all(design < 1.0)))

    def test_latin_hypercube_structure(self):
        n_points = 6
        design = maxpro_design(n_points, 3, iterations=20, seed=2)
        for j in range(design.shape[1]):
            cells = np.floor(design[:, j] * n_points).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n_points))

    def test_optimization_improves_criterion(self):
        n_points, n_factors = 8, 3
        design_0 = maxpro_design(n_points, n_factors, iterations=0, seed=0)
        design_opt = maxpro_design(n_points, n_factors, iterations=200, seed=0)
        self.assertLessEqual(_psi(design_opt), _psi(design_0))

    def test_reproducible_with_seed(self):
        design1 = maxpro_design(5, 2, iterations=50, seed=42)
        design2 = maxpro_design(5, 2, iterations=50, seed=42)
        np.testing.assert_array_equal(design1, design2)

    def test_invalid_n_points_raises(self):
        with self.assertRaises(ValueError):
            maxpro_design(1, 2)

    def test_invalid_n_factors_raises(self):
        with self.assertRaises(ValueError):
            maxpro_design(5, 0)

    def test_invalid_iterations_raises(self):
        with self.assertRaises(ValueError):
            maxpro_design(5, 2, iterations=-1)


if __name__ == "__main__":
    unittest.main()
