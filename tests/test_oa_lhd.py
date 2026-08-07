import unittest

import numpy as np

from pydoe import get_orthogonal_array, oa_lhd


class TestOaLhd(unittest.TestCase):
    def test_shape(self):
        oa = get_orthogonal_array("L9(3^4)")
        design = oa_lhd(oa, seed=0)
        self.assertEqual(design.shape, (9, 4))

    def test_unit_hypercube(self):
        oa = get_orthogonal_array("L9(3^4)")
        design = oa_lhd(oa, seed=1)
        self.assertTrue(bool(np.all(design >= 0.0)))
        self.assertTrue(bool(np.all(design < 1.0)))

    def test_columns_are_latin_hypercube_permutations(self):
        oa = get_orthogonal_array("L9(3^4)")
        design = oa_lhd(oa, seed=2)
        n_runs = oa.shape[0]
        for j in range(design.shape[1]):
            cells = np.floor(design[:, j] * n_runs).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n_runs))

    def test_reproducible_with_seed(self):
        oa = get_orthogonal_array("L9(3^4)")
        design1 = oa_lhd(oa, seed=42)
        design2 = oa_lhd(oa, seed=42)
        np.testing.assert_array_equal(design1, design2)

    def test_different_seeds_differ(self):
        oa = get_orthogonal_array("L9(3^4)")
        design1 = oa_lhd(oa, seed=1)
        design2 = oa_lhd(oa, seed=2)
        self.assertFalse(bool(np.all(design1 == design2)))

    def test_invalid_levels_raises(self):
        oa = np.array([[0, 1], [1, 2], [2, 0], [0, 1]])
        with self.assertRaises(ValueError):
            oa_lhd(oa)

    def test_l8_2_level_array(self):
        oa = get_orthogonal_array("L8(2^7)")
        design = oa_lhd(oa, seed=0)
        self.assertEqual(design.shape, (8, 7))
        n_runs = oa.shape[0]
        for j in range(design.shape[1]):
            cells = np.floor(design[:, j] * n_runs).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n_runs))


if __name__ == "__main__":
    unittest.main()
