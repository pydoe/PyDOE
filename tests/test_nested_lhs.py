import unittest

import numpy as np

from pydoe import nested_lhs


class TestNestedLhs(unittest.TestCase):
    def test_shapes(self):
        small, large = nested_lhs(2, 3, 2, seed=0)
        self.assertEqual(small.shape, (3, 2))
        self.assertEqual(large.shape, (6, 2))

    def test_unit_hypercube(self):
        small, large = nested_lhs(3, 4, 3, seed=1)
        for design in (small, large):
            self.assertTrue(bool(np.all(design >= 0.0)))
            self.assertTrue(bool(np.all(design < 1.0)))

    def test_small_is_latin_hypercube(self):
        small, _large = nested_lhs(2, 5, 2, seed=2)
        n1 = 5
        for j in range(small.shape[1]):
            cells = np.floor(small[:, j] * n1).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n1))

    def test_large_is_latin_hypercube(self):
        _small, large = nested_lhs(2, 5, 2, seed=2)
        n2 = 10
        for j in range(large.shape[1]):
            cells = np.floor(large[:, j] * n2).astype(int)
            np.testing.assert_array_equal(np.sort(cells), np.arange(n2))

    def test_nesting_property(self):
        n1, k = 4, 3
        small, large = nested_lhs(2, n1, k, seed=3)
        n2 = n1 * k
        for j in range(2):
            small_cells = np.floor(small[:, j] * n1).astype(int)
            large_cells = np.floor(large[:, j] * n2).astype(int)
            np.testing.assert_array_equal(
                large_cells // k, np.repeat(small_cells, k)
            )

    def test_reproducible_with_seed(self):
        small1, large1 = nested_lhs(2, 3, 2, seed=42)
        small2, large2 = nested_lhs(2, 3, 2, seed=42)
        np.testing.assert_array_equal(small1, small2)
        np.testing.assert_array_equal(large1, large2)

    def test_invalid_n_factors_raises(self):
        with self.assertRaises(ValueError):
            nested_lhs(0, 3, 2)

    def test_invalid_n1_raises(self):
        with self.assertRaises(ValueError):
            nested_lhs(2, 0, 2)

    def test_invalid_k_raises(self):
        with self.assertRaises(ValueError):
            nested_lhs(2, 3, 0)


if __name__ == "__main__":
    unittest.main()
