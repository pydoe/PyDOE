"""Tests for John's 3/4 fractional factorial design."""

import unittest

import numpy as np

from pydoe import john_three_quarter_design


class TestJohnThreeQuarterDesign(unittest.TestCase):
    # ------------------------------------------------------------------
    # Shape tests
    # ------------------------------------------------------------------

    def test_shape_k3(self):
        # 3 * 2^(3-2) = 3*2 = 6 runs, 3 factors
        design = john_three_quarter_design(3)
        self.assertEqual(design.shape, (6, 3))

    def test_shape_k4(self):
        # 3 * 2^(4-2) = 3*4 = 12 runs, 4 factors
        design = john_three_quarter_design(4)
        self.assertEqual(design.shape, (12, 4))

    def test_shape_k5(self):
        # 3 * 2^(5-2) = 3*8 = 24 runs, 5 factors
        design = john_three_quarter_design(5)
        self.assertEqual(design.shape, (24, 5))

    def test_shape_k6(self):
        # 3 * 2^(6-2) = 3*16 = 48 runs, 6 factors
        design = john_three_quarter_design(6)
        self.assertEqual(design.shape, (48, 6))

    def test_shape_formula(self):
        # Shape must be (3 * 2^(k-2), k) for all valid k
        for k in range(3, 8):
            design = john_three_quarter_design(k)
            expected_rows = 3 * 2 ** (k - 2)
            self.assertEqual(design.shape, (expected_rows, k))

    # ------------------------------------------------------------------
    # Value tests
    # ------------------------------------------------------------------

    def test_values_binary(self):
        # All entries must be exactly -1 or +1
        for k in [3, 4, 5]:
            design = john_three_quarter_design(k)
            self.assertTrue(
                np.all(np.isin(design, [-1, 1])),
                msg=f"Non-binary values found for k={k}",
            )

    def test_half_fraction_is_correct(self):
        # The first 2^(k-1) rows must form the half-fraction defined by
        # X_k = X_1 * X_2 * ... * X_{k-1}
        k = 4
        design = john_three_quarter_design(k)
        half = design[: 2 ** (k - 1)]  # first 8 rows
        # Verify the generator: last column = product of all others
        gen = np.prod(half[:, :-1], axis=1)
        np.testing.assert_array_equal(half[:, -1], gen)

    def test_augment_rows_have_fold_on_positive(self):
        # The augmented (last 2^(k-2)) rows must all have fold_on = +1
        k = 4
        fold_on = 1
        design = john_three_quarter_design(k, fold_on=fold_on)
        augment = design[2 ** (k - 1) :]  # last 4 rows
        j = fold_on - 1
        self.assertTrue(np.all(augment[:, j] == 1))

    def test_augment_rows_match_nist_k4(self):
        # Verify the four augmented rows. These are the 4 rows of the
        # half-fraction where X1=-1, with X1 flipped to +1 (semifoldover).
        # ff2n(3) orders X1 slowest (col0 = MSB), so X1=-1 rows come first.
        expected_augment = np.array([
            [1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0],
        ])
        design = john_three_quarter_design(4, fold_on=1)
        augment = design[8:]  # last 4 rows
        np.testing.assert_array_equal(augment, expected_augment)

    def test_base_half_fraction_nist_k4(self):
        # Verify the 8 base half-fraction rows (generator X4 = X1*X2*X3).
        # ff2n(3) has X1 (col0) alternating slowest, X3 (col2) fastest.
        expected_base = np.array([
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
        design = john_three_quarter_design(4, fold_on=1)
        np.testing.assert_array_equal(design[:8], expected_base)

    # ------------------------------------------------------------------
    # fold_on parameter
    # ------------------------------------------------------------------

    def test_fold_on_different_factors(self):
        # Folding on different factors should produce different designs
        # but all have the same shape and binary values
        k = 4
        designs = [
            john_three_quarter_design(k, fold_on=j) for j in range(1, k + 1)
        ]
        shapes = [d.shape for d in designs]
        self.assertTrue(all(s == (12, 4) for s in shapes))
        # Designs folded on different factors must differ
        self.assertFalse(np.array_equal(designs[0], designs[1]))

    def test_fold_on_last_factor(self):
        # The last factor is the generated one (X_k = product of free factors).
        # Folding on it is mathematically valid.
        design = john_three_quarter_design(4, fold_on=4)
        self.assertEqual(design.shape, (12, 4))
        self.assertTrue(np.all(np.isin(design, [-1, 1])))
        # All augmented rows have X4 = +1
        augment = design[8:]
        self.assertTrue(np.all(augment[:, 3] == 1))

    def test_fold_on_changes_aliasing(self):
        # After folding on X1, the augmented rows should contain no
        # duplicate rows from the base half-fraction (each new run is unique).
        design = john_three_quarter_design(4, fold_on=1)
        base = design[:8]
        augment = design[8:]
        for aug_row in augment:
            matches = np.all(base == aug_row, axis=1)
            self.assertFalse(
                np.any(matches), "Augmented row is duplicate of base row"
            )

    # ------------------------------------------------------------------
    # Run count = exactly 3/4 of 2^k
    # ------------------------------------------------------------------

    def test_run_count_is_three_quarters(self):
        for k in range(3, 7):
            design = john_three_quarter_design(k)
            full_runs = 2**k
            expected = (3 * full_runs) // 4
            self.assertEqual(design.shape[0], expected)

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_k_less_than_3(self):
        with self.assertRaises(ValueError):
            john_three_quarter_design(2)

    def test_raises_k_equals_1(self):
        with self.assertRaises(ValueError):
            john_three_quarter_design(1)

    def test_raises_fold_on_zero(self):
        with self.assertRaises(ValueError):
            john_three_quarter_design(4, fold_on=0)

    def test_raises_fold_on_too_large(self):
        with self.assertRaises(ValueError):
            john_three_quarter_design(4, fold_on=5)
