import unittest

import numpy as np
import pytest

from pydoe import ff2n, fracfact, fracfact_by_res, fracfact_opt, fullfact
from pydoe.factorial.factorial import validate_generator


class TestFactorial(unittest.TestCase):
    def test_factorial1(self):
        expected = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
            [0.0, 3.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 2.0],
            [0.0, 2.0, 2.0],
            [1.0, 2.0, 2.0],
            [0.0, 3.0, 2.0],
            [1.0, 3.0, 2.0],
        ]
        actual = fullfact([2, 4, 3])
        np.testing.assert_allclose(actual, expected)

    def test_factorial2(self):
        expected = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
        actual = ff2n(3)
        np.testing.assert_allclose(actual, expected)

    def test_factorial3(self):
        expected = [
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
        actual = fracfact("a b ab")
        np.testing.assert_allclose(actual, expected)

    def test_factorial4(self):
        expected = [
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
        actual = fracfact("A B AB")
        np.testing.assert_allclose(actual, expected)

    def test_factorial5(self):
        expected = [
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0, 1.0],
        ]
        actual = fracfact("a b -ab c +abc")
        np.testing.assert_allclose(actual, expected)

    def test_factorial6(self):
        expected = [
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        actual = fracfact_by_res(6, 3)
        np.testing.assert_allclose(actual, expected)

    def test_issue_9(self):
        ffo_doe = fracfact_opt(4, 1)
        self.assertEqual(ffo_doe[0], "a b c abc")
        self.assertEqual(
            ffo_doe[1],
            [
                "a = bcd",
                "b = acd",
                "c = abd",
                "d = abc",
                "ab = cd",
                "ac = bd",
                "ad = bc",
                "abcd",
            ],
        )
        np.testing.assert_array_equal(
            ffo_doe[2],
            np.array([0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def test_issue_152_saturated_designs(self):
        """A 2^(k-p) design with k = 2^m - 1 factors in 2^m runs is legal.

        These are the *saturated* resolution III designs: every one of the
        2^m - m - 1 interaction columns of the m main factors carries an
        erased factor.
        """
        for n_factors, n_erased, n_runs in [(3, 1, 4), (7, 4, 8), (15, 11, 16)]:
            gen, alias_map, alias_vector = fracfact_opt(n_factors, n_erased)
            design = fracfact(gen)
            self.assertEqual(design.shape, (n_runs, n_factors))
            self.assertEqual(len(gen.split()), n_factors)
            # a saturated design uses every available interaction column
            self.assertEqual(len(set(gen.split())), n_factors)
            # main effects must never be aliased with one another
            self.assertEqual(alias_vector[0], 0.0)
            self.assertTrue(alias_map)

    def test_issue_152_expected_saturated_generators(self):
        """The 2^(7-4) and 2^(15-11) designs use every interaction column."""
        gen, _, _ = fracfact_opt(7, 4)
        self.assertEqual(
            set(gen.split()), {"a", "b", "c", "ab", "ac", "bc", "abc"}
        )

        gen, _, _ = fracfact_opt(15, 11)
        self.assertEqual(len(set(gen.split())), 15)
        self.assertEqual(fracfact(gen).shape, (16, 15))

    def test_fracfact_opt_no_erased_factors(self):
        """n_erased = 0 must return the full factorial, not raise."""
        gen, alias_map, alias_vector = fracfact_opt(3, 0)
        self.assertEqual(gen, "a b c")
        np.testing.assert_allclose(fracfact(gen), ff2n(3))
        self.assertEqual(len(alias_map), 2**3 - 1)
        np.testing.assert_array_equal(alias_vector, np.zeros(6))

    def test_fracfact_opt_too_many_erased_factors(self):
        """Beyond the saturated design there is nothing left to alias."""
        with pytest.raises(
            ValueError, match="Too many erased factors to create aliasing"
        ):
            fracfact_opt(4, 2)  # 2 main factors provide only 1 interaction
        with pytest.raises(
            ValueError, match="Too many erased factors to create aliasing"
        ):
            fracfact_opt(8, 5)  # 3 main factors provide only 4 interactions

    def test_fracfact_by_res_resolution_III(self):
        """Test resolution III designs"""
        # 8 runs should support 4 factors at resolution III
        actual = fracfact_by_res(4, 3)
        self.assertEqual(actual.shape[0], 8)  # 2^3 runs
        self.assertEqual(actual.shape[1], 4)

        # 16 runs should support 7 factors at resolution III
        actual = fracfact_by_res(7, 3)
        self.assertEqual(actual.shape[0], 16)  # 2^4 runs
        self.assertEqual(actual.shape[1], 7)

        # 16 runs should support 10 factors at resolution III
        actual = fracfact_by_res(10, 3)
        self.assertEqual(actual.shape[0], 16)  # 2^4 runs
        self.assertEqual(actual.shape[1], 10)

    def test_fracfact_by_res_resolution_IV(self):
        """Test resolution IV designs"""
        # 16 runs should support 4 factors at resolution IV
        actual = fracfact_by_res(4, 4)
        self.assertEqual(actual.shape[0], 16)  # 2^4 runs
        self.assertEqual(actual.shape[1], 4)

        # 32 runs should support 8 factors at resolution IV
        actual = fracfact_by_res(8, 4)
        self.assertEqual(actual.shape[0], 32)  # 2^5 runs
        self.assertEqual(actual.shape[1], 8)

        # 64 runs should support 11 factors at resolution IV
        actual = fracfact_by_res(11, 4)
        self.assertEqual(actual.shape[0], 64)  # 2^6 runs
        self.assertEqual(actual.shape[1], 11)

    def test_fracfact_by_res_resolution_V(self):
        """Test resolution V designs"""
        # 32 runs should support 5 factors at resolution V
        actual = fracfact_by_res(5, 5)
        self.assertEqual(actual.shape[0], 32)  # 2^5 runs
        self.assertEqual(actual.shape[1], 5)

        # 128 runs should support 8 factors at resolution V
        actual = fracfact_by_res(8, 5)
        self.assertEqual(actual.shape[0], 128)  # 2^7 runs
        self.assertEqual(actual.shape[1], 8)

        # 256 runs should support 12 factors at resolution V
        actual = fracfact_by_res(12, 5)
        self.assertEqual(actual.shape[0], 256)  # 2^8 runs
        self.assertEqual(actual.shape[1], 12)

    def test_fracfact_by_res_higher_resolution_runs(self):
        """Test that higher resolution requires same or more runs"""
        # For 11 factors, resolution IV requires 64 runs, V requires 128
        iv_runs = fracfact_by_res(11, 4).shape[0]
        v_runs = fracfact_by_res(11, 5).shape[0]
        self.assertGreater(v_runs, iv_runs)
        self.assertEqual(v_runs, 128)
        self.assertEqual(iv_runs, 64)

        # For 8 factors, resolution IV requires 32 runs, V requires 128
        iv_runs = fracfact_by_res(8, 4).shape[0]
        v_runs = fracfact_by_res(8, 5).shape[0]
        self.assertGreater(v_runs, iv_runs)
        self.assertEqual(v_runs, 128)
        self.assertEqual(iv_runs, 32)

    def test_fracfact_by_res_invalid_n(self):
        with pytest.raises(ValueError, match="n must be at least 2"):
            fracfact_by_res(1, 3)

    def test_fracfact_by_res_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution must be >= 3"):
            fracfact_by_res(3, 2)

    def test_fracfact_by_res_table_lookup_path(self):
        # (6, 3) is explicitly in the DOE table -> k = 3
        design = fracfact_by_res(6, 3)
        self.assertEqual(design.shape, (8, 6))  # 2^3 runs, 6 factors
        self.assertEqual(set(np.unique(design)), {-1.0, 1.0})

    def test_fracfact_by_res_fallback_calculation_path(self):
        # (16, 3) is not in the table -> fallback calculation
        design = fracfact_by_res(16, 3)
        # k = ceil(log2(17)) = 5 -> 32 runs
        self.assertEqual(design.shape, (32, 16))
        self.assertEqual(set(np.unique(design)), {-1.0, 1.0})

    def test_fracfact_by_res_exceeds_base_factor_limit(self):
        with pytest.raises(ValueError, match="more than 26 base factors"):
            fracfact_by_res(2**27, 3)

    def test_fracfact_by_res_too_many_base_factors(self):
        # Force k > 26 via fallback logic
        with pytest.raises(ValueError, match="more than 26 base factors"):
            fracfact_by_res(2**27, 3)

    def test_fracfact_by_res_resolution_five(self):
        design = fracfact_by_res(6, 5)
        # From table: (6,5) -> k = 6 -> 64 runs
        self.assertEqual(design.shape, (64, 6))
        self.assertEqual(set(np.unique(design)), {-1.0, 1.0})

    def test_fracfact_by_res_columns_match_n(self):
        for n, res in [(5, 3), (7, 4), (9, 5)]:
            design = fracfact_by_res(n, res)
            self.assertEqual(design.shape[1], n)


@pytest.mark.parametrize(
    "n_factors, generator, message",
    [
        (2, "a b c", "Generator does not match the number of factors."),
        (2, "a a", "Main factors are confounded with each other."),
        (2, "a c", "Use the letters `a b` for the main factors."),
        (5, "a b c ab ab", "Generators are not unique."),
        (5, "a b c ab ad", "Generators are not valid."),
        (2, "ab ac", "At least one unconfounded main factor is needed."),
    ],
)
def test_validate_generator_invalid(
    n_factors: int, generator: str, message: str
):
    with pytest.raises(ValueError, match=message):
        validate_generator(n_factors, generator)
