import itertools
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

        # 8 runs should support 7 factors at resolution III: the saturated
        # 2^(7-4) design (NIST Table 3.17, Montgomery Table 8.14)
        actual = fracfact_by_res(7, 3)
        self.assertEqual(actual.shape[0], 8)  # 2^3 runs
        self.assertEqual(actual.shape[1], 7)

        # 16 runs should support 10 factors at resolution III
        actual = fracfact_by_res(10, 3)
        self.assertEqual(actual.shape[0], 16)  # 2^4 runs
        self.assertEqual(actual.shape[1], 10)

    def test_fracfact_by_res_resolution_IV(self):
        """Test resolution IV designs"""
        # 8 runs should support 4 factors at resolution IV: 2^(4-1)
        actual = fracfact_by_res(4, 4)
        self.assertEqual(actual.shape[0], 8)  # 2^3 runs
        self.assertEqual(actual.shape[1], 4)

        # 16 runs should support 8 factors at resolution IV: 2^(8-4)
        actual = fracfact_by_res(8, 4)
        self.assertEqual(actual.shape[0], 16)  # 2^4 runs
        self.assertEqual(actual.shape[1], 8)

        # 32 runs should support 11 factors at resolution IV: 2^(11-6)
        actual = fracfact_by_res(11, 4)
        self.assertEqual(actual.shape[0], 32)  # 2^5 runs
        self.assertEqual(actual.shape[1], 11)

    def test_fracfact_by_res_resolution_V(self):
        """Test resolution V designs"""
        # 16 runs should support 5 factors at resolution V: 2^(5-1)
        actual = fracfact_by_res(5, 5)
        self.assertEqual(actual.shape[0], 16)  # 2^4 runs
        self.assertEqual(actual.shape[1], 5)

        # 64 runs should support 8 factors at resolution V: 2^(8-2)
        actual = fracfact_by_res(8, 5)
        self.assertEqual(actual.shape[0], 64)  # 2^6 runs
        self.assertEqual(actual.shape[1], 8)

        # 256 runs should support 12 factors at resolution V
        actual = fracfact_by_res(12, 5)
        self.assertEqual(actual.shape[0], 256)  # 2^8 runs
        self.assertEqual(actual.shape[1], 12)

    def test_fracfact_by_res_higher_resolution_runs(self):
        """Test that higher resolution requires same or more runs"""
        # For 11 factors, resolution IV requires 32 runs, V requires 128
        iv_runs = fracfact_by_res(11, 4).shape[0]
        v_runs = fracfact_by_res(11, 5).shape[0]
        self.assertGreater(v_runs, iv_runs)
        self.assertEqual(v_runs, 128)
        self.assertEqual(iv_runs, 32)

        # For 8 factors, resolution IV requires 16 runs, V requires 64
        iv_runs = fracfact_by_res(8, 4).shape[0]
        v_runs = fracfact_by_res(8, 5).shape[0]
        self.assertGreater(v_runs, iv_runs)
        self.assertEqual(v_runs, 64)
        self.assertEqual(iv_runs, 16)

    def test_fracfact_by_res_invalid_n(self):
        with pytest.raises(ValueError, match="n must be at least 2"):
            fracfact_by_res(1, 3)

    def test_fracfact_by_res_invalid_resolution(self):
        with pytest.raises(ValueError, match="resolution must be >= 3"):
            fracfact_by_res(3, 2)

    def test_fracfact_by_res_saturated_design(self):
        # 6 factors is one short of saturating 8 runs
        design = fracfact_by_res(6, 3)
        self.assertEqual(design.shape, (8, 6))  # 2^3 runs, 6 factors
        self.assertEqual(set(np.unique(design)), {-1.0, 1.0})

    def test_fracfact_by_res_beyond_saturation(self):
        # 16 factors do not fit in 16 runs, so 32 are needed
        design = fracfact_by_res(16, 3)
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
        # 6 factors at resolution V need 32 runs: 2^(6-1)
        self.assertEqual(design.shape, (32, 6))
        self.assertEqual(set(np.unique(design)), {-1.0, 1.0})

    def test_fracfact_by_res_columns_match_n(self):
        for n, res in [(5, 3), (7, 4), (9, 5)]:
            design = fracfact_by_res(n, res)
            self.assertEqual(design.shape[1], n)

    def test_issue_153_seven_factors_resolution_III(self):
        """2^(7-4)_III needs 8 runs, not 16.

        https://www.itl.nist.gov/div898/handbook/pri/section3/eqns/2to7m4.txt
        """
        design = fracfact_by_res(7, 3)
        self.assertEqual(design.shape, (8, 7))
        self.assertEqual(_resolution(design), 3)
        # this is exactly the 8-run Plackett-Burman design
        np.testing.assert_allclose(design, fracfact("a b c ab ac bc abc"))

    def test_fracfact_by_res_minimum_runs(self):
        """Run counts must match the published minima for n = 2..15.

        Sources: NIST/SEMATECH e-Handbook Table 3.17 and Montgomery,
        *Design and Analysis of Experiments* (8th ed.), Table 8.14. The
        underlying theory is that 2^k runs carry at most 2^k - 1 factors at
        resolution III and at most 2^(k-1) at resolution IV; resolution V
        has no closed form.
        """
        minimum_runs = {
            3: [4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16],
            4: [4, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32],
            5: [4, 8, 16, 16, 32, 64, 64, 128, 128, 128, 256, 256, 256, 256],
        }
        for res, runs in minimum_runs.items():
            for n, expected in zip(range(2, 16), runs, strict=True):
                design = fracfact_by_res(n, res)
                self.assertEqual(
                    design.shape,
                    (expected, n),
                    msg=f"fracfact_by_res({n}, {res})",
                )

    def test_fracfact_by_res_achieves_requested_resolution(self):
        """The whole defining relation, not just each generator, must be
        long enough. Products of generators used to collapse: resolution V
        requests for 9 or more factors returned resolution IV designs.
        """
        for res in (3, 4, 5):
            for n in range(2, 13):
                design = fracfact_by_res(n, res)
                self.assertGreaterEqual(
                    _resolution(design), res, msg=f"fracfact_by_res({n}, {res})"
                )

    def test_fracfact_by_res_canonical_designs(self):
        """Reproduce the textbook designs exactly."""
        canonical = {
            (3, 3): "a b ab",
            (5, 3): "a b c ab ac",
            (6, 3): "a b c ab ac bc",
            (7, 3): "a b c ab ac bc abc",
            (4, 4): "a b c abc",
            (5, 5): "a b c d abcd",
            (8, 5): "a b c d e f abcd abef",
        }
        for (n, res), gen in canonical.items():
            np.testing.assert_allclose(
                fracfact_by_res(n, res),
                fracfact(gen),
                err_msg=f"fracfact_by_res({n}, {res}) != fracfact({gen!r})",
            )

    def test_fracfact_by_res_fewer_factors_than_base(self):
        """n <= k used to raise from islice(); it is a full factorial."""
        for n, res in [(2, 4), (2, 5), (3, 5), (4, 5), (3, 4)]:
            design = fracfact_by_res(n, res)
            np.testing.assert_allclose(design, ff2n(n))


def _resolution(design):
    """Length of the shortest word in the defining relation."""
    n_factors = design.shape[1]
    for size in range(1, n_factors + 1):
        for combination in itertools.combinations(range(n_factors), size):
            if np.all(np.prod(design[:, combination], axis=1) == 1):
                return size
    return np.inf  # full factorial: no defining word


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
