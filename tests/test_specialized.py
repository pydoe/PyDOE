import itertools
import unittest

import numpy as np

from pydoe import definitive_screening_design, supersaturated_design


class TestDefinitiveScreeningDesign(unittest.TestCase):
    def test_historical_four_factor_result_is_unchanged(self):
        expected = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0, -1.0],
            [-1.0, -1.0, 0.0, 1.0],
            [-1.0, 1.0, -1.0, 0.0],
            [0.0, -1.0, -1.0, -1.0],
            [1.0, 0.0, -1.0, 1.0],
            [1.0, 1.0, 0.0, -1.0],
            [1.0, -1.0, 1.0, 0.0],
        ])
        np.testing.assert_array_equal(definitive_screening_design(4), expected)

    def test_shape(self):
        for k in [4, 6, 8, 12]:
            design = definitive_screening_design(k)
            self.assertEqual(design.shape, (2 * k + 1, k))

    def test_levels_are_three(self):
        design = definitive_screening_design(4)
        self.assertEqual(set(np.unique(design).tolist()), {-1.0, 0.0, 1.0})

    def test_first_row_is_center(self):
        design = definitive_screening_design(6)
        np.testing.assert_array_equal(design[0], np.zeros(6))

    def test_main_effects_orthogonal(self):
        for k in [4, 6, 8]:
            design = definitive_screening_design(k)
            gram = design.T @ design
            expected = 2 * (k - 1) * np.eye(k)
            np.testing.assert_allclose(gram, expected)

    def test_foldover_structure(self):
        k = 4
        design = definitive_screening_design(k)
        np.testing.assert_allclose(design[1 : k + 1], -design[k + 1 :])

    def test_continuous_designs_have_defining_properties(self):
        for k in range(4, 18):
            with self.subTest(k=k):
                design = definitive_screening_design(k)
                rows = {tuple(row) for row in design}
                self.assertTrue(all(tuple(-row) in rows for row in design))

                np.testing.assert_allclose(design.sum(axis=0), 0)
                gram = design.T @ design
                np.testing.assert_allclose(gram - np.diag(np.diag(gram)), 0)

                for i, j in itertools.combinations(range(k), 2):
                    interaction = design[:, i] * design[:, j]
                    np.testing.assert_allclose(design.T @ interaction, 0)

    def test_previously_unsupported_counts_use_next_construction(self):
        self.assertEqual(definitive_screening_design(5).shape, (13, 5))
        self.assertEqual(definitive_screening_design(7).shape, (17, 7))

    def test_efficient_special_constructions(self):
        expected_runs = {9: 21, 10: 21, 15: 33, 16: 33, 25: 53, 26: 53}
        for k, n_runs in expected_runs.items():
            with self.subTest(k=k):
                design = definitive_screening_design(k)
                self.assertEqual(design.shape, (n_runs, k))

    def test_mixed_design_matches_normalized_upstream_result(self):
        expected = np.array([
            [0, 0, 0, -1, -1],
            [0, 0, 0, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 0, -1, 1, 1],
            [1, -1, 0, -1, 1],
            [1, 1, -1, 1, -1],
            [1, 1, 1, -1, 1],
            [1, -1, 1, 1, -1],
            [0, -1, -1, -1, -1],
            [-1, 0, 1, -1, -1],
            [-1, 1, 0, 1, -1],
            [-1, -1, 1, -1, 1],
            [-1, -1, -1, 1, -1],
            [-1, 1, -1, -1, 1],
        ])
        actual = definitive_screening_design(3, n_cat=2)
        np.testing.assert_array_equal(actual, expected)

    def test_mixed_design_levels_and_column_order(self):
        design = definitive_screening_design(3, n_cat=2)
        self.assertEqual(design.shape, (14, 5))
        self.assertEqual(
            set(np.unique(design[:, :3]).tolist()), {-1.0, 0.0, 1.0}
        )
        self.assertEqual(set(np.unique(design[:, 3:]).tolist()), {-1.0, 1.0})

    def test_orth_method_has_orthogonal_main_effects(self):
        design = definitive_screening_design(3, n_cat=2, method="orth")
        self.assertEqual(design.shape, (16, 5))
        gram = design.T @ design
        np.testing.assert_array_equal(
            gram - np.diag(np.diag(gram)), np.zeros_like(gram)
        )
        np.testing.assert_array_equal(design.sum(axis=0), np.zeros(5))

    def test_min_13_and_fake_factors_enlarge_design(self):
        default = definitive_screening_design(4)
        automatic = definitive_screening_design(4, min_13=True)
        explicit = definitive_screening_design(4, n_fake_factors=2)
        self.assertEqual(default.shape, (9, 4))
        self.assertEqual(automatic.shape, (13, 4))
        np.testing.assert_array_equal(automatic, explicit)

    def test_fake_columns_are_removed_before_categorical_columns(self):
        design = definitive_screening_design(3, n_cat=2, n_fake_factors=1)
        self.assertEqual(design.shape, (14, 5))
        self.assertEqual(set(np.unique(design[:, -2:]).tolist()), {-1.0, 1.0})

    def test_categorical_only_design(self):
        dsd = definitive_screening_design(0, n_cat=3)
        orth = definitive_screening_design(0, n_cat=3, method="orth")
        self.assertEqual(dsd.shape, (10, 3))
        self.assertEqual(orth.shape, (12, 3))
        self.assertEqual(set(np.unique(dsd).tolist()), {-1.0, 1.0})
        self.assertEqual(set(np.unique(orth).tolist()), {-1.0, 1.0})

    def test_result_is_deterministic_float_array(self):
        first = definitive_screening_design(5, n_cat=2)
        second = definitive_screening_design(5, n_cat=2)
        self.assertIsInstance(first, np.ndarray)
        self.assertTrue(np.issubdtype(first.dtype, np.floating))
        np.testing.assert_array_equal(first, second)

    def test_invalid_inputs_raise_value_error(self):
        for name, args, kwargs in [
            ("negative k", (-1,), {}),
            ("boolean k", (True,), {}),
            ("noninteger k", (3.5,), {}),
            ("negative n_cat", (3,), {"n_cat": -1}),
            ("boolean n_cat", (3,), {"n_cat": False}),
            ("negative fake", (3,), {"n_fake_factors": -1}),
            ("boolean fake", (3,), {"n_fake_factors": True}),
            ("no real factors", (0,), {}),
            ("bad method", (3,), {"method": "invalid"}),
            ("nonstring method", (3,), {"method": []}),
            ("nonboolean min_13", (3,), {"min_13": 1}),
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                definitive_screening_design(*args, **kwargs)


class TestSupersaturatedDesign(unittest.TestCase):
    def test_shape(self):
        design = supersaturated_design(8, 4, iterations=50, seed=0)
        self.assertEqual(design.shape, (4, 8))

    def test_levels_are_pm_one(self):
        design = supersaturated_design(8, 4, iterations=50, seed=0)
        self.assertEqual(set(np.unique(design).tolist()), {-1.0, 1.0})

    def test_reproducible_with_seed(self):
        d1 = supersaturated_design(8, 4, iterations=50, seed=42)
        d2 = supersaturated_design(8, 4, iterations=50, seed=42)
        np.testing.assert_array_equal(d1, d2)

    def test_raises_n_runs_too_small(self):
        with self.assertRaises(ValueError):
            supersaturated_design(8, 1)

    def test_raises_not_supersaturated(self):
        with self.assertRaises(ValueError):
            supersaturated_design(4, 4)
        with self.assertRaises(ValueError):
            supersaturated_design(4, 6)
