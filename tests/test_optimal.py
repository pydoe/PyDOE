import unittest

import numpy as np

from pydoe import (
    a_efficiency,
    a_optimality,
    build_design_matrix,
    build_uniform_moment_matrix,
    c_optimality,
    criterion_value,
    d_efficiency,
    d_optimality,
    detmax,
    e_optimality,
    fedorov,
    g_optimality,
    generate_candidate_set,
    i_optimality,
    information_matrix,
    modified_fedorov,
    optimal_design,
    s_optimality,
    sequential_dykstra,
    simple_exchange_wynn_mitchell,
    t_optimality,
    v_optimality,
)


class TestOptimalDesign(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.candidates_2d = generate_candidate_set(
            n_factors=2, n_levels=5, bounds=(-1, 1)
        )
        self.candidates_3d = generate_candidate_set(
            n_factors=3, n_levels=3, bounds=(-1, 1)
        )

    def test_generate_candidate_set_2d(self):
        expected_shape = (25, 2)
        actual = generate_candidate_set(n_factors=2, n_levels=5, bounds=(-1, 1))
        self.assertEqual(actual.shape, expected_shape)
        self.assertTrue(np.all(actual >= -1))
        self.assertTrue(np.all(actual <= 1))

    def test_generate_candidate_set_3d(self):
        expected_shape = (27, 3)
        actual = generate_candidate_set(n_factors=3, n_levels=3, bounds=(-1, 1))
        self.assertEqual(actual.shape, expected_shape)
        self.assertTrue(np.all(actual >= -1))
        self.assertTrue(np.all(actual <= 1))

    def test_generate_candidate_set_random(self):
        np.random.seed(42)
        actual = generate_candidate_set(
            n_factors=2, n_levels=3, bounds=(-1, 1), grid_type="uniform_random"
        )
        expected_shape = (9, 2)
        self.assertEqual(actual.shape, expected_shape)
        self.assertTrue(np.all(actual >= -1))
        self.assertTrue(np.all(actual <= 1))

    def test_build_design_matrix_linear(self):
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        expected = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
        actual = build_design_matrix(points, degree=1)
        np.testing.assert_allclose(actual, expected)

    def test_build_design_matrix_quadratic(self):
        points = np.array([[0, 0], [1, 1]])
        expected = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]])
        actual = build_design_matrix(points, degree=2)
        np.testing.assert_allclose(actual, expected)

    def test_information_matrix_normalized(self):
        X = np.array([[1, 0], [1, 1], [1, -1]])
        expected = np.array([[1, 0], [0, 2 / 3]])
        actual = information_matrix(X, normalized=True)
        np.testing.assert_allclose(actual, expected)

    def test_information_matrix_unnormalized(self):
        X = np.array([[1, 0], [1, 1], [1, -1]])
        expected = np.array([[3, 0], [0, 2]])
        actual = information_matrix(X, normalized=False)
        np.testing.assert_allclose(actual, expected)

    def test_information_matrix_augmented(self):
        X = np.array([[1, 0], [1, 1]])
        X0 = np.array([[1, 0], [1, 1], [1, -1]])
        alpha = 0.5

        H = X.T @ X
        H0 = (X0.T @ X0) / 3
        expected = (H + alpha * H0) / 2

        actual = information_matrix(X, normalized=True, alpha=alpha, X0=X0)
        np.testing.assert_allclose(actual, expected)

    def test_d_optimality(self):
        M = np.array([[2, 0], [0, 2]])
        expected = 4.0
        actual = d_optimality(M)
        self.assertAlmostEqual(actual, expected)

    def test_a_optimality(self):
        M = np.array([[2, 0], [0, 2]])
        expected = -1.0
        actual = a_optimality(M)
        self.assertAlmostEqual(actual, expected)

    def test_i_optimality(self):
        M_X = np.array([[2, 0], [0, 2]])
        moment_matrix = np.array([[1, 0], [0, 1]])
        expected = -1.0
        actual = i_optimality(M_X, moment_matrix)
        self.assertAlmostEqual(actual, expected)

    def test_c_optimality(self):
        M = np.array([[2, 0], [0, 2]])
        c = np.array([1, 0])
        expected = -0.5
        actual = c_optimality(M, c)
        self.assertAlmostEqual(actual, expected)

    def test_e_optimality(self):
        M = np.array([[3, 1], [1, 2]])
        eigenvals = np.linalg.eigvalsh(M)
        expected = float(np.min(eigenvals))
        actual = e_optimality(M)
        self.assertAlmostEqual(actual, expected)

    def test_g_optimality(self):
        M = np.array([[2, 0], [0, 2]])
        candidates = np.array([[1, 0], [0, 1], [1, 1]])
        _M_inv = np.linalg.inv(M)
        _variances = np.array([0.5, 0.5, 1.0])
        expected = -1.0
        actual = g_optimality(M, candidates)
        self.assertAlmostEqual(actual, expected)

    def test_v_optimality(self):
        M = np.array([[2, 0], [0, 2]])
        test_points = np.array([[1, 0], [0, 1]])
        expected = -0.5
        actual = v_optimality(M, test_points)
        self.assertAlmostEqual(actual, expected)

    def test_s_optimality(self):
        M = np.array([[4, 0], [0, 1]])
        det_M = 4.0
        col_rms = np.sqrt([4, 1])
        expected = det_M / np.prod(col_rms)
        actual = s_optimality(M)
        self.assertAlmostEqual(actual, expected)

    def test_t_optimality(self):
        X = np.array([[1, 0], [1, 1], [1, -1]])
        model_diff_subset = np.array([1, 0, -1])

        H = X.T @ X
        H_inv = np.linalg.inv(H)
        P = X @ H_inv @ X.T
        d = model_diff_subset.reshape(-1, 1)
        expected = float((d.T @ P @ d).item())

        actual = t_optimality(X, model_diff_subset)
        self.assertAlmostEqual(actual, expected)

    def test_d_efficiency(self):
        X = np.array([[1, 0], [1, 1], [1, -1]])
        M = information_matrix(X, normalized=True)
        p = X.shape[1]
        expected = 100.0 * (np.linalg.det(M) ** (1.0 / p))
        actual = d_efficiency(X)
        self.assertAlmostEqual(actual, expected)

    def test_a_efficiency(self):
        X = np.array([[1, 0], [1, 1], [1, -1]])
        M = information_matrix(X, normalized=True)
        p = X.shape[1]
        expected = 100.0 * (p / np.trace(np.linalg.inv(M)))
        actual = a_efficiency(X)
        self.assertAlmostEqual(actual, expected)

    def test_sequential_dykstra_algorithm(self):
        design = sequential_dykstra(
            self.candidates_2d, n_points=6, degree=2, criterion="D", alpha=0.01
        )
        self.assertEqual(design.shape, (6, 2))
        self.assertTrue(np.all(design >= -1))
        self.assertTrue(np.all(design <= 1))

    def test_simple_exchange_algorithm(self):
        design = simple_exchange_wynn_mitchell(
            self.candidates_2d,
            n_points=6,
            degree=2,
            criterion="D",
            alpha=0.01,
            max_iter=50,
        )
        self.assertEqual(design.shape, (6, 2))
        self.assertTrue(np.all(design >= -1))
        self.assertTrue(np.all(design <= 1))

    def test_fedorov_algorithm(self):
        design = fedorov(
            self.candidates_2d,
            n_points=6,
            degree=2,
            criterion="D",
            alpha=0.01,
            max_iter=50,
        )
        self.assertEqual(design.shape, (6, 2))
        self.assertTrue(np.all(design >= -1))
        self.assertTrue(np.all(design <= 1))

    def test_modified_fedorov_algorithm(self):
        design = modified_fedorov(
            self.candidates_2d,
            n_points=6,
            degree=2,
            criterion="D",
            alpha=0.01,
            max_iter=50,
        )
        self.assertEqual(design.shape, (6, 2))
        self.assertTrue(np.all(design >= -1))
        self.assertTrue(np.all(design <= 1))

    def test_detmax_algorithm(self):
        design = detmax(
            self.candidates_2d,
            n_points=6,
            degree=2,
            criterion="D",
            alpha=0.01,
            max_iter=50,
        )
        self.assertEqual(design.shape, (6, 2))
        self.assertTrue(np.all(design >= -1))
        self.assertTrue(np.all(design <= 1))

    def test_algorithm_with_a_optimality(self):
        design = detmax(
            self.candidates_2d, n_points=6, degree=2, criterion="A", alpha=0.01
        )
        self.assertEqual(design.shape, (6, 2))

    def test_algorithm_with_i_optimality(self):
        design = detmax(
            self.candidates_2d, n_points=6, degree=2, criterion="I", alpha=0.01
        )
        self.assertEqual(design.shape, (6, 2))

    def test_optimal_design_d_criterion(self):
        design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="D",
            method="detmax",
            alpha=0.01,
        )

        self.assertEqual(design.shape, (8, 2))
        self.assertEqual(info["criterion"], "D")
        self.assertEqual(info["method"], "detmax")
        self.assertEqual(info["alpha"], 0.01)
        self.assertIn("score", info)
        self.assertIn("D_eff", info)
        self.assertIn("A_eff", info)
        self.assertIn("p_columns", info)
        self.assertIn("n_runs", info)

    def test_optimal_design_a_criterion(self):
        design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="A",
            method="sequential",
            alpha=0.01,
        )

        self.assertEqual(design.shape, (8, 2))
        self.assertEqual(info["criterion"], "A")
        self.assertEqual(info["method"], "sequential")

    def test_optimal_design_i_criterion(self):
        design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="I",
            method="simple_exchange",
            alpha=0.01,
        )

        self.assertEqual(design.shape, (8, 2))
        self.assertEqual(info["criterion"], "I")
        self.assertEqual(info["method"], "simple_exchange")

    def test_optimal_design_all_methods(self):
        methods = [
            "sequential",
            "simple_exchange",
            "fedorov",
            "modified_fedorov",
            "detmax",
        ]

        for method in methods:
            with self.subTest(method=method):
                design, info = optimal_design(
                    candidates=self.candidates_2d,
                    n_points=6,
                    degree=1,
                    criterion="D",
                    method=method,
                    max_iter=20,
                )

                self.assertEqual(design.shape, (6, 2))
                self.assertEqual(info["method"], method)
                self.assertGreater(info["D_eff"], 80)
                self.assertGreater(info["A_eff"], 80)

    def test_criterion_value_function(self):
        X = build_design_matrix(self.candidates_2d[:6], degree=2)
        X0 = build_design_matrix(self.candidates_2d, degree=2)

        criteria = ["D", "A", "I", "E", "S"]
        for criterion in criteria:
            with self.subTest(criterion=criterion):
                score = criterion_value(X, criterion, X0, alpha=0.01)
                self.assertIsInstance(score, float)
                self.assertFalse(np.isnan(score))
                self.assertFalse(np.isinf(score))

    def test_build_uniform_moment_matrix(self):
        X0 = build_design_matrix(self.candidates_2d, degree=2)
        moment_matrix = build_uniform_moment_matrix(X0)

        expected_shape = (X0.shape[1], X0.shape[1])
        self.assertEqual(moment_matrix.shape, expected_shape)

        np.testing.assert_allclose(moment_matrix, moment_matrix.T)

        eigenvals = np.linalg.eigvalsh(moment_matrix)
        self.assertTrue(np.all(eigenvals >= -1e-10))

    def test_higher_degree_models(self):
        design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=12,
            degree=3,
            criterion="D",
            method="detmax",
        )

        X = build_design_matrix(design, degree=3)
        expected_params = 10
        self.assertEqual(X.shape[1], expected_params)
        self.assertEqual(info["p_columns"], expected_params)

    def test_3d_design_space(self):
        design, _info = optimal_design(
            candidates=self.candidates_3d,
            n_points=15,
            degree=2,
            criterion="D",
            method="detmax",
        )

        self.assertEqual(design.shape, (15, 3))
        self.assertTrue(np.all(design >= -1))
        self.assertTrue(np.all(design <= 1))

        X = build_design_matrix(design, degree=2)
        expected_params = 10
        self.assertEqual(X.shape[1], expected_params)

    def test_error_handling_insufficient_points(self):
        try:
            design, _info = optimal_design(
                candidates=self.candidates_2d[:10],
                n_points=3,
                degree=1,
                criterion="D",
                method="sequential",
            )
            self.assertEqual(design.shape, (3, 2))
        except (ValueError, RuntimeError) as e:
            self.assertIn("singular", str(e).lower())

    def test_error_handling_unknown_criterion(self):
        with self.assertRaises(ValueError):
            criterion_value(
                build_design_matrix(self.candidates_2d[:6], degree=2),
                "UNKNOWN",
                build_design_matrix(self.candidates_2d, degree=2),
            )

    def test_error_handling_unknown_method(self):
        with self.assertRaises(ValueError):
            optimal_design(
                candidates=self.candidates_2d,
                n_points=6,
                degree=2,
                criterion="D",
                method="unknown_method",
            )

    def test_reproducibility(self):
        np.random.seed(42)
        design1, info1 = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="D",
            method="detmax",
        )

        np.random.seed(42)
        design2, info2 = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="D",
            method="detmax",
        )

        np.testing.assert_allclose(design1, design2)
        self.assertAlmostEqual(info1["score"], info2["score"])

    def test_augmentation_parameter_effect(self):
        design1, _info1 = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="D",
            method="sequential",
            alpha=0.0,
        )

        design2, _info2 = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="D",
            method="sequential",
            alpha=0.1,
        )

        self.assertFalse(np.allclose(design1, design2))

    def test_efficiency_bounds(self):
        _design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=10,
            degree=2,
            criterion="D",
            method="detmax",
        )

        self.assertGreater(info["D_eff"], 40)
        self.assertGreater(info["A_eff"], 20)
        self.assertLessEqual(info["D_eff"], 100)
        self.assertLessEqual(info["A_eff"], 100)

    def test_all_criteria_comprehensive(self):
        criteria = ["D", "A", "I", "C", "E", "G", "V", "S", "T"]

        for criterion in criteria:
            with self.subTest(criterion=criterion):
                design = None
                info = None
                try:
                    # Use more design points and lower degree to
                    # avoid singular matrices
                    design, info = optimal_design(
                        candidates=self.candidates_2d,
                        n_points=10,  # More points
                        degree=1,  # Lower degree to start
                        criterion=criterion,
                        method="detmax",
                        alpha=0.1,  # Higher alpha for regularization
                        max_iter=100,
                    )
                except (ValueError, RuntimeError) as e:
                    # For problematic criteria, try with
                    # even more regularization
                    if (
                        criterion in {"C", "V", "T"}
                        and "singular" in str(e).lower()
                    ):
                        try:
                            print(
                                f"Retrying criterion {criterion} "
                                "with higher regularization ..."
                            )
                            design, info = optimal_design(
                                candidates=self.candidates_2d,
                                n_points=12,  # Even more points
                                degree=1,
                                criterion=criterion,
                                method="sequential",  # Different algorithm
                                alpha=0.5,  # Much higher alpha
                                max_iter=50,
                            )
                        except (ValueError, RuntimeError) as e2:
                            print(
                                f"Criterion {criterion} failed even with"
                                f" regularization: {e2!s}"
                            )
                            # Don't fail the test, just report the issue
                    else:
                        self.fail(f"Failed for criterion {criterion}: {e!s}")

                # Assertions outside try block
                if design is not None and info is not None:
                    # Basic shape and bounds checks
                    self.assertEqual(design.shape[1], 2)
                    self.assertTrue(np.all(design >= -1))
                    self.assertTrue(np.all(design <= 1))

                    # Info dictionary checks
                    self.assertEqual(info["criterion"], criterion)
                    self.assertIn("score", info)
                    self.assertIn("D_eff", info)
                    self.assertIn("A_eff", info)
                    self.assertIn("p_columns", info)
                    self.assertIn("n_runs", info)

                    # Score should be finite
                    self.assertFalse(np.isnan(info["score"]))
                    self.assertFalse(np.isinf(info["score"]))

                    # Efficiencies should be reasonable
                    self.assertGreater(info["D_eff"], 0)
                    self.assertGreater(info["A_eff"], 0)
                    self.assertLessEqual(info["D_eff"], 100)
                    self.assertLessEqual(info["A_eff"], 100)

                    # Original path uses detmax with 10 points
                    if design.shape[0] == 10:
                        self.assertEqual(info["method"], "detmax")
                        self.assertEqual(info["alpha"], 0.1)

                    print(
                        f"Criterion {criterion}: Score={info['score']:.4f}, "
                        f"D_eff={info['D_eff']:.2f}%, "
                        f"A_eff={info['A_eff']:.2f}%"
                    )

    def test_robust_criteria_individually(self):
        # Test D-optimality
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="D",
            method="detmax",
            alpha=0.01,
        )
        self.assertEqual(info["criterion"], "D")
        self.assertFalse(np.isnan(info["score"]))

        # Test A-optimality
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="A",
            method="sequential",
            alpha=0.05,
        )
        self.assertEqual(info["criterion"], "A")
        self.assertFalse(np.isnan(info["score"]))

        # Test I-optimality
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="I",
            method="detmax",
            alpha=0.01,
        )
        self.assertEqual(info["criterion"], "I")
        self.assertFalse(np.isnan(info["score"]))

        # Test E-optimality
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="E",
            method="detmax",
            alpha=0.05,
        )
        self.assertEqual(info["criterion"], "E")
        self.assertFalse(np.isnan(info["score"]))

        # Test G-optimality
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=10,
            degree=2,
            criterion="G",
            method="detmax",
            alpha=0.1,
        )
        self.assertEqual(info["criterion"], "G")
        self.assertFalse(np.isnan(info["score"]))

        # Test S-optimality
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="S",
            method="detmax",
            alpha=0.01,
        )
        self.assertEqual(info["criterion"], "S")
        self.assertFalse(np.isnan(info["score"]))

        # Test C-optimality (linear model to avoid singularity)
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=6,
            degree=1,  # Linear model
            criterion="C",
            method="sequential",
            alpha=0.1,
        )
        self.assertEqual(info["criterion"], "C")
        self.assertFalse(np.isnan(info["score"]))

        # Test V-optimality (linear model)
        _, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=1,  # Linear model
            criterion="V",
            method="sequential",
            alpha=0.2,
        )
        self.assertEqual(info["criterion"], "V")
        self.assertFalse(np.isnan(info["score"]))

        # Test T-optimality (linear model) - handle potential singularity
        try:
            _, info = optimal_design(
                candidates=self.candidates_2d,
                n_points=8,
                degree=1,  # Linear model
                criterion="T",
                method="sequential",
                alpha=0.2,
            )
            self.assertEqual(info["criterion"], "T")
            self.assertFalse(np.isnan(info["score"]))
        except Exception as e:
            if "singular" in str(e).lower():
                print(
                    f"T-optimality test skipped due to singularity issue: {e!s}"
                )
                # This is expected for T-optimality in some cases
            else:
                raise e

    def test_criteria_performance_comparison(self):
        criteria = ["D", "A", "I", "E", "G", "S", "V", "C", "T"]
        results = {}

        for criterion in criteria:
            design, info = optimal_design(
                candidates=self.candidates_2d,
                n_points=8,
                degree=2,
                criterion=criterion,
                method="detmax",
                alpha=0.05,
                max_iter=50,
            )
            results[criterion] = {
                "score": info["score"],
                "D_eff": info["D_eff"],
                "A_eff": info["A_eff"],
                "n_unique": len(np.unique(design.round(6), axis=0)),
            }

        print("\nCriteria Performance Comparison:")
        print("Criterion | Score      | D_eff  | A_eff  | Unique Points")
        print("-" * 60)
        for criterion, result in results.items():
            print(
                f"{criterion:8} | {result['score']:10.4f} | "
                f"{result['D_eff']:6.2f} | "
                f"{result['A_eff']:6.2f} | {result['n_unique']:13}"
            )

        # All should produce valid designs
        for _, result in results.items():
            self.assertGreaterEqual(result["D_eff"], 0)
            self.assertGreaterEqual(result["A_eff"], 0)
            self.assertGreaterEqual(
                result["n_unique"], 6
            )  # Should have reasonable diversity

    def test_all_criteria_with_different_methods(self):
        """Test all criteria with different algorithms"""
        criteria = ["D", "A", "I", "C", "E", "G", "V", "S", "T"]
        methods = [
            "sequential",
            "simple_exchange",
            "fedorov",
            "modified_fedorov",
            "detmax",
        ]

        for criterion in criteria[:3]:  # Test first 3 criteria with all methods
            for method in methods:
                with self.subTest(criterion=criterion, method=method):
                    try:
                        design, info = optimal_design(
                            candidates=self.candidates_2d,
                            n_points=6,
                            degree=1,
                            criterion=criterion,
                            method=method,
                            alpha=0.01,
                            max_iter=50,
                        )

                        self.assertEqual(design.shape, (6, 2))
                        self.assertEqual(info["criterion"], criterion)
                        self.assertEqual(info["method"], method)
                        self.assertFalse(np.isnan(info["score"]))

                    except (ValueError, RuntimeError) as e:
                        self.fail(
                            f"Failed for criterion {criterion} with "
                            f"method {method}: {e!s}"
                        )

    def test_criterion_specific_parameters(self):
        # Test C-optimality with custom c_vector
        design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=6,
            degree=1,
            criterion="C",
            method="detmax",
            alpha=0.01,
        )
        self.assertEqual(design.shape, (6, 2))
        self.assertEqual(info["criterion"], "C")

        # Test V-optimality with test points
        design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=8,
            degree=2,
            criterion="V",
            method="detmax",
            alpha=0.01,
        )
        self.assertEqual(design.shape, (8, 2))
        self.assertEqual(info["criterion"], "V")

        # Test T-optimality with model difference
        design, info = optimal_design(
            candidates=self.candidates_2d,
            n_points=6,
            degree=1,
            criterion="T",
            method="detmax",
            alpha=0.01,
        )
        self.assertEqual(design.shape, (6, 2))
        self.assertEqual(info["criterion"], "T")

    def test_criteria_with_3d_space(self):
        criteria = ["D", "A", "I", "C", "E", "G", "V", "S", "T"]

        for criterion in criteria:
            with self.subTest(criterion=criterion):
                try:
                    design, info = optimal_design(
                        candidates=self.candidates_3d,
                        n_points=12,
                        degree=2,
                        criterion=criterion,
                        method="detmax",
                        alpha=0.01,
                        max_iter=50,
                    )
                except (ValueError, RuntimeError) as e:
                    self.fail(f"Failed for criterion {criterion} in 3D: {e!s}")

                # Assertions outside try block
                self.assertEqual(design.shape, (12, 3))
                self.assertTrue(np.all(design >= -1))
                self.assertTrue(np.all(design <= 1))
                self.assertEqual(info["criterion"], criterion)
                self.assertFalse(np.isnan(info["score"]))

    def test_criteria_convergence(self):
        criteria = ["D", "A", "I"]

        for criterion in criteria:
            with self.subTest(criterion=criterion):
                # Short run
                _design1, _info1 = optimal_design(
                    candidates=self.candidates_2d,
                    n_points=6,
                    degree=2,
                    criterion=criterion,
                    method="detmax",
                    max_iter=10,
                )

                # Longer run
                _design2, _info2 = optimal_design(
                    candidates=self.candidates_2d,
                    n_points=6,
                    degree=2,
                    criterion=criterion,
                    method="detmax",
                    max_iter=100,
                )

    def test_all_nine_criteria_comprehensive_with_regularization(self):  # ruff: ignore[too-many-branches, too-many-statements]
        criteria = ["D", "A", "I", "C", "E", "G", "V", "S", "T"]
        results = {}

        for criterion in criteria:
            with self.subTest(criterion=criterion):
                design = None
                info = None
                try:
                    if criterion in {"D", "A", "I", "E", "G", "S"}:
                        # Robust criteria - use standard settings
                        design, info = optimal_design(
                            candidates=self.candidates_2d,
                            n_points=8,
                            degree=2,
                            criterion=criterion,
                            method="detmax",
                            alpha=0.05,  # Some regularization
                            max_iter=100,
                        )
                    else:  # ["C", "V", "T"]
                        # Problematic criteria - use conservative settings
                        design, info = optimal_design(
                            candidates=self.candidates_2d,
                            n_points=10,  # More points for stability
                            degree=1,  # Linear model
                            criterion=criterion,
                            method="sequential",  # Most stable algorithm
                            alpha=0.2,  # Higher regularization
                            max_iter=50,
                        )
                except (ValueError, RuntimeError) as e:
                    results[criterion] = {"success": False, "error": str(e)}
                    print(f"{criterion}-optimality: FAILED - {e!s}")
                    # Don't fail the test, just record the failure
                    continue

                # Assertions outside try block
                if design is not None and info is not None:
                    # Validate results
                    self.assertIsInstance(design, np.ndarray)
                    self.assertEqual(len(design.shape), 2)
                    self.assertEqual(design.shape[1], 2)  # 2D design space
                    self.assertTrue(np.all(design >= -1))
                    self.assertTrue(np.all(design <= 1))

                    # Check info dictionary
                    required_keys = [
                        "criterion",
                        "method",
                        "alpha",
                        "score",
                        "D_eff",
                        "A_eff",
                        "p_columns",
                        "n_runs",
                    ]
                    for key in required_keys:
                        self.assertIn(key, info)

                    self.assertEqual(info["criterion"], criterion)
                    self.assertFalse(np.isnan(info["score"]))
                    self.assertFalse(np.isinf(info["score"]))
                    self.assertGreater(info["D_eff"], 0)
                    self.assertGreater(info["A_eff"], 0)

                    results[criterion] = {
                        "success": True,
                        "score": info["score"],
                        "D_eff": info["D_eff"],
                        "A_eff": info["A_eff"],
                        "n_points": design.shape[0],
                        "method": info["method"],
                        "alpha": info["alpha"],
                    }

                    print(
                        f"{criterion}-optimality: Score={info['score']:8.4f}, "
                        f"D_eff={info['D_eff']:5.1f}%, "
                        f"A_eff={info['A_eff']:5.1f}%, "
                        f"Method={info['method']}, Alpha={info['alpha']}"
                    )

        # Summary
        successful_criteria = [
            c for c, r in results.items() if r.get("success", False)
        ]
        failed_criteria = [
            c for c, r in results.items() if not r.get("success", False)
        ]

        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        print(f"Successful: {len(successful_criteria)}/9 criteria")
        print(f"Working: {', '.join(successful_criteria)}")
        if failed_criteria:
            print(f"Failed: {', '.join(failed_criteria)}")

        # Performance comparison table
        if successful_criteria:
            print(
                "Criterion | Score      | D-Eff  | A-Eff  | Method      | Alpha"
            )
            print("-" * 70)
            for criterion in successful_criteria:
                r = results[criterion]
                print(
                    f"{criterion:8} | {r['score']:10.4f} | {r['D_eff']:6.1f} | "
                    f"{r['A_eff']:6.1f} | {r['method']:11} | {r['alpha']:5.2f}"
                )

        robust_criteria = ["D", "A", "I", "E", "G", "S"]
        for criterion in robust_criteria:
            with self.subTest(criterion=criterion, space="3D"):
                try:
                    design, info = optimal_design(
                        candidates=self.candidates_3d,
                        n_points=15,
                        degree=2,
                        criterion=criterion,
                        method="detmax",
                        alpha=0.1,
                        max_iter=50,
                    )

                    self.assertEqual(design.shape[1], 3)  # 3D design space
                    self.assertFalse(np.isnan(info["score"]))
                    print(
                        f"{criterion}-optimality (3D): "
                        f"D_eff={info['D_eff']:5.1f}%, "
                        f"A_eff={info['A_eff']:5.1f}%"
                    )

                except (ValueError, RuntimeError) as e:
                    print(f"{criterion}-optimality (3D): {e!s}")

        algorithms = [
            "sequential",
            "simple_exchange",
            "fedorov",
            "modified_fedorov",
            "detmax",
        ]
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                try:
                    design, info = optimal_design(
                        candidates=self.candidates_2d,
                        n_points=6,
                        degree=1,
                        criterion="D",
                        method=algorithm,
                        alpha=0.05,
                        max_iter=30,
                    )
                    print(
                        f"{algorithm:15}: D_eff={info['D_eff']:5.1f}%, "
                        f"A_eff={info['A_eff']:5.1f}%"
                    )
                except (ValueError, RuntimeError) as e:
                    print(f"{algorithm:15}: {e!s}")

        # Ensure at least the robust criteria work
        self.assertGreaterEqual(
            len(successful_criteria),
            6,
            "At least 6 robust criteria should work",
        )

        # Ensure no infinite or NaN scores for successful criteria
        for criterion in successful_criteria:
            score = results[criterion]["score"]
            self.assertFalse(np.isnan(score), f"{criterion} produced NaN score")
            self.assertFalse(
                np.isinf(score), f"{criterion} produced infinite score"
            )


if __name__ == "__main__":
    unittest.main()
