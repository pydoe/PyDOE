import unittest

import numpy as np

from pydoe.sequential import GaussianProcessRegressor, sequential_design


class TestGaussianProcessRegressor(unittest.TestCase):
    def test_fit_predict_close_to_training_values(self):
        X = np.linspace(0, 2 * np.pi, 8).reshape(-1, 1)
        y = np.sin(X).ravel()
        gp = GaussianProcessRegressor(length_scale=1.0).fit(X, y)
        mean = gp.predict(X)
        np.testing.assert_allclose(mean, y, atol=1e-3)

    def test_return_std_non_negative(self):
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 1.0, 0.0])
        gp = GaussianProcessRegressor(length_scale=0.5).fit(X, y)
        _, std = gp.predict(np.array([[0.0], [0.25], [2.0]]), return_std=True)
        self.assertTrue(np.all(std >= 0))

    def test_predict_before_fit_raises(self):
        gp = GaussianProcessRegressor()
        with self.assertRaises(ValueError):
            gp.predict(np.array([[0.0]]))

    def test_invalid_length_scale_raises(self):
        with self.assertRaises(ValueError):
            GaussianProcessRegressor(length_scale=0.0)
        with self.assertRaises(ValueError):
            GaussianProcessRegressor(length_scale=-1.0)

    def test_invalid_noise_raises(self):
        with self.assertRaises(ValueError):
            GaussianProcessRegressor(noise=-1e-6)

    def test_mismatched_lengths_raise(self):
        gp = GaussianProcessRegressor()
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0])
        with self.assertRaises(ValueError):
            gp.fit(X, y)

    def test_empty_X_raises(self):
        gp = GaussianProcessRegressor()
        X = np.empty((0, 1))
        y = np.empty((0,))
        with self.assertRaises(ValueError):
            gp.fit(X, y)


class TestSequentialDesign(unittest.TestCase):
    def neg_quadratic(self, x):
        return -float((x[0] - 0.3) ** 2)

    def test_output_shapes(self):
        bounds = np.array([[0.0, 1.0]])
        X, y = sequential_design(
            self.neg_quadratic, bounds, n_initial=4, n_iter=5, seed=0
        )
        self.assertEqual(X.shape, (9, 1))
        self.assertEqual(y.shape, (9,))

    def test_invalid_bounds_shape_raises(self):
        bounds = np.array([0.0, 1.0])
        with self.assertRaises(ValueError):
            sequential_design(self.neg_quadratic, bounds, n_initial=2, n_iter=1)

    def test_invalid_bounds_order_raises(self):
        bounds = np.array([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            sequential_design(self.neg_quadratic, bounds, n_initial=2, n_iter=1)

    def test_n_initial_too_small_raises(self):
        bounds = np.array([[0.0, 1.0]])
        with self.assertRaises(ValueError):
            sequential_design(self.neg_quadratic, bounds, n_initial=0, n_iter=1)

    def test_n_iter_negative_raises(self):
        bounds = np.array([[0.0, 1.0]])
        with self.assertRaises(ValueError):
            sequential_design(
                self.neg_quadratic, bounds, n_initial=2, n_iter=-1
            )

    def test_invalid_acquisition_raises(self):
        bounds = np.array([[0.0, 1.0]])
        with self.assertRaises(ValueError):
            sequential_design(
                self.neg_quadratic,
                bounds,
                n_initial=2,
                n_iter=1,
                acquisition="bogus",
            )

    def test_reproducibility(self):
        bounds = np.array([[0.0, 1.0]])
        X1, y1 = sequential_design(
            self.neg_quadratic, bounds, n_initial=4, n_iter=3, seed=0
        )
        X2, y2 = sequential_design(
            self.neg_quadratic, bounds, n_initial=4, n_iter=3, seed=0
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_sequential_improves_over_initial(self):
        bounds = np.array([[0.0, 1.0]])
        _, y = sequential_design(
            self.neg_quadratic, bounds, n_initial=4, n_iter=5, seed=0
        )
        self.assertGreaterEqual(y.max(), y[:4].max())

    def test_pi_and_ucb_acquisitions_run(self):
        bounds = np.array([[0.0, 1.0]])
        for acq in ("pi", "ucb"):
            X_acq, y_acq = sequential_design(
                self.neg_quadratic,
                bounds,
                n_initial=3,
                n_iter=2,
                acquisition=acq,
                seed=1,
            )
            self.assertEqual(X_acq.shape, (5, 1))
            self.assertEqual(y_acq.shape, (5,))

    def test_minimize(self):
        bounds = np.array([[0.0, 1.0]])

        def quadratic(x):
            return float((x[0] - 0.3) ** 2)

        _, y = sequential_design(
            quadratic, bounds, n_initial=4, n_iter=5, maximize=False, seed=0
        )
        self.assertLessEqual(y.min(), y[:4].min())

    def test_multidimensional(self):
        bounds = np.array([[0.0, 1.0], [-1.0, 1.0]])

        def objective(x):
            return -float((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2)

        X, y = sequential_design(
            objective, bounds, n_initial=5, n_iter=2, seed=0
        )
        self.assertEqual(X.shape, (7, 2))
        self.assertEqual(y.shape, (7,))


if __name__ == "__main__":
    unittest.main()
