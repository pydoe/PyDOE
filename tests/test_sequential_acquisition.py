import unittest

import numpy as np

from pydoe.sequential import (
    expected_improvement,
    probability_of_improvement,
    upper_confidence_bound,
)


class TestExpectedImprovement(unittest.TestCase):
    def test_shape(self):
        mean = np.array([0.5, 1.5, 2.5])
        std = np.array([0.1, 0.2, 0.3])
        ei = expected_improvement(mean, std, best_f=1.0)
        self.assertEqual(ei.shape, mean.shape)

    def test_higher_mean_higher_ei_maximize(self):
        mean = np.array([0.5, 1.5])
        std = np.array([0.2, 0.2])
        ei = expected_improvement(mean, std, best_f=1.0)
        self.assertGreater(ei[1], ei[0])

    def test_higher_mean_lower_ei_minimize(self):
        mean = np.array([0.5, 1.5])
        std = np.array([0.2, 0.2])
        ei = expected_improvement(mean, std, best_f=1.0, maximize=False)
        self.assertGreater(ei[0], ei[1])

    def test_std_zero_positive_improvement(self):
        mean = np.array([2.0])
        std = np.array([0.0])
        ei = expected_improvement(mean, std, best_f=1.0)
        self.assertAlmostEqual(ei[0], 1.0)

    def test_std_zero_no_improvement(self):
        mean = np.array([0.5])
        std = np.array([0.0])
        ei = expected_improvement(mean, std, best_f=1.0)
        self.assertAlmostEqual(ei[0], 0.0)

    def test_non_negative(self):
        mean = np.array([-1.0, 0.0, 1.0])
        std = np.array([0.1, 0.5, 1.0])
        ei = expected_improvement(mean, std, best_f=1.0)
        self.assertTrue(np.all(ei >= 0))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            expected_improvement(
                np.array([0.0, 1.0]), np.array([0.1]), best_f=0.0
            )

    def test_negative_std_raises(self):
        with self.assertRaises(ValueError):
            expected_improvement(np.array([0.0]), np.array([-0.1]), best_f=0.0)


class TestProbabilityOfImprovement(unittest.TestCase):
    def test_shape(self):
        mean = np.array([0.5, 1.5, 2.5])
        std = np.array([0.1, 0.2, 0.3])
        pi = probability_of_improvement(mean, std, best_f=1.0)
        self.assertEqual(pi.shape, mean.shape)

    def test_higher_mean_higher_pi_maximize(self):
        mean = np.array([0.5, 1.5])
        std = np.array([0.2, 0.2])
        pi = probability_of_improvement(mean, std, best_f=1.0)
        self.assertGreater(pi[1], pi[0])

    def test_higher_mean_lower_pi_minimize(self):
        mean = np.array([0.5, 1.5])
        std = np.array([0.2, 0.2])
        pi = probability_of_improvement(mean, std, best_f=1.0, maximize=False)
        self.assertGreater(pi[0], pi[1])

    def test_std_zero_positive_improvement(self):
        mean = np.array([2.0])
        std = np.array([0.0])
        pi = probability_of_improvement(mean, std, best_f=1.0)
        self.assertAlmostEqual(pi[0], 1.0)

    def test_std_zero_no_improvement(self):
        mean = np.array([0.5])
        std = np.array([0.0])
        pi = probability_of_improvement(mean, std, best_f=1.0)
        self.assertAlmostEqual(pi[0], 0.0)

    def test_bounded_in_unit_interval(self):
        mean = np.array([-1.0, 0.0, 1.0, 5.0])
        std = np.array([0.1, 0.5, 1.0, 2.0])
        pi = probability_of_improvement(mean, std, best_f=1.0)
        self.assertTrue(np.all(pi >= 0))
        self.assertTrue(np.all(pi <= 1))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            probability_of_improvement(
                np.array([0.0, 1.0]), np.array([0.1]), best_f=0.0
            )

    def test_negative_std_raises(self):
        with self.assertRaises(ValueError):
            probability_of_improvement(
                np.array([0.0]), np.array([-0.1]), best_f=0.0
            )


class TestUpperConfidenceBound(unittest.TestCase):
    def test_shape(self):
        mean = np.array([0.5, 1.5, 2.5])
        std = np.array([0.1, 0.2, 0.3])
        ucb = upper_confidence_bound(mean, std)
        self.assertEqual(ucb.shape, mean.shape)

    def test_higher_mean_higher_ucb_maximize(self):
        mean = np.array([0.5, 1.5])
        std = np.array([0.2, 0.2])
        ucb = upper_confidence_bound(mean, std)
        self.assertGreater(ucb[1], ucb[0])

    def test_higher_mean_lower_ucb_minimize(self):
        mean = np.array([0.5, 1.5])
        std = np.array([0.2, 0.2])
        ucb = upper_confidence_bound(mean, std, maximize=False)
        self.assertGreater(ucb[0], ucb[1])

    def test_higher_std_higher_ucb(self):
        mean = np.array([1.0, 1.0])
        std = np.array([0.1, 1.0])
        ucb = upper_confidence_bound(mean, std)
        self.assertGreater(ucb[1], ucb[0])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([0.0, 1.0]), np.array([0.1]))

    def test_negative_std_raises(self):
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([0.0]), np.array([-0.1]))

    def test_invalid_kappa_raises(self):
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([0.0]), np.array([0.1]), kappa=0.0)

    def test_negative_kappa_raises(self):
        with self.assertRaises(ValueError):
            upper_confidence_bound(np.array([0.0]), np.array([0.1]), kappa=-1.0)


if __name__ == "__main__":
    unittest.main()
