import unittest

import numpy as np

from pydoe import iman_conover


class TestImanConover(unittest.TestCase):
    def test_shape_preserved(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(500, 3))
        target = np.eye(3)
        reordered = iman_conover(data, target, seed=1)
        self.assertEqual(reordered.shape, data.shape)

    def test_marginals_unchanged(self):
        rng = np.random.default_rng(2)
        data = rng.exponential(size=(500, 2))
        target = np.array([[1.0, 0.5], [0.5, 1.0]])
        reordered = iman_conover(data, target, seed=2)
        for col in range(2):
            np.testing.assert_allclose(
                np.sort(reordered[:, col]), np.sort(data[:, col])
            )

    def test_induces_target_correlation(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(2000, 2))
        target = np.array([[1.0, 0.7], [0.7, 1.0]])
        reordered = iman_conover(data, target, seed=3)

        rank0 = reordered[:, 0].argsort().argsort()
        rank1 = reordered[:, 1].argsort().argsort()
        rank_corr = np.corrcoef(rank0, rank1)[0, 1]

        self.assertAlmostEqual(rank_corr, 0.7, delta=0.05)

    def test_negative_correlation(self):
        rng = np.random.default_rng(4)
        data = rng.uniform(size=(2000, 2))
        target = np.array([[1.0, -0.6], [-0.6, 1.0]])
        reordered = iman_conover(data, target, seed=4)

        rank0 = reordered[:, 0].argsort().argsort()
        rank1 = reordered[:, 1].argsort().argsort()
        rank_corr = np.corrcoef(rank0, rank1)[0, 1]

        self.assertAlmostEqual(rank_corr, -0.6, delta=0.05)

    def test_non_2d_data_raises(self):
        with self.assertRaises(ValueError):
            iman_conover(np.zeros(10), np.eye(2))

    def test_correlation_shape_mismatch_raises(self):
        data = np.zeros((10, 3))
        with self.assertRaises(ValueError):
            iman_conover(data, np.eye(2))

    def test_non_positive_definite_correlation_raises(self):
        data = np.zeros((10, 2))
        bad_corr = np.array([[1.0, 2.0], [2.0, 1.0]])
        with self.assertRaises(ValueError):
            iman_conover(data, bad_corr)
