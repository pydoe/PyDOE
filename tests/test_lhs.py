import unittest

import numpy as np

from pydoe import lhs


class TestLhs(unittest.TestCase):
    def test_lhs_vanilla_deprecated(self):
        expected = [
            [0.12484671, 0.95539205, 0.24399798],
            [0.53288616, 0.38533955, 0.86703834],
            [0.68602787, 0.31690477, 0.38533151],
        ]
        with self.assertWarns(DeprecationWarning):
            actual = lhs(3, random_state=42)
            np.testing.assert_allclose(actual, expected)

    def test_lhs_deprecated(self):
        expected = [
            [0.06242335, 0.19266575, 0.88202411, 0.89439364],
            [0.19266977, 0.53538985, 0.53030416, 0.49498498],
            [0.71737371, 0.75412607, 0.17634727, 0.71520486],
            [0.63874044, 0.85658231, 0.33676408, 0.31102936],
            [0.43351917, 0.45134543, 0.12199899, 0.53056742],
            [0.93530882, 0.15845238, 0.7386575, 0.09977641],
        ]
        with self.assertWarns(DeprecationWarning):
            actual = lhs(4, samples=6, random_state=42)
            np.testing.assert_allclose(actual, expected)

    def test_lhs_center_deprecated(self):
        expected = [[0.1, 0.9], [0.5, 0.5], [0.7, 0.1], [0.3, 0.7], [0.9, 0.3]]
        with self.assertWarns(DeprecationWarning):
            actual = lhs(2, samples=5, criterion="center", random_state=42)
            np.testing.assert_allclose(actual, expected)

    def test_lhs_maximin_deprecated(self):
        expected = [
            [0.69754389, 0.2997106, 0.96250964],
            [0.10585037, 0.09872038, 0.73157522],
            [0.25351996, 0.65148999, 0.07337204],
            [0.91276926, 0.97873992, 0.42783549],
        ]
        with self.assertWarns(DeprecationWarning):
            actual = lhs(3, samples=4, criterion="maximin", random_state=42)
            np.testing.assert_allclose(actual, expected)

    def test_lhs_correlation_deprecated(self):
        expected = [
            [0.4846803, 0.74226839, 0.23305339, 0.97000772],
            [0.98526018, 0.11735023, 0.75803511, 0.20312728],
            [0.10793843, 0.2592547, 0.98299194, 0.72119199],
            [0.25519984, 0.4789763, 0.19305106, 0.12140685],
            [0.63976848, 0.93021541, 0.45869763, 0.40281596],
        ]
        with self.assertWarns(DeprecationWarning):
            actual = lhs(
                4,
                samples=5,
                criterion="correlation",
                iterations=10,
                random_state=42,
            )
            np.testing.assert_allclose(actual, expected)

    def test_lhs_lhsmu_deprecated(self):
        expected = [
            [0.78593953, 0.00628584, 0.12728208, 0.96073442],
            [0.0628712, 0.38151329, 0.83731401, 0.76162408],
            [0.24985845, 0.9785118, 0.44575963, 0.28207658],
            [0.90786845, 0.41539598, 0.72668075, 0.10171414],
            [0.45795029, 0.77429212, 0.35111023, 0.43224426],
        ]
        with self.assertWarns(DeprecationWarning):
            actual = lhs(4, samples=5, criterion="lhsmu", random_state=42)
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_lhs_vanilla(self):
        expected = [
            [0.25798535, 0.14629281, 0.65854078],
            [0.56578934, 0.9286881, 0.28619931],
            [0.9203799, 0.36472578, 0.70937121],
        ]
        actual = lhs(3, seed=42)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_lhs(self):
        expected = [
            [0.60731085, 0.32927039, 0.8046052, 0.98218685],
            [0.35468561, 0.67730288, 0.14309965, 0.77194407],
            [0.18236289, 0.89242099, 0.29352328, 0.53787312],
            [0.75909746, 0.63712694, 0.395133, 0.48779416],
            [0.95968129, 0.40839766, 0.99511634, 0.116228],
            [0.12899267, 0.07314641, 0.57390237, 0.29767738],
        ]
        actual = lhs(4, samples=6, seed=42)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_lhs_center(self):
        expected = [[0.9, 0.1], [0.1, 0.7], [0.5, 0.9], [0.7, 0.5], [0.3, 0.3]]
        actual = lhs(2, samples=5, criterion="center", seed=42)
        np.testing.assert_allclose(actual, expected)

    def test_lhs_maximin(self):
        expected = [
            [0.91547913, 0.46335077, 0.05364617],
            [0.51457569, 0.10917935, 0.94597455],
            [0.35213216, 0.57034597, 0.30848487],
            [0.00770446, 0.88925804, 0.57339844],
        ]
        actual = lhs(3, samples=4, criterion="maximin", seed=42)
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_lhs_correlation(self):
        expected = [
            [0.15479121, 0.49007719, 0.68868284, 0.35721286],
            [0.21883547, 0.39512447, 0.4741596, 0.64544774],
            [0.42562273, 0.08777569, 0.35222794, 0.92633288],
            [0.91091696, 0.76455232, 0.96552623, 0.585353],
            [0.72877302, 0.81276345, 0.17171958, 0.13947361],
        ]
        actual = lhs(
            4, samples=5, criterion="correlation", iterations=10, seed=42
        )
        np.testing.assert_allclose(actual, expected)

    def test_lhs_lhsmu(self):
        expected = [
            [0.66119132, 0.95170391, 0.35575018, 0.41927819],
            [0.18171614, 0.71584391, 0.94389259, 0.34337804],
            [0.88641861, 0.2898723, 0.63535456, 0.77132286],
            [0.58052048, 0.49115526, 0.13994143, 0.05317399],
            [0.25444831, 0.19383528, 0.44047267, 0.92546177],
        ]
        actual = lhs(4, samples=5, criterion="lhsmu", seed=42)
        np.testing.assert_allclose(actual, expected)
