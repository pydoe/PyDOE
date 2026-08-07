"""Tests for mixture designs: simplex-lattice and simplex-centroid."""

import unittest
from math import comb

import numpy as np

from pydoe import (
    extreme_vertices_design,
    mixture_axial_design,
    mixture_process_design,
    simplex_centroid_design,
    simplex_lattice_design,
)


class TestSimplexLattice(unittest.TestCase):
    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def test_shape_q2_m1(self):
        # C(2+1-1, 1) = 2 points
        self.assertEqual(simplex_lattice_design(2, 1).shape, (2, 2))

    def test_shape_q2_m3(self):
        # C(2+3-1, 3) = C(4,3) = 4 points
        self.assertEqual(simplex_lattice_design(2, 3).shape, (4, 2))

    def test_shape_q3_m2(self):
        # C(3+2-1, 2) = C(4,2) = 6 points
        self.assertEqual(simplex_lattice_design(3, 2).shape, (6, 3))

    def test_shape_formula(self):
        for q in range(2, 6):
            for m in range(1, 5):
                design = simplex_lattice_design(q, m)
                expected_rows = comb(q + m - 1, m)
                self.assertEqual(design.shape, (expected_rows, q))

    # ------------------------------------------------------------------
    # Mixture constraint: rows sum to 1, all non-negative
    # ------------------------------------------------------------------

    def test_rows_sum_to_one(self):
        for q in [2, 3, 4]:
            for m in [1, 2, 3]:
                design = simplex_lattice_design(q, m)
                np.testing.assert_allclose(design.sum(axis=1), 1.0)

    def test_values_non_negative(self):
        design = simplex_lattice_design(3, 3)
        self.assertTrue(np.all(design >= 0))

    def test_values_at_most_one(self):
        design = simplex_lattice_design(4, 2)
        self.assertTrue(np.all(design <= 1.0))

    # ------------------------------------------------------------------
    # Exact values
    # ------------------------------------------------------------------

    def test_values_q2_m3(self):
        expected = np.array([
            [1.0, 0.0],
            [2 / 3, 1 / 3],
            [1 / 3, 2 / 3],
            [0.0, 1.0],
        ])
        np.testing.assert_allclose(simplex_lattice_design(2, 3), expected)

    def test_values_q3_m2(self):
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ])
        np.testing.assert_allclose(simplex_lattice_design(3, 2), expected)

    def test_m1_gives_vertices(self):
        # m=1 gives q pure-component vertices = identity matrix
        for q in [2, 3, 4, 5]:
            design = simplex_lattice_design(q, 1)
            np.testing.assert_allclose(design, np.eye(q))

    def test_lattice_values_on_grid(self):
        # Every value must be an integer multiple of 1/m
        m = 4
        design = simplex_lattice_design(3, m)
        residuals = (design * m) % 1
        np.testing.assert_allclose(residuals, 0, atol=1e-12)

    def test_q4_m3_shape(self):
        # C(4+3-1, 3) = C(6,3) = 20
        self.assertEqual(simplex_lattice_design(4, 3).shape, (20, 4))

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_q_less_than_2(self):
        with self.assertRaises(ValueError):
            simplex_lattice_design(1, 2)

    def test_raises_q_zero(self):
        with self.assertRaises(ValueError):
            simplex_lattice_design(0, 2)

    def test_raises_m_zero(self):
        with self.assertRaises(ValueError):
            simplex_lattice_design(3, 0)

    def test_raises_m_negative(self):
        with self.assertRaises(ValueError):
            simplex_lattice_design(3, -1)


class TestSimplexCentroid(unittest.TestCase):
    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    def test_shape_q2(self):
        # 2^2 - 1 = 3 points
        self.assertEqual(simplex_centroid_design(2).shape, (3, 2))

    def test_shape_q3(self):
        # 2^3 - 1 = 7 points
        self.assertEqual(simplex_centroid_design(3).shape, (7, 3))

    def test_shape_q4(self):
        # 2^4 - 1 = 15 points
        self.assertEqual(simplex_centroid_design(4).shape, (15, 4))

    def test_shape_formula(self):
        for q in range(2, 7):
            design = simplex_centroid_design(q)
            self.assertEqual(design.shape, (2**q - 1, q))

    # ------------------------------------------------------------------
    # Mixture constraint
    # ------------------------------------------------------------------

    def test_rows_sum_to_one(self):
        for q in [2, 3, 4, 5]:
            design = simplex_centroid_design(q)
            np.testing.assert_allclose(design.sum(axis=1), 1.0)

    def test_values_non_negative(self):
        design = simplex_centroid_design(4)
        self.assertTrue(np.all(design >= 0))

    def test_values_at_most_one(self):
        design = simplex_centroid_design(4)
        self.assertTrue(np.all(design <= 1.0))

    # ------------------------------------------------------------------
    # Exact values
    # ------------------------------------------------------------------

    def test_values_q3(self):
        expected = np.array([
            [1.0, 0.0, 0.0],  # vertex 1
            [0.0, 1.0, 0.0],  # vertex 2
            [0.0, 0.0, 1.0],  # vertex 3
            [0.5, 0.5, 0.0],  # edge centroid 12
            [0.5, 0.0, 0.5],  # edge centroid 13
            [0.0, 0.5, 0.5],  # edge centroid 23
            [1 / 3, 1 / 3, 1 / 3],  # overall centroid
        ])
        np.testing.assert_allclose(simplex_centroid_design(3), expected)

    def test_first_rows_are_vertices(self):
        # First q rows are the pure-component vertices (= identity matrix)
        for q in [2, 3, 4]:
            design = simplex_centroid_design(q)
            np.testing.assert_allclose(design[:q], np.eye(q))

    def test_last_row_is_overall_centroid(self):
        # Last row is always (1/q, ..., 1/q)
        for q in [2, 3, 4, 5]:
            design = simplex_centroid_design(q)
            np.testing.assert_allclose(design[-1], np.full(q, 1.0 / q))

    def test_binary_blends_have_exactly_two_nonzero(self):
        # Rows q through q + C(q,2) - 1 are binary blends (exactly 2 nonzeros)
        q = 4
        design = simplex_centroid_design(q)
        n_binary = comb(q, 2)
        binary_rows = design[q : q + n_binary]
        for row in binary_rows:
            nonzero = np.count_nonzero(row)
            self.assertEqual(nonzero, 2)
            np.testing.assert_allclose(row[row > 0], 0.5)

    def test_no_duplicate_rows(self):
        design = simplex_centroid_design(4)
        n = design.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                self.assertFalse(
                    np.allclose(design[i], design[j]),
                    msg=f"Rows {i} and {j} are duplicates",
                )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_raises_q_less_than_2(self):
        with self.assertRaises(ValueError):
            simplex_centroid_design(1)

    def test_raises_q_zero(self):
        with self.assertRaises(ValueError):
            simplex_centroid_design(0)


class TestMixtureAxialDesign(unittest.TestCase):
    def test_shape(self):
        for q in [2, 3, 4, 5]:
            design = mixture_axial_design(q)
            self.assertEqual(design.shape, (q + 1, q))

    def test_rows_sum_to_one(self):
        for q in [2, 3, 4]:
            design = mixture_axial_design(q, delta=0.3)
            np.testing.assert_allclose(design.sum(axis=1), 1.0)

    def test_first_row_is_centroid(self):
        q = 4
        design = mixture_axial_design(q)
        np.testing.assert_allclose(design[0], np.full(q, 1.0 / q))

    def test_values_non_negative(self):
        design = mixture_axial_design(5, delta=1.0)
        self.assertTrue(np.all(design >= 0))

    def test_delta_one_gives_vertices_after_centroid(self):
        q = 3
        design = mixture_axial_design(q, delta=1.0)
        np.testing.assert_allclose(design[1:], np.eye(q))

    def test_exact_values_q3(self):
        expected = np.array([
            [1 / 3, 1 / 3, 1 / 3],
            [2 / 3, 1 / 6, 1 / 6],
            [1 / 6, 2 / 3, 1 / 6],
            [1 / 6, 1 / 6, 2 / 3],
        ])
        np.testing.assert_allclose(mixture_axial_design(3, delta=0.5), expected)

    def test_raises_q_less_than_2(self):
        with self.assertRaises(ValueError):
            mixture_axial_design(1)

    def test_raises_invalid_delta(self):
        with self.assertRaises(ValueError):
            mixture_axial_design(3, delta=0.0)
        with self.assertRaises(ValueError):
            mixture_axial_design(3, delta=1.5)


class TestExtremeVerticesDesign(unittest.TestCase):
    def test_rows_sum_to_one(self):
        verts = extreme_vertices_design([0.1, 0.1, 0.1], [0.6, 0.6, 0.6])
        np.testing.assert_allclose(verts.sum(axis=1), 1.0)

    def test_within_bounds(self):
        lower = [0.0, 0.2, 0.0]
        upper = [1.0, 0.5, 0.7]
        verts = extreme_vertices_design(lower, upper)
        self.assertTrue(np.all(verts >= np.array(lower) - 1e-9))
        self.assertTrue(np.all(verts <= np.array(upper) + 1e-9))

    def test_exact_values_symmetric_region(self):
        expected = np.array([
            [0.1, 0.3, 0.6],
            [0.1, 0.6, 0.3],
            [0.3, 0.1, 0.6],
            [0.3, 0.6, 0.1],
            [0.6, 0.1, 0.3],
            [0.6, 0.3, 0.1],
        ])
        verts = extreme_vertices_design([0.1, 0.1, 0.1], [0.6, 0.6, 0.6])
        np.testing.assert_allclose(verts, expected)

    def test_unconstrained_region_gives_simplex_vertices(self):
        verts = extreme_vertices_design([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            np.sort(verts, axis=0), np.sort(np.eye(3), axis=0)
        )

    def test_no_duplicate_rows(self):
        verts = extreme_vertices_design([0.1, 0.1, 0.1], [0.6, 0.6, 0.6])
        self.assertEqual(len(np.unique(verts, axis=0)), len(verts))

    def test_raises_shape_mismatch(self):
        with self.assertRaises(ValueError):
            extreme_vertices_design([0.1, 0.1], [0.6, 0.6, 0.6])

    def test_raises_lower_exceeds_upper(self):
        with self.assertRaises(ValueError):
            extreme_vertices_design([0.7, 0.1, 0.1], [0.6, 0.6, 0.6])

    def test_raises_infeasible_lower_sum(self):
        with self.assertRaises(ValueError):
            extreme_vertices_design([0.5, 0.5, 0.5], [0.9, 0.9, 0.9])

    def test_raises_infeasible_upper_sum(self):
        with self.assertRaises(ValueError):
            extreme_vertices_design([0.0, 0.0, 0.0], [0.2, 0.2, 0.2])


class TestMixtureProcessDesign(unittest.TestCase):
    def test_shape(self):
        mixture = simplex_lattice_design(3, 1)
        process = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        design = mixture_process_design(mixture, process)
        self.assertEqual(design.shape, (3 * 4, 3 + 2))

    def test_mixture_columns_sum_to_one(self):
        mixture = simplex_lattice_design(3, 2)
        process = np.array([[-1.0], [1.0]])
        design = mixture_process_design(mixture, process)
        np.testing.assert_allclose(design[:, :3].sum(axis=1), 1.0)

    def test_exact_values(self):
        mixture = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        process = np.array([[-1.0], [1.0]])
        expected = np.array([
            [1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [0.5, 0.5, -1.0],
            [0.5, 0.5, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, 1.0, 1.0],
        ])
        np.testing.assert_allclose(
            mixture_process_design(mixture, process), expected
        )

    def test_full_factorial_pairing(self):
        mixture = np.eye(2)
        process = np.eye(3)
        design = mixture_process_design(mixture, process)
        self.assertEqual(design.shape, (6, 5))
        # First two rows pair mixture row 0 with each process row
        np.testing.assert_allclose(design[:3, :2], np.tile(mixture[0], (3, 1)))
