import unittest

import numpy as np

from pydoe import block_ccdesign, block_full_factorial, small_composite_design


class TestBlockFullFactorial(unittest.TestCase):
    def test_shape(self):
        design, blocks = block_full_factorial(3, [(0, 1, 2)])
        self.assertEqual(design.shape, (8, 3))
        self.assertEqual(blocks.shape, (8,))

    def test_two_blocks_have_equal_size(self):
        _design, blocks = block_full_factorial(3, [(0, 1, 2)])
        self.assertEqual(np.sum(blocks == 0), 4)
        self.assertEqual(np.sum(blocks == 1), 4)

    def test_exact_values(self):
        _design, blocks = block_full_factorial(3, [(0, 1, 2)])
        expected_blocks = np.array([1, 0, 0, 1, 0, 1, 1, 0])
        np.testing.assert_array_equal(blocks, expected_blocks)

    def test_confounded_interaction_constant_within_block(self):
        design, blocks = block_full_factorial(3, [(0, 1, 2)])
        contrast = np.sign(design.prod(axis=1))
        for b in np.unique(blocks):
            self.assertEqual(len(np.unique(contrast[blocks == b])), 1)

    def test_four_blocks_from_two_generators(self):
        design, blocks = block_full_factorial(4, [(0, 1), (1, 2, 3)])
        self.assertEqual(design.shape, (16, 4))
        self.assertEqual(set(np.unique(blocks).tolist()), {0, 1, 2, 3})
        for b in range(4):
            self.assertEqual(np.sum(blocks == b), 4)

    def test_raises_k_less_than_2(self):
        with self.assertRaises(ValueError):
            block_full_factorial(1, [(0,)])

    def test_raises_empty_generators(self):
        with self.assertRaises(ValueError):
            block_full_factorial(3, [])

    def test_raises_empty_generator_tuple(self):
        with self.assertRaises(ValueError):
            block_full_factorial(3, [()])

    def test_raises_repeated_index(self):
        with self.assertRaises(ValueError):
            block_full_factorial(3, [(0, 0)])

    def test_raises_out_of_range_index(self):
        with self.assertRaises(ValueError):
            block_full_factorial(3, [(0, 3)])

    def test_raises_too_many_blocks(self):
        with self.assertRaises(ValueError):
            block_full_factorial(2, [(0,), (1,), (0, 1)])


class TestBlockCcdesign(unittest.TestCase):
    def test_shape(self):
        design, blocks = block_ccdesign(2, center=(2, 2))
        self.assertEqual(design.shape, (4 + 2 + 4 + 2, 2))
        self.assertEqual(blocks.shape, (12,))

    def test_block_labels(self):
        _design, blocks = block_ccdesign(2, center=(2, 2))
        self.assertEqual(np.sum(blocks == 0), 6)
        self.assertEqual(np.sum(blocks == 1), 6)

    def test_exact_values_n2(self):
        expected = np.array([
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-np.sqrt(2), 0.0],
            [np.sqrt(2), 0.0],
            [0.0, -np.sqrt(2)],
            [0.0, np.sqrt(2)],
            [0.0, 0.0],
            [0.0, 0.0],
        ])
        design, _blocks = block_ccdesign(2, center=(2, 2))
        np.testing.assert_allclose(design, expected, atol=1e-8)

    def test_factorial_block_is_first(self):
        design, blocks = block_ccdesign(3, center=(4, 4))
        factorial_part = design[blocks == 0]
        self.assertEqual(factorial_part.shape, (8 + 4, 3))

    def test_raises_n_less_than_2(self):
        with self.assertRaises(ValueError):
            block_ccdesign(1)

    def test_raises_invalid_center(self):
        with self.assertRaises(ValueError):
            block_ccdesign(2, center=(1, 1, 1))


class TestSmallCompositeDesign(unittest.TestCase):
    def test_shape(self):
        design = small_composite_design(4, center=(2, 2))
        self.assertEqual(design.shape, (8 + 8 + 4, 4))

    def test_fewer_runs_than_full_ccd(self):
        # Full CCD on 2**4 has 16 + 8 + center runs, small composite uses
        # only 8 cube runs.
        design = small_composite_design(4, center=(2, 2))
        self.assertEqual(design.shape[0], 20)

    def test_cube_portion_is_plus_minus_one(self):
        design = small_composite_design(4, center=(2, 2))
        cube = design[:8]
        self.assertTrue(np.all(np.abs(cube) == 1))

    def test_star_points_present(self):
        design = small_composite_design(4, center=(2, 2))
        star_part = design[8 + 2 :]
        # 2*n star points should each have exactly one nonzero entry
        nonzero_counts = np.count_nonzero(star_part[:8], axis=1)
        np.testing.assert_array_equal(nonzero_counts, np.ones(8))

    def test_raises_n_less_than_3(self):
        with self.assertRaises(ValueError):
            small_composite_design(2)

    def test_raises_invalid_center(self):
        with self.assertRaises(ValueError):
            small_composite_design(4, center=(1,))
