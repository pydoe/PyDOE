import numpy as np

from pydoe import graeco_latin_square, hyper_graeco_latin_square, latin_square


class TestLatinSquare:
    def test_shape(self):
        for n in (2, 3, 4, 5, 7):
            square = latin_square(n)
            assert square.shape == (n, n)

    def test_values_in_range(self):
        n = 6
        square = latin_square(n)
        assert set(np.unique(square).tolist()) == set(range(n))

    def test_rows_are_permutations(self):
        n = 6
        square = latin_square(n)
        for row in square:
            assert sorted(row.tolist()) == list(range(n))

    def test_columns_are_permutations(self):
        n = 6
        square = latin_square(n)
        for col in square.T:
            assert sorted(col.tolist()) == list(range(n))

    def test_exact_values_n4(self):
        expected = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ])
        np.testing.assert_array_equal(latin_square(4), expected)

    def test_n_less_than_2_raises(self):
        try:
            latin_square(1)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")


class TestGraecoLatinSquare:
    def test_shapes(self):
        latin, graeco = graeco_latin_square(5)
        assert latin.shape == (5, 5)
        assert graeco.shape == (5, 5)

    def test_orthogonality(self):
        for n in (3, 5, 7):
            latin, graeco = graeco_latin_square(n)
            pairs = set(
                zip(
                    latin.ravel().tolist(), graeco.ravel().tolist(), strict=True
                )
            )
            assert len(pairs) == n * n

    def test_each_square_is_latin(self):
        latin, graeco = graeco_latin_square(5)
        for square in (latin, graeco):
            for row in square:
                assert sorted(row.tolist()) == list(range(5))
            for col in square.T:
                assert sorted(col.tolist()) == list(range(5))

    def test_exact_values_n3(self):
        expected_latin = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
        expected_graeco = np.array([[0, 2, 1], [1, 0, 2], [2, 1, 0]])
        latin, graeco = graeco_latin_square(3)
        np.testing.assert_array_equal(latin, expected_latin)
        np.testing.assert_array_equal(graeco, expected_graeco)

    def test_non_prime_raises(self):
        try:
            graeco_latin_square(4)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    def test_n_too_small_raises(self):
        try:
            graeco_latin_square(2)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")


class TestHyperGraecoLatinSquare:
    def test_shape(self):
        squares = hyper_graeco_latin_square(5, 3)
        assert squares.shape == (3, 5, 5)

    def test_pairwise_orthogonality(self):
        n, k = 7, 4
        squares = hyper_graeco_latin_square(n, k)
        for a in range(k):
            for b in range(a + 1, k):
                pairs = set(
                    zip(
                        squares[a].ravel().tolist(),
                        squares[b].ravel().tolist(),
                        strict=True,
                    )
                )
                assert len(pairs) == n * n

    def test_each_square_is_latin(self):
        squares = hyper_graeco_latin_square(5, 3)
        for square in squares:
            for row in square:
                assert sorted(row.tolist()) == list(range(5))
            for col in square.T:
                assert sorted(col.tolist()) == list(range(5))

    def test_exact_values_n5_k3(self):
        squares = hyper_graeco_latin_square(5, 3)
        expected_first = np.array([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0],
            [2, 3, 4, 0, 1],
            [3, 4, 0, 1, 2],
            [4, 0, 1, 2, 3],
        ])
        expected_third = np.array([
            [0, 3, 1, 4, 2],
            [1, 4, 2, 0, 3],
            [2, 0, 3, 1, 4],
            [3, 1, 4, 2, 0],
            [4, 2, 0, 3, 1],
        ])
        np.testing.assert_array_equal(squares[0], expected_first)
        np.testing.assert_array_equal(squares[2], expected_third)

    def test_non_prime_n_raises(self):
        try:
            hyper_graeco_latin_square(4, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

    def test_k_out_of_range_raises(self):
        try:
            hyper_graeco_latin_square(5, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")

        try:
            hyper_graeco_latin_square(5, 5)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
