import pytest
import numpy as np
import numpy.testing as npt

from cued_sf2_lab.laplacian_pyramid import (
    rowdec, rowdec2, beside, rowint, quant1, quantise)


def test_rowdec_odd():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 2, 1])
    npt.assert_equal(Y1,
        [[0, 0],
         [6, 10]])
    Y2 = rowdec2(X, [1, 2, 1])
    npt.assert_equal(Y2,
        [[0, 0],
         [8, 10]])

def test_rowdec_even():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 1])
    npt.assert_equal(Y1,
        [[0, 0],
         [3, 5]])

# TODO: fix this!
@pytest.mark.xfail(exception=IndexError)
def test_rowdec2_even():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y2 = rowdec2(X, [1, 1])
    npt.assert_equal(Y2,
        [[0, 0],
         [8, 10]])

# TODO: test `test_plot_laplacian_pyramid`

def test_beside():
    a = np.full((5, 5), 1)
    b = np.full((2, 2), 2)
    c = np.full((1, 1), 2)
    abc = beside(a, beside(b, c))
    npt.assert_equal(abc,
        [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 2, 2, 0, 2],
         [1, 1, 1, 1, 1, 0, 2, 2, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])

def test_rowint_odd():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 2, 1])
    Z = rowint(Y1, [1, 2, 1])
    npt.assert_equal(Z,
        [[ 0,  0,  0,  0],
         [12, 16, 20, 20]])

def test_rowint_even():
    X = np.array(
        [[0, 0, 0, 0],
         [2, 1, 4, 1]])
    Y1 = rowdec(X, [1, 1])
    Z = rowint(Y1, [1, 1])
    npt.assert_equal(Z,
        [[0, 0, 0, 0],
         [6, 3, 5, 5]])

def test_quant1():
    X = np.arange(-10, 11)
    Xq = quant1(X, 2.5)
    np.testing.assert_equal(Xq,
        [-4, -4, -3, -3,
         -2, -2, -2, -1, -1,
          0, 0, 0,
          1, 1, 2, 2, 2,
          3, 3, 4, 4])
    # TODO: should this return integers?
    # assert np.issubdtype(Xq.dtype, np.integer)

    # TODO: does this even make sense?
    Xq = quant1(X, 0)
    np.testing.assert_equal(Xq, X)

def test_quantise():
    X = np.arange(-10, 11)
    Xq = quantise(X, 2.5)
    np.testing.assert_equal(Xq,
        [-10.0, -10.0, -7.5, -7.5,
         -5.0, -5.0, -5.0, -2.5, -2.5,
         0.0, 0.0, 0.0,
         2.5, 2.5, 5.0, 5.0, 5.0,
         7.5, 7.5, 10.0, 10.0])

    Xq = quantise(X, 0)
    np.testing.assert_equal(Xq, X)

def test_bpp():
    # TODO: write this
    pass
