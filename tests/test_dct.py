import numpy as np
import numpy.testing as npt
import pytest

from cued_sf2_lab.dct import dct_ii, dct_iv, regroup, colxfm

class TestDctII:

    def test_basic(self):
        dct4 = dct_ii(4)
        npt.assert_allclose(dct4,
            [[0.5,      0.5,      0.5,      0.5    ],
             [0.65328,  0.27059, -0.27059, -0.65328],
             [0.5,     -0.5,     -0.5,      0.5    ],
             [0.27059, -0.65328,  0.65328, -0.27059]], atol=1e-5)

    @pytest.mark.xfail(raises=ZeroDivisionError)
    def test_zero(self):
        dct0 = dct_ii(0)
        assert dct0.shape == (0, 0)

    def test_one(self):
        dct1 = dct_ii(1)
        assert dct1.shape == (1, 1)
        npt.assert_allclose(dct1, 1)


class TestDCTIV:

    def test_basic(self):
        dct4 = dct_iv(4)
        npt.assert_allclose(dct4,
            [[ 0.69352 ,  0.587938,  0.392847,  0.13795 ],
             [ 0.587938, -0.13795 , -0.69352 , -0.392847],
             [ 0.392847, -0.69352 ,  0.13795 ,  0.587938],
             [ 0.13795 , -0.392847,  0.587938, -0.69352 ]], atol=1e-5)

    @pytest.mark.xfail(raises=ZeroDivisionError)
    def test_zero(self):
        dct0 = dct_iv(0)
        assert dct0.shape == (0, 0)

    def test_one(self):
        dct1 = dct_iv(1)
        assert dct1.shape == (1, 1)
        npt.assert_allclose(dct1, 1)


class TestRegroup:

    def check(self, x, y, m, n):
        assert x.shape == y.shape
        xm, xn = x.shape
        for xi in range(xm):
            i_div, i_mod = divmod(xi, m)
            yi = i_mod*(xm // m) + i_div
            for xj in range(xn):
                j_div, j_mod = divmod(xj, n)
                yj = j_mod*(xn // n) + j_div

                assert x[xi, xj] == y[yi, yj], (xi, xj, yi, yj)

    def test_roundtrip(self):
        x = np.arange(3*4*5*6).reshape(3*4, 5*6)
        y = regroup(x, [3, 5])
        self.check(x, y, 3, 5)

        # regrouping the other axes puts things back the way they were
        z = regroup(y, [4, 6])
        npt.assert_equal(z, x)

    def test_repeated(self):
        x = np.arange(3*4*3*6).reshape(3*4, 3*6)
        y = regroup(x, 3)
        self.check(x, y, 3, 3)

    def test_invalid(self):
        x = np.arange(3*4*5*6).reshape(3*4, 5*6)
        with pytest.raises(ValueError):
            regroup(x, 7)  # 7 is not a divisor


class TestColXFm:

    def test_basic(self):
        C = dct_ii(4)
        X = np.arange(8*2).reshape(8, 2)
        Y = colxfm(X, C)
        assert Y.shape == X.shape
        npt.assert_allclose(Y,
            [[ 6.    ,  8.    ],
             [-4.4609, -4.4609],
             [-0.    , -0.    ],
             [-0.317 , -0.317 ],
             [22.    , 24.    ],
             [-4.4609, -4.4609],
             [-0.    , -0.    ],
             [-0.317 , -0.317 ]], atol=1e-4)

    def test_invalid(self):
        C = dct_ii(5)
        X = np.arange(16).reshape(16, 1)
        with pytest.raises(ValueError):
            colxfm(X, C)  # 5 is not a divisor of 16
