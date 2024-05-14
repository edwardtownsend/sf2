import numpy as np
import numpy.testing as npt

from cued_sf2_lab.dwt import dwt, idwt

def test_dwt():
    X = np.arange(16).reshape(4, 4)
    Y = dwt(X)
    npt.assert_equal(Y,
        [[ 0.  ,  2.25,  0.  ,  0.5 ],
         [ 9.  , 11.25,  0.  ,  0.5 ],
         [ 0.  ,  0.  ,  0.  ,  0.  ],
         [ 2.  ,  2.  ,  0.  ,  0.  ]])
    Xr = idwt(Y)
    npt.assert_equal(Xr, X)
