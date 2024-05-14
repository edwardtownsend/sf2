import numpy.testing as npt
import pytest

from cued_sf2_lab.lbt import pot_ii


class TestPotII:

    def test_basic(self):
        Pr, Pf = pot_ii(4)
        npt.assert_allclose(Pr,
            [[ 1.16381485, -0.10772232,  0.10772232, -0.16381485],
             [ 0.3932168 ,  1.04555916, -0.04555916, -0.3932168 ],
             [-0.3932168 , -0.04555916,  1.04555916,  0.3932168 ],
             [-0.16381485,  0.10772232, -0.10772232,  1.16381485]])
        npt.assert_allclose(Pf,
            [[ 0.83717411, -0.24302135,  0.24302135,  0.16282589],
             [ 0.06657606,  0.91026014,  0.08973986, -0.06657606],
             [-0.06657606,  0.08973986,  0.91026014,  0.06657606],
             [ 0.16282589,  0.24302135, -0.24302135,  0.83717411]])

    def test_overlap_small(self):
        Pr, Pf = pot_ii(4, overlap=1)
        npt.assert_allclose(Pr,
            [[ 1.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  1.30901699, -0.30901699,  0.        ],
             [ 0.        , -0.30901699,  1.30901699,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  1.        ]])
        npt.assert_allclose(Pf,
            [[1.        , 0.        , 0.        , 0.        ],
             [0.        , 0.80901699, 0.19098301, 0.        ],
             [0.        , 0.19098301, 0.80901699, 0.        ],
             [0.        , 0.        , 0.        , 1.        ]])

    def test_illegal(self):
        with pytest.raises(ValueError, match="divisible by 2"):
            pot_ii(5)
        with pytest.raises(Exception):
            pot_ii(2.5)
        with pytest.raises(ValueError, match="overlap must satisfy"):
            pot_ii(8, overlap=5)
