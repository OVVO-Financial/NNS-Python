# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

from NNS import Partial_Moments as Partial_Moments


class TestPartialMoments(unittest.TestCase):
    COMPARISON_PRECISION = 7

    def assertAlmostEqualArray(
        self, x: [pd.Series, np.ndarray, list, dict], y: [pd.Series, np.ndarray, list, dict]
    ) -> None:
        if isinstance(x, dict):
            for i in x.keys():
                self.assertAlmostEqual(x[i], y[i])
        else:
            for i in range(len(x)):
                self.assertAlmostEqual(x[i], y[i])

    def test_LPM(self):
        x = self.load_default_data()["x"]
        # pandas
        self.assertAlmostEqual(Partial_Moments.LPM(0, None, x), 0.49)
        self.assertAlmostEqual(Partial_Moments.LPM(0, x.mean(), x), 0.49)
        self.assertAlmostEqual(Partial_Moments.LPM(1, x.mean(), x), 0.1032933)
        self.assertAlmostEqual(Partial_Moments.LPM(2, x.mean(), x), 0.02993767)

        # ndarray
        self.assertAlmostEqual(Partial_Moments.LPM(0, None, x.values), 0.49)
        self.assertAlmostEqual(Partial_Moments.LPM(0, x.mean(), x.values), 0.49)
        self.assertAlmostEqual(Partial_Moments.LPM(1, x.mean(), x.values), 0.1032933)
        self.assertAlmostEqual(Partial_Moments.LPM(2, x.mean(), x.values), 0.02993767)
        # pandas
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(0, x[4:10], x), [0.80, 0.41, 0.98, 0.74, 0.47, 0.35]
        )
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(1, x[4:10], x),
            [0.24289462, 0.06719648, 0.47952922, 0.21609444, 0.09339833, 0.05546457],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(2, x[4:10], x),
            [0.10301058, 0.01663970, 0.28997176, 0.08712359, 0.02590746, 0.01284660],
        )
        # ndarray
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(0, x[4:10].values, x.values), [0.80, 0.41, 0.98, 0.74, 0.47, 0.35]
        )
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(1, x[4:10].values, x.values),
            [0.24289462, 0.06719648, 0.47952922, 0.21609444, 0.09339833, 0.05546457],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(2, x[4:10].values, x.values),
            [0.10301058, 0.01663970, 0.28997176, 0.08712359, 0.02590746, 0.01284660],
        )
        # list
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(0, list(x[4:10].values), list(x.values)),
            [0.80, 0.41, 0.98, 0.74, 0.47, 0.35],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(1, list(x[4:10].values), list(x.values)),
            [0.24289462, 0.06719648, 0.47952922, 0.21609444, 0.09339833, 0.05546457],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.LPM(2, list(x[4:10].values), list(x.values)),
            [0.10301058, 0.01663970, 0.28997176, 0.08712359, 0.02590746, 0.01284660],
        )

    def test_UPM(self):
        x = self.load_default_data()["x"]
        # pandas
        self.assertAlmostEqual(Partial_Moments.UPM(0, None, x), 0.51)
        self.assertAlmostEqual(Partial_Moments.UPM(0, x.mean(), x), 0.51)
        self.assertAlmostEqual(Partial_Moments.UPM(1, x.mean(), x), 0.1032933)
        self.assertAlmostEqual(Partial_Moments.UPM(2, x.mean(), x), 0.03027411)
        # ndarray
        self.assertAlmostEqual(Partial_Moments.UPM(0, None, x.values), 0.51)
        self.assertAlmostEqual(Partial_Moments.UPM(0, x.mean(), x.values), 0.51)
        self.assertAlmostEqual(Partial_Moments.UPM(1, x.mean(), x.values), 0.1032933)
        self.assertAlmostEqual(Partial_Moments.UPM(2, x.mean(), x.values), 0.03027411)

        # pandas
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(0, x[4:10], x), [0.20, 0.59, 0.02, 0.26, 0.53, 0.65]
        )
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(1, x[4:10], x),
            [0.0248545343, 0.1455188996, 0.0001938987, 0.0326935823, 0.1138953065, 0.1647759354],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(2, x[4:10], x),
            [4.742673e-03, 4.970647e-02, 2.359908e-06, 6.724064e-03, 3.472444e-02, 5.931415e-02],
        )
        # ndarray
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(0, x[4:10].values, x.values), [0.20, 0.59, 0.02, 0.26, 0.53, 0.65]
        )
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(1, x[4:10].values, x.values),
            [0.0248545343, 0.1455188996, 0.0001938987, 0.0326935823, 0.1138953065, 0.1647759354],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(2, x[4:10].values, x.values),
            [4.742673e-03, 4.970647e-02, 2.359908e-06, 6.724064e-03, 3.472444e-02, 5.931415e-02],
        )
        # list
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(0, list(x[4:10].values), list(x.values)),
            [0.20, 0.59, 0.02, 0.26, 0.53, 0.65],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(1, list(x[4:10].values), list(x.values)),
            [0.0248545343, 0.1455188996, 0.0001938987, 0.0326935823, 0.1138953065, 0.1647759354],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.UPM(2, list(x[4:10].values), list(x.values)),
            [4.742673e-03, 4.970647e-02, 2.359908e-06, 6.724064e-03, 3.472444e-02, 5.931415e-02],
        )

    def test_Co_UPM(self):
        z = self.load_default_data()
        x, y = z["x"], z["y"]
        # pandas
        self.assertAlmostEqual(Partial_Moments.Co_UPM(0, 0, x, y, None, None), 0.28)
        self.assertAlmostEqual(Partial_Moments.Co_UPM(0, 0, x, y, x.mean(), y.mean()), 0.28)
        self.assertAlmostEqual(Partial_Moments.Co_UPM(0, 1, x, y, x.mean(), y.mean()), 0.06314884)
        self.assertAlmostEqual(Partial_Moments.Co_UPM(1, 0, x, y, x.mean(), y.mean()), 0.05017661)
        self.assertAlmostEqual(Partial_Moments.Co_UPM(1, 1, x, y, x.mean(), y.mean()), 0.01204606)
        self.assertAlmostEqual(Partial_Moments.Co_UPM(0, 2, x, y, x.mean(), y.mean()), 0.01741738)
        self.assertAlmostEqual(Partial_Moments.Co_UPM(2, 0, x, y, x.mean(), y.mean()), 0.01300084)
        self.assertAlmostEqual(Partial_Moments.Co_UPM(2, 2, x, y, x.mean(), y.mean()), 0.0009799173)
        # list
        self.assertAlmostEqual(Partial_Moments.Co_UPM(0, 0, list(x), list(y), None, None), 0.28)
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(0, 0, list(x), list(y), x.mean(), y.mean()), 0.28
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(0, 1, list(x), list(y), x.mean(), y.mean()), 0.06314884
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(1, 0, list(x), list(y), x.mean(), y.mean()), 0.05017661
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(1, 1, list(x), list(y), x.mean(), y.mean()), 0.01204606
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(0, 2, list(x), list(y), x.mean(), y.mean()), 0.01741738
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(2, 0, list(x), list(y), x.mean(), y.mean()), 0.01300084
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(2, 2, list(x), list(y), x.mean(), y.mean()), 0.0009799173
        )
        # ndarray
        self.assertAlmostEqual(Partial_Moments.Co_UPM(0, 0, x.values, y.values, None, None), 0.28)
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(0, 0, x.values, y.values, x.mean(), y.mean()), 0.28
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(0, 1, x.values, y.values, x.mean(), y.mean()), 0.06314884
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(1, 0, x.values, y.values, x.mean(), y.mean()), 0.05017661
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(1, 1, x.values, y.values, x.mean(), y.mean()), 0.01204606
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(0, 2, x.values, y.values, x.mean(), y.mean()), 0.01741738
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(2, 0, x.values, y.values, x.mean(), y.mean()), 0.01300084
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_UPM(2, 2, x.values, y.values, x.mean(), y.mean()), 0.0009799173
        )

        # pandas
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(0, 0, x, y, x[4:10], y[4:10]),
            [0.16, 0.18, 0.00, 0.10, 0.29, 0.05],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(0, 1, x, y, x[4:10], y[4:10]),
            [0.061739827, 0.022388081, 0.000000000, 0.012079921, 0.046203058, 0.003264875],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(1, 0, x, y, x[4:10], y[4:10]),
            [0.02350672, 0.04529021, 0.00000000, 0.01175768, 0.05593678, 0.01283602],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(1, 1, x, y, x[4:10], y[4:10]),
            [0.0077066976, 0.0053850329, 0.0000000000, 0.0016943266, 0.0098111799, 0.0009596762],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(0, 2, x, y, x[4:10], y[4:10]),
            [0.0343573169, 0.0036537229, 0.0000000000, 0.0021943104, 0.0106127604, 0.0002280662],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(2, 0, x, y, x[4:10], y[4:10]),
            [0.004591201, 0.015727129, 0.000000000, 0.002217705, 0.015175454, 0.004628285],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(2, 2, x, y, x[4:10], y[4:10]),
            [7.393054e-04, 3.113780e-04, 0.000000e00, 7.092159e-05, 7.024123e-04, 3.236049e-05],
        )
        # ndarray
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(0, 0, x, y, x[4:10].values, y[4:10].values),
            [0.16, 0.18, 0.00, 0.10, 0.29, 0.05],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(0, 1, x, y, x[4:10].values, y[4:10].values),
            [0.061739827, 0.022388081, 0.000000000, 0.012079921, 0.046203058, 0.003264875],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(1, 0, x, y, x[4:10].values, y[4:10].values),
            [0.02350672, 0.04529021, 0.00000000, 0.01175768, 0.05593678, 0.01283602],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(1, 1, x, y, x[4:10].values, y[4:10].values),
            [0.0077066976, 0.0053850329, 0.0000000000, 0.0016943266, 0.0098111799, 0.0009596762],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(0, 2, x, y, x[4:10].values, y[4:10].values),
            [0.0343573169, 0.0036537229, 0.0000000000, 0.0021943104, 0.0106127604, 0.0002280662],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(2, 0, x, y, x[4:10].values, y[4:10].values),
            [0.004591201, 0.015727129, 0.000000000, 0.002217705, 0.015175454, 0.004628285],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_UPM(2, 2, x, y, x[4:10], y[4:10]),
            [7.393054e-04, 3.113780e-04, 0.000000e00, 7.092159e-05, 7.024123e-04, 3.236049e-05],
        )

    def test_Co_LPM(self):
        z = self.load_default_data()
        x, y = z["x"], z["y"]
        # pandas
        self.assertAlmostEqual(Partial_Moments.Co_LPM(0, 0, x, y, None, None), 0.24)
        self.assertAlmostEqual(Partial_Moments.Co_LPM(0, 0, x, y, x.mean(), y.mean()), 0.24)
        self.assertAlmostEqual(Partial_Moments.Co_LPM(0, 1, x, y, x.mean(), y.mean()), 0.0539954)
        self.assertAlmostEqual(Partial_Moments.Co_LPM(1, 0, x, y, x.mean(), y.mean()), 0.04485831)
        self.assertAlmostEqual(Partial_Moments.Co_LPM(1, 1, x, y, x.mean(), y.mean()), 0.01058035)
        self.assertAlmostEqual(Partial_Moments.Co_LPM(0, 2, x, y, x.mean(), y.mean()), 0.01848438)
        self.assertAlmostEqual(Partial_Moments.Co_LPM(2, 0, x, y, x.mean(), y.mean()), 0.010676)
        self.assertAlmostEqual(Partial_Moments.Co_LPM(2, 2, x, y, x.mean(), y.mean()), 0.0008940764)
        # list
        self.assertAlmostEqual(Partial_Moments.Co_LPM(0, 0, list(x), list(y), None, None), 0.24)
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(0, 0, list(x), list(y), x.mean(), y.mean()), 0.24
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(0, 1, list(x), list(y), x.mean(), y.mean()), 0.0539954
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(1, 0, list(x), list(y), x.mean(), y.mean()), 0.04485831
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(1, 1, list(x), list(y), x.mean(), y.mean()), 0.01058035
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(0, 2, list(x), list(y), x.mean(), y.mean()), 0.01848438
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(2, 0, list(x), list(y), x.mean(), y.mean()), 0.010676
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(2, 2, list(x), list(y), x.mean(), y.mean()), 0.0008940764
        )
        # ndarray
        self.assertAlmostEqual(Partial_Moments.Co_LPM(0, 0, x.values, y.values, None, None), 0.24)
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(0, 0, x.values, y.values, x.mean(), y.mean()), 0.24
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(0, 1, x.values, y.values, x.mean(), y.mean()), 0.0539954
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(1, 0, x.values, y.values, x.mean(), y.mean()), 0.04485831
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(1, 1, x.values, y.values, x.mean(), y.mean()), 0.01058035
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(0, 2, x.values, y.values, x.mean(), y.mean()), 0.01848438
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(2, 0, x.values, y.values, x.mean(), y.mean()), 0.010676
        )
        self.assertAlmostEqual(
            Partial_Moments.Co_LPM(2, 2, x.values, y.values, x.mean(), y.mean()), 0.0008940764
        )

        # NNS::Co.LPM(0, 0, x, y, x[5:10], y[5:10])
        # is equal to
        # Partial_Moments.Co_LPM(0, 0, x, y, x[4:10], y[4:10])
        # cause R use a different slice method
        # pandas
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(0, 0, x, y, x[4:10], y[4:10]),
            [0.08, 0.27, 0.23, 0.48, 0.26, 0.33],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(0, 1, x, y, x[4:10], y[4:10]),
            [0.007927292, 0.095962199, 0.033041430, 0.136350213, 0.068840778, 0.124456005],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(1, 0, x, y, x[4:10], y[4:10]),
            [0.02806505, 0.04580165, 0.09941373, 0.14270236, 0.05234510, 0.05274968],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(1, 1, x, y, x[4:10], y[4:10]),
            [0.002703598, 0.013426491, 0.015289822, 0.039692579, 0.012319601, 0.017200653],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(0, 2, x, y, x[4:10], y[4:10]),
            [0.001028405, 0.044427617, 0.006357793, 0.057760425, 0.026193545, 0.068839353],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(2, 0, x, y, x[4:10], y[4:10]),
            [0.01192015, 0.01147958, 0.05527885, 0.05837932, 0.01401319, 0.01210954],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(2, 2, x, y, x[4:10], y[4:10]),
            [0.0001439027, 0.0010814400, 0.0017942588, 0.0063185186, 0.0010455681, 0.0016179699],
        )

        # ndarray
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(0, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.08, 0.27, 0.23, 0.48, 0.26, 0.33],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(0, 1, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.007927292, 0.095962199, 0.033041430, 0.136350213, 0.068840778, 0.124456005],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(1, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.02806505, 0.04580165, 0.09941373, 0.14270236, 0.05234510, 0.05274968],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(1, 1, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.002703598, 0.013426491, 0.015289822, 0.039692579, 0.012319601, 0.017200653],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(0, 2, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.001028405, 0.044427617, 0.006357793, 0.057760425, 0.026193545, 0.068839353],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(2, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.01192015, 0.01147958, 0.05527885, 0.05837932, 0.01401319, 0.01210954],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(2, 2, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.0001439027, 0.0010814400, 0.0017942588, 0.0063185186, 0.0010455681, 0.0016179699],
        )

        # list
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(
                0, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.08, 0.27, 0.23, 0.48, 0.26, 0.33],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(
                0, 1, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.007927292, 0.095962199, 0.033041430, 0.136350213, 0.068840778, 0.124456005],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(
                1, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.02806505, 0.04580165, 0.09941373, 0.14270236, 0.05234510, 0.05274968],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(
                1, 1, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.002703598, 0.013426491, 0.015289822, 0.039692579, 0.012319601, 0.017200653],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(
                0, 2, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.001028405, 0.044427617, 0.006357793, 0.057760425, 0.026193545, 0.068839353],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(
                2, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.01192015, 0.01147958, 0.05527885, 0.05837932, 0.01401319, 0.01210954],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.Co_LPM(
                2, 2, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.0001439027, 0.0010814400, 0.0017942588, 0.0063185186, 0.0010455681, 0.0016179699],
        )

    def test_D_LPM(self):
        z = self.load_default_data()
        x, y = z["x"], z["y"]
        # pandas
        self.assertAlmostEqual(Partial_Moments.D_LPM(0, 0, x, y, None, None), 0.23)
        self.assertAlmostEqual(Partial_Moments.D_LPM(0, 0, x, y, x.mean(), y.mean()), 0.23)
        self.assertAlmostEqual(Partial_Moments.D_LPM(0, 1, x, y, x.mean(), y.mean()), 0.06404049)
        self.assertAlmostEqual(Partial_Moments.D_LPM(1, 0, x, y, x.mean(), y.mean()), 0.05311669)
        self.assertAlmostEqual(Partial_Moments.D_LPM(1, 1, x, y, x.mean(), y.mean()), 0.01513793)
        self.assertAlmostEqual(Partial_Moments.D_LPM(0, 2, x, y, x.mean(), y.mean()), 0.02248309)
        self.assertAlmostEqual(Partial_Moments.D_LPM(2, 0, x, y, x.mean(), y.mean()), 0.01727327)
        self.assertAlmostEqual(Partial_Moments.D_LPM(2, 2, x, y, x.mean(), y.mean()), 0.001554909)
        # list
        self.assertAlmostEqual(Partial_Moments.D_LPM(0, 0, list(x), list(y), None, None), 0.23)
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(0, 0, list(x), list(y), x.mean(), y.mean()), 0.23
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(0, 1, list(x), list(y), x.mean(), y.mean()), 0.06404049
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(1, 0, list(x), list(y), x.mean(), y.mean()), 0.05311669
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(1, 1, list(x), list(y), x.mean(), y.mean()), 0.01513793
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(0, 2, list(x), list(y), x.mean(), y.mean()), 0.02248309
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(2, 0, list(x), list(y), x.mean(), y.mean()), 0.01727327
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(2, 2, list(x), list(y), x.mean(), y.mean()), 0.001554909
        )
        # ndarray
        self.assertAlmostEqual(Partial_Moments.D_LPM(0, 0, x.values, y.values, None, None), 0.23)
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(0, 0, x.values, y.values, x.mean(), y.mean()), 0.23
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(0, 1, x.values, y.values, x.mean(), y.mean()), 0.06404049
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(1, 0, x.values, y.values, x.mean(), y.mean()), 0.05311669
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(1, 1, x.values, y.values, x.mean(), y.mean()), 0.01513793
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(0, 2, x.values, y.values, x.mean(), y.mean()), 0.02248309
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(2, 0, x.values, y.values, x.mean(), y.mean()), 0.01727327
        )
        self.assertAlmostEqual(
            Partial_Moments.D_LPM(2, 2, x.values, y.values, x.mean(), y.mean()), 0.001554909
        )

        # pandas
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(0, 0, x, y, x[4:10], y[4:10]),
            [0.04, 0.41, 0.02, 0.16, 0.24, 0.60],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(0, 1, x, y, x[4:10], y[4:10]),
            [0.0033924100, 0.1228448321, 0.0005880316, 0.0585458775, 0.0795664636, 0.2168141323],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(1, 0, x, y, x[4:10], y[4:10]),
            [0.0013478162, 0.1002286860, 0.0001938987, 0.0209359065, 0.0579585233, 0.1519399110],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(1, 1, x, y, x[4:10], y[4:10]),
            [7.057350e-05, 3.485177e-02, 6.464252e-06, 8.244865e-03, 2.011508e-02, 5.933306e-02],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(0, 2, x, y, x[4:10], y[4:10]),
            [3.227829e-04, 5.633341e-02, 1.850276e-05, 2.809803e-02, 3.158834e-02, 1.181465e-01],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(2, 0, x, y, x[4:10], y[4:10]),
            [1.514718e-04, 3.397934e-02, 2.359908e-06, 4.506360e-03, 1.954899e-02, 5.468587e-02],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(2, 2, x, y, x[4:10], y[4:10]),
            [3.570080e-07, 6.165550e-03, 3.053570e-09, 7.683406e-04, 2.542728e-03, 1.305897e-02],
        )

        # ndarray
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(0, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.04, 0.41, 0.02, 0.16, 0.24, 0.60],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(0, 1, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.0033924100, 0.1228448321, 0.0005880316, 0.0585458775, 0.0795664636, 0.2168141323],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(1, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.0013478162, 0.1002286860, 0.0001938987, 0.0209359065, 0.0579585233, 0.1519399110],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(1, 1, x.values, y.values, x[4:10].values, y[4:10].values),
            [7.057350e-05, 3.485177e-02, 6.464252e-06, 8.244865e-03, 2.011508e-02, 5.933306e-02],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(0, 2, x.values, y.values, x[4:10].values, y[4:10].values),
            [3.227829e-04, 5.633341e-02, 1.850276e-05, 2.809803e-02, 3.158834e-02, 1.181465e-01],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(2, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [1.514718e-04, 3.397934e-02, 2.359908e-06, 4.506360e-03, 1.954899e-02, 5.468587e-02],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(2, 2, x.values, y.values, x[4:10].values, y[4:10].values),
            [3.570080e-07, 6.165550e-03, 3.053570e-09, 7.683406e-04, 2.542728e-03, 1.305897e-02],
        )

        # list
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(
                0, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.04, 0.41, 0.02, 0.16, 0.24, 0.60],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(
                0, 1, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.0033924100, 0.1228448321, 0.0005880316, 0.0585458775, 0.0795664636, 0.2168141323],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(
                1, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.0013478162, 0.1002286860, 0.0001938987, 0.0209359065, 0.0579585233, 0.1519399110],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(
                1, 1, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [7.057350e-05, 3.485177e-02, 6.464252e-06, 8.244865e-03, 2.011508e-02, 5.933306e-02],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(
                0, 2, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [3.227829e-04, 5.633341e-02, 1.850276e-05, 2.809803e-02, 3.158834e-02, 1.181465e-01],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(
                2, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [1.514718e-04, 3.397934e-02, 2.359908e-06, 4.506360e-03, 1.954899e-02, 5.468587e-02],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_LPM(
                2, 2, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [3.570080e-07, 6.165550e-03, 3.053570e-09, 7.683406e-04, 2.542728e-03, 1.305897e-02],
        )

    def test_D_UPM(self):
        z = self.load_default_data()
        x, y = z["x"], z["y"]
        # pandas
        self.assertAlmostEqual(Partial_Moments.D_UPM(0, 0, x, y, None, None), 0.25)
        self.assertAlmostEqual(Partial_Moments.D_UPM(0, 0, x, y, x.mean(), y.mean()), 0.25)
        self.assertAlmostEqual(Partial_Moments.D_UPM(0, 1, x, y, x.mean(), y.mean()), 0.05488706)
        self.assertAlmostEqual(Partial_Moments.D_UPM(1, 0, x, y, x.mean(), y.mean()), 0.05843498)
        self.assertAlmostEqual(Partial_Moments.D_UPM(1, 1, x, y, x.mean(), y.mean()), 0.01199175)
        self.assertAlmostEqual(Partial_Moments.D_UPM(0, 2, x, y, x.mean(), y.mean()), 0.01512857)
        self.assertAlmostEqual(Partial_Moments.D_UPM(2, 0, x, y, x.mean(), y.mean()), 0.01926167)
        self.assertAlmostEqual(Partial_Moments.D_UPM(2, 2, x, y, x.mean(), y.mean()), 0.0009941733)
        # list
        self.assertAlmostEqual(Partial_Moments.D_UPM(0, 0, list(x), list(y), None, None), 0.25)
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(0, 0, list(x), list(y), x.mean(), y.mean()), 0.25
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(0, 1, list(x), list(y), x.mean(), y.mean()), 0.05488706
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(1, 0, list(x), list(y), x.mean(), y.mean()), 0.05843498
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(1, 1, list(x), list(y), x.mean(), y.mean()), 0.01199175
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(0, 2, list(x), list(y), x.mean(), y.mean()), 0.01512857
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(2, 0, list(x), list(y), x.mean(), y.mean()), 0.01926167
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(2, 2, list(x), list(y), x.mean(), y.mean()), 0.0009941733
        )
        # ndarray
        self.assertAlmostEqual(Partial_Moments.D_UPM(0, 0, x.values, y.values, None, None), 0.25)
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(0, 0, x.values, y.values, x.mean(), y.mean()), 0.25
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(0, 1, x.values, y.values, x.mean(), y.mean()), 0.05488706
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(1, 0, x.values, y.values, x.mean(), y.mean()), 0.05843498
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(1, 1, x.values, y.values, x.mean(), y.mean()), 0.01199175
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(0, 2, x.values, y.values, x.mean(), y.mean()), 0.01512857
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(2, 0, x.values, y.values, x.mean(), y.mean()), 0.01926167
        )
        self.assertAlmostEqual(
            Partial_Moments.D_UPM(2, 2, x.values, y.values, x.mean(), y.mean()), 0.0009941733
        )

        # pandas
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(0, 0, x, y, x[4:10], y[4:10]),
            [0.71, 0.13, 0.74, 0.25, 0.20, 0.01],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(0, 1, x, y, x[4:10], y[4:10]),
            [0.3093049217, 0.0149820452, 0.2746970726, 0.0374187512, 0.0390594354, 0.0007309771],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(1, 0, x, y, x[4:10], y[4:10]),
            [0.214829569, 0.021394831, 0.380115487, 0.073392080, 0.041053231, 0.002714889],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(1, 1, x, y, x[4:10], y[4:10]),
            [0.0932774739, 0.0026736004, 0.1353388473, 0.0109792990, 0.0078132486, 0.0001984521],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(0, 2, x, y, x[4:10], y[4:10]),
            [1.672070e-01, 2.018018e-03, 1.252507e-01, 6.601065e-03, 9.106034e-03, 5.343275e-05],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(2, 0, x, y, x[4:10], y[4:10]),
            [0.091090436, 0.005160124, 0.234692911, 0.028744270, 0.011894273, 0.000737062],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(2, 2, x, y, x[4:10], y[4:10]),
            [2.150991e-02, 1.045718e-04, 3.685723e-02, 7.789023e-04, 5.293443e-04, 3.938325e-06],
        )

        # ndarray
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(0, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.71, 0.13, 0.74, 0.25, 0.20, 0.01],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(0, 1, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.3093049217, 0.0149820452, 0.2746970726, 0.0374187512, 0.0390594354, 0.0007309771],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(1, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.214829569, 0.021394831, 0.380115487, 0.073392080, 0.041053231, 0.002714889],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(1, 1, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.0932774739, 0.0026736004, 0.1353388473, 0.0109792990, 0.0078132486, 0.0001984521],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(0, 2, x.values, y.values, x[4:10].values, y[4:10].values),
            [1.672070e-01, 2.018018e-03, 1.252507e-01, 6.601065e-03, 9.106034e-03, 5.343275e-05],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(2, 0, x.values, y.values, x[4:10].values, y[4:10].values),
            [0.091090436, 0.005160124, 0.234692911, 0.028744270, 0.011894273, 0.000737062],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(2, 2, x.values, y.values, x[4:10].values, y[4:10].values),
            [2.150991e-02, 1.045718e-04, 3.685723e-02, 7.789023e-04, 5.293443e-04, 3.938325e-06],
        )
        # list
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(
                0, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.71, 0.13, 0.74, 0.25, 0.20, 0.01],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(
                0, 1, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.3093049217, 0.0149820452, 0.2746970726, 0.0374187512, 0.0390594354, 0.0007309771],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(
                1, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.214829569, 0.021394831, 0.380115487, 0.073392080, 0.041053231, 0.002714889],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(
                1, 1, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.0932774739, 0.0026736004, 0.1353388473, 0.0109792990, 0.0078132486, 0.0001984521],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(
                0, 2, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [1.672070e-01, 2.018018e-03, 1.252507e-01, 6.601065e-03, 9.106034e-03, 5.343275e-05],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(
                2, 0, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [0.091090436, 0.005160124, 0.234692911, 0.028744270, 0.011894273, 0.000737062],
        )
        self.assertAlmostEqualArray(
            Partial_Moments.D_UPM(
                2, 2, list(x.values), list(y.values), list(x[4:10].values), list(y[4:10].values)
            ),
            [2.150991e-02, 1.045718e-04, 3.685723e-02, 7.789023e-04, 5.293443e-04, 3.938325e-06],
        )

    def test_PM_matrix(self):
        z = self.load_default_data()
        for i in [True, False]:
            ret = Partial_Moments.PM_matrix(
                LPM_degree=1, UPM_degree=1, target="mean", variable=z if i else z.values
            )
            ret_default = self.load_default_PM_matrix_ret()
            if not i:
                ret_default["cupm"].columns = [0, 1, 2]
                ret_default["cupm"].index = [0, 1, 2]
                ret_default["dupm"].columns = [0, 1, 2]
                ret_default["dupm"].index = [0, 1, 2]
                ret_default["dlpm"].columns = [0, 1, 2]
                ret_default["dlpm"].index = [0, 1, 2]
                ret_default["clpm"].columns = [0, 1, 2]
                ret_default["clpm"].index = [0, 1, 2]
                ret_default["cov.matrix"].columns = [0, 1, 2]
                ret_default["cov.matrix"].index = [0, 1, 2]

            assert len(ret) == 5
            assert (
                "cupm" in ret
                and "dupm" in ret
                and "dlpm" in ret
                and "clpm" in ret
                and "cov.matrix" in ret
            )
            assert len(ret_default) == 5
            assert (
                "cupm" in ret_default
                and "dupm" in ret_default
                and "dlpm" in ret_default
                and "clpm" in ret_default
                and "cov.matrix" in ret_default
            )
            pd.testing.assert_frame_equal(
                ret["cupm"],
                ret_default["cupm"],
                check_exact=False,
                check_less_precise=6,
            )
            pd.testing.assert_frame_equal(
                ret["dupm"],
                ret_default["dupm"],
                check_exact=False,
                check_less_precise=6,
            )
            pd.testing.assert_frame_equal(
                ret["dlpm"],
                ret_default["dlpm"],
                check_exact=False,
                check_less_precise=6,
            )
            pd.testing.assert_frame_equal(
                ret["clpm"],
                ret_default["clpm"],
                check_exact=False,
                check_less_precise=6,
            )
            pd.testing.assert_frame_equal(
                ret["cov.matrix"],
                ret_default["cov.matrix"],
                check_exact=False,
                check_less_precise=6,
            )

    def test_LPM_ratio(self):
        x = self.load_default_data()["x"]
        self.assertAlmostEqual(Partial_Moments.LPM_ratio(degree=0, target="mean", variable=x), 0.49)
        self.assertAlmostEqual(
            Partial_Moments.LPM_ratio(degree=1, target="mean", variable=x), 0.5000000000000002
        )
        self.assertAlmostEqual(
            Partial_Moments.LPM_ratio(degree=2, target="mean", variable=x), 0.49720627
        )
        # list
        self.assertAlmostEqual(
            Partial_Moments.LPM_ratio(degree=0, target="mean", variable=list(x)), 0.49
        )
        self.assertAlmostEqual(
            Partial_Moments.LPM_ratio(degree=1, target="mean", variable=list(x)), 0.5000000000000002
        )
        self.assertAlmostEqual(
            Partial_Moments.LPM_ratio(degree=2, target="mean", variable=list(x)), 0.49720627
        )

    def test_UPM_ratio(self):
        x = self.load_default_data()["x"]
        self.assertAlmostEqual(Partial_Moments.UPM_ratio(degree=0, target="mean", variable=x), 0.51)
        self.assertAlmostEqual(
            Partial_Moments.UPM_ratio(degree=1, target="mean", variable=x), 0.4999999999999999
        )
        self.assertAlmostEqual(
            Partial_Moments.UPM_ratio(degree=2, target="mean", variable=x), 0.5027937984146681
        )
        # list
        self.assertAlmostEqual(
            Partial_Moments.UPM_ratio(degree=0, target="mean", variable=list(x)), 0.51
        )
        self.assertAlmostEqual(
            Partial_Moments.UPM_ratio(degree=1, target="mean", variable=list(x)), 0.4999999999999999
        )
        self.assertAlmostEqual(
            Partial_Moments.UPM_ratio(degree=2, target="mean", variable=list(x)), 0.5027937984146681
        )

    def test_NNS_PDF(self):
        print("TODO: Implement NNS_PDF")  # TODO

    def test_NNS_CDF(self):
        print("TODO: Implement NNS_CDF")  # TODO

    def load_default_PM_matrix_ret(self):
        return {
            "cupm": pd.DataFrame(
                [
                    [0.03027411, 0.01204606, 0.01241264],
                    [0.01204606, 0.03254594, 0.01268401],
                    [0.01241264, 0.01268401, 0.03535740],
                ],
                columns=["x", "y", "z"],
                index=["x", "y", "z"],
            ),
            "dupm": pd.DataFrame(
                [
                    [0.00000000, 0.01513793, 0.01041464],
                    [0.01199175, 0.00000000, 0.01854946],
                    [0.01178609, 0.01635133, 0.00000000],
                ],
                columns=["x", "y", "z"],
                index=["x", "y", "z"],
            ),
            "dlpm": pd.DataFrame(
                [
                    [0.00000000, 0.01199175, 0.01178609],
                    [0.01513793, 0.00000000, 0.01635133],
                    [0.01041464, 0.01854946, 0.00000000],
                ],
                columns=["x", "y", "z"],
                index=["x", "y", "z"],
            ),
            "clpm": pd.DataFrame(
                [
                    [0.02993767, 0.01058035, 0.01457109],
                    [0.01058035, 0.04096747, 0.01190255],
                    [0.01457109, 0.01190255, 0.04352933],
                ],
                columns=["x", "y", "z"],
                index=["x", "y", "z"],
            ),
            "cov.matrix": pd.DataFrame(
                [
                    [0.060211776, -0.00450327, 0.004783003],
                    [-0.004503270, 0.07351342, -0.010314231],
                    [0.004783003, -0.01031423, 0.078886725],
                ],
                columns=["x", "y", "z"],
                index=["x", "y", "z"],
            ),
        }

    def load_default_data(self):
        # R Code:
        # x <- c(0.6964691855978616, 0.28613933495037946, 0.2268514535642031, 0.5513147690828912, 0.7194689697855631, 0.42310646012446096, 0.9807641983846155, 0.6848297385848633, 0.48093190148436094, 0.3921175181941505, 0.3431780161508694, 0.7290497073840416, 0.4385722446796244, 0.05967789660956835, 0.3980442553304314, 0.7379954057320357, 0.18249173045349998, 0.17545175614749253, 0.5315513738418384, 0.5318275870968661, 0.6344009585513211, 0.8494317940777896, 0.7244553248606352, 0.6110235106775829, 0.7224433825702216, 0.3229589138531782, 0.3617886556223141, 0.22826323087895561, 0.29371404638882936, 0.6309761238544878, 0.09210493994507518, 0.43370117267952824, 0.4308627633296438, 0.4936850976503062, 0.425830290295828, 0.3122612229724653, 0.4263513069628082, 0.8933891631171348, 0.9441600182038796, 0.5018366758843366, 0.6239529517921112, 0.11561839507929572, 0.3172854818203209, 0.4148262119536318, 0.8663091578833659, 0.2504553653965067, 0.48303426426270435, 0.985559785610705, 0.5194851192598093, 0.6128945257629677, 0.12062866599032374, 0.8263408005068332, 0.6030601284109274, 0.5450680064664649, 0.3427638337743084, 0.3041207890271841, 0.4170222110247016, 0.6813007657927966, 0.8754568417951749, 0.5104223374780111, 0.6693137829622723, 0.5859365525622129, 0.6249035020955999, 0.6746890509878248, 0.8423424376202573, 0.08319498833243877, 0.7636828414433382, 0.243666374536874, 0.19422296057877086, 0.5724569574914731, 0.09571251661238711, 0.8853268262751396, 0.6272489720512687, 0.7234163581899548, 0.01612920669501683, 0.5944318794450425, 0.5567851923942887, 0.15895964414472274, 0.1530705151247731, 0.6955295287709109, 0.31876642638187636, 0.6919702955318197, 0.5543832497177721, 0.3889505741231446, 0.9251324896139861, 0.8416699969127163, 0.35739756668317624, 0.04359146379904055, 0.30476807341109746, 0.398185681917981, 0.7049588304513622, 0.9953584820340174, 0.35591486571745956, 0.7625478137854338, 0.5931769165622212, 0.6917017987001771, 0.15112745234808023, 0.39887629272615654, 0.24085589772362448, 0.34345601404832493)
        # y <- c(0.9290953494701337, 0.3001447577944899, 0.20646816984143224, 0.7712467017344186, 0.179207683251417, 0.7203696347073341, 0.2978651188274144, 0.6843301478774432, 0.6020774780838681, 0.8762070150459621, 0.7616916032270227, 0.6492402854114879, 0.3486146126960078, 0.5308900543442001, 0.31884300700035195, 0.6911215594221642, 0.7845248814489976, 0.8626202294885787, 0.4135895282244193, 0.8672153808700541, 0.8063467153755893, 0.7473209976914339, 0.08726848196743031, 0.023957638562143946, 0.050611236457549946, 0.4663642370285497, 0.4223981453920743, 0.474489623129292, 0.534186315014437, 0.7809131772951494, 0.8198754325768683, 0.7111791151322316, 0.49975889646204175, 0.5018097125708618, 0.7991356578408818, 0.03560152015693441, 0.921601798248779, 0.2733414160633679, 0.7824828518318679, 0.395582605302746, 0.48270235978971854, 0.5931259692926043, 0.2731798106977692, 0.8570159493264954, 0.5319561444631024, 0.1455315278392807, 0.6755524321238062, 0.27625359167650576, 0.2723010177649897, 0.6810977486565571, 0.9493047259244862, 0.807623816061548, 0.9451528088524095, 0.6402025296719795, 0.8258783277528565, 0.6300644920352498, 0.3893090155420259, 0.24163970305689175, 0.18402759570852467, 0.6031603131688895, 0.6566703304734626, 0.21177484928830181, 0.4359435889362071, 0.22965129132316398, 0.13087653733774363, 0.5989734941782344, 0.6688357426448118, 0.8093723729154483, 0.36209409565006223, 0.8513351315065957, 0.6551606487241549, 0.8554790691017261, 0.13596214615618918, 0.10883347378170816, 0.5448015917555307, 0.8728114143337533, 0.6621652225678912, 0.8701363950944805, 0.8453249339337617, 0.6283199211390311, 0.20690841095962864, 0.5176511518958, 0.6448515562981659, 0.42666354124364536, 0.9718610781333566, 0.24973274985042482, 0.05193778223157797, 0.6469719787522865, 0.3698392148054457, 0.8167218997483684, 0.710280810455504, 0.260673487453131, 0.4218711567383805, 0.793490082297006, 0.9398115107412777, 0.7625379749026492, 0.039750173274282985, 0.040137387046519146, 0.16805410857991787, 0.78433600580123)
        # z <- c(0.19999193561416084,0.6010279101158327,0.9788327513669298,0.8608964619298911,0.7601684508905298,0.12397506746787612,0.5394401401912896,0.8969279890952392,0.3839893553453263,0.5974293052436022,0.06516937735345008,0.15292545930437007,0.533669687225804,0.5430715864428796,0.8676197246411066,0.9298956526581725,0.6460088459791522,0.006548180072424414,0.6025139026895475,0.36841377074834125,0.44801794989436194,0.5048619249681798,0.4000809850582463,0.763740516980946,0.34083865579228434,0.5424284677884146,0.9587984735763967,0.5859672618993342,0.8422555318312421,0.5153219248350965,0.8358609378832195,0.787997995901579,0.2741451405223151,0.6444057500854898,0.02596405447571548,0.2797463018215405,0.10295252828980817,0.4354164588706081,0.26211152577662666,0.6998708543101617,0.37283691796585705,0.3227717548199931,0.1370286323274963,0.8070990185408966,0.7360223497043797,0.34991170542178995,0.9307716779643572,0.8134995545754865,0.32999762541477007,0.7009778150431946,0.9592132203954723,0.285109164298465,0.005404210183425628,0.7840965908154933,0.6534845192821737,0.22306404635944888,0.5599264352651063,0.9126415066887666,0.20749150526588522,0.769668024293192,0.7563728166813091,0.07231316109809582,0.44492578689736473,0.7211553193518122,0.8758657804680099,0.01890807847890197,0.11581293306751883,0.17126277092356368,0.8602241279326432,0.1371855605933343,0.5539492279716964,0.7663649743593801,0.19398868259207802,0.9569799507956978,0.24749785606958874,0.7610819645861326,0.567591973275089,0.7770410669374613,0.0733167994187951,0.845138899921509,0.867602249399254,0.32704688986389774,0.6298085331238098,0.019754547108759235,0.39450735124570824,0.5754821972966637,0.9506549185034494,0.6165089490060033,0.7456130158491189,0.8764042203221318,0.520223244392622,0.8123527374664891,0.8251058874981864,0.6842790562674221,0.4753605948189793,0.7491417107396956,0.4062763059892013,0.5738846393238041,0.32205678990789743,0.5765251949731963)
        return pd.DataFrame(
            {
                "x": [
                    0.6964691855978616,
                    0.28613933495037946,
                    0.2268514535642031,
                    0.5513147690828912,
                    0.7194689697855631,
                    0.42310646012446096,
                    0.9807641983846155,
                    0.6848297385848633,
                    0.48093190148436094,
                    0.3921175181941505,
                    0.3431780161508694,
                    0.7290497073840416,
                    0.4385722446796244,
                    0.05967789660956835,
                    0.3980442553304314,
                    0.7379954057320357,
                    0.18249173045349998,
                    0.17545175614749253,
                    0.5315513738418384,
                    0.5318275870968661,
                    0.6344009585513211,
                    0.8494317940777896,
                    0.7244553248606352,
                    0.6110235106775829,
                    0.7224433825702216,
                    0.3229589138531782,
                    0.3617886556223141,
                    0.22826323087895561,
                    0.29371404638882936,
                    0.6309761238544878,
                    0.09210493994507518,
                    0.43370117267952824,
                    0.4308627633296438,
                    0.4936850976503062,
                    0.425830290295828,
                    0.3122612229724653,
                    0.4263513069628082,
                    0.8933891631171348,
                    0.9441600182038796,
                    0.5018366758843366,
                    0.6239529517921112,
                    0.11561839507929572,
                    0.3172854818203209,
                    0.4148262119536318,
                    0.8663091578833659,
                    0.2504553653965067,
                    0.48303426426270435,
                    0.985559785610705,
                    0.5194851192598093,
                    0.6128945257629677,
                    0.12062866599032374,
                    0.8263408005068332,
                    0.6030601284109274,
                    0.5450680064664649,
                    0.3427638337743084,
                    0.3041207890271841,
                    0.4170222110247016,
                    0.6813007657927966,
                    0.8754568417951749,
                    0.5104223374780111,
                    0.6693137829622723,
                    0.5859365525622129,
                    0.6249035020955999,
                    0.6746890509878248,
                    0.8423424376202573,
                    0.08319498833243877,
                    0.7636828414433382,
                    0.243666374536874,
                    0.19422296057877086,
                    0.5724569574914731,
                    0.09571251661238711,
                    0.8853268262751396,
                    0.6272489720512687,
                    0.7234163581899548,
                    0.01612920669501683,
                    0.5944318794450425,
                    0.5567851923942887,
                    0.15895964414472274,
                    0.1530705151247731,
                    0.6955295287709109,
                    0.31876642638187636,
                    0.6919702955318197,
                    0.5543832497177721,
                    0.3889505741231446,
                    0.9251324896139861,
                    0.8416699969127163,
                    0.35739756668317624,
                    0.04359146379904055,
                    0.30476807341109746,
                    0.398185681917981,
                    0.7049588304513622,
                    0.9953584820340174,
                    0.35591486571745956,
                    0.7625478137854338,
                    0.5931769165622212,
                    0.6917017987001771,
                    0.15112745234808023,
                    0.39887629272615654,
                    0.24085589772362448,
                    0.34345601404832493,
                ],
                "y": [
                    0.9290953494701337,
                    0.3001447577944899,
                    0.20646816984143224,
                    0.7712467017344186,
                    0.179207683251417,
                    0.7203696347073341,
                    0.2978651188274144,
                    0.6843301478774432,
                    0.6020774780838681,
                    0.8762070150459621,
                    0.7616916032270227,
                    0.6492402854114879,
                    0.3486146126960078,
                    0.5308900543442001,
                    0.31884300700035195,
                    0.6911215594221642,
                    0.7845248814489976,
                    0.8626202294885787,
                    0.4135895282244193,
                    0.8672153808700541,
                    0.8063467153755893,
                    0.7473209976914339,
                    0.08726848196743031,
                    0.023957638562143946,
                    0.050611236457549946,
                    0.4663642370285497,
                    0.4223981453920743,
                    0.474489623129292,
                    0.534186315014437,
                    0.7809131772951494,
                    0.8198754325768683,
                    0.7111791151322316,
                    0.49975889646204175,
                    0.5018097125708618,
                    0.7991356578408818,
                    0.03560152015693441,
                    0.921601798248779,
                    0.2733414160633679,
                    0.7824828518318679,
                    0.395582605302746,
                    0.48270235978971854,
                    0.5931259692926043,
                    0.2731798106977692,
                    0.8570159493264954,
                    0.5319561444631024,
                    0.1455315278392807,
                    0.6755524321238062,
                    0.27625359167650576,
                    0.2723010177649897,
                    0.6810977486565571,
                    0.9493047259244862,
                    0.807623816061548,
                    0.9451528088524095,
                    0.6402025296719795,
                    0.8258783277528565,
                    0.6300644920352498,
                    0.3893090155420259,
                    0.24163970305689175,
                    0.18402759570852467,
                    0.6031603131688895,
                    0.6566703304734626,
                    0.21177484928830181,
                    0.4359435889362071,
                    0.22965129132316398,
                    0.13087653733774363,
                    0.5989734941782344,
                    0.6688357426448118,
                    0.8093723729154483,
                    0.36209409565006223,
                    0.8513351315065957,
                    0.6551606487241549,
                    0.8554790691017261,
                    0.13596214615618918,
                    0.10883347378170816,
                    0.5448015917555307,
                    0.8728114143337533,
                    0.6621652225678912,
                    0.8701363950944805,
                    0.8453249339337617,
                    0.6283199211390311,
                    0.20690841095962864,
                    0.5176511518958,
                    0.6448515562981659,
                    0.42666354124364536,
                    0.9718610781333566,
                    0.24973274985042482,
                    0.05193778223157797,
                    0.6469719787522865,
                    0.3698392148054457,
                    0.8167218997483684,
                    0.710280810455504,
                    0.260673487453131,
                    0.4218711567383805,
                    0.793490082297006,
                    0.9398115107412777,
                    0.7625379749026492,
                    0.039750173274282985,
                    0.040137387046519146,
                    0.16805410857991787,
                    0.78433600580123,
                ],
                "z": [
                    0.19999193561416084,
                    0.6010279101158327,
                    0.9788327513669298,
                    0.8608964619298911,
                    0.7601684508905298,
                    0.12397506746787612,
                    0.5394401401912896,
                    0.8969279890952392,
                    0.3839893553453263,
                    0.5974293052436022,
                    0.06516937735345008,
                    0.15292545930437007,
                    0.533669687225804,
                    0.5430715864428796,
                    0.8676197246411066,
                    0.9298956526581725,
                    0.6460088459791522,
                    0.006548180072424414,
                    0.6025139026895475,
                    0.36841377074834125,
                    0.44801794989436194,
                    0.5048619249681798,
                    0.4000809850582463,
                    0.763740516980946,
                    0.34083865579228434,
                    0.5424284677884146,
                    0.9587984735763967,
                    0.5859672618993342,
                    0.8422555318312421,
                    0.5153219248350965,
                    0.8358609378832195,
                    0.787997995901579,
                    0.2741451405223151,
                    0.6444057500854898,
                    0.02596405447571548,
                    0.2797463018215405,
                    0.10295252828980817,
                    0.4354164588706081,
                    0.26211152577662666,
                    0.6998708543101617,
                    0.37283691796585705,
                    0.3227717548199931,
                    0.1370286323274963,
                    0.8070990185408966,
                    0.7360223497043797,
                    0.34991170542178995,
                    0.9307716779643572,
                    0.8134995545754865,
                    0.32999762541477007,
                    0.7009778150431946,
                    0.9592132203954723,
                    0.285109164298465,
                    0.005404210183425628,
                    0.7840965908154933,
                    0.6534845192821737,
                    0.22306404635944888,
                    0.5599264352651063,
                    0.9126415066887666,
                    0.20749150526588522,
                    0.769668024293192,
                    0.7563728166813091,
                    0.07231316109809582,
                    0.44492578689736473,
                    0.7211553193518122,
                    0.8758657804680099,
                    0.01890807847890197,
                    0.11581293306751883,
                    0.17126277092356368,
                    0.8602241279326432,
                    0.1371855605933343,
                    0.5539492279716964,
                    0.7663649743593801,
                    0.19398868259207802,
                    0.9569799507956978,
                    0.24749785606958874,
                    0.7610819645861326,
                    0.567591973275089,
                    0.7770410669374613,
                    0.0733167994187951,
                    0.845138899921509,
                    0.867602249399254,
                    0.32704688986389774,
                    0.6298085331238098,
                    0.019754547108759235,
                    0.39450735124570824,
                    0.5754821972966637,
                    0.9506549185034494,
                    0.6165089490060033,
                    0.7456130158491189,
                    0.8764042203221318,
                    0.520223244392622,
                    0.8123527374664891,
                    0.8251058874981864,
                    0.6842790562674221,
                    0.4753605948189793,
                    0.7491417107396956,
                    0.4062763059892013,
                    0.5738846393238041,
                    0.32205678990789743,
                    0.5765251949731963,
                ],
            }
        )
