# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd

import NNS


class TestNNS_term_matrix(unittest.TestCase):
    COMPARISON_PRECISION = 7

    def test_NNS_term_matrix(self):
        x = self.load_default_data()
        ret = NNS.NNS_term_matrix(x["df"], names=True)
        assert len(ret) == 2
        assert "IV" in ret and "DV" in ret
        pd.testing.assert_frame_equal(
            ret["IV"],
            x["ok_IV"],
            check_exact=False,
            check_less_precise=6,
        )
        self.assertListEqual(list(ret["DV"]), list(x["ok_DV"]))
        # OOS
        ret = NNS.NNS_term_matrix(x["df"], names=True, oos=x["OOS"])
        assert len(ret) == 3
        assert "IV" in ret and "DV" in ret and "OOS" in ret
        pd.testing.assert_frame_equal(
            ret["IV"],
            x["ok_IV"],
            check_exact=False,
            check_less_precise=6,
        )
        self.assertListEqual(list(ret["DV"]), list(x["ok_DV"]))
        pd.testing.assert_frame_equal(
            ret["OOS"],
            x["ok_OOS"],
            check_exact=False,
            check_less_precise=6,
        )

    def load_default_data(self):
        return {
            "df": pd.DataFrame({"X1": ["sunny windy", "rainy cloudy"], "X2": ["1", "-1"]}),
            "ok_IV": pd.DataFrame(
                {
                    "sunny": [1, 0],
                    "windy": [1, 0],
                    "rainy": [0, 1],
                    "cloudy": [0, 1],
                }
            ),
            "ok_DV": np.array(["1", "-1"]),
            "OOS": pd.Series(["sunny"]),
            "ok_OOS": pd.DataFrame(
                {
                    "sunny": [1],
                    "windy": [0],
                    "rainy": [0],
                    "cloudy": [0],
                }
            ),
        }
