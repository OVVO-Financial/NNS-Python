# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd

from NNS.Numerical_Differentiation import NNS_diff


class TestNumerical_Differentiation(unittest.TestCase):
    COMPARISON_PRECISION = 7

    def test_Numerical_Differentiation(self):
        r = NNS_diff(f=lambda x: x ** 2, point=5, h=0.1, tol=1e-10, digits=12, print_trace=True)
