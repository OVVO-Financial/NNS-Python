# -*- coding: utf-8 -*-
import unittest
import numpy as np
import pandas as pd

from NNS.Numerical_Differentiation import NNS_diff


class TestNumerical_Differentiation(unittest.TestCase):
    COMPARISON_PRECISION = 7

    def test_Numerical_Differentiation(self):
        # TODO
        r = NNS_diff(f=lambda x: x ** 2, point=5, h=0.1, tol=1e-10, digits=12, print_trace=True)
        ret_ok = {
            "Value of f(x) at point": 2.50e01,
            "Final y-intercept (B)": -2.50e01,
            "DERIVATIVE": 1.00e01,
            "Inferred h": 9.30e-11,
            "iterations": 3.10e01,
            "f(x-h)": 9.90e00,
            "f(x+h)": 1.01e01,
            "Averaged Finite Step Initial h": 1.00e01,
            "Inferred h.f(x-h)": 1.00e01,
            "Inferred h.f(x+h)": 1.00e01,
            "Inferred h Averaged Finite Step": 1.00e01,
            "Complex Step Derivative (Initial h)": 1.00e01,
        }
        self.assertDictEqual(r, ret_ok)
        print(r)
