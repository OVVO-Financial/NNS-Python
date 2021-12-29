# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tdigest
from .Partial_Moments import LPM, LPM_ratio
import scipy.optimize


def _LPM_VaR(
    percentile: [float, int], degree: [float, int, str, None], x: [pd.Series, np.ndarray]
) -> float:
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max:
        return x_min

    percentile = max(min(percentile, 1.0), 0.0)
    if percentile <= 0:
        return x_min
    elif percentile >= 1:
        return x_max
    if degree == 0:
        # td = tdigest.TDigest()
        # td.batch_update(x)
        # return td.percentile(percentile)
        return np.quantile(x, percentile, interpolation="linear")
    # degree > 0
    x0 = np.mean(x)
    x_range = x_max - x_min
    x1 = x_min if x_range == 0 else (x_range * percentile + x_min)

    def _func(b):
        return LPM_ratio(degree, max(x_min, min(x_max, b)), x) - percentile

    ret = scipy.optimize.root_scalar(
        f=_func,
        # method='bisect',
        bracket=[x_min, x_max],
        x0=x0,
        x1=x1,
    )
    if not ret.converged:
        raise Exception(f"Root find didn't converged: {ret}")
    return ret.root


_vec_LPM_VaR = np.vectorize(_LPM_VaR, excluded=["degree", "x"])


def LPM_VaR(
    percentile: [float, int, np.array, pd.Series, list],
    degree: [float, int, str, None],
    x: [pd.Series, np.ndarray],
) -> [float, np.array]:
    r"""
    LPM VaR

    Generates a value at risk (VaR) quantile based on the Lower Partial Moment ratio.

    @param percentile numeric [0, 1]; The percentile for left-tail VaR (vectorized).
    @param degree integer; \code{(degree = 0)} for discrete distributions, \code{(degree = 1)} for continuous distributions.
    @param x a numeric vector.
    @return Returns a numeric value representing the point at which \code{"percentile"} of the area of \code{x} is below.
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100)

    ## For 5% quantile, left-tail
    LPM.VaR(0.05, 0, x)
    @export
    """
    func = _LPM_VaR
    if isinstance(percentile, (np.ndarray, pd.Series, list)):
        func = _vec_LPM_VaR
    return func(percentile=percentile, degree=degree, x=x)


def _UPM_VaR(
    percentile: [float, int],
    degree: [float, int, str, None],
    x: [pd.Series, np.array],
) -> float:
    x_min = np.min(x)
    x_max = np.max(x)
    if x_min == x_max:
        return x_min

    percentile = max(min(percentile, 1.0), 0.0)
    if percentile <= 0:
        return x_max
    elif percentile >= 1:
        return x_min

    if degree == 0:
        # td = tdigest.TDigest()
        # td.batch_update(x)
        # return td.percentile(percentile)
        return np.quantile(x, 1 - percentile, interpolation="linear")

    # degree > 0
    x0 = np.mean(x)
    x_range = x_max - x_min
    x1 = x_min if x_range == 0 else (x_range * (1 - percentile) + x_min)

    def _func(b):
        return LPM_ratio(degree, max(x_min, min(x_max, b)), x) - (1 - percentile)

    ret = scipy.optimize.root_scalar(
        f=_func,
        # method='bisect',
        bracket=[x_min, x_max],
        x0=x0,
        x1=x1,
    )
    if not ret.converged:
        raise Exception(f"Root find didn't converged: {ret}")
    return ret.root


_vec_UPM_VaR = np.vectorize(_UPM_VaR, excluded=["degree", "x"])


def UPM_VaR(
    percentile: [float, int, np.array, pd.Series, list],
    degree: [float, int, str, None],
    x: [pd.Series, np.array],
) -> [float, np.array]:
    r"""
    UPM VaR

    Generates an upside value at risk (VaR) quantile based on the Upper Partial Moment ratio
    @param percentile numeric [0, 1]; The percentile for right-tail VaR (vectorized).
    @param degree integer; \code{(degree = 0)} for discrete distributions, \code{(degree = 1)} for continuous distributions.
    @param x a numeric vector.
    @return Returns a numeric value representing the point at which \code{"percentile"} of the area of \code{x} is above.
    @examples
    set.seed(123)
    x <- rnorm(100)

    ## For 5% quantile, right-tail
    UPM.VaR(0.05, 0, x)
    @export
    """
    func = _UPM_VaR
    if isinstance(percentile, (np.ndarray, pd.Series, list)):
        func = _vec_UPM_VaR
    return func(percentile=percentile, degree=degree, x=x)


__all__ = [
    "LPM_VaR",
    "UPM_VaR",
]
