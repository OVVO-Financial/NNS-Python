# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .Partial_Moments import LPM, LPM_ratio
import scipy.optimize

# TODO: TESTS, tdigest::tdigest from R:
def LPM_VaR(percentile: [float, int], degree: [float, int, str, None], x: pd.Series) -> float:
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

    #percentile <- pmax(pmin(percentile, 1), 0)
    percentile = np.maximum(np.minimum(percentile, 1), 0)
    l = len(x)
    if(degree == 0):
        # TODO: tdigest::tdigest from R:
        # Create a new t-Digest histogram from a vector
        # Description
        # The t-Digest construction algorithm, by Dunning et al., uses a variant of 1-dimensional k-means
        # clustering to produce a very compact data structure that allows accurate estimation of quantiles.
        # This t-Digest data structure can be used to estimate quantiles, compute other rank statistics or
        # even to estimate related measures like trimmed means. The advantage of the t-Digest over previous
        # digests for this purpose is that the t-Digest handles data with full floating point resolution.
        # The accuracy of quantile estimates produced by t-Digests can be orders of magnitude more accurate
        # than those produced by previous digest algorithms. Methods are provided to create and update t-Digests
        # and retrieve quantiles from the accumulated distributions.
        #
        # Usage
        # tdigest(vec, compression = 100)
        #
        # ## S3 method for class 'tdigest'
        # print(x, ...)
        # Arguments
        # vec - vector (will be converted to double if not already double). NOTE that this is ALTREP-aware and
        #               will not materialize the passed-in object in order to add the values to the t-Digest.)
        #
        # compression -  the input compression value; should be >= 1.0; this will control how aggressively the
        #               t-Digest compresses data together. The original t-Digest paper suggests using a value of
        #               100 for a good balance between precision and efficiency. It will land at very small
        #               (think like 1e-6 percentile points) errors at extreme points in the distribution, and
        #               compression ratios of around 500 for large data sets (~1 million datapoints).
        #               Defaults to 100.
        #
        # x	-  tdigest object
        #
        # ...	-  unused
        #
        # Value -  a tdigest object
        #
        # References - Computing Extremely Accurate Quantiles Using t-Digests
        #
        # Examples
        # set.seed(1492)
        # x <- sample(0:100, 1000000, replace = TRUE)
        # td <- tdigest(x, 1000)
        # tquantile(td, c(0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1))
        # quantile(td)


        td = tdigest::tdigest(x, compression = max(100, np.log(l, 10)*100))
        try:
            q = tdigest::tquantile(td, percentile)
        except Exception:
            q = x.quantile(percentile)
        return q
    def _func(b):
        return abs(LPM_ratio(degree, b, x) - percentile)
    ret = scipy.optimize.minimize(_func, x.mean(), bounds=(x.min(), x.max()))
    return ret.x
    #return(optimize(_func, c(min(x),max(x)))$minimum)


def UPM_VaR(percentile: [float, int], degree: [float, int, str, None], x: pd.Series) -> float:
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

    percentile = np.maximum(np.minimum(percentile, 1), 0)
    l = len(x)
    if(degree==0):
        # TODO: tdigest::tdigest from R:
        td = tdigest::tdigest(x, compression = max(100, np.log(l,10)*100))
        try:
            q = tdigest::tquantile(td, 1 - percentile)
        except Exception:
            q = x.quantile(1 - percentile)
        return q

    def _func(b):
        return abs(LPM_ratio(degree, b, x) - (1-percentile))
    ret = scipy.optimize.minimize(_func, x.mean(), bounds=(x.min(), x.max()))


    ret = scipy.optimize.minimize(_func, x.mean(), bounds=(x.min(), x.max()))
    return ret.x
    #return(optimize(_func, c(min(x),max(x)))$minimum)

