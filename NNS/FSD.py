# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .Partial_Moments import LPM_ratio
import matplotlib.pyplot as plt

# TODO: test / matplotlib
def NNS_FSD(x: pd.Series, y: pd.Series, type_cdf: str = "discrete", use_plot: bool = True) -> str:
    r"""
    NNS FSD Test

    Bi-directional test of first degree stochastic dominance using lower partial moments.

    @param x a numeric vector.
    @param y a numeric vector.
    @param type options: ("discrete", "continuous"); \code{"discrete"} (default) selects the type of CDF.
    @return Returns one of the following FSD results: \code{"X FSD Y"}, \code{"Y FSD X"}, or \code{"NO FSD EXISTS"}.
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2016) "LPM Density Functions for the Computation of the SD Efficient Set." Journal of Mathematical Finance, 6, 105-126. \url{http://www.scirp.org/Journal/PaperInformation.aspx?PaperID=63817}.

    Viole, F. (2017) "A Note on Stochastic Dominance." \url{https://ssrn.com/abstract=3002675}.
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    NNS.FSD(x, y)
    @export
    """
    type_cdf = type_cdf.lower()
    if type_cdf not in ["discrete", "continuous"]:
        raise Exception("type needs to be either 'discrete' or 'continuous'")

    x_sort = np.sort(x)
    y_sort = np.sort(y)
    Combined_sort = np.unique(np.append(x_sort, y_sort))
    ## Indicator function ***for all values of x and y*** as the continuous CDF target
    degree = 0 if type_cdf == "discrete" else 1
    LPM_x_sort = LPM_ratio(degree, Combined_sort, x)
    LPM_y_sort = LPM_ratio(degree, Combined_sort, y)

    x_fsd_y = np.any(LPM_x_sort > LPM_y_sort)
    y_fsd_x = np.any(LPM_y_sort > LPM_x_sort)

    if use_plot:
        plt.title("FSD")
        plt.ylabel("Area of Cumulative Distribution")
        plt.plot(Combined_sort, LPM_x_sort, label="<Combined Sort> vs <LPM X Sort>")
        plt.plot(Combined_sort, LPM_y_sort, label="<Combined Sort> vs <LPM Y Sort>")
        plt.legend()
        # plot(
        #    Combined_sort,
        #    LPM_x_sort,
        #    type="l",
        #    lwd=3,
        #    col="red",
        #    main="FSD",
        #    ylab="Probability of Cumulative Distribution",
        #    ylim=c(0, 1),
        # )
        # lines(Combined_sort, LPM_y_sort, type="l", lwd=3, col="blue")
        # legend("topleft", c("X", "Y"), lwd=10, col=c("red", "blue"))

    ## Verification of ***0 instances*** of CDFx > CDFy, and conversely of CDFy > CDFx
    if (not x_fsd_y) and (x_sort[0] >= y_sort[0]) and (not np.equal(LPM_x_sort, LPM_y_sort).all()):
        return "X FSD Y"
    if (not y_fsd_x) and (y_sort[0] >= x_sort[0]) and (not np.equal(LPM_x_sort, LPM_y_sort).all()):
        return "Y FSD X"
    return "NO FSD EXISTS"


__all__ = ["NNS_FSD"]
