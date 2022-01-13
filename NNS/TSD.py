# -*- coding: utf-8 -*-

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd

from .Partial_Moments import LPM


# TODO: TEST / implement matplotlib graph
def NNS_TSD(x: pd.Series, y: pd.Series, use_plot: bool = True) -> str:
    r"""
    NNS TSD Test

    Bi-directional test of third degree stochastic dominance using lower partial moments.

    @param x a numeric vector.
    @param y a numeric vector.
    @return Returns one of the following TSD results: \code{"X TSD Y"}, \code{"Y TSD X"}, or \code{"NO TSD EXISTS"}.
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2016) "LPM Density Functions for the Computation of the SD Efficient Set." Journal of Mathematical Finance, 6, 105-126. \url{http://www.scirp.org/Journal/PaperInformation.aspx?PaperID=63817}.
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    NNS.TSD(x, y)
    @export
    """
    x_sort = np.sort(x)
    y_sort = np.sort(y)
    Combined_sort = np.unique(np.append(x_sort, y_sort))
    LPM_x_sort = LPM(2, Combined_sort, x)
    LPM_y_sort = LPM(2, Combined_sort, y)
    x_tsd_y = np.any(LPM_x_sort > LPM_y_sort)
    y_tsd_x = np.any(LPM_y_sort > LPM_x_sort)
    if use_plot and plt is not None:
        plt.title("TSD")
        plt.ylabel("Area of Cumulative Distribution")
        plt.plot(Combined_sort, LPM_x_sort, label="<Combined Sort> vs <LPM X Sort>")
        plt.plot(Combined_sort, LPM_y_sort, label="<Combined Sort> vs <LPM Y Sort>")
        plt.legend()
        # plot(
        #    LPM_x_sort,
        #    type="l",
        #    lwd=3,
        #    col="red",
        #    main="TSD",
        #    ylab="Area of Cumulative Distribution",
        #    ylim=c(min(c(LPM_y_sort, LPM_x_sort)), max(c(LPM_y_sort, LPM_x_sort))),
        # )
        # lines(LPM_y_sort, type="l", lwd=3, col="blue")
        # legend("topleft", c("X", "Y"), lwd=10, col=c("red", "blue"))
    if (
        (not x_tsd_y)
        and (x_sort[0] >= y_sort[0])
        and (x.mean() >= y.mean())
        and (not np.equal(LPM_x_sort, LPM_y_sort).all())
    ):
        return "X TSD Y"
    if (
        (not y_tsd_x)
        and (y_sort[0] >= x_sort[0])
        and (y.mean() >= x.mean())
        and (not np.equal(LPM_x_sort, LPM_y_sort).all())
    ):
        return "Y TSD X"
    return "NO TSD EXISTS"


__all__ = ["NNS_TSD"]
