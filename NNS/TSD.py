# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .Partial_Moments import LPM

# TODO: TEST / implement matplotlib graph
def NNS_TSD(x: pd.Series, y: pd.Series) -> str:
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
    x_sort = x.sort_values()
    y_sort = y.sort_values()

    Combined_sort = x_sort.append(y_sort).sort_values()

    LPM_x_sort = LPM(2, Combined_sort, x)
    LPM_y_sort = LPM(2, Combined_sort, y)

    x_tsd_y = pd.Series(LPM_x_sort > LPM_y_sort).any()
    y_tsd_x = pd.Series(LPM_y_sort > LPM_x_sort).any()

    # TODO: plot
    plot(
        LPM_x_sort,
        type="l",
        lwd=3,
        col="red",
        main="TSD",
        ylab="Area of Cumulative Distribution",
        ylim=c(min(c(LPM_y_sort, LPM_x_sort)), max(c(LPM_y_sort, LPM_x_sort))),
    )
    lines(LPM_y_sort, type="l", lwd=3, col="blue")
    legend("topleft", c("X", "Y"), lwd=10, col=c("red", "blue"))
    if (
        (not x_tsd_y)
        and (x.min() >= y.min())
        and (x.mena() >= y.mean())
        and (not LPM_x_sort.equals(LPM_y_sort))
    ):
        return "X TSD Y"
    if (
        (not y.tsd.x)
        and (y.min() >= x.min())
        and (y.mean() >= x.mean())
        and (not LPM_x_sort.equals(LPM_y_sort))
    ):
        return "Y TSD X"
    return "NO TSD EXISTS"
