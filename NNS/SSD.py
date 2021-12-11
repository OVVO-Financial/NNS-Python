# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .Partial_Moments import LPM

# TODO: test / matplotlib graph
def NNS_SSD(x: pd.Series, y: pd.Series, use_plot: bool = True) -> str:
    r"""
    NNS SSD Test

    Bi-directional test of second degree stochastic dominance using lower partial moments.

    @param x a numeric vector.
    @param y a numeric vector.
    @return Returns one of the following SSD results: \code{"X SSD Y"}, \code{"Y SSD X"}, or \code{"NO SSD EXISTS"}.
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2016) "LPM Density Functions for the Computation of the SD Efficient Set." Journal of Mathematical Finance, 6, 105-126. \url{http://www.scirp.org/Journal/PaperInformation.aspx?PaperID=63817}.
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    NNS.SSD(x, y)
    @export
    """
    x_sort = np.sort(x)
    y_sort = np.sort(y)

    Combined_sort = np.unique(np.append(x_sort, y_sort))
    LPM_x_sort = LPM(1, Combined_sort, x)
    LPM_y_sort = LPM(1, Combined_sort, y)
    x_ssd_y = np.any(LPM_x_sort > LPM_y_sort)
    y_ssd_x = np.any(LPM_y_sort > LPM_x_sort)
    if use_plot:
        plt.title("SSD")
        plt.ylabel("Area of Cumulative Distribution")
        plt.plot(Combined_sort, LPM_x_sort, label="<Combined Sort> vs <LPM X Sort>")
        plt.plot(Combined_sort, LPM_y_sort, label="<Combined Sort> vs <LPM Y Sort>")
        plt.legend()
        # plot(
        #    Combined_sort,
        #    LPM_x_sort,
        #    type = "l",
        #    lwd = 3,
        #    col = "red",
        #    main = "SSD",
        #    ylab = "Area of Cumulative Distribution",
        #    ylim = c(min(c(LPM_y_sort, LPM_x_sort)), max(c(LPM_y_sort, LPM_x_sort)))
        # )
        # lines(Combined_sort, LPM_y_sort, type = "l", lwd = 3,col = "blue")
        # legend("topleft", c("X", "Y"), lwd = 10, col = c("red", "blue"))

    if (
        (not x_ssd_y)
        and (x_sort[0] >= y_sort[0])
        and (x.mean() >= y.mean())
        and (not np.equal(LPM_x_sort, LPM_y_sort).all())
    ):
        return "X SSD Y"
    if (
        (not y_ssd_x)
        and (y_sort[0] >= x_sort[0])
        and (y.mean() >= x.mean())
        and (not np.equal(LPM_x_sort, LPM_y_sort).all())
    ):
        return "Y SSD X"
    return "NO SSD EXISTS"
