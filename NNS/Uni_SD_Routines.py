# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .Partial_Moments import LPM, UPM


def NNS_FSD_uni(x: pd.Series, y: pd.set_option, type_test: str = "discrete") -> int:
    r"""
    NNS FSD Test uni-directional

    Uni-directional test of first degree stochastic dominance using lower partial moments used in SD Efficient Set routine.

    @param x a numeric vector.
    @param y a numeric vector.
    @param type_test: ("discrete", "continuous"); \code{"discrete"} (default) selects the type of CDF.
    @return Returns (1) if \code{"X FSD Y"}, else (0).
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2016) "LPM Density Functions for the Computation of the SD Efficient Set." Journal of Mathematical Finance, 6, 105-126. \url{http://www.scirp.org/Journal/PaperInformation.aspx?PaperID=63817}.

    Viole, F. (2017) "A Note on Stochastic Dominance." \url{https://ssrn.com/abstract=3002675}.
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    NNS.FSD.uni(x, y)
    @export
    """
    type_test = type_test.lower()

    if type_test not in ["discrete", "continuous"]:
        raise ValueError("type needs to be either discrete or continuous")

    if y.min() > x.min():
        return 0

    x_sort = x.sort_values(ascending=True)  # TODO: , decreasing = FALSE)
    y_sort = y.sort_values(ascending=True)

    Combined_sort = x_sort.append(y_sort).sort_values(ascending=True)  # TODO:, decreasing = FALSE)

    if type == "discrete":
        degree = 0
    else:
        degree = 1

    L_x = LPM(degree, Combined_sort, x)
    LPM_x_sort = L_x / (UPM(degree, Combined_sort, x) + L_x)
    L_y = LPM(degree, Combined_sort, y)
    LPM_y_sort = L_y / (UPM(degree, Combined_sort, y) + L_y)

    x_fsd_y = pd.Series(LPM_x_sort > LPM_y_sort).any()

    if (not x_fsd_y) and (x.min() >= y.min()) and (not pd.Series(LPM_x_sort).equals(LPM_y_sort)):
        return 1
    return 0


def NNS_SSD_uni(x: pd.Series, y: pd.Series) -> int:
    r"""
    NNS SSD Test uni-directional

    Uni-directional test of second degree stochastic dominance using lower partial moments used in SD Efficient Set routine.
    @param x a numeric vector.
    @param y a numeric vector.
    @return Returns (1) if \code{"X SSD Y"}, else (0).
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2016) "LPM Density Functions for the Computation of the SD Efficient Set."
        Journal of Mathematical Finance, 6, 105-126. \url{http://www.scirp.org/Journal/PaperInformation.aspx?PaperID=63817}.
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    NNS.SSD.uni(x, y)
    @export
    """
    if y.min() > x.min() or y.mean() > x.mean():
        return 0
    x_sort = x.sort_values()  # TODO:, decreasing = FALSE)
    y_sort = y.sort_values()  # TODO:, decreasing = FALSE)

    Combined_sort = np.unique(x_sort.append(y_sort).values)

    LPM_x_sort = LPM(1, Combined_sort, x)
    LPM_y_sort = LPM(1, Combined_sort, y)

    x_ssd_y = pd.Series(LPM_x_sort > LPM_y_sort).any()

    if (not x_ssd_y) and (x.min() >= y.min()) and (not pd.Series(LPM_x_sort).equals(LPM_y_sort)):
        return 1
    return 0


def NNS_TSD_uni(x: pd.Series, y: pd.Series) -> int:
    r"""
    NNS TSD Test uni-directional

    Uni-directional test of third degree stochastic dominance using lower partial moments used in SD Efficient Set routine.
    @param x a numeric vector.
    @param y a numeric vector.
    @return Returns (1) if \code{"X TSD Y"}, else (0).
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2016) "LPM Density Functions for the Computation of the SD Efficient Set."
        Journal of Mathematical Finance, 6, 105-126. \url{http://www.scirp.org/Journal/PaperInformation.aspx?PaperID=63817}.
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    NNS.TSD.uni(x, y)
    @export
    """
    if y.min() > x.min() or y.mean() > x.mean():
        return 0
    x_sort = x.sort_values(ascending=True)  # TODO:, decreasing = FALSE)
    y_sort = y.sort_values(ascending=True)  # TODO:, decreasing = FALSE)

    Combined_sort = np.unique(x_sort.append(y_sort).values)  # TODO:, decreasing = FALSE)

    LPM_x_sort = LPM(2, Combined_sort, x)
    LPM_y_sort = LPM(2, Combined_sort, y)

    x_tsd_y = pd.Series(LPM_x_sort > LPM_y_sort).any()

    if (not x_tsd_y) and (x.min() >= y.min()) and (not pd.Series(LPM_x_sort).equals(LPM_y_sort)):
        return 1
    return 0
