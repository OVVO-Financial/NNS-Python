# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .Partial_Moments import LPM, UPM


def NNS_FSD_uni(
    x: [pd.Series, np.ndarray], y: [pd.Series, np.ndarray], type_test: str = "discrete"
) -> int:
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

    if np.min(y) > np.min(x):
        return 0

    x_sort = np.sort(x)
    y_sort = np.sort(y)

    Combined_sort = np.unique(np.append(x_sort, y_sort))
    degree = 0 if type_test == "discrete" else 1
    L_x = LPM(degree, Combined_sort, x)
    LPM_x_sort = L_x / (UPM(degree, Combined_sort, x) + L_x)
    L_y = LPM(degree, Combined_sort, y)
    LPM_y_sort = L_y / (UPM(degree, Combined_sort, y) + L_y)
    x_fsd_y = np.any(LPM_x_sort > LPM_y_sort)

    if (not x_fsd_y) and (x_sort[0] >= y_sort[0]) and (not np.equal(LPM_x_sort, LPM_y_sort).all()):
        return 1
    return 0


def NNS_SSD_uni(x: [pd.Series, np.ndarray], y: [pd.Series, np.ndarray]) -> int:
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
    if np.min(y) > np.min(x) or np.mean(y) > np.mean(x):
        return 0
    x_sort = np.sort(x)
    y_sort = np.sort(y)
    Combined_sort = np.unique(np.append(x_sort, y_sort))
    LPM_x_sort = LPM(1, Combined_sort, x)
    LPM_y_sort = LPM(1, Combined_sort, y)
    x_ssd_y = np.any(LPM_x_sort > LPM_y_sort)
    if (not x_ssd_y) and (x_sort[0] >= y_sort[0]) and (not np.equal(LPM_x_sort, LPM_y_sort).all()):
        return 1
    return 0


def NNS_TSD_uni(x: [pd.Series, np.ndarray], y: [pd.Series, np.ndarray]) -> int:
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
    if np.min(y) > np.min(x) or np.mean(y) > np.mean(x):
        return 0
    x_sort = np.sort(x)
    y_sort = np.sort(y)
    Combined_sort = np.unique(np.append(x_sort, y_sort))
    LPM_x_sort = LPM(2, Combined_sort, x)
    LPM_y_sort = LPM(2, Combined_sort, y)
    x_tsd_y = np.any(LPM_x_sort > LPM_y_sort)

    if (not x_tsd_y) and (x_sort[0] >= y_sort[0]) and (not np.equal(LPM_x_sort, LPM_y_sort).all()):
        return 1
    return 0


__all__ = [
    "NNS_FSD_uni",
    "NNS_SSD_uni",
    "NNS_TSD_uni"
]
