# -*- coding: utf-8 -*-
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .Partial_Moments import PM_matrix

_tmp = Axes3D, Poly3DCollection  # just o avoid import lint errors
del _tmp

# https://stackoverflow.com/questions/49277753/python-matplotlib-plotting-cuboids
def cuboid_data(o, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
        [o[0], o[0] + l, o[0] + l, o[0], o[0]],
    ]
    y = [
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1] + w, o[1] + w, o[1]],
        [o[1], o[1], o[1], o[1], o[1]],
        [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w],
    ]
    z = [
        [o[2], o[2], o[2], o[2], o[2]],
        [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
        [o[2], o[2], o[2] + h, o[2] + h, o[2]],
    ]
    return np.array(x), np.array(y), np.array(z)


def plotCubeAt(pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
    # Plotting a cube element at position pos
    if ax is not None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)


def NNS_copula(
    x: [pd.DataFrame, pd.Series, np.ndarray],
    continuous: bool = True,
    plot: bool = True,
    independence_overlay: bool = False,
) -> [float, None]:
    r"""NNS Co-Partial Moments Higher Dimension Dependence

    Determines higher dimension dependence coefficients based on co-partial moment matrices ratios.

    @param x a numeric matrix or data frame.
    @param continuous logical; \code{TRUE} (default) Generates a continuous measure using degree 1 \link{PM.matrix}, while discrete \code{FALSE} uses degree 0 \link{PM.matrix}.
    @param plot logical; \code{FALSE} (default) Generates a 3d scatter plot with regression points using \link{plot3d}.
    @param independence_overlay logical; \code{FALSE} (default) Creates and overlays independent \link{Co.LPM} and \link{Co.UPM} regions to visually reference the difference in dependence from the data.frame of variables being analyzed.  Under independence, the light green and red shaded areas would be occupied by green and red data points respectively.

    @return Returns a multivariate dependence value [0,1].

    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. (2016) "Beyond Correlation: Using the Elements of Variance for Conditional Means and Probabilities"  \url{https://www.ssrn.com/abstract=2745308}.
    @examples
    set.seed(123)
    x <- rnorm(1000) ; y <- rnorm(1000) ; z <- rnorm(1000)
    A <- data.frame(x, y, z)
    NNS.copula(A, plot = TRUE, independence.overlay = TRUE)
    @export
    """

    if np.any(np.isnan(x)):
        raise Exception("You have some missing values, please address.")
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(x, np.ndarray) and len(x.shape) == 1:
        x = x.reshape(-1, 1)  # series to matrix like

    A = x
    n = A.shape[1]
    colnames = x.columns if isinstance(x, pd.DataFrame) else list(range(x.shape[1]))
    # l = A.shape[0]

    # if(is.null(colnames(A))){
    #    colnames.list <- list()
    #    for(i in 1 : n){
    #        colnames.list[i] <- paste0("Var ", i)
    #    }
    #    colnames(A) <- c(colnames.list)
    # }

    # if(continuous) degree <- 1 else degree <- 0
    degree = 1 if continuous else 0

    # Generate partial moment matrices
    # pm_cov <- PM.matrix(degree, degree, variable = x, pop.adj = TRUE)
    pm_cov = PM_matrix(degree, degree, variable=x, pop_adj=True)

    # Isolate the upper triangles from each of the partial moment matrices
    Co_pm = np.sum(np.triu(pm_cov["cupm"], 1)) + np.sum(np.triu(pm_cov["clpm"], 1))
    D_pm = np.sum(np.triu(pm_cov["dupm"], 1)) + np.sum(np.triu(pm_cov["dlpm"], 1))
    if plot and n == 3 and plt is not None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        B = A.values if isinstance(A, pd.DataFrame) else A
        tmp_color_1l = B[:, 0] <= np.mean(B[:, 0])
        tmp_color_2l = B[:, 1] <= np.mean(B[:, 1])
        tmp_color_3l = B[:, 2] <= np.mean(B[:, 2])
        tmp_color_1g = B[:, 0] > np.mean(B[:, 0])
        tmp_color_2g = B[:, 1] > np.mean(B[:, 1])
        tmp_color_3g = B[:, 2] > np.mean(B[:, 2])
        tmp_color = [
            "red"
            if (tmp_color_1l[i] and tmp_color_2l[i] and tmp_color_3l[i])
            else (
                "green"
                if (tmp_color_1g[i] and tmp_color_2g[i] and tmp_color_3g[i])
                else "steelblue"
            )
            for i in range(len(tmp_color_1l))
        ]
        del tmp_color_1g, tmp_color_1l, tmp_color_2g, tmp_color_2l, tmp_color_3g, tmp_color_3l
        ax.scatter(
            B[:, 0],
            B[:, 1],
            B[:, 2],
            c=tmp_color,
        )
        ax.set_xlabel(colnames[0])
        ax.set_ylabel(colnames[1])
        ax.set_zlabel(colnames[2])
        if independence_overlay:
            positions = [
                (np.min(B[:, 0]), np.min(B[:, 1]), np.min(B[:, 2])),
                (np.mean(B[:, 0]), np.mean(B[:, 1]), np.mean(B[:, 2])),
            ]
            sizes = [
                (
                    np.mean(B[:, 0]) - np.min(B[:, 0]),
                    np.mean(B[:, 1]) - np.min(B[:, 1]),
                    np.mean(B[:, 2]) - np.min(B[:, 2]),
                ),
                (
                    np.max(B[:, 0]) - np.mean(B[:, 0]),
                    np.max(B[:, 1]) - np.mean(B[:, 1]),
                    np.max(B[:, 2]) - np.mean(B[:, 2]),
                ),
            ]
            colors = ["red", "green"]
            for p, s, c in zip(positions, sizes, colors):
                plotCubeAt(pos=p, size=s, ax=ax, color=c, alpha=0.25)

    if np.any(np.isnan(Co_pm)) or Co_pm is None:
        Co_pm = 0
    if np.any(np.isnan(D_pm)) or D_pm is None:
        D_pm = 0
    if Co_pm == D_pm:
        return 0
    elif Co_pm == 0 or D_pm == 0:
        return 1
    elif Co_pm < D_pm:
        return 1 - Co_pm / D_pm
    elif Co_pm > D_pm:
        return 1 - D_pm / Co_pm
    return None


__all__ = ["cuboid_data", "plotCubeAt", "NNS_copula"]
