# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import numba
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from . import Internal_Functions


def pd_fill_diagonal(df_matrix: pd.DataFrame, value) -> None:
    n = min(df_matrix.shape[0], df_matrix.shape[1])
    df_matrix.values[tuple([np.arange(n)] * 2)] = value


@numba.jit(parallel=True, nopython=True)
def numba_LPM(degree: [int, float], target: np.ndarray, variable: np.ndarray) -> np.ndarray:
    ret = np.zeros(shape=(target.shape[0]), dtype=np.float32)
    for i in numba.prange(target.shape[0]):
        # ugly implementation, but working
        for ll in range(variable.shape[0]):
            if variable[ll] <= target[i]:
                ret[i] += (target[i] - variable[ll]) ** degree
        ret[i] /= variable.shape[0]

        # TODO:  variable[variable <= target[i]] didn't work at numba
        # ret[i] += (
        #    (target[i] - (variable[variable <= target[i]])) ** degree
        # ).sum() / variable.shape[0]
    return ret


def LPM(
    degree: [int, float],
    target: [int, float, str, None, pd.Series, np.ndarray, list],
    variable: [pd.Series, np.ndarray, list],
) -> [float, np.ndarray]:
    r"""
    Lower Partial Moment

    This function generates a univariate lower partial moment for any degree or target.
    @param degree integer; \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param target numeric; Typically set to mean, but does not have to be. (Vectorized)
    @param variable a numeric vector.
    @return LPM of variable
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100)
    LPM(0, mean(x), x)
    @export
    """

    if target is None:
        target = np.mean(variable)
    if isinstance(target, str):  # "mean"
        target = getattr(np, target)(variable)
    if isinstance(target, list):
        target = np.array(target)
    if isinstance(variable, list):
        variable = np.array(variable)
    if degree == 0:
        if isinstance(target, (np.ndarray, pd.Series, list)):
            return np.array([np.mean(variable <= i) for i in target])
        return np.mean(variable <= target)
    if isinstance(target, (np.ndarray, list)):
        return numba_LPM(
            degree=degree,
            target=target,
            variable=variable if not hasattr(variable, "values") else variable.values,
        )
    elif isinstance(target, pd.Series):
        return numba_LPM(
            degree=degree,
            target=target.values,
            variable=variable if not hasattr(variable, "values") else variable.values,
        )
    return numba_LPM(
        degree=degree,
        target=np.array([target]),
        variable=variable if not hasattr(variable, "values") else variable.values,
    )[0]


@numba.jit(parallel=True, nopython=True)
def numba_UPM(degree: [int, float], target: np.ndarray, variable: np.ndarray) -> np.ndarray:
    ret = np.zeros(shape=(target.shape[0]), dtype=np.float32)
    for i in numba.prange(target.shape[0]):
        # ugly implementation, but working
        for ll in range(variable.shape[0]):
            if variable[ll] > target[i]:
                ret[i] += (variable[ll] - target[i]) ** degree
        ret[i] /= variable.shape[0]
        # TODO:  variable[variable > target[i]] didn't work at numba
        # ret[i] += (
        #    ((variable[variable > target[i]]) - target[i]) ** degree
        # ).sum() / variable.shape[0]
    return ret


def UPM(
    degree: [int, float],
    target: [int, float, str, None, pd.Series, np.ndarray, list],
    variable: [pd.Series, np.ndarray, list],
) -> [float, np.ndarray]:
    r"""
    Upper Partial Moment

    This function generates a univariate upper partial moment for any degree or target.
    @param degree integer; \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param target numeric; Typically set to mean, but does not have to be. (Vectorized)
    @param variable a numeric vector.
    @return UPM of variable
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100)
    UPM(0, mean(x), x)
    @export
    """
    if target is None:
        target = np.mean(variable)
    if isinstance(target, str):  # "mean"
        target = getattr(np, target)(variable)
    if isinstance(target, list):
        target = np.array(target)
    if isinstance(variable, list):
        variable = np.array(variable)
    if degree == 0:
        if isinstance(target, (np.ndarray, pd.Series, list)):
            return np.array([np.mean(variable > i) for i in target])
        return np.mean(variable > target)
    if isinstance(target, (np.ndarray, list)):
        return numba_UPM(
            degree=degree,
            target=target,
            variable=variable if not hasattr(variable, "values") else variable.values,
        )
    elif isinstance(target, pd.Series):
        return numba_UPM(
            degree=degree,
            target=target.values,
            variable=variable if not hasattr(variable, "values") else variable.values,
        )
    return numba_UPM(
        degree=degree,
        target=np.array([target]),
        variable=variable if not hasattr(variable, "values") else variable.values,
    )[0]


def _Co_UPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = np.mean(x)
    if target_y is None:
        target_y = np.mean(y)
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(np, target_x)(x)
    if isinstance(target_y, str):  # "mean"
        target_y = getattr(np, target_y)(y)

    z = pd.DataFrame({"x": x, "y": y})
    z["x"] = z["x"] - target_x
    z["y"] = z["y"] - target_y

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / len(x)


# Co.UPM <- Vectorize(Co.UPM, vectorize.args = c('target.x', 'target.y'))
_vec_Co_UPM = np.vectorize(_Co_UPM, excluded=["degree_x", "degree_y", "x", "y"])


def Co_UPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None, pd.Series, np.ndarray, list] = None,
    target_y: [int, float, str, None, pd.Series, np.ndarray, list] = None,
) -> [float, np.ndarray]:
    r"""
    Co-Upper Partial Moment
    (Upper Right Quadrant 1)

    This function generates a co-upper partial moment between two equal length variables for any degree or target.
    @param degree_x integer; Degree for variable X.  \code{(degree.x = 0)} is frequency, \code{(degree.x = 1)} is area.
    @param degree_y integer; Degree for variable Y.  \code{(degree.y = 0)} is frequency, \code{(degree.y = 1)} is area.
    @param x a numeric vector.
    @param y a numeric vector of equal length to \code{x}.
    @param target_x numeric; Typically the mean of Variable X for classical statistics equivalences, but does not have to be. (Vectorized)
    @param target_y numeric; Typically the mean of Variable Y for classical statistics equivalences, but does not have to be. (Vectorized)
    @return Co-UPM of two variables
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    Co.UPM(0, 0, x, y, mean(x), mean(y))
    @export
    """
    func = _Co_UPM
    if isinstance(target_x, list):
        target_x = np.array(target_x)
    if isinstance(target_y, list):
        target_y = np.array(target_y)
    if isinstance(target_y, (np.ndarray, pd.Series)) or isinstance(
        target_x, (np.ndarray, pd.Series)
    ):
        func = _vec_Co_UPM
    return func(
        degree_x=degree_x,
        degree_y=degree_y,
        x=x,
        y=y,
        target_x=target_x,
        target_y=target_y,
    )


def _Co_LPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = np.mean(x)
    if target_y is None:
        target_y = np.mean(y)
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(np, target_x)(x)
    if isinstance(target_y, str):  # "mean"
        target_y = getattr(np, target_y)(y)

    z = pd.DataFrame({"x": x, "y": y})
    # z <- t(c(target.x, target.y) - t(z))
    z["x"] = target_x - z["x"]
    z["y"] = target_y - z["y"]

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / len(x)


# Co.LPM <- Vectorize(Co.LPM, vectorize.args = c('target.x', 'target.y'))
_vec_Co_LPM = np.vectorize(_Co_LPM, excluded=["degree_x", "degree_y", "x", "y"])


def Co_LPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None, np.ndarray, pd.Series, list] = None,
    target_y: [int, float, str, None, np.ndarray, pd.Series, list] = None,
) -> [float, np.ndarray]:
    r"""
    Co-Lower Partial Moment
    (Lower Left Quadrant 4)

    This function generates a co-lower partial moment for between two equal length variables for any degree or target.
    @param degree_x integer; Degree for variable X.  \code{(degree.x = 0)} is frequency, \code{(degree.x = 1)} is area.
    @param degree_y integer; Degree for variable Y.  \code{(degree.y = 0)} is frequency, \code{(degree.y = 1)} is area.
    @param x a numeric vector.
    @param y a numeric vector of equal length to \code{x}.
    @param target_x numeric; Typically the mean of Variable X for classical statistics equivalences, but does not have to be. (Vectorized)
    @param target_y numeric; Typically the mean of Variable Y for classical statistics equivalences, but does not have to be. (Vectorized)
    @return Co-LPM of two variables
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    Co.LPM(0, 0, x, y, mean(x), mean(y))
    @export
    """
    func = _Co_LPM
    if isinstance(target_x, list):
        target_x = np.array(target_x)
    if isinstance(target_y, list):
        target_y = np.array(target_y)
    if isinstance(target_y, (np.ndarray, pd.Series)) or isinstance(
        target_x, (np.ndarray, pd.Series)
    ):
        func = _vec_Co_LPM
    return func(
        degree_x=degree_x,
        degree_y=degree_y,
        x=x,
        y=y,
        target_x=target_x,
        target_y=target_y,
    )


def _D_LPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = np.mean(x)
    if target_y is None:
        target_y = np.mean(y)
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(np, target_x)(x)
    if isinstance(target_y, str):  # "mean"
        target_y = getattr(np, target_y)(y)

    z = pd.DataFrame({"x": x, "y": y})
    #   z[,1] <- z[,1] - target.x
    #   z[,2] <- target.y - z[,2]
    z["x"] = z["x"] - target_x
    z["y"] = target_y - z["y"]

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / len(x)


# D.LPM <- Vectorize(D.LPM, vectorize.args = c('target.x', 'target.y'))
_vec_D_LPM = np.vectorize(_D_LPM, excluded=["degree_x", "degree_y", "x", "y"])


def D_LPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None, pd.Series, np.ndarray, list] = None,
    target_y: [int, float, str, None, pd.Series, np.ndarray, list] = None,
) -> float:
    r"""
    Divergent-Lower Partial Moment
    (Lower Right Quadrant 3)

    This function generates a divergent lower partial moment between two equal length variables for any degree or target.
    @param degree_x integer; Degree for variable X.  \code{(degree.x = 0)} is frequency, \code{(degree.x = 1)} is area.
    @param degree_y integer; Degree for variable Y.  \code{(degree.y = 0)} is frequency, \code{(degree.y = 1)} is area.
    @param x a numeric vector.
    @param y a numeric vector of equal length to \code{x}.
    @param target_x numeric; Typically the mean of Variable X for classical statistics equivalences, but does not have to be. (Vectorized)
    @param target_y numeric; Typically the mean of Variable Y for classical statistics equivalences, but does not have to be. (Vectorized)
    @return Divergent LPM of two variables
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    D.LPM(0, 0, x, y, mean(x), mean(y))
    @export
    """
    func = _D_LPM
    if isinstance(target_x, list):
        target_x = np.array(target_x)
    if isinstance(target_y, list):
        target_y = np.array(target_y)
    if isinstance(target_y, (np.ndarray, pd.Series)) or isinstance(
        target_x, (np.ndarray, pd.Series)
    ):
        func = _vec_D_LPM
    return func(
        degree_x=degree_x,
        degree_y=degree_y,
        x=x,
        y=y,
        target_x=target_x,
        target_y=target_y,
    )


def _D_UPM(
    degree_x: [int, float],
    degree_y: [int, float],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = np.mean(x)
    if target_y is None:
        target_y = np.mean(y)
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(np, target_x)(x)
    if isinstance(target_y, str):  # "mean"
        target_y = getattr(np, target_y)(y)
    z = pd.DataFrame({"x": x, "y": y})
    # z[,1] <- target.x - z[,1]
    # z[,2] <- z[,2] - target.y
    z["x"] = target_x - z["x"]
    z["y"] = z["y"] - target_y

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / len(x)


# D.UPM <- Vectorize(D.UPM, vectorize.args = c('target.x', 'target.y'))
_vec_D_UPM = np.vectorize(_D_UPM, excluded=["degree_x", "degree_y", "x", "y"])


def D_UPM(
    degree_x: [int, float],
    degree_y: [int, float],
    x: [pd.Series, np.ndarray, list],
    y: [pd.Series, np.ndarray, list],
    target_x: [int, float, str, None, pd.Series, np.ndarray, list] = None,
    target_y: [int, float, str, None, pd.Series, np.ndarray, list] = None,
) -> float:
    r"""
    Divergent-Upper Partial Moment
    (Upper Left Quadrant 2)

    This function generates a divergent upper partial moment between two equal length variables for any degree or target.
    @param degree_x integer; Degree for variable X.  \code{(degree.x = 0)} is frequency, \code{(degree.x = 1)} is area.
    @param degree_y integer; Degree for variable Y.  \code{(degree.y = 0)} is frequency, \code{(degree.y = 1)} is area.
    @param x a numeric vector.
    @param y a numeric vector of equal length to \code{x}.
    @param target_x numeric; Typically the mean of Variable X for classical statistics equivalences, but does not have to be. (Vectorized)
    @param target_y numeric; Typically the mean of Variable Y for classical statistics equivalences, but does not have to be. (Vectorized)
    @return Divergent UPM of two variables
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    D.UPM(0, 0, x, y, mean(x), mean(y))
    @export
    """
    func = _D_UPM
    if isinstance(target_x, list):
        target_x = np.array(target_x)
    if isinstance(target_y, list):
        target_y = np.array(target_y)
    if isinstance(target_y, (np.ndarray, pd.Series)) or isinstance(
        target_x, (np.ndarray, pd.Series)
    ):
        func = _vec_D_UPM
    return func(
        degree_x=degree_x,
        degree_y=degree_y,
        x=x,
        y=y,
        target_x=target_x,
        target_y=target_y,
    )


def PM_matrix(
    LPM_degree: [int, float],
    UPM_degree: [int, float],
    target: [str, dict, list, float, int, pd.Series, np.array, list] = "mean",
    variable: [pd.Series, pd.DataFrame, np.ndarray, None, list] = None,
    pop_adj: bool = False,
) -> dict:
    r"""
    Partial Moment Matrix


    This function generates a co-partial moment matrix for the specified co-partial moment.
    @param LPM_degree integer; Degree for \code{variable} below \code{target} deviations.  \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param UPM_degree integer; Degree for \code{variable} above \code{target} deviations.  \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param target numeric; Typically the mean of Variable X for classical statistics equivalences, but does not have to be. (Vectorized)  \code{(target = "mean")} (default) will set the target as the mean of every variable.
    @param variable a numeric matrix or data.frame.
    @param pop_adj logical; \code{FALSE} (default) Adjusts the sample co-partial moment matrices for population statistics.
    @return Matrix of partial moment quadrant values (CUPM, DUPM, DLPM, CLPM), and overall covariance matrix.  Uncalled quadrants will return a matrix of zeros.
    @note For divergent asymmetical \code{"D.LPM" and "D.UPM"} matrices, matrix is \code{D.LPM(column,row,...)}.
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @references Viole, F. (2017) "Bayes' Theorem From Partial Moments"
    \url{https://ssrn.com/abstract=3457377}
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100) ; z <- rnorm(100)
    A <- cbind(x,y,z)
    PM.matrix(LPM.degree = 1, UPM.degree = 1, target = "mean", variable = A)

    ## Use of vectorized numeric targets (target_x, target_y, target_z)
    PM.matrix(LPM.degree = 1, UPM.degree = 1, target = c(0, 0.15, .25), variable = A)

    ## Calling Individual Partial Moment Quadrants
    cov.mtx <- PM.matrix(LPM.degree = 1, UPM.degree = 1, target = "mean", variable = A)
    cov.mtx$cupm

    ## Full covariance matrix
    cov.mtx$cov.matrix
    @export
    """

    if variable is None:
        return {"cupm": None, "dupm": None, "dlpm": None, "clpm": None, "cov.matrix": None}
    if isinstance(variable, list):
        variable = np.column_stack(variable)
    if isinstance(target, list):
        target = np.array(target)
    if isinstance(variable, pd.Series):
        variable = variable.to_frame()
    if isinstance(variable, np.ndarray) and len(variable.shape) == 1:
        variable = variable.reshape(-1, 1)  # series to matrix like
    assert isinstance(
        variable, (pd.DataFrame, np.ndarray, np.matrix)
    ), "supply a matrix-like (pd.DataFrame, np.ndarray, np.matrix) 'variable'"

    n = variable.shape[1]
    variable_columns = (
        variable.columns if isinstance(variable, pd.DataFrame) else list(range(variable.shape[1]))
    )

    # target dict
    if isinstance(target, (list, pd.Series, np.ndarray)):
        target = {i: v for i, v in enumerate(target)}
    elif isinstance(target, str):
        # mean / median / mode
        if isinstance(variable, pd.DataFrame):
            target = {i: getattr(np, target)(variable.values[:, i]) for i in range(n)}
        else:
            target = {i: getattr(np, target)(variable[:, i]) for i in range(n)}
    elif isinstance(target, (int, float)):
        target = {i: target for i in range(n)}
    # Partial moments lists
    clpms, cupms, dlpms, dupms = [], [], [], []
    for cur_var in range(n):
        clpms.append([])
        cupms.append([])
        dlpms.append([])
        dupms.append([])
        # sapply(X, FUN, ..., simplify = TRUE, USE.NAMES = TRUE)
        # clpms[[i]] <- sapply(1 : n, function(b) Co.LPM(x = variable[ , i], y = variable[ , b], degree.x = LPM.degree, degree.y = LPM.degree, target.x = target[i], target.y = target[b]))
        for cur_var2 in range(n):
            if isinstance(variable, pd.DataFrame):
                x = variable.values[:, cur_var]
                y = variable.values[:, cur_var2]
            else:
                x = variable[:, cur_var]
                y = variable[:, cur_var2]
            clpms[cur_var].append(
                Co_LPM(
                    x=x,
                    y=y,
                    degree_x=LPM_degree,
                    degree_y=LPM_degree,
                    target_x=target[cur_var],
                    target_y=target[cur_var2],
                )
            )
            cupms[cur_var].append(
                Co_UPM(
                    x=x,
                    y=y,
                    degree_x=UPM_degree,
                    degree_y=UPM_degree,
                    target_x=target[cur_var],
                    target_y=target[cur_var2],
                )
            )
            #            dlpms[[i]] <- sapply(1 : n, function(b)
            #            D.LPM(x = variable[ , i], y = variable[ , b], degree.x = UPM.degree, degree.y = LPM.degree, target.x = target[i], target.y = target[b]))
            if cur_var == cur_var2:
                dlpms[cur_var].append(0.0)
                dupms[cur_var].append(0.0)
            else:
                dlpms[cur_var].append(
                    D_LPM(
                        x=x,
                        y=y,
                        degree_x=UPM_degree,
                        degree_y=LPM_degree,
                        target_x=target[cur_var],
                        target_y=target[cur_var2],
                    )
                )
                dupms[cur_var].append(
                    D_UPM(
                        x=x,
                        y=y,
                        degree_x=LPM_degree,
                        degree_y=UPM_degree,
                        target_x=target[cur_var],
                        target_y=target[cur_var2],
                    )
                )

    # clpm.matrix <- matrix(unlist(clpms), n, n)
    # colnames(clpm.matrix) <- colnames(variable)
    # rownames(clpm.matrix) <- colnames(variable)
    clpm_matrix = pd.DataFrame(clpms, index=variable_columns, columns=variable_columns).T

    # cupm.matrix <- matrix(unlist(cupms), n, n)
    # colnames(cupm.matrix) <- colnames(variable)
    # rownames(cupm.matrix) <- colnames(variable)
    cupm_matrix = pd.DataFrame(cupms, index=variable_columns, columns=variable_columns).T

    # dlpm.matrix <- matrix(unlist(dlpms), n, n)
    # diag(dlpm.matrix) <- 0
    # colnames(dlpm.matrix) <- colnames(variable)
    # rownames(dlpm.matrix) <- colnames(variable)
    dlpm_matrix = pd.DataFrame(dlpms, index=variable_columns, columns=variable_columns).T
    # pd_fill_diagonal(dlpm_matrix, 0.0)

    # dupm.matrix <- matrix(unlist(dupms), n, n)
    # diag(dupm.matrix) <- 0
    # colnames(dupm.matrix) <- colnames(variable)
    # rownames(dupm.matrix) <- colnames(variable)
    dupm_matrix = pd.DataFrame(dupms, index=variable_columns, columns=variable_columns).T
    # pd_fill_diagonal(dupm_matrix, 0.0)

    if pop_adj:
        # adjustment <- length(variable[ , 1]) / (length(variable[ , 1]) - 1)
        adjustment = variable.shape[0] / (variable.shape[0] - 1)
        clpm_matrix *= adjustment
        cupm_matrix *= adjustment
        dlpm_matrix *= adjustment
        dupm_matrix *= adjustment

    # cov.matrix <- cupm.matrix + clpm.matrix - dupm.matrix - dlpm.matrix
    cov_matrix = cupm_matrix + clpm_matrix - dupm_matrix - dlpm_matrix

    return {
        "cupm": cupm_matrix,
        "dupm": dupm_matrix,
        "dlpm": dlpm_matrix,
        "clpm": clpm_matrix,
        "cov.matrix": cov_matrix,
    }


def LPM_ratio(
    degree: [int, float],
    target: [int, float, str, None, pd.Series, np.ndarray, list],
    variable: [pd.Series, np.ndarray, list],
) -> [float, np.ndarray]:
    r"""
    Lower Partial Moment RATIO

    This function generates a standardized univariate lower partial moment for any degree or target.
    @param degree integer; \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param target numeric; Typically set to mean, but does not have to be. (Vectorized)
    @param variable a numeric vector.
    @return Standardized LPM of variable
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @references Viole, F. (2017) "Continuous CDFs and ANOVA with NNS"
    \url{https://ssrn.com/abstract=3007373}
    @examples
    set.seed(123)
    x <- rnorm(100)
    LPM.ratio(0, mean(x), x)

    \dontrun{
    ## Empirical CDF (degree = 0)
    lpm_cdf <- LPM.ratio(0, sort(x), x)
    plot(sort(x), lpm_cdf)

    ## Continuous CDF (degree = 1)
    lpm_cdf_1 <- LPM.ratio(1, sort(x), x)
    plot(sort(x), lpm_cdf_1)

    ## Joint CDF
    x <- rnorm(5000) ; y <- rnorm(5000)
    plot3d(x, y, Co.LPM(0, 0, sort(x), sort(y), x, y), col = "blue", xlab = "X", ylab = "Y",
    zlab = "Probability", box = FALSE)
    }
    @export
    """
    lpm = LPM(degree=degree, target=target, variable=variable)
    if degree > 0:
        area = lpm + UPM(degree=degree, target=target, variable=variable)
    else:
        area = 1
    return lpm / area


def UPM_ratio(
    degree: [int, float],
    target: [int, float, str, None, list, np.ndarray, pd.Series],
    variable: [pd.Series, np.ndarray, list],
) -> [float, np.ndarray]:
    r"""
    Upper Partial Moment RATIO

    This function generates a standardized univariate upper partial moment for any degree or target.
    @param degree integer; \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param target numeric; Typically set to mean, but does not have to be. (Vectorized)
    @param variable a numeric vector.
    @return Standardized UPM of variable
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples
    set.seed(123)
    x <- rnorm(100)
    UPM.ratio(0, mean(x), x)

    ## Joint Upper CDF
    \dontrun{
    x <- rnorm(5000) ; y <- rnorm(5000)
    plot3d(x, y, Co.UPM(0, 0, sort(x), sort(y), x, y), col = "blue", xlab = "X", ylab = "Y",
    zlab = "Probability", box = FALSE)
    }
    @export
    """
    upm = UPM(degree, target, variable)
    if degree > 0:
        area = LPM(degree, target, variable) + upm
    else:
        area = 1
    return upm / area


def NNS_PDF(
    variable: pd.Series,
    degree: [int, float] = 1,
    target: [float, int, str, None] = None,
    bins: [float, int, None] = None,
    plot: bool = True,
) -> pd.DataFrame:
    # TODO: CONVERT/TEST
    if target is None:
        target = variable.sort_values()
    # d/dx approximation
    if bins is None:
        # bins < - density(variable)$bw
        bins = Internal_Functions.bw_nrd0(variable)
        # tgt < - seq(min(target), max(target), bins)
        tgt = np.arange(np.min(target), np.max(target), bins)
    else:
        d_dx = (np.abs(np.max(target)) + np.abs(np.min(target))) / bins
        tgt = np.arange(np.min(target), np.max(target), d_dx)

    # TODO: Check return of function CDF and ['function'] dict
    CDF = NNS_CDF(variable=variable, degree=degree, target=None, type="CDF", plot=False)["Function"]

    # TODO: find function dy.dx()
    PDF = np.maximum(
        dy.dx(unlist(CDF[:, 1]), unlist(CDF[:, 2]), eval_point=tgt, deriv_method="FD")["First"], 0.0
    )
    if plot:
        # TODO: convert ```{R} plot(tgt, PDF, col = 'steelblue', type = 'l', lwd = 3, xlab = "X", ylab = "Density")```
        # plot(tgt, PDF, col = 'steelblue', type = 'l', lwd = 3, xlab = "X", ylab = "Density")
        pass
    # TODO: convert ```{R} return (data.table::data.table(cbind("Intervals" = tgt, PDF))) ```
    return pd.DataFrame()


def NNS_CDF(
    variable: [pd.Series, pd.DataFrame, np.ndarray, list],
    degree: [float, int] = 0,
    target: [float, int, None] = None,
    _type: str = "CDF",
    plot: bool = True,
    **kwargs,
) -> dict:
    if "type" in kwargs:
        # R compatibility
        _type = kwargs["type"]

    # TODO: CONVERT/TEST
    if isinstance(variable, list) and isinstance(variable[0], (np.ndarray, pd.Series)):
        variable = np.column_stack(variable)
    single_variable = (
        isinstance(variable, pd.Series)
        or (
            hasattr(variable, "shape")
            and (
                (len(variable.shape) == 2 and variable.shape[1] == 1) or (len(variable.shape) == 1)
            )
        )
        or (
            isinstance(variable, list)
            and not isinstance(variable[0], (np.ndarray, pd.Series, list))
        )
    )

    if target is not None:
        if isinstance(target, list):
            if isinstance(target[0], list):
                target = np.column_stack([np.array(i) for i in target])
            else:
                target = np.column_stack(target)

        if single_variable:
            if np.any(target < np.min(variable)) or np.any(target > np.max(variable)):
                raise ValueError(
                    "Please make sure target is within the observed values of variable."
                )
        else:
            if isinstance(variable, pd.DataFrame):
                # pandas, using .values[]
                if np.any(target[:, 0] < np.min(variable.values[:, 0])) or np.any(
                    target[:, 0] > np.max(variable.values[:, 0])
                ):
                    raise ValueError(
                        "Please make sure target 1 is within the observed values of variable 1."
                    )
                if np.any(target[:, 1] < np.min(variable.values[:, 1])) or np.any(
                    target[:, 1] > np.max(variable.values[:, 1])
                ):
                    raise ValueError(
                        "Please make sure target 2 is within the observed values of variable 2."
                    )
            else:
                if np.any(target[:, 0] < np.min(variable[:, 0])) or np.any(
                    target[:, 0] > np.max(variable[:, 0])
                ):
                    raise ValueError(
                        "Please make sure target 1 is within the observed values of variable 1."
                    )
                if np.any(target[:, 1] < np.min(variable[:, 1])) or np.any(
                    target[:, 1] > np.max(variable[:, 1])
                ):
                    raise ValueError(
                        "Please make sure target 2 is within the observed values of variable 2."
                    )

    cdf_type = _type.lower().strip()
    if cdf_type not in ["cdf", "survival", "hazard", "cumulative hazard"]:
        raise ValueError("Please select a type from: [CDF, survival, hazard, cumulative hazard]")

    if single_variable:
        # single variable
        overall_target = np.sort(variable)  # variable.sort_values()
        x = overall_target
        if degree > 0:
            CDF = LPM_ratio(degree=degree, target=overall_target, variable=variable)
        else:
            # TODO: ecdf function
            cdf_fun = Internal_Functions.ecdf_function(x)
            CDF = cdf_fun(overall_target)

        # if hasattr(variable, 'name'):
        #    values = pd.DataFrame({variable.name: overall_target, "CDF": CDF})
        # else:
        #    values = pd.DataFrame({0: overall_target, "CDF": CDF})

        if target is not None:
            P = LPM_ratio(degree=degree, target=target, variable=variable)
        else:
            P = None
        ylabel = "Probability"
        if cdf_type == "survival":
            CDF = 1 - CDF
            if P is not None:
                P = 1 - P
        elif cdf_type == "hazard":
            _density = Internal_Functions.density_kde(x)
            # CDF <- exp(log(density(x, n = length(x))$y)-log(1-CDF))
            CDF = np.exp(np.log(_density[1]) - np.log(1 - CDF))
            ylabel = "h(x)"
            # TODO
            P = NNS_reg(
                x[-x.shape[1]],
                CDF[-x.shape[1]],
                order="max",
                point_est=[x[x.shape[1]], target],
                plot=False,
            )["Point.est"]
            CDF[np.isinf(CDF)] = P[1]
            P = P[-1]
        elif cdf_type == "cumulative hazard":
            CDF = -np.log((1 - CDF))
            ylabel = "H(x)"
            # TODO
            P = NNS_reg(
                x[-x.shape[1]],
                CDF[-x.shape[1]],
                order="max",
                point_est=[x[x.shape[1]], target],
                plot=False,
            )["Point.est"]
            CDF[np.isinf(CDF)] = P[1]
            P = P[-1]
        if plot:
            plt.title(_type.upper())
            plt.xlabel(variable.name if hasattr(variable, "name") else "X")
            plt.ylabel(ylabel)
            plt.step(x, CDF, color="steelblue", linewidth=2, marker=".", where="post")
            plt.plot(x, CDF, color="steelblue", linestyle=":")
            # plot(
            #    x,
            #    CDF,
            #    pch=19,
            #    col="steelblue",
            #    xlab=deparse(substitute(variable)),
            #    ylab=ylabel,
            #    main=toupper(_type),
            #    type="s",
            #    lwd=2,
            # )
            # points(x, CDF, pch=19, col="steelblue")
            # lines(x, CDF, lty=2, col="steelblue")
            if target is not None:
                min_var = np.min(variable)
                min_p = np.min(P)
                # segments(target, 0, target, P, col="red", lwd=2, lty=2)
                # segments(min(variable), P, target, P, col="red", lwd=2, lty=2)
                # points(target, P, col="green", pch=19)
                # mtext(text=round(P, 4), col="red", side=2, at=P, las=2)
                # mtext(text=round(target, 4), col="red", side=1, at=target, las=1)
                for i, t in enumerate(target):
                    plt.plot([t, t], [0, P[i]], linestyle="--", linewidth=2, color="red")
                    plt.plot([min_var, t], [P[i], P[i]], linestyle="--", linewidth=2, color="red")
                    plt.text(min_var, P[i], f"{P[i]:.4f}", color="red")
                    plt.text(t, min_p, f"{t:.4f}", color="red", rotation=90)
                    # plt.text(t, P[i], f"({t:.4f}, {P[i]:.4f})", color='red')
                plt.scatter(target, P, color="lightgreen")

                # plt.xticks(list(plt.xticks()[0]) + list(target))
                # plt.yticks(list(plt.yticks()[0]) + list(P))

        if hasattr(variable, "name"):
            values = pd.DataFrame({variable.name: x, ylabel: CDF})
        else:
            values = pd.DataFrame({0: x, ylabel: CDF})
        return {"Function": values, "target.value": P}
    else:
        if isinstance(variable, pd.DataFrame):
            # pandas
            overall_target_1 = variable.values[:, 0]
            overall_target_2 = variable.values[:, 1]
            x_sorted = np.sort(variable.values[:, 0])
            y_sorted = np.sort(variable.values[:, 1])
        else:
            overall_target_1 = variable[:, 0]
            overall_target_2 = variable[:, 1]
            x_sorted = np.sort(variable[:, 0])
            y_sorted = np.sort(variable[:, 1])
        CDF = Co_LPM(
            degree_x=degree,
            degree_y=degree,
            x=x_sorted,
            y=y_sorted,
            target_x=overall_target_1,
            target_y=overall_target_2,
        ) / (
            Co_LPM(
                degree_x=degree,
                degree_y=degree,
                x=x_sorted,
                y=y_sorted,
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
            + Co_UPM(
                degree_x=degree,
                degree_y=degree,
                x=x_sorted,
                y=y_sorted,
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
            + D_UPM(
                degree_x=degree,
                degree_y=degree,
                x=x_sorted,
                y=y_sorted,
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
            + D_LPM(
                degree_x=degree,
                degree_y=degree,
                x=x_sorted,
                y=y_sorted,
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
        )
        if cdf_type == "survival":
            CDF = 1 - CDF
        elif cdf_type == "hazard":
            CDF = Internal_Functions.alt_cbind(np.sort(variable.reshape(-1)), 1 - CDF)
            CDF = (CDF[0] / CDF[1]).values[0 : len(overall_target_2)]
            # np.sort(variable.reshape(-1)) / (1 - CDF)
        elif cdf_type == "cumulative hazard":
            CDF = -np.log(1 - CDF)
        if target is not None:
            up = Co_LPM(
                degree_x=degree,
                degree_y=degree,
                x=overall_target_1,
                y=overall_target_2,
                target_x=target[:, 0],
                target_y=target[:, 1],
            )
            down = (
                Co_LPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=overall_target_1,
                    y=overall_target_2,
                    target_x=target[:, 0],
                    target_y=target[:, 1],
                )
                + Co_UPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=overall_target_1,
                    y=overall_target_2,
                    target_x=target[:, 0],
                    target_y=target[:, 1],
                )
                + D_LPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=overall_target_1,
                    y=overall_target_2,
                    target_x=target[:, 0],
                    target_y=target[:, 1],
                )
                + D_UPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=overall_target_1,
                    y=overall_target_2,
                    target_x=target[:, 0],
                    target_y=target[:, 1],
                )
            )
            P = up / down
        else:
            P = None

        if plot:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")
            plt.title(_type.upper())
            ax.scatter(
                overall_target_1,
                overall_target_2,
                zs=CDF,
                color="steelblue",
                marker="o",
            )
            ax.set_xlabel("X" if not hasattr(variable, "columns") else variable.columns[0])
            ax.set_ylabel("Y" if not hasattr(variable, "columns") else variable.columns[1])
            ax.set_zlabel("Probability")

            # plot3d(
            #    variable[:, 1],
            #    variable[:, 2],
            #    CDF,
            #    col="steelblue",
            #    xlab=variable[:, 1].name,
            #    ylab=variable[:, 2].name,
            #    zlab="Probability",
            #    box=False,
            #    pch=19,
            # )
            if target is not None:
                ax.scatter(target[:, 0], target[:, 1], zs=P, color="lightgreen")
                ax.scatter(target[:, 0], target[:, 1], zs=np.zeros(target.shape[0]), color="red")
                # points3d(target[1], target[2], P, col="green", pch=19)
                # points3d(target[1], target[2], 0, col="red", pch=15, cex=2)
                for P_i in P:
                    ax.plot(
                        [target[0, 0], np.max(overall_target_1)],
                        [target[0, 1], np.max(overall_target_2)],
                        zs=[P_i, P_i],
                        color="red",
                        linewidth="2",
                    )
                # lines3d(
                #    x=c(target[1], variable[:, 1].max()),
                #    y=c(target[2], variable[:, 2].max()),
                #    z=c(P, P),
                #    col="red",
                #    lwd=2,
                #    lty=3,
                # )
                for P_i in P:
                    ax.plot(
                        [target[0, 0], target[0, 0]],
                        [target[0, 1], target[0, 1]],
                        zs=[0, P_i],
                        color="red",
                        linewidth="2",
                    )
                # lines3d(
                #    x=c(target[1], target[1]),
                #    y=c(target[2], target[2]),
                #    z=c(0, P),
                #    col="red",
                #    lwd=1,
                #    lty=3,
                # )
                for P_i in P:
                    ax.text(
                        np.max(overall_target_1),
                        np.max(overall_target_2),
                        P_i,
                        f"P = {P_i:.4f}",
                        color="red",
                    )
                # text3d(
                #    variable[:, 1].max(),
                #    variable[:, 2].max(),
                #    P,
                #    texts=f"P = {round(P, 4)}",
                #    pos=4,
                #    col="red",
                # )
        if not isinstance(variable, pd.DataFrame):
            return {
                "CDF": pd.concat([pd.DataFrame(variable), pd.Series(CDF, name="CDF")], axis=1),
                "P": P,
            }
        else:
            return {"CDF": pd.concat([variable, pd.Series(CDF, name="CDF")], axis=1), "P": P}


__all__ = [
    "pd_fill_diagonal",
    "LPM",
    "UPM",
    "Co_UPM",
    "Co_LPM",
    "D_LPM",
    "D_UPM",
    "PM_matrix",
    "LPM_ratio",
    "UPM_ratio",
    # "NNS_PDF", # TODO
    "NNS_CDF",  # TODO: partial
]
