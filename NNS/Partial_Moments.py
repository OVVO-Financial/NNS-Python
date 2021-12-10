# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import numba


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
    target: [int, float, str, None, pd.Series, np.ndarray],
    variable: [pd.Series, np.ndarray],
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
        target = variable.mean()
    if isinstance(target, str):  # "mean"
        target = getattr(variable, target)()
    if degree == 0:
        if isinstance(target, (np.ndarray, pd.Series, list)):
            return [pd.Series(variable <= i).mean() for i in target]
        return pd.Series(variable <= target).mean()
    if isinstance(target, (np.ndarray, list)):
        return numba_LPM(
            degree=degree,
            target=target,
            variable=variable if isinstance(variable, np.ndarray) else variable.values,
        )
    elif isinstance(target, pd.Series):
        return numba_LPM(
            degree=degree,
            target=target.values,
            variable=variable if isinstance(variable, np.ndarray) else variable.values,
        )
    return numba_LPM(
        degree=degree,
        target=np.array([target]),
        variable=variable if isinstance(variable, np.ndarray) else variable.values,
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
    target: [int, float, str, None, pd.Series, np.ndarray],
    variable: [pd.Series, np.ndarray],
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
        target = variable.mean()
    if isinstance(target, str):  # "mean"
        target = getattr(variable, target)()
    if degree == 0:
        if isinstance(target, (np.ndarray, pd.Series, list)):
            return [(variable > i).mean() for i in target]
        return (variable > target).mean()
    if isinstance(target, (np.ndarray, list)):
        return numba_UPM(
            degree=degree,
            target=target,
            variable=variable if isinstance(variable, np.ndarray) else variable.values,
        )
    elif isinstance(target, pd.Series):
        return numba_UPM(
            degree=degree,
            target=target.values,
            variable=variable if isinstance(variable, np.ndarray) else variable.values,
        )
    return numba_UPM(
        degree=degree,
        target=np.array([target]),
        variable=variable if isinstance(variable, np.ndarray) else variable.values,
    )[0]


def _Co_UPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = x.mean()
    if target_y is None:
        target_y = y.mean()
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(x, target_x)()
    if isinstance(target_y, str):  # "mean"
        target_Y = getattr(y, target_y)()

    z = pd.DataFrame({"x": x, "y": y})
    z["x"] = z["x"] - target_x
    z["y"] = z["y"] - target_y

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / x.shape[0]


# Co.UPM <- Vectorize(Co.UPM, vectorize.args = c('target.x', 'target.y'))
_vec_Co_UPM = np.vectorize(_Co_UPM, excluded=["degree_x", "degree_y", "x", "y"])


def Co_UPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None, pd.Series, np.ndarray] = None,
    target_y: [int, float, str, None, pd.Series, np.ndarray] = None,
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
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = x.mean()
    if target_y is None:
        target_y = y.mean()
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(x, target_x)()
    if isinstance(target_y, str):  # "mean"
        target_y = getattr(y, target_y)()

    z = pd.DataFrame({"x": x, "y": y})
    # z <- t(c(target.x, target.y) - t(z))
    z["x"] = target_x - z["x"]
    z["y"] = target_y - z["y"]

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / x.shape[0]


# Co.LPM <- Vectorize(Co.LPM, vectorize.args = c('target.x', 'target.y'))
_vec_Co_LPM = np.vectorize(_Co_LPM, excluded=["degree_x", "degree_y", "x", "y"])


def Co_LPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None, np.ndarray, pd.Series] = None,
    target_y: [int, float, str, None, np.ndarray, pd.Series] = None,
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
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = x.mean()
    if target_y is None:
        target_y = y.mean()
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(x, target_x)()
    if isinstance(target_y, str):  # "mean"
        target_y = getattr(y, target_y)()
    z = pd.DataFrame({"x": x, "y": y})
    #   z[,1] <- z[,1] - target.x
    #   z[,2] <- target.y - z[,2]
    z["x"] = z["x"] - target_x
    z["y"] = target_y - z["y"]

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / x.shape[0]


# D.LPM <- Vectorize(D.LPM, vectorize.args = c('target.x', 'target.y'))
_vec_D_LPM = np.vectorize(_D_LPM, excluded=["degree_x", "degree_y", "x", "y"])


def D_LPM(
    degree_x: [float, int],
    degree_y: [float, int],
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None, pd.Series, np.ndarray] = None,
    target_y: [int, float, str, None, pd.Series, np.ndarray] = None,
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
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None] = None,
    target_y: [int, float, str, None] = None,
) -> float:
    if target_x is None:
        target_x = x.mean()
    if target_y is None:
        target_y = y.mean()
    if isinstance(target_x, str):  # "mean"
        target_x = getattr(x, target_x)()
    if isinstance(target_y, str):  # "mean"
        target_y = getattr(y, target_y)()
    z = pd.DataFrame({"x": x, "y": y})
    # z[,1] <- target.x - z[,1]
    # z[,2] <- z[,2] - target.y
    z["x"] = target_x - z["x"]
    z["y"] = z["y"] - target_y

    z.loc[z["x"] <= 0, "x"] = np.nan
    z.loc[z["y"] <= 0, "y"] = np.nan

    z.dropna(inplace=True)

    return (z["x"] ** degree_x).dot(z["y"] ** degree_y) / x.shape[0]


# D.UPM <- Vectorize(D.UPM, vectorize.args = c('target.x', 'target.y'))
_vec_D_UPM = np.vectorize(_D_UPM, excluded=["degree_x", "degree_y", "x", "y"])


def D_UPM(
    degree_x: [int, float],
    degree_y: [int, float],
    x: pd.Series,
    y: pd.Series,
    target_x: [int, float, str, None, pd.Series, np.ndarray] = None,
    target_y: [int, float, str, None, pd.Series, np.ndarray] = None,
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
    target: [str, dict] = "mean",
    variable: [pd.Series, pd.DataFrame, None] = None,
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
        return {
            "cupm": None,
            "dupm": None,
            "dlpm": None,
            "clpm": None,
            "cov.matrix": None,
        }
    if isinstance(variable, pd.Series):
        variable = variable.to_frame()
    assert isinstance(variable, pd.DataFrame), "supply a matrix-like (pd.DataFrame) 'variable'"
    n = variable.shape[1]

    # target dict
    if isinstance(target, (list, pd.Series, np.ndarray)):
        target = {c: target[i] for i, c in enumerate(variable.columns)}
    elif isinstance(target, str):
        target = {i: getattr(variable[i], target)() for i in variable.columns}
    elif isinstance(target, (int, float)):
        target = {i: target for i in variable.columns}

    # Partial moments lists
    clpms, cupms, dlpms, dupms = [], [], [], []
    for i, cur_var in enumerate(variable.columns):
        clpms.append([])
        cupms.append([])
        dlpms.append([])
        dupms.append([])
        # sapply(X, FUN, ..., simplify = TRUE, USE.NAMES = TRUE)
        # clpms[[i]] <- sapply(1 : n, function(b) Co.LPM(x = variable[ , i], y = variable[ , b], degree.x = LPM.degree, degree.y = LPM.degree, target.x = target[i], target.y = target[b]))
        for b, cur_var2 in enumerate(variable.columns):
            clpms[i].append(
                Co_LPM(
                    x=variable[cur_var],
                    y=variable[cur_var2],
                    degree_x=LPM_degree,
                    degree_y=LPM_degree,
                    target_x=target[cur_var],
                    target_y=target[cur_var2],
                )
            )
            cupms[i].append(
                Co_UPM(
                    x=variable[cur_var],
                    y=variable[cur_var2],
                    degree_x=UPM_degree,
                    degree_y=UPM_degree,
                    target_x=target[cur_var],
                    target_y=target[cur_var2],
                )
            )
            #            dlpms[[i]] <- sapply(1 : n, function(b)
            #            D.LPM(x = variable[ , i], y = variable[ , b], degree.x = UPM.degree, degree.y = LPM.degree, target.x = target[i], target.y = target[b]))
            if cur_var == cur_var2:
                dlpms[i].append(0.0)
                dupms[i].append(0.0)
            else:
                dlpms[i].append(
                    D_LPM(
                        x=variable[cur_var],
                        y=variable[cur_var2],
                        degree_x=UPM_degree,
                        degree_y=LPM_degree,
                        target_x=target[cur_var],
                        target_y=target[cur_var2],
                    )
                )
                dupms[i].append(
                    D_UPM(
                        x=variable[cur_var],
                        y=variable[cur_var2],
                        degree_x=LPM_degree,
                        degree_y=UPM_degree,
                        target_x=target[cur_var],
                        target_y=target[cur_var2],
                    )
                )

    # clpm.matrix <- matrix(unlist(clpms), n, n)
    # colnames(clpm.matrix) <- colnames(variable)
    # rownames(clpm.matrix) <- colnames(variable)
    clpm_matrix = pd.DataFrame(clpms, index=variable.columns, columns=variable.columns).T

    # cupm.matrix <- matrix(unlist(cupms), n, n)
    # colnames(cupm.matrix) <- colnames(variable)
    # rownames(cupm.matrix) <- colnames(variable)
    cupm_matrix = pd.DataFrame(cupms, index=variable.columns, columns=variable.columns).T

    # dlpm.matrix <- matrix(unlist(dlpms), n, n)
    # diag(dlpm.matrix) <- 0
    # colnames(dlpm.matrix) <- colnames(variable)
    # rownames(dlpm.matrix) <- colnames(variable)
    dlpm_matrix = pd.DataFrame(dlpms, index=variable.columns, columns=variable.columns).T
    # pd_fill_diagonal(dlpm_matrix, 0.0)

    # dupm.matrix <- matrix(unlist(dupms), n, n)
    # diag(dupm.matrix) <- 0
    # colnames(dupm.matrix) <- colnames(variable)
    # rownames(dupm.matrix) <- colnames(variable)
    dupm_matrix = pd.DataFrame(dupms, index=variable.columns, columns=variable.columns).T
    # pd_fill_diagonal(dupm_matrix, 0.0)

    if pop_adj:
        # adjustment <- length(variable[ , 1]) / (length(variable[ , 1]) - 1)
        adjustment = variable.shape[1] / (variable.shape[1] - 1)
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


def LPM_ratio(degree: [int, float], target: [int, float, str, None], variable: pd.Series) -> float:
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


def UPM_ratio(degree: [int, float], target: [int, float, str, None], variable: pd.Series) -> float:
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
    bins: [int, None] = None,
    plot: bool = True,
) -> pd.DataFrame:
    # TODO: CONVERT/TEST
    if target is None:
        target = variable.sort_values()
    # d/dx approximation
    if bins is None:
        # python:
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
        # https://stackoverflow.com/questions/9141732/how-does-numpy-histogram-work
        # https://stackoverflow.com/questions/33967513/creating-density-estimate-in-numpy
        # https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html

        #
        ## Default S3 method:
        # density(x, bw = "nrd0", adjust = 1,
        #        kernel = c("gaussian", "epanechnikov", "rectangular",
        #                   "triangular", "biweight",
        #                   "cosine", "optcosine"),
        #        weights = NULL, window = kernel, width,
        #        give.Rkern = FALSE,
        #        n = 512, from, to, cut = 3, na.rm = FALSE, ...)
        # bins <- density(variable)$bw
        # $bw = the bandwidth used.
        #
        # > bw.nrd0
        # function (x)
        # {
        #     if (length(x) < 2L)
        #         stop("need at least 2 data points")
        #     hi <- sd(x)
        #     if (!(lo <- min(hi, IQR(x)/1.34)))
        #         (lo <- hi) || (lo <- abs(x[1L])) || (lo <- 1)
        #     0.9 * lo * length(x)^(-0.2)
        # }
        # > IQR
        # function (x, na.rm = FALSE, type = 7)
        # diff(quantile(as.numeric(x), c(0.25, 0.75), na.rm = na.rm, names = FALSE,
        #     type = type))

        # TODO: Understand R density()$bw function
        bins = 4
        # bins = density(variable)$bw
        tgt = np.arange(target.min(), target.max(), bins)
    else:
        d_dx = (target.max().abs() + target.min().abs()) / bins
        tgt = np.arange(target.min(), target.max(), d_dx)

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
    variable: [pd.Series, pd.DataFrame],
    degree: [float, int] = 0,
    target: [float, int, None] = None,
    type: str = "CDF",
    plot: bool = True,
) -> dict:
    # TODO: CONVERT/TEST
    if target is not None:
        # TODO: use iloc[] / loc[]
        if isinstance(variable, pd.Series) or variable.shape[1] == 1:
            if target < variable.min() or target > variable.max():
                raise ValueError(
                    "Please make sure target is within the observed values of variable."
                )
        else:
            if target[1] < variable[:, 1].min() or target[1] > variable[:, 1].max():
                raise ValueError(
                    "Please make sure target 1 is within the observed values of variable 1."
                )
            if target[2] < variable[:, 2].min() or target[2] > variable[:, 2].max():
                raise ValueError(
                    "Please make sure target 2 is within the observed values of variable 2."
                )

    cdf_type = type.lower().strip()
    if cdf_type not in ["cdf", "survival", "hazard", "cumulative hazard"]:
        raise ValueError("Please select a type from: [CDF, survival, hazard, cumulative hazard]")

    if isinstance(variable, pd.Series) or variable.shape[1] == 1:
        overall_target = variable.sort_values()
        x = overall_target
        if degree > 0:
            # TODO: check if target can be an pd.series/np.ndarray
            CDF = LPM_ratio(degree=degree, target=overall_target, variable=variable)
        else:
            # TODO: ecdf function
            cdf_fun = ecdf(x)
            CDF = cdf_fun(overall_target)

        values = pd.DataFrame({variable.name: variable.sort_values(), "CDF": CDF})
        if target is not None:
            P = LPM_ratio(degree=degree, target=target, variable=variable)
        else:
            P = None

        ylabel = "Probability"

        if cdf_type == "survival":
            CDF = 1 - CDF
            P = 1 - P
        elif cdf_type == "hazard":
            # TODO: R density function
            CDF = np.exp(np.log(density(x, n=x.shaṕe[1])["y"]) - np.log(1 - CDF))
            ylabel = "h(x)"
            P = NNS_reg(
                x[-x.shape[1]],
                CDF[-x.shape[1]],
                order="max",
                point_est=c(x[x.shape[1]], target),
                plot=False,
            )["Point.est"]
            CDF[np.isinf(CDF)] = P[1]
            P = P[-1]
        elif cdf_type == "cumulative hazard":
            CDF = -np.log((1 - CDF))
            ylabel = "H(x)"
            P = NNS_reg(
                x[-x.shape[1]],
                CDF[-x.shape[1]],
                order="max",
                point_est=c(x[x.shape[1]], target),
                plot=False,
            )["Point.est"]
            CDF[np.isinf(CDF)] = P[1]
            P = P[-1]
        if plot:
            # TODO matplotlib like:
            pass
            if False:
                plot(
                    x,
                    CDF,
                    pch=19,
                    col="steelblue",
                    xlab=deparse(substitute(variable)),
                    ylab=ylabel,
                    main=toupper(type),
                    type="s",
                    lwd=2,
                )
                points(x, CDF, pch=19, col="steelblue")
                lines(x, CDF, lty=2, col="steelblue")
                if target is not None:
                    segments(target, 0, target, P, col="red", lwd=2, lty=2)
                    segments(min(variable), P, target, P, col="red", lwd=2, lty=2)
                    points(target, P, col="green", pch=19)
                    mtext(text=round(P, 4), col="red", side=2, at=P, las=2)
                    mtext(text=round(target, 4), col="red", side=1, at=target, las=1)
        values = pd.DataFrame({variable.name: x, ylabel: CDF})
        return {"Function": values, "target.value": P}
    else:
        overall_target_1 = variable[:, 1]
        overall_target_2 = variable[:, 2]
        CDF = Co_LPM(
            degree_x=degree,
            degree_y=degree,
            x=variable[:, 1].sort_values(),
            y=variable[:, 2].sort_values(),
            target_x=overall_target_1,
            target_y=overall_target_2,
        ) / (
            Co_LPM(
                degree_x=degree,
                degree_y=degree,
                x=variable[:, 1].sort_values(),
                y=variable[:, 2].sort_values(),
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
            + Co_UPM(
                degree_x=degree,
                degree_y=degree,
                x=variable[:, 1].sort_values(),
                y=variable[:, 2].sort_values(),
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
            + D_UPM(
                degree_x=degree,
                degree_y=degree,
                x=variable[:, 1].sort_values(),
                y=variable[:, 2].sort_values(),
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
            + D_LPM(
                degree_x=degree,
                degree_y=degree,
                x=variable[:, 1].sort_values(),
                y=variable[:, 2].sort_values(),
                target_x=overall_target_1,
                target_y=overall_target_2,
            )
        )
        if cdf_type == "survival":
            CDF = 1 - CDF
        elif cdf_type == "hazard":
            CDF = variable.sort_values() / (1 - CDF)
        elif cdf_type == "cumulative hazard":
            CDF = -np.log(1 - CDF)
        if target is not None:
            P = Co_LPM(
                degree_x=degree,
                degree_y=degree,
                x=variable[:, 1],
                y=variable[:, 2],
                target_x=target[1],
                target_y=target[2],
            ) / (
                Co_LPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=variable[:, 1],
                    y=variable[:, 2],
                    target_x=target[1],
                    target_y=target[2],
                )
                + Co_UPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=variable[:, 1],
                    y=variable[:, 2],
                    target_x=target[1],
                    target_y=target[2],
                )
                + D_LPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=variable[:, 1],
                    y=variable[:, 2],
                    target_x=target[1],
                    target_y=target[2],
                )
                + D_UPM(
                    degree_x=degree,
                    degree_y=degree,
                    x=variable[:, 1],
                    y=variable[:, 2],
                    target_x=target[1],
                    target_y=target[2],
                )
            )
        else:
            P = None

        if plot:
            # TODO: Matplotlib like
            pass
            if False:
                plot3d(
                    variable[:, 1],
                    variable[:, 2],
                    CDF,
                    col="steelblue",
                    xlab=variable[:, 1].name,
                    ylab=variable[:, 2].name,
                    zlab="Probability",
                    box=False,
                    pch=19,
                )
                if target is not None:
                    points3d(target[1], target[2], P, col="green", pch=19)
                    points3d(target[1], target[2], 0, col="red", pch=15, cex=2)
                    lines3d(
                        x=c(target[1], variable[:, 1].max()),
                        y=c(target[2], variable[:, 2].max()),
                        z=c(P, P),
                        col="red",
                        lwd=2,
                        lty=3,
                    )
                    lines3d(
                        x=c(target[1], target[1]),
                        y=c(target[2], target[2]),
                        z=c(0, P),
                        col="red",
                        lwd=1,
                        lty=3,
                    )
                    text3d(
                        variable[:, 1].max(),
                        variable[:, 2].max(),
                        P,
                        texts=f"P = {round(P, 4)}",
                        pos=4,
                        col="red",
                    )
    return {"CDF": pd.DataFrame({"variable": variable, "CDF": CDF}), "P": P}
