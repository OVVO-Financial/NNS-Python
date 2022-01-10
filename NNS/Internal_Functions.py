# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy
import scipy.stats
import KDEpy


def fivenum(v: [pd.Series, np.ndarray]) -> list:
    """Tukey Five-Number Summaries

    Returns Tukey's five number summary (minimum, lower-hinge, median, upper-hinge, maximum) for the input data."""
    x = v[~np.isnan(v)]
    x = np.sort(x)
    n = x.shape[0]
    if n == 0:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

    n4 = np.floor((n + 3) / 2) / 2
    d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n])
    return list(0.5 * (x[np.floor(d).astype(int) - 1] + x[np.ceil(d).astype(int) - 1]))


def bw_nrd0(x: [scipy.stats.kde.gaussian_kde, np.ndarray, pd.Series]):
    """Kernel density bandwidth estimator - like R default option of density function"""
    # https://stackoverflow.com/questions/38337804/is-there-a-scipy-numpy-alternative-to-rs-nrd0
    if isinstance(x, pd.Series):
        tmpx = x.values if len(x.shape) == 1 else x.values[0, :]
    elif isinstance(x, np.ndarray):
        tmpx = x if len(x.shape) == 1 else x[0, :]
    else:
        tmpx = x.dataset if len(x.dataset.shape) == 1 else x.dataset[0, :]
    if len(tmpx) < 2:
        raise (Exception("need at least 2 data points"))
    hi = np.std(tmpx, ddof=1)
    q75, q25 = np.percentile(tmpx, [75, 25])
    iqr = q75 - q25
    lo = min(hi, iqr / 1.34)
    if not (lo != 0):
        if hi != 0:
            lo = hi
        elif abs(x[0]) != 0:
            lo = abs(x[0])
        else:
            lo = 1
    bw = 0.9 * lo * len(tmpx) ** -0.2
    return bw


def mode(x: [pd.Series, np.ndarray]) -> float:
    """Continuous Mode of a distribution"""

    #    d = tryCatch(
    #        density(
    #            as.numeric(x),
    #            na.rm = TRUE,
    #            n = 128,
    #            from = min(x),
    #            to = max(x)),
    #            error = function(e) (
    #                median(x) + mean(fivenum(x)[2:4])
    #            )/2
    #    )
    #    tryCatch(
    #        d$x[which.max(d$y)],
    #        error = function(e) d
    #    )

    if x.shape[0] == 2:
        return np.median(x)

    if (
        True
    ):  # TODO Check if KDEpy exists, if not use the old scipy.stats.gaussian_kde (with bad outputs btw)
        if isinstance(x, pd.Series):
            a, b = KDEpy.FFTKDE(kernel="gaussian", bw=bw_nrd0(x.values)).fit(x.values).evaluate()
        else:
            a, b = KDEpy.FFTKDE(kernel="gaussian", bw=bw_nrd0(x)).fit(x).evaluate()
        mode_value = a[np.argmax(b)]
    elif (
        False
    ):  # TODO Check if KDEpy exists, if not use the old scipy.stats.gaussian_kde (with bad outputs btw)
        # https://rmflight.github.io/post/finding-modes-using-kernel-density-estimates/
        kernel = scipy.stats.gaussian_kde(x, bw_method=bw_nrd0)
        height = kernel.pdf(x)
        mode_value = x[np.argmax(height)]
    return mode_value


def mode_class(x: [np.ndarray, pd.Series]) -> float:
    """Classification Mode of a distribution"""
    if x.shape[0] == 0:
        return np.nan
    unique, indexes, counts = np.unique(x[~np.isnan(x)], return_counts=True, return_index=True)
    max_uniques = unique[counts == np.max(counts)]
    return x[np.isin(x, max_uniques)][0]


def gravity(x: [pd.Series, np.ndarray]) -> float:
    """Central Tendency"""
    return (np.mean(x) + np.median(x) + mode(x)) / 3.0


def gravity_class(x: [pd.Series, np.ndarray]) -> float:
    """Central Tendency Class"""
    return (np.mean(x) + np.mean(fivenum(x)[1:4])) / 2.0


def alt_cbind(
    x: [pd.Series, np.ndarray], y: [pd.Series, np.ndarray], first: bool = False
) -> pd.DataFrame:
    """### cbind different length vectors"""
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    len_x = len(x)
    len_y = len(y)
    if len_x < len_y:
        v = pd.Series(np.full(fill_value=np.nan, shape=(len_y - len_x), dtype=float))
        if first:
            x = pd.Series(v).append(x).rename(x.name)
        else:
            x = pd.concat([x, v], axis=0).rename(x.name)
        i = len_x
        while i < len_y:
            next_i = i + len_x
            if next_i >= len_y:
                next_i = len_y
            x[i:next_i] = x[0 : next_i - i]
            if next_i >= len_y:
                break
            i = next_i

    if len_y < len_x:
        v = pd.Series(np.full(fill_value=np.nan, shape=(len_x - len_y), dtype=float))
        if first:
            y = pd.Series(v).append(y).rename(y.name)
        else:
            y = pd.concat([y, v], axis=0).rename(y.name)
        i = len_y
        while i < len_x:
            next_i = i + len_y
            if next_i >= len_x:
                next_i = len_x
            y[i:next_i] = y[0 : next_i - i]
            if next_i >= len_x:
                break
            i = next_i
    df = pd.DataFrame({0: x.values, 1: y.values})
    df.columns = (0 if x.name is None else x.name, 1 if y.name is None else y.name)
    return df


def factor_2_dummy(x: pd.Series) -> [pd.Series, pd.DataFrame]:
    """Factor to dummy variable"""
    if x.dtype.name == "category" and x.nunique() > 1:
        return pd.get_dummies(x, prefix=x.name).iloc[:, 1:]
    return x.astype(float)


def factor_2_dummy_FR(x: pd.Series) -> [pd.Series, pd.DataFrame]:
    """Factor to dummy variable FULL RANK"""
    if x.dtype.name == "category" and x.nunique() > 1:
        return pd.get_dummies(x, prefix=x.name)
    return x.astype(float)


def generate_vectors(
    x: [np.ndarray, pd.Series], l: [np.ndarray, pd.Series, list, float, int]
) -> dict:
    r"""Generator for 1:length(lag) vectors in NNS.ARMA"""
    if isinstance(l, (int, float)):
        l = [l]
    Component_series = {}
    Component_index = {}
    len_x = len(x)
    len_l = len(l)
    for i in range(len_l):
        CS = np.array([x[i] for i in range(len_x, -1, -l[i]) if len_x > i >= 0])[::-1]
        CS = CS[~np.isnan(CS)]
        Component_series[f"Series.{i+1}"] = CS
        Component_index[f"Index.{i+1}"] = np.arange(1, len(CS) + 1)
    return {"Component.index": Component_index, "Component.series": Component_series}


def ARMA_seas_weighting(sf: [int, float, bool], mat: dict) -> dict:
    # TODO
    ### Weight and lag function for seasonality in NNS.ARMA
    sf_true = sf is True or (not isinstance(sf, bool) and sf != 0)
    if (
        "all.periods" not in mat
        or "best.period" not in mat
        or "periods" not in mat
        or len(mat["all.periods"].shape) != 2
    ):
        raise Exception("An matrix was expected at mat['all.periods']")

    n = mat["all.periods"].shape[0]
    if n <= 0:
        return {"lag": mat["all.periods"]["Period"], "Weights": 1}
    elif n == 1:
        return {"lag": 1, "Weights": 1}
    else:
        if sf_true:
            return {"lag": mat["all.periods"]["Period"].iloc[0], "Weights": 1}
        # Determine lag from seasonality test

        lag = mat["all.periods"]["Period"][~np.isnan(mat["all.periods"]["Period"])]
        Observation_weighting = 1 / np.sqrt(lag)
        if (
            np.any(np.isnan(mat["all.periods"]["Coefficient.of.Variation"]))
            and len(mat["all.periods"]["Coefficient.of.Variation"]) == 1
        ):
            Lag_weighting = 1
        else:
            Lag_weighting = (
                mat["all.periods"]["Variable.Coefficient.of.Variation"]
                - mat["all.periods"]["Coefficient.of.Variation"]
            )
        Weights = (Lag_weighting * Observation_weighting) / np.sum(
            Lag_weighting * Observation_weighting
        )
        return {"lag": lag, "Weights": Weights}


def lag_mtx(x, tau):
    # TODO: Translate
    r"""
    Lag matrix generator for NNS.VAR
    Vector of tau for single different tau per variables tau = c(1, 4)
    List of tau vectors for multiple different tau per variables tau = list(c(1,2,3), c(4,5,6))
    """
    colheads = None
    # TODO: Translate ```{R} max(unlist(tau)) ```
    max_tau = max(unlist(tau))
    if is_null(dim(x)[2]):
        # TODO: Translate ```{R} noquote(as_character(deparse(substitute(x)))) ```
        colheads = noquote(as_character(deparse(substitute(x))))
        x = t(t(x))
    j_vectors = []
    # TODO: Convert ```{R} ncol(x)```
    for j in range(ncol(x)):
        if is_null(colheads):
            # TODO: Convert
            colheads = colnames(x)[j]
            colheads = noquote(as_character(deparse(substitute(colheads))))

        x.vectors = list()
        heads = paste0(colheads, "_tau_")
        heads = gsub('"', "", heads)

        for i in range(max_tau):
            x_vectors[[paste(heads, i, sep="")]] = numeric(0)
            start = max_tau - i + 1
            end = length(x[:, j]) - i
            x_vectors[[i + 1]] = x[start:end, j]

        j_vectors[[j]] = do.call(cbind, x.vectors)
        colheads = None
    mtx = pd.DataFrame(do.call(cbind, j.vectors))

    relevant_lags = list(length(tau))
    if length(unlist(tau)) > 1:
        for i in range(length(tau)):
            relevant_lags[[i]] = c((i - 1) * max_tau + i, (i - 1) * max_tau + unlist(tau[[i]]) + i)
        relevant_lags = sort(unlist(relevant_lags))
        mtx = mtx[:, relevant_lags]
    vars = which(grepl("tau_0", colnames(mtx)))

    everything_else = seq_len(dim(mtx)[2])[-vars]
    mtx = mtx[:, c(vars, everything_else)]
    return mtx


def RP(x: pd.Series, rows=None, cols=None, na_rm=False):
    r"""Row products for NNS.dep.hd"""
    if rows is not None and cols is not None:
        pass
        # x = x[rows, cols, drop = False]
    elif rows is not None:
        pass
        # x = x[rows, , drop = False]
    elif cols is not None:
        pass
        # x = x[, cols, drop = False]

    n = nrow(x)
    y = double(length=n)
    if n == 0:
        return y
    for ii in range(seq_len(n)):
        pass
        # y[ii] = prod(x[ii, , drop = True], na_rm = na_rm)
    return y


def NNS_meboot_part(x: pd.Series, n, z, xmin, xmax, desintxb, reachbnd):
    r"""Refactored meboot::meboot.part function using tdigest"""
    # Generate random numbers from the [0,1] uniform interval
    # TODO: Convert
    p = runif(n, min=0, max=1)
    # Assign quantiles of x from p
    # td = tdigest::tdigest(x, compression = max(100, np.log(n,10)*100))
    q = np.quantile(x, p)
    # try:
    #    q = tdigest::tquantile(td, p)
    # except Exception:
    #    q = x.quantile(p)
    # TODO: Convert
    ref1 = which(p <= (1 / n))
    if length(ref1) > 0:
        qq = approx(c(0, 1 / n), c(xmin, z[1]), p[ref1])["y"]
        q[ref1] = qq
        if not reachbnd:
            q[ref1] = qq + desintxb[1] - 0.5 * (z[1] + xmin)
    ref4 = which(p == ((n - 1) / n))
    if len(ref4) > 0:
        q[ref4] < -z[n - 1]

    ref5 = which(p > ((n - 1) / n))
    if len(ref5) > 0:
        # Right tail proportion p[i]
        qq = approx(c((n - 1) / n, 1), c(z[n - 1], xmax), p[ref5])["y"]
        q[ref5] = qq  # this implicitly shifts xmax for algorithm
        if not reachbnd:
            q[ref5] = qq + desintxb[n] - 0.5 * (z[n - 1] + xmax)
        # such that the algorithm gives xmax when p[i]=1
        # this is the meaning of reaching the bounds xmax and xmin
    return q


def NNS_meboot_expand_sd(x: pd.Series, ensemble, fiv=5):
    # TODO: Convert
    r"""Refactored meboot::expand.sd function"""
    if is_null(ncol(x)):
        sdx = x.std()
    else:
        sdx = apply(x, 2, np.std)

    sdf = c(sdx, apply(ensemble, 2, np.std))
    sdfa = sdf / sdf[1]  # ratio of actual sd to that of original data
    sdfd = sdf[1] / sdf  # ratio of desired sd to actual sd

    # expansion is needed since some of these are <1 due to attenuation
    mx = 1 + (fiv / 100)
    # following are expansion factors
    id = which(sdfa < 1)
    if len(id) > 0:
        sdfa[id] = runif(n=length(id), min=1, max=mx)

    sdfdXsdfa = sdfd[-1] * sdfa[-1]  #  TODO: this looks like some keyboard with problem ahhaha
    id = which(np.floor(sdfdXsdfa) > 0)
    if len(id) > 0:
        if len(id) > 1:
            pass
            # ensemble[:,id] = ensemble[:,id].dot(np.diag(sdfdXsdfa[id]))    # TODO convert ```{R} diag(sdfdXsdfa[id])```
        else:
            pass
            # ensemble[:,id] = ensemble[:,id] * sdfdXsdfa[id]
    if is_ts(x):
        ensemble = ts(ensemble, frequency=frequency(x), start=start(x))
    return ensemble


__all__ = [
    "bw_nrd0",
    "mode",
    "mode_class",
    "gravity",
    "alt_cbind",
    "factor_2_dummy",
    "factor_2_dummy_FR",
    "generate_vectors",
    "ARMA_seas_weighting",
    "lag_mtx",
    "RP",
    "NNS_meboot_part",
    "NNS_meboot_expand_sd",
]
