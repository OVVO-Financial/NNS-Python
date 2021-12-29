# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def mode(x:pd.Series) -> float:
    """Continuous Mode of a distribution"""
    # TODO: Convert
    return x.mode(dropna=True)[0]
    #d <-tryCatch(density(na.omit(as.numeric(x))), error = function(e) { median(x)})
    #tryCatch(d$x[which.max(d$y)], error = function(e) {d})

def mode_class(x1):
    # TODO: Convert
    """Classification Mode of a distribution"""
    raise Exception("Not implemented")
    x = x1.dropna()
    ux = x.unique()
    return ux[which.max(tabulate(match(x, ux)))]

def gravity(x: pd.Series) -> float:
    # TODO: Test
    """Central Tendency"""
    return (x.mean() + x.median() + mode(x)) / 3

def alt_cbind(x: pd.Series, y: pd.Series, first=False) -> pd.DataFrame:
    """### cbind different length vectors"""
    if(len(x)<len(y)):
        if(first):
            x = pd.Series([np.nan] * (len(y) - len(x))).append(x).rename(x.name)
        else:
            x = x.append([np.nan] * (len(y) - len(x)))
    if(len(y)<len(x)):
        if(first):
            y = pd.Series([np.nan] * (len(x) - len(y))).append(y).rename(y.name)
        else:
            y = y.append([np.nan] * (len(x) - len(y)))
    return pd.DataFrame([x, y])

def factor_2_dummy(x:pd.Series) -> pd.Series:
    """Factor to dummy variable"""
    # TODO: check how to see if dtype is category
    if(x.dtype == "category" and x.nunique() > 1):
        # TODO: what model.matrix do?
        return model_matrix(~(x) -1, x)[:,-1]
    return x.astype(float)

def factor_2_dummy_FR(x: pd.Series) -> pd.Series:
    """Factor to dummy variable FULL RANK"""
    # TODO: check how to see if dtype is category
    if (x.dtype == "category" and x.nunique() > 1):
        # TODO: what model.matrix do?
        return model_matrix(~(x) -1, x)
    return x.astype(float)



def generate_vectors(x: pd.Series, l: pd.Series) -> dict:
    r"""Generator for 1:length(lag) vectors in NNS.ARMA"""
    Component_series = []
    Component_index = []
    for i in range(len(l)):
        # TODO: convert ```{R} CS = rev(x[seq(length(x)+1, 1, -l[i])]) ```
        CS = rev(x[seq(length(x)+1, 1, -l[i])])
        CS.dropna(inplace=True)
        # TODO: convert ```{R} Component_series[[paste('Series.', i, sep = "")]] = CS```
        Component_series[[paste('Series.', i, sep = "")]] = CS
        # TODO: convert ```{R} Component_index[[paste('Index.', i, sep = "")]] = (1 : length(CS))```
        Component_index[[paste('Index.', i, sep = "")]] = range(len(CS))
    return {
        "Component.index": Component_index,
        "Component.series": Component_series
    }


def ARMA_seas_weighting(sf,mat):
    # TODO: convert this to python

    ### Weight and lag function for seasonality in NNS.ARMA
    M = mat
    n = ncol(M)
    if(is.null(n)):
        return {
            "lag": M[1],
            "Weights": 1
        }
    if(n == 1):
        return {
            "lag": 1,
            "Weights": 1
        }
    if(n > 1):
        if(sf):
            lag = M$all.periods$Period[1]
            Weights = 1
            return {
                "lag": lag,
                "Weights": Weights
            }
        # Determine lag from seasonality test
        lag = na.omit(M$Period)
        Observation_weighting = (1 / np.sqrt(lag))
        if(is.na(M$Coefficient.of.Variation) and len(M$Coefficient.of.Variation)==1):
            Lag_weighting = 1
        else:
            Lag_weighting = (M$Variable.Coefficient.of.Variation - M$Coefficient.of.Variation)
        Weights = (Lag_weighting * Observation_weighting) / (Lag_weighting * Observation_weighting).sum()
        return {
            "lag": lag,
            "Weights": Weights
        }
    return None

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
    if(is.null(dim(x)[2])):
        # TODO: Translate ```{R} noquote(as.character(deparse(substitute(x)))) ```
        colheads = noquote(as.character(deparse(substitute(x))))
        x = t(t(x))
    j_vectors = []
    # TODO: Convert ```{R} ncol(x)```
    for j in range(ncol(x)):
        if(is.null(colheads)):
            # TODO: Convert
            colheads = colnames(x)[j]
            colheads = noquote(as.character(deparse(substitute(colheads))))

        x.vectors = list()
        heads = paste0(colheads, "_tau_")
        heads = gsub('"', '' ,heads)

        for i in range(max_tau):
            x_vectors[[paste(heads, i, sep = "")]] = numeric(0L)
            start = max_tau - i + 1
            end = length(x[,j]) - i
            x_vectors[[i + 1]] = x[start : end, j]

        j_vectors[[j]] = do.call(cbind, x.vectors)
        colheads = None
    mtx = pd.DataFrame(do.call(cbind, j.vectors))

    relevant_lags = list(length(tau))
    if(length(unlist(tau)) > 1):
        for i in range(length(tau)):
            relevant_lags[[i]] = c((i-1)*max_tau + i, (i-1)*max_tau + unlist(tau[[i]]) + i)
        relevant_lags = sort(unlist(relevant_lags))
        mtx = mtx[: , relevant_lags]
    vars = which(grepl("tau_0", colnames(mtx)))

    everything_else = seq_len(dim(mtx)[2])[-vars]
    mtx = mtx[,c(vars, everything_else)]
    return mtx

def RP(x: pd.Series, rows = None, cols = None, na_rm = False):
    r"""Row products for NNS.dep.hd"""

    if rows is not None and cols is not None:
        x = x[rows, cols, drop = False]
    elif rows is not None:
        x = x[rows, , drop = False]
    elif cols is not None:
        x = x[, cols, drop = False]

    n = nrow(x)
    y = double(length = n)
    if (n == 0L)
        return(y)
    for ii in range(seq_len(n)):
        y[ii] = prod(x[ii, , drop = True], na_rm = na_rm)
    return y


def NNS_meboot_part(x: pd.Series, n, z, xmin, xmax, desintxb, reachbnd):
    r"""Refactored meboot::meboot.part function using tdigest"""
    # Generate random numbers from the [0,1] uniform interval
    # TODO: Convert
    p = runif(n, min=0, max=1)
    # Assign quantiles of x from p
    td = tdigest::tdigest(x, compression = max(100, np.log(n,10)*100))
    try:
        q = tdigest::tquantile(td, p)
    except Exception:
        q = x.quantile(p)
    # TODO: Convert
    ref1 = which(p <= (1/n))
    if(length(ref1) > 0):
        qq = approx(c(0,1/n), c(xmin,z[1]), p[ref1])['y']
        q[ref1] = qq
        if(not reachbnd):
            q[ref1] = qq + desintxb[1]-0.5*(z[1]+xmin)
    ref4 = which(p == ((n-1)/n))
    if(len(ref4) > 0):
        q[ref4] <- z[n-1]

    ref5 = which(p > ((n-1)/n))
    if(len(ref5) > 0):
        # Right tail proportion p[i]
        qq = approx(c((n-1)/n,1), c(z[n-1],xmax), p[ref5])['y']
        q[ref5] = qq   # this implicitly shifts xmax for algorithm
        if(not reachbnd):
            q[ref5] = qq + desintxb[n]-0.5*(z[n-1]+xmax)
        # such that the algorithm gives xmax when p[i]=1
        # this is the meaning of reaching the bounds xmax and xmin
    return q

def NNS_meboot_expand_sd(x: pd.Series, ensemble, fiv=5):
    # TODO: Convert
    r"""Refactored meboot::expand.sd function"""
    if (is.null(ncol(x))):
        sdx = x.std()
    else:
        sdx = apply(x, 2, np.std)

    sdf = c(sdx, apply(ensemble, 2, np.std))
    sdfa = sdf/sdf[1]  # ratio of actual sd to that of original data
    sdfd = sdf[1]/sdf  # ratio of desired sd to actual sd

    # expansion is needed since some of these are <1 due to attenuation
    mx = 1+(fiv/100)
    # following are expansion factors
    id = which(sdfa < 1)
    if (len(id) > 0):
        sdfa[id] = runif(n=length(id), min=1, max=mx)

    sdfdXsdfa = sdfd[-1]*sdfa[-1]  #  TODO: this looks like some keyboard with problem ahhaha
    id = which(np.floor(sdfdXsdfa) > 0)
    if (len(id) > 0):
        if(len(id) > 1):
            ensemble[:,id] = ensemble[:,id].dot(np.diag(sdfdXsdfa[id]))    # TODO convert ```{R} diag(sdfdXsdfa[id])```
        else:
            ensemble[:,id] = ensemble[:,id] * sdfdXsdfa[id]
    if(is.ts(x)):
        ensemble <- ts(ensemble, frequency=frequency(x), start=start(x))
    return ensemble


__all__ = [
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
    "NNS_meboot_expand_sd"
]
