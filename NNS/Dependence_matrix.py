# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def NNS_dep_matrix(x: [np.ndarray, pd.Series], order=None, degree=None, asym: bool=False):
    if len(x.shape) == 1:
        raise Exception("supply both 'x' and 'y' or a matrix-like 'x'")
    n = x.shape[1]
    if isinstance(x, pd.Series):
        x = x.to_frame()
    if isinstance(x, np.ndarray) and len(x.shape) == 1:
        x = x.reshape(-1, 1)  # series to matrix like

    if(x.shape[0] < 20 ):
        order=2

    raw_rhos_lower = {}
    raw_deps_lower = {}
    raw_both_lower = {}
    raw_rhos_upper = {}
    raw_deps_upper = {}
    raw_both_upper = {}

    for i in range(n):
        raw_both_lower[i] = sapply((i + 1) : n, function(b) NNS_dep(x[ , i], x[ , b], print.map = FALSE, asym = asym))
        raw_both_upper[i] = sapply((i + 1) : n, function(b) NNS_dep(x[ , b], x[ , i], print.map = FALSE, asym = asym))
        raw_rhos_upper[i] = unlist(raw_both_upper[i][row_names(raw_both_upper[i])=="Correlation"])
        raw_deps_upper[i] = unlist(raw_both_upper[i][row_names(raw_both_upper[i])=="Dependence"])
        raw_rhos_lower[i] = unlist(raw_both_lower[i][row_names(raw_both_lower[i])=="Correlation"])
        raw_deps_lower[i] = unlist(raw_both_lower[i][row_names(raw_both_lower[i])=="Dependence"])

    rhos = matrix(, n, n)
    deps = matrix(0, n, n)
    if not asym:
        rhos[lower.tri(rhos, diag = FALSE)] = (unlist(raw.rhos_upper) + unlist(raw.rhos_lower)) / 2
        deps[lower.tri(deps, diag = FALSE)] = (unlist(raw.deps_upper) + unlist(raw.deps_lower)) / 2
        rhos = pmax(rhos, t(rhos), na.rm = TRUE)
        deps = pmax(deps, t(deps), na.rm = TRUE)
    else:
        rhos[lower.tri(rhos, diag = FALSE)] = unlist(raw.rhos_lower)
        deps[lower.tri(deps, diag = FALSE)] = unlist(raw.deps_lower)

        rhos_upper = matrix(0, n, n)
        deps_upper = matrix(0, n, n)

        rhos[is.na(rhos)] = 0
        deps[is.na(deps)] = 0

        rhos_upper[lower.tri(rhos_upper, diag=FALSE)] = unlist(raw.rhos_upper)
        rhos_upper = t(rhos_upper)

        deps_upper[lower.tri(deps_upper, diag=FALSE)] = unlist(raw.deps_upper)
        deps_upper = t(deps_upper)

        rhos = rhos + rhos_upper
        deps = deps + deps_upper

    diag(rhos) = 1
    diag(deps) = 1

    colnames(rhos) = colnames(x)
    colnames(deps) = colnames(x)
    rownames(rhos) = colnames(x)
    rownames(deps) = colnames(x)

    return {
        "Correlation": rhos,
        "Dependence": deps
    }

__all__ = [
    "NNS_dep_matrix"
]