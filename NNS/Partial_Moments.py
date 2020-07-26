# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def LPM(degree: [int, float], target: pd.Series, variable: [pd.Series, float, int]) -> pd.Series:
    """Lower Partial Moment

    This function generates a univariate lower partial moment for any degree or target.

    @param degree integer; \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param target numeric; Typically set to mean, but does not have to be. (Vectorized)
    @param variable a numeric vector.
    @return LPM of variable
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples{R}
    set.seed(123)
    x <- rnorm(100)
    LPM(0, mean(x), x)
    @export
    """
    if(degree == 0):
        return (variable <= target).mean()

    return ((target - (variable[variable <= target])) ** degree).sum() / variable.shape[0]

def LPM(degree: [int, float], target: pd.Series, variable: [pd.Series, float, int]) -> pd.Series:
    """Upper Partial Moment

    This function generates a univariate upper partial moment for any degree or target.
    @param degree integer; \code{(degree = 0)} is frequency, \code{(degree = 1)} is area.
    @param target numeric; Typically set to mean, but does not have to be. (Vectorized)
    @param variable a numeric vector.
    @return UPM of variable
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}
    @examples{R}
    set.seed(123)
    x <- rnorm(100)
    UPM(0, mean(x), x)
    @export
    """

  if(degree == 0) return(mean(variable > target))

  sum(((variable[variable > target]) - target) ^ degree) / length(variable)
