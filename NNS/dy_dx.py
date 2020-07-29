# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# TODO TEST dydx
def dy_dx(x: pd.Series, y: pd.Series, eval_point: [float, int, str, None] = None, deriv_method: str = "FD") -> dict:
    r"""
    Partial Derivative dy/dx

    Returns the numerical partial derivate of \code{y} wrt \code{x} for a point of interest.

    @param x a numeric vector.
    @param y a numeric vector.
    @param eval.point numeric or ("overall"); \code{x} point to be evaluated.  Defaults to \code{(eval.point = median(x))}.  Set to \code{(eval.point = "overall")} to find an overall partial derivative estimate (1st derivative only).
    @param deriv.method method of derivative estimation, options: ("NNS", "FD"); Determines the partial derivative from the coefficient of the \link{NNS.reg} output when \code{(deriv.method = "NNS")} or generates a partial derivative using the finite difference method \code{(deriv.method = "FD")} (Defualt).
    @return Returns a list of both 1st and 2nd derivative:
    \itemize{
    \item{\code{dy.dx(...)$First}} the 1st derivative.
    \item{\code{dy.dx(...)$Second}} the 2nd derivative.
    }

    @note If a vector of derivatives is required, ensure \code{(deriv.method = "FD")}.
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995}

    Vinod, H. and Viole, F. (2017) "Nonparametric Regression Using Clusters"
    \url{https://link.springer.com/article/10.1007/s10614-017-9713-5}

    @examples
    \dontrun{
    x <- seq(0, 2 * pi, pi / 100) ; y <-sin(x)
    dy.dx(x, y, eval.point = 1.75)

    # Vector of derivatives
    dy.dx(x, y, eval.point = c(1.75, 2.5), deriv.method = "FD")}
    @export
    """
    if eval_point is None:
        eval_point = x.median()
    order = None
    # TODO NNS_dep
    dep = NNS_dep(x, y)['Dependence']
    if(dep > 0.85):
        h = 0.01
    elif(dep > 0.5):
        h = 0.05
    else:
        h = 0.2

    if isinstance(x, pd.Series)::
        x = pd.DataFrame(x, columns=[x.name])

    # TODO type of eval_point
    if (len(eval_point) > 1 and deriv_method == "NNS"):
        deriv_method = "FD"

    if isinstance(eval_point, str):
        # TODO NNS_reg
        ranges = NNS_reg(x, y, order = order, plot = False)['derivative']
        # TODO Convert
        ranges[ , interval := seq(1 : len(ranges['Coefficient']))]

        # TODO Convert
        range_weights = pd.DataFrame(x, 'interval' = findInterval(x, ranges[: , X_Lower_Range]))

        # TODO Convert
        ranges = ranges[interval in range_weights['interval'], ]

        # TODO Convert
        range_weights = range_weights[ , .N, by = 'interval']

        # TODO Convert
        range_weights = range_weights['N'] / range_weights['N'].sum()

        # TODO Convert
        return {
            "First": (ranges[:,Coefficient]*range_weights).sum()
        }

    original_eval_point_min = eval_point
    original_eval_point_max = eval_point

    # TODO Convert
    h_step = LPM_ratio(1, unlist(eval_point), x)
    # TODO Convert
    h_step = LPM_VaR(h_step + h, 1, x) - LPM_VaR(h_step - h, 1, x)

    # TODO Convert
    eval_point_min = original_eval_point_min - h_step

    # TODO Convert
    eval_point_max = h_step + original_eval_point_max

    # TODO Convert
    deriv_points = cbind(eval_point_min, eval_point, eval_point.max)

    # TODO Convert
    n = dim(deriv_points)[1]

    # TODO Convert
    run = eval_point_max - eval_point_min

    if (run==0).any():
        # TODO Convert
        z = which(run == 0)
        eval_point_max[z] = (abs((max(x) - min(x)) * h)) + eval_point[z]
        eval_point_max[z] = eval_point[z] - (abs((max(x) - min(x)) * h))
        run[z] = eval_point_max[z] - eval_point_min[z]

    # TODO Convert + NNS_reg
    reg_output = NNS_reg(x, y, plot = False, return_values = True, order = order, point_est = pd.Series(deriv_points))

    # TODO Convert
    estimates_min = reg_output['Point.est'][1:n]
    estimates_max = reg_output['Point.est'][(2*n+1):(3*n)]
    estimates = reg_output['Point.est'][(n+1):(2*n)]

    if deriv_method == "FD":
        # TODO Convert
        rise = estimates_max - estimates_min
        first_deriv =  rise / run
    else:
        # TODO Convert
        output = reg_output['derivative']
        if (len(output[: , Coefficient]) == 1):
            first_deriv = output[: , Coefficient]
        if((output[ , X_Upper_Range][which(eval.point < output[ , X_Upper_Range]) - 1][1]) < eval_point):
            first_deriv =  output[: , Coefficient][which(eval_point < output[: , X_Upper_Range])][1]
        else:
            first_deriv =  mean(c(output[: , Coefficient][which(eval_point < output[: , X_Upper_Range])][1], output[: , X_Lower_Range][which(eval_point < output[: , X_Upper_Range]) - 1][1]))

    # TODO Convert
    ## Second derivative form:
    # [f(x+h) - 2(f(x)) + f(x-h)] / h^2
    f_x__h = estimates_min
    two_f_x = 2 * estimates
    f_x_h = estimates_max
    second_deriv = (f_x_h - two_f_x + f_x__h) / (h_step ** 2)

    return {
        "First": first_deriv,
        "Second": second_deriv
    }

