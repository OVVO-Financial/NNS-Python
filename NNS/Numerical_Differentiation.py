# -*- coding: utf-8 -*-
from typing import Callable

import numpy as np
import scipy.linalg
import scipy.optimize


def NNS_diff(
    f: Callable,
    point: float,
    h: float = 0.1,
    tol: float = 1e-10,
    digits: int = 12,  # not in use
    print_trace: bool = False,
) -> dict:
    r"""NNS Numerical Differentiation

    Determines numerical derivative of a given univariate function using projected secant lines on the y-axis.  These projected points infer finite steps \code{h}, in the finite step method.

    @param f an expression or call or a formula with no lhs.
    @param point numeric; Point to be evaluated for derivative of a given function \code{f}.
    @param h numeric [0, ...]; Initial step for secant projection.  Defaults to \code{(h = 0.1)}.
    @param tol numeric; Sets the tolerance for the stopping condition of the inferred \code{h}.  Defaults to \code{(tol = 1e-10)}.
    @param digits numeric; Sets the number of digits specification of the output.  Defaults to \code{(digits = 12)}.
    @param print.trace logical; \code{FALSE} (default) Displays each iteration, lower y-intercept, upper y-intercept and inferred \code{h}.
    @return Returns a matrix of values, intercepts, derivatives, inferred step sizes for multiple methods of estimation.
    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995/ref=cm_sw_su_dp}
    @examples
    f <- function(x) sin(x) / x
    NNS.diff(f, 4.1)
    @export

    """

    def Finite_step(p: float, h: float) -> dict:
        f_x = f(p)
        f_x_h_min = f(p - h)
        f_x_h_pos = f(p + h)

        neg_step = (f_x - f_x_h_min) / h
        pos_step = (f_x_h_pos - f_x) / h

        return {
            "f(x-h)": neg_step,
            "f(x+h)": pos_step,
            "next": np.mean([neg_step, pos_step]),
        }

    # Bs <- numeric()
    # Bl <- numeric()
    # Bu <- numeric()
    Bs, Bl, Bu = {}, {}, {}
    ### Step 1 initalize the boundaries for B

    ### Initial step size h
    f_x = f(point)
    f_x_h = f(point - h)

    ### Y = mX + B
    (f_x - f_x_h) / h
    ((f_x - f_x_h) / h) * point
    f_x - ((f_x - f_x_h) / h) * point
    ### Initial interval for B given inputted h-step-value

    f_x_h_lower = f(point - h)
    f_x_h_upper = f(point + h)

    B1 = f_x - ((f_x - f_x_h_lower) / h) * point
    B2 = f_x - ((f_x_h_upper - f_x) / h) * point

    low_B = min(B1, B2)
    high_B = max(B1, B2)

    lower_B = low_B
    upper_B = high_B

    ## Return "Derivative Does Not Exist" if lower.B and upper.B are identical to 20 digits
    if lower_B == upper_B:
        # TODO graph
        # original_par = par(no.readonly = TRUE)
        # par(mfrow = c(1, 2))
        # plot(f, xlim = c(point - (100 * h), point + (100 * h)), col = 'blue', ylab = 'f(x)')
        # points(point, f.x, pch = 19, col = 'red')
        # plot(f, xlim = c(point - 1, point + 1), col = 'blue', ylab = 'f(x)')
        # points(point, f.x, pch = 19, col = 'red')
        # par(original.par)
        raise Exception("Derivative Does Not Exist")

    new_B = np.mean([lower_B, upper_B])
    i = 1
    while i >= 1:
        Bl[i] = lower_B
        Bu[i] = upper_B
        # TODO: understand function(x)
        # new.f <- function(x) - f.x + ((f.x - f(point - x)) / x) * point + new.B
        ###  SOLVE FOR h, we just need the negative or positive sign from the tested B
        # inferred_h = uniroot(new_f, c(-2 * h, 2 * h), extendInt="yes")["root"]

        inferred_h = scipy.optimize.root_scalar(
            f=lambda x: -f_x + ((f_x - f(point - x)) / x) * point + new_B,
            # method='bisect',
            bracket=[-2 * h, 2 * h],
        ).root

        if print_trace:
            print("Iteration", i, "h", inferred_h, "Lower B", lower_B, "Upper B", upper_B)

        Bs[i] = new_B

        ## Stop when the inferred h is within the tolerance level
        if abs(inferred_h) < tol:
            final_B = np.mean([upper_B, lower_B])
            # TODO: scipy solver
            slope = scipy.linalg.solve(point, f_x - final_B)[0]
            z = complex(real=point, imag=inferred_h)

            # TODO: graph
            # original_par <- par(no.readonly = TRUE)
            # par(mfrow=c(1, 3))
            ## Plot #1
            # plot(f, xlim = c(min(c(point - (100 * h), point + (100 * h)), 0), max(c(point - (100 * h), point + (100 * h)), 0)), col = 'azure4', ylab = 'f(x)', lwd = 2, ylim = c(min(c(min(c(B1, B2)), min(na.omit(f((point - (100 * h)) : (point + (100 * h))))))), max(c(max(na.omit(f((point - (100 * h)) : (point + (100 * h))))), max(c(B1, B2))))), main = 'f(x) and initial y-intercept range')
            # abline(h = 0, v = 0, col = 'grey')
            # points(point, f.x, pch = 19, col = 'green')
            # points(point - h, f.x.h.lower, col = ifelse(B1 == high.B, 'blue', 'red'), pch = 19)
            # points(point + h, f.x.h.upper, col = ifelse(B1 == high.B, 'red', 'blue'), pch = 19)
            # points(x = rep(0, 2), y = c(B1, B2), col = c(ifelse(B1 == high.B, 'blue', 'red'), ifelse(B1 == high.B, 'red', 'blue')), pch = 1)
            # segments(0, B1, point - h, f.x.h.lower, col = ifelse(B1 == high.B, 'blue','red'), lty = 2)
            # segments(0, B2, point + h, f.x.h.upper, col = ifelse(B1 == high.B, 'red','blue'), lty = 2)

            ### Plot #2
            # plot(f, col = 'azure4', ylab = 'f(x)', lwd = 3, main = 'f(x) narrowed range and secant lines', xlim = c(min(c(point - h, point + h,  0)), max(c(point + h,point - h, 0))), ylim= c(min(c(B1, B2, f.x.h.lower, f.x.h.upper)), max(c(B1, B2, f.x.h.lower, f.x.h.upper))))
            # abline(h = 0, v = 0, col = 'grey')
            # points(point,f.x, pch = 19, col = 'red')
            # points(point - h, f.x.h.lower, col = ifelse(B1 == high.B, 'blue', 'red'), pch = 19)
            # points(point + h, f.x.h.upper, col = ifelse(B1 == high.B, 'red', 'blue'), pch = 19)
            # points(point, f.x, pch = 19, col = 'green')
            # segments(0, B1, point - h, f.x.h.lower, col = ifelse(B1 == high.B, 'blue', 'red'), lty = 2)
            # segments(0, B2, point + h, f.x.h.upper, col = ifelse(B1 == high.B, 'red', 'blue'), lty = 2)
            # points(x = rep(0, 2), y = c(B1, B2), col = c(ifelse(B1 == high.B, 'blue', 'red'), ifelse(B1 == high.B, 'red', 'blue')), pch = 1)

            ## Plot #3
            # plot(Bs, ylim = c(min(c(Bl, Bu)), max(c(Bl, Bu))), xlab = "Iterations", ylab = "y-inetercept", col = 'green', pch = 19, main = 'Iterated range of y-intercept')
            # points(Bl, col = 'red', ylab = '')
            # points(Bu, col = 'blue', ylab = '')
            # legend('topright', c("Upper y-intercept", "Lower y-intercept", "Mean y-intercept"), col = c('blue', 'red', 'green'), pch = c(1, 1, 19), bty = 'n')
            # par(original.par)
            ret = Finite_step(point, h)
            ret.update(
                {
                    "Value of f(x) at point": f(point),
                    "Final y-intercept (B)": final_B,
                    "DERIVATIVE": slope,
                    "Inferred.h": inferred_h,
                    "iterations": i,
                    "Averaged Finite Step Initial h": Finite_step(point, h)["next"],
                    "Inferred h": Finite_step(point, inferred_h),
                    "Inferred h Averaged Finite Step": Finite_step(point, inferred_h)["next"],
                    "Complex Step Derivative (Initial h)": complex(f(z)).imag / complex(z).imag,
                }
            )
            return ret

        ## NARROW THE RANGE OF B BASED ON SIGN OF INFERRED.H
        if B1 == high_B:
            if np.sign(inferred_h) < 0:
                lower_B = new_B
                upper_B = upper_B
            else:
                upper_B = new_B
                lower_B = lower_B
        else:
            if np.sign(inferred_h) < 0:
                lower_B = lower_B
                upper_B = new_B
            else:
                upper_B = upper_B
                lower_B = new_B

        new_B = np.mean([lower_B, upper_B])
        i += 1


__all__ = ["NNS_diff"]
