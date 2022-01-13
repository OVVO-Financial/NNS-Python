# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def NNS_part(
    x: [np.ndarray, pd.Series], y: [np.ndarray, pd.Series],
    Voronoi: bool=False,
    _type=None,
    order=None,
    obs_req: int = 8,
    min_obs_stop: bool = True,
    noise_reduction: str = "off"
):
    """
    NNS Partition Map

    Creates partitions based on partial moment quadrant centroids, iteratively assigning identifications to observations based on those quadrants (unsupervised partitional and hierarchial clustering method).  Basis for correlation, dependence \link{NNS.dep}, regression \link{NNS.reg} routines.

    @param x a numeric vector.
    @param y a numeric vector with compatible dimensions to \code{x}.
    @param Voronoi logical; \code{FALSE} (default) Displays a Voronoi type diagram using partial moment quadrants.
    @param _type \code{NULL} (default) Controls the partitioning basis.  Set to \code{(type = "XONLY")} for X-axis based partitioning.  Defaults to \code{NULL} for both X and Y-axis partitioning.
    @param order integer; Number of partial moment quadrants to be generated.  \code{(order = "max")} will institute a perfect fit.
    @param obs_req integer; (8 default) Required observations per cluster where quadrants will not be further partitioned if observations are not greater than the entered value.  Reduces minimum number of necessary observations in a quadrant to 1 when \code{(obs.req = 1)}.
    @param min_obs_stop logical; \code{TRUE} (default) Stopping condition where quadrants will not be further partitioned if a single cluster contains less than the entered value of \code{obs.req}.
    @param noise_reduction the method of determining regression points options for the dependent variable \code{y}: ("mean", "median", "mode", "off"); \code{(noise.reduction = "mean")} uses means for partitions.  \code{(noise.reduction = "median")} uses medians instead of means for partitions, while \code{(noise.reduction = "mode")} uses modes instead of means for partitions.  Defaults to \code{(noise.reduction = "off")} where an overall central tendency measure is used, which is the default for the independent variable \code{x}.
    @return Returns:
     \itemize{
      \item{\code{"dt"}} a \link{data.table} of \code{x} and \code{y} observations with their partition assignment \code{"quadrant"} in the 3rd column and their prior partition assignment \code{"prior.quadrant"} in the 4th column.
      \item{\code{"regression.points"}} the \link{data.table} of regression points for that given \code{(order = ...)}.
      \item{\code{"order"}}  the \code{order} of the final partition given \code{"min.obs.stop"} stopping condition.
     }

    @note \code{min.obs.stop = FALSE} will not generate regression points due to unequal partitioning of quadrants from individual cluster observations.

    @author Fred Viole, OVVO Financial Systems
    @references Viole, F. and Nawrocki, D. (2013) "Nonlinear Nonparametric Statistics: Using Partial Moments"
    \url{https://www.amazon.com/dp/1490523995/ref=cm_sw_su_dp}
    @examples
    set.seed(123)
    x <- rnorm(100) ; y <- rnorm(100)
    NNS.part(x, y)

    ## Data.table of observations and partitions
    NNS.part(x, y, order = 1)$dt

    ## Regression points
    NNS.part(x, y, order = 1)$regression.points

    ## Voronoi style plot
    NNS.part(x, y, Voronoi = TRUE)

    ## Examine final counts by quadrant
    DT <- NNS.part(x, y)$dt
    DT[ , counts := .N, by = quadrant]
    DT
    @export
    """


    noise_reduction = noise_reduction.lower()
    if noise_reduction not in ["mean", "median", "mode", "off", "mode_class"]:
        raise Exception("Please ensure noise_reduction is from  [mean, median, mode, off, mode_class]")

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    if obs_req is None:
        obs_req = 8

    if not order is None and order == 0:
        order = 1
    if Voronoi:
        # TODO
        x.label = deparse(substitute(x))
        y.label = deparse(substitute(y))

    # TODO
    PART = data.table::data.table(
        x, y,
        quadrant = "q",
        prior.quadrant = "pq"
    )[, `:=`(counts, .N), by = "quadrant"][, `:=`(old.counts, .N), by = "prior.quadrant"]

    if Voronoi:
        # TODO: matplotlib
        plot(x, y, col = "steelblue", cex.lab = 1.5, xlab = x.label, ylab = y.label)

    if x.shape[0] <= 8:
        if order is None:
            order = 1
            hard_stop = max(np.ceil(np.log(x.shape[0], 2)), 1)
        else:
            obs_req = 0
            hard_stop = x.shape[0]

    if order is None:
        order = np.max(np.ceil(np.log(x.shape[0], 2)), 1)

    # TODO is numeric?
    #if(!is.numeric(order)):
    if False:
        obs_req = 0
        hard_stop = np.max(np.ceil(np.log(x.shape[0], 2)), 1) + 2
    else:
        #obs_req = obs_req
        hard_stop = 2*np.max(np.ceil(np.log(x.shape[0], 2)), 1) + 2

    RP = None
    if _type is None:
        i = 0
        while (i >= 0):
            if i == order or i == hard_stop:
                break
            # TODO
            PART[counts >= obs_req, `:=`(counts, .N), by = quadrant]
            PART[old_counts >= obs_req, `:=`(old_counts, .N), by = prior_quadrant]
            l_PART = max(PART$counts)

            obs_req_rows = PART[counts >= obs_req, which = TRUE]
            old_obs_req_rows = PART[old_counts >= obs_req, which = TRUE]

            if len(obs_req_rows)==0:
                break
            if min_obs_stop and obs_req > 0 and (len(obs_req_rows) < len(old_obs_req_rows)):
                break

            if noise_reduction == "off":
                if Voronoi:
                    if l_PART > obs_req:
                        PART[obs_req_rows, {
                                segments(min(x), gravity(y), max(x), gravity(y), lty = 3)
                                segments(gravity(x), min(y), gravity(x), max(y), lty = 3)
                        }, by = quadrant]
                 RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = gravity), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            if noise.reduction == "mean":
                if Voronoi:
                    if l_PART > obs_req:
                        PART[
                            obs.req.rows, {
                            segments(min(x), mean(y), max(x), mean(y), lty = 3)
                            segments(gravity(x), min(y), gravity(x), max(y), lty = 3)
                        }, by = quadrant]
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = mean), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            if noise_reduction == "median":
                if Voronoi:
                    if l_PART > obs_req:
                        PART[
                            obs.req.rows, {
                                segments(min(x), median(y), max(x), median(y), lty = 3)
                                segments(gravity(x), min(y), gravity(x), max(y), lty = 3)
                            },
                            by = quadrant
                        ]
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = median), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            if noise_reduction == "mode":
                if Voronoi:
                    if l_PART > obs_req:
                        PART[
                            obs.req.rows, {
                                segments(min(x), mode(y), max(x), mode(y), lty = 3)
                                segments(gravity(x), min(y), gravity(x), max(y), lty = 3)
                            },
                            by = quadrant
                        ]
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = mode), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            if noise_reduction == "mode_class":
                if Voronoi:
                    if l_PART > obs_req:
                        PART[
                            obs.req.rows, {
                                segments(min(x), mode_class(y), max(x), mode_class(y), lty = 3)
                                segments(gravity_class(x), min(y), gravity_class(x), max(y), lty = 3)
                            },
                            by = quadrant
                        ]
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity_class, y = mode_class), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            RP[, `:=`(prior.quadrant, (quadrant))]
            PART[obs.req.rows, `:=`(prior.quadrant, (quadrant))]
            old_parts = length(unique(PART$quadrant))

            PART[
                RP,
                on = .(quadrant), `:=`(
                    q_new, {
                        lox = x.x <= i.x
                        loy = x.y <= i.y
                        1L + lox + loy * 2L
                    }
               )
            ]
            PART[
                obs.req.rows,
                `:=`(quadrant, paste0(quadrant, q_new))
            ]
            new_parts = len(np.unique(PART$quadrant))
            if min(PART$counts) <= obs_req) and i >= 1:
                break
            i += 1

        if RP is None:
            RP = PART[, c("quadrant", "x", "y")]
        # TODO: is.numeric(order) ?
        if (!is.numeric(order) || is.null(dim(RP))):
            RP = PART[, c("quadrant", "x", "y")]
        else:
            RP[, `:=`(prior.quadrant = NULL)]

        PART[, `:=`(counts = NULL, old.counts = NULL, q_new = NULL)]
        RP = data.table::setorder(RP[], quadrant)[]
        if Voronoi:
            # TODO: Matplotlib
            title(main = paste0("NNS Order = ", i), cex.main = 2)
            if min_obs_stop:
                points(RP$x, RP$y, pch = 15, lwd = 2, col = "red")
        if min_obs_stop == False:
            RP = None
        return{
            "order": i,
            "dt": PART[],
            "regression.points": RP
        }

    if not _type is None:
        i = 0
        while i >= 0:
            if i == order or i == hard_stop:
                break
            PART[counts > obs.req/2, `:=`(counts, .N), by = quadrant]
            PART[old.counts > obs.req/2, `:=`(old.counts, .N), by = prior.quadrant]
            obs_req_rows = PART[counts > obs.req/2, which = TRUE]
            old_obs_req_rows = PART[old.counts > obs.req/2, which = TRUE]

            if len(obs_req_rows) == 0:
                break
            if obs_req > 0 and (len(obs_req_rows) < len(old_obs_req_rows)):
                break
            if noise_reduction == "off":
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = gravity), .SD),
                    by = quadrant,
                   .SDcols = x:y
                ]

            if noise_reduction == "mean":
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = mean), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            if noise_reduction == "mode":
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = mode), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            if noise_reduction == "mode_class":
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity_class, y = mode_class), .SD),
                    by = quadrant,
                    .SDcols=x:y
                ]

            if noise_reduction == "median":
                RP = PART[
                    obs.req.rows,
                    mapply(function(f,z) as.list(f(z)), list(x = gravity, y = median), .SD),
                    by = quadrant,
                    .SDcols = x:y
                ]

            RP[, `:=`(prior.quadrant, (quadrant))]
            PART[obs.req.rows, `:=`(prior.quadrant, (quadrant))]
            old_parts = len(np.unique(PART$quadrant))
            PART[
                RP,
                on = .(quadrant), `:=`(q_new, {
                    lox = x.x > i.x
                    1L + lox
                })
            ]
            PART[
                obs_req_rows,
                `:=`(quadrant, paste0(quadrant, q_new))
            ]
            new_parts = len(np.unique(PART$quadrant))
            if min(PART$counts) <= obs_req and i >= 1:
                break
            i += 1

        if RP is None:
            RP = PART[, c("quadrant", "x", "y")]
        # TODO is.numeric(order)?
        if not is_numeric(order) or is_null(dim(RP)):
            RP = PART[, c("quadrant", "x", "y")]
        else:
            RP[, `:=`(prior.quadrant = NULL)]

        PART[, `:=`(counts = NULL, old.counts = NULL, q_new = NULL)]
        RP = data.table::setorder(RP[], quadrant)[]
        if np.mean([
                len(np.unique(np.diff(x))),
                len(np.unique(x))
            ]) < .33 * len(x):
            RP$x = ifelse(RP$x%%1 < .5, floor(RP$x), ceiling(RP$x))

        if Voronoi:
            abline(v = c(PART[ ,min(x), by=prior.quadrant]$V1,max(x)), lty = 3)
            if min_obs_stop:
                points(RP$x, RP$y, pch = 15, lwd = 2, col = "red")
            title(main = paste0("NNS Order = ", i), cex.main = 2)
        if min_obs_stop is False:
            RP = None
        return {
            "order": i,
            "dt": PART[],
            "regression.points": RP
        }
    return {}


__all__ = ["NNS_part"]
