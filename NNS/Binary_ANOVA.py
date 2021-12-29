from typing import Union, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .LPM_UPM_VaR import LPM_VaR, UPM_VaR
from .Partial_Moments import LPM_ratio, UPM_ratio


def NNS_ANOVA_bin(
    control: [pd.Series, np.ndarray],
    treatment: [pd.Series, np.ndarray],
    mean_of_means: [float, None] = None,
    upper_25_target: [float, None] = None,
    lower_25_target: [float, None] = None,
    upper_125_target: [float, None] = None,
    lower_125_target: [float, None] = None,
    confidence_interval: [float, None] = None,
    tails: [str, None] = None,
    plot: bool = True,
    par=None,  # NO USE
) -> dict:
    if tails is None:
        tails = "both"
    if tails not in [None, "both", "left", "right"]:
        raise Exception("Tails must be None, both, left or right")

    if upper_25_target is None and lower_25_target is None:
        mean_of_means = np.mean([np.mean(control), np.mean(treatment)])
        upper_25_target = np.mean([UPM_VaR(0.25, 1, control), UPM_VaR(0.25, 1, treatment)])
        lower_25_target = np.mean([LPM_VaR(0.25, 1, control), LPM_VaR(0.25, 1, treatment)])
        upper_125_target = np.mean([UPM_VaR(0.125, 1, control), UPM_VaR(0.125, 1, treatment)])
        lower_125_target = np.mean([LPM_VaR(0.125, 1, control), LPM_VaR(0.125, 1, treatment)])

    # Continuous CDF for each variable from Mean of Means
    LPM_ratio_1 = LPM_ratio(1, mean_of_means, control)
    LPM_ratio_2 = LPM_ratio(1, mean_of_means, treatment)

    Upper_25_ratio_1 = UPM_ratio(1, upper_25_target, control)
    Upper_25_ratio_2 = UPM_ratio(1, upper_25_target, treatment)
    # Upper_25_ratio = np.mean([Upper_25_ratio_1, Upper_25_ratio_2])

    Lower_25_ratio_1 = LPM_ratio(1, lower_25_target, control)
    Lower_25_ratio_2 = LPM_ratio(1, lower_25_target, treatment)
    # Lower_25_ratio = np.mean([Lower_25_ratio_1, Lower_25_ratio_2])

    Upper_125_ratio_1 = UPM_ratio(1, upper_125_target, control)
    Upper_125_ratio_2 = UPM_ratio(1, upper_125_target, treatment)
    # Upper_125_ratio = np.mean([Upper_125_ratio_1, Upper_125_ratio_2])

    Lower_125_ratio_1 = LPM_ratio(1, lower_125_target, control)
    Lower_125_ratio_2 = LPM_ratio(1, lower_125_target, treatment)
    # Lower_125_ratio = np.mean([Lower_125_ratio_1, Lower_125_ratio_2])

    # Continuous CDF Deviation from 0.5
    MAD_CDF = min(0.5, max(abs(LPM_ratio_1 - 0.5), abs(LPM_ratio_2 - 0.5)))
    upper_25_CDF = min(0.25, max(abs(Upper_25_ratio_1 - 0.25), abs(Upper_25_ratio_2 - 0.25)))
    lower_25_CDF = min(0.25, max(abs(Lower_25_ratio_1 - 0.25), abs(Lower_25_ratio_2 - 0.25)))
    upper_125_CDF = min(0.125, max(abs(Upper_125_ratio_1 - 0.125), abs(Upper_125_ratio_2 - 0.125)))
    lower_125_CDF = min(0.125, max(abs(Lower_125_ratio_1 - 0.125), abs(Lower_125_ratio_2 - 0.125)))

    # Certainty associated with samples
    NNS_ANOVA_rho = (
        np.sum(
            [
                ((0.5 - MAD_CDF) ** 2) / 0.25,
                0.5 * (((0.25 - upper_25_CDF) ** 2) / 0.25 ** 2),
                0.5 * (((0.25 - lower_25_CDF) ** 2) / 0.25 ** 2),
                0.25 * (((0.125 - upper_125_CDF) ** 2) / 0.125 ** 2),
                0.25 * (((0.125 - lower_125_CDF) ** 2) / 0.125 ** 2),
            ]
        )
        / 2.5
    )

    pop_adjustment = ((len(control) + len(treatment) - 2) / (len(control) + len(treatment))) ** 2

    # Graphs
    if plot:
        plt.title("NNS ANOVA and Effect Side")
        plt.boxplot(
            [control, treatment],
            labels=["Control", "Treatment"],
            vert=False,
        )
        plt.xlabel("Means")
        plt.axvline(mean_of_means, color="red", linewidth=4, label="Grand Mean")
        # if par is None:
        #    original_par = par(no.readonly = TRUE)
        # else:
        #    original_par = par
        # boxplot(list(control, treatment), las = 2, names = c("Control", "Treatment"), xlab = "Means", horizontal = TRUE, main = "NNS ANOVA and Effect Size", col = c("grey", "white"), cex.axis = 0.75)
        # For ANOVA Visualization
        # abline(v = mean_of_means, col = "red", lwd = 4)
        # mtext("Grand Mean", side = 3, col = "red", at = mean_of_means)

    if confidence_interval is None:
        return {
            "Control Mean": np.mean(control),
            "Treatment Mean": np.mean(treatment),
            "Grand Mean": mean_of_means,
            "Control CDF": LPM_ratio_1,
            "Treatment CDF": LPM_ratio_2,
            "Certainty": min(1, NNS_ANOVA_rho * pop_adjustment),
        }
    else:
        # Upper end of CDF confidence interval for control mean
        Upper_Bound_Effect, Lower_Bound_Effect, CI = None, None, 0
        if tails == "both":
            CI = confidence_interval + (1 - confidence_interval) / 2
        elif tails in ["left", "right"]:
            CI = confidence_interval
        a = UPM_VaR(1 - CI, 1, control)
        b = np.mean(control)
        if plot:
            if tails in ["both", "right"]:
                plt.axvline(max(a, b), color="green", linewidth=4, label="mu+", linestyle=":")
                # abline(v = max(a, b), col = "green", lwd = 4, lty = 3)
                # text(max(a, b), pos = 2, 0.75, "mu+", col = "green")
                # text(max(a, b), pos = 4, 0.75, paste0((1 - CI) * 100, "% --->"), col = "green")}

        # Lower end of CDF confidence interval for control mean
        c = LPM_VaR(1 - CI, 1, control)
        d = np.mean(control)

        if plot:
            if tails in ["both", "left"]:
                plt.axvline(min(c, d), color="blue", linewidth=4, label="mu-", linestyle=":")
                # abline(v = min(c, d), col = "blue", lwd = 4, lty = 3)
                # text(min(c, d), pos = 4, 0.75, "mu-", col = "blue")
                # text(min(c, d), pos=2, 0.75, paste0( "<--- ", (1 - CI) * 100, "%"), col = 'blue')}
            # par(original.par)

        # Effect Size Lower Bound
        if tails in ["both", "right"]:
            Lower_Bound_Effect = np.mean(treatment) - max(a, b)
        elif tails == "left":
            Lower_Bound_Effect = np.mean(treatment) - max(c, d)

        # Effect Size Upper Bound
        if tails in ["both", "left"]:
            Upper_Bound_Effect = np.mean(treatment) - min(c, d)
        elif tails == "right":
            Upper_Bound_Effect = np.mean(treatment) - min(a, b)

        # Certainty Statistic and Effect Size Given Confidence Interval
        return {
            "Control Mean": np.mean(control),
            "Treatment Mean": np.mean(treatment),
            "Grand Mean": mean_of_means,
            "Control CDF": LPM_ratio_1,
            "Treatment CDF": LPM_ratio_2,
            "Certainty": min(1, NNS_ANOVA_rho * pop_adjustment),
            "Lower Bound Effect": Lower_Bound_Effect,
            "Upper Bound Effect": Upper_Bound_Effect,
        }


__all__ = ["NNS_ANOVA_bin"]
