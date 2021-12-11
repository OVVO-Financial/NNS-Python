# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .Uni_SD_Routines import NNS_FSD_uni, NNS_SSD_uni, NNS_TSD_uni
from .Partial_Moments import LPM


def NNS_SD_efficient_set(
    x: pd.DataFrame, degree: int, _type: str = "discrete", status: bool = True
) -> list:
    _type = _type.lower()

    if _type not in ["discrete", "continuous"]:
        raise Exception("_type needs to be either 'discrete' or 'continuous'")

    if degree not in [1, 2, 3]:
        raise Exception("degree needs to be 1, 2, or 3")

    n = x.shape[0]
    max_target = max(x)

    # LPM_order <- numeric()
    # Dominated_set <- numeric()
    # current_base <- numeric()
    Dominated_set = {}
    current_base = []

    LPM_order = [LPM(1, max_target, x[:, i]) for i in range(n)]
    LPM_order_argsort = np.argsort(LPM_order)
    final_ranked = x[:, LPM_order_argsort]
    # TODO
    current_base.append(0)
    for i in range(n):
        if status:
            print(f"Checking {i} of {n-1}\r", end="")
            if i == (n - 1):
                print("                                        ", end="\n")

        # TODO
        # base <- final_ranked[ , tail(current_base, 1)]
        base = final_ranked[:, current_base[-1]]
        challenger = final_ranked[:, i]
        sd_test = 0
        if degree == 1:
            sd_test = NNS_FSD_uni(base, challenger, _type=type)
        elif degree == 2:
            sd_test = NNS_SSD_uni(base, challenger)
        elif degree == 3:
            sd_test = NNS_TSD_uni(base, challenger)
        if sd_test == 1:
            Dominated_set[i] = i + 1
        elif sd_test == 0:
            I = False
            for j in current_base:
                base = final_ranked[:, j]
                new_base_sd_test = 0
                if degree == 1:
                    new_base_sd_test = NNS_FSD_uni(base, challenger, _type=type)
                elif degree == 2:
                    new_base_sd_test = NNS_SSD_uni(base, challenger)
                elif degree == 3:
                    new_base_sd_test = NNS_TSD_uni(base, challenger)
                if new_base_sd_test == 0:
                    I = False
                    continue
                else:
                    I = True
                    Dominated_set[i] = i + 1
                    break

            # TODO
            # if(!I) current_base <- c(current_base, i + 1)
            if not I:
                current_base.append(i + 1)

    if len(Dominated_set) > 0:
        return final_ranked.columns.drop(Dominated_set)
    return final_ranked.columns
