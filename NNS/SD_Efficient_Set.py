# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from .Uni_SD_Routines import NNS_FSD_uni, NNS_SSD_uni, NNS_TSD_uni
from .Partial_Moments import LPM


def NNS_SD_efficient_set(
    x: [pd.DataFrame, np.ndarray],
    degree: int,
    type_first_degree: str = "discrete",
    status: bool = True,
) -> [list, np.ndarray]:
    type_first_degree = type_first_degree.lower()

    if len(x.shape) != 2:
        raise Exception(f"x shape should contains 2 elements (dataframe like): {x.shape}")

    if type_first_degree not in ["discrete", "continuous"]:
        raise Exception("type_first_degree needs to be either 'discrete' or 'continuous'")

    if degree not in [1, 2, 3]:
        raise Exception("degree needs to be 1, 2, or 3")

    n = x.shape[1]
    max_target = np.max(np.max(x))

    Dominated_set = []
    current_base = []
    if isinstance(x, pd.DataFrame):
        LPM_order = [LPM(1, max_target, x.values[:, i]) for i in range(n)]
        LPM_order_argsort = np.argsort(LPM_order)
        final_ranked = x.values[:, LPM_order_argsort]
    else:
        LPM_order = [LPM(1, max_target, x[:, i]) for i in range(n)]
        LPM_order_argsort = np.argsort(LPM_order)
        final_ranked = x[:, LPM_order_argsort]
    current_base.append(0)
    for i in range(n - 1):
        if status:
            print(f"Checking {i} of {n-1}\r", end="")
            if i == (n - 1):
                print("                                        ", end="\n")
        base, challenger = final_ranked[:, current_base[-1]], final_ranked[:, i + 1]
        sd_test = 0
        if degree == 1:
            sd_test = NNS_FSD_uni(base, challenger, type_test=type_first_degree)
        elif degree == 2:
            sd_test = NNS_SSD_uni(base, challenger)
        elif degree == 3:
            sd_test = NNS_TSD_uni(base, challenger)
        if sd_test == 1:
            Dominated_set.append(i + 1)
        elif sd_test == 0:
            sd_found = False
            for j in current_base:
                base = final_ranked[:, j]
                new_base_sd_test = 0
                if degree == 1:
                    new_base_sd_test = NNS_FSD_uni(base, challenger, type_test=type_first_degree)
                elif degree == 2:
                    new_base_sd_test = NNS_SSD_uni(base, challenger)
                elif degree == 3:
                    new_base_sd_test = NNS_TSD_uni(base, challenger)
                if new_base_sd_test != 0:
                    sd_found = True
                    Dominated_set.append(i + 1)
                    break
            if not sd_found:
                current_base.append(i + 1)

    if len(Dominated_set) > 0:
        if isinstance(x, pd.DataFrame):
            return list(
                x.columns[
                    LPM_order_argsort[
                        [i for i in range(len(LPM_order_argsort)) if i not in Dominated_set]
                    ]
                ]
            )
        else:
            return LPM_order_argsort[
                [i for i in range(len(LPM_order_argsort)) if i not in Dominated_set]
            ]
    elif isinstance(x, pd.DataFrame):
        return list(x.columns[LPM_order_argsort])
    else:
        return LPM_order_argsort


__all__ = [
    "NNS_SD_efficient_set",
]
