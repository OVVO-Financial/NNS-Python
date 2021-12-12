# NNS-Python
Nonlinear Nonparametric Statistics

From R Version of 2020-07-26 (Version: 0.5.4.3, Date: 2020-07-25) 

Implemented Functions:

* dy_dx
    * dy_dx: TODO
    
* Internal Functions
    * mode: Must TEST
    * mode_class: TODO
    * gravity: Must TEST
    * alt_cbind: Must TEST
    * factor_2_dummy: TODO
    * factor_2_dummy_FR: TODO
    * generate_vectors: TODO
    * ARMA_seas_weighting: TODO
    * lag_mtx: TODO
    * RP: TODO
    * NNS_meboot_part: TODO
    * NNS_meboot_expand_sd: TODO

* LPM UPM VaR
    * LPM_VaR: OK (used np.quantile instead of tdigest and root_scalar instead of optimize)
    * UPM_VaR: OK

* Partial Moments
    * pd_fill_diagonal: OK (Internal use)
    * LPM: OK Tested
        * numba_LPM: Numba version (Internal use)
        * LPM: Vectorized / pandas / numpy friendly
    * UPM: OK Tested
        * numba_UPM: Numba version (Internal use)
        * UPM: Vectorized / pandas / numpy friendly
    * Co_UPM: OK Tested
        * _Co_UPM: Internal Use
        * _vec_Co_UPM: numpy.vectorized
        * Co_UPM: Vectorized / pandas / numpy friendly
    * Co_LPM: OK Tested
        * _Co_LPM: Internal Use
        * _vec_Co_LPM: numpy.vectorized
        * Co_LPM: Vectorized / pandas / numpy friendly
    * D_LPM: OK Tested
        * _D_LPM: Internal User
        * _vec_D_LPM: numpy.vectorized
        * D_LPM: Vectorized / pandas / numpy friendly 
    * D_UPM: OK Tested
        * _D_UPM: Internal User
        * _vec_D_UPM: numpy.vectorized
        * D_UPM: Vectorized / pandas / numpy friendly 
    * PM_matrix: OK
    * LPM_ratio: OK
    * UPM_ratio: OK
    * NNS_PDF: Todo (d/dx approximation, density)
    * NNS_CDF: Todo (ecdf, density, matplotlib, NNS_reg)

* SD Efficient Set
    * NNS_SD_efficient_set: OK
    
* FSD, SSD, TSD
    * NNS_FSD: OK
    * NNS_SSD: OK
    * NNS_TSD: OK

* Uni SD Routines
    * NNS_FSD_uni: OK
    * NNS_SSD_uni: OK
    * NNS_TSD_uni: OK
