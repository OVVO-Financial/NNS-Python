# NNS-Python
Nonlinear Nonparametric Statistics

From R Version of 2020-07-26 (Version: 0.5.4.3, Date: 2020-07-25) 

Implemented Functions:

* dy_dx
    * dy_dx: TODO
    
* FSD
    * NNS_FSD: TODO
    
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
    * LPM_VaR: TODO (tdigest)
    * UPM_VaR: TODO (tdigest)

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
    * PM_matrix: OK Tested
    * LPM_ratio: OK Tested
    * UPM_ratio: OK Tested
    * NNS_PDF: Todo (d/dx approximation, density)
    * NNS_CDF: Todo (ecdf, density, matplotlib, NNS_reg)

* SD Efficient Set
    * NNS_SD_efficient_set: TODO
    
* SSD
    * NNS_SSD: TODO (matplotlib, numpy friendly)
    
* TSD
    * NNS_TSD: TODO (matplotlib, numpy friendly)

* Uni SD Routines
    * NNS_FSD_uni: Must Test (TODO: numpy friendly)
    * NNS_SSD_uni: Must Test (TODO: numpy friendly)
    * NNS_TSD_uni: Must Test (TODO: numpy friendly)
    
