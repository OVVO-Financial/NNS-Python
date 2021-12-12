# NNS-Python
Nonlinear Nonparametric Statistics

From R Version of 2020-07-26 (Version: 0.5.4.3, Date: 2020-07-25) 

Implemented Functions:

* ANOVA
    * NNS.ANOVA: TODO (deps: NNS.ANOVA.bin)
    
* ARMA
    * NNS.ARMA: TODO (deps: NNS.seas, ARMA.seas.weighting, NNS.meboot)
    
* ARMA_optim:
    * NNS.ARMA.optim: TODO (deps: NNS.ARMA)
    
* Binary_ANOVA
    * NNS.ANOVA.bin: OK

* Boost
    * NNS.boost: TODO (deps: NNS.caus, NNS.reg, NNS.stack)
    
* Causal_Matrix
    * NNS.caus.matrix: TODO (deps: NNS.caus)

* Causation
    * NNS.caus: TODO (deps: Uni.caus, NNS.caus.matrix)

* Copula
    * NNS.copula: OK
    
* Dependence
    * NNS.dep: TODO (deps: NNS.part, NNS.dep.matrix)
    
* Dependence_matrix
    * NNS.dep.matrix: TODO (deps: NNS.dep)
    
* dy_d_wrt
    * dy.d_: TODO (deps: NNS.reg)

* dy_dx
    * dy_dx: TODO (deps: NNS.dep, NNS.reg)

* Internal Functions
    * mode: TEST
    * mode_class: TODO
    * gravity: TEST
    * gravity_class: TODO
    * factor_2_dummy: TODO
    * factor_2_dummy_FR: TODO
    * generate_vectors: TODO
    * ARMA_seas_weighting: TODO
    * is.discrete: TODO
    * lag_mtx: TODO
    * NNS_meboot_part: TODO
    * NNS_meboot_expand_sd: TODO
    * alt_cbind: TEST (not in newest version, maybe R related)
    * RP: TODO (not in newest version)

* LPM UPM VaR
    * LPM_VaR: OK
    * UPM_VaR: OK
    * used np.quantile instead of tdigest, and root_scalar instead of optimize

* Multivariate_Regression
    * NNS.M.reg: TODO (deps: NNS.part, NNS.dep, NNS::NNS.distance, NNS.copula, NNS.reg)

* NNS_Distance
    * NNS.distance: TODO (deps: dtw, Rfast)

* NNS_meboot
    * NNS.meboot: TODO (deps: NNS.dep, NNS.meboot.expand.sd)
    
* NNS_term_matrix
    * NNS.term.matrix: OK

* NNS_VAR
    * NNS.VAR: TODO (deps: NNS.reg, NNS.seas, NNS.ARMA.optim, NNS.ARMA, NNS.stack, NNS.dep, NNS.caus)

* Normalization
    * NNS.norm: TODO (deps: NNS.dep, Rfast)

* Nowcast
    * NNS.nowcast: TODO (deps: Quandl, NNS.VAR)

* Numerical Differentiation
    * NNS.diff: TODO (nodeps)
    
* Partition_Map
    * NNS.part: TODO (deps: internal functions: gravity_class, gravity, mode_class)

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
    * NNS_PDF: TODO (deps: d/dx approximation, density)
    * NNS_CDF: TODO (deps: ecdf, density, matplotlib, NNS_reg)

* Regression
    * NNS.reg: TODO (deps: NNS.M.reg, NNS.dep, NNS.part, Uni.caus)

* SD Efficient Set
    * NNS_SD_efficient_set: OK (TODO: numba version?)

* Seasonality_Test
    * NNS.seas: TODO (nodeps)
    
* Stack
    * NNS.stack: TODO (deps: NNS.reg, NNS::NNS.distance)
    
* Uni_Causation
    * Uni.caus: TODO (deps: NNS.norm, NNS.dep)
    
* FSD, SSD, TSD
    * NNS_FSD: OK (TODO: numba version?)
    * NNS_SSD: OK (TODO: numba version?)
    * NNS_TSD: OK (TODO: numba version?)

* Uni SD Routines
    * NNS_FSD_uni: OK (TODO: numba version?)
    * NNS_SSD_uni: OK (TODO: numba version?)
    * NNS_TSD_uni: OK (TODO: numba version?)
