#!/usr/bin/env python

import numpy as np
import stats
import time
import copy
import matplotlib.pyplot as plt

# covariance matrix computation
def compute_cov(F, t=True):
    if t:
        return np.cov(F.T)
    else:
        return np.cov(F)

def eof_decomp(field, eigvec, neof):
    pcomp = [field @ eigvec[:,j] for j in range(len(eigvec))]
    field_reconstr = np.zeros((field.shape[0], len(eigvec)))
    for k in range(neof):
        field_reconstr += np.dot(np.expand_dims(pcomp[k], axis=0).T,
                        np.expand_dims(eigvec[:,k].T, axis=0))
    return field_reconstr

def reconstruct_field(field_tp, datai, mask_tp, mask_cv, neof):
    # EOF decomposition
    c = True
    econv = 1e-4
    e = []
    e.append(2)
    e.append(1)
    i = 1
    j = 0
    neof = 1
    itr = 0

    fields = []
    rms_cv = []
    rmscv = []
    dataii = copy.copy(field_tp)
    
    print('rmse crss-v')
    start_t = time.time()
    while e[i] < e[i-1]:
        field = copy.copy(field_tp)
        eigv, eigval, eigvt = np.linalg.svd(compute_cov(field),
                                            full_matrices=True)
        field = eof_decomp(field, eigv, neof)
        field_tp[mask_tp] = field[mask_tp]
    
        # rms computation in function of neof &/or iterations
        rms_cv.append(stats.rmse_cross_v(field[mask_cv],
                                         datai[mask_cv],
                                         datai.shape[1],
                                         50))
        print ('%0.08f' %rms_cv[j])
        rmscv.append(rms_cv[j])
        # algorithm to stop reconstruction 
        if rms_cv[j] > e[i]:
            end_t = time.time()
            print('procedure stopped! Error augmented')
            break
        j += 1
        itr += 1
        #neof += 1
        if j > 1:
            if abs(rms_cv[j-1]-rms_cv[j-2]) > econv:
                continue
            else:
                e.append(rms_cv[j-1])
                if (1 - e[i+1]/e[i]) < 0.05:
                    end_t = time.time()
                    print('procedure stopped! Minimum of error difference was reached')
                    break
                fields.append(field)
                neof += 1
                i += 1
                rms_cv = []
                j = 0
            
    print ('%d iterations - %0.06f seconds' %(itr, end_t - start_t))
    print ('%d EOFs for reconstruction' %(neof-1))
    plt.figure()
    plt.plot(rmscv)
    plt.show()
    return(fields)
