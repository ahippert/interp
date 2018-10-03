#!/usr/bin/env python

# eof_interp.py
#
# This code performs interpolation of a spatio-temporal field
# containing missing data. 

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import math
from scipy import signal
import time

from gen_field import *
import noisegen as ng
import gaps_gen as gg
from fdist import *
from scipy.stats import gaussian_kde, pearsonr, norm
import corr_noise as cn

from matplotlib import rcParams, cycler

#from fatm import *

# function that computes variance contained in any mode
# based on Beckers and Rixen 2001
#
def compute_variance_by_mode(n_EOF, singular_val, cov=True):
    tot_variance = 0
    variance = []
    # singular values of field matrix X are the square roots
    # of the eigenvalues of Xt*X
    if not cov :
        singular_val *= singular_val
    for i in range(0, len(singular_val)):
        tot_variance += singular_val[i]
    return [100*((singular_val[i])/tot_variance) for i in range(0, n_EOF)]

""""""""""""""""""""""""""""""""""""""
""" plot x, y data in form of bars """
""""""""""""""""""""""""""""""""""""""
def plot_bars(x, y, x_label, y_label):#, x_text, y_text):
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8
    rects1 = ax.bar(x, y, bar_width, alpha=0.6, color='b')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    #ax.text(x_text, y_text, '%.0f EOFs = %2.1f prct of variance' % (max(x),100*sum(y)))
    ax.set_title('Signal representation by mode')

# compute root mean square deviation between prediction and observation fields
#
def rmse(pred, obs, t, nx, ny, col=True):
    sum_of_dif = 0
    for i in range(0, t):
        if col :
            sum_of_dif += np.linalg.norm(pred[:,i] - obs[:,i])**2
        else :
            sum_of_dif += np.linalg.norm(pred[i] - obs[i])**2
        # same as sum_of_dif += sum(abs(pred[:,i] - obs[:,i])**2)
    return np.sqrt(sum_of_dif/(t*nx*ny))

def rmse_zero_mode(pred, obs, t, nx, ny):
    sum_of_dif = 0
    for i in range(0, t):
        sum_of_dif += np.linalg.norm(np.mean(pred[:,i])-obs[:,i])**2
    return np.sqrt(sum_of_dif/(t*nx*ny))

def rmse_cross_v(pred, obs, N):
    sum_of_dif = 0 
    for i in range(0, N):
        sum_of_dif += np.linalg.norm(pred[i] - obs[i])**2
    return np.sqrt(sum_of_dif/N)

def rmse_cv_zero_mode(pred, obs, N):
    sum_of_dif = 0 
    #for i in range(0, t) :
    #    sum_of_dif += np.linalg.norm(np.mean(pred[i*N:(i+1)*N]) - obs[i*N:(i+1)*N])**2
    for i in range(0, N):
        sum_of_dif += np.linalg.norm(np.mean(pred) - obs[i])**2
    return np.sqrt(sum_of_dif/(t*N))

def compute_image_rmse(im1, im2, nx, ny):
    rmse_im = 0
    for i in range(nx):
        for j in range(ny):
            rmse_im += np.linalg.norm(im1[i][j] - im2[i][j])**2
    return np.sqrt(rmse_im/(nx*ny))

def compute_mean(matrix, n, col, nozeros=True):
    matrix_mean = []
    for i in range(0, n):
        if col :
            if nozeros:
                matrix_mean.append(np.nanmean(matrix[:,i]))
            else:
                matrix_mean.append(np.mean(matrix[:,i]))
        else :
            if nozeros:
                matrix_mean.append(np.nanmean(matrix[i]))
            else:
                matrix_mean.append(np.mean(matrix[i]))
    return matrix_mean

def compute_mean0(matrix, n, col, nozeros=True):
    matrix_mean = []
    for i in range(0, n):
        if col:
            if nozeros:
                matrix_mean.append(matrix[:,i][matrix[:,i].nonzero()].mean())
            else:
                matrix_mean.append(np.mean(matrix[:,i]))
        else:
            if nozeros:
                matrix_mean.append(matrix[i][matrix[i].nonzero()].mean())
            else:
                matrix_mean.append(np.mean(matrix[i]))
    return matrix_mean
    

def remove_mean(matrix, t_mean, n, col):
    if col: 
        for i in range(0, n):
            matrix[:,i] -= t_mean[i]
    else:
        for i in range(0, n):
            matrix[i] -= t_mean[i]
    return matrix

def add_mean(data, mean, n, col):
    if col:
        for i in range(0, n):
            if np.isnan(mean[i]):
                data[:,i] += 0.
            else:
                data[:,i] += mean[i]
    else:
        for i in range(0, n):
            if np.isnan(mean[i]):
                data[i] += 0. 
            else:
                data[i] += mean[i]
    return data

def chi2(obs, pred, n):
    chi = 0.
    for i in range(n):
        if pred[i]==0. or obs[i]==0. or (pred[i]==0. and obs[i]!=0.):
            print ("Bad expected number in chi-2")
        else :
            chi += (1/np.var(obs))*(obs[i] - pred[i])**2
    return chi

def ftest(data1, data2):
    var1, var2 = np.var(data1), np.var(data2)
    if var1 > var2:
        return var1/var2
    else:
        return var2/var1


# covariance matrix computation
def compute_cov(F, t=True):
    if t:
        return np.cov(F.T)
    else:
        return np.cov(F)

# step 2 : svd decomposition of cov matrix
# step 3 : retrieve principal components
# step 4 : reconstruct field using k EOFs
def eof_decomp(field, eigvec, neof):
    pcomp = [field @ eigvec[:,j] for j in range(len(eigvec))]
    field_reconstr = np.zeros((field.shape[0], len(eigvec)))
    for k in range(0, neof):
        field_reconstr += np.dot(np.expand_dims(pcomp[k], axis=0).T,
                        np.expand_dims(eigvec[:,k].T, axis=0))
    return field_reconstr


#__________________________
#__________________________

# field dimensions 
nx, ny = 100, 100
nobs = nx*ny # i from 1... m : nb of points in each map

#nts = range(1, 80, 3)#[5, 10, 15, 20, 25] # j from 1... n : nb of times
nts = [30]
nt = 30
x, y = np.meshgrid(np.linspace(-1,1,nx), np.linspace(-1,1,ny)) #Grillage de l'espace
h1, k1 = 1, 1
h2, k2 = -1, -1
r1 = np.sqrt(x**2+y**2)
r2 = np.sqrt((x-h1)**2+(y-k1)**2) # Grillage des distances a l'origine
r3 = np.sqrt((x-h2)**2+(y-k2)**2)
r = r1# + r2 + r3

# time series of field
mode = 2
modes = 5 # in [2,7]
if mode == 1:
    truth = np.array([volcan(r, (t/nt)+30/nt) for t in range(10, nt+10)])
elif mode == 2:
    truth = np.array([volcan(r, (t/nt)+30/nt) +
                      volcan2(r, (t/nt)+30/nt, modes) for t in range(10, nt+10)])
elif mode > 2:
    truth = np.array([deterministic_field(r, (t/nt)+30/nt) for t in range(10, nt+10)])
else:
    truth = np.array([
        [deterministic_field(i, j, nobs, nt)
         for j in range(nt)]
        for i in range(nobs)])

# Construct and tune noise
BUILDNOISE = True
NOISE_TYPE = 'corr' # 'rand' or 'corr'
rms_all = np.array([])
sn_ratio = np.array([])
rms_nt = []
rms = []
rms_eof = []
rms_cv = []
chsq_truth = []
chsq = []
ftst = []
fields = []
eigvals = []
errdif = []
rmseof = []
rmscv = []
rmsinit = []

# Construct other noise useful for correlated gaps generation
mu, sigma = 0, 1
b = np.random.normal(mu, sigma, (nx, ny))
e = np.linspace(1.1, 1.9, nt)
corr = [ng.geo(r, e[i]) for i in range(nt)]
gaps = ng.gen_noise_series2(corr, b, nt)

pct = 9
pourcent = np.linspace(0, 80, pct)
#pourcent = [10, 20]
col = True
#init_zero = [True, False] 
inits = ['zero'] # 'noise' or 'zero'
gens = ['random', 'correlated'] # 'random' or 'correlated'
tirage = 100
    
if not BUILDNOISE: 
    noise = np.zeros((nt,nx,ny))
else:
    blanc = np.random.normal(mu, sigma, (nt, nx, ny))
    expo = 1.4 # exponent in correlation function (as: 1/(r)**expo)
    geo_noise = ng.geo(r, expo)
    if NOISE_TYPE == 'corr':
        noise = ng.gen_noise_series(geo_noise, blanc, nt)
        #noise = cn.gen_corr_noise(nt, 10, 0.01, 0.9)
    elif NOISE_TYPE == 'rand':
        noise = blanc
    
# time series of evolving noise
sn_ratio = []
inter = 9
mul = np.linspace(2, 15, inter)
#mul = [2, 3]
noises = [noise*i for i in mul]
for n in range(len(mul)): 
    # total displacement field
    data = truth + noises[n]
            
    # form initial spatio temporal field
    datai = np.reshape(data, (nt, nobs)).T

    truthi = np.reshape(copy.copy(truth), (nt, nobs)).T
    truth_mean = compute_mean(truthi, nt, col)
    ftruth = remove_mean(truthi, truth_mean, nt, col)
            
    dataimean = compute_mean(datai, nt, col)
    dataii = copy.copy(datai)
    dataii = remove_mean(dataii, dataimean, nt, col)

    for k in pourcent:
        for gen in gens:
            fdispl = copy.copy(datai)
            # 1. generate correlated gaps using noise
            if gen == 'correlated':
                t_start, t_end = 8, 14
                seuil = norm.ppf(k/100., np.mean(gaps), np.std(gaps))
                print ('seuil: %0.2f' %seuil)
                mask0 = gg.gen_correlated_gaps(gaps, seuil, t_start, t_end)
                mask0 = np.reshape(mask0, (nt, nobs)).T
                                
            # 2. Generate random gaps 
            elif gen == 'random':
                ngaps = np.arange(int(nobs*nt*k/100.))
                mask0 = gg.gen_random_gaps(np.zeros((nobs, nt), dtype=bool), nobs, nt, ngaps)
            
            # mask for cross validation
            ngaps = [30]
            ngaps_cv = [np.arange(i) for i in ngaps]
            for m in range(len(ngaps)):
                tng = ngaps[m]*nt
                mask_temp = copy.copy(mask0)
                for i in range(nt):
                    mask_temp[:,i] = gg.gen_cv_mask(mask_temp[:,i],
                                                    nobs,
                                                    ngaps_cv[m])
                                
                # Generate mask for cross validation
                mask_cv = np.logical_xor(mask_temp, mask0)
            
                # Create mask where data exists for later use
                mask_data = np.invert(mask_temp)
                n_data = len(mask_data[mask_data==True])

                # Apply mask on displacement field
                fdispl = gg.mask_field(fdispl, mask_temp, np.nan)
                displ_mean = compute_mean(fdispl, nt, col)
                fdispl = remove_mean(fdispl, displ_mean, nt, col)
            
                noise_init = np.random.normal(mu, sigma, (nt, nx, ny))                                 
                noise_init *= 8
                fval = [0.0, np.reshape(noise_init, (nt, nobs)).T]
                for init in inits: # loop for type of gap initialization
                    # Gap initialization
                    fdispltp = copy.deepcopy(fdispl)
                    if init == 'zero':
                        fdispltp[mask_temp == True] = fval[0]
                    elif init == 'noise':
                        fdispltp[mask_temp == True] = fval[1][mask_temp == True]
                    
                    # EOF decomposition
                    c = True
                    econv = 1e-6
                    e = []
                    e.append(2)
                    e.append(1)
                    i = 1
                    j = 0
                    neof = 1
                    itr = 0
                    rms_cv = []
                    rms_eof = []

                    print('rmse real  | rmse crss-v')
                    start_t = time.time()
                    while e[i] < e[i-1]:
                        field = copy.copy(fdispltp)
                        eigv, eigval, eigvt = np.linalg.svd(compute_cov(field),
                                                            full_matrices=True)
                        field = eof_decomp(field, eigv, neof)
                        fdispltp[mask_temp] = field[mask_temp]

                        # rms computation in function of neof &/or iterations
                        rms_eof.append(rmse(field, ftruth, nt, nx, ny))
                        rms_cv.append(rmse_cross_v(field[mask_cv], dataii[mask_cv], tng))
                        print ('%0.08f | %0.08f' %(rms_eof[j], rms_cv[j]))
                        rmscv.append(rms_cv[j]) # just make a copy
                        rmseof.append(rms_eof[j])

                        # algorithm to stop reconstruction 
                        if rms_cv[j] > e[i]:
                            end_t = time.time()
                            print('procedure stopped!')
                            break
                        
                        j += 1
                        itr += 1

                        if j > 1:
                            if abs(rms_cv[j-1]-rms_cv[j-2]) > econv:
                                continue
                            else:
                                e.append(rms_cv[j-1])
                                if (1 - e[i+1]/e[i]) < 0.05:
                                    end_t = time.time()
                                    print('procedure stopped!')
                                    break
                                fields.append(field)
                                neof += 1
                                i += 1
                                rms_cv = []
                                rms_eof = []
                                j = 0
                                                        
                    print ('%d iterations - %0.06f seconds' %(itr, end_t - start_t))
                    print ('%d EOFs for reconstruction' %(neof-1))
            rms.append(e[neof])
        # Compute signal to noise ratio
        print ('SNR: %0.03f' %(np.std(truth)/np.std(noises[n])))
    sn_ratio.append((np.std(truth)/np.std(noises[n])))


"""""""""""""""""""""---GRAPHICS---"""""""""""""""""""""

##### ERROR PLOTS #####

X, Y = np.meshgrid(sn_ratio, pourcent) 
rms1 = np.array(rms)
Z1 = np.reshape(rms1[0::2],(X.shape[0], X.shape[1]))
Z2 = np.reshape(rms1[1::2],(X.shape[0], X.shape[1]))
Z = [Z1, Z2]
title = ['Random', 'Correlated']

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 3.5))
i=0
levels = [0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32]
for ax in axs.flat:
    CS = ax.contourf(X, Y, Z[i].T, levels, cmap=plt.cm.coolwarm, extend='both')
    ax.set_title(title[i])
    ax.set_xlabel('SNR', fontsize=12)
    ax.set_ylabel('% of gaps', fontsize=12)
    i+=1

fig.subplots_adjust(right=0.8)
cbar = fig.add_axes([0.82, 0.12, 0.03, 0.75])
fig.colorbar(CS, cax=cbar)
#cbar.ax.set_ylabel('Cross-RMSE')
plt.show()
