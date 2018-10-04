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
modes = 3 # in [2,7]
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

# noise_init = np.random.normal(mu, sigma, (nt, nx, ny))                                 
# noise_init *= 0.1
# fval = [0.0, np.reshape(noise_init, (nt, nobs)).T] # two types of gaps generation
pourcent = [20]
col = True
#init_zero = [True, False] 
inits = ['zero','noise'] # 'noise' or 'zero'
gens = ['random'] # 'random' or 'correlated'

#for i in gens: # loop for gap generation type
for ti in range(1):
    
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
    inter = 3
    mul = np.linspace(1, 8, inter)
    mul = [6]
    noises = [noise*i for i in mul]
    
    for n in range(len(mul)): # loop for different noise levels
            
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
            fdispl = copy.copy(datai)
            # 1. generate correlated gaps using noise
            # if gen == 'correlated':
            #     t_start, t_end = 8, 14
            #     seuil = norm.ppf(k/100., np.mean(gaps), np.std(gaps))
            #     print ('seuil: %0.2f' %seuil)
            #     mask0 = gg.gen_correlated_gaps(gaps, seuil, t_start, t_end)
            #     mask0 = np.reshape(mask0, (nt, nobs)).T
                                
            # # 2. Generate random gaps 
            # elif gen == 'random':
            ngaps = np.arange(int(nobs*nt*k/100.))
            mask0 = gg.gen_random_gaps(np.zeros((nobs, nt), dtype=bool),
                                       nobs,
                                       nt,
                                       ngaps)
            
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

                # Create a blank image
                #fdispl[:,10][1000:30000] = np.nan

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
                    
                    # u, d, v = np.linalg.svd(compute_cov(fdispltp),
                    #                         full_matrices=True)
                    # plot_bars(range(nt),
                    #           compute_variance_by_mode(nt, d),'i','j')
                                  
                    elif init == 'noise':
                        fdispltp[mask_temp == True] = fval[1][mask_temp == True]
                    
                    #fdispltp[:,10][1000:30000] = fval[1][:,10][1000:30000]
                    # u1, d1, v1 = np.linalg.svd(compute_cov(fdispltp),
                    #                            full_matrices=True)
                    # plot_bars(range(nt),
                    #           compute_variance_by_mode(nt, d1),'k','l')

                    # u2, d2, v2 = np.linalg.svd(compute_cov(dataii),
                    #                            full_matrices=True)
                    # plot_bars(range(nt),
                    #           compute_variance_by_mode(nt, d2),'k','l')
                    # plt.figure()
                    # plt.plot(d, 'ro', alpha=0.9)
                    # plt.plot(d1,'ko', alpha=0.2)
                    # plt.plot(d2,'bo')
                    # plt.figure()
                    # plt.plot(d-d1)
                    # plt.show()

                    
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
                    #for i in range(0, itr):
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
                    #rmsinit.append(e[neof+1])
                #print (rmsinit)
            # Compute RMSE image to image 
            #rms_all = np.append(rms_all, [compute_image_rmse(field[i], ftruth[i], nx, ny) for i in range(nt)])
            # Compute signal to noise ratio
            #sn_ratio = np.append(sn_ratio, np.std(truth)/np.std(noises[n]))
            #print ('SNR: %0.03f - NSR: %0.03f' %(sn_ratio, 1./sn_ratio))
            print ('SNR: %0.03f' %(np.std(truth)/np.std(noises[n])))



plt.figure()
plt.plot(rmsinit[::2],'ro')
plt.plot(rmsinit[1::2],'ko')
plt.show()
            

# Energy bar plots            
#eigvals.append(np.linalg.eigvals(np.cov(field.T)))
#variance = compute_variance_by_mode(nt, eigval)
#plot_bars(range(1, nt+1), variance, 'Mode', 'Contained variance in mode k (%)')

# STEP 5
for i in range(len(fields)):
   fields[i] = add_mean(fields[i], displ_mean, nt, col)
   fields[i] = np.reshape(fields[i], (nx, ny, nt)).T

# field = add_mean(field, displ_mean, N, col)
# field = np.reshape(field, (nx, ny, nt)).T
    
# Add temporal mean that was substracted
ftruth = add_mean(ftruth, truth_mean, nt, col)
fdispl = add_mean(fdispl, displ_mean, nt, col)
fdispltp = add_mean(fdispltp, displ_mean, nt, col)
dataii = add_mean(dataii, dataimean, nt, col)


# reshape data matrix into a time series of images
ftruth = np.reshape(ftruth, (nx, ny, nt)).T # pour afficher l'image: im[i].T
fdispl = np.reshape(fdispl, (nx, ny, nt)).T
fdispltp = np.reshape(fdispltp, (nx, ny, nt)).T
dataii = np.reshape(dataii, (nx, ny, nt)).T

#ftest = []
#for i in range(0, nEOF-2):
#    ftest.append((1./tng)*(chsq[i]-chsq[i+1])/chsq[i+1])


"""""""""""""""""""""---GRAPHICS---"""""""""""""""""""""

##### SCATTER PLOT TESTS #####

# snr = sn_ratio[10::50]
# rms = rms_all[10::50]
# xy = np.vstack([snr, rms])
# z = gaussian_kde(xy)(xy)
# idx = z.argsort()
# snr, rms, z = snr[idx], rms[idx], z[idx]

# rcoef = (pearsonr(snr, rms))
# fig, ax = plt.subplots()
# ax.scatter(snr, rms, c=z, s=50, edgecolor='')
# ax.set_xlabel('SNR')
# ax.set_ylabel('RMSE')
# ax.set_title('Error vs noise scatter plot <r>=%0.2f' % (rcoef[0])) 
# plt.show()

##### ERROR PLOTS #####

# if len(nts) > 1 :
#     plt.figure(2)
#     plt.xticks(range(0, len(nts)), nts)
#     plt.plot(range(0, len(nts)), rms_nt, 'k-o', linewidth=1)
#     plt.title('Root Mean Square Error vs. number of images')


# print (rms_vc)
# fig, ax = plt.subplots()
# #plt.xticks(ngaps)
# ax.plot(100*(ngaps/nobs), rms_cv, 'k-')
# ax.plot(100*(ngaps/nobs), rms_eof, 'r-')
# ax.set_xlabel('% of point used in cross validation per image')
# ax.set_ylabel('RMSE')
# ax.set_title('Interpolation error vs. number of points used in ross validation')

# plt.figure()
# plt.plot(sn_ratio, rms_cv, 'k-')
# plt.plot(sn_ratio, rms_eof, 'r-')
# plt.title('RMSE')

#st, end = 0, itr
#st1, end1 = nEOF, nEOF*2
#fig, ax = plt.subplots(figsize=(7,5))
#ax.plot(range(0, itr), rms_vc[st:end], 'k-', label='correlated gaps + noise init')
#ax.plot(range(1,nEOF+1), rms_vc[st1:end1], 'k--', label='correlated gaps + noise init')
#ax.plot(range(1,nEOF+1), rms_vc[end1:nEOF*3], 'b-', label='random gaps + 0 init')
#ax.plot(range(1,nEOF+1), rms_vc[nEOF*3:nEOF*4], 'b--', label='random gaps + noise init')

# # ax.plot(range(nEOF), rms_it[st:end], 'b-', label='ideal cross-v error')
# # ax.plot(range(nEOF), rms_it[st1:end1], 'b--', label='ideal cross-v error')

#ax.plot(range(0, itr), rms_eof[st:end], 'r-', label='ideal error')
#ax.plot(range(0, itr), rms[st:end], 'b-', label='non-missing data error')


#ax.set_xlabel('Number of EOFs used in reconstruction')
#ax.set_xlabel('iterations')
#ax.set_ylabel('rmse')
# ax.set_title('%d-order displacement field with %d%% gaps, SNR=%0.02f' %(modes, pourcent[0], sn_ratio[0]))
# plt.xticks(np.arange(1, nEOF+1, 2))
# plt.ylim(0, 0.4)
#plt.legend()

# pct = []
# for i in range(len(e)-1):
#     pct.append(100*(1-e[i+1]/e[i]))
# plt.figure()
# plt.plot(pct[2:], 'r-')


plt.figure()
plt.title('RMSE vs iterations')
plt.plot(rmscv[:-1], label='cross-v error')
plt.plot(rmseof[:-1], label='real error')
plt.legend()
# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(range(1,nEOF), chsq_truth)
# ax.plot(range(1,nEOF), chsq, 'r')
# ax.set_title('2nd order displacement field with %d%% gaps, SNR=%0.02f' %(pourcent[0], sn_ratio[0]))
# ax.set_xlabel('Number of EOFs retained in reconstruction')
# ax.set_ylabel('Chi-square')

# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(range(0,nEOF-1), ftst)

# ax.set_title('2nd order displacement field with %d%% gaps, SNR=%0.02f' %(pourcent[0], sn_ratio[0]))
# ax.set_xlabel('Step')
# ax.set_ylabel('F-test')
#plt.show()

## EIGENVALUES PLOTS ##
# plt.figure()
# for i in range(len(eigvals)):
#     plt.plot(range(nt), np.sort(np.real(eigvals[i]))[::-1])




# X, Y = np.meshgrid(sn_ratio[0:len(pourcent)], pourcent) 
# rms_eof_g1, rms_it_g1, rms_vc_g1 = [], [], []
# rms_eof_g2, rms_it_g2, rms_vc_g2 = [], [], []
# s = X.shape[0]
# for i in range(X.shape[0]):
#     rms_eof_g1 += rms_eof[2*s*i:s*(2*i+1)]
#     rms_it_g1 += rms_it[2*s*i:s*(2*i+1)]
#     rms_vc_g1 += rms_vc[2*s*i:s*(2*i+1)]
#     rms_eof_g2 += rms_eof[s*(2*i+1):2*s*(i+1)]
#     rms_it_g2 += rms_it[s*(2*i+1):2*s*(i+1)]
#     rms_vc_g2 += rms_vc[s*(2*i+1):2*s*(i+1)]
# Z1 = np.reshape(rms_eof_g1,(X.shape[0], X.shape[0]))
# Z2 = np.reshape(rms_it_g1,(X.shape[0], X.shape[0]))
# Z3 = np.reshape(rms_vc_g1,(X.shape[0], X.shape[0]))
# Z4 = np.reshape(rms_eof_g2,(X.shape[0], X.shape[0]))
# Z5 = np.reshape(rms_it_g2,(X.shape[0], X.shape[0]))
# Z6 = np.reshape(rms_vc_g2,(X.shape[0], X.shape[0]))
# Z = [Z1, Z2, Z3, Z4, Z5, Z6]
# col = ['r', 'b', 'k', 'r', 'b', 'k']

# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
# i=0
# #levels
# for ax in axs.flat:
#     CS = ax.contour(X, Y, Z[i].T, colors=col[i])
#     i+=1
#     plt.clabel(CS, inline=1, fontsize=8)
# plt.show()


    

##### plot fields #####
# ind = 10
# img = [ind]
# cmaps = ['RdGy', 'inferno', 'viridis', 'seismic']
# axtitle = ['Truth', 'Real', 'Reconstructed']
# cm = 0
# row1, col1 = 1, 3
# mat1 = [ftruth, fdispl, fields[len(fields)-2]]
# row2, col2 = 1, 2
# mat2 = [ftruth-fields[len(fields)-2], noise]
# max1, min1 = np.nanmax(fdispl[ind]), np.nanmin(fdispl[ind])
# max2, min2 = np.max(noise[ind])+0.3, np.min(noise[ind])-0.3
# savefig = False

# for i in img :
#     j=0
#     fig, axs = plt.subplots(nrows=row1, ncols=col1, figsize=(15, 5))
#     #fig.suptitle('%d EOF - %d%% %s gaps - SNR=%0.02f' %(nEOF, pourcent[0], gen, sn_ratio[0]),
#     #             fontsize = 16)
#     fig.suptitle('%d EOF - %d%% gaps - SNR=%0.02f - %d iterations' %(neof-1, pourcent[0], sn_ratio[0], itr),
#                  fontsize = 16)
#     for ax in axs.flat:
#         im = ax.imshow(mat1[j][i].T, vmin=min1, vmax=max1, cmap = cmaps[cm])
#         ax.set_title(axtitle[j], fontsize=16)
#         j+=1
#         fig.subplots_adjust(right=0.8)
#         cb_ax = fig.add_axes([0.84, 0.2, 0.03, 0.59])
#         cbar = fig.colorbar(im, cax=cb_ax)
#     j=0
#     if savefig :
#         plt.savefig('/home/hipperta/Documents/img/reconstruction/g%d%%_snr%0.01f.png' %(pourcent[0], sn_ratio[0]))
#     fig, axs = plt.subplots(nrows=row2, ncols=col2, figsize=(10, 5))
#     for ax in axs.flat:
#         im = ax.imshow(mat2[j][i].T, vmin=min2, vmax=max2, cmap = cmaps[cm])
#         j+=1
#         fig.subplots_adjust(right=0.8)
#         cb_ax = fig.add_axes([0.85, 0.11, 0.03, 0.76])
#         cbar = fig.colorbar(im, cax=cb_ax)

        
# for i in range(len(fields)):
#      plt.figure()
#      plt.imshow(fields[i][ind].T, vmin=min1, vmax=max1, cmap = cmaps[cm])
# plt.show()


# tstruth = []
# tsdispl = []
# tsfield = []
# tsdataii = []
# x = np.random.randint(50)
# y = np.random.randint(50)
# for i in range(0, nt):
#     tstruth.append(ftruth[i][x][y])
#     tsdispl.append(fdispl[i][x][y])
#     tsdataii.append(dataii[i][x][y])
#     tsfield.append(fields[len(fields)-2][i][x][y])
# fig, ax = plt.subplots(figsize=(7,5))
# ax.plot(tstruth, 'k-', label='original field')
# ax.plot(tsdispl, 'r-', label='gappy + noisy field')
# ax.plot(tsdataii, 'r--', label='value of field in gaps')
# ax.plot(tsfield, 'b-', label='reconstructed field')
# plt.legend()


##### plot SNR and error of temporal series #####
# color = 'tab:blue'
# fig, ax1 = plt.subplots()
# plt.xticks(range(0, nt+1))
# ax1.set_ylabel('S/N ratio', color=color)
# ax1.plot(range(0, nt), sn_ratio, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# color = 'tab:red'
# ax2 = ax1.twinx()
# ax2.set_ylabel('RMSE', color=color)  # we already handled the x-label with ax1
# ax2.plot(range(0, nt), rms_all, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()

plt.show()
    
