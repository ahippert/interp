#!/usr/bin/env python

# AH 08/10/2018
# Last updated : 09/10/2018

import numpy as np
import matplotlib.pyplot as plt


''' 
Generates time series of correlated noise using
Cholesky decomposition
Input :
    shape : tuple of ints, shape of the desired noise matrix 
    r : correlation coefficient in ]0,1[
    corr : type of correlation
           0 --> time-correlated noise
           1 --> space-correlated noise
Output:
    y : noise matrix of dimension (nt, ns) or (ns, nt)
        depending on the type of correlation
'''
def gen_corr_noise(r, shape, corr):
    if corr==0:
        R = gen_covariance_matrix(shape[0], r)
    else:
        R = gen_covariance_matrix(shape[1], r)
    L = np.linalg.cholesky(R)
    x = np.random.normal(0, 0.1, (shape[0],shape[1]*shape[2]))
    y = np.dot(L, x)
    return np.reshape(y,shape)

''' 
Generates covariance matrix for a given
correlation correlation coefficient
Input :
    m : size of covariance matrix (should be symmetric
        and positive semi definite)
    corcoef : correlation coefficient in ]0,1[ interval
Output :
    cov : Covariance matrix
'''
def gen_covariance_matrix(m, corcoef):
    cov = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            cov[i][j] = corcoef**(abs(i-j))
    #R = np.kron(Rt, Rs)
    return cov

''' 
Apply patches on a time series
Input :
    patches : array of patches in format (nt, ns, ns)  
    nt : time dimension (number of images)
    ns : space dimension (number of points)
    pw : patch width
Output:
    img : time series of patches
'''
def apply_patch_on_image(patches, nt, ns, pw):
    img = np.zeros((nt, ns*pw, ns*pw))
    for i in range(nt):
        for j in range(ns):
            for k in range(ns):
                for c in range(ns):
                    img[i][j+c*ns][ns*k:ns*(k+1)] = patches[k+c*ns][i][j]
    return img


### TEST CODE ###

# nt = 2 # time dimension
# ns = 20 # image side length
# pl = 10 # patch length, condition : ns/np = integer
# #if (ns/pl).is_integer() == False:
# #    print('Number of patches cannot match image size')
# #    print('An error will be thrown')
# #np = (ns/pl)**2 # number of patches
# np = 4
# all_tseries = []

# Rt = np.zeros((nt, nt))
# Rs = np.zeros((ns, ns))
# rot = 0.1
# ros = 0.9
# for i in range(nt):
#     for j in range(nt):
#         Rt[i][j] = rot**(abs(i-j))
# for i in range(ns):
#     for j in range(ns):
#         Rs[i][j] = ros**(abs(i-j))

# R = np.kron(Rt, Rs)

# for i in range(ns):
#     x = np.random.normal(0, 1, (nt*ns))
#     xv = np.vstack(x)
#     L = np.linalg.cholesky(R)
#     y = np.dot(L, xv)
#     yh = np.hstack(y)
#     #tseries = np.reshape(yh, (tlen, s, s))
#     all_tseries.append(np.reshape(yh, (nt, ns, ns)))
    
# # test patching
# im = np.zeros((nt, ns, ns))
# for k in range(nt):
#     for i in range(ns):
#         for j in range(ns):
#             for c in range(ns):
#                 im[k][i+c*ns][ns*j:ns*(j+1)] = all_tseries[j+c*ns][k][i]

### Graphic tests ###

# plt.figure(1)
# plt.imshow(Rt)

# plt.figure(2)
# plt.imshow(Rs)

# plt.figure(3)
# plt.imshow(R)

# plt.figure(4)
# plt.imshow(all_tseries[0][0])

# for i in range(nt):
#     plt.figure()
#     plt.imshow(im[i])
# plt.show()
