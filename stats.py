#!/usr/bin/env python

#########################################
#
# interp.stats
#
# Statistics in python.
#
# AH 07-09-2018
# Last update : 03-11-2018
#########################################


import numpy as np


def rmse(pred, obs, nt, nx, ny, col=True):
    """Compute the Root-Mean-Square Error (RMSE) 
    between prediction and observation fields 
    """
    sum_of_dif = 0
    for i in range(0, nt):
        if col :
            sum_of_dif += np.linalg.norm(pred[:,i] - obs[:,i])**2
        else :
            sum_of_dif += np.linalg.norm(pred[i] - obs[i])**2
    return np.sqrt(sum_of_dif/(nt*nx*ny))


def rmse_zero_mode(pred, obs, nt, nx, ny):
    """Compute RMSE between the mean of prediction and observation fields
    """
    sum_of_dif = 0
    for i in range(0, nt):
        sum_of_dif += np.linalg.norm(np.mean(pred[:,i])-obs[:,i])**2
    return np.sqrt(sum_of_dif/(nt*nx*ny))


def rmse_cross_v(pred, obs, N):
    """Compute cross-RMSE between prediction and observation fields
    """
    sum_of_dif = 0 
    for i in range(0, N):
        sum_of_dif += np.linalg.norm(pred[i] - obs[i])**2
    return np.sqrt(sum_of_dif/N)


def rmse_cv_zero_mode(pred, obs, N):
    """Compute cross-RMSE between the mean of prediction and observation fields
    """
    sum_of_dif = 0 
    for i in range(0, N):
        sum_of_dif += np.linalg.norm(np.mean(pred) - obs[i])**2
    return np.sqrt(sum_of_dif/(t*N))

def compute_image_rmse(im1, im2, nx, ny):
    """ Compute RMSE between two images
    """
    rmse_im = 0
    for i in range(nx):
        for j in range(ny):
            rmse_im += np.linalg.norm(im1[i][j] - im2[i][j])**2
    return np.sqrt(rmse_im/(nx*ny))


def compute_mean(data, col, nozeros=True):
    """Compute the column/line mean of a matrix without NaNs
    """
    datamean = []
    if not col: data = data
    else: data = data.T
    for i in range(0, data.shape[0]):
        if not nozeros:
            datamean.append(np.mean(data[i])) 
        else:
            datamean.append(np.nanmean(data[i]))
    return datamean


def compute_mean0(matrix, n, col, nozeros=True):
    """Compute the column/line mean of a matrix without zeros
    """
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


def remove_mean(data, mean, col):
    """Remove the column/line mean of a matrix
    """
    if col: 
        for i in range(0, data.shape[1]):
            data[:,i] -= mean[i]
    else:
        for i in range(0, data.shape[0]):
            data[i] -= mean[i]
    return data


def remove_t_mean(data, mean):
    """Remove the temporal mean of a time series matrix
    """
    n = max(data.shape[0],data.shape[1])
    for i in range(0, n):
        if data.shape[0]==n:
            data[i] -= mean[i]
        else :
            data[:,i] -= mean[i]
    return data


def remove_s_mean(data, mean):
    """Remove the spatial mean of a time series matrix
    """
    n = min(data.shape[0],data.shape[1])
    for i in range(0, n):
        if data.shape[0]==n:
            data[i] -= mean[i]
        else :
            data[:,i] -= mean[i]
    return data

def add_mean(data, mean, col):
    """Add the column/line mean of a matrix 
    """
    if col:
        for i in range(0, data.shape[1]):
            if np.isnan(mean[i]):
                data[:,i] += 0.
            else:
                data[:,i] += mean[i]
    else:
        for i in range(0, data.shape[0]):
            if np.isnan(mean[i]):
                data[i] += 0. 
            else:
                data[i] += mean[i]
    return data


def compute_variance_by_mode(n_EOF, singular_val, cov=True):
    """Compute variance contained in any mode (EOF).
    Based on Beckers and Rixen 2001
    """
    tot_variance = 0
    variance = []
    # singular values of field matrix X are the square roots
    # of the eigenvalues of Xt*X
    if not cov :
        singular_val *= singular_val
    for i in range(0, len(singular_val)):
        tot_variance += singular_val[i]
    return [100*((singular_val[i])/tot_variance) for i in range(0, n_EOF)]


def chi2(obs, pred, n):
    """Compute statistical Chi-2 between observations and predictions
    """
    chi = 0.
    for i in range(n):
        if pred[i]==0. or obs[i]==0. or (pred[i]==0. and obs[i]!=0.):
            print ("Bad expected number in chi-2")
        else :
            chi += (1/np.var(obs))*(obs[i] - pred[i])**2
    return chi


def ftest(data1, data2):
    """Compute the F-Test between the variances of two datasets
    """
    var1, var2 = np.var(data1), np.var(data2)
    if var1 > var2:
        return var1/var2
    else:
        return var2/var1
